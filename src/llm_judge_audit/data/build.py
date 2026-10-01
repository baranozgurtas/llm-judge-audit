"""Canonical pair construction, human-vote aggregation, and stratified sampling.

All outputs are pure functions of the pinned raw files and `config/study.json`;
`build_artifacts` returns {filename: bytes} so determinism can be checked byte-for-byte.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_judge_audit.config import (
    ATTRIBUTION_FILE,
    AUDIT_FILE,
    LICENSE_FILE,
    MANIFEST_FILE,
    MANIFEST_META_FILE,
    PAIRS_FILE,
    PROJECT_ROOT,
)
from llm_judge_audit.data.schema import (
    Question,
    SourceRow,
    load_parquet_rows,
    load_questions,
    validate_rows,
)
from llm_judge_audit.io_utils import canonical_json, pretty_json, sha256_bytes, sha256_text

CANDIDATES = ("cand_1", "cand_2")
HUMAN_LABELS = ("cand_1", "cand_2", "tie")

AGGREGATION_RULE_TEXT = (
    "aggregation-v1: (1) Only turn-1 rows are used. (2) Each vote is mapped from source "
    "model_a/model_b to the canonical candidate holding that exact response text; 'tie' stays "
    "'tie'. (3) Per (pair, annotator): several identical votes count once (lowest source row "
    "kept, the rest logged as duplicate_vote_same_annotator); differing votes are all dropped "
    "and logged as conflicting_votes_same_annotator. (4) Over retained votes, the label "
    "(cand_1, cand_2 or tie) with a strictly larger count than each other label is the "
    "aggregate; if no strict plurality exists, or no votes remain, the aggregate is "
    "'unresolved'. A tie aggregate stays 'tie' and 'unresolved' stays 'unresolved'; neither is "
    "ever converted to a decisive label. Only cand_1/cand_2 aggregates count as resolved."
)
SAMPLING_RULE_TEXT = (
    "sampling-v1: eligible pool = turn-1 canonical pairs whose two candidate texts differ. "
    "Strata = the 8 MT-bench categories; equal allocation n_pairs / 8 per stratum. Within a "
    "stratum, pairs are ranked by sha256('{seed}|{category}|{pair_id}') ascending and the first "
    "k are selected. The build stops if any stratum has fewer than k eligible pairs."
)
LENGTH_RULE_TEXT = "whitespace-token count = len(text.split()) on the exact candidate text"


class SamplingError(RuntimeError):
    pass


def whitespace_tokens(text: str) -> int:
    return len(text.split())


@dataclass
class Vote:
    annotator: str
    vote: str
    raw_winner: str
    source_row: int


@dataclass
class _Pair:
    question_id: int
    turn: int
    question: str
    texts: dict[str, str]  # sha -> text
    models: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    votes: list[Vote] = field(default_factory=list)
    source_rows: list[int] = field(default_factory=list)

    @property
    def shas(self) -> tuple[str, str]:
        ordered = sorted(self.texts)
        return (ordered[0], ordered[-1])

    @property
    def pair_id(self) -> str:
        h1, h2 = self.shas
        return "p_" + sha256_text(f"{self.question_id}|{self.turn}|{h1}|{h2}")[:16]


def _candidate_for(pair: _Pair, text: str) -> str:
    h1, _ = pair.shas
    return "cand_1" if sha256_text(text) == h1 else "cand_2"


def aggregate_votes(
    pair_id: str, votes: list[Vote]
) -> tuple[list[Vote], list[dict[str, Any]], str, dict[str, int]]:
    """Apply aggregation-v1. Returns (retained votes, adjustment log, aggregate, counts)."""
    by_annotator: dict[str, list[Vote]] = defaultdict(list)
    for v in votes:
        by_annotator[v.annotator].append(v)
    retained: list[Vote] = []
    log: list[dict[str, Any]] = []
    for annotator in sorted(by_annotator):
        group = sorted(by_annotator[annotator], key=lambda v: v.source_row)
        if len(group) == 1:
            retained.append(group[0])
            continue
        if len({v.vote for v in group}) == 1:
            retained.append(group[0])
            log.append(_adj(pair_id, group[0], "kept", "duplicate_vote_same_annotator"))
            log.extend(
                _adj(pair_id, v, "dropped", "duplicate_vote_same_annotator") for v in group[1:]
            )
        else:
            log.extend(
                _adj(pair_id, v, "dropped", "conflicting_votes_same_annotator") for v in group
            )
    counts = {label: 0 for label in HUMAN_LABELS}
    for v in retained:
        counts[v.vote] += 1
    best = max(counts.values())
    leaders = [label for label in HUMAN_LABELS if counts[label] == best]
    aggregate = leaders[0] if best > 0 and len(leaders) == 1 else "unresolved"
    return retained, log, aggregate, counts


def _adj(pair_id: str, v: Vote, action: str, reason: str) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "annotator": v.annotator,
        "source_row": v.source_row,
        "vote": v.vote,
        "action": action,
        "reason": reason,
    }


def construct_pairs(
    rows: list[SourceRow], questions: dict[int, Question], turn: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build canonical pair records (eligible and ineligible) plus the audit log."""
    row_exclusions: list[dict[str, Any]] = []
    pairs: dict[tuple[int, int, str, str], _Pair] = {}
    identical: dict[tuple[int, int, str], dict[str, Any]] = {}

    for r in rows:
        if r.turn != turn:
            row_exclusions.append(
                {
                    "source_row": r.source_row,
                    "question_id": r.question_id,
                    "turn": r.turn,
                    "reason": f"turn_{r.turn}_out_of_scope",
                }
            )
            continue
        ha, hb = sha256_text(r.text_a), sha256_text(r.text_b)
        if ha == hb:
            row_exclusions.append(
                {
                    "source_row": r.source_row,
                    "question_id": r.question_id,
                    "turn": r.turn,
                    "reason": "identical_candidate_texts",
                }
            )
            key_i = (r.question_id, r.turn, ha)
            entry = identical.setdefault(
                key_i,
                {
                    "question_id": r.question_id,
                    "turn": r.turn,
                    "text_sha256": ha,
                    "source_models": set(),
                    "source_rows": [],
                },
            )
            entry["source_models"].update({r.model_a, r.model_b})
            entry["source_rows"].append(r.source_row)
            continue
        h1, h2 = sorted((ha, hb))
        key = (r.question_id, r.turn, h1, h2)
        pair = pairs.get(key)
        if pair is None:
            pair = _Pair(r.question_id, r.turn, r.question_turn1, {ha: r.text_a, hb: r.text_b})
            pairs[key] = pair
        pair.models[ha].add(r.model_a)
        pair.models[hb].add(r.model_b)
        pair.source_rows.append(r.source_row)
        if r.winner == "tie":
            vote = "tie"
        else:
            vote = _candidate_for(pair, r.text_a if r.winner == "model_a" else r.text_b)
        pair.votes.append(Vote(r.annotator, vote, r.winner, r.source_row))

    records: list[dict[str, Any]] = []
    adjustments: list[dict[str, Any]] = []
    for pair in pairs.values():
        pid = pair.pair_id
        retained, log, aggregate, counts = aggregate_votes(pid, pair.votes)
        adjustments.extend(log)
        dropped_rows = {e["source_row"] for e in log if e["action"] == "dropped"}
        cands: dict[str, Any] = {}
        for name, sha in zip(CANDIDATES, pair.shas, strict=True):
            text = pair.texts[sha]
            cands[name] = {
                "text": text,
                "sha256": sha,
                "whitespace_tokens": whitespace_tokens(text),
                "source_models": sorted(pair.models[sha]),
            }
        n1, n2 = cands["cand_1"]["whitespace_tokens"], cands["cand_2"]["whitespace_tokens"]
        records.append(
            {
                "pair_id": pid,
                "question_id": pair.question_id,
                "turn": pair.turn,
                "category": questions[pair.question_id].category,
                "question": pair.question,
                "candidates": cands,
                "longer_candidate": "equal" if n1 == n2 else ("cand_1" if n1 > n2 else "cand_2"),
                "eligible": True,
                "exclusion_reason": None,
                "human": {
                    "aggregate": aggregate,
                    "counts": counts,
                    "n_votes_source": len(pair.votes),
                    "n_votes_retained": len(retained),
                    "votes": [
                        {
                            "annotator": v.annotator,
                            "vote": v.vote,
                            "raw_winner": v.raw_winner,
                            "source_row": v.source_row,
                            "retained": v.source_row not in dropped_rows,
                        }
                        for v in sorted(pair.votes, key=lambda v: v.source_row)
                    ],
                },
                "source_rows": sorted(pair.source_rows),
            }
        )

    pair_exclusions: list[dict[str, Any]] = []
    for entry in identical.values():
        pid = (
            "p_"
            + sha256_text(
                f"{entry['question_id']}|{entry['turn']}|{entry['text_sha256']}|{entry['text_sha256']}"
            )[:16]
        )
        pair_exclusions.append(
            {
                "pair_id": pid,
                "question_id": entry["question_id"],
                "turn": entry["turn"],
                "text_sha256": entry["text_sha256"],
                "source_models": sorted(entry["source_models"]),
                "source_rows": sorted(entry["source_rows"]),
                "reason": "identical_candidate_texts",
            }
        )

    records.sort(key=lambda rec: (rec["question_id"], rec["pair_id"]))
    pair_exclusions.sort(key=lambda e: (e["question_id"], e["pair_id"]))
    adjustments.sort(key=lambda e: (e["pair_id"], e["annotator"], e["source_row"]))
    row_exclusions.sort(key=lambda e: e["source_row"])
    audit = {
        "row_exclusions": row_exclusions,
        "pair_exclusions": pair_exclusions,
        "vote_adjustments": adjustments,
    }
    return records, audit


def stratified_sample(
    eligible: list[dict[str, Any]], categories: list[str], n: int, seed: int
) -> list[dict[str, Any]]:
    if n % len(categories) != 0:
        raise SamplingError(f"n={n} not divisible by {len(categories)} categories")
    k = n // len(categories)
    pool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in eligible:
        if rec["category"] not in categories:
            raise SamplingError(f"unexpected category {rec['category']!r}")
        pool[rec["category"]].append(rec)
    shortfalls = {c: len(pool[c]) for c in categories if len(pool[c]) < k}
    if shortfalls:
        raise SamplingError(
            f"cannot sample {k} per category; eligible counts below {k}: {shortfalls}"
        )
    selected: list[dict[str, Any]] = []
    for cat in categories:
        ranked = sorted(pool[cat], key=lambda r: sha256_text(f"{seed}|{cat}|{r['pair_id']}"))
        selected.extend(ranked[:k])
    if len({r["pair_id"] for r in selected}) != n:
        raise SamplingError("sample is not unique")
    return selected


def _attribution(cfg: dict[str, Any]) -> str:
    ds, qs = cfg["source"]["dataset"], cfg["source"]["questions"]
    return f"""# Dataset attribution

This directory contains artifacts derived from:

**MT-Bench Human Judgments** — `{ds["hf_repo"]}` (Hugging Face), split `{ds["split"]}`,
revision `{ds["hf_revision"]}`, file `{ds["path"]}` (sha256 `{ds["sha256"]}`).
License: **{ds["license"]}** (https://creativecommons.org/licenses/by/4.0/), as declared in the
dataset card metadata at that revision. Authors: Lianmin Zheng, Wei-Lin Chiang, Ying Sheng,
Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing,
Hao Zhang, Joseph E. Gonzalez, Ion Stoica. "Judging LLM-as-a-judge with MT-Bench and Chatbot
Arena", arXiv:2306.05685 (2023).

**MT-bench question categories** — `{qs["repo"]}` at commit `{qs["revision"]}`, file
`{qs["path"]}` (sha256 `{qs["sha256"]}`). License: **{qs["license"]}**.

Changes made: rows restricted to turn 1; candidate responses deduplicated by exact text;
votes mapped to canonical candidates and aggregated (aggregation-v1); identical-text pairs
excluded; a 200-pair stratified sample drawn (sampling-v1). Human labels are not modified.

Note: candidate responses were generated by third-party models (GPT-4, GPT-3.5, Claude-v1,
Vicuna-13B, Alpaca-13B, LLaMA-13B). This project uses them only for non-commercial research
evaluation; no model is trained on them.
"""


def build_artifacts(
    cfg: dict[str, Any], root: Path = PROJECT_ROOT
) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Build every derived artifact in memory. Returns ({filename: bytes}, summary)."""
    ds, qs, sp = cfg["source"]["dataset"], cfg["source"]["questions"], cfg["sampling"]
    raw_path, q_path = root / ds["local_file"], root / qs["local_file"]
    questions = load_questions(q_path)
    categories = list(sp["categories"])
    if sorted({q.category for q in questions.values()}) != sorted(categories):
        raise SamplingError("question categories differ from configured categories")
    rows = validate_rows(load_parquet_rows(raw_path), questions)
    records, audit = construct_pairs(rows, questions, turn=sp["turn"])
    selected = stratified_sample(records, categories, sp["n_pairs"], sp["seed"])

    manifest_records = []
    for idx, rec in enumerate(selected):
        m = {k: v for k, v in rec.items() if k not in ("eligible", "exclusion_reason")}
        m["manifest_index"] = idx
        manifest_records.append(m)

    def jsonl(recs: list[dict[str, Any]]) -> bytes:
        return "".join(canonical_json(r) + "\n" for r in recs).encode("utf-8")

    pairs_bytes = jsonl(records)
    manifest_bytes = jsonl(manifest_records)
    raw_sha = sha256_bytes(raw_path.read_bytes())
    q_sha = sha256_bytes(q_path.read_bytes())

    agg_pool = Counter(r["human"]["aggregate"] for r in records)
    agg_sample = Counter(r["human"]["aggregate"] for r in manifest_records)
    eligible_by_cat = Counter(r["category"] for r in records)
    sample_by_cat = Counter(r["category"] for r in manifest_records)
    adj = audit["vote_adjustments"]
    excl_reasons = Counter(e["reason"] for e in audit["row_exclusions"])
    totals = {
        "source_rows": len(rows),
        "rows_in_scope_turn": sum(1 for r in rows if r.turn == sp["turn"]),
        "row_exclusions_by_reason": dict(sorted(excl_reasons.items())),
        "canonical_pairs_total": len(records) + len(audit["pair_exclusions"]),
        "eligible_pairs": len(records),
        "excluded_pairs_identical_text": len(audit["pair_exclusions"]),
        "eligible_pairs_by_category": {c: eligible_by_cat[c] for c in categories},
        "eligible_pool_human_aggregate": {k: agg_pool[k] for k in sorted(agg_pool)},
        "vote_adjustment_rows_by_reason": dict(
            sorted(Counter(f"{e['reason']}:{e['action']}" for e in adj).items())
        ),
        "pairs_with_vote_adjustments": len({e["pair_id"] for e in adj}),
        "votes_retained_in_eligible_pool": sum(r["human"]["n_votes_retained"] for r in records),
    }
    audit_doc = {
        "aggregation_rule_version": cfg["aggregation"]["rule_version"],
        "aggregation_rule": AGGREGATION_RULE_TEXT,
        "sampling_rule_version": sp["rule_version"],
        "sampling_rule": SAMPLING_RULE_TEXT,
        "totals": totals,
        **audit,
    }
    audit_bytes = pretty_json(audit_doc).encode("utf-8")
    license_doc = {
        "dataset": {
            "name": "MT-Bench Human Judgments",
            "source": f"https://huggingface.co/datasets/{ds['hf_repo']}",
            "revision": ds["hf_revision"],
            "split": ds["split"],
            "file": ds["path"],
            "sha256": ds["sha256"],
            "license": ds["license"],
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "license_evidence": "dataset card YAML metadata `license: cc-by-4.0` at the pinned "
            "revision; Hub API tag `license:cc-by-4.0`; dataset not gated",
            "citation": "Zheng et al., Judging LLM-as-a-judge with MT-Bench and Chatbot Arena, "
            "arXiv:2306.05685, 2023",
        },
        "questions": {
            "name": "MT-bench questions (categories only)",
            "source": f"https://github.com/{qs['repo']}",
            "revision": qs["revision"],
            "file": qs["path"],
            "sha256": qs["sha256"],
            "license": qs["license"],
        },
        "gpt4_judgment_split_used": False,
    }
    meta = {
        "study_name": cfg["study_name"],
        "n_pairs": len(manifest_records),
        "sampling_rule_version": sp["rule_version"],
        "sampling_rule": SAMPLING_RULE_TEXT,
        "sampling_seed": sp["seed"],
        "aggregation_rule_version": cfg["aggregation"]["rule_version"],
        "length_rule": LENGTH_RULE_TEXT,
        "source": {
            "hf_repo": ds["hf_repo"],
            "hf_revision": ds["hf_revision"],
            "raw_file": ds["local_file"],
            "raw_sha256": raw_sha,
            "questions_repo": qs["repo"],
            "questions_revision": qs["revision"],
            "questions_file": qs["local_file"],
            "questions_sha256": q_sha,
            "license": ds["license"],
        },
        "artifacts_sha256": {
            MANIFEST_FILE: sha256_bytes(manifest_bytes),
            PAIRS_FILE: sha256_bytes(pairs_bytes),
            AUDIT_FILE: sha256_bytes(audit_bytes),
        },
        "sample_by_category": {c: sample_by_cat[c] for c in categories},
        "sample_human_aggregate": {k: agg_sample.get(k, 0) for k in (*HUMAN_LABELS, "unresolved")},
        "sample_longer_candidate": dict(
            sorted(Counter(r["longer_candidate"] for r in manifest_records).items())
        ),
        "pair_ids": [r["pair_id"] for r in manifest_records],
    }
    artifacts = {
        PAIRS_FILE: pairs_bytes,
        MANIFEST_FILE: manifest_bytes,
        AUDIT_FILE: audit_bytes,
        MANIFEST_META_FILE: pretty_json(meta).encode("utf-8"),
        LICENSE_FILE: pretty_json(license_doc).encode("utf-8"),
        ATTRIBUTION_FILE: _attribution(cfg).encode("utf-8"),
    }
    return artifacts, {"totals": totals, "meta": meta}
