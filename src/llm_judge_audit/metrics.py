"""Registered metrics (metrics-v1). Definitions are fixed before any benchmark inference.

Units: a *cell* is one judge call for (pair, order); a *pair* has two cells. Only cells with
status `valid` carry a verdict. Human labels come from the manifest aggregate: `cand_1`/`cand_2`
are resolved; `tie` and `unresolved` are reported separately and never scored as correct or
incorrect. A judge `tie` on a resolved pair is a valid, incorrect decision.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from llm_judge_audit.stats import cluster_bootstrap, cohen_kappa, percentile, rate

METRICS_VERSION = "metrics-v1"
RESOLVED = ("cand_1", "cand_2")
JUDGE_LABELS = ("cand_1", "cand_2", "tie")
ORDERS = ("original", "swapped")

DEFINITIONS: dict[str, str] = {
    "main_accuracy": "correct valid cells / valid cells, over pairs with a resolved human "
    "preference (both orders pooled). Wilson interval treats cells as independent; the "
    "pair-cluster bootstrap interval resamples pairs with both orders together.",
    "coverage": "valid cells / expected cells (2 x resolved pairs).",
    "accuracy_by_order": "per order: correct / valid decisions on resolved pairs; coverage = "
    "valid decisions / resolved pairs.",
    "cohen_kappa": "unweighted kappa between human aggregate (cand_1/cand_2) and judge mapped "
    "verdict (cand_1/cand_2/tie) over valid cells of resolved pairs; pair-cluster bootstrap.",
    "order_inconsistency": "pairs whose two mapped-back verdicts differ / pairs with two valid "
    "cells (all sampled pairs, regardless of human label).",
    "position_choice": "displayed verdict A (first) / B (second) / tie per order and pooled, "
    "over valid cells.",
    "consistent_accuracy": "resolved pairs with two valid, equal mapped verdicts: correct / "
    "consistent pairs; coverage = consistent pairs / resolved pairs. Shown beside main accuracy "
    "and coverage; abstention on inconsistent pairs is not an accuracy improvement by itself.",
    "verbosity": "unequal-length pairs (whitespace tokens, frozen pre-inference). Judge: "
    "valid cells choosing the longer candidate / valid decisive cells (A or B), and / all valid "
    "cells. Human: resolved pairs where the longer candidate won / resolved unequal pairs. "
    "Association only, not causal.",
    "baseline_lopo_majority": "for each resolved pair, predict the majority human label "
    "(cand_1 vs cand_2) among all *other* resolved pairs in the sample; ties in that count "
    "predict cand_1. Accuracy = correct / resolved pairs. Descriptive context, not a judge.",
    "latency": "median and p95 (numpy linear percentile) of per-cell wall latency over attempted "
    "cells; token counts as reported by Ollama.",
}


def _correct(verdict: str, human: str) -> bool:
    return verdict == human


def lopo_majority_baseline(labels: dict[str, str]) -> dict[str, Any]:
    """Leave-one-pair-out majority over resolved human labels {pair_id: cand_1|cand_2}."""
    total = Counter(labels.values())
    correct = 0
    predictions: dict[str, str] = {}
    for pid, label in sorted(labels.items()):
        others = total.copy()
        others[label] -= 1
        pred = "cand_1" if others["cand_1"] >= others["cand_2"] else "cand_2"
        predictions[pid] = pred
        correct += pred == label
    return {
        "rule": DEFINITIONS["baseline_lopo_majority"],
        "accuracy": rate(correct, len(labels)),
        "predictions": predictions,
    }


def compute_metrics(
    manifest: list[dict[str, Any]],
    accepted: dict[tuple[str, str], dict[str, Any]],
    integrity_counts: dict[str, int],
    sessions: list[dict[str, Any]],
    analysis_cfg: dict[str, Any],
) -> dict[str, Any]:
    seed = int(analysis_cfg["bootstrap_seed"])
    reps = int(analysis_cfg["bootstrap_resamples"])
    pairs = {m["pair_id"]: m for m in manifest}
    human = {pid: m["human"]["aggregate"] for pid, m in pairs.items()}
    resolved = sorted(pid for pid, h in human.items() if h in RESOLVED)

    def verdict(pid: str, order: str) -> str | None:
        rec = accepted.get((pid, order))
        if rec is None or rec["status"] != "valid":
            return None
        out: str = rec["mapped_verdict"]
        return out

    def displayed(pid: str, order: str) -> str | None:
        rec = accepted.get((pid, order))
        if rec is None or rec["status"] != "valid":
            return None
        out: str = rec["verdict_displayed"]
        return out

    # ---- Human agreement -------------------------------------------------------------
    def acc_stat(ids: list[str]) -> float | None:
        n = k = 0
        for pid in ids:
            for o in ORDERS:
                v = verdict(pid, o)
                if v is not None:
                    n += 1
                    k += _correct(v, human[pid])
        return k / n if n else None

    def kappa_stat(ids: list[str]) -> float | None:
        h, j = [], []
        for pid in ids:
            for o in ORDERS:
                v = verdict(pid, o)
                if v is not None:
                    h.append(human[pid])
                    j.append(v)
        return cohen_kappa(h, j, JUDGE_LABELS)

    valid_res = [
        (pid, o, v) for pid in resolved for o in ORDERS if (v := verdict(pid, o)) is not None
    ]
    correct_res = sum(_correct(v, human[pid]) for pid, _, v in valid_res)
    by_order: dict[str, Any] = {}
    for o in ORDERS:
        vs = [(pid, v) for pid in resolved if (v := verdict(pid, o)) is not None]
        by_order[o] = {
            "accuracy": rate(sum(_correct(v, human[pid]) for pid, v in vs), len(vs)),
            "coverage": rate(len(vs), len(resolved)),
            "judge_tie_verdicts": sum(v == "tie" for _, v in vs),
        }

    def acc_diff_stat(ids: list[str]) -> float | None:
        n = d = 0
        for pid in ids:
            vo, vs_ = verdict(pid, "original"), verdict(pid, "swapped")
            if vo is not None and vs_ is not None:
                n += 1
                d += int(_correct(vo, human[pid])) - int(_correct(vs_, human[pid]))
        return d / n if n else None

    tie_pairs = sorted(pid for pid, h in human.items() if h == "tie")
    tie_cells = [v for pid in tie_pairs for o in ORDERS if (v := verdict(pid, o)) is not None]
    sample_dropped_votes = sum(
        sum(1 for v in pairs[pid]["human"]["votes"] if not v["retained"]) for pid in pairs
    )
    agreement = {
        "resolved_pairs": len(resolved),
        "main_accuracy": {
            **rate(correct_res, len(valid_res)),
            "pair_cluster_bootstrap": cluster_bootstrap(resolved, acc_stat, seed, reps),
        },
        "coverage": rate(len(valid_res), 2 * len(resolved)),
        "judge_tie_on_resolved": rate(sum(v == "tie" for _, _, v in valid_res), len(valid_res)),
        "by_order": by_order,
        "accuracy_original_minus_swapped": cluster_bootstrap(
            resolved, acc_diff_stat, seed + 1, reps
        ),
        "cohen_kappa": cluster_bootstrap(resolved, kappa_stat, seed + 2, reps),
        "human_label_counts": dict(Counter(human.values())),
        "human_tie_pairs": {
            "pairs": len(tie_pairs),
            "valid_judge_cells": len(tie_cells),
            "judge_verdicts": dict(Counter(tie_cells)),
            "note": "not scored as correct/incorrect",
        },
        "human_unresolved_pairs": sum(h == "unresolved" for h in human.values()),
        "excluded_annotations_in_sample": sample_dropped_votes,
    }

    # ---- Order robustness -------------------------------------------------------------
    all_ids = sorted(pairs)
    both = [(pid, verdict(pid, "original"), verdict(pid, "swapped")) for pid in all_ids]
    both_valid = [(pid, a, b) for pid, a, b in both if a is not None and b is not None]
    changed = [pid for pid, a, b in both_valid if a != b]
    decisions = {
        o: dict(Counter(v for pid in all_ids if (v := verdict(pid, o)) is not None)) for o in ORDERS
    }
    position: dict[str, Any] = {}
    pooled: Counter[str] = Counter()
    for o in ORDERS:
        c = Counter(d for pid in all_ids if (d := displayed(pid, o)) is not None)
        pooled.update(c)
        n = sum(c.values())
        position[o] = {
            "first_A": rate(c["A"], n),
            "second_B": rate(c["B"], n),
            "tie": rate(c["tie"], n),
        }
    n_pooled = sum(pooled.values())
    position["pooled"] = {
        "first_A": rate(pooled["A"], n_pooled),
        "second_B": rate(pooled["B"], n_pooled),
        "tie": rate(pooled["tie"], n_pooled),
        "first_among_decisive": rate(pooled["A"], pooled["A"] + pooled["B"]),
    }
    resolved_set = set(resolved)
    consistent = [(pid, a) for pid, a, b in both_valid if a == b and pid in resolved_set]
    consistent_correct = sum(_correct(a, human[pid]) for pid, a in consistent)
    order = {
        "pairs_total": len(all_ids),
        "pairs_two_valid": len(both_valid),
        "order_inconsistency": rate(len(changed), len(both_valid)),
        "changed_pair_ids": changed,
        "mapped_decisions_by_order": decisions,
        "position_choice": position,
        "consistent_accuracy": {
            "accuracy": rate(consistent_correct, len(consistent)),
            "coverage": rate(len(consistent), len(resolved)),
        },
        "all_valid_accuracy": {
            "accuracy": agreement["main_accuracy"],
            "coverage": agreement["coverage"],
        },
    }

    # ---- Verbosity association ----------------------------------------------------------
    unequal = sorted(pid for pid, m in pairs.items() if m["longer_candidate"] != "equal")
    longer = {pid: pairs[pid]["longer_candidate"] for pid in unequal}
    j_cells = [(pid, v) for pid in unequal for o in ORDERS if (v := verdict(pid, o)) is not None]
    j_decisive = [(pid, v) for pid, v in j_cells if v != "tie"]
    j_longer = sum(v == longer[pid] for pid, v in j_decisive)
    h_res = [pid for pid in unequal if human[pid] in RESOLVED]
    h_longer = sum(human[pid] == longer[pid] for pid in h_res)

    def longer_diff(ids: list[str]) -> float | None:
        jn = jk = hn = hk = 0
        for pid in ids:
            hn += 1
            hk += human[pid] == longer[pid]
            for o in ORDERS:
                v = verdict(pid, o)
                if v is not None and v != "tie":
                    jn += 1
                    jk += v == longer[pid]
        if not jn or not hn:
            return None
        return jk / jn - hk / hn

    verbosity = {
        "length_rule": "whitespace tokens (len(text.split())), frozen in manifest",
        "unequal_length_pairs": len(unequal),
        "equal_length_pairs": len(pairs) - len(unequal),
        "judge_longer_among_decisive": rate(j_longer, len(j_decisive)),
        "judge_longer_among_all_valid": rate(j_longer, len(j_cells)),
        "judge_tie_cells": len(j_cells) - len(j_decisive),
        "human_longer_among_resolved": rate(h_longer, len(h_res)),
        "human_tie_or_unresolved_unequal_pairs": len(unequal) - len(h_res),
        "judge_minus_human_longer_rate_on_resolved_unequal": cluster_bootstrap(
            h_res, longer_diff, seed + 3, reps
        ),
        "interpretation": "association only; not causal evidence of length bias",
    }

    # ---- Operational --------------------------------------------------------------------
    attempted = list(accepted.values())
    lat = [float(r["latency_s"]) for r in attempted if r.get("latency_s") is not None]
    ptok = [int(r["prompt_tokens"]) for r in attempted if r.get("prompt_tokens") is not None]
    ctok = [
        int(r["completion_tokens"]) for r in attempted if r.get("completion_tokens") is not None
    ]
    invalid_reasons = Counter(
        str(r.get("error")).split(":")[0] for r in attempted if r["status"] != "valid"
    )
    operational = {
        "cell_counts": integrity_counts,
        "latency_s": {
            "n": len(lat),
            "median": percentile(lat, 50),
            "p95": percentile(lat, 95),
            "max": max(lat) if lat else None,
        },
        "prompt_tokens": _token_summary(ptok),
        "completion_tokens": _token_summary(ctok),
        "failure_reasons": dict(invalid_reasons),
        "sessions": [
            {
                k: s.get(k)
                for k in ("session_start_utc", "session_end_utc", "cells_attempted", "aborted")
            }
            for s in sessions
        ],
        "cost_usd": 0.0,
        "cost_note": "local inference; no API cost. Electricity and hardware time not priced.",
    }

    labels = {pid: human[pid] for pid in resolved}
    baseline = lopo_majority_baseline(labels)
    baseline.pop("predictions")
    return {
        "metrics_version": METRICS_VERSION,
        "definitions": DEFINITIONS,
        "human_agreement": agreement,
        "order_robustness": order,
        "verbosity": verbosity,
        "operational": operational,
        "baseline": baseline,
    }


def _token_summary(values: list[int]) -> dict[str, Any]:
    return {
        "n": len(values),
        "total": sum(values),
        "median": percentile([float(v) for v in values], 50),
        "p95": percentile([float(v) for v in values], 95),
        "max": max(values) if values else None,
    }
