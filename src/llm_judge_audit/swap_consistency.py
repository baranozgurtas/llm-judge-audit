"""Supplementary swap-consistency aggregation (swap-consistency-v1). Exploratory and post-hoc.

Rule, applied per pair to the two order-specific verdicts *after* each is mapped back to the
original candidate identities (cand_1 / cand_2 / tie):

- both orders name the same candidate -> that candidate is the aggregated prediction;
- both orders say tie                 -> tie (a judge tie on a resolved pair scores incorrect);
- the mapped verdicts differ          -> abstained (no prediction);
- either cell lacks a valid verdict   -> unavailable (no prediction; reported separately).

Predictions are built from judge verdicts only; human labels are read afterwards, solely to
score. This module never edits metrics.json or any registered artifact; `stats.py` (frozen) is
imported, not modified.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Any

from llm_judge_audit.io_utils import sha256_text
from llm_judge_audit.prompt import render_cell
from llm_judge_audit.stats import cluster_bootstrap, rate

SWAP_CONSISTENCY_VERSION = "swap-consistency-v1"
# Declared before this analysis was computed: registered bootstrap_seed (7331) + 200, so it
# collides with neither the registered seeds (7331-7334) nor supplementary-v1 (7431-7437).
BOOTSTRAP_SEED = 7531
ORDERS = ("original", "swapped")
RESOLVED = ("cand_1", "cand_2")
JUDGE_LABELS = ("cand_1", "cand_2", "tie")
# Written out independently of prompt.DISPLAY_MAPPING so the stored mapping is cross-checked.
EXPECTED_MAPPING: dict[str, dict[str, str]] = {
    "original": {"A": "cand_1", "B": "cand_2"},
    "swapped": {"A": "cand_2", "B": "cand_1"},
}

COVERED, ABSTAINED, UNAVAILABLE = "covered", "abstained", "unavailable"

DEFINITIONS: dict[str, str] = {
    "rule": "per pair: both mapped-back verdicts equal -> that verdict (cand_1, cand_2 or tie); "
    "mapped-back verdicts differ -> abstained; a missing/failed cell -> unavailable. Human "
    "labels are not inputs to the rule.",
    "resolved_pairs": "pairs whose human aggregate is cand_1 or cand_2. Human-tie and "
    "unresolved pairs are reported separately and never scored.",
    "accuracy_among_covered": "correct aggregated predictions / covered resolved pairs. An "
    "aggregated tie on a resolved pair is incorrect.",
    "coverage": "covered resolved pairs / all resolved pairs.",
    "accuracy_all_resolved": "correct aggregated predictions / all resolved pairs; abstained "
    "and unavailable pairs count as unanswered (not correct).",
    "original_order_accuracy": "correct original-order mapped verdicts / all resolved pairs "
    "(same pair set; a missing original cell counts as unanswered).",
    "differences": "paired, on the same resolved pairs: aggregated minus original-order. "
    "Pair-level percentile bootstrap.",
    "intervals": "Wilson 95% for every rate (pairs are the units, so cells are not double "
    "counted) and a percentile pair-level bootstrap 95% for accuracy and coverage.",
}


class MappingError(ValueError):
    """Raised when stored records do not support an unambiguous answer-identity mapping."""


def map_back(displayed: str, order: str) -> str:
    """Displayed verdict (A/B/tie) -> original candidate identity, from EXPECTED_MAPPING."""
    if displayed == "tie":
        return "tie"
    return EXPECTED_MAPPING[order][displayed]


def aggregate_pair(original: str | None, swapped: str | None) -> tuple[str, str | None]:
    """(status, prediction) from the two mapped-back verdicts; None means no valid verdict."""
    if original is None or swapped is None:
        return UNAVAILABLE, None
    for v in (original, swapped):
        if v not in JUDGE_LABELS:
            raise MappingError(f"unexpected mapped verdict {v!r}")
    if original == swapped:
        return COVERED, original
    return ABSTAINED, None


def verify_mapping(
    manifest: list[dict[str, Any]], accepted: dict[tuple[str, str], dict[str, Any]]
) -> dict[str, Any]:
    """Fail loudly unless every valid cell maps back unambiguously to the original answers.

    Checks per cell: the stored display mapping equals EXPECTED_MAPPING, the stored mapped
    verdict equals an independent re-mapping of the displayed verdict, and the stored rendered-
    prompt hash equals a fresh render of the manifest pair in that order (so the answer shown
    as A/B really was the claimed candidate).
    """
    pairs = {m["pair_id"]: m for m in manifest}
    checked = prompt_checked = 0
    for (pid, order), rec in sorted(accepted.items()):
        if pid not in pairs or order not in ORDERS:
            raise MappingError(f"cell {(pid, order)} is not in the manifest schedule")
        if rec.get("display_mapping") != EXPECTED_MAPPING[order]:
            raise MappingError(f"{pid}/{order}: display mapping {rec.get('display_mapping')}")
        if rec["status"] != "valid":
            continue
        if rec["mapped_verdict"] != map_back(rec["verdict_displayed"], order):
            raise MappingError(f"{pid}/{order}: mapped verdict disagrees with displayed verdict")
        stored = rec.get("rendered_prompt_sha256")
        if stored is not None:
            if sha256_text(render_cell(pairs[pid], order).prompt) != stored:
                raise MappingError(f"{pid}/{order}: rendered prompt hash does not match")
            prompt_checked += 1
        checked += 1
    return {
        "valid_cells_checked": checked,
        "rendered_prompt_hashes_matched": prompt_checked,
        "expected_mapping": EXPECTED_MAPPING,
    }


def aggregate_predictions(
    accepted: dict[tuple[str, str], dict[str, Any]], pair_ids: list[str]
) -> dict[str, dict[str, Any]]:
    """Per-pair aggregation from judge records only (no human-label argument by design)."""

    def verdict(pid: str, order: str) -> str | None:
        rec = accepted.get((pid, order))
        if rec is None or rec["status"] != "valid":
            return None
        return map_back(rec["verdict_displayed"], order)

    out: dict[str, dict[str, Any]] = {}
    for pid in sorted(pair_ids):
        o, s = verdict(pid, "original"), verdict(pid, "swapped")
        status, pred = aggregate_pair(o, s)
        out[pid] = {"original": o, "swapped": s, "status": status, "prediction": pred}
    return out


def _status_counts(preds: dict[str, dict[str, Any]], ids: list[str]) -> dict[str, Any]:
    status = Counter(preds[pid]["status"] for pid in ids)
    return {
        "pairs": len(ids),
        COVERED: status[COVERED],
        ABSTAINED: status[ABSTAINED],
        UNAVAILABLE: status[UNAVAILABLE],
        "covered_predictions": dict(
            sorted(
                Counter(preds[pid]["prediction"] for pid in ids if preds[pid]["prediction"]).items()
            )
        ),
    }


def swap_consistency_analysis(
    manifest: list[dict[str, Any]],
    accepted: dict[tuple[str, str], dict[str, Any]],
    resamples: int,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    mapping_check = verify_mapping(manifest, accepted)
    all_ids = sorted(m["pair_id"] for m in manifest)
    preds = aggregate_predictions(accepted, all_ids)  # built before any label is read

    human = {m["pair_id"]: m["human"]["aggregate"] for m in manifest}
    unknown = {h for h in human.values() if h not in (*RESOLVED, "tie", "unresolved")}
    if unknown:
        raise MappingError(f"unexpected human aggregate values {sorted(unknown)}")
    resolved = [pid for pid in all_ids if human[pid] in RESOLVED]
    ties = [pid for pid in all_ids if human[pid] == "tie"]
    unresolved = [pid for pid in all_ids if human[pid] == "unresolved"]

    def covered(ids: list[str]) -> list[str]:
        return [pid for pid in ids if preds[pid]["status"] == COVERED]

    def n_correct(ids: list[str]) -> int:
        return sum(preds[pid]["prediction"] == human[pid] for pid in covered(ids))

    def n_orig_correct(ids: list[str]) -> int:
        return sum(preds[pid]["original"] == human[pid] for pid in ids)

    # Bootstrap statistics: each takes a resample of resolved pair IDs (with repeats).
    def acc_covered(ids: list[str]) -> float | None:
        c = covered(ids)
        return n_correct(ids) / len(c) if c else None

    def coverage(ids: list[str]) -> float | None:
        return len(covered(ids)) / len(ids) if ids else None

    def acc_all(ids: list[str]) -> float | None:
        return n_correct(ids) / len(ids) if ids else None

    def acc_orig(ids: list[str]) -> float | None:
        return n_orig_correct(ids) / len(ids) if ids else None

    def diff(a: Callable[[list[str]], float | None]) -> Callable[[list[str]], float | None]:
        def stat(ids: list[str]) -> float | None:
            x, y = a(ids), acc_orig(ids)
            return None if x is None or y is None else x - y

        return stat

    def boot(stat: Callable[[list[str]], float | None], i: int) -> dict[str, Any]:
        return cluster_bootstrap(resolved, stat, seed + i, resamples)

    cov_res = covered(resolved)
    orig_valid = [pid for pid in resolved if preds[pid]["original"] is not None]
    abst_res = [pid for pid in resolved if preds[pid]["status"] == ABSTAINED]
    n_res = len(resolved)
    primary = {
        "counts": _status_counts(preds, resolved),
        "accuracy_among_covered": {
            **rate(n_correct(resolved), len(cov_res)),
            "pair_bootstrap": boot(acc_covered, 0),
        },
        "coverage": {**rate(len(cov_res), n_res), "pair_bootstrap": boot(coverage, 1)},
        "accuracy_all_resolved": {
            **rate(n_correct(resolved), n_res),
            "pair_bootstrap": boot(acc_all, 2),
        },
        "aggregated_tie_predictions_scored_incorrect": sum(
            preds[pid]["prediction"] == "tie" for pid in cov_res
        ),
    }
    comparison = {
        "original_order_accuracy": {
            **rate(n_orig_correct(resolved), n_res),
            "pair_bootstrap": boot(acc_orig, 3),
        },
        "original_order_coverage": rate(len(orig_valid), n_res),
        "accuracy_among_covered_minus_original": boot(diff(acc_covered), 4),
        "accuracy_all_resolved_minus_original": boot(diff(acc_all), 5),
        "original_order_on_covered_pairs": rate(n_orig_correct(cov_res), len(cov_res)),
        "original_order_on_abstained_pairs": rate(n_orig_correct(abst_res), len(abst_res)),
        "note": "On covered pairs the aggregated prediction equals the original-order verdict "
        "by construction, so any accuracy-among-covered gain comes only from abstaining on "
        "the order-inconsistent pairs.",
    }

    def not_scored(ids: list[str], label: str) -> dict[str, Any]:
        return {**_status_counts(preds, ids), "human_aggregate": label, "scored": False}

    return {
        "analysis_version": SWAP_CONSISTENCY_VERSION,
        "status": "EXPLORATORY / POST-HOC supplementary analysis, not preregistered; registered "
        "metrics-v1 are unchanged",
        "definitions": DEFINITIONS,
        "bootstrap": {
            "seed_base": seed,
            "seeds": {
                "accuracy_among_covered": seed,
                "coverage": seed + 1,
                "accuracy_all_resolved": seed + 2,
                "original_order_accuracy": seed + 3,
                "accuracy_among_covered_minus_original": seed + 4,
                "accuracy_all_resolved_minus_original": seed + 5,
            },
            "resamples": resamples,
            "unit": "pair ID (resolved pairs only; both orders travel with their pair)",
            "method": "percentile 2.5/97.5, llm_judge_audit.stats.cluster_bootstrap",
        },
        "mapping_verification": mapping_check,
        "all_pairs": _status_counts(preds, all_ids),
        "resolved": primary,
        "comparison_original_order": comparison,
        "human_tie_pairs": not_scored(ties, "tie"),
        "human_unresolved_pairs": not_scored(unresolved, "unresolved"),
        "pairs": [{"pair_id": pid, **preds[pid], "human_aggregate": human[pid]} for pid in all_ids],
    }


def registered_cross_checks(result: dict[str, Any], registered: dict[str, Any]) -> dict[str, bool]:
    """The rule coincides with registered consistent-pair figures; confirm they agree."""
    o = registered["order_robustness"]
    r = result["resolved"]

    def same(a: dict[str, Any], b: dict[str, Any]) -> bool:
        return (a["numerator"], a["denominator"]) == (b["numerator"], b["denominator"])

    return {
        "accuracy_among_covered_equals_registered_consistent_accuracy": same(
            r["accuracy_among_covered"], o["consistent_accuracy"]["accuracy"]
        ),
        "coverage_equals_registered_consistent_coverage": same(
            r["coverage"], o["consistent_accuracy"]["coverage"]
        ),
        "abstained_all_pairs_equals_registered_order_inconsistency": (
            result["all_pairs"][ABSTAINED] == o["order_inconsistency"]["numerator"]
        ),
        "original_order_accuracy_equals_registered": same(
            result["comparison_original_order"]["original_order_accuracy"],
            registered["human_agreement"]["by_order"]["original"]["accuracy"],
        ),
    }
