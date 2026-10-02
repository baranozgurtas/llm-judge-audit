"""Supplementary analysis v1: verbosity-statistic audit and interval-method audit.

Post-hoc and additive. It recomputes figures from the immutable cell log and manifest, verifies
them against the registered metrics-v1 artifact, and adds clearly labelled supplementary pair-
cluster bootstrap intervals for cell-level rates whose registered Wilson interval treats the two
order judgments of a pair as independent. It never edits metrics.json or replaces a registered
metric; `metrics.py` and `stats.py` (frozen) are imported, not modified.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from llm_judge_audit.stats import cluster_bootstrap, rate

SUPPLEMENTARY_VERSION = "supplementary-analysis-v1"
ORDERS = ("original", "swapped")
RESOLVED = ("cand_1", "cand_2")
SEED_OFFSET = 100  # supplementary seeds = registered bootstrap_seed + 100 + i (declared here)

VERBOSITY_FORMULAS = {
    "judge_longer_all_unequal": "sum over the 196 unequal-length pairs and both orders of "
    "1[mapped verdict = longer candidate], divided by the number of those cells whose verdict "
    "is decisive (not tie). Registered as verbosity.judge_longer_among_decisive.",
    "human_longer_resolved_unequal": "number of unequal-length pairs with a resolved human "
    "aggregate whose aggregate is the longer candidate, divided by the number of such pairs. "
    "Registered as verbosity.human_longer_among_resolved.",
    "judge_longer_resolved_unequal": "judge rate as above, restricted to the same resolved "
    "unequal-length pairs as the human rate. Not stored separately in metrics-v1; it is the "
    "first term of the registered paired difference.",
    "paired_difference": "judge_longer_resolved_unequal - human_longer_resolved_unequal, on the "
    "same pairs. Registered as verbosity.judge_minus_human_longer_rate_on_resolved_unequal; "
    "interval = percentile pair-cluster bootstrap (resample resolved unequal pair IDs, both "
    "orders kept with their pair).",
    "raw_headline_difference": "judge_longer_all_unequal - human_longer_resolved_unequal. The two "
    "rates use different pair sets (all unequal vs resolved unequal), so this is a descriptive "
    "contrast of the headline numbers, not a paired comparison.",
}


def _verdict(accepted: dict[tuple[str, str], dict[str, Any]], pid: str, order: str) -> str | None:
    rec = accepted.get((pid, order))
    if rec is None or rec["status"] != "valid":
        return None
    out: str = rec["mapped_verdict"]
    return out


def _judge_longer(
    pids: list[str], accepted: dict[tuple[str, str], dict[str, Any]], longer: dict[str, str]
) -> tuple[int, int, int]:
    """(cells choosing longer, decisive valid cells, tie cells) over both orders of `pids`."""
    k = n = ties = 0
    for pid in pids:
        for order in ORDERS:
            v = _verdict(accepted, pid, order)
            if v is None:
                continue
            if v == "tie":
                ties += 1
                continue
            n += 1
            k += v == longer[pid]
    return k, n, ties


def reproduce_bootstrap(
    cluster_ids: list[str], statistic: Callable[[list[str]], float | None], seed: int, reps: int
) -> dict[str, Any]:
    """Independent re-implementation of the registered percentile pair-cluster bootstrap."""
    ids = sorted(cluster_ids)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(reps):
        draw = [ids[i] for i in rng.integers(0, len(ids), size=len(ids))]
        v = statistic(draw)
        if v is not None:
            values.append(v)
    lo, hi = np.percentile(np.array(values), [2.5, 97.5])
    return {"estimate": statistic(ids), "ci95": [float(lo), float(hi)]}


def _close(a: Any, b: Any) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, float) or isinstance(b, float):
        return bool(np.isclose(a, b, rtol=0, atol=1e-12))
    return bool(a == b)


def supplementary_analysis(
    manifest: list[dict[str, Any]],
    accepted: dict[tuple[str, str], dict[str, Any]],
    registered: dict[str, Any],
    analysis_cfg: dict[str, Any],
) -> dict[str, Any]:
    seed = int(analysis_cfg["bootstrap_seed"])
    reps = int(analysis_cfg["bootstrap_resamples"])
    sup_seed = seed + SEED_OFFSET
    pairs = {m["pair_id"]: m for m in manifest}
    human: dict[str, str] = {pid: m["human"]["aggregate"] for pid, m in pairs.items()}
    longer: dict[str, str] = {pid: m["longer_candidate"] for pid, m in pairs.items()}
    unequal = sorted(pid for pid in pairs if longer[pid] != "equal")
    res_unequal = [pid for pid in unequal if human[pid] in RESOLVED]
    other_unequal = [pid for pid in unequal if human[pid] not in RESOLVED]
    resolved = sorted(pid for pid in pairs if human[pid] in RESOLVED)

    # ---- 1. Verbosity decomposition ------------------------------------------------------
    jk_all, jn_all, jt_all = _judge_longer(unequal, accepted, longer)
    jk_res, jn_res, jt_res = _judge_longer(res_unequal, accepted, longer)
    jk_oth, jn_oth, jt_oth = _judge_longer(other_unequal, accepted, longer)
    hk = sum(human[pid] == longer[pid] for pid in res_unequal)
    hn = len(res_unequal)
    judge_all, judge_res, human_rate = rate(jk_all, jn_all), rate(jk_res, jn_res), rate(hk, hn)

    def paired_diff(ids: list[str]) -> float | None:
        k, n, _ = _judge_longer(ids, accepted, longer)
        h = sum(human[pid] == longer[pid] for pid in ids)
        return (k / n - h / len(ids)) if n and ids else None

    def judge_rate_stat(ids: list[str]) -> float | None:
        k, n, _ = _judge_longer(ids, accepted, longer)
        return k / n if n else None

    def raw_diff_stat(ids: list[str]) -> float | None:
        k, n, _ = _judge_longer(ids, accepted, longer)
        res = [pid for pid in ids if human[pid] in RESOLVED]
        if not n or not res:
            return None
        return k / n - sum(human[pid] == longer[pid] for pid in res) / len(res)

    reg_v = registered["verbosity"]
    reg_paired = reg_v["judge_minus_human_longer_rate_on_resolved_unequal"]
    repro_paired = reproduce_bootstrap(res_unequal, paired_diff, seed + 3, reps)
    verbosity = {
        "formulas": VERBOSITY_FORMULAS,
        "pairs": {
            "unequal_length": len(unequal),
            "resolved_unequal": hn,
            "tie_or_unresolved_unequal": len(other_unequal),
        },
        "judge_longer_all_unequal": {
            **judge_all,
            "tie_cells": jt_all,
            "interval": "Wilson (cells as units)",
        },
        "judge_longer_resolved_unequal": {
            **judge_res,
            "tie_cells": jt_res,
            "interval": "Wilson (cells as units)",
        },
        "judge_longer_tie_or_unresolved_unequal": {**rate(jk_oth, jn_oth), "tie_cells": jt_oth},
        "human_longer_resolved_unequal": {**human_rate, "interval": "Wilson (pairs as units)"},
        "paired_difference": {
            "estimate": paired_diff(res_unequal),
            "registered": reg_paired,
            "reproduced": repro_paired,
            "interval": "registered percentile pair-cluster bootstrap, seed "
            f"{reg_paired['seed']}, {reg_paired['resamples']} resamples, unit = pair ID",
        },
        "raw_headline_difference": {
            "estimate": judge_all["rate"] - human_rate["rate"],
            "supplementary_pair_cluster_bootstrap": cluster_bootstrap(
                unequal, raw_diff_stat, sup_seed + 1, reps
            ),
            "note": "unpaired contrast of rates on different pair sets; supplementary only",
        },
        "checks": {
            "judge_all_matches_registered": _close(judge_all, reg_v["judge_longer_among_decisive"]),
            "human_matches_registered": _close(human_rate, reg_v["human_longer_among_resolved"]),
            "paired_estimate_matches_registered": _close(
                paired_diff(res_unequal), reg_paired["estimate"]
            ),
            "paired_ci_reproduced_exactly": _close(repro_paired["ci95"], reg_paired["ci95"]),
            "paired_estimate_equals_subset_judge_minus_human": _close(
                paired_diff(res_unequal), judge_res["rate"] - human_rate["rate"]
            ),
        },
    }

    # ---- 2. Registered bootstrap reproduction --------------------------------------------
    ag = registered["human_agreement"]

    def acc(ids: list[str]) -> float | None:
        cells = [
            (pid, v) for pid in ids for o in ORDERS if (v := _verdict(accepted, pid, o)) is not None
        ]
        return sum(v == human[pid] for pid, v in cells) / len(cells) if cells else None

    def acc_diff(ids: list[str]) -> float | None:
        d = [
            int(_verdict(accepted, p, "original") == human[p])
            - int(_verdict(accepted, p, "swapped") == human[p])
            for p in ids
            if _verdict(accepted, p, "original") is not None
            and _verdict(accepted, p, "swapped") is not None
        ]
        return sum(d) / len(d) if d else None

    def kappa(ids: list[str]) -> float | None:
        from llm_judge_audit.stats import cohen_kappa

        h, j = [], []
        for pid in ids:
            for o in ORDERS:
                v = _verdict(accepted, pid, o)
                if v is not None:
                    h.append(human[pid])
                    j.append(v)
        return cohen_kappa(h, j, ("cand_1", "cand_2", "tie"))

    reproductions = {}
    for name, ids, fn, s, reg in (
        ("main_accuracy", resolved, acc, seed, ag["main_accuracy"]["pair_cluster_bootstrap"]),
        (
            "accuracy_original_minus_swapped",
            resolved,
            acc_diff,
            seed + 1,
            ag["accuracy_original_minus_swapped"],
        ),
        ("cohen_kappa", resolved, kappa, seed + 2, ag["cohen_kappa"]),
        ("verbosity_paired_difference", res_unequal, paired_diff, seed + 3, reg_paired),
    ):
        rep = reproduce_bootstrap(ids, fn, s, reps)
        reproductions[name] = {
            "seed": s,
            "resamples": reps,
            "unit": "pair ID (both orders kept together)",
            "n_clusters": len(ids),
            "registered": {"estimate": reg["estimate"], "ci95": reg["ci95"]},
            "reproduced": rep,
            "matches": _close(rep["estimate"], reg["estimate"])
            and _close(rep["ci95"], reg["ci95"]),
        }

    # ---- 3. Interval-method audit + supplementary clustered intervals --------------------
    pc = registered["order_robustness"]["position_choice"]["pooled"]
    all_ids = sorted(pairs)

    def displayed_first(ids: list[str], decisive_only: bool) -> float | None:
        k = n = 0
        for pid in ids:
            for o in ORDERS:
                rec = accepted.get((pid, o))
                if rec is None or rec["status"] != "valid":
                    continue
                d = rec["verdict_displayed"]
                if decisive_only and d == "tie":
                    continue
                n += 1
                k += d == "A"
        return k / n if n else None

    def judge_tie(ids: list[str]) -> float | None:
        vs = [v for pid in ids for o in ORDERS if (v := _verdict(accepted, pid, o)) is not None]
        return sum(v == "tie" for v in vs) / len(vs) if vs else None

    clustered = {}
    for i, (name, reg, ids, fn) in enumerate(
        (
            (
                "human_agreement.judge_tie_on_resolved",
                ag["judge_tie_on_resolved"],
                resolved,
                judge_tie,
            ),
            (
                "order_robustness.position_choice.pooled.first_A",
                pc["first_A"],
                all_ids,
                lambda ids: displayed_first(ids, False),
            ),
            (
                "order_robustness.position_choice.pooled.first_among_decisive",
                pc["first_among_decisive"],
                all_ids,
                lambda ids: displayed_first(ids, True),
            ),
            (
                "verbosity.judge_longer_among_decisive",
                reg_v["judge_longer_among_decisive"],
                unequal,
                judge_rate_stat,
            ),
            (
                "verbosity.judge_longer_resolved_unequal (not registered)",
                judge_res,
                res_unequal,
                judge_rate_stat,
            ),
        ),
        start=2,
    ):
        boot = cluster_bootstrap(ids, fn, sup_seed + i, reps)
        clustered[name] = {
            "numerator": reg["numerator"],
            "denominator": reg["denominator"],
            "rate": reg["rate"],
            "registered_wilson95": reg.get("wilson95"),
            "supplementary_pair_cluster_bootstrap": boot,
            "n_pairs": len(ids),
        }

    interval_audit = [
        {
            "metric": "Rates with one unit per pair (order inconsistency, consistent-pair "
            "accuracy/coverage, per-order accuracy/coverage/position, human longer rate, "
            "baseline)",
            "registered_method": "Wilson score 95%",
            "unit": "pair",
            "assessment": "appropriate: units are distinct pairs",
        },
        {
            "metric": "Cell-level rates pooling both orders (main accuracy, coverage, judge tie on "
            "resolved, pooled position choice, judge longer rate)",
            "registered_method": "Wilson score 95%",
            "unit": "cell (2 per pair)",
            "assessment": "registered as specified (Wilson for rates) but treats a pair's two "
            "order judgments as independent, so it can be too narrow; main accuracy also has a "
            "registered pair-cluster bootstrap; supplementary pair-cluster bootstraps given for "
            "the others (coverage is 308/308, a degenerate rate, so none is added)",
        },
        {
            "metric": "Kappa, main-accuracy bootstrap, accuracy original-minus-swapped, verbosity "
            "paired difference",
            "registered_method": "percentile pair-cluster bootstrap, 2000 "
            "resamples, seeds 7331-7334",
            "unit": "pair ID; both orders resampled together",
            "assessment": "matches the specification; reproduced exactly from raw records",
        },
    ]
    return {
        "supplementary_version": SUPPLEMENTARY_VERSION,
        "status": "post-hoc supplementary analysis; registered metrics-v1 are unchanged",
        "registered_metrics_version": registered["metrics_version"],
        "supplementary_seed_base": sup_seed,
        "resamples": reps,
        "verbosity": verbosity,
        "registered_bootstrap_reproduction": reproductions,
        "interval_audit": interval_audit,
        "supplementary_clustered_intervals": clustered,
    }
