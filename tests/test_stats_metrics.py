"""Statistics and registered metrics on hand-computable SAMPLE DATA fixtures."""

from __future__ import annotations

from typing import Any

import pytest

from llm_judge_audit.metrics import compute_metrics, lopo_majority_baseline
from llm_judge_audit.prompt import map_verdict
from llm_judge_audit.sample_data import synthetic_manifest
from llm_judge_audit.stats import cluster_bootstrap, cohen_kappa, rate, wilson

ANALYSIS = {"bootstrap_seed": 1, "bootstrap_resamples": 300}


def test_wilson_known_values() -> None:
    lo, hi = wilson(8, 10) or (0, 0)
    assert lo == pytest.approx(0.4902, abs=1e-4)
    assert hi == pytest.approx(0.9433, abs=1e-4)
    assert wilson(0, 0) is None
    assert rate(0, 0) == {"numerator": 0, "denominator": 0, "rate": None, "wilson95": None}
    lo0, _ = wilson(0, 5) or (1, 1)
    assert lo0 == 0.0


def test_kappa_known_values() -> None:
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"], ["a", "b"]) == 1.0
    # p_o = 0.5, p_e = 0.5 -> 0
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"], ["a", "b"]) == 0.0
    assert cohen_kappa(["a", "a"], ["a", "a"], ["a", "b"]) is None
    k = cohen_kappa(["a", "a", "a", "b"], ["a", "a", "b", "b"], ["a", "b"])
    assert k == pytest.approx(0.5)


def test_bootstrap_is_deterministic_and_resamples_clusters() -> None:
    data = {"p1": [1, 1], "p2": [0, 0], "p3": [1, 0]}
    seen: list[list[str]] = []

    def stat(ids: list[str]) -> float | None:
        seen.append(ids)
        vals = [v for i in ids for v in data[i]]
        return sum(vals) / len(vals)

    a = cluster_bootstrap(list(data), stat, seed=5, resamples=100)
    b = cluster_bootstrap(list(data), stat, seed=5, resamples=100)
    assert a == b
    assert a["estimate"] == pytest.approx(0.5)
    assert all(len(ids) == 3 for ids in seen)


def test_lopo_baseline_never_uses_own_label() -> None:
    out = lopo_majority_baseline({"a": "cand_1", "b": "cand_1", "c": "cand_2"})
    # a: others {c1:1,c2:1} -> cand_1 (tie rule); b: same; c: others {c1:2} -> cand_1
    assert out["predictions"] == {"a": "cand_1", "b": "cand_1", "c": "cand_1"}
    assert out["accuracy"]["numerator"] == 2


def _cell(pid: str, order: str, verdict: str | None, status: str = "valid") -> dict[str, Any]:
    return {
        "pair_id": pid,
        "order": order,
        "status": status,
        "verdict_displayed": verdict,
        "mapped_verdict": map_verdict(verdict, order) if verdict else None,
        "latency_s": 1.0,
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "error": None if status == "valid" else "x",
    }


def test_metrics_hand_computed() -> None:
    manifest = synthetic_manifest(6)  # aggregates: c1, c2, c1, c2, tie, unresolved
    ids = [m["pair_id"] for m in manifest]
    cells = {
        # p0 human cand_1: consistent correct (A then B)
        (ids[0], "original"): _cell(ids[0], "original", "A"),
        (ids[0], "swapped"): _cell(ids[0], "swapped", "B"),
        # p1 human cand_2: always first position -> inconsistent, one correct
        (ids[1], "original"): _cell(ids[1], "original", "A"),
        (ids[1], "swapped"): _cell(ids[1], "swapped", "A"),
        # p2 human cand_1: one invalid, one tie
        (ids[2], "original"): _cell(ids[2], "original", None, "invalid_output"),
        (ids[2], "swapped"): _cell(ids[2], "swapped", "tie"),
        # p3 human cand_2: consistent wrong
        (ids[3], "original"): _cell(ids[3], "original", "A"),
        (ids[3], "swapped"): _cell(ids[3], "swapped", "B"),
        # p4 human tie
        (ids[4], "original"): _cell(ids[4], "original", "A"),
        # p5 unresolved: missing entirely
    }
    counts = {
        "expected": 12,
        "attempted": 9,
        "valid": 8,
        "invalid_output": 1,
        "runtime_error": 0,
        "timeout": 0,
        "missing": 3,
        "complete": 8,
    }
    m = compute_metrics(manifest, cells, counts, [], ANALYSIS)
    ag = m["human_agreement"]
    assert ag["resolved_pairs"] == 4
    # valid resolved cells: p0 x2 (correct), p1 x2 (swapped A -> cand_2 correct), p2 x1 (tie),
    # p3 x2 (wrong) -> 3 correct / 7
    assert ag["main_accuracy"]["numerator"] == 3
    assert ag["main_accuracy"]["denominator"] == 7
    assert ag["coverage"]["numerator"] == 7 and ag["coverage"]["denominator"] == 8
    assert ag["by_order"]["original"]["accuracy"]["denominator"] == 3
    assert ag["by_order"]["original"]["coverage"]["numerator"] == 3
    assert ag["judge_tie_on_resolved"]["numerator"] == 1
    assert ag["human_tie_pairs"]["pairs"] == 1
    assert ag["human_unresolved_pairs"] == 1
    order = m["order_robustness"]
    # pairs with two valid: p0, p1, p3 -> p1 changed
    assert order["order_inconsistency"]["numerator"] == 1
    assert order["order_inconsistency"]["denominator"] == 3
    cons = order["consistent_accuracy"]
    assert cons["accuracy"]["numerator"] == 1 and cons["accuracy"]["denominator"] == 2
    assert cons["coverage"]["denominator"] == 4
    pos = order["position_choice"]["pooled"]
    assert pos["first_A"]["numerator"] == 5 and pos["first_A"]["denominator"] == 8
    assert m["operational"]["cell_counts"] == counts
    assert m["baseline"]["accuracy"]["denominator"] == 4


def test_verbosity_counts() -> None:
    manifest = synthetic_manifest(6)
    for mm in manifest:
        mm["longer_candidate"] = "cand_1"
    manifest[5]["longer_candidate"] = "equal"
    ids = [mm["pair_id"] for mm in manifest]
    cells = {
        (ids[0], "original"): _cell(ids[0], "original", "A"),
        (ids[1], "original"): _cell(ids[1], "original", "B"),
        (ids[2], "original"): _cell(ids[2], "original", "tie"),
    }
    m = compute_metrics(manifest, cells, {}, [], ANALYSIS)["verbosity"]
    assert m["unequal_length_pairs"] == 5
    assert m["judge_longer_among_decisive"]["numerator"] == 1
    assert m["judge_longer_among_decisive"]["denominator"] == 2
    assert m["judge_longer_among_all_valid"]["denominator"] == 3
    # human resolved unequal: p0 c1, p1 c2, p2 c1, p3 c2 -> longer (c1) won 2/4
    assert m["human_longer_among_resolved"]["numerator"] == 2
    assert m["human_longer_among_resolved"]["denominator"] == 4
