"""Supplementary verbosity / interval audit: hand-computed SAMPLE DATA and the real saved run."""

from __future__ import annotations

import re
from typing import Any

import pytest

from llm_judge_audit.analysis import SUPPLEMENTARY_DIR, SUPPLEMENTARY_FILE
from llm_judge_audit.config import DATA_DERIVED, PROJECT_ROOT, RUNS_DIR, load_config
from llm_judge_audit.integrity import check_integrity
from llm_judge_audit.io_utils import read_json, read_jsonl, sha256_file
from llm_judge_audit.metrics import compute_metrics
from llm_judge_audit.prompt import map_verdict
from llm_judge_audit.sample_data import synthetic_manifest
from llm_judge_audit.supplementary import supplementary_analysis

ANALYSIS = {"bootstrap_seed": 11, "bootstrap_resamples": 300}


def _cell(pid: str, order: str, verdict: str) -> dict[str, Any]:
    return {
        "pair_id": pid,
        "order": order,
        "status": "valid",
        "verdict_displayed": verdict,
        "mapped_verdict": map_verdict(verdict, order),
        "latency_s": 1.0,
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "error": None,
    }


def _fixture() -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    """SAMPLE DATA: 6 pairs, all with cand_1 longer.

    Human aggregates: p0 c1, p1 c2, p2 c1, p3 c2, p4 tie, p5 unresolved.
    Judge picks the longer answer (cand_1) everywhere except p1 swapped and p3 both orders.
    """
    manifest = synthetic_manifest(6)
    for m in manifest:
        m["longer_candidate"] = "cand_1"
    ids = [m["pair_id"] for m in manifest]
    pick = {  # displayed verdict per (pair index, order)
        (0, "original"): "A",
        (0, "swapped"): "B",
        (1, "original"): "A",
        (1, "swapped"): "A",  # swapped A -> cand_2
        (2, "original"): "A",
        (2, "swapped"): "B",
        (3, "original"): "B",
        (3, "swapped"): "A",  # cand_2 both
        (4, "original"): "A",
        (4, "swapped"): "B",
        (5, "original"): "A",
        (5, "swapped"): "tie",
    }
    cells = {(ids[i], o): _cell(ids[i], o, v) for (i, o), v in pick.items()}
    return manifest, cells


def test_paired_vs_raw_difference_hand_computed() -> None:
    manifest, cells = _fixture()
    registered = compute_metrics(manifest, cells, {}, [], ANALYSIS)
    sup = supplementary_analysis(manifest, cells, registered, ANALYSIS)
    v = sup["verbosity"]
    # All 6 pairs unequal; decisive cells 11 (p5 swapped tie); longer chosen: 11 - 1 - 2 = 8
    assert (
        v["judge_longer_all_unequal"]["numerator"],
        v["judge_longer_all_unequal"]["denominator"],
    ) == (8, 11)
    # Resolved unequal pairs p0..p3: 8 decisive cells, longer chosen 8 - 1 - 2 = 5
    assert (
        v["judge_longer_resolved_unequal"]["numerator"],
        v["judge_longer_resolved_unequal"]["denominator"],
    ) == (5, 8)
    # Humans: p0, p2 chose cand_1 (longer) -> 2/4
    assert (
        v["human_longer_resolved_unequal"]["numerator"],
        v["human_longer_resolved_unequal"]["denominator"],
    ) == (2, 4)
    assert v["paired_difference"]["estimate"] == pytest.approx(5 / 8 - 2 / 4)
    assert v["raw_headline_difference"]["estimate"] == pytest.approx(8 / 11 - 2 / 4)
    reg_paired = registered["verbosity"]["judge_minus_human_longer_rate_on_resolved_unequal"]
    assert reg_paired["estimate"] == pytest.approx(5 / 8 - 2 / 4)
    assert all(v["checks"].values())
    assert all(r["matches"] for r in sup["registered_bootstrap_reproduction"].values())


def test_supplementary_is_deterministic_and_never_edits_registered() -> None:
    manifest, cells = _fixture()
    registered = compute_metrics(manifest, cells, {}, [], ANALYSIS)
    snapshot = repr(registered)
    a = supplementary_analysis(manifest, cells, registered, ANALYSIS)
    b = supplementary_analysis(manifest, cells, registered, ANALYSIS)
    assert a == b
    assert repr(registered) == snapshot


def test_cluster_bootstrap_keeps_both_orders_together() -> None:
    """Every resample's judge denominator must be twice the number of drawn pairs."""
    from llm_judge_audit.supplementary import _judge_longer, reproduce_bootstrap

    manifest, cells = _fixture()
    longer = {m["pair_id"]: m["longer_candidate"] for m in manifest}
    ids = [m["pair_id"] for m in manifest[:4]]  # no ties among these pairs
    seen: list[tuple[int, int]] = []

    def stat(draw: list[str]) -> float | None:
        _, n, _ = _judge_longer(draw, cells, longer)
        seen.append((len(draw), n))
        return 0.0

    reproduce_bootstrap(ids, stat, seed=3, reps=50)
    assert all(n == 2 * k for k, n in seen)


# ---- Real saved run (immutable records) ------------------------------------------------
RUN = RUNS_DIR / "bench-4e35d65ba747"
real = pytest.mark.skipif(not (RUN / "metrics.json").exists(), reason="no saved benchmark run")


@pytest.fixture(scope="module")
def real_sup() -> dict[str, Any]:
    manifest = read_jsonl(DATA_DERIVED / "manifest.jsonl")
    meta = read_json(RUN / "run_meta.json")
    report = check_integrity(
        RUN / "cells.jsonl", [m["pair_id"] for m in manifest], meta["identity"]
    )
    assert report["completion_state"] == "COMPLETE"
    out: dict[str, Any] = supplementary_analysis(
        manifest, report["accepted"], read_json(RUN / "metrics.json"), load_config()["analysis"]
    )
    return out


@real
def test_real_verbosity_counts_and_checks(real_sup: dict[str, Any]) -> None:
    v = real_sup["verbosity"]
    assert v["pairs"] == {
        "unequal_length": 196,
        "resolved_unequal": 151,
        "tie_or_unresolved_unequal": 45,
    }
    counts = {
        k: (v[k]["numerator"], v[k]["denominator"])
        for k in (
            "judge_longer_all_unequal",
            "judge_longer_resolved_unequal",
            "judge_longer_tie_or_unresolved_unequal",
            "human_longer_resolved_unequal",
        )
    }
    assert counts == {
        "judge_longer_all_unequal": (267, 387),
        "judge_longer_resolved_unequal": (204, 299),
        "judge_longer_tie_or_unresolved_unequal": (63, 88),
        "human_longer_resolved_unequal": (102, 151),
    }
    assert v["paired_difference"]["estimate"] == pytest.approx(204 / 299 - 102 / 151)
    assert v["raw_headline_difference"]["estimate"] == pytest.approx(267 / 387 - 102 / 151)
    assert round(100 * v["paired_difference"]["estimate"], 1) == 0.7
    assert round(100 * v["raw_headline_difference"]["estimate"], 1) == 1.4
    assert all(v["checks"].values())
    assert all(r["matches"] for r in real_sup["registered_bootstrap_reproduction"].values())


@real
def test_saved_supplementary_artifact_matches_recomputation(real_sup: dict[str, Any]) -> None:
    saved = read_json(RUN / SUPPLEMENTARY_DIR / SUPPLEMENTARY_FILE)
    inputs = saved.pop("inputs_sha256")
    assert saved == real_sup
    assert inputs["cells.jsonl"] == sha256_file(RUN / "cells.jsonl")
    assert inputs["metrics.json"] == sha256_file(RUN / "metrics.json")


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


@real
def test_readme_agrees_with_saved_artifacts(real_sup: dict[str, Any]) -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    m = read_json(RUN / "metrics.json")
    ag, o, v = m["human_agreement"], m["order_robustness"], m["verbosity"]

    def wilson_text(r: dict[str, Any]) -> str:
        lo, hi = r["wilson95"]
        return f"{_pct(lo)}–{_pct(hi)}%"

    for r in (
        ag["main_accuracy"],
        o["order_inconsistency"],
        v["judge_longer_among_decisive"],
        v["human_longer_among_resolved"],
        m["baseline"]["accuracy"],
        o["consistent_accuracy"]["accuracy"],
    ):
        assert f"{r['numerator']}/{r['denominator']}" in readme
        assert f"{_pct(r['rate'])}%" in readme
        assert wilson_text(r) in readme
    kappa = ag["cohen_kappa"]
    assert f"{kappa['estimate']:.3f}" in readme
    assert f"{kappa['ci95'][0]:.3f}–{kappa['ci95'][1]:.3f}" in readme
    paired = v["judge_minus_human_longer_rate_on_resolved_unequal"]
    lo, hi = (100 * x for x in paired["ci95"])
    assert f"+{_pct(paired['estimate'])} pp" in readme
    assert f"−{abs(lo):.1f} to +{hi:.1f} pp" in readme
    raw = real_sup["verbosity"]["raw_headline_difference"]
    assert f"+{_pct(raw['estimate'])} pp" in readme
    assert "204/299" in readme
    for c in real_sup["supplementary_clustered_intervals"].values():
        lo_c, hi_c = c["supplementary_pair_cluster_bootstrap"]["ci95"]
        if c["numerator"] in (267, 204):
            assert f"{_pct(lo_c)}–{_pct(hi_c)}%" in readme
    # The headline-rate contrast must never be presented as the paired statistic.
    assert not re.search(r"Paired judge − human difference: \+0\.7", readme)
