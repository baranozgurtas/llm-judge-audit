"""swap-consistency-v1: hand-computed SAMPLE DATA fixtures and checks on the saved real run."""

from __future__ import annotations

import copy
import re
from typing import Any

import numpy as np
import pytest

from llm_judge_audit.analysis import (
    SUPPLEMENTARY_DIR,
    SWAP_CONSISTENCY_CODE,
    SWAP_CONSISTENCY_FILE,
)
from llm_judge_audit.config import DATA_DERIVED, PROJECT_ROOT, RUNS_DIR, load_config
from llm_judge_audit.integrity import check_integrity
from llm_judge_audit.io_utils import read_json, read_jsonl, sha256_file, sha256_text
from llm_judge_audit.prompt import render_cell
from llm_judge_audit.sample_data import synthetic_manifest
from llm_judge_audit.swap_consistency import (
    ABSTAINED,
    BOOTSTRAP_SEED,
    COVERED,
    UNAVAILABLE,
    MappingError,
    aggregate_pair,
    aggregate_predictions,
    map_back,
    registered_cross_checks,
    swap_consistency_analysis,
)

MAPPING = {"original": {"A": "cand_1", "B": "cand_2"}, "swapped": {"A": "cand_2", "B": "cand_1"}}


def _cell(pair: dict[str, Any], order: str, displayed: str | None) -> dict[str, Any]:
    """SAMPLE DATA cell; displayed=None models a failed (invalid_output) cell."""
    mapped = None if displayed is None else map_back(displayed, order)
    return {
        "pair_id": pair["pair_id"],
        "order": order,
        "status": "valid" if displayed is not None else "invalid_output",
        "display_mapping": dict(MAPPING[order]),
        "verdict_displayed": displayed,
        "mapped_verdict": mapped,
        "rendered_prompt_sha256": sha256_text(render_cell(pair, order).prompt),
    }


# Hand-built SAMPLE DATA: (human aggregate, original displayed, swapped displayed).
# Expected mapped-back verdicts and aggregation are written out beside each row.
CASES = [
    ("cand_1", "A", "B"),  # p0: c1, c1 -> covered c1, correct
    ("cand_2", "B", "A"),  # p1: c2, c2 -> covered c2, correct
    ("cand_1", "A", "A"),  # p2: c1, c2 -> abstained (same displayed slot); original correct
    ("cand_2", "tie", "tie"),  # p3: tie, tie -> covered tie, incorrect; original incorrect
    ("cand_1", "B", "A"),  # p4: c2, c2 -> covered c2, incorrect; original incorrect
    ("cand_2", "B", "tie"),  # p5: c2, tie -> abstained; original correct
    ("tie", "A", "B"),  # p6: human tie -> covered c1, not scored
    ("unresolved", "A", "A"),  # p7: unresolved -> abstained, not scored
    ("cand_1", "A", None),  # p8: swapped cell failed -> unavailable; original correct
]


def _fixture() -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    manifest = synthetic_manifest(len(CASES))
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for m, (human, o, s) in zip(manifest, CASES, strict=True):
        m["human"]["aggregate"] = human
        cells[(m["pair_id"], "original")] = _cell(m, "original", o)
        cells[(m["pair_id"], "swapped")] = _cell(m, "swapped", s)
    return manifest, cells


def _ns(r: dict[str, Any]) -> tuple[int, int]:
    return r["numerator"], r["denominator"]


def test_map_back_follows_displayed_position() -> None:
    assert [map_back(d, "original") for d in ("A", "B", "tie")] == ["cand_1", "cand_2", "tie"]
    assert [map_back(d, "swapped") for d in ("A", "B", "tie")] == ["cand_2", "cand_1", "tie"]


def test_aggregate_pair_rule() -> None:
    assert aggregate_pair("cand_1", "cand_1") == (COVERED, "cand_1")
    assert aggregate_pair("cand_2", "cand_2") == (COVERED, "cand_2")
    assert aggregate_pair("tie", "tie") == (COVERED, "tie")
    assert aggregate_pair("cand_1", "cand_2") == (ABSTAINED, None)
    assert aggregate_pair("cand_2", "tie") == (ABSTAINED, None)
    assert aggregate_pair("cand_1", None) == (UNAVAILABLE, None)
    assert aggregate_pair(None, None) == (UNAVAILABLE, None)
    with pytest.raises(MappingError):
        aggregate_pair("A", "A")  # displayed labels must never reach the rule unmapped


def test_per_pair_predictions_hand_computed() -> None:
    manifest, cells = _fixture()
    preds = aggregate_predictions(cells, [m["pair_id"] for m in manifest])
    got = [(p["original"], p["swapped"], p["status"], p["prediction"]) for p in preds.values()]
    assert got == [
        ("cand_1", "cand_1", COVERED, "cand_1"),
        ("cand_2", "cand_2", COVERED, "cand_2"),
        ("cand_1", "cand_2", ABSTAINED, None),
        ("tie", "tie", COVERED, "tie"),
        ("cand_2", "cand_2", COVERED, "cand_2"),
        ("cand_2", "tie", ABSTAINED, None),
        ("cand_1", "cand_1", COVERED, "cand_1"),
        ("cand_1", "cand_2", ABSTAINED, None),
        ("cand_1", None, UNAVAILABLE, None),
    ]


def test_counts_and_denominators_hand_computed() -> None:
    manifest, cells = _fixture()
    out = swap_consistency_analysis(manifest, cells, resamples=200, seed=5)
    r, c = out["resolved"], out["comparison_original_order"]
    # Resolved pairs: p0 p1 p2 p3 p4 p5 p8 = 7; covered p0 p1 p3 p4; abstained p2 p5; p8 n/a.
    assert {k: r["counts"][k] for k in ("pairs", COVERED, ABSTAINED, UNAVAILABLE)} == {
        "pairs": 7,
        COVERED: 4,
        ABSTAINED: 2,
        UNAVAILABLE: 1,
    }
    assert r["counts"]["covered_predictions"] == {"cand_1": 1, "cand_2": 2, "tie": 1}
    assert _ns(r["accuracy_among_covered"]) == (2, 4)  # p0, p1 correct
    assert _ns(r["coverage"]) == (4, 7)
    assert _ns(r["accuracy_all_resolved"]) == (2, 7)  # abstained/unavailable = unanswered
    assert r["aggregated_tie_predictions_scored_incorrect"] == 1  # p3
    # Original order on the same 7 pairs: p0 p1 p2 p5 p8 correct.
    assert _ns(c["original_order_accuracy"]) == (5, 7)
    assert _ns(c["original_order_coverage"]) == (7, 7)
    assert _ns(c["original_order_on_covered_pairs"]) == (2, 4)
    assert _ns(c["original_order_on_abstained_pairs"]) == (2, 2)
    assert c["accuracy_all_resolved_minus_original"]["estimate"] == pytest.approx(2 / 7 - 5 / 7)
    assert c["accuracy_among_covered_minus_original"]["estimate"] == pytest.approx(2 / 4 - 5 / 7)
    # Human ties / unresolved are reported apart and never scored.
    t, u = out["human_tie_pairs"], out["human_unresolved_pairs"]
    assert (t["pairs"], t[COVERED], t["covered_predictions"], t["scored"]) == (
        1,
        1,
        {"cand_1": 1},
        False,
    )
    assert (u["pairs"], u[ABSTAINED], u["scored"]) == (1, 1, False)
    a = out["all_pairs"]
    assert (a["pairs"], a[COVERED], a[ABSTAINED], a[UNAVAILABLE]) == (9, 5, 3, 1)
    assert out["mapping_verification"]["valid_cells_checked"] == 17
    assert out["mapping_verification"]["rendered_prompt_hashes_matched"] == 17


def test_predictions_ignore_human_labels() -> None:
    manifest, cells = _fixture()
    base = swap_consistency_analysis(manifest, cells, resamples=50, seed=5)
    flipped = copy.deepcopy(manifest)
    swap = {"cand_1": "cand_2", "cand_2": "cand_1", "tie": "unresolved", "unresolved": "tie"}
    for m in flipped:
        m["human"]["aggregate"] = swap[m["human"]["aggregate"]]
        m["human"]["votes"] = []
    other = swap_consistency_analysis(flipped, cells, resamples=50, seed=5)
    keep = ("pair_id", "original", "swapped", "status", "prediction")
    assert [{k: p[k] for k in keep} for p in base["pairs"]] == [
        {k: p[k] for k in keep} for p in other["pairs"]
    ]


def test_mapping_verification_fails_loudly() -> None:
    manifest, cells = _fixture()
    key = (manifest[0]["pair_id"], "swapped")
    for field, value in (
        ("mapped_verdict", "cand_2"),
        ("display_mapping", dict(MAPPING["original"])),
        ("rendered_prompt_sha256", "0" * 64),
    ):
        bad = copy.deepcopy(cells)
        bad[key][field] = value
        with pytest.raises(MappingError):
            swap_consistency_analysis(manifest, bad, resamples=10)


def test_bootstrap_reproducible_and_independently_reimplemented() -> None:
    manifest, cells = _fixture()
    a = swap_consistency_analysis(manifest, cells, resamples=300, seed=11)
    b = swap_consistency_analysis(manifest, cells, resamples=300, seed=11)
    assert a == b
    assert swap_consistency_analysis(manifest, cells, resamples=300)["bootstrap"]["seed_base"] == (
        BOOTSTRAP_SEED
    )
    # Independent re-implementation of the coverage interval (seed 11 + 1, pair resampling).
    resolved = sorted(
        m["pair_id"] for m in manifest if m["human"]["aggregate"] in ("cand_1", "cand_2")
    )
    covered = {p["pair_id"] for p in a["pairs"] if p["status"] == COVERED}
    rng = np.random.default_rng(12)
    vals = []
    for _ in range(300):
        draw = [resolved[i] for i in rng.integers(0, len(resolved), size=len(resolved))]
        vals.append(sum(pid in covered for pid in draw) / len(draw))
    lo, hi = np.percentile(np.array(vals), [2.5, 97.5])
    assert a["resolved"]["coverage"]["pair_bootstrap"]["ci95"] == [float(lo), float(hi)]
    assert a["resolved"]["coverage"]["pair_bootstrap"]["seed"] == 12


# ---- Real saved run (immutable records) ------------------------------------------------
RUN = RUNS_DIR / "bench-4e35d65ba747"
ARTIFACT = RUN / SUPPLEMENTARY_DIR / SWAP_CONSISTENCY_FILE
real = pytest.mark.skipif(not ARTIFACT.exists(), reason="no saved swap-consistency artifact")


@pytest.fixture(scope="module")
def real_out() -> dict[str, Any]:
    manifest = read_jsonl(DATA_DERIVED / "manifest.jsonl")
    meta = read_json(RUN / "run_meta.json")
    report = check_integrity(
        RUN / "cells.jsonl", [m["pair_id"] for m in manifest], meta["identity"]
    )
    assert report["completion_state"] == "COMPLETE"
    out: dict[str, Any] = swap_consistency_analysis(
        manifest, report["accepted"], int(load_config()["analysis"]["bootstrap_resamples"])
    )
    return out


@real
def test_real_counts(real_out: dict[str, Any]) -> None:
    r, c = real_out["resolved"], real_out["comparison_original_order"]
    assert real_out["mapping_verification"]["rendered_prompt_hashes_matched"] == 400
    assert (r["counts"]["pairs"], r["counts"][COVERED], r["counts"][ABSTAINED]) == (154, 116, 38)
    assert _ns(r["accuracy_among_covered"]) == (100, 116)
    assert _ns(r["coverage"]) == (116, 154)
    assert _ns(r["accuracy_all_resolved"]) == (100, 154)
    assert _ns(c["original_order_accuracy"]) == (120, 154)
    assert real_out["human_tie_pairs"]["pairs"] == 32
    assert real_out["human_unresolved_pairs"]["pairs"] == 14
    assert all(registered_cross_checks(real_out, read_json(RUN / "metrics.json")).values())


@real
def test_saved_artifact_matches_recomputation_and_hashes(real_out: dict[str, Any]) -> None:
    saved = read_json(ARTIFACT)
    inputs, code = saved.pop("inputs_sha256"), saved.pop("code_sha256")
    checks = saved.pop("registered_cross_checks")
    assert (saved.pop("run_id"), saved.pop("completion_state")) == (RUN.name, "COMPLETE")
    assert saved == real_out
    assert all(checks.values())
    for rel_path, digest in inputs.items():
        assert sha256_file(PROJECT_ROOT / rel_path) == digest, rel_path
    assert set(code) == set(SWAP_CONSISTENCY_CODE)
    for rel_path, digest in code.items():
        assert sha256_file(PROJECT_ROOT / rel_path) == digest, rel_path


@real
def test_readme_reports_swap_consistency_from_artifact() -> None:
    readme = re.sub(r"\s+", " ", (PROJECT_ROOT / "README.md").read_text(encoding="utf-8"))
    out = read_json(ARTIFACT)
    r, c = out["resolved"], out["comparison_original_order"]

    def pct(x: float) -> str:
        return f"{100 * x:.1f}"

    def row(x: dict[str, Any]) -> str:
        b = x["pair_bootstrap"]["ci95"]
        return (
            f"{x['numerator']}/{x['denominator']} | {pct(x['rate'])}% | pair bootstrap "
            f"{pct(b[0])}–{pct(b[1])}%"
        )

    assert "post-hoc" in readme.lower() and out["analysis_version"] in readme
    for x in (
        r["accuracy_among_covered"],
        r["coverage"],
        r["accuracy_all_resolved"],
        c["original_order_accuracy"],
    ):
        assert row(x) in readme, row(x)

    def pp(b: dict[str, Any]) -> str:
        def f(v: float) -> str:
            return f"{'+' if v >= 0 else '−'}{abs(100 * v):.1f}"

        return f"{f(b['estimate'])} pp (pair bootstrap {f(b['ci95'][0])} to {f(b['ci95'][1])} pp)"

    assert pp(c["accuracy_all_resolved_minus_original"]) in readme
    assert pp(c["accuracy_among_covered_minus_original"]) in readme
    assert f"seed {out['bootstrap']['seed_base']}" in readme
    ab, t, u = (
        c["original_order_on_abstained_pairs"],
        out["human_tie_pairs"],
        out["human_unresolved_pairs"],
    )
    assert f"correct in {ab['numerator']}/{ab['denominator']} ({pct(ab['rate'])}%)" in readme
    n = r["counts"]
    assert (
        f"{n['covered']} covered, {n['abstained']} abstained, {n['unavailable']} unavailable"
        in readme
    )
    assert (
        f"{t['pairs']} human-tie pairs ({t['covered']} covered, {t['abstained']} abstained) and "
        f"{u['pairs']} unresolved pairs ({u['covered']} covered, {u['abstained']} abstained)"
    ) in readme
    assert "No improvement is claimed" in readme
