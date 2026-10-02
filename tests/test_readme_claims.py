"""Every number in the README project page must be recomputable from committed artifacts."""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from typing import Any

import pytest

from llm_judge_audit.config import DATA_DERIVED, PROJECT_ROOT, RUNS_DIR
from llm_judge_audit.io_utils import read_json, read_jsonl

RUN = RUNS_DIR / "bench-4e35d65ba747"
pytestmark = pytest.mark.skipif(not (RUN / "metrics.json").exists(), reason="no saved run")


@pytest.fixture(scope="module")
def readme() -> str:
    # Whitespace-normalised so that line wrapping in the Markdown source does not matter.
    return re.sub(r"\s+", " ", (PROJECT_ROOT / "README.md").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def m() -> dict[str, Any]:
    out: dict[str, Any] = read_json(RUN / "metrics.json")
    return out


@pytest.fixture(scope="module")
def cells() -> dict[tuple[str, str], dict[str, Any]]:
    return {(r["pair_id"], r["order"]): r for r in read_jsonl(RUN / "cells.jsonl")}


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def wilson(r: dict[str, Any]) -> str:
    return f"{pct(r['wilson95'][0])}–{pct(r['wilson95'][1])}%"


def test_status_line(readme: str) -> None:
    integ = read_json(RUN / "integrity.json")
    c = integ["counts"]
    assert integ["completion_state"] == "COMPLETE" and integ["integrity_ok"]
    assert c["valid"] == c["expected"] == 400
    assert "**COMPLETE.** 400/400 valid cells" in readme
    assert (
        f"{c['expected']} expected · {c['attempted']} attempted · {c['valid']} valid · "
        f"{c['invalid_output']} invalid output · {c['runtime_error']} runtime error · "
        f"{c['timeout']} timeout · {c['missing']} missing"
    ) in readme


def test_agreement_table(readme: str, m: dict[str, Any]) -> None:
    ag = m["human_agreement"]
    rows = [
        ag["main_accuracy"],
        ag["coverage"],
        ag["by_order"]["original"]["accuracy"],
        ag["by_order"]["swapped"]["accuracy"],
        ag["judge_tie_on_resolved"],
        m["baseline"]["accuracy"],
    ]
    for r in rows:
        line = f"| {r['numerator']}/{r['denominator']} | {pct(r['rate'])}% | Wilson {wilson(r)}"
        assert line in readme, line
    b = ag["main_accuracy"]["pair_cluster_bootstrap"]
    assert f"pair bootstrap {pct(b['ci95'][0])}–{pct(b['ci95'][1])}%" in readme
    d = ag["accuracy_original_minus_swapped"]
    assert (
        f"+{pct(d['estimate'])} pp, pair bootstrap −{abs(100 * d['ci95'][0]):.1f} to "
        f"+{pct(d['ci95'][1])} pp"
    ) in readme
    assert f"{ag['resolved_pairs']} pairs with a resolved human preference" in readme
    t = ag["human_tie_pairs"]
    assert (
        f"{t['pairs']} human-tie pairs (the judge said tie on {t['judge_verdicts']['tie']} "
        f"of their {t['valid_judge_cells']} cells)"
    ) in readme
    assert f"{ag['human_unresolved_pairs']} unresolved pairs" in readme
    assert f"{ag['excluded_annotations_in_sample']} excluded annotations" in readme


def test_order_claims(readme: str, m: dict[str, Any], cells: dict[Any, Any]) -> None:
    o = m["order_robustness"]
    inc = o["order_inconsistency"]
    assert (
        f"{inc['numerator']}/{inc['denominator']} pairs ({pct(inc['rate'])}%, Wilson 95% CI "
        f"{wilson(inc)})" in readme
    )
    first = o["position_choice"]["pooled"]["first_A"]
    assert (
        f"{first['numerator']}/{first['denominator']} calls ({pct(first['rate'])}%, "
        f"Wilson {wilson(first)})"
    ) in readme
    ca, cc = o["consistent_accuracy"]["accuracy"], o["consistent_accuracy"]["coverage"]
    assert (
        f"{ca['numerator']}/{ca['denominator']} ({pct(ca['rate'])}%, Wilson {wilson(ca)})" in readme
    )
    assert (
        f"{cc['numerator']}/{cc['denominator']} resolved pairs ({pct(cc['rate'])}%, "
        f"Wilson {wilson(cc)})" in readme
    )
    # Post-hoc slot breakdown, recomputed from the raw cell log.
    slots: Counter[str] = Counter()
    for pid in o["changed_pair_ids"]:
        a, b = cells[(pid, "original")], cells[(pid, "swapped")]
        assert a["mapped_verdict"] != b["mapped_verdict"]
        if a["verdict_displayed"] == b["verdict_displayed"] != "tie":
            slots[a["verdict_displayed"]] += 1
    same = slots["A"] + slots["B"]
    assert (same, slots["A"], slots["B"]) == (54, 25, 29)
    assert (
        f"In {same} of the {inc['numerator']} flipped pairs the judge picked the *same "
        f"displayed slot* both times: {slots['A']} always the first answer, {slots['B']} "
        "always the second."
    ) in readme


def test_error_review_claims(readme: str, cells: dict[Any, Any]) -> None:
    er = read_json(RUN / "error_review.json")
    n_sel = len(er["selected_pair_ids"])
    assert f"{n_sel} of {er['eligible_incorrect_pairs']} incorrect resolved pairs" in readme
    same_slot = sum(
        1
        for c in er["cases"]
        if len({j["displayed"] for j in c["judge"].values()}) == 1
        and len({j["mapped"] for j in c["judge"].values()}) == 2
    )
    assert f"{same_slot} cases chose the same slot in both orders" in readme
    assert (
        "in 5 maths or reasoning items" in readme and "on 5 math/reasoning items" in er["summary"]
    )
    # The README's worked example must match the saved cells.
    pid = "p_d43e1f57ba7a61ba"
    assert cells[(pid, "original")]["verdict_displayed"] == "B"
    assert cells[(pid, "swapped")]["verdict_displayed"] == "B"
    assert cells[(pid, "original")]["mapped_verdict"] != cells[(pid, "swapped")]["mapped_verdict"]


def test_verbosity_table(readme: str, m: dict[str, Any]) -> None:
    v = m["verbosity"]
    sup = read_json(RUN / "supplementary" / "supplementary_analysis_v1.json")["verbosity"]
    for r in (
        v["judge_longer_among_decisive"],
        v["human_longer_among_resolved"],
        sup["judge_longer_resolved_unequal"],
    ):
        assert (
            f"{r['numerator']}/{r['denominator']} | {pct(r['rate'])}% | Wilson {wilson(r)}"
            in readme
        )
    assert (
        f"({v['unequal_length_pairs']} pairs with unequal length; "
        f"{sup['pairs']['resolved_unequal']} have" in readme
    )


def test_operational_and_sample_claims(readme: str, m: dict[str, Any]) -> None:
    op = m["operational"]
    lat, pt, ct = op["latency_s"], op["prompt_tokens"], op["completion_tokens"]
    assert (
        f"median {lat['median']:.1f} s · p95 {lat['p95']:.1f} s · max {lat['max']:.1f} s" in readme
    )
    assert (
        f"{pt['total']:,} prompt (median {pt['median']:.0f}, max {pt['max']:,}) · "
        f"{ct['total']:,} completion (median {ct['median']:.0f})"
    ) in readme
    s = op["sessions"][0]
    start = datetime.fromisoformat(s["session_start_utc"])
    end = datetime.fromisoformat(s["session_end_utc"])
    assert f"{(end - start).total_seconds() / 60:.1f} min" in readme
    manifest = read_jsonl(DATA_DERIVED / "manifest.jsonl")
    single = sum(1 for p in manifest if p["human"]["n_votes_retained"] == 1)
    assert f"{single} of the 200 pairs rest on a single retained vote" in readme
    agg = Counter(p["human"]["aggregate"] for p in manifest)
    assert f"{agg['tie']} pairs are human ties and {agg['unresolved']} are unresolved" in readme
    assert Counter(p["category"] for p in manifest) == dict.fromkeys(
        {p["category"] for p in manifest}, 25
    )


def test_screenshots_exist_and_are_referenced(readme: str) -> None:
    refs = re.findall(r"\((docs/screenshots/[a-z_]+\.png)\)", readme)
    for view in ("overview", "agreement", "order", "verbosity", "pair_explorer", "integrity"):
        assert f"docs/screenshots/{view}.png" in refs
    for ref in refs:
        data = (PROJECT_ROOT / ref).read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 50_000
    # The Overview screenshot appears before the first section heading.
    assert readme.index("docs/screenshots/overview.png") < readme.index(" ## ")
