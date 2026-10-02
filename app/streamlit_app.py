"""LLM Judge Audit dashboard. Reads saved artifacts only; never starts inference.

Launch: uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from llm_judge_audit.ui_data import (
    SAMPLE_SOURCE,
    Bundle,
    fmt_boot,
    fmt_diff_pp,
    fmt_rate,
    list_sources,
    load_bundle,
)

# Reference categorical slots (validated palette): blue, orange; neutral gray for ties.
COLOR_FIRST = "#2a78d6"
COLOR_SECOND = "#eb6834"
COLOR_TIE = "#8a8984"

STATE_TEXT = {
    "COMPLETE": "COMPLETE: all 400 cells valid and integrity checks passed.",
    "PARTIAL": "PARTIAL: some cells are missing or failed; results are descriptive only.",
    "NOT_STARTED": "NOT STARTED: no benchmark cells recorded.",
    "INTEGRITY_FAILED": "INTEGRITY FAILED: the cell log has corrupt, duplicate or "
    "mismatched records.",
}

st.set_page_config(page_title="LLM Judge Audit", page_icon="⚖️", layout="wide")


@st.cache_data(show_spinner=False)
def _bundle(source: str) -> Bundle:
    return load_bundle(source)


def rate_row(name: str, r: dict[str, Any] | None, note: str = "") -> dict[str, Any]:
    r = r or {}
    ci = r.get("wilson95")
    return {
        "Metric": name,
        "Numerator": r.get("numerator"),
        "Denominator": r.get("denominator"),
        "Rate": None if r.get("rate") is None else round(100 * r["rate"], 1),
        "Wilson 95% CI (%)": f"{100 * ci[0]:.1f}–{100 * ci[1]:.1f}" if ci else "—",
        "Note": note,
    }


def rate_table(rows: list[dict[str, Any]]) -> None:
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={"Rate": st.column_config.NumberColumn("Rate (%)", format="%.1f")},
    )


def ci_chart(rows: list[tuple[str, dict[str, Any] | None]], title: str) -> None:
    """Point + Wilson interval per rate, on a shared 0–100% axis."""
    data = []
    for name, r in rows:
        if r and r.get("rate") is not None and r.get("wilson95"):
            data.append(
                {
                    "metric": name,
                    "rate": 100 * r["rate"],
                    "lo": 100 * r["wilson95"][0],
                    "hi": 100 * r["wilson95"][1],
                    "counts": f"{r['numerator']}/{r['denominator']}",
                }
            )
    if not data:
        return
    df = pd.DataFrame(data)
    base = alt.Chart(df).encode(
        y=alt.Y("metric:N", title=None, sort=None, axis=alt.Axis(labelLimit=360)),
        tooltip=[
            alt.Tooltip("metric:N", title="Metric"),
            alt.Tooltip("counts:N", title="n/N"),
            alt.Tooltip("rate:Q", title="Rate %", format=".1f"),
            alt.Tooltip("lo:Q", title="CI low %", format=".1f"),
            alt.Tooltip("hi:Q", title="CI high %", format=".1f"),
        ],
    )
    rule = base.mark_rule(strokeWidth=2, color=COLOR_FIRST).encode(
        x=alt.X("lo:Q", title="Rate (%) with Wilson 95% CI", scale=alt.Scale(domain=[0, 100])),
        x2="hi:Q",
    )
    point = base.mark_point(filled=True, size=90, color=COLOR_FIRST).encode(x="rate:Q")
    st.altair_chart(
        (rule + point).properties(title=title, height=alt.Step(40)), use_container_width=True
    )


def need_metrics(b: Bundle) -> dict[str, Any] | None:
    if b.metrics is None:
        st.info(
            "No metrics saved for this source yet. Metrics appear after an "
            "integrity-checked run is analysed (`uv run lja analyze`)."
        )
    return b.metrics


# --------------------------------------------------------------------------- sidebar
st.sidebar.title("LLM Judge Audit")
sources = list_sources()
source = st.sidebar.selectbox(
    "Result source",
    sources,
    index=0,
    help="Benchmark runs found under results/runs, or synthetic "
    "SAMPLE DATA for demonstrating the UI.",
)
VIEWS = [
    "Overview",
    "Agreement & Coverage",
    "Order & Verbosity",
    "Pair Explorer",
    "Run Integrity & Provenance",
]
VIEW_SLUGS = ["overview", "agreement", "order", "pairs", "integrity"]
# Optional deep links, e.g. ?view=pairs&pair=p_d43e1f57ba7a61ba (read-only; no inference).
_slug = st.query_params.get("view", "overview")
view = st.sidebar.radio(
    "View",
    VIEWS,
    index=VIEW_SLUGS.index(_slug) if _slug in VIEW_SLUGS else 0,
)
st.sidebar.caption("Read-only dashboard. Loading it never starts model inference.")

bundle = _bundle(source)
if bundle.is_sample:
    st.warning(
        "**SAMPLE DATA** — synthetic texts, fake labels and a fake judge for "
        "demonstrating the dashboard. These are not study results.",
        icon="🧪",
    )

state = (bundle.integrity or {}).get("completion_state", "NOT_STARTED")
counts = (bundle.integrity or {}).get("counts", {})


# --------------------------------------------------------------------------- views
def flip_breakdown() -> dict[str, int]:
    """Post-hoc description of order-changed pairs by displayed slot (from saved cells)."""
    out = {"changed": 0, "same_slot": 0, "first": 0, "second": 0}
    for m in bundle.manifest:
        o = bundle.cells.get((m["pair_id"], "original"))
        s = bundle.cells.get((m["pair_id"], "swapped"))
        if not (o and s and o["status"] == s["status"] == "valid"):
            continue
        if o["mapped_verdict"] == s["mapped_verdict"]:
            continue
        out["changed"] += 1
        if o["verdict_displayed"] == s["verdict_displayed"] != "tie":
            out["same_slot"] += 1
            out["first" if o["verdict_displayed"] == "A" else "second"] += 1
    return out


def view_overview() -> None:
    st.title("LLM Judge Audit")
    st.markdown(
        "**Question.** On 200 MT-bench pairwise comparisons with expert human preferences, how "
        "often does one local LLM judge (Ollama `gemma3:4b`) agree with the human preference, "
        "how often does swapping answer positions change its decision, and does it favour the "
        "first-shown or the longer answer?"
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sampled pairs (N)", len(bundle.manifest))
    c2.metric("Expected judge calls", counts.get("expected", 2 * len(bundle.manifest)))
    c3.metric("Valid cells", f"{counts.get('valid', 0)}/{counts.get('expected', 400)}")
    c4.metric("Completion state", state)
    (st.success if state == "COMPLETE" else st.warning)(STATE_TEXT.get(state, state))

    st.subheader("Study design")
    st.markdown(
        "- **Data:** `lmsys/mt_bench_human_judgments`, human split, CC-BY-4.0, revision "
        "`f7d2896`; turn-1 pairs only; 25 pairs from each of 8 MT-bench categories "
        "(seeded, stratified).\n"
        "- **Judge:** one local model, `gemma3:4b` via Ollama, temperature 0, JSON-schema "
        "constrained output, no retries.\n"
        "- **Design:** each pair is judged twice, in original and swapped answer order "
        "(400 cells); verdicts are mapped back to the original answers.\n"
        "- **Human labels:** strict plurality of expert votes; ties and unresolved pairs are "
        "reported separately, never scored."
    )
    m = bundle.metrics
    if m:
        st.subheader("Headline results")
        st.caption(m.get("scope_note", ""))
        ag, order = m["human_agreement"], m["order_robustness"]
        st.markdown(
            f"- **Agreement with humans (all valid decisions):** {fmt_rate(ag['main_accuracy'])}\n"
            f"- **Coverage:** {fmt_rate(ag['coverage'])}\n"
            f"- **Cohen's κ:** {fmt_boot(ag['cohen_kappa'])}\n"
            f"- **Order inconsistency:** {fmt_rate(order['order_inconsistency'])}\n"
            f"- **Majority baseline (descriptive):** {fmt_rate(m['baseline']['accuracy'])}"
        )
        if bundle.cells:
            flips = flip_breakdown()
            inc = order["order_inconsistency"]
            st.info(
                f"**Key finding: order sensitivity.** Swapping the answer positions changed the "
                f"mapped-back verdict for {inc['numerator']} of {inc['denominator']} pairs "
                f"({inc['rate']:.1%}, Wilson 95% CI {inc['wilson95'][0]:.1%}–"
                f"{inc['wilson95'][1]:.1%}). In {flips['same_slot']} of those {flips['changed']} "
                f"pairs the judge picked the same displayed slot both times ({flips['first']} "
                f"always first, {flips['second']} always second). Pooled, the first slot was "
                f"chosen in {fmt_rate(order['position_choice']['pooled']['first_A'])} of calls, "
                "so there is no net first-position preference, but the decision depends on "
                "position for many individual pairs. The slot breakdown is a post-hoc "
                "description computed from the saved cells, not a registered metric."
            )
    st.subheader("Limitations")
    st.markdown(
        "- Exploratory study of **one** small local judge on **200** pairs from one dataset; "
        "results do not generalise to other judges, prompts or data.\n"
        "- Human labels are noisy: 109 of 200 pairs rest on a single vote; ties and "
        "disagreement are common.\n"
        "- MT-bench and its answers are public since 2023 and may be in Gemma's training data "
        "(possible contamination).\n"
        "- Temperature 0 does not guarantee determinism.\n"
        "- Length findings are associations, not causal evidence."
    )


def view_agreement() -> None:
    st.title("Agreement & Coverage")
    m = need_metrics(bundle)
    if not m:
        return
    st.caption(m.get("scope_note", ""))
    ag = m["human_agreement"]
    st.markdown(
        f"Pairs with a resolved human preference: **{ag['resolved_pairs']}** of "
        f"{len(bundle.manifest)}. Human labels: {label_counts(ag['human_label_counts'])}."
    )
    rows = [
        rate_row(
            "Main accuracy (cells, both orders)",
            ag["main_accuracy"],
            "correct / valid cells on resolved pairs",
        ),
        rate_row("Coverage (cells)", ag["coverage"], "valid cells / 2 × resolved pairs"),
        rate_row("Accuracy — original order", ag["by_order"]["original"]["accuracy"]),
        rate_row("Coverage — original order", ag["by_order"]["original"]["coverage"]),
        rate_row("Accuracy — swapped order", ag["by_order"]["swapped"]["accuracy"]),
        rate_row("Coverage — swapped order", ag["by_order"]["swapped"]["coverage"]),
        rate_row(
            "Judge 'tie' on resolved pairs",
            ag["judge_tie_on_resolved"],
            "valid ties count as incorrect",
        ),
        rate_row(
            "Majority baseline (leave-one-pair-out)",
            m["baseline"]["accuracy"],
            "descriptive context, not a judge",
        ),
    ]
    rate_table(rows)
    ci_chart(
        [
            (r["Metric"], src)
            for r, src in zip(
                rows,
                [
                    ag["main_accuracy"],
                    ag["coverage"],
                    ag["by_order"]["original"]["accuracy"],
                    ag["by_order"]["original"]["coverage"],
                    ag["by_order"]["swapped"]["accuracy"],
                    ag["by_order"]["swapped"]["coverage"],
                    ag["judge_tie_on_resolved"],
                    m["baseline"]["accuracy"],
                ],
                strict=True,
            )
        ],
        "Rates with Wilson 95% intervals",
    )
    st.markdown(
        f"- **Main accuracy, pair-cluster bootstrap:** "
        f"{fmt_boot(ag['main_accuracy']['pair_cluster_bootstrap'], pct=True)}\n"
        f"- **Cohen's κ (human vs judge, valid resolved cells):** {fmt_boot(ag['cohen_kappa'])}\n"
        f"- **Accuracy original − swapped (paired):** "
        f"{fmt_diff_pp(ag['accuracy_original_minus_swapped'])}"
    )
    st.subheader("Reported separately (not scored)")
    t = ag["human_tie_pairs"]
    st.markdown(
        f"- **Human-tie pairs:** {t['pairs']}; judge verdicts on their {t['valid_judge_cells']} "
        f"valid cells: {label_counts(t['judge_verdicts'])}\n"
        f"- **Unresolved human pairs:** {ag['human_unresolved_pairs']}\n"
        f"- **Excluded annotations in sample** (duplicate/conflicting votes): "
        f"{ag['excluded_annotations_in_sample']}"
    )
    st.caption("Baseline rule: " + m["baseline"]["rule"])


def view_order() -> None:
    st.title("Order & Verbosity")
    m = need_metrics(bundle)
    if not m:
        return
    st.caption(m.get("scope_note", ""))
    o, v = m["order_robustness"], m["verbosity"]
    st.subheader("Order robustness")
    st.markdown(
        f"**Changed mapped-back verdict after swapping:** "
        f"{fmt_rate(o['order_inconsistency'])}  \n"
        f"Pairs with two valid cells: {o['pairs_two_valid']} of {o['pairs_total']}."
    )
    dec = pd.DataFrame(o["mapped_decisions_by_order"]).fillna(0).astype(int)
    dec = dec.rename(
        index=LABEL_TEXT, columns={"original": "Original order", "swapped": "Swapped order"}
    )
    dec.index.name = "Mapped-back verdict"
    st.markdown("**Mapped-back decisions by order** (counts of valid cells)")
    st.dataframe(dec, width="stretch")

    pc = o["position_choice"]
    pos_rows, chart_rows = [], []
    for order in ("original", "swapped", "pooled"):
        for key, label in (("first_A", "First (A)"), ("second_B", "Second (B)"), ("tie", "Tie")):
            pos_rows.append(rate_row(f"{order}: {label}", pc[order][key]))
            r = pc[order][key]
            if r["denominator"]:
                chart_rows.append(
                    {
                        "order": order,
                        "choice": label,
                        "share": 100 * r["rate"],
                        "counts": f"{r['numerator']}/{r['denominator']}",
                    }
                )
    pos_rows.append(rate_row("pooled: first among decisive", pc["pooled"]["first_among_decisive"]))
    st.markdown("**Position choice by displayed position**")
    if chart_rows:
        chart = (
            alt.Chart(pd.DataFrame(chart_rows))
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                y=alt.Y("order:N", title=None, sort=["original", "swapped", "pooled"]),
                x=alt.X(
                    "share:Q",
                    title="Share of valid cells (%)",
                    stack="zero",
                    scale=alt.Scale(domain=[0, 100]),
                ),
                color=alt.Color(
                    "choice:N",
                    title="Judge chose",
                    scale=alt.Scale(
                        domain=["First (A)", "Second (B)", "Tie"],
                        range=[COLOR_FIRST, COLOR_SECOND, COLOR_TIE],
                    ),
                ),
                order=alt.Order("choice:N"),
                tooltip=["order", "choice", "counts", alt.Tooltip("share:Q", format=".1f")],
            )
            .properties(height=150)
        )
        st.altair_chart(chart, use_container_width=True)
    rate_table(pos_rows)

    st.markdown("**Accuracy on order-consistent pairs vs all valid decisions**")
    rate_table(
        [
            rate_row("All valid decisions: accuracy", o["all_valid_accuracy"]["accuracy"]),
            rate_row("All valid decisions: coverage", o["all_valid_accuracy"]["coverage"]),
            rate_row("Order-consistent pairs: accuracy", o["consistent_accuracy"]["accuracy"]),
            rate_row(
                "Order-consistent pairs: coverage",
                o["consistent_accuracy"]["coverage"],
                "abstains on inconsistent pairs",
            ),
        ]
    )
    st.caption(
        "Higher accuracy on consistent pairs comes with lower coverage; abstaining alone "
        "is not an accuracy improvement."
    )

    st.subheader("Verbosity association")
    st.markdown(
        f"Unequal-length pairs: **{v['unequal_length_pairs']}** "
        f"(equal: {v['equal_length_pairs']}); length = {v['length_rule']}."
    )
    vrows = [
        rate_row("Judge chose longer (decisive valid cells)", v["judge_longer_among_decisive"]),
        rate_row(
            "Judge chose longer (all valid cells, ties count as no)",
            v["judge_longer_among_all_valid"],
        ),
        rate_row("Humans chose longer (resolved pairs)", v["human_longer_among_resolved"]),
    ]
    rate_table(vrows)
    ci_chart(
        [
            (r["Metric"], s)
            for r, s in zip(
                vrows,
                [
                    v["judge_longer_among_decisive"],
                    v["judge_longer_among_all_valid"],
                    v["human_longer_among_resolved"],
                ],
                strict=True,
            )
        ],
        "Longer-answer selection",
    )
    paired = v["judge_minus_human_longer_rate_on_resolved_unequal"]
    sup = bundle.supplementary
    if sup is None:
        st.markdown(
            "**Paired difference (registered):** judge longer-rate on the resolved unequal pairs "
            "minus the human rate on the same pairs (not the difference of the two headline "
            f"rates above, which use different pair sets): {fmt_diff_pp(paired)}"
        )
    else:
        sv = sup["verbosity"]
        jr, hr = sv["judge_longer_resolved_unequal"], sv["human_longer_resolved_unequal"]
        ja, raw = sv["judge_longer_all_unequal"], sv["raw_headline_difference"]
        st.markdown(
            f"**Paired difference (registered), same {hr['denominator']} resolved unequal "
            f"pairs:** judge {jr['numerator']}/{jr['denominator']} ({jr['rate']:.2%}) − human "
            f"{hr['numerator']}/{hr['denominator']} ({hr['rate']:.2%}) = "
            f"{fmt_diff_pp(paired)}  \n"
            f"**Raw headline difference (supplementary, unpaired):** judge "
            f"{ja['numerator']}/{ja['denominator']} on all {sv['pairs']['unequal_length']} "
            f"unequal pairs ({ja['rate']:.2%}) − human ({hr['rate']:.2%}) = "
            f"{fmt_diff_pp(raw['supplementary_pair_cluster_bootstrap'])}. "
            "The two rates use different pair sets, so this is not a paired comparison."
        )
        st.markdown(
            "**Supplementary pair-cluster bootstrap intervals for cell-level rates** "
            f"({sup['supplementary_version']}; registered Wilson intervals unchanged)"
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Metric": SUPP_LABELS.get(name, name),
                        "Count": f"{c['numerator']}/{c['denominator']}",
                        "Rate (%)": f"{100 * c['rate']:.1f}",
                        "Registered Wilson 95% (%)": "–".join(
                            f"{100 * x:.1f}" for x in c["registered_wilson95"]
                        ),
                        "Pair-cluster bootstrap 95% (%)": "–".join(
                            f"{100 * x:.1f}"
                            for x in c["supplementary_pair_cluster_bootstrap"]["ci95"]
                        ),
                        "Pairs": c["n_pairs"],
                    }
                    for name, c in sup["supplementary_clustered_intervals"].items()
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    st.caption("Association only; this design cannot show that length causes the judge's choice.")


SUPP_LABELS = {
    "human_agreement.judge_tie_on_resolved": "Judge 'tie' on resolved pairs",
    "order_robustness.position_choice.pooled.first_A": "Chose first-shown answer (pooled)",
    "order_robustness.position_choice.pooled.first_among_decisive": "Chose first-shown answer, "
    "decisive only (pooled)",
    "verbosity.judge_longer_among_decisive": "Judge chose longer, all 196 unequal pairs",
    "verbosity.judge_longer_resolved_unequal (not registered)": "Judge chose longer, 151 resolved "
    "unequal pairs (not registered)",
}
LABEL_TEXT = {"cand_1": "Answer 1", "cand_2": "Answer 2", "tie": "Tie", "unresolved": "Unresolved"}


def label_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{LABEL_TEXT.get(k, k)}: {v}" for k, v in sorted(counts.items()))


HUMAN_TEXT = {
    "cand_1": "Answer 1 preferred",
    "cand_2": "Answer 2 preferred",
    "tie": "Tie",
    "unresolved": "Unresolved (no strict plurality)",
}
MAPPED_TEXT = {"cand_1": "Answer 1", "cand_2": "Answer 2", "tie": "Tie", None: "—"}


def changed(pid: str) -> bool:
    o, s = bundle.cells.get((pid, "original")), bundle.cells.get((pid, "swapped"))
    return bool(o and s and o["mapped_verdict"] != s["mapped_verdict"])


def view_pairs() -> None:
    st.title("Pair Explorer")
    if not bundle.manifest:
        st.info("No manifest found.")
        return
    cats = sorted({m["category"] for m in bundle.manifest})
    c1, c2, c3 = st.columns(3)
    cat = c1.selectbox("Category", ["all", *cats])
    behaviour = c3.selectbox(
        "Order behaviour",
        ["all", "changed after swap", "same after swap"],
        help="Whether the mapped-back verdict changed when answer positions were swapped.",
    )
    lab = c2.selectbox(
        "Human label",
        ["all", "cand_1", "cand_2", "tie", "unresolved"],
        format_func=lambda x: HUMAN_TEXT.get(x, x),
    )
    pairs = [
        m
        for m in bundle.manifest
        if (cat == "all" or m["category"] == cat)
        and (lab == "all" or m["human"]["aggregate"] == lab)
        and (behaviour == "all" or (behaviour == "changed after swap") == changed(m["pair_id"]))
    ]
    if not pairs:
        st.info("No pairs match the filters.")
        return
    linked = st.query_params.get("pair")
    ids = [m["pair_id"] for m in pairs]
    idx = st.selectbox(
        "Pair",
        range(len(pairs)),
        index=ids.index(linked) if linked in ids else 0,
        format_func=lambda i: f"#{pairs[i]['manifest_index']:03d} · "
        f"{pairs[i]['category']} · {pairs[i]['pair_id']}",
    )
    p = pairs[idx]
    h = p["human"]
    st.markdown(
        f"**Human aggregate:** {HUMAN_TEXT[h['aggregate']]} — retained votes: "
        f"Answer 1 = {h['counts']['cand_1']}, Answer 2 = {h['counts']['cand_2']}, "
        f"tie = {h['counts']['tie']} (annotator identities hidden)"
    )
    st.markdown("**Question**")
    st.text(p["question"])
    a1, a2 = st.columns(2)
    for col, key, name in ((a1, "cand_1", "Answer 1"), (a2, "cand_2", "Answer 2")):
        c = p["candidates"][key]
        col.markdown(f"**{name}** · {c['whitespace_tokens']} whitespace tokens")
        col.container(height=380, border=True).text(c["text"])
    st.subheader("Judge outputs")
    cols = st.columns(2)
    for col, order, disp in (
        (cols[0], "original", "A = Answer 1, B = Answer 2"),
        (cols[1], "swapped", "A = Answer 2, B = Answer 1"),
    ):
        rec = bundle.cells.get((p["pair_id"], order)) or {}
        box = col.container(border=True)
        box.markdown(f"**{order.capitalize()} order** · {disp}")
        box.markdown(
            f"Status: `{rec.get('status', 'missing')}` · displayed verdict: "
            f"**{rec.get('verdict_displayed') or '—'}** → "
            f"**{MAPPED_TEXT.get(rec.get('mapped_verdict'))}** · "
            f"{rec.get('latency_s', 0):.1f} s"
        )
        box.markdown(f"> {(rec.get('parse') or {}).get('rationale') or '—'}")
    verdict_changed = changed(p["pair_id"])
    (st.warning if verdict_changed else st.info)(
        "The mapped-back verdict **changed** when the answers were swapped."
        if verdict_changed
        else "The mapped-back verdict was the same in both orders."
    )
    with st.expander("Raw visible responses"):
        for order in ("original", "swapped"):
            raw_rec = bundle.cells.get((p["pair_id"], order))
            st.markdown(f"**{order}**")
            st.code((raw_rec or {}).get("raw_response") or "(none)", language="json")


def view_integrity() -> None:
    st.title("Run Integrity & Provenance")
    (st.success if state == "COMPLETE" else st.warning)(STATE_TEXT.get(state, state))
    if bundle.integrity:
        ig = bundle.integrity
        st.subheader("Cell status counts")
        names = {
            "expected": "Expected",
            "attempted": "Attempted",
            "valid": "Valid",
            "invalid_output": "Invalid output",
            "runtime_error": "Runtime error",
            "timeout": "Timeout",
            "missing": "Missing",
            "complete": "Complete",
            "log_lines": "Log lines",
        }
        st.dataframe(
            pd.DataFrame([{names[k]: ig["counts"][k] for k in names if k in ig["counts"]}]),
            hide_index=True,
            width="stretch",
        )
        st.markdown(
            f"Integrity checks: **{'PASS' if ig['integrity_ok'] else 'FAIL'}** — corrupt "
            f"{len(ig['corrupt'])}, duplicate {len(ig['duplicates'])}, mismatched "
            f"{len(ig['mismatched'])}, out-of-manifest {len(ig['out_of_manifest'])}, "
            f"inconsistent {len(ig['inconsistent'])}, missing {len(ig['missing_cells'])}."
        )
        if ig["missing_cells"]:
            with st.expander(f"Missing cells ({len(ig['missing_cells'])})"):
                st.dataframe(pd.DataFrame(ig["missing_cells"]), hide_index=True)
    else:
        st.info("No integrity report yet (no analysed run for this source).")
    if bundle.metrics:
        op = bundle.metrics["operational"]
        st.subheader("Operational quality")
        lat = op["latency_s"]

        def fmt(x: float | None) -> str:
            return "—" if x is None else f"{x:.2f}"

        def tok(t: dict[str, Any]) -> str:
            return (
                f"total {t['total']:,}, median {t['median']:.0f}, p95 {t['p95']:.0f}, "
                f"max {t['max']:,} over {t['n']} cells"
            )

        def fail_text(f: dict[str, int]) -> str:
            return ", ".join(f"{k}: {v}" for k, v in f.items()) if f else "none"

        st.markdown(
            f"- Latency (s): median {fmt(lat['median'])}, p95 {fmt(lat['p95'])}, "
            f"max {fmt(lat['max'])} over {lat['n']} attempted cells\n"
            f"- Prompt tokens: {tok(op['prompt_tokens'])}\n"
            f"- Completion tokens: {tok(op['completion_tokens'])}\n"
            f"- Failure reasons: {fail_text(op['failure_reasons'])}\n"
            f"- Cost: ${op['cost_usd']:.2f} — {op['cost_note']}"
        )
    prov = bundle.provenance
    if prov:
        st.subheader("Provenance")
        st.markdown(f"Run `{prov['run_id']}` · UTC {prov['utc_start']} → {prov['utc_end']}")
        st.json(
            {
                k: prov[k]
                for k in (
                    "source",
                    "manifest_sha256",
                    "sampling",
                    "model",
                    "hashes",
                    "versions",
                    "decoding",
                    "hardware",
                    "git_at_run_start",
                    "git_at_analysis",
                )
            },
            expanded=False,
        )
    elif bundle.freeze:
        st.subheader("Frozen study (no run yet)")
        st.json(bundle.freeze, expanded=False)
    if bundle.preflight:
        st.subheader(f"Preflight report: {bundle.preflight['verdict']}")
        st.json(bundle.preflight, expanded=False)
    if bundle.manifest_meta:
        st.subheader("Sample manifest")
        mm = bundle.manifest_meta
        st.markdown(
            f"N = {mm['n_pairs']}, seed {mm['sampling_seed']}, "
            f"{mm['sampling_rule_version']}, {mm['aggregation_rule_version']}"
        )
        st.json(mm["artifacts_sha256"])
    if bundle.error_review:
        er = bundle.error_review
        st.subheader("Error review (seeded selection)")
        st.markdown(
            f"{len(er['selected_pair_ids'])} of {er['eligible_incorrect_pairs']} "
            f"incorrect resolved pairs selected with seed {er['seed']}."
        )
        if er.get("summary"):
            st.markdown(f"**Summary.** {er['summary']}")
        st.markdown(
            "\n".join(
                f"- **{c['category']}** · `{c['pair_id']}` · human: "
                f"{HUMAN_TEXT.get(c['human_aggregate'], c['human_aggregate'])} — "
                f"{c['observation'] or '—'}"
                for c in er["cases"]
            )
        )


{
    "Overview": view_overview,
    "Agreement & Coverage": view_agreement,
    "Order & Verbosity": view_order,
    "Pair Explorer": view_pairs,
    "Run Integrity & Provenance": view_integrity,
}[view]()

if source == SAMPLE_SOURCE:
    st.caption("SAMPLE DATA — synthetic.")
