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
        y=alt.Y("metric:N", title=None, sort=None),
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
        (rule + point).properties(title=title, height=44 * len(data) + 40), use_container_width=True
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
view = st.sidebar.radio(
    "View",
    [
        "Overview",
        "Agreement & Coverage",
        "Order & Verbosity",
        "Pair Explorer",
        "Run Integrity & Provenance",
    ],
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
        f"{len(bundle.manifest)}. Human label counts: `{ag['human_label_counts']}`."
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
        f"{fmt_boot(ag['accuracy_original_minus_swapped'], pct=True)}"
    )
    st.subheader("Reported separately (not scored)")
    t = ag["human_tie_pairs"]
    st.markdown(
        f"- **Human-tie pairs:** {t['pairs']}; judge verdicts on their {t['valid_judge_cells']} "
        f"valid cells: `{t['judge_verdicts']}`\n"
        f"- **Unresolved human pairs:** {ag['human_unresolved_pairs']}\n"
        f"- **Excluded annotations in sample** (duplicate/conflicting votes): "
        f"{ag['excluded_annotations_in_sample']}"
    )
    st.caption(m["baseline"]["rule"])


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
    dec.index.name = "Mapped verdict"
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
    st.markdown(
        f"**Judge − human longer-selection rate (resolved unequal pairs, paired):** "
        f"{fmt_boot(v['judge_minus_human_longer_rate_on_resolved_unequal'], pct=True)}"
    )
    st.caption("Association only; this design cannot show that length causes the judge's choice.")


HUMAN_TEXT = {
    "cand_1": "Answer 1 preferred",
    "cand_2": "Answer 2 preferred",
    "tie": "Tie",
    "unresolved": "Unresolved (no strict plurality)",
}
MAPPED_TEXT = {"cand_1": "Answer 1", "cand_2": "Answer 2", "tie": "Tie", None: "—"}


def view_pairs() -> None:
    st.title("Pair Explorer")
    if not bundle.manifest:
        st.info("No manifest found.")
        return
    cats = sorted({m["category"] for m in bundle.manifest})
    c1, c2 = st.columns(2)
    cat = c1.selectbox("Category", ["all", *cats])
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
    ]
    if not pairs:
        st.info("No pairs match the filters.")
        return
    idx = st.selectbox(
        "Pair",
        range(len(pairs)),
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
        col.text_area(name, c["text"], height=320, disabled=True, label_visibility="collapsed")
    st.subheader("Judge outputs")
    rows = []
    for order, disp in (
        ("original", "A = Answer 1, B = Answer 2"),
        ("swapped", "A = Answer 2, B = Answer 1"),
    ):
        rec = bundle.cells.get((p["pair_id"], order))
        rows.append(
            {
                "Order": order,
                "Displayed as": disp,
                "Status": rec["status"] if rec else "missing",
                "Displayed verdict": (rec or {}).get("verdict_displayed") or "—",
                "Mapped verdict": MAPPED_TEXT.get((rec or {}).get("mapped_verdict")),
                "Rationale": ((rec or {}).get("parse") or {}).get("rationale") or "—",
                "Latency (s)": (rec or {}).get("latency_s"),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with st.expander("Raw visible responses"):
        for order in ("original", "swapped"):
            rec = bundle.cells.get((p["pair_id"], order))
            st.markdown(f"**{order}**")
            st.code((rec or {}).get("raw_response") or "(none)", language="json")


def view_integrity() -> None:
    st.title("Run Integrity & Provenance")
    (st.success if state == "COMPLETE" else st.warning)(STATE_TEXT.get(state, state))
    if bundle.integrity:
        ig = bundle.integrity
        st.subheader("Cell status counts")
        st.dataframe(pd.DataFrame([ig["counts"]]), hide_index=True, width="stretch")
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

        st.markdown(
            f"- Latency (s): median {fmt(lat['median'])}, p95 {fmt(lat['p95'])}, "
            f"max {fmt(lat['max'])} over {lat['n']} attempted cells\n"
            f"- Prompt tokens: `{op['prompt_tokens']}`\n"
            f"- Completion tokens: `{op['completion_tokens']}`\n"
            f"- Failure reasons: `{op['failure_reasons']}`\n"
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
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "pair_id": c["pair_id"],
                        "category": c["category"],
                        "human": c["human_aggregate"],
                        "observation": c["observation"] or "—",
                    }
                    for c in er["cases"]
                ]
            ),
            hide_index=True,
            width="stretch",
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
