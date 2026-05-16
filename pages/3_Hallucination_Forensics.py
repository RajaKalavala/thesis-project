"""Page 3 — Hallucination Forensics with τ slider.

The thesis's central novelty contribution made visible: the confidence-aware
rejection layer on Multi-Hop. The page has three parts:

  1. **τ slider** + signal-config selector → live accept-rate, accuracy on
     accepted, and recall of wrong rejected. Sourced from
     `results/exp_09_confidence_rejection/exp_05_multi_hop_rag__golden_234__threshold_sweeps.csv`.
  2. **Per-question table** of all 234 golden questions on Multi-Hop, with the
     verdict (ACCEPTED / REJECTED) recomputed live from the chosen signal
     config and threshold (using `src.confidence.signals.combine_signals`).
  3. **Forensics drill-down** on a selected row: retrieved chunks (LIME-coloured
     when available), hallucination taxonomy category + rationale.

Bottom: stacked bar of the 5-architecture hallucination taxonomy
(`results/exp_15_taxonomy_analysis/table7_proportions.csv`).
"""
from __future__ import annotations

import sys

import pandas as pd
import plotly.express as px
import streamlit as st

from app.utils.highlight import coef_to_colour, lime_chunk_index, normalise_coefs
from app.utils.loaders import (
    REPO_ROOT,
    get_chunk,
    load_confidence_signals,
    load_golden,
    load_lime,
    load_predictions,
    load_retrieval,
    load_taxonomy,
    load_taxonomy_proportions,
    load_threshold_sweeps,
)
from app.utils.theme import PALETTE, fmt_metric

# Allow importing src/ from the page directly so the signal-combining math
# matches what exp_09 used.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.confidence.signals import (  # noqa: E402
    RAGAS_SIGNALS,
    RETRIEVAL_SIGNALS,
    combine_signals,
)

st.set_page_config(page_title="Hallucination Forensics", layout="wide")

st.title("Hallucination Forensics")
st.caption(
    "Multi-Hop on golden_234 with confidence-aware rejection. Drag the τ slider "
    "to see how the safety-grade operating point moves: at τ=0.5 with the "
    "combined signal, Multi-Hop reaches 0.9669 accuracy on the 121 accepted "
    "questions (vs the unfiltered 0.901 baseline)."
)

# ---------------------------------------------------------------------------
# τ slider + config selector — top control bar
# ---------------------------------------------------------------------------
SIGNAL_CONFIGS = {
    "combined":          "All 8 signals, equal-weighted",
    "RAGAS_only":        "RAGAS only (faithfulness · CP · CR · answer_relevancy)",
    "faithfulness_only": "RAGAS Faithfulness alone",
    "retrieval_only":    "Retrieval scores only (mean · max · var · n_chunks)",
}

sweeps = load_threshold_sweeps()
signals = load_confidence_signals()
golden = load_golden()
preds = load_predictions("MultiHop", "golden_234")

cc1, cc2, cc3 = st.columns([1.2, 1, 2.5])
with cc1:
    config = st.selectbox(
        "Signal configuration",
        options=list(SIGNAL_CONFIGS.keys()),
        index=0,
        format_func=lambda k: f"{k} — {SIGNAL_CONFIGS[k]}",
    )
with cc2:
    tau = st.slider("τ (rejection threshold)", 0.30, 0.90, 0.50, 0.05)

with cc3:
    sweep_at_tau = sweeps[(sweeps["config"] == config) & (sweeps["threshold"] == tau)]
    if not sweep_at_tau.empty:
        row = sweep_at_tau.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accept rate", f"{1 - row['rejection_rate']:.1%}")
        c2.metric(
            "Accuracy on accepted",
            f"{row['accuracy_on_accepted']:.1%}" if pd.notna(row["accuracy_on_accepted"]) else "—",
            delta=f"{row['accuracy_uplift']*100:+.2f} pp",
        )
        c3.metric("Wrong-Q rejected", f"{row['recall_of_wrong_rejected']:.1%}")
        c4.metric("Accepted / Total", f"{int(row['n_accepted'])} / {int(row['n_total'])}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Live recompute: per-question confidence + verdict
# ---------------------------------------------------------------------------
if config == "combined":
    cols = tuple(list(RETRIEVAL_SIGNALS) + list(RAGAS_SIGNALS))
    conf = combine_signals(signals, signal_columns=cols)
elif config == "RAGAS_only":
    conf = combine_signals(signals, signal_columns=RAGAS_SIGNALS)
elif config == "faithfulness_only":
    conf = signals["faithfulness"]
else:  # retrieval_only
    conf = combine_signals(signals, signal_columns=RETRIEVAL_SIGNALS)

table = pd.DataFrame({
    "question_id": signals["question_id"],
    "gold": signals["gold_letter"],
    "pred": signals["pred_letter"],
    "is_correct": signals["is_correct"].astype(bool),
    "confidence": conf,
    "faithfulness": signals["faithfulness_raw"],
})
table["verdict"] = table["confidence"].apply(
    lambda c: "REJECTED" if pd.isna(c) or c < tau else "ACCEPTED"
)

# Join question text + question_type + multihop flag for filtering
qtext = golden[["question_id", "question_type", "meta_info", "requires_multihop"]]
table = table.merge(qtext, on="question_id", how="left")

# ---------------------------------------------------------------------------
# Pareto-style curve — accept-rate vs accuracy-on-accepted, by config
# ---------------------------------------------------------------------------
st.subheader("Rejection trade-off curve")
curve = sweeps.copy()
curve["accept_rate"] = 1 - curve["rejection_rate"]
curve = curve[curve["accuracy_on_accepted"].notna()]
fig = px.line(
    curve,
    x="accept_rate",
    y="accuracy_on_accepted",
    color="config",
    markers=True,
    hover_data=["threshold"],
)
# Highlight the active operating point
if not sweep_at_tau.empty:
    op = sweep_at_tau.iloc[0]
    fig.add_scatter(
        x=[1 - op["rejection_rate"]],
        y=[op["accuracy_on_accepted"]],
        mode="markers",
        marker=dict(size=18, symbol="star", color="#f43f5e"),
        name=f"τ={tau} (active)",
    )
fig.update_layout(
    xaxis_title="Accept rate (1 − rejection rate)",
    yaxis_title="Accuracy on accepted",
    height=350,
    margin=dict(l=0, r=0, t=20, b=0),
    legend_title_text="Signal config",
)
st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# Question filter + selection
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Per-question verdict table")

fc1, fc2 = st.columns([3, 1])
with fc1:
    verdict_filter = st.radio(
        "Show",
        options=["All", "Wrong answers (originally)", "Rejected (at current τ)", "Wrong AND rejected"],
        horizontal=True,
    )
with fc2:
    show_n = st.number_input("Rows", min_value=10, max_value=234, value=30, step=10)

view = table.copy()
if verdict_filter == "Wrong answers (originally)":
    view = view[~view["is_correct"]]
elif verdict_filter == "Rejected (at current τ)":
    view = view[view["verdict"] == "REJECTED"]
elif verdict_filter == "Wrong AND rejected":
    view = view[(~view["is_correct"]) & (view["verdict"] == "REJECTED")]

view = view.sort_values("confidence", ascending=True, na_position="first").head(show_n)

display = view[[
    "question_id", "gold", "pred", "is_correct",
    "confidence", "faithfulness", "verdict", "question_type", "requires_multihop",
]].copy()
display["is_correct"] = display["is_correct"].map({True: "✓", False: "✗"})
display["confidence"] = display["confidence"].apply(
    lambda x: f"{x:.3f}" if pd.notna(x) else "NaN"
)
display["faithfulness"] = display["faithfulness"].apply(
    lambda x: f"{x:.3f}" if pd.notna(x) else "—"
)

st.dataframe(display, hide_index=True, width="stretch", height=320)

# ---------------------------------------------------------------------------
# Forensics drill-down on a single question
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Forensics for a selected question")

qid_options = view["question_id"].tolist()
if not qid_options:
    st.info("No rows match the active filter.")
    st.stop()

chosen = st.selectbox("Question", qid_options, index=0)
selected_row = table[table["question_id"] == chosen].iloc[0].to_dict()
golden_row = golden[golden["question_id"] == chosen].iloc[0].to_dict()

# Header strip
verdict_colour = "#ef4444" if selected_row["verdict"] == "REJECTED" else "#10b981"
correctness_colour = "#10b981" if selected_row["is_correct"] else "#ef4444"
st.markdown(
    f"""
<div style='padding:10px 14px; border-radius:8px; background:#f8fafc;
            border:1px solid #cbd5e1;'>
  <div style='display:flex; gap:18px; flex-wrap:wrap;'>
    <span><b>{chosen}</b></span>
    <span style='background:{correctness_colour}; color:white; padding:2px 8px;
                 border-radius:6px;'>pred {selected_row['pred']} {'✓' if selected_row['is_correct'] else '✗'}</span>
    <span style='background:#1e293b; color:white; padding:2px 8px;
                 border-radius:6px;'>gold {selected_row['gold']}</span>
    <span style='background:{verdict_colour}; color:white; padding:2px 8px;
                 border-radius:6px;'>{selected_row['verdict']} at τ={tau}</span>
    <span>confidence = <b>{fmt_metric(selected_row['confidence'], 3)}</b></span>
    <span>faithfulness = <b>{fmt_metric(selected_row['faithfulness'], 3)}</b></span>
  </div>
  <div style='margin-top:8px; font-size:0.9em; color:#475569;'>
    {golden_row['question'][:600]}{'…' if len(golden_row['question']) > 600 else ''}
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# Taxonomy lookup (only available for wrong answers)
tax = load_taxonomy("MultiHop")
if not selected_row["is_correct"] and not tax.empty:
    trow = tax[tax["question_id"] == chosen]
    if not trow.empty:
        cat = trow.iloc[0]["category"]
        rationale = trow.iloc[0]["rationale"]
        st.markdown(
            f"**Hallucination taxonomy:** `{cat}`  \n*{rationale}*"
        )

# Retrieved chunks + LIME if available
retr_df = load_retrieval("MultiHop", "golden_234")
retr_row = retr_df[retr_df["question_id"] == chosen]
if retr_row.empty:
    st.info("No retrieved chunks for this question.")
else:
    chunk_ids = retr_row.iloc[0]["retrieved_chunk_ids"]
    chunk_scores = retr_row.iloc[0]["retrieved_chunk_scores"]

    lime_df = load_lime("MultiHop")
    lime_row = None
    if not lime_df.empty:
        our_set = set(chunk_ids)
        for _, r in lime_df.iterrows():
            row_ids = [p["chunk_id"] for p in r.get("passages", [])]
            if set(row_ids) == our_set:
                lime_row = r.to_dict()
                break
    lime_idx = lime_chunk_index(lime_row) if lime_row else {}
    if lime_idx:
        coefs = [lime_idx[c]["correctness_coef"] if c in lime_idx else 0.0 for c in chunk_ids]
        coefs_norm = normalise_coefs(coefs)
        st.caption(
            "LIME passage attribution available · green = pushed toward "
            "predicted letter · red = pushed away."
        )
    else:
        coefs_norm = [0.0] * len(chunk_ids)

    with st.expander(f"Retrieved chunks ({len(chunk_ids)})", expanded=False):
        for i, (cid, cscore) in enumerate(zip(chunk_ids, chunk_scores)):
            chunk = get_chunk(cid)
            text = chunk["text"] if chunk else f"(chunk text not found: {cid})"
            book = chunk["book_name"] if chunk else "?"
            bg = coef_to_colour(coefs_norm[i]) if lime_idx else "rgba(241,245,249,0.6)"
            coef_str = (
                f" · LIME coef: <b>{lime_idx[cid]['correctness_coef']:+.3f}</b>"
                if lime_idx and cid in lime_idx
                else ""
            )
            st.markdown(
                f"<div style='font-size:0.85em; color:#475569; margin-bottom:2px;'>"
                f"<b>#{i+1}</b> · {cid} · {book} · score: {cscore:.3f}{coef_str}</div>"
                f"<div style='background:{bg}; padding:8px 10px; border-radius:6px; "
                f"margin-bottom:8px; white-space:pre-wrap; font-size:0.85em; "
                f"line-height:1.45;'>{text[:1000]}{'…' if len(text) > 1000 else ''}</div>",
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# Footer — 5-arch hallucination taxonomy (Table 7)
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Wrong-answer taxonomy across architectures (Table 7)")
st.caption(
    "Proportional breakdown of the 4 observed hallucination categories on each "
    "architecture's wrong-answer set. Better retrieval (Hybrid → Multi-Hop) shifts "
    "errors from `wrong_reasoning_chain` to `unsupported_treatment`: the model "
    "now reads grounded chunks but picks the wrong option."
)
tax_props = load_taxonomy_proportions()
tax_long = tax_props.melt(id_vars="category", var_name="architecture", value_name="proportion")
fig2 = px.bar(
    tax_long,
    x="architecture",
    y="proportion",
    color="category",
    barmode="stack",
    color_discrete_sequence=px.colors.qualitative.Set2,
)
fig2.update_layout(
    yaxis_title="Proportion of wrong answers",
    xaxis_title=None,
    height=380,
    margin=dict(l=0, r=0, t=20, b=0),
    legend_title_text="Category",
)
st.plotly_chart(fig2, width="stretch")
