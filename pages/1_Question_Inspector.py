"""Page 1 — Clinical Question Inspector.

Pick one of the 234 golden-set clinical questions. The page shows how each of
the 5 fixed architectures (NoRAG, Naive, Sparse, Hybrid, Multi-Hop) answers
it, side-by-side, with:

  - predicted letter + ✓/✗ vs the gold answer
  - RAGAS Faithfulness (red < 0.30 < amber < 0.70 < green)
  - retrieval-score summary
  - **memorisation badge** when the model is right but the chunks don't
    support it (Faithfulness < 0.5) — the central thesis claim made tangible

Selecting a card reveals the retrieved passages (with text) for that arch.

LIME passage-level attribution is shown when available (the cross-architecture
LIME runs only cover ~186 questions per arch on the test-split retrieval-changed
surface; the join with golden_234 only hits when the question is also in test).
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from app.utils.highlight import coef_to_colour, lime_chunk_index, normalise_coefs
from app.utils.loaders import (
    ARCH_TO_EXP_GOLDEN,
    REPO_ROOT,
    get_chunk,
    get_inspector_view,
    load_golden,
    load_lime,
)
from app.utils.theme import (
    PALETTE,
    correctness_badge,
    correctness_colour,
    faith_colour,
    faith_label,
    fmt_metric,
    memorisation_flag,
)

st.set_page_config(page_title="Question Inspector", layout="wide")

# ----------------------------------------------------------------------------
# Sidebar — question filters
# ----------------------------------------------------------------------------
st.title("Question Inspector")
st.caption(
    "Pick a clinical question · see how 5 architectures answer it · spot the "
    "**memorisation badge** where a correct answer is unsupported by the "
    "retrieved chunks."
)

golden = load_golden()

with st.sidebar:
    st.subheader("Filter questions")
    step_options = ["(all)"] + sorted(golden["meta_info"].dropna().unique().tolist())
    type_options = ["(all)"] + sorted(golden["question_type"].dropna().unique().tolist())
    mh_options = ["(all)", "yes", "no"]

    step_filter = st.selectbox("USMLE step", step_options, index=0)
    type_filter = st.selectbox("Question type", type_options, index=0)
    mh_filter = st.selectbox("Requires multi-hop", mh_options, index=0)

    filtered = golden.copy()
    if step_filter != "(all)":
        filtered = filtered[filtered["meta_info"] == step_filter]
    if type_filter != "(all)":
        filtered = filtered[filtered["question_type"] == type_filter]
    if mh_filter != "(all)":
        filtered = filtered[filtered["requires_multihop"] == mh_filter]

    st.caption(f"{len(filtered)} of {len(golden)} questions match.")

# ----------------------------------------------------------------------------
# Curated picks — "show me a juicy one" rotation
# ----------------------------------------------------------------------------
picks_path = REPO_ROOT / "app" / "curated" / "inspector_picks.json"
if picks_path.exists():
    with open(picks_path) as f:
        curated = json.load(f)
else:
    curated = []

c0, c1 = st.columns([2, 1])
with c0:
    qid_options = filtered["question_id"].tolist()
    if not qid_options:
        st.warning("No questions match the active filters. Loosen them in the sidebar.")
        st.stop()

    label_map = {
        qid: f"{qid} · {filtered.loc[filtered['question_id'] == qid, 'meta_info'].iloc[0]} · "
        f"{filtered.loc[filtered['question_id'] == qid, 'question_type'].iloc[0]}"
        for qid in qid_options
    }
    default_qid = st.session_state.get("inspector_qid", qid_options[0])
    if default_qid not in qid_options:
        default_qid = qid_options[0]

    chosen_qid = st.selectbox(
        "Pick a question",
        qid_options,
        index=qid_options.index(default_qid),
        format_func=lambda q: label_map[q],
    )
    st.session_state["inspector_qid"] = chosen_qid

with c1:
    st.markdown("")  # vertical alignment
    st.markdown("")
    if curated and st.button(
        "Surprise me — show a juicy one",
        help="Cycle through curated questions where MultiHop wins but Naive memorises.",
        width="stretch",
    ):
        idx = st.session_state.get("curated_idx", 0)
        pick = curated[idx % len(curated)]
        st.session_state["curated_idx"] = idx + 1
        # Force the selectbox onto a curated pick (must be in filter result;
        # otherwise we widen the filter automatically by storing and rerunning).
        st.session_state["inspector_qid"] = pick["question_id"]
        st.rerun()

view = get_inspector_view(chosen_qid)
if not view:
    st.error(f"No data for {chosen_qid}.")
    st.stop()

q = view["question"]

# ----------------------------------------------------------------------------
# Question header
# ----------------------------------------------------------------------------
st.markdown("---")
st.markdown(f"#### {chosen_qid}")
st.markdown(q["question"])

opts = q["options"] if isinstance(q["options"], dict) else {}
gold_letter = q["gold_answer_letter"]
opt_html = "  ".join(
    f"<span style='padding:4px 10px; border-radius:6px; background:#f1f5f9; "
    f"margin-right:4px;'>"
    f"<b>{letter}.</b> {opts.get(letter, '')}</span>"
    for letter in ["A", "B", "C", "D"]
    if letter in opts
)
st.markdown(opt_html, unsafe_allow_html=True)

meta_row = (
    f"**Gold:** `{gold_letter}` · "
    f"**Type:** `{q['question_type']}` · "
    f"**Step:** `{q['meta_info']}` · "
    f"**Multi-hop:** `{q['requires_multihop']}`"
)
st.markdown(meta_row)

# ----------------------------------------------------------------------------
# Card grid — 5 architectures
# ----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### How each architecture answers")

cols = st.columns(len(view["archs"]))
selected_arch_key = "inspector_arch"
if selected_arch_key not in st.session_state:
    st.session_state[selected_arch_key] = view["archs"][-1]["arch"]  # default = MultiHop

for col, av in zip(cols, view["archs"]):
    arch = av["arch"]
    pred = av["pred"]
    retr = av["retrieval"]
    ragas = av["ragas"] or {}

    pred_letter = pred.get("pred_letter", "?")
    is_correct = bool(pred.get("is_correct", False))
    faith = ragas.get("faithfulness")
    cp = ragas.get("context_precision")
    cr = ragas.get("context_recall")
    ans_correct = ragas.get("answer_correctness")
    ans_rel = ragas.get("answer_relevancy")
    n_chunks = len(retr["retrieved_chunk_ids"]) if retr else 0
    mean_score = (
        sum(retr["retrieved_chunk_scores"]) / len(retr["retrieved_chunk_scores"])
        if retr and retr["retrieved_chunk_scores"]
        else None
    )
    memo = memorisation_flag(is_correct, faith)

    border_colour = PALETTE.get(arch, "#cbd5e1")
    pred_colour = correctness_colour(is_correct)
    fc = faith_colour(faith)
    flabel = faith_label(faith)

    memo_badge = (
        "<div style='display:inline-block; background:#fde68a; color:#92400e; "
        "padding:2px 8px; border-radius:4px; font-size:0.75em; "
        "margin-top:6px;'>memorisation</div>"
        if memo
        else ""
    )

    with col:
        st.markdown(
            f"""
<div style='border:2px solid {border_colour}; border-radius:10px;
            padding:10px 12px; min-height:240px;'>
  <div style='display:flex; justify-content:space-between; align-items:center;'>
    <span style='font-weight:600;'>{arch}</span>
    <span style='background:{pred_colour}; color:white; padding:2px 8px;
                 border-radius:6px; font-weight:700; font-size:1.1em;'>
      {pred_letter} {correctness_badge(is_correct)}
    </span>
  </div>
  <div style='margin-top:8px;'>
    <div style='display:flex; align-items:center; gap:6px;'>
      <span style='width:10px; height:10px; border-radius:50%;
                   background:{fc}; display:inline-block;'></span>
      <span style='font-size:0.85em;'>Faith: <b>{fmt_metric(faith, 2)}</b>
        <span style='color:#64748b;'>({flabel})</span></span>
    </div>
    <div style='font-size:0.8em; color:#475569; margin-top:4px;'>
      Context P/R: {fmt_metric(cp, 2)} / {fmt_metric(cr, 2)}<br>
      Ans Correctness: {fmt_metric(ans_correct, 2)}<br>
      Retrieved chunks: {n_chunks} · mean score: {fmt_metric(mean_score, 2)}
    </div>
    {memo_badge}
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        if st.button(
            "View chunks",
            key=f"view_{arch}",
            width="stretch",
            type=("primary" if st.session_state[selected_arch_key] == arch else "secondary"),
        ):
            st.session_state[selected_arch_key] = arch
            st.rerun()

# ----------------------------------------------------------------------------
# Retrieved-chunk drill-down
# ----------------------------------------------------------------------------
st.markdown("---")
selected_arch = st.session_state[selected_arch_key]
selected_av = next((av for av in view["archs"] if av["arch"] == selected_arch), None)

if not selected_av:
    st.info("Click 'View chunks' on a card above.")
    st.stop()

st.markdown(f"### Retrieved chunks — {selected_arch}")

retr = selected_av["retrieval"]
if selected_arch == "NoRAG" or retr is None or not retr.get("retrieved_chunk_ids"):
    st.info("This architecture retrieved no chunks (NoRAG baseline).")
else:
    chunk_ids = retr["retrieved_chunk_ids"]
    chunk_scores = retr["retrieved_chunk_scores"]

    # LIME join — best effort. LIME is keyed by medqa_NNNNN; the golden
    # question_id is golden_NNN. We match by question text instead.
    lime_df = load_lime(selected_arch)
    lime_row = None
    if not lime_df.empty:
        # Join on the gold answer letter as a cheap filter, then accept the
        # row whose retrieved_chunk_ids set matches ours exactly. (LIME
        # passages are listed for the same retrieved set.)
        our_set = set(chunk_ids)
        for _, row in lime_df.iterrows():
            row_chunk_ids = [p["chunk_id"] for p in row.get("passages", [])]
            if set(row_chunk_ids) == our_set:
                lime_row = row.to_dict()
                break

    lime_idx = lime_chunk_index(lime_row) if lime_row else {}
    if lime_idx:
        coefs = [lime_idx[c]["correctness_coef"] if c in lime_idx else 0.0 for c in chunk_ids]
        coefs_norm = normalise_coefs(coefs)
        st.caption(
            f"LIME passage attribution available (n_samples={lime_row.get('n_samples')}). "
            "Green = pushed answer toward predicted letter · Red = pushed away."
        )
    else:
        coefs_norm = [0.0] * len(chunk_ids)

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
        header = (
            f"<div style='font-size:0.85em; color:#475569; margin-bottom:2px;'>"
            f"<b>#{i+1}</b> · {cid} · {book} · retrieval score: {cscore:.3f}{coef_str}"
            f"</div>"
        )
        body = (
            f"<div style='background:{bg}; padding:8px 10px; border-radius:6px; "
            f"margin-bottom:10px; white-space:pre-wrap; font-size:0.85em; "
            f"line-height:1.45;'>{text[:1200]}{'…' if len(text) > 1200 else ''}</div>"
        )
        st.markdown(header + body, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Reference answer (collapsible)
# ----------------------------------------------------------------------------
with st.expander("Reference answer (Phase 3 gold-context construction)"):
    st.markdown(f"**Reference answer:** {q.get('reference_answer', '—')}")
    st.markdown(f"**Explanation:** {q.get('reference_explanation', '—')}")
    if q.get("evidence_keywords"):
        st.markdown(f"**Evidence keywords:** {', '.join(q['evidence_keywords'])}")
