"""Page 2 — Pareto Explorer + weight sliders.

Two-column layout:

  - LEFT: Pareto plot of accuracy_test_1273 vs groq_calls_per_q. Frontier
    architectures (NoRAG · Adaptive_A · MultiHop) are filled; dominated ones
    are open circles.
  - RIGHT: six sliders for the plan §11 weight vector + a "Reset to locked"
    button. The ranking table re-sorts in real time. A regime preset dropdown
    snaps the sliders to the three alternative weight regimes recorded in
    `results/exp_16_final_synthesis/sensitivity_ranks.csv`.

The component scores used for re-ranking are the **already-normalised** ones
written by `src/synthesis/normaliser.py` (see `component_scores_normalised.csv`).
This is exactly the math behind `table12_final_ranking.csv`, so the page
reproduces the canonical ranking when sliders are at the locked weights.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app.utils.loaders import load_synthesis
from app.utils.theme import LOCKED_WEIGHTS, PALETTE

st.set_page_config(page_title="Pareto Explorer", layout="wide")

st.title("Pareto Explorer")
st.caption(
    "Drag the weight sliders to see how the ranking shifts. The locked plan §11 "
    "weights (Accuracy 0.25 · Faithfulness 0.25 · Retrieval 0.20 · Safety 0.15 "
    "· Explainability 0.10 · Latency 0.05) reproduce Table 12. Adjust them and "
    "watch the leader swap — that is the sensitivity argument made tangible."
)

synth = load_synthesis()
components = synth["components_normalised"].set_index("architecture")
sensitivity = synth["sensitivity"].set_index("architecture")
pareto = synth["pareto"]

# Plan-doc style regime presets (mirrors regimes used in sensitivity_ranks.csv).
REGIME_WEIGHTS: dict[str, dict[str, float]] = {
    "plan_default":   {"Accuracy": 0.25, "Faithfulness": 0.25, "Retrieval": 0.20, "Safety": 0.15, "Explainability": 0.10, "Latency": 0.05},
    "accuracy_heavy": {"Accuracy": 0.50, "Faithfulness": 0.20, "Retrieval": 0.15, "Safety": 0.05, "Explainability": 0.05, "Latency": 0.05},
    "safety_heavy":   {"Accuracy": 0.20, "Faithfulness": 0.30, "Retrieval": 0.15, "Safety": 0.30, "Explainability": 0.05, "Latency": 0.00},
    "compute_heavy":  {"Accuracy": 0.20, "Faithfulness": 0.20, "Retrieval": 0.15, "Safety": 0.10, "Explainability": 0.15, "Latency": 0.20},
}

if "weights" not in st.session_state:
    st.session_state["weights"] = dict(LOCKED_WEIGHTS)
if "regime" not in st.session_state:
    st.session_state["regime"] = "plan_default"


def apply_regime(name: str) -> None:
    if name in REGIME_WEIGHTS:
        st.session_state["weights"] = dict(REGIME_WEIGHTS[name])
        st.session_state["regime"] = name


# ----------------------------------------------------------------------------
# Layout — Pareto plot (left) + slider panel (right)
# ----------------------------------------------------------------------------
left, right = st.columns([3, 2], gap="large")

with right:
    st.subheader("Weight vector")

    rcol1, rcol2 = st.columns([3, 1])
    with rcol1:
        regime = st.selectbox(
            "Preset",
            options=list(REGIME_WEIGHTS.keys()),
            index=list(REGIME_WEIGHTS.keys()).index(st.session_state["regime"]),
            help=(
                "Apply one of the four weight regimes used in plan §11's "
                "sensitivity analysis. Snaps every slider at once."
            ),
        )
    with rcol2:
        st.markdown(" ")
        if st.button("Apply", width="stretch"):
            apply_regime(regime)
            st.rerun()

    if st.button("Reset to plan §11 locked", width="stretch"):
        apply_regime("plan_default")
        st.rerun()

    w = st.session_state["weights"]
    new_w: dict[str, float] = {}
    for dim, default in LOCKED_WEIGHTS.items():
        new_w[dim] = st.slider(
            dim,
            min_value=0.0,
            max_value=1.0,
            value=float(w[dim]),
            step=0.05,
            key=f"slider_{dim}",
        )
    st.session_state["weights"] = new_w

    weight_sum = sum(new_w.values())
    if abs(weight_sum - 1.0) > 0.001:
        st.warning(
            f"Weights sum to **{weight_sum:.2f}** (not 1.00). The ranking is "
            "computed from the un-normalised slider values — interpret with care."
        )

# Re-compute the ranking from the normalised component scores
weighted = pd.DataFrame(index=components.index)
for dim, weight in new_w.items():
    weighted[dim] = components[dim] * weight
weighted["final_score"] = weighted.sum(axis=1)
weighted = weighted.sort_values("final_score", ascending=False)
weighted["rank"] = range(1, len(weighted) + 1)

leader = weighted.iloc[0]
locked_top = synth["ranking"].iloc[0]["architecture"]
leader_swapped = leader.name != locked_top

with left:
    st.subheader("Pareto frontier")

    pareto_plot = pareto.copy()
    pareto_plot["status_label"] = pareto_plot["pareto_status"].map(
        {"frontier": "Frontier", "DOMINATED": "Dominated"}
    )
    fig = px.scatter(
        pareto_plot,
        x="groq_calls_per_q",
        y="accuracy_test_1273",
        color="architecture",
        symbol="status_label",
        symbol_map={"Frontier": "circle", "Dominated": "circle-open"},
        color_discrete_map=PALETTE,
        text="architecture",
        size_max=20,
    )
    fig.update_traces(textposition="top center", marker=dict(size=14))
    fig.update_layout(
        xaxis_title="Groq calls per question (cost proxy)",
        yaxis_title="Accuracy on test_1273",
        showlegend=False,
        height=460,
        margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig, width="stretch")

    st.caption(
        "Frontier (filled): NoRAG (cheapest) · Adaptive_A (mid-frontier) · "
        "MultiHop (highest accuracy). Adaptive_B and Naive/Sparse/Hybrid are "
        "dominated."
    )

# ----------------------------------------------------------------------------
# Re-computed ranking table
# ----------------------------------------------------------------------------
st.markdown("---")
st.subheader("Live ranking (re-weighted)")

if leader_swapped:
    st.info(
        f"**Leader has swapped:** under your current weights, **{leader.name}** "
        f"is rank-1 (final_score = {leader['final_score']:.4f}). "
        f"Under plan §11 locked weights, **{locked_top}** wins."
    )
else:
    st.success(
        f"**{leader.name}** remains rank-1 under your current weights "
        f"(final_score = {leader['final_score']:.4f}). The ranking is robust."
    )

display = weighted.reset_index().rename(columns={"index": "architecture"})
display = display[["rank", "architecture"] + list(LOCKED_WEIGHTS.keys()) + ["final_score"]]
st.dataframe(
    display.style.format({c: "{:.4f}" for c in list(LOCKED_WEIGHTS.keys()) + ["final_score"]}),
    hide_index=True,
    width="stretch",
)

# ----------------------------------------------------------------------------
# Sensitivity matrix — recorded ranks across the four regimes
# ----------------------------------------------------------------------------
st.markdown("---")
st.subheader("Sensitivity across regimes (recorded ranks)")
st.caption(
    "Multi-Hop holds rank-1 under three of the four regimes; Naive takes rank-1 "
    "only under compute_heavy (latency weight quadrupled from 0.05 to 0.20)."
)
st.dataframe(sensitivity, width="stretch")
