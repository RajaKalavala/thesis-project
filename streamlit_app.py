"""Streamlit Cloud entry point for the thesis demo.

Run locally:
    .venv/bin/streamlit run streamlit_app.py

The native Streamlit ``pages/`` directory mechanism is used — see ``pages/``
at the repo root.
"""
from __future__ import annotations

import streamlit as st

from app.utils.loaders import load_synthesis

st.set_page_config(
    page_title="Adaptive RAG with Confidence-Aware Hallucination Detection",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    st.title("Adaptive RAG with Confidence-Aware Hallucination Detection")
    st.caption(
        "MSc Thesis demo · MedQA · LLaMA 3.3 70B generator · BGE-large retriever · "
        "Claude Sonnet 4.6 RAGAS judge · Cached-mode, no live LLM calls."
    )

    synth = load_synthesis()
    ranking = synth["ranking"]
    summary = synth["summary"]

    st.markdown("### Three findings, three pages")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 1. Memorisation vs grounding")
        st.markdown(
            "**88% of correct Naive RAG answers are *ungrounded*** — LLaMA "
            "answers from pre-training, not retrieved chunks. Multi-Hop is the "
            "architecture that breaks the pattern."
        )
        st.page_link("pages/1_Question_Inspector.py", label="→ Open Question Inspector")
    with c2:
        st.markdown("#### 2. The locked ranking")
        top = ranking.iloc[0]
        st.markdown(
            f"**{top['architecture']} wins** ({top['final_score']:.4f}) under "
            f"plan §11 weights. The proposal's *“Adaptive should be best balanced”* "
            f"expectation is empirically **falsified**."
        )
        st.page_link("pages/2_Pareto_Explorer.py", label="→ Open Pareto Explorer")
    with c3:
        st.markdown("#### 3. Confidence-aware rejection")
        st.markdown(
            "**Multi-Hop + τ=0.5 rejection → 96.7% accuracy on accepted** "
            "(48% rejection rate). The thesis's central novelty contribution "
            "turns Multi-Hop into a safety-grade clinical-decision system."
        )
        st.page_link("pages/3_Hallucination_Forensics.py", label="→ Open Forensics")

    st.markdown("---")

    st.markdown("### Headline ranking (plan §11 locked weights)")
    fmt = ranking.copy()
    for col in fmt.columns:
        if fmt[col].dtype == float:
            fmt[col] = fmt[col].round(4)
    st.dataframe(fmt, hide_index=True, width="stretch")

    st.caption(
        f"Weighted score = "
        f"{summary['weights']['Accuracy']}·Accuracy "
        f"+ {summary['weights']['Faithfulness']}·Faithfulness "
        f"+ {summary['weights']['Retrieval']}·Retrieval "
        f"+ {summary['weights']['Safety']}·Safety "
        f"+ {summary['weights']['Explainability']}·Explainability "
        f"+ {summary['weights']['Latency']}·Latency."
    )

    with st.expander("Demo scope & methodology disclaimer"):
        st.markdown(
            "This is a **research-presentation tool**, not a clinical deployment. "
            "The proposal (§6) explicitly excludes real-time clinical use and "
            "patient-facing testing. The application reads cached experiment "
            "outputs (`results/exp_*/`), never makes a live LLM call, and every "
            "displayed answer shows the system's confidence score and a "
            "safety-rejection indicator. See `plan.md §12` for the full "
            "Phase 10 scope guardrails."
        )


if __name__ == "__main__":
    main()
