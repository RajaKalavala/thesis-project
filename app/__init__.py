"""Streamlit demo app for the thesis (Phase 10).

Three-page viva demo:
  - Question Inspector  (memorisation vs grounding gap)
  - Pareto Explorer     (MultiHop wins under locked weights, robust under 3/4 regimes)
  - Hallucination Forensics  (confidence-aware rejection)

Cached-mode only: reads from results/exp_*/ artefacts on disk. No live LLM calls.
"""
