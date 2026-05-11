"""Phase 9 / EXP_16 — Final synthesis: aggregate every architecture into one
weighted ranking row.

Public API
==========

`aggregator.collect_architecture_metrics(repo_root)`
    Read every `summary.json` (test_1273 + golden_234) and the auxiliary
    Phase 5/6 artefacts; return a tidy `ArchitectureMetrics` table — one row per
    architecture, raw values, with NaN where a metric was not measured.

`normaliser.normalise(table, scheme)`
    Map raw metric columns to [0, 1]. `scheme='raw_with_minmax_latency'` keeps
    every RAGAS-style metric at its native [0, 1] value and only min-max-
    rescales the lower-is-better latency column.

`ranker.weighted_score(table, weights)`
    Apply the plan §11 weights and produce a `final_score` column + rank
    ordering. Also reports the Pareto frontier on (accuracy, mean_latency_s)
    so the ranking comes with a sanity check that picks up trade-off rows.

`recommender.use_case_recommendations(table)`
    Map the ranked table to the four use-case picks from plan §11:
    lowest-cost, highest-accuracy, lowest-hallucination, highest-explainability,
    best-balanced.
"""
from src.synthesis.aggregator import (
    ArchitectureMetrics,
    ARCHITECTURES,
    collect_architecture_metrics,
)
from src.synthesis.normaliser import normalise
from src.synthesis.ranker import (
    DEFAULT_WEIGHTS,
    pareto_frontier,
    weighted_score,
)
from src.synthesis.recommender import use_case_recommendations

__all__ = [
    "ArchitectureMetrics",
    "ARCHITECTURES",
    "collect_architecture_metrics",
    "normalise",
    "DEFAULT_WEIGHTS",
    "pareto_frontier",
    "weighted_score",
    "use_case_recommendations",
]
