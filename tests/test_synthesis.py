"""Unit tests for `src/synthesis/{normaliser,ranker,recommender}.py`.

    .venv/bin/python tests/test_synthesis.py

The aggregator is exercised end-to-end by `notebooks/09_exp16_final_ranking.ipynb`
(it reads actual on-disk artefacts and is not easily mocked); these tests cover
the pure-Python transforms.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from src.synthesis.normaliser import normalise
from src.synthesis.ranker import DEFAULT_WEIGHTS, pareto_frontier, weighted_score
from src.synthesis.recommender import use_case_recommendations


def _toy_raw() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # NoRAG: high acc, zero retrieval/safety, fast.
            {
                "architecture": "NoRAG",
                "accuracy_test_1273": 0.77,
                "mean_latency_s_test_1273": 0.30,
                "groq_calls_per_q": 1.0,
                "faithfulness_golden_234": float("nan"),
                "hallucination_rate_golden_234": float("nan"),
                "context_precision_golden_234": float("nan"),
                "context_recall_golden_234": float("nan"),
                "answer_correctness_golden_234": 0.87,
                "answer_relevance_golden_234": 0.60,
                "lime_shap_spearman": float("nan"),
                "n_explainability_questions": 0,
            },
            # MultiHop: best on every measured dimension, slow.
            {
                "architecture": "MultiHop",
                "accuracy_test_1273": 0.80,
                "mean_latency_s_test_1273": 0.70,
                "groq_calls_per_q": 3.0,
                "faithfulness_golden_234": 0.28,
                "hallucination_rate_golden_234": 0.74,
                "context_precision_golden_234": 0.37,
                "context_recall_golden_234": 0.71,
                "answer_correctness_golden_234": 0.87,
                "answer_relevance_golden_234": 0.60,
                "lime_shap_spearman": 0.63,
                "n_explainability_questions": 134,
            },
        ]
    )


def test_normalise_keeps_native_scale_for_higher_better() -> None:
    raw = _toy_raw()
    out = normalise(raw)
    # Accuracy should still be the raw value for both rows (both in [0,1]).
    assert math.isclose(out.loc[0, "Accuracy"], 0.77)
    assert math.isclose(out.loc[1, "Accuracy"], 0.80)


def test_normalise_inverts_latency() -> None:
    raw = _toy_raw()
    out = normalise(raw)
    # Faster (lower latency) should score higher.
    no_rag_lat = out.loc[out["architecture"] == "NoRAG", "Latency"].iloc[0]
    mhop_lat = out.loc[out["architecture"] == "MultiHop", "Latency"].iloc[0]
    assert no_rag_lat > mhop_lat
    # Min-max scaled → min latency = 1.0, max latency = 0.0.
    assert math.isclose(no_rag_lat, 1.0)
    assert math.isclose(mhop_lat, 0.0)


def test_normalise_nan_floor_zero() -> None:
    raw = _toy_raw()
    out = normalise(raw)
    # NoRAG had NaN Faithfulness / Retrieval / Safety / Explainability →
    # the default nan_floor=0 should land them at 0.
    no_rag = out.loc[out["architecture"] == "NoRAG"].iloc[0]
    assert math.isclose(no_rag["Faithfulness"], 0.0)
    assert math.isclose(no_rag["Retrieval"], 0.0)
    assert math.isclose(no_rag["Safety"], 0.0)
    assert math.isclose(no_rag["Explainability"], 0.0)


def test_normalise_retrieval_composite_is_mean_of_cp_cr() -> None:
    raw = _toy_raw()
    out = normalise(raw)
    mhop = out.loc[out["architecture"] == "MultiHop"].iloc[0]
    # Multi-Hop: CP=0.37, CR=0.71 → composite 0.54.
    assert math.isclose(mhop["Retrieval"], (0.37 + 0.71) / 2.0, rel_tol=1e-9)


def test_normalise_safety_is_one_minus_hr() -> None:
    raw = _toy_raw()
    out = normalise(raw)
    mhop = out.loc[out["architecture"] == "MultiHop"].iloc[0]
    assert math.isclose(mhop["Safety"], 1.0 - 0.74, rel_tol=1e-9)


def test_weighted_score_sums_per_plan_weights() -> None:
    raw = _toy_raw()
    out = normalise(raw)
    ranked = weighted_score(out)
    mhop = ranked.loc[ranked["architecture"] == "MultiHop"].iloc[0]
    expected = sum(mhop[c] * w for c, w in DEFAULT_WEIGHTS.items())
    assert math.isclose(mhop["final_score"], expected, rel_tol=1e-9)


def test_weighted_score_ranks_descending() -> None:
    raw = _toy_raw()
    out = normalise(raw)
    ranked = weighted_score(out)
    # First row should have the highest final_score (rank 1).
    assert ranked["rank"].tolist() == [1, 2]
    scores = ranked["final_score"].tolist()
    assert scores[0] >= scores[1]


def test_weighted_score_rejects_unsummed_weights() -> None:
    raw = _toy_raw()
    out = normalise(raw)
    try:
        weighted_score(out, weights={c: 0.1 for c in DEFAULT_WEIGHTS})
    except ValueError:
        return
    raise AssertionError("expected ValueError on non-1.0 weights")


def test_pareto_frontier_tags_dominated_rows() -> None:
    raw = pd.DataFrame(
        [
            {"architecture": "Cheap", "accuracy_test_1273": 0.77, "groq_calls_per_q": 1.0},
            {"architecture": "Mid", "accuracy_test_1273": 0.78, "groq_calls_per_q": 1.8},
            {"architecture": "Expensive", "accuracy_test_1273": 0.80, "groq_calls_per_q": 3.0},
            {"architecture": "Dominated", "accuracy_test_1273": 0.76, "groq_calls_per_q": 2.0},
        ]
    )
    out = pareto_frontier(raw)
    status = dict(zip(out["architecture"], out["pareto_status"]))
    assert status["Cheap"] == "frontier"
    assert status["Mid"] == "frontier"
    assert status["Expensive"] == "frontier"
    assert status["Dominated"] == "DOMINATED"


def test_recommendations_pick_extreme_archs() -> None:
    raw = _toy_raw()
    out = normalise(raw)
    ranked = weighted_score(out)
    recs = use_case_recommendations(raw, ranked).set_index("use_case")["architecture"].to_dict()
    # Cheapest = NoRAG (1.0 vs 3.0 calls). Highest acc = MultiHop (0.80 > 0.77).
    # Only Multi-Hop has a measured hallucination rate → lowest = MultiHop.
    # Only Multi-Hop has explainability → highest = MultiHop.
    assert recs["Lowest cost"] == "NoRAG"
    assert recs["Highest accuracy"] == "MultiHop"
    assert recs["Lowest hallucination"] == "MultiHop"
    assert recs["Highest explainability"] == "MultiHop"


def test_default_weights_sum_to_one() -> None:
    assert math.isclose(sum(DEFAULT_WEIGHTS.values()), 1.0, abs_tol=1e-9)


if __name__ == "__main__":
    tests = [
        test_default_weights_sum_to_one,
        test_normalise_keeps_native_scale_for_higher_better,
        test_normalise_inverts_latency,
        test_normalise_nan_floor_zero,
        test_normalise_retrieval_composite_is_mean_of_cp_cr,
        test_normalise_safety_is_one_minus_hr,
        test_weighted_score_sums_per_plan_weights,
        test_weighted_score_ranks_descending,
        test_weighted_score_rejects_unsummed_weights,
        test_pareto_frontier_tags_dominated_rows,
        test_recommendations_pick_extreme_archs,
    ]
    for t in tests:
        t()
        print(f"  ✓ {t.__name__}")
    print(f"\nAll {len(tests)} synthesis tests passed.")
