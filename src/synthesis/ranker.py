"""Weighted scoring and Pareto-frontier sanity check for EXP_16."""
from __future__ import annotations

import pandas as pd


# Plan §11 weights — they sum to 1.0.
DEFAULT_WEIGHTS: dict[str, float] = {
    "Accuracy": 0.25,
    "Faithfulness": 0.25,
    "Retrieval": 0.20,
    "Safety": 0.15,
    "Explainability": 0.10,
    "Latency": 0.05,
}


def weighted_score(
    normalised: pd.DataFrame,
    *,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Apply weights and return a `final_score`-augmented table sorted by rank.

    Args:
        normalised: output of `normaliser.normalise`.
        weights: defaults to plan §11 weights (sum to 1.0). If supplied, must
            contain a value for every component column.

    Returns:
        DataFrame with the original component columns + `final_score` +
        `rank` (1 = best), sorted by `rank`.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    missing = set(weights) - set(normalised.columns)
    if missing:
        raise KeyError(f"Weights reference unknown components: {sorted(missing)}")

    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 1e-9:
        raise ValueError(f"Weights must sum to 1.0; got {total_weight:.6f}")

    out = normalised.copy()
    out["final_score"] = sum(out[c] * w for c, w in weights.items())
    out = out.sort_values("final_score", ascending=False).reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    return out


def pareto_frontier(
    raw: pd.DataFrame,
    *,
    benefit_col: str = "accuracy_test_1273",
    cost_col: str = "groq_calls_per_q",
) -> pd.DataFrame:
    """Tag each row as 'frontier' or 'dominated' on (benefit, cost).

    A point is dominated if some other row has higher-or-equal benefit AND
    lower-or-equal cost, with at least one strict inequality. The returned
    table keeps the input columns plus a `pareto_status` column.
    """
    points = raw[["architecture", benefit_col, cost_col]].to_dict("records")
    statuses = []
    for me in points:
        dominated = False
        for other in points:
            if other["architecture"] == me["architecture"]:
                continue
            if (
                other[benefit_col] >= me[benefit_col]
                and other[cost_col] <= me[cost_col]
                and (
                    other[benefit_col] > me[benefit_col]
                    or other[cost_col] < me[cost_col]
                )
            ):
                dominated = True
                break
        statuses.append("DOMINATED" if dominated else "frontier")
    out = raw.copy()
    out["pareto_status"] = statuses
    return out
