"""Map the ranked architecture table to use-case recommendations.

Plan §11 specifies five categories:
- Lowest cost           → cheapest architecture that still produces an answer
                          (here: fewest Groq calls per question)
- Highest accuracy      → max accuracy on test_1273
- Lowest hallucination  → min hallucination_rate (or, equivalently, max safety)
- Highest explainability → max LIME-SHAP Spearman ρ
- Best balanced         → top-ranked architecture in the weighted score
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd


def _idx_argmax(df: pd.DataFrame, col: str, *, ascending: bool = False) -> str:
    sub = df.dropna(subset=[col])
    if sub.empty:
        return "n/a"
    if ascending:
        return str(sub.loc[sub[col].idxmin()]["architecture"])
    return str(sub.loc[sub[col].idxmax()]["architecture"])


def use_case_recommendations(
    raw: pd.DataFrame,
    ranked: pd.DataFrame,
) -> pd.DataFrame:
    """Map the four use cases from plan §11 to specific architectures.

    Args:
        raw: the unnormalised metrics table from
            `aggregator.collect_architecture_metrics`. Used for the
            lowest-cost / highest-accuracy / lowest-hallucination /
            highest-explainability picks.
        ranked: the weighted-score table from `ranker.weighted_score`. Used
            for the best-balanced pick.

    Returns:
        A long-form DataFrame with columns `use_case`, `architecture`,
        `metric`, `value`, `rationale`.
    """
    rows: list[dict] = []

    cheapest = _idx_argmax(raw, "groq_calls_per_q", ascending=True)
    cheapest_value = float(raw.set_index("architecture").loc[cheapest, "groq_calls_per_q"]) if cheapest != "n/a" else float("nan")
    rows.append(
        {
            "use_case": "Lowest cost",
            "architecture": cheapest,
            "metric": "groq_calls_per_q",
            "value": cheapest_value,
            "rationale": "Fewest Groq calls per question (cheapest deployment).",
        }
    )

    best_acc = _idx_argmax(raw, "accuracy_test_1273")
    best_acc_value = float(raw.set_index("architecture").loc[best_acc, "accuracy_test_1273"]) if best_acc != "n/a" else float("nan")
    rows.append(
        {
            "use_case": "Highest accuracy",
            "architecture": best_acc,
            "metric": "accuracy_test_1273",
            "value": best_acc_value,
            "rationale": "Top exact-match on the 1,273-question contamination-clean test split.",
        }
    )

    safest = _idx_argmax(raw, "hallucination_rate_golden_234", ascending=True)
    safest_value = float(raw.set_index("architecture").loc[safest, "hallucination_rate_golden_234"]) if safest != "n/a" else float("nan")
    rows.append(
        {
            "use_case": "Lowest hallucination",
            "architecture": safest,
            "metric": "hallucination_rate_golden_234",
            "value": safest_value,
            "rationale": "Lowest RAGAS Hallucination_Rate (fraction with Faithfulness < 0.5) on golden_234.",
        }
    )

    most_explainable = _idx_argmax(raw, "lime_shap_spearman")
    explain_value = float(raw.set_index("architecture").loc[most_explainable, "lime_shap_spearman"]) if most_explainable != "n/a" else float("nan")
    rows.append(
        {
            "use_case": "Highest explainability",
            "architecture": most_explainable,
            "metric": "lime_shap_spearman",
            "value": explain_value,
            "rationale": "Highest LIME-SHAP Spearman ρ on the EXP_12 retrieval-changed surface (Multi-Hop measured directly; Adaptive variants inherit by routing share).",
        }
    )

    if not ranked.empty:
        top_row = ranked.iloc[0]
        rows.append(
            {
                "use_case": "Best balanced",
                "architecture": str(top_row["architecture"]),
                "metric": "final_score",
                "value": float(top_row["final_score"]),
                "rationale": (
                    "Top of the plan §11 weighted ranking "
                    "(0.25·Accuracy + 0.25·Faithfulness + 0.20·Retrieval + "
                    "0.15·Safety + 0.10·Explainability + 0.05·Latency)."
                ),
            }
        )

    return pd.DataFrame(rows)
