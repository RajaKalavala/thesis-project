"""Map raw architecture metrics to [0, 1] component scores ready for weighting.

The default scheme keeps every metric at its native [0, 1] interpretation
wherever possible and only rescales the lower-is-better latency column.
Missing values (NaN) are conservatively floored to 0 so the weighted score
penalises architectures that didn't measure a dimension — the alternative
(re-normalising weights per-row) would silently reward unmeasured archs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# Component name → (raw-column, direction). 'higher' means raw is already a
# [0, 1] score; 'lower' means it must be inverted (e.g. latency).
COMPONENT_COLUMNS: dict[str, tuple[str, str]] = {
    "Accuracy": ("accuracy_test_1273", "higher"),
    "Faithfulness": ("faithfulness_golden_234", "higher"),
    "Retrieval": ("_retrieval_composite", "higher"),
    "Safety": ("_safety_score", "higher"),
    "Explainability": ("lime_shap_spearman", "higher"),
    "Latency": ("mean_latency_s_test_1273", "lower"),
}


def _retrieval_composite(row: pd.Series) -> float:
    """Mean of Context Precision and Context Recall (RAGAS retrieval-quality
    composite). NaN if either is missing.
    """
    cp = row.get("context_precision_golden_234")
    cr = row.get("context_recall_golden_234")
    if pd.isna(cp) or pd.isna(cr):
        return float("nan")
    return float((cp + cr) / 2.0)


def _safety_score(row: pd.Series) -> float:
    """1 − Hallucination_Rate. NaN if the rate is missing."""
    hr = row.get("hallucination_rate_golden_234")
    if pd.isna(hr):
        return float("nan")
    return float(1.0 - hr)


def normalise(
    raw: pd.DataFrame,
    *,
    scheme: str = "raw_with_minmax_latency",
    nan_floor: float = 0.0,
) -> pd.DataFrame:
    """Return a tidy [0, 1]-scored table.

    The output keeps the `architecture` column plus six component columns
    (`Accuracy`, `Faithfulness`, `Retrieval`, `Safety`, `Explainability`,
    `Latency`) — all in [0, 1].

    Args:
        raw: output of `aggregator.collect_architecture_metrics`.
        scheme: 'raw_with_minmax_latency' (default) keeps RAGAS-style metrics
            at their native value and only min-max-rescales latency.
            'minmax_all' rescales every column against the in-set min/max
            (sensitive to which architectures are included).
        nan_floor: value to substitute when a raw metric is NaN. 0.0 is the
            conservative default (a missing measurement scores zero).

    Returns:
        DataFrame with rows in the same order as `raw`.
    """
    if scheme not in {"raw_with_minmax_latency", "minmax_all"}:
        raise ValueError(f"Unknown scheme: {scheme}")

    out = pd.DataFrame({"architecture": raw["architecture"].values})

    # Derived columns
    enriched = raw.copy()
    enriched["_retrieval_composite"] = enriched.apply(_retrieval_composite, axis=1)
    enriched["_safety_score"] = enriched.apply(_safety_score, axis=1)

    for component, (col, direction) in COMPONENT_COLUMNS.items():
        values = pd.to_numeric(enriched[col], errors="coerce")
        if direction == "higher":
            if scheme == "minmax_all":
                normed = _minmax(values, invert=False)
            else:
                normed = values.clip(lower=0.0, upper=1.0)
        else:  # 'lower' → invert
            normed = _minmax(values, invert=True)
        out[component] = normed.fillna(nan_floor)

    return out


def _minmax(values: pd.Series, *, invert: bool) -> pd.Series:
    """Min-max scale to [0, 1]. If `invert=True`, lower raw → higher score."""
    finite = values.dropna()
    if finite.empty:
        return pd.Series([np.nan] * len(values), index=values.index)
    lo, hi = float(finite.min()), float(finite.max())
    if hi == lo:
        # All identical → everyone gets full marks (or zero — pick 1 so
        # ties don't drag down the weighted score for a uniform column).
        return values.where(values.isna(), 1.0)
    scaled = (values - lo) / (hi - lo)
    if invert:
        scaled = 1.0 - scaled
    return scaled.clip(lower=0.0, upper=1.0)
