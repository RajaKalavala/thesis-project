"""LIME-coefficient → coloured chunk-text rendering.

Used by Page 1 (Inspector) and Page 3 (Forensics) to show which retrieved
passages drove the model's answer. Coefficients come from
``results/exp_10_lime_passage/*.jsonl`` — positive = pushed toward predicted
letter, negative = pushed away.
"""
from __future__ import annotations

import math
from typing import Any


def normalise_coefs(coefs: list[float]) -> list[float]:
    """Scale coefs to [-1, 1] by dividing by max |coef|. Zero stays zero."""
    if not coefs:
        return []
    m = max(abs(c) for c in coefs)
    if m == 0 or math.isnan(m):
        return [0.0] * len(coefs)
    return [c / m for c in coefs]


def coef_to_colour(coef_norm: float) -> str:
    """Map normalised coef in [-1, 1] to a hex background colour.

    +1 → strong green (chunk pushed answer toward predicted letter)
    -1 → strong red   (chunk pushed away)
     0 → light grey   (neutral)
    """
    if coef_norm != coef_norm:  # NaN
        return "rgba(203,213,225,0.25)"
    if coef_norm > 0:
        alpha = min(0.65, 0.15 + 0.5 * coef_norm)
        return f"rgba(16,185,129,{alpha:.2f})"
    if coef_norm < 0:
        alpha = min(0.65, 0.15 + 0.5 * abs(coef_norm))
        return f"rgba(239,68,68,{alpha:.2f})"
    return "rgba(203,213,225,0.20)"


def lime_chunk_index(lime_row: dict[str, Any]) -> dict[str, dict[str, float]]:
    """chunk_id → {'correctness_coef', 'sameletter_coef', 'rank'}.

    Empty dict if ``lime_row`` is None or has no ``passages`` field.
    """
    if not lime_row or "passages" not in lime_row:
        return {}
    out: dict[str, dict[str, float]] = {}
    for p in lime_row["passages"]:
        out[p["chunk_id"]] = {
            "correctness_coef": p.get("correctness_coef", 0.0),
            "sameletter_coef": p.get("sameletter_coef", 0.0),
            "rank": p.get("rank", -1),
        }
    return out
