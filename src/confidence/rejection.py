"""Confidence-aware rejection: threshold sweep + accept/reject metrics (EXP_09).

Given a per-question confidence score in [0, 1] and a ground-truth
`is_correct` label, sweep a threshold τ and report:

- `n_accepted`           — questions with confidence ≥ τ
- `n_rejected`           — questions with confidence < τ
- `rejection_rate`       — `n_rejected / n_total`
- `accuracy_on_accepted` — `is_correct` rate among accepted
- `accuracy_uplift`      — `accuracy_on_accepted − baseline_accuracy`
- `recall_of_correct`    — fraction of originally-correct questions accepted
                            (= n_correct_accepted / n_total_correct)
- `recall_of_wrong_rejected` — fraction of originally-wrong questions rejected

Rows with NaN confidence are *always rejected* (the rejection layer has no
signal on them). The thresholds default to {0.3, 0.4, 0.5, 0.6, 0.7} but
any iterable is accepted.

The output table is tiny (one row per τ) and is meant to be pasted into
Table 11 of the Excel workbook + plotted in the notebook.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


@dataclass
class RejectionRow:
    threshold: float
    n_total: int
    n_accepted: int
    n_rejected: int
    rejection_rate: float
    accuracy_on_accepted: float  # NaN if n_accepted == 0
    accuracy_uplift: float  # vs full-population accuracy
    recall_of_correct: float
    recall_of_wrong_rejected: float


def sweep_thresholds(
    confidence: pd.Series,
    is_correct: pd.Series,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
) -> pd.DataFrame:
    """Run the threshold sweep over `(confidence, is_correct)` aligned by index.

    Rows where `confidence` is NaN are treated as REJECTED at every threshold
    (the layer has no signal). This is conservative and matches the thesis
    "fail safe" framing — if confidence can't be computed, the question is
    flagged for review.

    Returns a dataframe with one row per threshold (columns match
    `RejectionRow` fields).
    """
    if len(confidence) != len(is_correct):
        raise ValueError(
            f"length mismatch: confidence={len(confidence)} is_correct={len(is_correct)}"
        )
    c = pd.to_numeric(confidence, errors="coerce").to_numpy()
    y = is_correct.astype(bool).to_numpy()
    n_total = int(len(y))
    baseline_accuracy = float(y.mean()) if n_total else float("nan")
    n_correct_total = int(y.sum())
    n_wrong_total = n_total - n_correct_total

    rows: list[RejectionRow] = []
    for tau in thresholds:
        accept_mask = (~np.isnan(c)) & (c >= tau)
        n_accepted = int(accept_mask.sum())
        n_rejected = n_total - n_accepted
        if n_accepted > 0:
            acc_on_accepted = float(y[accept_mask].mean())
        else:
            acc_on_accepted = float("nan")
        uplift = (
            acc_on_accepted - baseline_accuracy
            if not np.isnan(acc_on_accepted)
            else float("nan")
        )
        n_correct_accepted = int((y & accept_mask).sum())
        n_wrong_rejected = int((~y & ~accept_mask).sum())
        rows.append(
            RejectionRow(
                threshold=float(tau),
                n_total=n_total,
                n_accepted=n_accepted,
                n_rejected=n_rejected,
                rejection_rate=n_rejected / n_total if n_total else float("nan"),
                accuracy_on_accepted=acc_on_accepted,
                accuracy_uplift=uplift,
                recall_of_correct=(
                    n_correct_accepted / n_correct_total
                    if n_correct_total
                    else float("nan")
                ),
                recall_of_wrong_rejected=(
                    n_wrong_rejected / n_wrong_total
                    if n_wrong_total
                    else float("nan")
                ),
            )
        )
    return pd.DataFrame([r.__dict__ for r in rows])


def baseline_no_rejection(is_correct: pd.Series) -> dict:
    """Reference row at τ = −∞ (accept everything). Mirrors the sweep schema."""
    y = is_correct.astype(bool).to_numpy()
    n = len(y)
    acc = float(y.mean()) if n else float("nan")
    return {
        "threshold": float("-inf"),
        "n_total": n,
        "n_accepted": n,
        "n_rejected": 0,
        "rejection_rate": 0.0,
        "accuracy_on_accepted": acc,
        "accuracy_uplift": 0.0,
        "recall_of_correct": 1.0,
        "recall_of_wrong_rejected": 0.0,
    }
