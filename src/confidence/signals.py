"""Per-question confidence-signal extraction (EXP_08).

Reads three existing artefacts and assembles a wide DataFrame of normalised
per-question signals for the rejection threshold sweep in EXP_09.

## Inputs (all already on disk; no LLM calls)

For a given architecture (e.g. `exp_05_multi_hop_rag`) + surface (e.g.
`golden_234`):

1. **`predictions.jsonl`** — `question_id`, `gold_letter`, `pred_letter`,
   `is_correct`. Provides the ground-truth label the rejection layer is
   trying to predict.
2. **`retrieval.jsonl`** — `question_id`, `retrieved_chunk_ids`,
   `retrieved_chunk_scores`. Provides four retrieval-quality features:
   `retrieval_score_mean`, `retrieval_score_max`, `retrieval_score_var`,
   `n_chunks`.
3. **`ragas_scores.csv`** — `question_id`, `faithfulness`,
   `context_precision`, `context_recall`, `answer_relevancy`. Provides four
   RAGAS grounding features.

We *exclude* `answer_correctness` from the confidence vector because it's
the RAGAS judge's overall correctness score — including it would be
quasi-circular (LLM judge predicting gold correctness, then we measure
the rejection layer against gold).

## Normalisation

Each signal is min-max scaled to [0, 1] using the per-architecture
distribution from the **same surface**. This avoids cross-architecture
leakage (Multi-Hop's Faithfulness scale differs from Naive's) while
keeping the threshold sweep in a uniform [0, 1] space.

NaN handling: rows where any signal is NaN keep the NaN. Downstream
rejection logic can choose to (a) drop NaN rows or (b) impute the
median per signal. Default behaviour in `rejection.py` is to drop.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Signal columns in the output dataframe (in canonical order)
RETRIEVAL_SIGNALS = (
    "retrieval_score_mean",
    "retrieval_score_max",
    "retrieval_score_var",
    "n_chunks",
)
RAGAS_SIGNALS = (
    "faithfulness",
    "context_precision",
    "context_recall",
    "answer_relevancy",
)
DEFAULT_SIGNAL_COLUMNS = RETRIEVAL_SIGNALS + RAGAS_SIGNALS


@dataclass
class SignalArtefacts:
    """Paths to the three input files for a (architecture, surface) pair."""

    predictions_path: Path
    retrieval_path: Path
    ragas_scores_path: Path

    @classmethod
    def for_run(cls, results_root: Path, run_dir: str) -> SignalArtefacts:
        base = Path(results_root) / run_dir
        return cls(
            predictions_path=base / "predictions.jsonl",
            retrieval_path=base / "retrieval.jsonl",
            ragas_scores_path=base / "ragas_scores.csv",
        )


def load_retrieval_features(retrieval_path: Path) -> pd.DataFrame:
    """Build per-question retrieval-quality features from `retrieval.jsonl`.

    Returns a dataframe with columns `question_id`, `retrieval_score_mean`,
    `retrieval_score_max`, `retrieval_score_var`, `n_chunks`.
    """
    rows = []
    for line in Path(retrieval_path).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        scores = r.get("retrieved_chunk_scores") or []
        if scores:
            arr = np.array(scores, dtype=float)
            mean = float(arr.mean())
            mx = float(arr.max())
            var = float(arr.var()) if len(arr) > 1 else 0.0
        else:
            mean = mx = var = 0.0
        rows.append({
            "question_id": r["question_id"],
            "retrieval_score_mean": mean,
            "retrieval_score_max": mx,
            "retrieval_score_var": var,
            "n_chunks": int(len(scores)),
        })
    return pd.DataFrame(rows)


def load_ragas_features(ragas_scores_path: Path) -> pd.DataFrame:
    """Load RAGAS per-question features (the four non-circular signals)."""
    df = pd.read_csv(ragas_scores_path)
    cols_keep = ["question_id"] + list(RAGAS_SIGNALS)
    missing = [c for c in cols_keep if c not in df.columns]
    if missing:
        raise KeyError(f"ragas_scores.csv missing columns: {missing}")
    return df[cols_keep].copy()


def load_predictions(predictions_path: Path) -> pd.DataFrame:
    """Load `question_id`, `is_correct` from predictions.jsonl."""
    rows = []
    for line in Path(predictions_path).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        rows.append({
            "question_id": r["question_id"],
            "gold_letter": r.get("gold_letter"),
            "pred_letter": r.get("pred_letter"),
            "is_correct": bool(r.get("is_correct", False)),
        })
    return pd.DataFrame(rows)


def build_signal_table(
    artefacts: SignalArtefacts, *, normalise: bool = True
) -> pd.DataFrame:
    """Join predictions + retrieval + RAGAS into one per-question dataframe.

    If `normalise=True` (default), each signal column is min-max scaled
    to [0, 1] using the column's own min/max on the joined data. Rows
    with NaN in any signal column are preserved as NaN; downstream
    rejection logic decides how to handle them.

    Output columns:
        question_id, gold_letter, pred_letter, is_correct,
        retrieval_score_mean, retrieval_score_max, retrieval_score_var, n_chunks,
        faithfulness, context_precision, context_recall, answer_relevancy
        (plus the un-normalised `_raw` copies if `normalise=True`)
    """
    preds = load_predictions(artefacts.predictions_path)
    retr = load_retrieval_features(artefacts.retrieval_path)
    rag = load_ragas_features(artefacts.ragas_scores_path)

    df = preds.merge(retr, on="question_id", how="left").merge(
        rag, on="question_id", how="left"
    )

    if normalise:
        for col in DEFAULT_SIGNAL_COLUMNS:
            df[f"{col}_raw"] = df[col]
            df[col] = _min_max(df[col])

    return df


def _min_max(series: pd.Series) -> pd.Series:
    """Min-max scale to [0, 1]. Constant series → all 0.5 (neutral signal)."""
    s = pd.to_numeric(series, errors="coerce")
    s_min = s.min(skipna=True)
    s_max = s.max(skipna=True)
    if pd.isna(s_min) or pd.isna(s_max) or s_max - s_min < 1e-12:
        return pd.Series(np.where(s.notna(), 0.5, np.nan), index=s.index)
    return (s - s_min) / (s_max - s_min)


def combine_signals(
    df: pd.DataFrame,
    *,
    signal_columns: tuple[str, ...] = DEFAULT_SIGNAL_COLUMNS,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """Build a single confidence score in [0, 1] from the per-question vector.

    Default: equal-weighted mean of all signal columns (skipping NaN per row).

    A weighted mean is supported via `weights` (dict signal_name → weight);
    weights are renormalised to sum to 1 across the non-NaN signals for each
    row, so missing signals don't bias the score.

    Rows where every signal is NaN return NaN.
    """
    cols = list(signal_columns)
    sub = df[cols]
    if weights is None:
        w = np.ones(len(cols), dtype=float)
    else:
        w = np.array([weights.get(c, 0.0) for c in cols], dtype=float)
        if (w < 0).any():
            raise ValueError("weights must be non-negative")
        if w.sum() == 0:
            raise ValueError("at least one weight must be positive")

    # Per-row: take values + weights for the non-NaN columns
    out = np.full(len(sub), np.nan, dtype=float)
    for i, (_, row) in enumerate(sub.iterrows()):
        vals = row.to_numpy(dtype=float)
        mask = ~np.isnan(vals)
        if not mask.any():
            continue
        w_row = w[mask]
        v_row = vals[mask]
        if w_row.sum() == 0:
            continue
        out[i] = float(np.dot(v_row, w_row) / w_row.sum())
    return pd.Series(out, index=df.index, name="confidence")
