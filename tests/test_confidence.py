"""Unit tests for `src/confidence/signals.py` + `src/confidence/rejection.py`.

    .venv/bin/python tests/test_confidence.py
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from src.confidence.rejection import (
    DEFAULT_THRESHOLDS,
    baseline_no_rejection,
    sweep_thresholds,
)
from src.confidence.signals import (
    DEFAULT_SIGNAL_COLUMNS,
    SignalArtefacts,
    _min_max,
    build_signal_table,
    combine_signals,
    load_predictions,
    load_ragas_features,
    load_retrieval_features,
)


# ------------------- signals.py --------------------


def test_min_max_basic() -> None:
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    out = _min_max(s)
    assert out.iloc[0] == 0.0
    assert out.iloc[-1] == 1.0
    assert math.isclose(out.iloc[1], 1.0 / 3.0)


def test_min_max_constant_returns_neutral() -> None:
    s = pd.Series([0.5, 0.5, 0.5, 0.5])
    out = _min_max(s)
    assert (out == 0.5).all()


def test_min_max_with_nan() -> None:
    s = pd.Series([1.0, 2.0, np.nan, 4.0])
    out = _min_max(s)
    assert out.iloc[0] == 0.0
    assert math.isclose(out.iloc[-1], 1.0)
    assert math.isnan(out.iloc[2])


def _write_temp_artefacts(td: Path) -> SignalArtefacts:
    """Build a minimal predictions/retrieval/ragas trio for testing."""
    preds_p = td / "predictions.jsonl"
    ret_p = td / "retrieval.jsonl"
    rag_p = td / "ragas_scores.csv"
    with preds_p.open("w") as f:
        for qid, gold, pred, is_corr in [
            ("q1", "A", "A", True),
            ("q2", "B", "C", False),
            ("q3", "C", "C", True),
        ]:
            f.write(json.dumps({
                "question_id": qid, "gold_letter": gold, "pred_letter": pred,
                "is_correct": is_corr, "raw_response": pred, "latency_s": 0.0, "was_cached": False,
            }) + "\n")
    with ret_p.open("w") as f:
        f.write(json.dumps({"question_id": "q1", "retrieved_chunk_ids": ["a", "b"], "retrieved_chunk_scores": [0.9, 0.7]}) + "\n")
        f.write(json.dumps({"question_id": "q2", "retrieved_chunk_ids": ["c"], "retrieved_chunk_scores": [0.5]}) + "\n")
        f.write(json.dumps({"question_id": "q3", "retrieved_chunk_ids": ["d", "e", "f"], "retrieved_chunk_scores": [0.8, 0.75, 0.6]}) + "\n")
    pd.DataFrame([
        {"question_id": "q1", "faithfulness": 0.8, "context_precision": 0.6, "context_recall": 0.9, "answer_relevancy": 0.7},
        {"question_id": "q2", "faithfulness": 0.1, "context_precision": 0.2, "context_recall": 0.3, "answer_relevancy": 0.5},
        {"question_id": "q3", "faithfulness": 0.5, "context_precision": 0.4, "context_recall": 0.6, "answer_relevancy": 0.6},
    ]).to_csv(rag_p, index=False)
    return SignalArtefacts(preds_p, ret_p, rag_p)


def test_load_helpers_round_trip() -> None:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        art = _write_temp_artefacts(td)
        preds = load_predictions(art.predictions_path)
        retr = load_retrieval_features(art.retrieval_path)
        rag = load_ragas_features(art.ragas_scores_path)
        assert len(preds) == 3
        assert {"is_correct", "gold_letter", "pred_letter"}.issubset(preds.columns)
        assert set(retr.columns) == {"question_id", "retrieval_score_mean", "retrieval_score_max", "retrieval_score_var", "n_chunks"}
        assert retr.set_index("question_id").loc["q1", "n_chunks"] == 2
        assert math.isclose(retr.set_index("question_id").loc["q1", "retrieval_score_max"], 0.9)
        assert {"question_id", "faithfulness", "context_precision", "context_recall", "answer_relevancy"} <= set(rag.columns)


def test_build_signal_table_normalises_and_preserves_raw() -> None:
    with tempfile.TemporaryDirectory() as td:
        art = _write_temp_artefacts(Path(td))
        df = build_signal_table(art, normalise=True)
        # Normalised cols in [0, 1]
        for c in DEFAULT_SIGNAL_COLUMNS:
            vals = df[c].dropna()
            if len(vals) > 1:
                assert vals.min() >= -1e-9 and vals.max() <= 1.0 + 1e-9
        # Raw cols preserved
        for c in DEFAULT_SIGNAL_COLUMNS:
            assert f"{c}_raw" in df.columns
        # Sanity: q2 has lowest faithfulness raw → normalised 0
        f_norm = df.set_index("question_id")["faithfulness"]
        assert math.isclose(f_norm.loc["q2"], 0.0)
        assert math.isclose(f_norm.loc["q1"], 1.0)


def test_combine_signals_equal_weights() -> None:
    df = pd.DataFrame({
        "faithfulness": [1.0, 0.0, 0.5],
        "context_precision": [1.0, 0.0, 0.5],
    })
    out = combine_signals(df, signal_columns=("faithfulness", "context_precision"))
    assert math.isclose(out.iloc[0], 1.0)
    assert math.isclose(out.iloc[1], 0.0)
    assert math.isclose(out.iloc[2], 0.5)


def test_combine_signals_weighted() -> None:
    """Faithfulness has weight 3x more than CP."""
    df = pd.DataFrame({
        "faithfulness": [1.0, 0.0],
        "context_precision": [0.0, 1.0],
    })
    out = combine_signals(
        df,
        signal_columns=("faithfulness", "context_precision"),
        weights={"faithfulness": 3.0, "context_precision": 1.0},
    )
    # Row 0: 1*3/4 + 0*1/4 = 0.75
    # Row 1: 0*3/4 + 1*1/4 = 0.25
    assert math.isclose(out.iloc[0], 0.75)
    assert math.isclose(out.iloc[1], 0.25)


def test_combine_signals_handles_nan_per_row() -> None:
    """NaN in one signal column should not pull the row's score down."""
    df = pd.DataFrame({
        "faithfulness": [1.0, 1.0],
        "context_precision": [1.0, np.nan],
    })
    out = combine_signals(df, signal_columns=("faithfulness", "context_precision"))
    assert math.isclose(out.iloc[0], 1.0)
    # Row 1 uses only the non-NaN signal: 1.0
    assert math.isclose(out.iloc[1], 1.0)


# ------------------- rejection.py --------------------


def test_sweep_at_zero_threshold_accepts_all_with_signal() -> None:
    """At τ=0, all rows with valid confidence are accepted."""
    conf = pd.Series([0.1, 0.5, 0.9, np.nan])
    y = pd.Series([True, True, False, True])
    out = sweep_thresholds(conf, y, thresholds=[0.0])
    row = out.iloc[0]
    # NaN row is rejected → n_accepted = 3
    assert row["n_accepted"] == 3
    assert math.isclose(row["accuracy_on_accepted"], 2.0 / 3.0)
    assert math.isclose(row["rejection_rate"], 0.25)


def test_sweep_at_high_threshold_rejects_all() -> None:
    conf = pd.Series([0.1, 0.5, 0.9])
    y = pd.Series([True, False, True])
    out = sweep_thresholds(conf, y, thresholds=[1.5])
    row = out.iloc[0]
    assert row["n_accepted"] == 0
    assert math.isnan(row["accuracy_on_accepted"])
    assert row["rejection_rate"] == 1.0


def test_sweep_accuracy_uplift_positive_when_rejection_helps() -> None:
    """Confidence aligns with correctness — accuracy uplift should be positive."""
    # 4 right + 4 wrong; high conf == correct
    conf = pd.Series([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])
    y = pd.Series([True, True, True, True, False, False, False, False])
    out = sweep_thresholds(conf, y, thresholds=[0.5])
    row = out.iloc[0]
    # τ=0.5 should accept exactly the 4 correct → accuracy 1.0
    assert math.isclose(row["accuracy_on_accepted"], 1.0)
    # baseline 4/8 = 0.5 → uplift 0.5
    assert math.isclose(row["accuracy_uplift"], 0.5)
    assert math.isclose(row["recall_of_correct"], 1.0)
    assert math.isclose(row["recall_of_wrong_rejected"], 1.0)


def test_sweep_with_default_thresholds() -> None:
    conf = pd.Series(np.linspace(0.0, 1.0, 100))
    y = pd.Series([i % 2 == 0 for i in range(100)])  # alternating
    out = sweep_thresholds(conf, y, thresholds=DEFAULT_THRESHOLDS)
    assert len(out) == len(DEFAULT_THRESHOLDS)
    assert (out["threshold"] == list(DEFAULT_THRESHOLDS)).all()
    # rejection rate monotonically non-decreasing in τ
    rates = out["rejection_rate"].to_list()
    assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))


def test_baseline_no_rejection_matches_full_accuracy() -> None:
    y = pd.Series([True, False, True, True])
    row = baseline_no_rejection(y)
    assert row["n_accepted"] == 4
    assert row["n_rejected"] == 0
    assert math.isclose(row["accuracy_on_accepted"], 0.75)


def test_sweep_length_mismatch_raises() -> None:
    conf = pd.Series([0.5, 0.5])
    y = pd.Series([True, True, False])
    try:
        sweep_thresholds(conf, y)
    except ValueError:
        return
    raise AssertionError("expected ValueError on length mismatch")


if __name__ == "__main__":
    tests = [
        test_min_max_basic,
        test_min_max_constant_returns_neutral,
        test_min_max_with_nan,
        test_load_helpers_round_trip,
        test_build_signal_table_normalises_and_preserves_raw,
        test_combine_signals_equal_weights,
        test_combine_signals_weighted,
        test_combine_signals_handles_nan_per_row,
        test_sweep_at_zero_threshold_accepts_all_with_signal,
        test_sweep_at_high_threshold_rejects_all,
        test_sweep_accuracy_uplift_positive_when_rejection_helps,
        test_sweep_with_default_thresholds,
        test_baseline_no_rejection_matches_full_accuracy,
        test_sweep_length_mismatch_raises,
    ]
    for t in tests:
        t()
        print(f"  ✓ {t.__name__}")
    print(f"\nAll {len(tests)} confidence tests passed.")
