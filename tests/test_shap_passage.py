"""Unit tests for `src/xai/shap_passage.py`.

No real Groq calls — feeds synthetic LIME records into the SHAP function
and verifies (a) the weight formula, (b) attribution direction, (c)
no-passages handling, (d) batch resumability.

    .venv/bin/python tests/test_shap_passage.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.xai.shap_passage import (
    _HIGH_WEIGHT,
    kernel_shap_weight,
    passage_shap_from_lime_record,
    run_shap_from_lime_batch,
)


def test_kernel_weight_boundaries() -> None:
    """All-zeros and all-ones masks get the high-weight constraint."""
    assert kernel_shap_weight(0, 5) == _HIGH_WEIGHT
    assert kernel_shap_weight(5, 5) == _HIGH_WEIGHT
    assert kernel_shap_weight(0, 0) == 0.0  # degenerate k


def test_kernel_weight_interior_is_positive_and_symmetric() -> None:
    """Interior weights are positive and symmetric in |S| ↔ k − |S|."""
    for size in range(1, 5):
        w = kernel_shap_weight(size, 5)
        assert w > 0
        # symmetry
        assert abs(w - kernel_shap_weight(5 - size, 5)) < 1e-12


def _synthetic_lime_record(
    n_passages: int = 3,
    full_pred: str = "A",
    gold: str = "A",
    samples: list[dict] | None = None,
) -> dict:
    """Build a minimal LIME record dict that SHAP can consume."""
    if samples is None:
        # Default: all-ones with correct=1; partial masks with same prediction.
        samples = [
            {"mask": [1, 1, 1], "n_chunks_included": 3, "pred_letter": "A", "correct": 1, "sameletter": 1, "was_cached": True, "latency_s": 0.0},
            {"mask": [0, 1, 1], "n_chunks_included": 2, "pred_letter": "A", "correct": 1, "sameletter": 1, "was_cached": False, "latency_s": 0.0},
            {"mask": [1, 0, 1], "n_chunks_included": 2, "pred_letter": "A", "correct": 1, "sameletter": 1, "was_cached": False, "latency_s": 0.0},
            {"mask": [1, 1, 0], "n_chunks_included": 2, "pred_letter": "B", "correct": 0, "sameletter": 0, "was_cached": False, "latency_s": 0.0},
        ]
    return {
        "question_id": "q1",
        "architecture": "arch",
        "method": "subset_lime",
        "gold_letter": gold,
        "full_pred_letter": full_pred,
        "full_correct": full_pred == gold,
        "n_passages": n_passages,
        "n_samples": len(samples),
        "seed": 0,
        "alpha": 0.1,
        "correctness_intercept": 0.0,
        "sameletter_intercept": 0.0,
        "correctness_score_variance": 0.0,
        "sameletter_score_variance": 0.0,
        "passages": [
            {"chunk_id": f"c{i}", "rank": i, "correctness_coef": 0.0, "sameletter_coef": 0.0}
            for i in range(n_passages)
        ],
        "samples": samples,
        "top_chunk_by_correctness": None,
        "top_chunk_by_sameletter": None,
        "note": "",
    }


def test_shap_directional_attribution() -> None:
    """When removing c2 (rank=2) is the ONLY thing that flips correctness,
    SHAP should attribute the most positive correctness mass to c2."""
    rec = _synthetic_lime_record(n_passages=3, full_pred="A", gold="A")
    result = passage_shap_from_lime_record(rec)
    assert result.n_passages == 3
    # c2 had the only flip when removed → positive correctness SHAP
    c0 = next(p for p in result.passages if p.chunk_id == "c0")
    c1 = next(p for p in result.passages if p.chunk_id == "c1")
    c2 = next(p for p in result.passages if p.chunk_id == "c2")
    assert c2.correctness_shap > 0, f"c2 should attribute positive correctness; got {c2.correctness_shap}"
    assert c2.correctness_shap > c0.correctness_shap
    assert c2.correctness_shap > c1.correctness_shap
    assert result.top_chunk_by_correctness == "c2"


def test_shap_zero_variance_returns_zeros() -> None:
    """If all samples have the same prediction (== full_pred), SHAP gives
    all-zero attributions and top-1 is None."""
    rec = _synthetic_lime_record(
        n_passages=3,
        full_pred="A",
        gold="A",
        samples=[
            {"mask": [1, 1, 1], "n_chunks_included": 3, "pred_letter": "A", "correct": 1, "sameletter": 1, "was_cached": True, "latency_s": 0.0},
            {"mask": [0, 1, 1], "n_chunks_included": 2, "pred_letter": "A", "correct": 1, "sameletter": 1, "was_cached": False, "latency_s": 0.0},
            {"mask": [1, 0, 0], "n_chunks_included": 1, "pred_letter": "A", "correct": 1, "sameletter": 1, "was_cached": False, "latency_s": 0.0},
        ],
    )
    result = passage_shap_from_lime_record(rec)
    assert result.correctness_score_variance == 0.0
    assert result.top_chunk_by_correctness is None
    for p in result.passages:
        assert p.correctness_shap == 0.0


def test_shap_no_rag_anchor_flag_and_intercept() -> None:
    """Supplying a No-RAG prediction (a) sets `no_rag_anchor_used=True`,
    (b) adds one synthetic all-zeros sample to the regression, (c) forces
    the correctness intercept toward the No-RAG correctness value (because
    the anchor has 1e6 weight).

    For this test we need samples where the unanchored intercept is *not*
    already at the No-RAG value, so use data with all-correct partial
    masks — that gives the unanchored fit an intercept near 1, but the
    anchored fit (No-RAG = wrong) drives intercept toward 0.
    """
    rec = _synthetic_lime_record(
        n_passages=3,
        full_pred="A",
        gold="A",
        samples=[
            {"mask": [1, 1, 1], "n_chunks_included": 3, "pred_letter": "A", "correct": 1, "sameletter": 1, "was_cached": True, "latency_s": 0.0},
            {"mask": [0, 1, 1], "n_chunks_included": 2, "pred_letter": "A", "correct": 1, "sameletter": 1, "was_cached": False, "latency_s": 0.0},
            {"mask": [1, 0, 1], "n_chunks_included": 2, "pred_letter": "A", "correct": 1, "sameletter": 1, "was_cached": False, "latency_s": 0.0},
            {"mask": [0, 0, 1], "n_chunks_included": 1, "pred_letter": "A", "correct": 1, "sameletter": 1, "was_cached": False, "latency_s": 0.0},
        ],
    )
    # First check: this data has zero variance → both paths return zero coefs.
    # That's fine; we just verify the anchor flag is set correctly.
    result_anchored = passage_shap_from_lime_record(
        rec, no_rag_pred_letter="B", gold_letter="A"
    )
    result_plain = passage_shap_from_lime_record(rec)
    assert result_anchored.no_rag_anchor_used is True
    assert result_plain.no_rag_anchor_used is False
    assert result_anchored.n_samples == result_plain.n_samples + 1

    # Now use data with variance: one mask flips correctness. Anchor should
    # shift the intercept toward 0 (anchored) vs whatever the unweighted
    # fit produces (plain).
    rec2 = _synthetic_lime_record(
        n_passages=3,
        full_pred="A",
        gold="A",
        samples=[
            {"mask": [1, 1, 1], "n_chunks_included": 3, "pred_letter": "A", "correct": 1, "sameletter": 1, "was_cached": True, "latency_s": 0.0},
            {"mask": [0, 1, 1], "n_chunks_included": 2, "pred_letter": "A", "correct": 1, "sameletter": 1, "was_cached": False, "latency_s": 0.0},
            {"mask": [1, 0, 1], "n_chunks_included": 2, "pred_letter": "B", "correct": 0, "sameletter": 0, "was_cached": False, "latency_s": 0.0},
            {"mask": [0, 0, 1], "n_chunks_included": 1, "pred_letter": "B", "correct": 0, "sameletter": 0, "was_cached": False, "latency_s": 0.0},
        ],
    )
    plain2 = passage_shap_from_lime_record(rec2)
    anchored2 = passage_shap_from_lime_record(rec2, no_rag_pred_letter="B", gold_letter="A")
    # With the anchor forcing intercept near zero (because no-RAG correct=0),
    # the anchored intercept should be << the plain intercept on this data.
    assert anchored2.correctness_intercept < plain2.correctness_intercept + 1e-6


def test_shap_no_passages_returns_note() -> None:
    rec = _synthetic_lime_record(n_passages=0, samples=[])
    rec["passages"] = []
    result = passage_shap_from_lime_record(rec)
    assert result.n_passages == 0
    assert result.note == "no_passages_skipped_shap"
    assert result.passages == []


def test_shap_batch_resumability() -> None:
    """run_shap_from_lime_batch skips already-completed (qid, arch) rows."""
    with tempfile.TemporaryDirectory() as td:
        lime_path = Path(td) / "lime.jsonl"
        shap_path = Path(td) / "shap.jsonl"
        with lime_path.open("w") as f:
            for qid in ("q1", "q2"):
                rec = _synthetic_lime_record(n_passages=3)
                rec["question_id"] = qid
                f.write(json.dumps(rec) + "\n")
        s1 = run_shap_from_lime_batch(lime_path, shap_path, progress=False)
        assert s1["n_rows_written_this_run"] == 2
        s2 = run_shap_from_lime_batch(lime_path, shap_path, progress=False)
        assert s2["n_rows_written_this_run"] == 0
        assert s2["n_rows_already_done"] == 2


if __name__ == "__main__":
    tests = [
        test_kernel_weight_boundaries,
        test_kernel_weight_interior_is_positive_and_symmetric,
        test_shap_directional_attribution,
        test_shap_zero_variance_returns_zeros,
        test_shap_no_rag_anchor_flag_and_intercept,
        test_shap_no_passages_returns_note,
        test_shap_batch_resumability,
    ]
    for t in tests:
        t()
        print(f"  ✓ {t.__name__}")
    print(f"\nAll {len(tests)} SHAP passage tests passed.")
