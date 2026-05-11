"""Unit tests for `src/xai/agreement.py`.

    .venv/bin/python tests/test_agreement.py
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

from src.xai.agreement import (
    _top_n_chunk_ids,
    _topn_overlap,
    agreement_from_records,
    run_agreement_batch,
)


def _lime_rec(qid: str, coefs: list[float]) -> dict:
    """Build a minimal LIME record with given correctness_coef values."""
    n = len(coefs)
    top_idx = max(range(n), key=lambda i: abs(coefs[i]))
    return {
        "question_id": qid,
        "architecture": "arch",
        "method": "subset_lime",
        "n_passages": n,
        "n_samples": 16,
        "passages": [
            {
                "chunk_id": f"c{i}",
                "rank": i,
                "correctness_coef": coefs[i],
                "sameletter_coef": coefs[i],  # for simplicity
            }
            for i in range(n)
        ],
        "top_chunk_by_correctness": f"c{top_idx}",
        "top_chunk_by_sameletter": f"c{top_idx}",
    }


def _shap_rec(qid: str, coefs: list[float]) -> dict:
    n = len(coefs)
    top_idx = max(range(n), key=lambda i: abs(coefs[i]))
    return {
        "question_id": qid,
        "architecture": "arch",
        "method": "kernel_shap",
        "n_passages": n,
        "n_samples": 16,
        "passages": [
            {
                "chunk_id": f"c{i}",
                "rank": i,
                "correctness_shap": coefs[i],
                "sameletter_shap": coefs[i],
            }
            for i in range(n)
        ],
        "top_chunk_by_correctness": f"c{top_idx}",
        "top_chunk_by_sameletter": f"c{top_idx}",
    }


def test_top1_match_when_same_argmax() -> None:
    """LIME and SHAP both pick c2 as top by |coef|."""
    lime = _lime_rec("q1", [0.1, -0.05, 0.7, -0.2, 0.0])
    shap = _shap_rec("q1", [0.2, -0.1, 0.6, -0.3, 0.05])
    r = agreement_from_records(lime, shap)
    assert r is not None
    assert r.correctness_top1 == 1.0
    assert r.sameletter_top1 == 1.0


def test_top1_mismatch() -> None:
    lime = _lime_rec("q1", [0.1, 0.7, 0.2])  # top c1
    shap = _shap_rec("q1", [0.7, 0.1, 0.2])  # top c0
    r = agreement_from_records(lime, shap)
    assert r is not None
    assert r.correctness_top1 == 0.0


def test_top3_overlap() -> None:
    """k=5 passages; top-3 sets share 2 chunks → overlap = 2/3."""
    lime = _lime_rec("q1", [0.9, 0.8, 0.7, 0.1, 0.0])  # top3 = c0, c1, c2
    shap = _shap_rec("q1", [0.9, 0.8, 0.0, 0.7, 0.1])  # top3 = c0, c1, c3
    r = agreement_from_records(lime, shap)
    assert r is not None
    assert math.isclose(r.correctness_top3_overlap, 2.0 / 3.0)


def test_spearman_full_agreement() -> None:
    """When LIME and SHAP have identical ranking, Spearman = 1."""
    lime = _lime_rec("q1", [0.1, 0.5, 0.3, 0.9])
    shap = _shap_rec("q1", [0.15, 0.4, 0.32, 0.95])
    r = agreement_from_records(lime, shap)
    assert r is not None
    assert math.isclose(r.correctness_spearman, 1.0, abs_tol=1e-9)


def test_spearman_anti_agreement() -> None:
    """Reversed rankings give Spearman = -1."""
    lime = _lime_rec("q1", [0.1, 0.3, 0.5, 0.9])  # increasing
    shap = _shap_rec("q1", [0.9, 0.5, 0.3, 0.1])  # decreasing
    r = agreement_from_records(lime, shap)
    assert r is not None
    assert math.isclose(r.correctness_spearman, -1.0, abs_tol=1e-9)


def test_no_top_returns_nan() -> None:
    """When either method has top_chunk = None (no attribution), top1 is NaN."""
    lime = _lime_rec("q1", [0.5, 0.2, 0.1])
    lime["top_chunk_by_correctness"] = None
    shap = _shap_rec("q1", [0.5, 0.2, 0.1])
    r = agreement_from_records(lime, shap)
    assert r is not None
    assert math.isnan(r.correctness_top1)


def test_qid_mismatch_raises() -> None:
    lime = _lime_rec("q1", [0.5, 0.2])
    shap = _shap_rec("q2", [0.5, 0.2])
    try:
        agreement_from_records(lime, shap)
    except ValueError:
        return
    raise AssertionError("expected ValueError on qid mismatch")


def test_top_n_chunk_ids_tie_breaks_by_rank() -> None:
    """When two chunks tie on |coef|, the lower-rank chunk wins the tiebreak."""
    passages = [
        {"chunk_id": "a", "rank": 0, "correctness_coef": 0.5},
        {"chunk_id": "b", "rank": 1, "correctness_coef": -0.5},  # same |coef| as a
        {"chunk_id": "c", "rank": 2, "correctness_coef": 0.1},
    ]
    top = _top_n_chunk_ids(passages, "correctness_coef", 2)
    assert top[0] == "a"  # rank 0 wins the tie
    assert top[1] == "b"


def test_batch_runs_end_to_end() -> None:
    with tempfile.TemporaryDirectory() as td:
        lime_p = Path(td) / "lime.jsonl"
        shap_p = Path(td) / "shap.jsonl"
        agr_p = Path(td) / "agr.jsonl"
        with lime_p.open("w") as f:
            for qid in ("q1", "q2"):
                f.write(json.dumps(_lime_rec(qid, [0.1, 0.5, 0.2])) + "\n")
        with shap_p.open("w") as f:
            for qid in ("q1", "q2"):
                f.write(json.dumps(_shap_rec(qid, [0.1, 0.5, 0.2])) + "\n")
        s = run_agreement_batch(lime_p, shap_p, agr_p, progress=False)
        assert s["n_rows_written"] == 2
        rows = [json.loads(l) for l in agr_p.read_text().splitlines()]
        assert all(r["correctness_top1"] == 1.0 for r in rows)


if __name__ == "__main__":
    tests = [
        test_top1_match_when_same_argmax,
        test_top1_mismatch,
        test_top3_overlap,
        test_spearman_full_agreement,
        test_spearman_anti_agreement,
        test_no_top_returns_nan,
        test_qid_mismatch_raises,
        test_top_n_chunk_ids_tie_breaks_by_rank,
        test_batch_runs_end_to_end,
    ]
    for t in tests:
        t()
        print(f"  ✓ {t.__name__}")
    print(f"\nAll {len(tests)} agreement tests passed.")
