"""Structure tests for `src/eval/ragas_eval.py` — **no LLM calls**.

Validates the dataset-construction half of RAGAS scoring (which is where most
wiring bugs live). The actual judge call is exercised by the gated smoke cell
in `notebooks/04a_exp01_ragas.ipynb`, not here, because it costs real Anthropic
credit. These tests run in <1 s, no API keys required.

Run:

    .venv/bin/python tests/test_ragas_eval.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval.ragas_eval import (
    ALL_METRIC_NAMES,
    CONTEXT_DEPENDENT,
    CONTEXT_INDEPENDENT,
    _golden_qid_from_pred,
    _load_golden_by_qid,
    _load_predictions,
    _load_retrieval,
    _merge_partial_scores,
    _nan_question_ids,
    applicable_metrics,
    build_ragas_rows,
)


def test_id_parser() -> None:
    assert _golden_qid_from_pred("golden_000") == 0
    assert _golden_qid_from_pred("golden_233") == 233
    assert _golden_qid_from_pred("golden_001") == 1
    assert _golden_qid_from_pred("medqa_00000") is None
    assert _golden_qid_from_pred("smoke_3") is None
    assert _golden_qid_from_pred("garbage") is None


def test_metric_sets_partition() -> None:
    """The two subsets must partition `ALL_METRIC_NAMES` cleanly."""
    assert CONTEXT_DEPENDENT.isdisjoint(CONTEXT_INDEPENDENT)
    assert set(ALL_METRIC_NAMES) == CONTEXT_DEPENDENT | CONTEXT_INDEPENDENT


def test_build_rows_against_real_exp01() -> None:
    """Joins all 234 EXP_01 predictions with the golden file. No-RAG ⇒
    `_has_context` is False everywhere."""
    pred_dir = _REPO_ROOT / "results" / "exp_01_base_llm__golden_234"
    if not pred_dir.exists():
        return  # EXP_01 hasn't been run yet — skip silently
    preds = _load_predictions(pred_dir / "predictions.jsonl")
    rets = _load_retrieval(pred_dir / "retrieval.jsonl")
    golden_by_qid = _load_golden_by_qid(_REPO_ROOT / "data" / "processed" / "golden_ragas_300.jsonl")

    rows = build_ragas_rows(preds, rets, golden_by_qid)
    assert len(rows) == 234, f"expected 234 joined rows, got {len(rows)}"
    assert (~rows["_has_context"]).all(), "EXP_01 must have no retrieved context"

    # Every required RAGAS field is non-null
    for col in ("user_input", "response", "retrieved_contexts", "reference"):
        assert rows[col].notna().all(), f"{col} has nulls"

    # Prose-conversion: when prediction is correct, response should equal
    # reference (prose form of the right option). When wrong, they differ.
    correct = rows[rows["_is_correct"]]
    if len(correct):
        match_rate = (correct["response"] == correct["reference"]).mean()
        assert match_rate > 0.95, f"correct rows should mostly match reference; got {match_rate:.2%}"


def test_applicable_metrics_no_context() -> None:
    """Synthetic case: all rows have empty context ⇒ context-independent only."""
    import pandas as pd

    rows = pd.DataFrame({"_has_context": [False, False, False]})
    assert set(applicable_metrics(rows)) == CONTEXT_INDEPENDENT


def test_applicable_metrics_with_context() -> None:
    """Even one context-bearing row flips us to the full metric set."""
    import pandas as pd

    rows = pd.DataFrame({"_has_context": [False, True, False]})
    assert set(applicable_metrics(rows)) == set(ALL_METRIC_NAMES)


def test_build_rows_with_synthetic_chunks() -> None:
    """Mock a 2-row RAG-style join to exercise the chunk-text lookup."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # Synthetic predictions: 1 golden_000, 1 golden_001
        (tmp / "predictions.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "question_id": "golden_000",
                            "gold_letter": "B",
                            "pred_letter": "B",
                            "raw_response": "B",
                            "latency_s": 0.1,
                            "was_cached": False,
                            "is_correct": True,
                        }
                    ),
                    json.dumps(
                        {
                            "question_id": "golden_001",
                            "gold_letter": "C",
                            "pred_letter": "A",
                            "raw_response": "A",
                            "latency_s": 0.1,
                            "was_cached": False,
                            "is_correct": False,
                        }
                    ),
                ]
            )
        )
        (tmp / "retrieval.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "question_id": "golden_000",
                            "retrieved_chunk_ids": ["c1", "c2"],
                            "retrieved_chunk_scores": [0.9, 0.8],
                        }
                    ),
                    json.dumps(
                        {
                            "question_id": "golden_001",
                            "retrieved_chunk_ids": [],
                            "retrieved_chunk_scores": [],
                        }
                    ),
                ]
            )
        )
        # Synthetic golden subset (only 2 rows of the right shape)
        golden_path = tmp / "golden.jsonl"
        golden_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "question_id": 0,
                            "question": "Q0?",
                            "options": {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
                            "gold_answer_letter": "B",
                            "gold_answer_text": "beta",
                            "final_status": "accepted",
                        }
                    ),
                    json.dumps(
                        {
                            "question_id": 1,
                            "question": "Q1?",
                            "options": {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
                            "gold_answer_letter": "C",
                            "gold_answer_text": "gamma",
                            "final_status": "accepted",
                        }
                    ),
                ]
            )
        )
        preds = _load_predictions(tmp / "predictions.jsonl")
        rets = _load_retrieval(tmp / "retrieval.jsonl")
        golden_by_qid = _load_golden_by_qid(golden_path)

        rows = build_ragas_rows(
            preds,
            rets,
            golden_by_qid,
            chunks_by_id={"c1": "first chunk text", "c2": "second chunk text"},
        )
        assert len(rows) == 2
        # Row 0: correct + has retrieved chunks
        r0 = rows.iloc[0]
        assert r0["response"] == "beta"
        assert r0["reference"] == "beta"
        assert r0["retrieved_contexts"] == ["first chunk text", "second chunk text"]
        assert bool(r0["_has_context"]) is True
        # Row 1: wrong + no retrieved chunks (placeholder context)
        r1 = rows.iloc[1]
        assert r1["response"] == "alpha"
        assert r1["reference"] == "gamma"
        assert r1["retrieved_contexts"] == [""]
        assert bool(r1["_has_context"]) is False

        # Mixed surface ⇒ all 5 metrics applicable
        assert set(applicable_metrics(rows)) == set(ALL_METRIC_NAMES)


def test_nan_question_ids_finds_all_nan_rows() -> None:
    """`_nan_question_ids` returns the set of question_ids whose row has NaN
    in any active-metric column."""
    import pandas as pd

    existing = pd.DataFrame(
        {
            "question_id": ["golden_000", "golden_001", "golden_002", "golden_003"],
            "answer_relevancy": [0.6, float("nan"), 0.8, 0.5],
            "answer_correctness": [0.9, 0.95, float("nan"), float("nan")],
        }
    )
    metric_names = ["answer_relevancy", "answer_correctness"]
    nan_qids = _nan_question_ids(existing, metric_names)
    # Row 0: both clean → NOT included
    # Row 1: AR is NaN → included
    # Row 2: AC is NaN → included
    # Row 3: AC is NaN → included
    assert nan_qids == {"golden_001", "golden_002", "golden_003"}, nan_qids


def test_nan_question_ids_handles_missing_metric_column() -> None:
    """If `existing` is missing a metric column entirely, it's not used in
    the NaN check (treated as 'we never asked for that one')."""
    import pandas as pd

    existing = pd.DataFrame(
        {
            "question_id": ["golden_000", "golden_001"],
            "answer_relevancy": [0.6, float("nan")],
            # answer_correctness column absent
        }
    )
    nan_qids = _nan_question_ids(existing, ["answer_relevancy", "answer_correctness"])
    assert nan_qids == {"golden_001"}


def test_merge_partial_replaces_nans_only() -> None:
    """`_merge_partial_scores` replaces NaN cells with new values; preserves
    already-good cells; ignores partial rows that don't match any existing
    question_id."""
    import pandas as pd

    existing = pd.DataFrame(
        {
            "question_id": ["golden_000", "golden_001", "golden_002"],
            "answer_relevancy": [0.6, float("nan"), 0.8],
            "answer_correctness": [0.9, float("nan"), float("nan")],
        }
    )
    partial = pd.DataFrame(
        {
            "question_id": ["golden_001", "golden_002", "golden_999"],
            "answer_relevancy": [0.7, 0.85, 0.5],          # 002 was already 0.8 → preserved
            "answer_correctness": [0.88, 0.92, 0.4],       # 999 doesn't exist → ignored
        }
    )
    merged = _merge_partial_scores(existing, partial, ["answer_relevancy", "answer_correctness"])

    # Row 0 untouched
    assert merged.loc[0, "answer_relevancy"] == 0.6
    assert merged.loc[0, "answer_correctness"] == 0.9
    # Row 1: both filled from partial
    assert merged.loc[1, "answer_relevancy"] == 0.7
    assert merged.loc[1, "answer_correctness"] == 0.88
    # Row 2: AR was already 0.8 (NOT overwritten) — AC was NaN, now 0.92
    assert merged.loc[2, "answer_relevancy"] == 0.8
    assert merged.loc[2, "answer_correctness"] == 0.92
    # Phantom golden_999 didn't add a new row
    assert len(merged) == 3


def test_merge_partial_preserves_bookkeeping_columns() -> None:
    """Bookkeeping columns (_pred_letter, _is_correct, etc.) survive the merge."""
    import pandas as pd

    existing = pd.DataFrame(
        {
            "question_id": ["golden_000", "golden_001"],
            "answer_relevancy": [float("nan"), 0.7],
            "_pred_letter": ["B", "C"],
            "_is_correct": [True, False],
        }
    )
    partial = pd.DataFrame(
        {
            "question_id": ["golden_000"],
            "answer_relevancy": [0.55],
        }
    )
    merged = _merge_partial_scores(existing, partial, ["answer_relevancy"])
    assert merged.loc[0, "_pred_letter"] == "B"
    assert bool(merged.loc[0, "_is_correct"]) is True
    assert merged.loc[0, "answer_relevancy"] == 0.55


if __name__ == "__main__":
    test_id_parser()
    print("✓ test_id_parser")
    test_metric_sets_partition()
    print("✓ test_metric_sets_partition")
    test_build_rows_against_real_exp01()
    print("✓ test_build_rows_against_real_exp01")
    test_applicable_metrics_no_context()
    print("✓ test_applicable_metrics_no_context")
    test_applicable_metrics_with_context()
    print("✓ test_applicable_metrics_with_context")
    test_build_rows_with_synthetic_chunks()
    print("✓ test_build_rows_with_synthetic_chunks")
    test_nan_question_ids_finds_all_nan_rows()
    print("✓ test_nan_question_ids_finds_all_nan_rows")
    test_nan_question_ids_handles_missing_metric_column()
    print("✓ test_nan_question_ids_handles_missing_metric_column")
    test_merge_partial_replaces_nans_only()
    print("✓ test_merge_partial_replaces_nans_only")
    test_merge_partial_preserves_bookkeeping_columns()
    print("✓ test_merge_partial_preserves_bookkeeping_columns")
    print("\nAll 10 RAGAS structure tests passed.")
