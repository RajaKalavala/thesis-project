"""3-row real-data tests for the EXP_01 module slice.

Per `docs/architecture.md` §11: every `src/` module gets a 3-row test against
real data. Mocks would just confirm the mock; real-data tests catch schema
drift the moment it happens.

Two of the tests hit the live Groq API but every call is disk-cached, so
running this file twice in a row makes 0 net API calls. The first run uses
~6 cache slots (3 questions × 2 invocations: runner full-path + runner
golden-path). To run:

    .venv/bin/python -m pytest tests/test_exp01_modules.py -xvs

`pytest` is not in `requirements.txt` so the file is also runnable as a plain
script (each `test_*` function is invoked at the bottom).
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

# Make `src` importable when this file is invoked directly (`python tests/...`).
# Pytest doesn't need this — it uses conftest discovery — but the script form does.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(_REPO_ROOT / ".env")  # Groq + Anthropic + OpenAI keys

from src.data.loaders import load_chunks, load_golden, load_medqa_4opt
from src.eval.non_llm_metrics import accuracy, exact_match, mrr, ndcg_at_k, recall_at_k
from src.eval.runner import run_experiment
from src.retrieval.base import Chunk, Retriever
from src.retrieval.none import NoRetrieval


def test_loaders_shape_and_types() -> None:
    df = load_medqa_4opt()
    assert df.shape == (12723, 9), df.shape
    assert df["question_id"].iloc[0] == "medqa_00000"
    assert isinstance(df["options"].iloc[0], dict)
    assert set(df["options"].iloc[0].keys()) == {"A", "B", "C", "D"}

    golden = load_golden()
    assert len(golden) == 234, len(golden)
    assert all(r["final_status"] == "accepted" for r in golden)
    assert isinstance(golden[0]["options"], dict)

    chunks = load_chunks()
    assert chunks.shape[0] == 67599, chunks.shape
    assert chunks["chunk_id"].iloc[0].endswith("_chunk_00000")


def test_no_retrieval_returns_empty() -> None:
    r = NoRetrieval()
    assert isinstance(r, Retriever)
    assert r.retrieve("anything", k=5) == []


def test_non_llm_metrics_basic() -> None:
    assert exact_match("a", "A") is True
    assert exact_match(None, "A") is False
    assert accuracy(["A", "B", "C"], ["A", "B", "X"]) == 2 / 3

    retrieved = ["c1", "c2", "c3", "c4", "c5"]
    gold = ["c3", "c9"]
    assert recall_at_k(retrieved, gold, 3) == 0.5
    assert mrr(retrieved, gold) == 1 / 3
    # nDCG: hit at rank 3 ⇒ DCG = 1/log2(4); ideal ⇒ ranks 1, 2 ⇒ 1 + 1/log2(3)
    import math

    expected = (1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert abs(ndcg_at_k(retrieved, gold, 5) - expected) < 1e-9


def test_runner_end_to_end_on_3_medqa_rows() -> None:
    df = load_medqa_4opt().head(3)
    out = Path(tempfile.mkdtemp(prefix="exp01_test_"))
    try:
        summary = run_experiment(
            retriever=NoRetrieval(),
            dataset=df,
            output_dir=out,
            experiment_id="EXP_01_BASE_LLM_test",
            dataset_label="full_12723",
            progress=False,
        )
        # File artefacts exist
        assert (out / "predictions.jsonl").exists()
        assert (out / "retrieval.jsonl").exists()
        assert (out / "summary.json").exists()

        # Summary shape: every locked key present
        for key in (
            "experiment_id",
            "dataset",
            "n_questions",
            "Generator_Model",
            "Acuuracy",
            "Exact_Match",
            "RAGAS_Faithfulness",  # null but present
            "mean_latency_s",
            "timestamp_utc",
        ):
            assert key in summary, f"missing key: {key}"

        assert summary["n_questions"] == 3
        assert 0.0 <= summary["Acuuracy"] <= 1.0
        assert summary["RAGAS_Faithfulness"] is None  # EXP_01 never fills RAGAS

        # Predictions: each row has all required fields
        preds = [json.loads(line) for line in (out / "predictions.jsonl").read_text().splitlines()]
        assert len(preds) == 3
        for p in preds:
            assert set(p.keys()) >= {
                "question_id",
                "gold_letter",
                "pred_letter",
                "is_correct",
                "latency_s",
                "was_cached",
            }
            assert p["question_id"].startswith("medqa_")

        # Retrieval rows exist with empty lists (No-RAG)
        rets = [json.loads(line) for line in (out / "retrieval.jsonl").read_text().splitlines()]
        assert len(rets) == 3
        assert all(r["retrieved_chunk_ids"] == [] for r in rets)

        # Resume idempotence: re-running makes 0 calls
        summary2 = run_experiment(
            retriever=NoRetrieval(),
            dataset=df,
            output_dir=out,
            experiment_id="EXP_01_BASE_LLM_test",
            dataset_label="full_12723",
            progress=False,
        )
        assert summary2["n_calls_this_run"] == 0
        assert summary2["n_questions"] == 3
    finally:
        shutil.rmtree(out)


def test_runner_handles_golden_row_shape() -> None:
    """Golden rows have a different shape (options is already a dict;
    `gold_answer_letter` instead of `answer_idx`; question_id is int).
    Runner must parse them via `_row_to_inputs` golden branch."""
    golden = load_golden()[:2]
    out = Path(tempfile.mkdtemp(prefix="exp01_golden_test_"))
    try:
        summary = run_experiment(
            retriever=NoRetrieval(),
            dataset=golden,
            output_dir=out,
            experiment_id="EXP_01_BASE_LLM_test_golden",
            dataset_label="golden_234",
            progress=False,
        )
        assert summary["n_questions"] == 2
        preds = [json.loads(line) for line in (out / "predictions.jsonl").read_text().splitlines()]
        assert all(p["question_id"].startswith("golden_") for p in preds)
    finally:
        shutil.rmtree(out)


if __name__ == "__main__":
    test_loaders_shape_and_types()
    print("✓ test_loaders_shape_and_types")
    test_no_retrieval_returns_empty()
    print("✓ test_no_retrieval_returns_empty")
    test_non_llm_metrics_basic()
    print("✓ test_non_llm_metrics_basic")
    test_runner_end_to_end_on_3_medqa_rows()
    print("✓ test_runner_end_to_end_on_3_medqa_rows")
    test_runner_handles_golden_row_shape()
    print("✓ test_runner_handles_golden_row_shape")
    print("\nAll 5 tests passed.")
