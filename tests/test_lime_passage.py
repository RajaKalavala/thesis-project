"""Unit tests for `src/xai/lime_passage.py`.

Uses a stub LLM (no real Groq calls) so the test is fast and deterministic.
Verifies:
- LOO mechanics: full prompt + k LOO prompts called, predictions threaded through
- correctness_attribution sign: positive when removing a helpful chunk
- sameletter_attribution: 1 when LOO flips the letter
- top_chunk_by_correctness picks the right chunk
- Resumability via _load_completed_keys
- No-passages path (No-RAG) returns early with a note

    .venv/bin/python tests/test_lime_passage.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.retrieval.base import Chunk
from src.xai.lime_passage import (
    _load_completed_keys,
    lime_passage_loo,
    passage_subset_lime,
    run_lime_passage_batch,
    run_subset_lime_batch,
)


class _StubLLM:
    """LLM stub keyed on a substring of the prompt → returned letter.

    Calls are deterministic; we track `n_calls` and the prompts seen so the
    test can assert on call structure.
    """

    def __init__(self, rules: list[tuple[str, str]]) -> None:
        # rules: list of (prompt_must_contain_substring, return_letter).
        # First matching rule wins; falls back to "B" otherwise.
        self.rules = rules
        self.n_calls = 0
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> tuple[str, float, bool]:
        self.n_calls += 1
        self.prompts.append(prompt)
        for needle, letter in self.rules:
            if needle in prompt:
                return letter, 0.01, False
        return "B", 0.01, False


def _make_chunks() -> list[Chunk]:
    return [
        Chunk(chunk_id="c0", book_name="b", text="alpha unique marker", score=0.9),
        Chunk(chunk_id="c1", book_name="b", text="beta unique marker", score=0.8),
        Chunk(chunk_id="c2", book_name="b", text="gamma unique marker", score=0.7),
    ]


def test_loo_call_count() -> None:
    chunks = _make_chunks()
    stub = _StubLLM([])
    result = lime_passage_loo(
        question_id="q1",
        question="Test question?",
        options={"A": "x", "B": "y", "C": "z", "D": "w"},
        gold_letter="A",
        chunks=chunks,
        architecture="arch",
        llm_callable=stub,
    )
    # 1 full + 3 LOO = 4 calls
    assert stub.n_calls == 4, f"expected 4 calls, got {stub.n_calls}"
    assert result.n_passages == 3
    assert len(result.passages) == 3


def test_correctness_attribution_positive_when_removal_hurts() -> None:
    """Full prompt → correct letter A; LOO of c0 → wrong letter B; LOO of
    c1/c2 → correct A. So c0's correctness_attribution should be +1, others 0."""
    chunks = _make_chunks()
    stub = _StubLLM(
        [
            # If "alpha unique marker" is missing, we return "B" (wrong); otherwise "A".
            # Match the LOO-of-c0 case first by checking absence... simpler:
            # since first-match wins, we rule on whether "alpha unique marker" is present.
            # Trick: rule list is checked in order; we want to special-case "no alpha".
        ]
    )

    # Custom rule: if "alpha unique marker" appears in prompt -> "A"; else "B"
    class _ConditionalStub(_StubLLM):
        def __call__(self, prompt: str) -> tuple[str, float, bool]:
            self.n_calls += 1
            self.prompts.append(prompt)
            return ("A", 0.01, False) if "alpha unique marker" in prompt else ("B", 0.01, False)

    stub2 = _ConditionalStub([])
    result = lime_passage_loo(
        question_id="q1",
        question="Test question?",
        options={"A": "x", "B": "y", "C": "z", "D": "w"},
        gold_letter="A",
        chunks=chunks,
        architecture="arch",
        llm_callable=stub2,
    )
    assert result.full_pred_letter == "A"
    assert result.full_correct is True
    # c0 (alpha) is the only essential chunk:
    p_c0 = next(p for p in result.passages if p.chunk_id == "c0")
    p_c1 = next(p for p in result.passages if p.chunk_id == "c1")
    p_c2 = next(p for p in result.passages if p.chunk_id == "c2")
    assert p_c0.correctness_attribution == 1, "c0 should be +1 (removing it hurt)"
    assert p_c0.sameletter_attribution == 1, "c0 LOO changed letter A→B"
    assert p_c0.loo_correct is False, "c0 LOO predicted B, gold is A"
    assert p_c1.correctness_attribution == 0, "c1 should be 0 (removing it didn't hurt)"
    assert p_c2.correctness_attribution == 0
    assert result.top_chunk_by_correctness == "c0"
    assert result.top_chunk_by_sameletter == "c0"


def test_no_passages_path() -> None:
    """No chunks → returns early with note; full prompt is no-RAG style."""
    stub = _StubLLM([])
    result = lime_passage_loo(
        question_id="q1",
        question="Test question?",
        options={"A": "x", "B": "y", "C": "z", "D": "w"},
        gold_letter="A",
        chunks=[],
        architecture="no_rag",
        llm_callable=stub,
    )
    assert result.n_passages == 0
    assert result.note == "no_passages_skipped_loo"
    assert stub.n_calls == 1, "only the no-RAG prompt should fire"
    assert result.passages == []


def test_resumability_round_trip() -> None:
    """Write 2 LIME rows, then resume — second batch should skip both."""
    chunks = _make_chunks()
    stub = _StubLLM([])
    rows = [
        {
            "question_id": "q1",
            "question": "Test?",
            "options": {"A": "x", "B": "y", "C": "z", "D": "w"},
            "gold_letter": "A",
            "chunks": chunks,
            "architecture": "arch",
        },
        {
            "question_id": "q2",
            "question": "Test?",
            "options": {"A": "x", "B": "y", "C": "z", "D": "w"},
            "gold_letter": "A",
            "chunks": chunks,
            "architecture": "arch",
        },
    ]
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "lime.jsonl"
        s1 = run_lime_passage_batch(rows, out, stub, progress=False)
        assert s1["n_rows_written_this_run"] == 2
        assert s1["n_rows_already_done"] == 0
        # Resume — both should be skipped
        s2 = run_lime_passage_batch(rows, out, stub, progress=False)
        assert s2["n_rows_written_this_run"] == 0
        assert s2["n_rows_already_done"] == 2
        # File still has exactly 2 rows
        assert sum(1 for _ in out.read_text().splitlines()) == 2


def test_load_completed_keys_tolerates_garbage() -> None:
    """Final partial line shouldn't crash the loader."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(json.dumps({"question_id": "q1", "architecture": "a"}) + "\n")
        f.write("not-json-line\n")
        f.write('{"question_id": "q2", "architecture": "a"}\n')  # well-formed
        path = Path(f.name)
    keys = _load_completed_keys(path)
    assert ("q1", "a") in keys
    assert ("q2", "a") in keys
    path.unlink()


# ---------------------------------------------------------------------------
#  Subset-sampling LIME tests
# ---------------------------------------------------------------------------


def test_subset_lime_call_count_and_shape() -> None:
    """N=8 samples → 8 LLM calls; passage rows match chunk count."""
    chunks = _make_chunks()  # 3 chunks
    stub = _StubLLM([])  # default "B"
    result = passage_subset_lime(
        question_id="q1",
        question="Test?",
        options={"A": "x", "B": "y", "C": "z", "D": "w"},
        gold_letter="A",
        chunks=chunks,
        architecture="arch",
        llm_callable=stub,
        n_samples=8,
        seed=0,
    )
    assert stub.n_calls == 8, f"expected 8 calls, got {stub.n_calls}"
    assert len(result.samples) == 8
    assert len(result.passages) == 3
    assert result.method == "subset_lime"
    # First sample's mask is always all-ones (the full-prediction anchor)
    assert result.samples[0].mask == [1, 1, 1]


def test_subset_lime_attributes_essential_chunk() -> None:
    """If the LLM picks gold A only when chunk c0 is present, ridge should
    assign c0 the largest absolute coefficient on the correctness score."""
    chunks = _make_chunks()

    class _EssentialC0(_StubLLM):
        def __call__(self, prompt: str) -> tuple[str, float, bool]:
            self.n_calls += 1
            self.prompts.append(prompt)
            return ("A", 0.01, False) if "alpha unique marker" in prompt else ("B", 0.01, False)

    stub = _EssentialC0([])
    result = passage_subset_lime(
        question_id="q1",
        question="Test?",
        options={"A": "x", "B": "y", "C": "z", "D": "w"},
        gold_letter="A",
        chunks=chunks,
        architecture="arch",
        llm_callable=stub,
        n_samples=16,
        seed=0,
    )
    # The full prediction (all-ones mask) returns "A" (c0 is present)
    assert result.full_pred_letter == "A"
    # c0's correctness_coef should be the largest by absolute value
    coefs = {p.chunk_id: p.correctness_coef for p in result.passages}
    abs_coefs = {k: abs(v) for k, v in coefs.items()}
    top = max(abs_coefs, key=abs_coefs.get)
    assert top == "c0", f"expected c0 to dominate, got {top}; coefs={coefs}"
    assert result.top_chunk_by_correctness == "c0"
    # Variance should be > 0 (some subsets return A, others B)
    assert result.correctness_score_variance > 0


def test_subset_lime_zero_variance_when_letter_is_constant() -> None:
    """When the LLM returns the same letter regardless of mask, both
    score variances are 0 and the top-1 selections are None."""
    chunks = _make_chunks()
    stub = _StubLLM([])  # always "B"
    result = passage_subset_lime(
        question_id="q1",
        question="Test?",
        options={"A": "x", "B": "y", "C": "z", "D": "w"},
        gold_letter="A",
        chunks=chunks,
        architecture="arch",
        llm_callable=stub,
        n_samples=8,
        seed=0,
    )
    assert result.correctness_score_variance == 0.0
    assert result.sameletter_score_variance == 0.0
    assert result.top_chunk_by_correctness is None
    assert result.top_chunk_by_sameletter is None


def test_subset_lime_no_passages_path() -> None:
    """k=0 short-circuits to no-RAG: one LLM call, note set, no samples."""
    stub = _StubLLM([])
    result = passage_subset_lime(
        question_id="q1",
        question="Test?",
        options={"A": "x", "B": "y", "C": "z", "D": "w"},
        gold_letter="A",
        chunks=[],
        architecture="no_rag",
        llm_callable=stub,
        n_samples=16,
    )
    assert result.n_passages == 0
    assert result.note == "no_passages_skipped_subset"
    assert stub.n_calls == 1
    assert result.samples == []


def test_subset_lime_batch_resumability() -> None:
    """run_subset_lime_batch skips already-completed (qid, arch) rows."""
    chunks = _make_chunks()
    stub = _StubLLM([])
    rows = [
        {
            "question_id": "q1",
            "question": "Test?",
            "options": {"A": "x", "B": "y", "C": "z", "D": "w"},
            "gold_letter": "A",
            "chunks": chunks,
            "architecture": "arch",
        },
        {
            "question_id": "q2",
            "question": "Test?",
            "options": {"A": "x", "B": "y", "C": "z", "D": "w"},
            "gold_letter": "A",
            "chunks": chunks,
            "architecture": "arch",
        },
    ]
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "subset_lime.jsonl"
        s1 = run_subset_lime_batch(rows, out, stub, n_samples=4, progress=False)
        assert s1["n_rows_written_this_run"] == 2
        s2 = run_subset_lime_batch(rows, out, stub, n_samples=4, progress=False)
        assert s2["n_rows_written_this_run"] == 0
        assert s2["n_rows_already_done"] == 2


if __name__ == "__main__":
    tests = [
        test_loo_call_count,
        test_correctness_attribution_positive_when_removal_hurts,
        test_no_passages_path,
        test_resumability_round_trip,
        test_load_completed_keys_tolerates_garbage,
        test_subset_lime_call_count_and_shape,
        test_subset_lime_attributes_essential_chunk,
        test_subset_lime_zero_variance_when_letter_is_constant,
        test_subset_lime_no_passages_path,
        test_subset_lime_batch_resumability,
    ]
    for t in tests:
        t()
        print(f"  ✓ {t.__name__}")
    print(f"\nAll {len(tests)} LIME passage tests passed.")
