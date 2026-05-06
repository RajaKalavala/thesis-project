"""3-row real-data test for `src/retrieval/naive.py`.

Loads BGE-large + the persistent ChromaDB collection — heavyweight (~3 s on
M1 Pro), so this lives in its own file rather than being mixed into the fast
`test_exp01_modules.py` suite. Run when you change the retriever or the
indices, not on every commit.

    .venv/bin/python tests/test_naive_retriever.py
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Quiet ChromaDB telemetry — same pattern Notebook 03 uses.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
logging.getLogger("chromadb").setLevel(logging.WARNING)

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.embedder import best_device, load_bge
from src.data.indices import load_chroma
from src.data.loaders import load_medqa_4opt
from src.retrieval.base import Chunk, Retriever
from src.retrieval.naive import NaiveRetriever


def _load_retriever() -> NaiveRetriever:
    chroma_coll = load_chroma(_REPO_ROOT / "data" / "indices" / "chroma_textbooks")
    model = load_bge(device=best_device())
    return NaiveRetriever(chroma_coll, model)


def test_naive_retriever_returns_chunk_objects() -> None:
    r = _load_retriever()
    assert isinstance(r, Retriever)

    medqa = load_medqa_4opt().head(3)
    for _, row in medqa.iterrows():
        chunks = r.retrieve(row["question"], k=5)
        assert len(chunks) == 5, f"expected 5 chunks, got {len(chunks)}"
        for c in chunks:
            assert isinstance(c, Chunk)
            assert c.chunk_id and c.text and c.book_name
            assert 0.0 <= c.score <= 1.0, f"score out of [0,1]: {c.score}"


def test_naive_retriever_k_zero_returns_empty() -> None:
    r = _load_retriever()
    assert r.retrieve("any question", k=0) == []


def test_naive_retriever_score_ordering() -> None:
    """Scores should be monotonically non-increasing (best first).
    ChromaDB returns ranked results; we just flip distance → similarity."""
    r = _load_retriever()
    chunks = r.retrieve(
        "What is the first-line treatment for community-acquired pneumonia?", k=10
    )
    scores = [c.score for c in chunks]
    assert scores == sorted(scores, reverse=True), f"scores not sorted desc: {scores}"


if __name__ == "__main__":
    test_naive_retriever_returns_chunk_objects()
    print("✓ test_naive_retriever_returns_chunk_objects")
    test_naive_retriever_k_zero_returns_empty()
    print("✓ test_naive_retriever_k_zero_returns_empty")
    test_naive_retriever_score_ordering()
    print("✓ test_naive_retriever_score_ordering")
    print("\nAll 3 NaiveRetriever real-data tests passed.")
