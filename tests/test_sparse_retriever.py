"""3-row real-data test for `src/retrieval/sparse.py`.

Loads the BM25 pickle (~106 MB) + chunks parquet — heavyweight (~3 s to start),
so this lives in its own file rather than being mixed into the fast
`test_exp01_modules.py` suite. Run when you change the retriever or the BM25
index, not on every commit.

    .venv/bin/python tests/test_sparse_retriever.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.indices import load_bm25
from src.data.loaders import load_chunks, load_medqa_4opt
from src.retrieval.base import Chunk, Retriever
from src.retrieval.sparse import SparseRetriever


def _load_retriever() -> SparseRetriever:
    chunks_df = load_chunks()
    bm25_payload = load_bm25(_REPO_ROOT / "data" / "indices" / "bm25.pkl")
    return SparseRetriever(bm25_payload, chunks_df)


def test_sparse_retriever_returns_chunk_objects() -> None:
    r = _load_retriever()
    assert isinstance(r, Retriever)

    medqa = load_medqa_4opt().head(3)
    for _, row in medqa.iterrows():
        chunks = r.retrieve(row["question"], k=5)
        assert len(chunks) == 5, f"expected 5 chunks, got {len(chunks)}"
        for c in chunks:
            assert isinstance(c, Chunk)
            assert c.chunk_id and c.text and c.book_name
            # BM25 raw scores can be negative for queries with no good keyword
            # match, but are typically 30–200+ for medical-question keyword overlap.
            assert isinstance(c.score, float)


def test_sparse_retriever_k_zero_returns_empty() -> None:
    r = _load_retriever()
    assert r.retrieve("any question", k=0) == []


def test_sparse_retriever_score_ordering() -> None:
    """Scores should be monotonically non-increasing (best first).
    `bm25_top_k` returns argsorted by descending score."""
    r = _load_retriever()
    chunks = r.retrieve(
        "What is the first-line treatment for community-acquired pneumonia?", k=10
    )
    scores = [c.score for c in chunks]
    assert scores == sorted(scores, reverse=True), f"scores not sorted desc: {scores}"


def test_sparse_retriever_recovers_rare_terms() -> None:
    """BM25 should put a chunk containing rare keywords near the top.
    Test: a query with a specific drug name should retrieve a chunk
    mentioning it. This validates BM25's keyword-matching superpower
    (which the dense retriever may miss for rare terms)."""
    r = _load_retriever()
    chunks = r.retrieve("What is the mechanism of action of cisplatin?", k=10)
    # At least one of the top-5 should mention 'cisplatin' literally.
    text_blob = " ".join(c.text.lower() for c in chunks[:5])
    assert "cisplatin" in text_blob, (
        f"expected 'cisplatin' literal match in top-5; got chunks from books: "
        f"{[c.book_name for c in chunks[:5]]}"
    )


if __name__ == "__main__":
    test_sparse_retriever_returns_chunk_objects()
    print("✓ test_sparse_retriever_returns_chunk_objects")
    test_sparse_retriever_k_zero_returns_empty()
    print("✓ test_sparse_retriever_k_zero_returns_empty")
    test_sparse_retriever_score_ordering()
    print("✓ test_sparse_retriever_score_ordering")
    test_sparse_retriever_recovers_rare_terms()
    print("✓ test_sparse_retriever_recovers_rare_terms")
    print("\nAll 4 SparseRetriever real-data tests passed.")
