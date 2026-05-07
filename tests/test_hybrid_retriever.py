"""3-row real-data test for `src/retrieval/hybrid.py::HybridRetriever`.

Loads BGE-large + ChromaDB + BM25 — heavyweight (~5 s to start), so this
lives in its own file. Run when you change the retriever or the indices.

    .venv/bin/python tests/test_hybrid_retriever.py
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
logging.getLogger("chromadb").setLevel(logging.WARNING)

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.embedder import best_device, load_bge
from src.data.indices import load_bm25, load_chroma
from src.data.loaders import load_chunks, load_medqa_4opt
from src.retrieval.base import Chunk, Retriever
from src.retrieval.hybrid import HybridRetriever, hybrid_top_k


def _load_retriever() -> HybridRetriever:
    chunks_df = load_chunks()
    chroma = load_chroma(_REPO_ROOT / "data" / "indices" / "chroma_textbooks")
    bm25 = load_bm25(_REPO_ROOT / "data" / "indices" / "bm25.pkl")
    embedder = load_bge(device=best_device())
    return HybridRetriever(embedder, chroma, bm25, chunks_df)


def test_hybrid_retriever_returns_chunk_objects() -> None:
    r = _load_retriever()
    assert isinstance(r, Retriever)

    medqa = load_medqa_4opt().head(3)
    for _, row in medqa.iterrows():
        chunks = r.retrieve(row["question"], k=5)
        assert len(chunks) == 5
        for c in chunks:
            assert isinstance(c, Chunk)
            assert c.chunk_id and c.text and c.book_name
            # RRF scores are positive but small — bounded by 2/rrf_k = 2/60 ≈ 0.033.
            assert 0.0 <= c.score <= 0.05, f"unexpected RRF score: {c.score}"


def test_hybrid_retriever_k_zero_returns_empty() -> None:
    r = _load_retriever()
    assert r.retrieve("any question", k=0) == []


def test_hybrid_retriever_score_ordering() -> None:
    r = _load_retriever()
    chunks = r.retrieve(
        "What is the first-line treatment for community-acquired pneumonia?", k=10
    )
    scores = [c.score for c in chunks]
    assert scores == sorted(scores, reverse=True), f"scores not sorted desc: {scores}"


def test_hybrid_function_api_still_works() -> None:
    """Backwards-compatibility check — the existing `hybrid_top_k` function
    is used by Notebook 04 golden construction. Don't break it."""
    from src.data.embedder import load_bge as _bge
    from src.data.indices import load_bm25 as _bm25

    chunks_df = load_chunks()
    chroma = load_chroma(_REPO_ROOT / "data" / "indices" / "chroma_textbooks")
    bm25 = _bm25(_REPO_ROOT / "data" / "indices" / "bm25.pkl")
    embedder = _bge(device=best_device())

    pairs = hybrid_top_k(
        "What is the mechanism of action of cisplatin?",
        embedder_model=embedder,
        chroma_coll=chroma,
        bm25_payload=bm25,
        k=5,
    )
    assert len(pairs) == 5
    for cid, score in pairs:
        assert isinstance(cid, str)
        assert isinstance(score, float)


if __name__ == "__main__":
    test_hybrid_retriever_returns_chunk_objects()
    print("✓ test_hybrid_retriever_returns_chunk_objects")
    test_hybrid_retriever_k_zero_returns_empty()
    print("✓ test_hybrid_retriever_k_zero_returns_empty")
    test_hybrid_retriever_score_ordering()
    print("✓ test_hybrid_retriever_score_ordering")
    test_hybrid_function_api_still_works()
    print("✓ test_hybrid_function_api_still_works")
    print("\nAll 4 HybridRetriever real-data tests passed.")
