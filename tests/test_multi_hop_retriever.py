"""3-row real-data test for `src/retrieval/multi_hop.py`.

Heavyweight: loads BGE-large + ChromaDB AND makes real Groq calls for
sub-query generation (~6 calls for the 2-row test, ~$0 on Groq free tier).
Cached, so re-runs are free.

    .venv/bin/python tests/test_multi_hop_retriever.py
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

from dotenv import load_dotenv

load_dotenv(_REPO_ROOT / ".env")

from src.data.embedder import best_device, load_bge
from src.data.indices import load_chroma
from src.data.loaders import load_medqa_4opt
from src.retrieval.base import Chunk, Retriever
from src.retrieval.multi_hop import (
    DEFAULT_MAX_HOPS,
    DEFAULT_PER_HOP_K,
    MultiHopRetriever,
    _clean_subquery,
)


def test_clean_subquery_handles_typical_llm_outputs() -> None:
    """Pure-Python — no LLM calls."""
    assert _clean_subquery("antibiotics for community-acquired pneumonia") == (
        "antibiotics for community-acquired pneumonia"
    )
    assert _clean_subquery("1. macrolide vs beta-lactam choice") == "macrolide vs beta-lactam choice"
    assert _clean_subquery("2) treatment of legionella") == "treatment of legionella"
    assert _clean_subquery("- pulmonary fibrosis") == "pulmonary fibrosis"
    assert _clean_subquery("* macrolide use") == "macrolide use"
    assert _clean_subquery("Follow-up query: pulmonary fibrosis pathology") == (
        "pulmonary fibrosis pathology"
    )
    assert _clean_subquery("") == ""
    assert _clean_subquery("\n\n  query here  \n\n") == "query here"
    # Multi-line — take first non-empty line
    assert _clean_subquery("\nfirst query\nsecond query") == "first query"


def test_multi_hop_constructor_validates_inputs() -> None:
    """Pure-Python — no LLM calls."""
    chroma = None  # we won't call retrieve
    embedder = None
    # Boundary: max_hops < 1 should raise
    try:
        MultiHopRetriever(embedder, chroma, max_hops=0)
        raise AssertionError("expected ValueError for max_hops=0")
    except ValueError:
        pass
    try:
        MultiHopRetriever(embedder, chroma, per_hop_k=0)
        raise AssertionError("expected ValueError for per_hop_k=0")
    except ValueError:
        pass


def _load_retriever() -> MultiHopRetriever:
    chroma = load_chroma(_REPO_ROOT / "data" / "indices" / "chroma_textbooks")
    embedder = load_bge(device=best_device())
    return MultiHopRetriever(embedder, chroma, max_hops=DEFAULT_MAX_HOPS, per_hop_k=DEFAULT_PER_HOP_K)


def test_multi_hop_returns_chunk_objects_and_dedups() -> None:
    """Real Groq calls (cached)."""
    r = _load_retriever()
    assert isinstance(r, Retriever)

    medqa = load_medqa_4opt().head(2)
    for _, row in medqa.iterrows():
        chunks = r.retrieve(row["question"], k=15)
        # Must be 1..max_hops*per_hop_k = 1..15
        assert 1 <= len(chunks) <= DEFAULT_MAX_HOPS * DEFAULT_PER_HOP_K
        # All unique (dedup contract)
        assert len({c.chunk_id for c in chunks}) == len(chunks), "dedup failed"
        for c in chunks:
            assert isinstance(c, Chunk)
            assert c.chunk_id and c.text and c.book_name
            assert 0.0 <= c.score <= 1.0


def test_multi_hop_k_zero_returns_empty() -> None:
    r = _load_retriever()
    assert r.retrieve("any question", k=0) == []


def test_multi_hop_k_truncation() -> None:
    """If caller asks for fewer than `max_hops × per_hop_k` chunks, return only
    the first `k` (best-first across hops)."""
    r = _load_retriever()
    chunks = r.retrieve("first-line treatment community acquired pneumonia", k=3)
    assert len(chunks) == 3


if __name__ == "__main__":
    test_clean_subquery_handles_typical_llm_outputs()
    print("✓ test_clean_subquery_handles_typical_llm_outputs")
    test_multi_hop_constructor_validates_inputs()
    print("✓ test_multi_hop_constructor_validates_inputs")
    test_multi_hop_returns_chunk_objects_and_dedups()
    print("✓ test_multi_hop_returns_chunk_objects_and_dedups")
    test_multi_hop_k_zero_returns_empty()
    print("✓ test_multi_hop_k_zero_returns_empty")
    test_multi_hop_k_truncation()
    print("✓ test_multi_hop_k_truncation")
    print("\nAll 5 MultiHopRetriever real-data tests passed.")
