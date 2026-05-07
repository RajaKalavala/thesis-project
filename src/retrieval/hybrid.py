"""Hybrid retrieval = Reciprocal Rank Fusion of dense (Chroma/BGE) + sparse (BM25).

RRF formula (Cormack, Clarke & Buettcher 2009):

    score(d) = Σ over rankers r:  1 / (k + rank_r(d))

with `k = 60` per `plan.md §0 #6`. RRF only uses **ranks**, not raw similarity
scores, so the dense and sparse retrievers don't need score-calibration.

This module exposes two APIs that share the same fusion logic:

1. **`hybrid_top_k(...)`** — function-based, used by Notebook 04 golden-set
   construction (where the caller manages embedder/chroma/bm25 lifecycles
   directly). Kept stable for backwards compatibility.

2. **`HybridRetriever`** — class conforming to the `Retriever` ABC, used by
   the runner in `src/eval/runner.py` for EXP_04. Wraps `hybrid_top_k`
   internally and joins back to `chunks_df` to populate `Chunk.text` /
   `Chunk.book_name`.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from src.data.embedder import embed_queries
from src.data.indices import bm25_top_k
from src.retrieval.base import Chunk, Retriever

RRF_K = 60


def hybrid_top_k(
    query: str,
    *,
    embedder_model,
    chroma_coll,
    bm25_payload: dict,
    k: int = 5,
    fetch_per_retriever: int | None = None,
    rrf_k: int = RRF_K,
) -> list[tuple[str, float]]:
    """Run BGE-dense + BM25-sparse, fuse with RRF, return top-k as [(chunk_id, rrf_score), …].

    `fetch_per_retriever` defaults to `max(k * 4, 20)` so RRF has enough overlap
    candidates to fuse meaningfully. Construction-time callers in Notebook 04
    typically use k=10 with fetch=10 (dense + sparse each return 10).
    """
    if fetch_per_retriever is None:
        fetch_per_retriever = max(k * 4, 20)

    # 1. Dense (ChromaDB / BGE)
    q_emb = embed_queries(embedder_model, [query])
    dense_res = chroma_coll.query(
        query_embeddings=q_emb.tolist(),
        n_results=fetch_per_retriever,
    )
    dense_ids = list(dense_res["ids"][0])

    # 2. Sparse (BM25)
    sparse_pairs = bm25_top_k(query, bm25_payload, k=fetch_per_retriever)
    sparse_ids = [cid for cid, _ in sparse_pairs]

    # 3. RRF fusion
    rrf: dict[str, float] = defaultdict(float)
    for rank, cid in enumerate(dense_ids, start=1):
        rrf[cid] += 1.0 / (rrf_k + rank)
    for rank, cid in enumerate(sparse_ids, start=1):
        rrf[cid] += 1.0 / (rrf_k + rank)

    fused = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)
    return fused[:k]


def hybrid_top_k_with_text(
    query: str,
    *,
    embedder_model,
    chroma_coll,
    bm25_payload: dict,
    chunks_df: Any,
    k: int = 5,
    fetch_per_retriever: int | None = None,
    rrf_k: int = RRF_K,
) -> list[dict]:
    """Same as `hybrid_top_k` but joins back to chunks_df to also return text.

    Returns: [{"chunk_id": str, "text": str, "rrf_score": float, "book_name": str}, …].
    """
    pairs = hybrid_top_k(
        query,
        embedder_model=embedder_model,
        chroma_coll=chroma_coll,
        bm25_payload=bm25_payload,
        k=k,
        fetch_per_retriever=fetch_per_retriever,
        rrf_k=rrf_k,
    )
    by_id = chunks_df.set_index("chunk_id")
    out = []
    for cid, score in pairs:
        if cid in by_id.index:
            row = by_id.loc[cid]
            out.append({
                "chunk_id": cid,
                "rrf_score": float(score),
                "text": str(row["text"]),
                "book_name": str(row["book_name"]),
                "n_tokens": int(row["n_tokens"]),
            })
    return out


# ---------------------------------------------------------------------------
# `Retriever` ABC subclass — used by the EXP_04 runner
# ---------------------------------------------------------------------------


class HybridRetriever(Retriever):
    """Hybrid dense + sparse retrieval via RRF, conforming to the `Retriever`
    ABC. Used by the runner for EXP_04.

    Constructor is heavyweight: BGE-large + ChromaDB collection + BM25 payload
    + chunks lookup. Build once at notebook startup and reuse across all
    questions in a run.

    Score semantics:
        `Chunk.score = RRF score` ∈ [0, 2/rrf_k] = [0, ~0.033]. RRF scores are
        comparable WITHIN a question (top-k ordering is meaningful) but not
        across questions or retrievers — they're just rank-fusion weights.
        Higher = better, per the project-wide `Retriever` convention.
    """

    def __init__(
        self,
        embedder_model: Any,
        chroma_collection: Any,
        bm25_payload: dict,
        chunks_df: pd.DataFrame,
        *,
        fetch_per_retriever: int | None = None,
        rrf_k: int = RRF_K,
    ) -> None:
        self._embedder = embedder_model
        self._chroma = chroma_collection
        self._bm25 = bm25_payload
        self._fetch_per_retriever = fetch_per_retriever
        self._rrf_k = rrf_k
        # Pre-compute chunk_id → text/book_name lookup as a dict (~10× faster
        # than repeated DataFrame.loc on a 67k-row frame).
        self._by_id = (
            chunks_df.set_index("chunk_id")[["text", "book_name"]].to_dict("index")
        )

    def retrieve(self, question: str, k: int) -> list[Chunk]:
        if k <= 0:
            return []
        pairs = hybrid_top_k(
            question,
            embedder_model=self._embedder,
            chroma_coll=self._chroma,
            bm25_payload=self._bm25,
            k=k,
            fetch_per_retriever=self._fetch_per_retriever,
            rrf_k=self._rrf_k,
        )
        chunks: list[Chunk] = []
        for cid, score in pairs:
            row = self._by_id.get(cid, {})
            chunks.append(
                Chunk(
                    chunk_id=str(cid),
                    book_name=str(row.get("book_name", "unknown")),
                    text=str(row.get("text", "")),
                    score=float(score),
                )
            )
        return chunks
