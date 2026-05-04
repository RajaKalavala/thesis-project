"""Hybrid retrieval = Reciprocal Rank Fusion of dense (Chroma/BGE) + sparse (BM25).

RRF formula (Cormack, Clarke & Buettcher 2009):

    score(d) = Σ over rankers r:  1 / (k + rank_r(d))

with `k = 60` per `plan.md §0 #6`. RRF only uses **ranks**, not raw similarity
scores, so the dense and sparse retrievers don't need score-calibration.

Used by:
- Notebook 04 (Stage B — golden-set construction-time retrieval)
- EXP_04 Hybrid RAG experiment (Phase 4)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from src.data.embedder import embed_queries
from src.data.indices import bm25_top_k

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
