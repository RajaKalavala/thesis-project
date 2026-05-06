"""Naive RAG retriever — single-shot ChromaDB top-k with the BGE-large query prefix.

This is the EXP_02 retriever. Conforms to the `Retriever` ABC so the runner
in `src/eval/runner.py` swaps it in for `NoRetrieval` with no other changes.

The pattern was inlined inside Notebook 03's smoke test; this is the refactor
into `src/` per the architecture rule "code that's used twice belongs in src/".

Construction is heavyweight (loads BGE model + opens ChromaDB persistent client),
so callers should build the retriever once at notebook startup and reuse it
across all questions in a run.

Score semantics:
    `Chunk.score = 1 - cosine_distance`  (cosine **similarity** in [0, 1]).
    Higher is more relevant. ChromaDB returns cosine *distance*; we flip
    so retrievers across the project share a "higher = better" convention.
"""
from __future__ import annotations

from typing import Any

from src.data.embedder import embed_queries
from src.retrieval.base import Chunk, Retriever


class NaiveRetriever(Retriever):
    """Dense top-k over BGE-large embeddings stored in ChromaDB."""

    def __init__(self, chroma_collection: Any, embedder_model: Any) -> None:
        self._chroma = chroma_collection
        self._embedder = embedder_model

    def retrieve(self, question: str, k: int) -> list[Chunk]:
        if k <= 0:
            return []
        # BGE query prefix is applied inside `embed_queries` per
        # `src/data/embedder.py` (BGE_QUERY_PREFIX).
        q_emb = embed_queries(self._embedder, [question])
        res = self._chroma.query(
            query_embeddings=q_emb.tolist(),
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        ids = res["ids"][0]
        docs = res["documents"][0]
        dists = res["distances"][0]
        metas = res.get("metadatas", [None])[0] or [{} for _ in ids]

        chunks: list[Chunk] = []
        for cid, doc, dist, meta in zip(ids, docs, dists, metas):
            book = (meta or {}).get("book_name", "unknown")
            # Cosine distance → similarity. ChromaDB cosine distance is in [0, 2];
            # for L2-normalised vectors (which BGE-large outputs) it's [0, 2] but
            # in practice typically [0, 1] for any real query/passage pair.
            score = float(1.0 - dist) if dist is not None else 0.0
            chunks.append(
                Chunk(chunk_id=str(cid), book_name=str(book), text=str(doc), score=score)
            )
        return chunks
