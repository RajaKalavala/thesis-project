"""Sparse RAG retriever — BM25 top-k over chunk text.

This is the EXP_03 retriever. Conforms to the `Retriever` ABC so the runner
in `src/eval/runner.py` swaps it in for `NaiveRetriever` / `NoRetrieval` with
no other changes.

Wraps the existing `bm25_top_k(query, payload, k)` helper from
`src/data/indices.py` (which returns `[(chunk_id, score), …]`) and joins the
chunk_id back to the chunks DataFrame to populate `Chunk.text` + `Chunk.book_name`.

Why BM25 alongside dense (Naive RAG): dense embeddings cluster on *semantic*
similarity, which can miss rare medical terms (drug names, anatomical
structures, eponyms) where the question's exact term is the strongest signal.
Sparse keyword matching catches these cases. EXP_03 tests the falsifiable
hypothesis from EXP_02's analysis: *Context Precision will improve on
rare-term questions where keyword matching beats embedding similarity.*

Score semantics:
    `Chunk.score` is the **raw BM25 score** — typically in [0, ~30]; can be
    negative for queries with no good keyword match. Higher = better, but
    scores are NOT comparable across retrievers (BM25 is unnormalised; dense
    cosine is in [0, 1]). RRF fusion (EXP_04 hybrid) only uses ranks, not
    raw scores, so the unnormalised range is fine.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.indices import bm25_top_k
from src.retrieval.base import Chunk, Retriever


class SparseRetriever(Retriever):
    """BM25 keyword retrieval over the same 67,599-chunk corpus the dense
    retriever uses. Constructor is heavyweight (loads the BM25 pickle + builds
    a chunk_id → text/book_name lookup), so build once at notebook startup
    and reuse across all questions in a run."""

    def __init__(self, bm25_payload: dict, chunks_df: pd.DataFrame) -> None:
        self._bm25 = bm25_payload
        # chunk_id → {"text", "book_name"} lookup. Pre-computing as a dict
        # is ~10× faster than repeated DataFrame.loc on a 67k-row frame.
        self._by_id = (
            chunks_df.set_index("chunk_id")[["text", "book_name"]].to_dict("index")
        )

    def retrieve(self, question: str, k: int) -> list[Chunk]:
        if k <= 0:
            return []
        pairs = bm25_top_k(question, self._bm25, k=k)
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
