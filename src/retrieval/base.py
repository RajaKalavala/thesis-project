"""Retriever ABC + Chunk dataclass — the contract every retrieval strategy honours.

Every concrete retriever (`none.py`, `naive.py`, `sparse.py`, `hybrid.py`,
`multi_hop.py`, `adaptive.py`) implements `retrieve(question, k) -> list[Chunk]`
so the runner in `src/eval/runner.py` can swap retrievers without per-experiment
branching. EXP_07 Adaptive RAG composes the others under this same contract.

`score` semantics differ per retriever (cosine for dense, BM25 for sparse, RRF
for hybrid) — only the *ranking* is contractually meaningful across retrievers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    book_name: str
    text: str
    score: float


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, question: str, k: int) -> list[Chunk]:
        """Return up to `k` chunks ranked best-first. May return fewer than `k`
        (or zero) chunks; callers must tolerate short lists."""
        raise NotImplementedError
