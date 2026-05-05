"""No-retrieval baseline for EXP_01.

`NoRetrieval.retrieve(...)` always returns `[]`. The runner detects an empty
chunk list and switches to `build_no_rag_prompt` instead of the
evidence-grounded template. Conforms to the `Retriever` contract so EXP_01
shares the same code path as EXP_02–EXP_05.
"""
from __future__ import annotations

from src.retrieval.base import Chunk, Retriever


class NoRetrieval(Retriever):
    def retrieve(self, question: str, k: int) -> list[Chunk]:
        return []
