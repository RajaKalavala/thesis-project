"""Multi-Hop RAG retriever — iterative dense retrieval with sub-query generation.

This is the EXP_05 retriever. Conforms to the `Retriever` ABC; the runner
swaps it in for `NaiveRetriever` / `SparseRetriever` / `HybridRetriever` with
no other changes.

## Algorithm

```
hop 1: query = original_question
       → BGE-large dense top-k chunks
       → accumulate (deduped by chunk_id)
hop 2: query = LLM-generated follow-up given (question + accumulated chunks)
       → BGE-large dense top-k NEW chunks (skip already-seen)
       → accumulate
hop 3: same iterative pattern
       → if a hop returns 0 new chunks, stop early (no progress = done)
```

Final return: first `k` accumulated chunks (best-first across hops).

## Design choices

- **3 hops max** per `plan.md §0 #7`. The proposal §7.6.4 names this number.
- **k=5 per hop** per `plan.md §6` table; total `accumulated <= 15` chunks.
- **Sub-query LLM = the same LLaMA 3.3 70B (Groq, T=0)** — keeping the
  generator family stable across the architecture; the sub-query call is
  just a smaller variant of the answer call.
- **Cached** through `groq_complete` → `data/cache/groq/`. Restarts free.
- **Score** for `Chunk.score` = the dense cosine similarity from the *first*
  hop that retrieved it. Higher = better, project-wide convention.

## Prompt

Sub-query generation uses `build_multi_hop_subquery_prompt` from
`src/generation/prompts.py`. Output is one short line of plain text — we
strip + collapse whitespace + truncate at the first newline at the call site.

## Cost characteristics

Per question on test_1273: ~2 sub-query Groq calls (hops 2 + 3) + 1 final
answer Groq call = 3 calls. 1,273 questions × 3 = ~3,800 Groq calls per run.
Free tier handles this in ~30–45 min wall time. Cost: $0.

For thesis defensibility: this is a deliberately conservative multi-hop
implementation — no LLM-judge "do I need another hop?" gate (which would add
calls), no agent loop, no tool use. Just decompose-and-accumulate.
"""
from __future__ import annotations

import re
from typing import Any

from src.data.embedder import embed_queries
from src.generation.groq_client import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    groq_complete,
)
from src.generation.prompts import build_multi_hop_subquery_prompt
from src.retrieval.base import Chunk, Retriever

DEFAULT_MAX_HOPS = 3
DEFAULT_PER_HOP_K = 5
SUBQUERY_MAX_TOKENS = 80  # follow-up queries are short — bound the spend


def _clean_subquery(raw: str) -> str:
    """Take the LLM's raw output and turn it into a plain-text query.
    Strips whitespace, takes the first non-empty line, drops a leading bullet
    or number prefix if the model emitted one, and caps at 200 chars.
    """
    if not raw:
        return ""
    # Take first non-empty line
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Drop common prefixes the model sometimes adds despite instructions
        line = re.sub(r"^(\d+[\.\)]\s*|[-*•]\s*|Follow-up query[:\s]*)", "", line, flags=re.IGNORECASE)
        return line[:200].strip()
    return ""


class MultiHopRetriever(Retriever):
    """Iterative dense retrieval over BGE-large + ChromaDB. Generates
    follow-up sub-queries via the same Groq LLaMA 3.3 70B answerer.

    Constructor is heavyweight (BGE-large model + ChromaDB persistent client).
    Build once at notebook startup and reuse across all questions.
    """

    def __init__(
        self,
        embedder_model: Any,
        chroma_collection: Any,
        *,
        max_hops: int = DEFAULT_MAX_HOPS,
        per_hop_k: int = DEFAULT_PER_HOP_K,
        subquery_model: str = DEFAULT_MODEL,
        subquery_temperature: float = DEFAULT_TEMPERATURE,
        subquery_max_tokens: int = SUBQUERY_MAX_TOKENS,
    ) -> None:
        if max_hops < 1:
            raise ValueError("max_hops must be >= 1")
        if per_hop_k < 1:
            raise ValueError("per_hop_k must be >= 1")
        self._embedder = embedder_model
        self._chroma = chroma_collection
        self._max_hops = max_hops
        self._per_hop_k = per_hop_k
        self._subquery_model = subquery_model
        self._subquery_temperature = subquery_temperature
        self._subquery_max_tokens = subquery_max_tokens

    def retrieve(self, question: str, k: int) -> list[Chunk]:
        """Return up to `k` accumulated chunks across ≤`max_hops` retrieval
        rounds. Note that `k` here is the FINAL chunk count returned to the
        runner; per-hop fan-out is `self._per_hop_k`."""
        if k <= 0:
            return []
        accumulated: list[Chunk] = []
        seen_ids: set[str] = set()
        current_query = question

        for hop_n in range(self._max_hops):
            new_chunks = self._dense_top_k(current_query, self._per_hop_k)
            new_unique = [c for c in new_chunks if c.chunk_id not in seen_ids]
            if not new_unique and hop_n > 0:
                # No progress — early stop (`plan.md §15` cap to prevent
                # iterative-retrieval loops).
                break
            accumulated.extend(new_unique)
            seen_ids.update(c.chunk_id for c in new_unique)

            # Skip sub-query generation after the LAST hop
            if hop_n + 1 < self._max_hops:
                next_query = self._generate_subquery(question, accumulated)
                if not next_query:
                    # Sub-query generation failed (empty / parse error) — stop early
                    break
                current_query = next_query

        return accumulated[:k]

    def _dense_top_k(self, query: str, n: int) -> list[Chunk]:
        """Same body as `NaiveRetriever.retrieve` — kept inline because the
        runtime cost is dominated by Groq, not by code reuse, and in-lining
        avoids a circular import with `src/retrieval/naive.py`."""
        q_emb = embed_queries(self._embedder, [query])
        res = self._chroma.query(
            query_embeddings=q_emb.tolist(),
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        ids = res["ids"][0]
        docs = res["documents"][0]
        dists = res["distances"][0]
        metas = res.get("metadatas", [None])[0] or [{} for _ in ids]

        chunks: list[Chunk] = []
        for cid, doc, dist, meta in zip(ids, docs, dists, metas):
            book = (meta or {}).get("book_name", "unknown")
            score = float(1.0 - dist) if dist is not None else 0.0
            chunks.append(
                Chunk(chunk_id=str(cid), book_name=str(book), text=str(doc), score=score)
            )
        return chunks

    def _generate_subquery(self, question: str, accumulated: list[Chunk]) -> str:
        """Ask Groq to generate a follow-up search query targeting an
        evidence gap. Returns the cleaned query string (or empty on
        parse failure)."""
        prompt = build_multi_hop_subquery_prompt(question, [c.text for c in accumulated])
        text, _latency_s, _was_cached = groq_complete(
            prompt,
            model=self._subquery_model,
            temperature=self._subquery_temperature,
            max_tokens=self._subquery_max_tokens,
        )
        return _clean_subquery(text)
