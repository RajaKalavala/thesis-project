"""Adaptive RAG dispatcher (EXP_07) — routes each question to a different
underlying retriever based on a precomputed complexity bucket.

This is the EXP_07 retriever. Conforms to the `Retriever` ABC; the runner
swaps it in for `NaiveRetriever` / `MultiHopRetriever` / etc. with no other
changes — the dispatch is invisible from the runner's perspective.

## How it works

At construction time, `AdaptiveRetriever` takes:

1. ``question_to_bucket`` — a `dict[str, str]` mapping the verbatim question
   text to its complexity bucket label (one of ``Simple`` / ``Moderate`` /
   ``Complex``). Built from the EXP_06 output `complexity_labels.parquet`
   joined to the source question text in `medqa_4opt.parquet`.
2. ``bucket_to_retriever`` — a `dict[str, Retriever]` defining the routing
   table — which underlying retriever fires for each bucket.

At call time, ``retrieve(question, k)`` looks up the bucket and delegates.

If a question is not present in the lookup, the call falls back to
``default_bucket`` (`"Moderate"` by default). This handles edge cases
silently rather than raising — but ``unknown_question_count`` exposes the
miss rate so the notebook can flag it if it gets non-trivial.

## Two routing tables tested in EXP_07

### Variant A — proposal as-is

```python
{
    "Simple":   NaiveRetriever(...),     # k=5 dense BGE-large
    "Moderate": HybridRetriever(...),    # RRF dense + BM25
    "Complex":  MultiHopRetriever(...),  # 3-hop iterative dense
}
```

### Variant B — data-driven binary

```python
{
    "Simple":   NoRetrieval(),           # No-RAG (empty chunks)
    "Moderate": MultiHopRetriever(...),
    "Complex":  MultiHopRetriever(...),
}
```

Both variants are run by the EXP_07 notebook and reported side-by-side
in Table 1 row 6 + Table 10.

## Integration with the runner

`src/eval/runner.py` is unchanged. When `retrieve()` returns an empty list
(NoRetrieval branch in Variant B), the runner falls back to the No-RAG
prompt automatically — same code path as EXP_01. Score semantics are
inherited from whichever underlying retriever fires; cross-bucket scores
are not comparable but that's fine because the runner only uses
`Chunk.text` for prompt construction.

## Why a `dict[question_text, bucket]` lookup, not a runtime classifier

The complexity classifier is deterministic and side-effect-free, but
re-running it per question wastes ~12,723 dict accesses' worth of
re-parsing. Pre-computing once at notebook startup and indexing by
question text is the simplest contract that gives O(1) per-question
dispatch and makes the routing decision auditable from the parquet.
"""
from __future__ import annotations

from typing import Mapping

from src.retrieval.base import Chunk, Retriever


class AdaptiveRetriever(Retriever):
    """Per-question routing over a fixed table of underlying retrievers.

    Uses an external question-text → bucket lookup (from EXP_06's
    `complexity_labels.parquet`) to decide which of several retrievers to
    fire per question. Conforms to `Retriever` so the standard runner
    works unchanged.
    """

    def __init__(
        self,
        question_to_bucket: Mapping[str, str],
        bucket_to_retriever: Mapping[str, Retriever],
        *,
        default_bucket: str = "Moderate",
    ) -> None:
        if default_bucket not in bucket_to_retriever:
            raise ValueError(
                f"default_bucket {default_bucket!r} not in routing table "
                f"keys {sorted(bucket_to_retriever.keys())}"
            )
        # Any bucket label that appears in the lookup must have a routing entry,
        # otherwise an unrouteable question would error mid-run after hours of
        # wall time. Fail fast at construction.
        seen_buckets = set(question_to_bucket.values())
        missing = seen_buckets - set(bucket_to_retriever.keys())
        if missing:
            raise ValueError(
                f"buckets in lookup but not in routing table: {sorted(missing)}"
            )

        self._lookup: Mapping[str, str] = question_to_bucket
        self._table: Mapping[str, Retriever] = bucket_to_retriever
        self._default: str = default_bucket
        self._dispatch_counts: dict[str, int] = {b: 0 for b in bucket_to_retriever}
        self._unknown_question_count: int = 0

    def retrieve(self, question: str, k: int) -> list[Chunk]:
        bucket = self.bucket_for(question)
        retriever = self._table[bucket]
        self._dispatch_counts[bucket] += 1
        return retriever.retrieve(question, k)

    def bucket_for(self, question: str) -> str:
        """Return the bucket assigned to `question`. Falls back to
        `default_bucket` (and increments `unknown_question_count`) if the
        question text is not in the lookup."""
        bucket = self._lookup.get(question)
        if bucket is None:
            self._unknown_question_count += 1
            return self._default
        return bucket

    @property
    def dispatch_counts(self) -> dict[str, int]:
        """Per-bucket invocation counts — useful for the EXP_07 notebook to
        sanity-check that the routing fan-out matches `complexity_labels.parquet`."""
        return dict(self._dispatch_counts)

    @property
    def unknown_question_count(self) -> int:
        """Number of `retrieve()` calls that hit the default-bucket fallback
        because the question text was not in the lookup. Should be 0 on
        test_1273 / golden_234 surfaces; non-zero suggests a join bug."""
        return self._unknown_question_count
