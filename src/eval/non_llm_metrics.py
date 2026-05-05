"""Non-LLM evaluation metrics.

Two families:

- **Answer-side**: `exact_match`, `accuracy` — used by every Phase-4 experiment
  on the full 12,723 surface. EXP_01 (No-RAG) uses only these.

- **Retrieval-side**: `recall_at_k`, `mrr`, `ndcg_at_k` — used by EXP_02→EXP_05.
  Computed against the **golden subset** because only golden rows have
  `gold_chunks` (the per-question ground-truth chunk IDs from Phase 3).

Per `docs/dataset.md` §4: retrieval recall is golden-only by construction —
the raw 12,723 has no chunk-level ground truth.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def exact_match(pred: str | None, gold: str) -> bool:
    """Letter-level exact match. Case-insensitive; `None` predictions miss."""
    if pred is None:
        return False
    return pred.strip().upper() == gold.strip().upper()


def accuracy(predictions: Sequence[str | None], golds: Sequence[str]) -> float:
    """Fraction in [0, 1]. Empty input returns 0.0."""
    if not predictions:
        return 0.0
    if len(predictions) != len(golds):
        raise ValueError(f"length mismatch: {len(predictions)} preds vs {len(golds)} golds")
    correct = sum(1 for p, g in zip(predictions, golds) if exact_match(p, g))
    return correct / len(predictions)


def recall_at_k(retrieved_ids: Sequence[str], gold_ids: Iterable[str], k: int) -> float:
    """Fraction of `gold_ids` that appear in the top-`k` of `retrieved_ids`.

    Per-question metric — average across questions to get the headline figure.
    """
    gold_set = set(gold_ids)
    if not gold_set:
        return 0.0
    top = list(retrieved_ids)[:k]
    hits = sum(1 for cid in gold_set if cid in top)
    return hits / len(gold_set)


def mrr(retrieved_ids: Sequence[str], gold_ids: Iterable[str]) -> float:
    """Reciprocal rank of the first gold-chunk hit. 0.0 if none of the gold
    chunks are retrieved at all. Per-question; average across questions.
    """
    gold_set = set(gold_ids)
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in gold_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], gold_ids: Iterable[str], k: int) -> float:
    """Binary-relevance nDCG@k: gain = 1 if a retrieved chunk is in `gold_ids`,
    else 0. Discount = `1 / log2(rank + 1)`. Ideal DCG assumes all gold chunks
    fill the top positions (capped at `k`).
    """
    gold_set = set(gold_ids)
    if not gold_set:
        return 0.0
    top = list(retrieved_ids)[:k]
    dcg = sum(1.0 / math.log2(rank + 1) for rank, cid in enumerate(top, start=1) if cid in gold_set)
    ideal_hits = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0
