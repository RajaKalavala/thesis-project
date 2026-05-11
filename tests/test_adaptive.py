"""Real-data test for `src/retrieval/adaptive.py`.

Builds a tiny routing table over stub retrievers, plus one end-to-end check
on real medqa_4opt rows + EXP_06 labels. Lightweight (~1 s) — no models or
indices loaded; the underlying retrievers are stubs.

    .venv/bin/python tests/test_adaptive.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from src.retrieval.adaptive import AdaptiveRetriever
from src.retrieval.base import Chunk, Retriever
from src.retrieval.none import NoRetrieval


class _StubRetriever(Retriever):
    """Retriever that returns a single Chunk identifying which lane fired —
    useful to confirm AdaptiveRetriever dispatched to the right underlying."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.calls = 0

    def retrieve(self, question: str, k: int) -> list[Chunk]:
        self.calls += 1
        return [
            Chunk(chunk_id=f"{self.label}_chunk", book_name="stub", text=self.label, score=1.0)
        ]


def _build_lookup() -> dict[str, str]:
    return {
        "factoid q": "Simple",
        "vignette q": "Moderate",
        "complex case q": "Complex",
    }


def test_dispatches_to_correct_lane() -> None:
    lookup = _build_lookup()
    table = {
        "Simple": _StubRetriever("naive"),
        "Moderate": _StubRetriever("hybrid"),
        "Complex": _StubRetriever("multihop"),
    }
    r = AdaptiveRetriever(lookup, table)
    out_simple = r.retrieve("factoid q", k=1)
    out_moderate = r.retrieve("vignette q", k=1)
    out_complex = r.retrieve("complex case q", k=1)
    assert out_simple[0].text == "naive"
    assert out_moderate[0].text == "hybrid"
    assert out_complex[0].text == "multihop"
    assert r.dispatch_counts == {"Simple": 1, "Moderate": 1, "Complex": 1}
    assert r.unknown_question_count == 0


def test_unknown_question_falls_back_to_default() -> None:
    lookup = _build_lookup()
    table = {
        "Simple": _StubRetriever("naive"),
        "Moderate": _StubRetriever("hybrid"),
        "Complex": _StubRetriever("multihop"),
    }
    r = AdaptiveRetriever(lookup, table, default_bucket="Moderate")
    out = r.retrieve("question never seen before", k=1)
    assert out[0].text == "hybrid"
    assert r.unknown_question_count == 1
    assert r.dispatch_counts["Moderate"] == 1


def test_variant_b_with_no_retrieval() -> None:
    """Variant B routes Simple → NoRetrieval. The runner sees empty chunks
    and falls back to No-RAG prompt — verify NoRetrieval returns []."""
    lookup = _build_lookup()
    table = {
        "Simple": NoRetrieval(),
        "Moderate": _StubRetriever("multihop"),
        "Complex": _StubRetriever("multihop"),
    }
    r = AdaptiveRetriever(lookup, table)
    out = r.retrieve("factoid q", k=5)
    assert out == [], f"NoRetrieval should return empty list; got {out}"
    assert r.bucket_for("factoid q") == "Simple"


def test_default_bucket_must_exist_in_table() -> None:
    lookup = _build_lookup()
    table = {
        "Simple": _StubRetriever("naive"),
        "Moderate": _StubRetriever("hybrid"),
        # No Complex — but the lookup contains it, so construction should fail.
    }
    try:
        AdaptiveRetriever(lookup, table)
    except ValueError as e:
        assert "Complex" in str(e)
        return
    raise AssertionError("expected ValueError on missing routing-table entry")


def test_bucket_for_does_not_dispatch() -> None:
    lookup = _build_lookup()
    table = {
        "Simple": _StubRetriever("naive"),
        "Moderate": _StubRetriever("hybrid"),
        "Complex": _StubRetriever("multihop"),
    }
    r = AdaptiveRetriever(lookup, table)
    assert r.bucket_for("factoid q") == "Simple"
    # bucket_for is a peek; it should NOT increment dispatch counts
    assert r.dispatch_counts == {"Simple": 0, "Moderate": 0, "Complex": 0}


def test_real_data_lookup_join() -> None:
    """End-to-end check: build the lookup from the production
    `complexity_labels.parquet` joined to `medqa_4opt.parquet` question text.
    Then dispatch on 5 real test-split questions."""
    REPO = Path(__file__).resolve().parent.parent
    labels = pd.read_parquet(REPO / "data/processed/complexity_labels.parquet")
    md = pd.read_parquet(REPO / "data/processed/medqa_4opt.parquet")
    md = md.reset_index(drop=False).rename(columns={"index": "row_idx"})
    md["question_id"] = "medqa_" + md["row_idx"].astype(str)
    # `split` is in both — keep labels' version, only pull `question` from md
    joined = labels.merge(md[["question_id", "question"]], on="question_id")
    test = joined[joined.split == "test"]
    assert len(test) == 1273, f"expected 1273 test questions, got {len(test)}"

    lookup = dict(zip(test["question"], test["complexity"].astype(str)))
    table = {
        "Simple": _StubRetriever("naive"),
        "Moderate": _StubRetriever("hybrid"),
        "Complex": _StubRetriever("multihop"),
    }
    r = AdaptiveRetriever(lookup, table)

    sample = test.sample(n=5, random_state=0)
    for _, row in sample.iterrows():
        bucket = r.bucket_for(row["question"])
        assert bucket == row["complexity"], (
            f"mismatch on {row['question_id']}: lookup says {bucket} "
            f"but parquet says {row['complexity']}"
        )
        out = r.retrieve(row["question"], k=1)
        assert len(out) == 1


if __name__ == "__main__":
    tests = [
        test_dispatches_to_correct_lane,
        test_unknown_question_falls_back_to_default,
        test_variant_b_with_no_retrieval,
        test_default_bucket_must_exist_in_table,
        test_bucket_for_does_not_dispatch,
        test_real_data_lookup_join,
    ]
    for t in tests:
        t()
        print(f"  ✓ {t.__name__}")
    print(f"\nAll {len(tests)} adaptive retriever tests passed.")
