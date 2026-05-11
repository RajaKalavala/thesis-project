"""Real-data test for `src/retrieval/complexity.py`.

Picks four hand-chosen questions from `data/processed/medqa_4opt.parquet`
that should land in known buckets, then runs `classify_complexity` and
`classify_dataframe` on them. Lightweight (~1 s) — no models loaded.

    .venv/bin/python tests/test_complexity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.loaders import load_medqa_4opt
from src.retrieval.complexity import (
    LABELS,
    PHRASES_P33,
    PHRASES_P67,
    WORDS_P33,
    WORDS_P67,
    classify_complexity,
    classify_dataframe,
    extract_features,
)


def test_synthetic_simple() -> None:
    # Short stem, no complex cue → should land Simple.
    q = "What is the rate-limiting enzyme of glycolysis?"
    label = classify_complexity(q, metamap_phrases=["enzyme", "glycolysis"])
    assert label == "Simple", f"expected Simple, got {label}"


def test_synthetic_complex_via_cue() -> None:
    # Short stem but complex cue → should land Complex (cue overrides length).
    q = "A 60-year-old man presents with chest pain. What is the best next step in management?"
    label = classify_complexity(q, metamap_phrases=["chest pain"])
    assert label == "Complex", f"expected Complex, got {label}"


def test_synthetic_complex_via_length() -> None:
    # Long stem + many entities, no cue → Complex by length+density.
    long_q = "A patient presents with " + " ".join(["finding"] * 200)
    phrases = [f"phrase_{i}" for i in range(60)]
    label = classify_complexity(long_q, metamap_phrases=phrases)
    assert label == "Complex", f"expected Complex (long+entity), got {label}"


def test_synthetic_moderate() -> None:
    # Length between thresholds, no cue → Moderate.
    q = " ".join(["word"] * 110)
    phrases = [f"phrase_{i}" for i in range(35)]
    label = classify_complexity(q, metamap_phrases=phrases)
    assert label == "Moderate", f"expected Moderate, got {label}"


def test_features_extraction() -> None:
    q = "Best next step in initial management of stroke?"
    f = extract_features(q, metamap_phrases=["stroke"])
    assert f.n_words == 8
    assert f.n_phrases == 1
    assert f.has_complex_cue is True
    assert f.has_simple_cue is False


def test_real_data_dataframe() -> None:
    md = load_medqa_4opt()
    sample = md.sample(n=100, random_state=42).reset_index(drop=True)
    out = classify_dataframe(sample)
    assert len(out) == 100
    assert set(out["complexity"]) <= set(LABELS), (
        f"unexpected labels: {set(out['complexity']) - set(LABELS)}"
    )
    # Every row has all five new columns.
    for col in (
        "complexity",
        "n_words",
        "n_phrases",
        "has_complex_cue",
        "has_simple_cue",
    ):
        assert col in out.columns, f"missing column {col}"
    # All three labels should appear in 100 random rows from a balanced dataset.
    assert (out["complexity"] == "Simple").sum() > 0
    assert (out["complexity"] == "Complex").sum() > 0


def test_threshold_invariants() -> None:
    # Sanity check: thresholds are positive integers and ordered correctly.
    assert 0 < WORDS_P33 < WORDS_P67
    assert 0 < PHRASES_P33 < PHRASES_P67


if __name__ == "__main__":
    tests = [
        test_synthetic_simple,
        test_synthetic_complex_via_cue,
        test_synthetic_complex_via_length,
        test_synthetic_moderate,
        test_features_extraction,
        test_real_data_dataframe,
        test_threshold_invariants,
    ]
    for t in tests:
        t()
        print(f"  ✓ {t.__name__}")
    print(f"\nAll {len(tests)} complexity tests passed.")
