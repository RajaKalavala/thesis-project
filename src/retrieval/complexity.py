"""Rule-based question-complexity classifier (EXP_06).

This module assigns each MedQA question one of three complexity labels —
``Simple``, ``Moderate``, ``Complex`` — using a hand-written rule over four
features available at notebook time (no LLM calls):

1. ``n_words``    — word count of the question stem
2. ``n_phrases``  — count of MetaMap medical phrases (from `medqa_4opt.parquet`)
3. ``has_complex_cue`` — presence of clinical-decision cue phrases
   (e.g., "best next step in management", "most appropriate initial …")
4. ``has_simple_cue``  — presence of factoid / mechanism cue phrases
   (e.g., "mechanism of action", "rate-limiting", "derived from")

Why a rule and not a classifier: the labels feed Phase 5 EXP_07 adaptive
routing. A learned classifier would (a) need its own training data, (b)
introduce a model-of-a-model dependency, and (c) make the routing decision
opaque to a viva examiner. A transparent rule based on length + entity
density + cue words is defensible end-to-end.

Threshold calibration: the word/phrase cutoffs are anchored to the 33rd /
67th percentiles of the **full MedQA-US 4-option dataset** (12,723 questions
in `data/processed/medqa_4opt.parquet`, computed 2026-05-10):

    n_words   p33=93   p67=133
    n_phrases p33=28   p67=41

This produces a roughly even three-way split (~30 / 33 / 38 %). Picking
percentiles of the population avoids over-fitting to any one stratum
(step1 vs step2&3) and keeps the buckets balanced enough for per-bucket
analysis in EXP_07.

Caveat for the methodology footnote: MedQA is overwhelmingly clinical
vignettes — even the "Simple" bucket is mostly *short-vignette* questions,
not pure factoids. The proposal's terminology (Simple / Moderate / Complex)
is preserved for plan-alignment but the rule is honestly a *length + entity
density + cue-word* proxy for complexity, not a deep-semantic measure.

Routing tables (EXP_07, both variants tested):

    Variant A (proposal):     Simple → Naive · Moderate → Hybrid · Complex → Multi-Hop
    Variant B (data-driven):  Simple → No-RAG · Moderate → Multi-Hop · Complex → Multi-Hop
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd

# Percentile-anchored thresholds (computed on the full MedQA-US 4-opt set,
# 12,723 questions, 2026-05-10).
WORDS_P33 = 93
WORDS_P67 = 133
PHRASES_P33 = 28
PHRASES_P67 = 41

# Cue phrases — lowercase substring match on the question stem.
COMPLEX_CUES: tuple[str, ...] = (
    "best next step",
    "next step in management",
    "next best step",
    "most appropriate management",
    "most appropriate next",
    "most appropriate initial",
    "initial management",
    "best initial",
    "further management",
)

SIMPLE_CUES: tuple[str, ...] = (
    "mechanism of action",
    "rate-limiting",
    "rate limiting",
    "class of drug",
    "type of receptor",
    "derived from",
    "most likely originated",
    "best describes the",
    "pathway is",
    "is the function",
    "is involved in",
)

LABELS = ("Simple", "Moderate", "Complex")


@dataclass(frozen=True)
class ComplexityFeatures:
    """Intermediate feature record produced before bucket assignment.

    Stored alongside the label in the output parquet so a reviewer can see
    *why* a question landed in a bucket without re-running the rule.
    """

    n_words: int
    n_phrases: int
    has_complex_cue: bool
    has_simple_cue: bool


def _lower_strip(question: str) -> str:
    # Collapse internal whitespace + lowercase; cheap normalisation so cue-word
    # matching is robust to formatting variation in the source data.
    return re.sub(r"\s+", " ", question).strip().lower()


def extract_features(
    question: str, metamap_phrases: Iterable[str] | None = None
) -> ComplexityFeatures:
    q_lower = _lower_strip(question)
    n_words = len(question.split())
    n_phrases = sum(1 for _ in (metamap_phrases or ()))
    has_complex = any(c in q_lower for c in COMPLEX_CUES)
    has_simple = any(c in q_lower for c in SIMPLE_CUES)
    return ComplexityFeatures(
        n_words=n_words,
        n_phrases=n_phrases,
        has_complex_cue=has_complex,
        has_simple_cue=has_simple,
    )


def classify_features(f: ComplexityFeatures) -> str:
    """Apply the rule to an already-extracted feature record.

    Order matters: Complex cues override length-based bucketing because a
    short stem with "best next step in management" is asking for clinical
    decision-making and benefits from multi-hop retrieval.
    """
    long_stem = f.n_words >= WORDS_P67
    short_stem = f.n_words <= WORDS_P33
    high_entity = f.n_phrases >= PHRASES_P67
    low_entity = f.n_phrases <= PHRASES_P33

    if f.has_complex_cue or (long_stem and high_entity):
        return "Complex"
    if (short_stem and low_entity) or (f.has_simple_cue and short_stem):
        return "Simple"
    return "Moderate"


def classify_complexity(
    question: str, metamap_phrases: Iterable[str] | None = None
) -> str:
    """Single-question entry point — extract features and classify."""
    return classify_features(extract_features(question, metamap_phrases))


def _coerce_metamap(value: Any) -> list[str]:
    """Tolerate the two storage formats found in `medqa_4opt.parquet`:

    - ``metamap_phrases`` column already-decoded list (some loaders)
    - ``metamap_phrases_json`` column JSON-string list (the parquet on disk)
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
            return decoded if isinstance(decoded, list) else []
        except json.JSONDecodeError:
            return []
    return []


def classify_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised classification over a MedQA dataframe.

    Expects columns:
        - ``question``                   (str)
        - ``metamap_phrases_json`` OR ``metamap_phrases`` (one of the two)

    Returns a new dataframe with the same index plus columns:
        ``complexity``, ``n_words``, ``n_phrases``,
        ``has_complex_cue``, ``has_simple_cue``.
    """
    if "metamap_phrases" in df.columns:
        phrases = df["metamap_phrases"].apply(_coerce_metamap)
    elif "metamap_phrases_json" in df.columns:
        phrases = df["metamap_phrases_json"].apply(_coerce_metamap)
    else:
        raise KeyError(
            "classify_dataframe expects 'metamap_phrases' or "
            "'metamap_phrases_json' column"
        )

    features = [
        extract_features(q, p) for q, p in zip(df["question"], phrases, strict=False)
    ]
    out = df.copy()
    out["n_words"] = [f.n_words for f in features]
    out["n_phrases"] = [f.n_phrases for f in features]
    out["has_complex_cue"] = [f.has_complex_cue for f in features]
    out["has_simple_cue"] = [f.has_simple_cue for f in features]
    out["complexity"] = [classify_features(f) for f in features]
    return out
