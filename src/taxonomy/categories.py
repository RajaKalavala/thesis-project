"""The 6-category hallucination error taxonomy (EXP_13).

Single source of truth for the category names, definitions, and exemplars.
The labeller (`src/taxonomy/labeller.py`) loads these into the gpt-4o-mini
prompt; the analysis module (`src/taxonomy/analysis.py`) imports the
canonical ordering for cross-tab rows.

The 6 categories are the *workbook* categories from
[`docs/thesis-files/Raja Kalavala Final Thesis Project Sheet.xlsx`] Table 7,
locked at the proposal §7.8 stage. The descriptions below are tightened
versions of the proposal text — they're what gets sent to the labeller.

## How to decide between adjacent categories (rater guidance)

- **unsupported_diagnosis vs context_omission**: if the *gold* diagnosis is
  also unsupported by the retrieved chunks → context_omission (retrieval
  missed the right evidence). If chunks DO carry the gold answer but the
  LLM picked a different diagnosis with no chunk support → unsupported_diagnosis.

- **unsupported_treatment** is the management analogue of `unsupported_diagnosis`.

- **wrong_reasoning_chain**: chunks DO support the gold answer, AND the LLM
  appears to acknowledge them in its prose but draws an incorrect conclusion.
  This is the "the model can read the evidence but reasons wrong" bucket.

- **partial_evidence_misuse**: a chunk fragment fits the LLM's chosen answer
  superficially, but reading the chunk in context reveals the LLM is
  misinterpreting it. Distinct from `wrong_reasoning_chain` (where chunks
  support gold, not pred).

- **option_mismatch**: rare. The LLM's prose explanation argues for option
  X, but it outputs letter Y. Pure parsing / generation artefact, not a
  retrieval-level error.

- **context_omission**: the gold answer requires a fact none of the
  retrieved chunks contain. The model can't be expected to know it from
  the retrieved evidence; it's a retrieval failure.
"""
from __future__ import annotations

from dataclasses import dataclass

# Canonical ordering — used by analysis.py for cross-tab rows + Table 7 paste.
CATEGORY_ORDER: tuple[str, ...] = (
    "unsupported_diagnosis",
    "unsupported_treatment",
    "wrong_reasoning_chain",
    "partial_evidence_misuse",
    "option_mismatch",
    "context_omission",
)


@dataclass(frozen=True)
class CategoryDef:
    name: str
    short_description: str  # one sentence; used in the labeller prompt
    long_description: str  # paragraph; used in the rater guideline doc


CATEGORIES: dict[str, CategoryDef] = {
    "unsupported_diagnosis": CategoryDef(
        name="unsupported_diagnosis",
        short_description=(
            "The LLM picked a diagnosis (or a 'most likely cause') that none of the "
            "retrieved chunks supports. Use when the question_type is a diagnosis "
            "question and the LLM's chosen answer has no chunk-level grounding."
        ),
        long_description=(
            "Diagnosis-flavoured questions ('most likely diagnosis', 'most likely "
            "cause', 'most likely organism') where the chosen letter's content is "
            "not supported by anything the retriever surfaced. The LLM either "
            "fabricated the diagnosis or relied entirely on pretraining memorisation. "
            "Distinguish from `context_omission`: if the GOLD diagnosis is also not "
            "in the chunks, it's `context_omission` (retrieval missed it); if only "
            "the LLM's choice is unsupported and gold IS in the chunks, it's "
            "`unsupported_diagnosis` (model ignored relevant evidence)."
        ),
    ),
    "unsupported_treatment": CategoryDef(
        name="unsupported_treatment",
        short_description=(
            "The LLM picked a treatment/management option not supported by any "
            "retrieved chunk. Use for treatment, management, and 'next-step' "
            "questions where the chosen letter has no chunk-level grounding."
        ),
        long_description=(
            "Treatment- or management-flavoured questions ('best next step', "
            "'initial management', 'most appropriate treatment') where the chosen "
            "letter's content lacks chunk support. Same distinction vs "
            "`context_omission` as `unsupported_diagnosis`."
        ),
    ),
    "wrong_reasoning_chain": CategoryDef(
        name="wrong_reasoning_chain",
        short_description=(
            "The retrieved chunks support the GOLD answer, and the LLM appears to "
            "acknowledge them in its response, but it draws an incorrect inference "
            "to a different letter. The model can read the evidence but reasons "
            "incorrectly from it."
        ),
        long_description=(
            "The retrieval surfaced the right evidence. The LLM's response prose "
            "engages with the chunks but arrives at the wrong letter via faulty "
            "clinical reasoning, incorrect probability weighting between competing "
            "differentials, or a misapplied algorithm. The error is at the "
            "reasoning step, not at retrieval or perception."
        ),
    ),
    "partial_evidence_misuse": CategoryDef(
        name="partial_evidence_misuse",
        short_description=(
            "A fragment of a retrieved chunk superficially fits the LLM's chosen "
            "letter but the chunk in context does NOT support that choice. The "
            "LLM cherry-picked a phrase out of its proper context."
        ),
        long_description=(
            "Most common in long Multi-Hop chunks. The chunk discusses condition A "
            "in passing while explaining condition B; the LLM lifts the phrase "
            "about A and treats it as primary evidence. The chunk *taken as a "
            "whole* does not support the LLM's letter; only an isolated fragment "
            "does. Distinct from `wrong_reasoning_chain` because the misuse is at "
            "the chunk-comprehension level, not the inference level."
        ),
    ),
    "option_mismatch": CategoryDef(
        name="option_mismatch",
        short_description=(
            "The LLM's response text argues for one option but outputs a different "
            "letter. A pure parsing / generation artefact — not a retrieval or "
            "reasoning error per se."
        ),
        long_description=(
            "Rare. The LLM's free-text explanation clearly endorses option X, but "
            "the final letter it emits is Y. The retrieved chunks and reasoning may "
            "actually be correct; the failure is at the output-formatting step. "
            "Worth recording separately because it suggests a different mitigation "
            "(e.g., post-hoc letter-from-explanation parsing) than the other "
            "categories."
        ),
    ),
    "context_omission": CategoryDef(
        name="context_omission",
        short_description=(
            "None of the retrieved chunks contain the evidence needed for the gold "
            "answer. The model can't be expected to answer correctly from the "
            "retrieved evidence; this is a retrieval-level failure, not a "
            "generation-level one."
        ),
        long_description=(
            "Retrieval missed the right evidence entirely. The gold answer requires "
            "knowledge of a specific fact, finding, or guideline that none of the "
            "retrieved chunks references. Whatever letter the LLM picks, it's "
            "guessing — possibly from pretraining memorisation, possibly randomly. "
            "Phase 4's 'memorisation cases' overlap heavily with this category."
        ),
    ),
}


# Sanity: every name in CATEGORY_ORDER must have a CategoryDef.
assert set(CATEGORY_ORDER) == set(CATEGORIES.keys()), (
    f"CATEGORY_ORDER {CATEGORY_ORDER} does not match CATEGORIES keys "
    f"{tuple(CATEGORIES.keys())}"
)


def format_categories_for_prompt() -> str:
    """Render the 6 categories as a numbered block for the labeller prompt."""
    lines = []
    for i, name in enumerate(CATEGORY_ORDER, start=1):
        cd = CATEGORIES[name]
        lines.append(f"{i}. `{cd.name}` — {cd.short_description}")
    return "\n".join(lines)


def is_valid_category(label: str) -> bool:
    return label in CATEGORIES
