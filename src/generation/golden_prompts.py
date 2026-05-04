"""Three-pass golden-set construction prompts (Notebook 04 / Phase 3).

Constructor: `openai/gpt-oss-120b` via Groq (per `plan.md §0 #10`, recalibrated
2026-05-04 from `gpt-4o`). The model is a *reasoning* model whose internal
"thinking" tokens are hidden by Groq but counted in `completion_tokens` —
this means **strict JSON mode (`response_format`) does not work**; we use
**instructed JSON** instead, with `parse_json_with_reasoning_leak` as the
robust parser that recovers the JSON block even if reasoning preamble leaks.

Prompts:
- `pass1_evidence_selection`     — select 1–3 strongest passages, extract keywords
- `pass2_reference_answer`       — write reference answer + explanation + check points
- `pass3_validation`             — score 0–5 and decide accept/needs_review/reject
"""
from __future__ import annotations

import json
import re
from typing import Any

# ----------------------------------------------------------------------------
# JSON parser with reasoning-leak recovery
# ----------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", re.DOTALL)


def parse_json_with_reasoning_leak(text: str) -> dict | None:
    """Robust JSON parser for reasoning-model output.

    Tries strict JSON first (clean instructed output). If that fails because
    reasoning tokens leaked into the visible text, finds the first balanced
    `{...}` block and parses that. Returns None if no JSON can be recovered.
    """
    if not text:
        return None
    text = text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Reasoning-leak fallback: find the first balanced {...} block
    m = _JSON_BLOCK_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


# ----------------------------------------------------------------------------
# Prompt rendering helpers
# ----------------------------------------------------------------------------

def _format_candidates(candidates: list[dict]) -> str:
    """Render the candidate passages block. `candidates` is a list of dicts
    with keys: chunk_id, text, book_name, rrf_score (optional)."""
    lines = []
    for i, c in enumerate(candidates, start=1):
        text = c["text"].strip().replace("\n", " ")
        lines.append(f"[{i}] chunk_id={c['chunk_id']}  book={c.get('book_name','')}\n    {text}")
    return "\n\n".join(lines)


def _format_options(options: dict[str, str]) -> str:
    return "\n".join(f"{k}) {v}" for k, v in sorted(options.items()))


# ----------------------------------------------------------------------------
# Pass 1 — Evidence selection (temp 0)
# ----------------------------------------------------------------------------

PASS1_TEMPLATE = """You are a careful medical editor building a USMLE evidence set.

You are given:
- A multiple-choice question and its CORRECT answer letter and text
- Up to 10 retrieved candidate passages from medical textbooks

Your task: pick the 1–3 passages that BEST support the correct answer, extract 3–8 specific medical keywords that link the passages to the answer, and decide whether the evidence is sufficient to write a faithful reference answer.

QUESTION:
{question}

OPTIONS:
{options_block}

CORRECT ANSWER: {correct_letter}) {correct_answer_text}

CANDIDATE PASSAGES:
{candidates_block}

Output ONLY a JSON object — no preamble, no explanation, no thinking. Use exactly these keys:
{{
  "selected_chunk_ids": [list of 1 to 3 chunk_id strings, taken from the candidates above],
  "evidence_keywords": [list of 3 to 8 specific medical terms that appear in the selected passages and tie them to the answer],
  "is_evidence_sufficient": true or false,
  "reasoning_one_line": "one short sentence explaining the choice"
}}"""


def build_pass1_prompt(
    question: str,
    options: dict[str, str],
    correct_letter: str,
    correct_answer_text: str,
    candidates: list[dict],
) -> str:
    return PASS1_TEMPLATE.format(
        question=question.strip(),
        options_block=_format_options(options),
        correct_letter=correct_letter,
        correct_answer_text=correct_answer_text,
        candidates_block=_format_candidates(candidates),
    )


# ----------------------------------------------------------------------------
# Pass 2 — Reference answer + explanation (temp 0.2)
# ----------------------------------------------------------------------------

PASS2_TEMPLATE = """You are writing the reference answer for a USMLE multiple-choice question. Your output will be used as ground truth for evaluating retrieval-augmented language models, so it must be FAITHFUL to the gold context — do not introduce facts that are not in the passages below.

QUESTION:
{question}

OPTIONS:
{options_block}

CORRECT ANSWER: {correct_letter}) {correct_answer_text}

GOLD CONTEXT (the only passages you may rely on):
{gold_context_block}

Output ONLY a JSON object — no preamble, no explanation, no thinking. Use exactly these keys:
{{
  "reference_answer": "one sentence stating the correct answer with its key clinical justification",
  "reference_explanation": "3 to 6 sentences grounded in the gold context, explaining WHY the correct answer is correct",
  "why_other_options_are_less_suitable": "2 to 4 sentences contrasting the correct answer with each distractor (use the option text, not just letters)",
  "hallucination_check_points": [list of 3 to 6 atomic factual claims that any correct generated answer MUST cover; each item is a short string],
  "question_type": "one of: diagnosis, treatment, mechanism, management, other",
  "requires_multihop": "yes or no — yes ONLY if answering requires combining at least 2 distinct facts from at least 2 different passages"
}}"""


def build_pass2_prompt(
    question: str,
    options: dict[str, str],
    correct_letter: str,
    correct_answer_text: str,
    gold_context: list[dict],
) -> str:
    return PASS2_TEMPLATE.format(
        question=question.strip(),
        options_block=_format_options(options),
        correct_letter=correct_letter,
        correct_answer_text=correct_answer_text,
        gold_context_block=_format_candidates(gold_context),
    )


# ----------------------------------------------------------------------------
# Pass 3 — Validation (temp 0)
# ----------------------------------------------------------------------------

PASS3_TEMPLATE = """You are a senior medical editor validating a draft golden-set entry. Score it strictly — this entry will be used as ground truth in evaluating language models, so any flaw propagates.

QUESTION:
{question}

CORRECT ANSWER: {correct_letter}) {correct_answer_text}

GOLD CONTEXT:
{gold_context_block}

DRAFT REFERENCE ANSWER:
{reference_answer}

DRAFT REFERENCE EXPLANATION:
{reference_explanation}

Output ONLY a JSON object — no preamble, no explanation, no thinking. Use exactly these keys:
{{
  "evidence_relevance_score": integer 0 to 5 (does the gold context actually support the answer?),
  "faithfulness_score": integer 0 to 5 (is the reference explanation grounded in the gold context with no hallucinated facts?),
  "explanation_quality_score": integer 0 to 5 (is the explanation clinically accurate and clearly written?),
  "hallucination_risk": "one of: low, medium, high",
  "final_status": "one of: accepted, needs_review, rejected",
  "rejection_reason_if_any": "empty string if accepted, otherwise one short sentence"
}}"""


def build_pass3_prompt(
    question: str,
    correct_letter: str,
    correct_answer_text: str,
    gold_context: list[dict],
    reference_answer: str,
    reference_explanation: str,
) -> str:
    return PASS3_TEMPLATE.format(
        question=question.strip(),
        correct_letter=correct_letter,
        correct_answer_text=correct_answer_text,
        gold_context_block=_format_candidates(gold_context),
        reference_answer=reference_answer.strip(),
        reference_explanation=reference_explanation.strip(),
    )


# ----------------------------------------------------------------------------
# Schema validators — minimal type / key checks
# ----------------------------------------------------------------------------

def validate_pass1(d: dict) -> tuple[bool, str]:
    """Return (ok, reason). Validates Pass 1 output structure."""
    if not isinstance(d, dict):
        return False, "not a dict"
    if not isinstance(d.get("selected_chunk_ids"), list) or not (1 <= len(d["selected_chunk_ids"]) <= 3):
        return False, "selected_chunk_ids must be a list of 1-3"
    if not all(isinstance(x, str) for x in d["selected_chunk_ids"]):
        return False, "selected_chunk_ids must be strings"
    if not isinstance(d.get("evidence_keywords"), list) or not (3 <= len(d["evidence_keywords"]) <= 8):
        return False, "evidence_keywords must be a list of 3-8"
    if not isinstance(d.get("is_evidence_sufficient"), bool):
        return False, "is_evidence_sufficient must be bool"
    return True, ""


def validate_pass2(d: dict) -> tuple[bool, str]:
    if not isinstance(d, dict):
        return False, "not a dict"
    required_str_fields = ["reference_answer", "reference_explanation",
                            "why_other_options_are_less_suitable", "question_type", "requires_multihop"]
    for f in required_str_fields:
        if not isinstance(d.get(f), str) or not d[f].strip():
            return False, f"{f} must be a non-empty string"
    if d["question_type"].lower() not in {"diagnosis", "treatment", "mechanism", "management", "other"}:
        return False, f"question_type {d['question_type']!r} not in allowed set"
    if d["requires_multihop"].lower() not in {"yes", "no"}:
        return False, f"requires_multihop {d['requires_multihop']!r} must be 'yes' or 'no'"
    if not isinstance(d.get("hallucination_check_points"), list) or len(d["hallucination_check_points"]) < 3:
        return False, "hallucination_check_points must be a list of >= 3"
    return True, ""


def validate_pass3(d: dict) -> tuple[bool, str]:
    if not isinstance(d, dict):
        return False, "not a dict"
    for f in ["evidence_relevance_score", "faithfulness_score", "explanation_quality_score"]:
        v = d.get(f)
        if not isinstance(v, int) or not (0 <= v <= 5):
            return False, f"{f} must be int 0-5, got {v!r}"
    if d.get("hallucination_risk") not in {"low", "medium", "high"}:
        return False, f"hallucination_risk {d.get('hallucination_risk')!r} invalid"
    if d.get("final_status") not in {"accepted", "needs_review", "rejected"}:
        return False, f"final_status {d.get('final_status')!r} invalid"
    return True, ""
