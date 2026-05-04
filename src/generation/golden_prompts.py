"""Three-pass golden-set construction prompts (Phase 3 / Notebook 04).

Constructor: `gpt-4o` via OpenAI API (locked 2026-05-04 after the gpt-oss-120b
A/B comparison — see `docs/output_notes/04_notebook_output.md`).

Prompts (this module):
- `pass1_prompt(...)` — evidence selection: structured `selected_chunks`,
  `best_gold_context` (verbatim concatenation by the constructor), keywords,
  sufficiency flag.
- `pass2_prompt(...)` — reference answer + explanation, with a tightened
  `requires_multihop` definition to address the over-labelling observed in
  both A/B pilots.
- `pass3_prompt(...)` — validation: `answer_match` boolean + 0-5 scores +
  final status.

Pipeline pattern (Notebook 04):
- Each pass reads the previous stage's JSONL and writes the next, so re-running
  Pass 3 with a tightened prompt only re-bills Pass 3 (Passes 1 & 2 stay
  cached on disk via the disk cache).

JSON parsing is via `parse_json_with_reasoning_leak` — the parser strips
markdown fences and recovers from reasoning-model preamble leak. We use
**instructed JSON** (no strict `response_format`); GPT-4o follows the schema
reliably without strict mode.
"""
from __future__ import annotations

import json
import re

# ----------------------------------------------------------------------------
# JSON parser with reasoning-leak recovery
# ----------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", re.DOTALL)


def parse_json_with_reasoning_leak(text: str) -> dict | None:
    """Robust JSON parser. Handles clean output, markdown fences, and
    reasoning-model preamble leak. Returns None if no JSON can be recovered.
    """
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _JSON_BLOCK_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


# ----------------------------------------------------------------------------
# Rendering helpers — used by the Notebook 04 cells directly
# ----------------------------------------------------------------------------

def format_options(opts: dict[str, str]) -> str:
    """Render `{"A": "...", "B": "..."}` as `A) ...` lines."""
    return "\n".join(f"{k}) {v}" for k, v in sorted(opts.items()))


def format_candidates(cands: list[dict]) -> str:
    """Render a list of candidate-chunk dicts as numbered passage blocks.
    Each candidate dict needs `chunk_id`, `book_name`, `text`. `rank` is
    optional — falls back to enumeration order.
    """
    blocks = []
    for i, c in enumerate(cands, start=1):
        rank = c.get("rank", i)
        blocks.append(
            f"[{rank}] chunk_id={c['chunk_id']} | book={c['book_name']}\n{c['text']}"
        )
    return "\n\n".join(blocks)


# ----------------------------------------------------------------------------
# Pass 1 — Evidence selection (temp 0)
# ----------------------------------------------------------------------------

PASS1_PROMPT = """You are a medical evidence verification assistant.

You will receive a USMLE-style multiple-choice question, the verified correct answer, and 10 candidate textbook passages retrieved from a medical knowledge base.

Task:
- Identify which passages directly support the correct answer.
- Reject passages that are irrelevant, too general, or insufficient on their own.
- Return STRICT JSON only — no prose, no markdown.

Question:
{question}

Options:
{options}

Correct answer ({answer_idx}): {correct_answer}

Candidate passages:
{candidates}

Return JSON with this exact schema:
{{
  "selected_chunks": [
    {{"chunk_id": "<from candidate>", "book_name": "<from candidate>", "support_level": "strong|moderate|weak", "reason": "<1 sentence>"}}
  ],
  "best_gold_context": "<concatenated text of the 1-3 strongest selected passages, verbatim>",
  "evidence_keywords": ["<medical term>", ...],
  "is_evidence_sufficient": true|false,
  "review_note": "<short note if anything is concerning>"
}}"""


def pass1_prompt(
    question: str,
    options: dict[str, str],
    answer_idx: str,
    correct_answer: str,
    candidates: list[dict],
) -> str:
    return PASS1_PROMPT.format(
        question=question.strip(),
        options=format_options(options),
        answer_idx=answer_idx,
        correct_answer=correct_answer,
        candidates=format_candidates(candidates),
    )


# ----------------------------------------------------------------------------
# Pass 2 — Reference answer + explanation (temp 0.2)
# ----------------------------------------------------------------------------

PASS2_PROMPT = """You are creating a reference answer for a medical question-answering evaluation dataset.

Use ONLY the provided question, correct answer, and gold textbook evidence. Do not introduce medical claims that the evidence does not support. Write in clear academic English.

For the `requires_multihop` field: label "yes" ONLY when answering requires combining ≥2 distinct facts from ≥2 different gold passages AND the answer cannot be inferred from any single passage alone. If a single passage supports the answer, label "no".

Question:
{question}

Options:
{options}

Correct answer ({answer_idx}): {correct_answer}

Gold evidence:
{gold_context}

Return STRICT JSON only — no prose, no markdown — with this schema:
{{
  "reference_answer": "<single-sentence answer that names the correct option>",
  "reference_explanation": "<3-6 sentence explanation grounded in the evidence>",
  "why_other_options_are_less_suitable": "<1-3 sentences>",
  "hallucination_check_points": ["<a claim a faithful answer must support>", ...],
  "question_type": "diagnosis|treatment|mechanism|management|other",
  "requires_multihop": "yes|no"
}}"""


def pass2_prompt(
    question: str,
    options: dict[str, str],
    answer_idx: str,
    correct_answer: str,
    gold_context: str,
) -> str:
    return PASS2_PROMPT.format(
        question=question.strip(),
        options=format_options(options),
        answer_idx=answer_idx,
        correct_answer=correct_answer,
        gold_context=gold_context.strip(),
    )


# ----------------------------------------------------------------------------
# Pass 3 — Validation (temp 0)
# ----------------------------------------------------------------------------

PASS3_PROMPT = """You are validating a golden dataset sample for medical RAG evaluation.

Check whether:
1. The reference answer matches the original correct answer.
2. The explanation is supported by the gold evidence.
3. The evidence is genuinely relevant to the question.
4. There are no unsupported medical claims.
5. The sample is suitable for RAGAS evaluation.

Question: {question}
Correct answer ({answer_idx}): {correct_answer}
Gold evidence: {gold_context}
Reference answer: {reference_answer}
Reference explanation: {reference_explanation}

Return STRICT JSON only:
{{
  "answer_match": true|false,
  "evidence_relevance_score": 0-5,
  "faithfulness_score": 0-5,
  "explanation_quality_score": 0-5,
  "hallucination_risk": "low|medium|high",
  "final_status": "accepted|needs_review|rejected",
  "reason": "<short>"
}}"""


def pass3_prompt(
    question: str,
    answer_idx: str,
    correct_answer: str,
    gold_context: str,
    reference_answer: str,
    reference_explanation: str,
) -> str:
    return PASS3_PROMPT.format(
        question=question.strip(),
        answer_idx=answer_idx,
        correct_answer=correct_answer,
        gold_context=gold_context.strip(),
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
    sc = d.get("selected_chunks")
    if not isinstance(sc, list) or not (1 <= len(sc) <= 5):
        return False, f"selected_chunks must be list of 1-5"
    for i, item in enumerate(sc):
        if not isinstance(item, dict):
            return False, f"selected_chunks[{i}] must be dict"
        for f in ("chunk_id", "book_name", "support_level", "reason"):
            if not isinstance(item.get(f), str):
                return False, f"selected_chunks[{i}].{f} must be string"
        if item["support_level"].lower() not in {"strong", "moderate", "weak"}:
            return False, f"selected_chunks[{i}].support_level invalid"
    if not isinstance(d.get("best_gold_context"), str) or not d["best_gold_context"].strip():
        return False, "best_gold_context must be non-empty string"
    if not isinstance(d.get("evidence_keywords"), list) or not (3 <= len(d["evidence_keywords"]) <= 10):
        return False, "evidence_keywords must be list of 3-10"
    if not isinstance(d.get("is_evidence_sufficient"), bool):
        return False, "is_evidence_sufficient must be bool"
    return True, ""


def validate_pass2(d: dict) -> tuple[bool, str]:
    if not isinstance(d, dict):
        return False, "not a dict"
    str_fields = ["reference_answer", "reference_explanation",
                  "why_other_options_are_less_suitable", "question_type", "requires_multihop"]
    for f in str_fields:
        if not isinstance(d.get(f), str) or not d[f].strip():
            return False, f"{f} must be non-empty string"
    if d["question_type"].lower() not in {"diagnosis", "treatment", "mechanism", "management", "other"}:
        return False, f"question_type {d['question_type']!r} invalid"
    if d["requires_multihop"].lower() not in {"yes", "no"}:
        return False, f"requires_multihop must be yes/no"
    if not isinstance(d.get("hallucination_check_points"), list) or len(d["hallucination_check_points"]) < 1:
        return False, "hallucination_check_points must be a non-empty list"
    return True, ""


def validate_pass3(d: dict) -> tuple[bool, str]:
    if not isinstance(d, dict):
        return False, "not a dict"
    if not isinstance(d.get("answer_match"), bool):
        return False, "answer_match must be bool"
    for f in ("evidence_relevance_score", "faithfulness_score", "explanation_quality_score"):
        v = d.get(f)
        if not isinstance(v, int) or not (0 <= v <= 5):
            return False, f"{f} must be int 0-5"
    if d.get("hallucination_risk") not in {"low", "medium", "high"}:
        return False, f"hallucination_risk invalid"
    if d.get("final_status") not in {"accepted", "needs_review", "rejected"}:
        return False, f"final_status invalid"
    return True, ""
