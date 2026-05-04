"""Prompt templates and answer parsing for the thesis.

Three template flavours (only `evidence_grounded` is used in Notebook 03):
- `evidence_grounded`: question + retrieved chunks + options → single-letter answer
- `no_rag`: question + options only → single-letter answer (EXP_01)
- `multi_hop`: decomposition prompt for EXP_05

Letter parsing is permissive: it grabs the first A–E in the response (case-insensitive)
because LLaMA sometimes adds preamble like "The answer is B." despite the instruction.
"""
from __future__ import annotations

import re
from typing import Iterable

EVIDENCE_GROUNDED_TEMPLATE = """You are a medical expert answering a USMLE multiple-choice question.

Use ONLY the evidence below to choose the best option. If the evidence is insufficient, still pick the single most plausible option based on the question. Do not explain your reasoning.

EVIDENCE:
{evidence_block}

QUESTION:
{question}

OPTIONS:
{options_block}

Output exactly one letter (A, B, C, D, or E). Nothing else."""


NO_RAG_TEMPLATE = """You are a medical expert answering a USMLE multiple-choice question.

QUESTION:
{question}

OPTIONS:
{options_block}

Output exactly one letter (A, B, C, D, or E). Nothing else."""


def format_evidence_block(chunks: Iterable[str]) -> str:
    """Render retrieved chunk texts as a numbered evidence block."""
    lines = []
    for i, text in enumerate(chunks, start=1):
        text = text.strip().replace("\n", " ")
        lines.append(f"[{i}] {text}")
    return "\n\n".join(lines)


def format_options_block(options: dict[str, str]) -> str:
    """Render `{"A": "...", "B": "..."}` as `A) ...` lines, sorted by letter."""
    return "\n".join(f"{k}) {v}" for k, v in sorted(options.items()))


def build_evidence_grounded_prompt(
    question: str,
    options: dict[str, str],
    chunk_texts: list[str],
) -> str:
    return EVIDENCE_GROUNDED_TEMPLATE.format(
        evidence_block=format_evidence_block(chunk_texts),
        question=question.strip(),
        options_block=format_options_block(options),
    )


def build_no_rag_prompt(question: str, options: dict[str, str]) -> str:
    return NO_RAG_TEMPLATE.format(
        question=question.strip(),
        options_block=format_options_block(options),
    )


_LETTER_RE = re.compile(r"\b([A-E])\b", re.IGNORECASE)


def parse_letter(response: str) -> str | None:
    """Return the first standalone A–E found in the response, uppercased.
    Returns None if no letter is found."""
    if not response:
        return None
    m = _LETTER_RE.search(response)
    return m.group(1).upper() if m else None
