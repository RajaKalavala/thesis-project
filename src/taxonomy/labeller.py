"""gpt-4o-mini hallucination classifier (EXP_14).

Given a wrong-answer question + its retrieved chunks + the LLM's predicted
letter + gold letter, ask `gpt-4o-mini` (via OpenAI JSON-mode) to assign one
of the 6 categories from `src/taxonomy/categories.py`, plus a 1-2 sentence
rationale.

## Why gpt-4o-mini

- Per `plan.md §14`, the Phase 8 budget is ~$3 for ~157 question-arch labels.
  gpt-4o-mini at ~$0.15/1M input + ~$0.60/1M output tokens lands within
  that ceiling with margin.
- Different model family from generator (LLaMA via Groq), constructor
  (gpt-4o via OpenAI), and judge (Claude Sonnet 4.6) — but `gpt-4o-mini`
  shares the OpenAI family with the constructor. Acceptable methodologically
  because (a) we're labelling errors not grading correctness, (b) the
  constructor wrote the golden answers months earlier from different
  prompts and chunks, (c) mini and 4o are distinct models with different
  parameter counts and training cut-offs.

## Output schema (JSON-mode enforced)

```json
{
  "category": "one of CATEGORY_ORDER",
  "rationale": "1-2 sentence explanation citing chunk numbers or response text"
}
```

Parsing is tolerant: if `category` is missing or invalid, we record `null`
and the analysis pipeline drops those rows with a count of "labeller
parse failures".

## Caching + resumability

Each call is cached on `sha256(provider + model + temperature + prompt)` via
the existing `src/utils/cache.py`. Re-runs over the same question set are
free. The batch runner streams JSONL output and skips already-labelled
`(question_id, architecture)` rows.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from openai import OpenAI
from tqdm.auto import tqdm

from src.taxonomy.categories import (
    CATEGORIES,
    CATEGORY_ORDER,
    format_categories_for_prompt,
    is_valid_category,
)
from src.utils.cache import DEFAULT_CACHE_DIR, cache_get, cache_put

PROVIDER = "openai"
DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 400


_client_singleton: OpenAI | None = None


def _client() -> OpenAI:
    global _client_singleton
    if _client_singleton is None:
        import os
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to .env and "
                "`load_dotenv()` before calling the labeller."
            )
        _client_singleton = OpenAI(api_key=api_key)
    return _client_singleton


# ---------------------------------------------------------------------------
#  Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_MESSAGE = (
    "You are an expert medical-education examiner classifying the failure "
    "mode of a USMLE multiple-choice answer that was wrong. You will be "
    "given the question, the answer options, the chunks of textbook "
    "evidence the model retrieved, the model's predicted letter + response "
    "text, and the correct gold letter. Your job: pick the single "
    "category that best describes WHY the model got this question wrong, "
    "from the list below.\n\n"
    f"{format_categories_for_prompt()}\n\n"
    "Decision principles:\n"
    "- Read the retrieved chunks carefully. They are the model's only "
    "evidence beyond pretraining knowledge.\n"
    "- If the gold answer's supporting evidence is NOT in any chunk, the "
    "category is `context_omission` regardless of what the model picked.\n"
    "- If chunks DO carry the gold answer's evidence and the model's "
    "response engages with them but reasons wrong → `wrong_reasoning_chain`.\n"
    "- If chunks don't support the model's letter at all → "
    "`unsupported_diagnosis` (diagnosis question) or "
    "`unsupported_treatment` (treatment/management question).\n"
    "- If the model's response prose argues for one letter but it emits "
    "a different one → `option_mismatch`.\n"
    "- If a chunk fragment fits superficially but the chunk in context "
    "does not support the model's letter → `partial_evidence_misuse`.\n\n"
    "Output a JSON object with exactly two keys: `category` (a string "
    "from the list above) and `rationale` (1-2 sentence justification, "
    "citing chunk numbers like '[3]' or quoting from the response). No "
    "other keys, no preamble."
)


def build_user_message(
    question: str,
    options: dict[str, str],
    chunks: list[dict] | list[str],
    pred_letter: str,
    pred_text: str,
    gold_letter: str,
    *,
    chunk_text_char_limit: int = 1200,
) -> str:
    """Render the per-question payload for the labeller.

    Chunks can be a list of dicts (with `text` keys) or raw strings.
    Each chunk is truncated to ~1200 chars to keep the prompt under
    8k tokens even with k=15 Multi-Hop chunks.
    """
    chunk_lines = []
    for i, c in enumerate(chunks, start=1):
        text = c["text"] if isinstance(c, dict) else str(c)
        text = text.replace("\n", " ").strip()
        if len(text) > chunk_text_char_limit:
            text = text[:chunk_text_char_limit] + "..."
        chunk_lines.append(f"[{i}] {text}")
    chunk_block = "\n\n".join(chunk_lines) if chunk_lines else "(no chunks retrieved — No-RAG)"
    options_block = "\n".join(f"({k}) {v}" for k, v in sorted(options.items()))
    return (
        f"QUESTION:\n{question.strip()}\n\n"
        f"OPTIONS:\n{options_block}\n\n"
        f"RETRIEVED EVIDENCE CHUNKS:\n{chunk_block}\n\n"
        f"MODEL'S PREDICTED LETTER: {pred_letter}\n"
        f"MODEL'S RESPONSE TEXT: {pred_text.strip()[:600]}\n\n"
        f"GOLD CORRECT LETTER: {gold_letter}\n\n"
        "Classify the error and produce the JSON object."
    )


# ---------------------------------------------------------------------------
#  Single-question labeller
# ---------------------------------------------------------------------------


@dataclass
class TaxonomyLabel:
    question_id: str
    architecture: str
    gold_letter: str
    pred_letter: str
    category: str | None  # one of CATEGORY_ORDER, or None on parse failure
    rationale: str
    raw_response: str  # full model output for audit
    was_cached: bool
    latency_s: float
    model: str
    parse_ok: bool

    def to_jsonl_row(self) -> dict:
        return asdict(self)


def classify_one(
    *,
    question_id: str,
    architecture: str,
    question: str,
    options: dict[str, str],
    chunks: list[dict] | list[str],
    pred_letter: str,
    pred_text: str,
    gold_letter: str,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
) -> TaxonomyLabel:
    """Label one wrong-answer question with a category + rationale.

    Always returns a `TaxonomyLabel`; on parse failure `category=None` and
    `parse_ok=False`. The raw OpenAI response is preserved for the audit
    trail.
    """
    user_msg = build_user_message(
        question=question,
        options=options,
        chunks=chunks,
        pred_letter=pred_letter,
        pred_text=pred_text,
        gold_letter=gold_letter,
    )
    # cache key: combine system + user message (both vary per question; system
    # is constant within a labeller version, but we include it so prompt edits
    # invalidate the cache cleanly)
    cache_key_payload = f"SYSTEM:\n{_SYSTEM_MESSAGE}\n\nUSER:\n{user_msg}"

    if use_cache:
        cached = cache_get(PROVIDER, model, temperature, cache_key_payload, cache_dir=cache_dir)
        if cached is not None:
            return _payload_to_label(cached, question_id, architecture, gold_letter, pred_letter, model, was_cached=True)

    t0 = time.time()
    resp = _client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_MESSAGE},
            {"role": "user", "content": user_msg},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    latency = time.time() - t0
    text = resp.choices[0].message.content or ""

    payload = {
        "text": text,
        "latency_s": latency,
        "model": model,
        "temperature": temperature,
        "usage": {
            "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
            "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
            "total_tokens": resp.usage.total_tokens if resp.usage else None,
        },
    }
    if use_cache:
        cache_put(PROVIDER, model, temperature, cache_key_payload, payload, cache_dir=cache_dir)

    return _payload_to_label(payload, question_id, architecture, gold_letter, pred_letter, model, was_cached=False)


def _payload_to_label(
    payload: dict,
    question_id: str,
    architecture: str,
    gold_letter: str,
    pred_letter: str,
    model: str,
    *,
    was_cached: bool,
) -> TaxonomyLabel:
    raw = payload.get("text", "")
    category: str | None = None
    rationale = ""
    parse_ok = False
    try:
        data = json.loads(raw)
        cat = data.get("category")
        rat = data.get("rationale", "")
        if isinstance(cat, str) and is_valid_category(cat):
            category = cat
            rationale = str(rat)[:1000]
            parse_ok = True
        else:
            rationale = f"INVALID_CATEGORY: {cat!r}"
    except (json.JSONDecodeError, TypeError, AttributeError) as e:
        rationale = f"PARSE_FAILURE: {e}"

    return TaxonomyLabel(
        question_id=question_id,
        architecture=architecture,
        gold_letter=gold_letter,
        pred_letter=pred_letter,
        category=category,
        rationale=rationale,
        raw_response=raw,
        was_cached=was_cached,
        latency_s=float(payload.get("latency_s", 0.0)),
        model=model,
        parse_ok=parse_ok,
    )


# ---------------------------------------------------------------------------
#  Batch labeller with resumability
# ---------------------------------------------------------------------------


def _load_completed_keys(output_path: Path) -> set[tuple[str, str]]:
    if not output_path.exists():
        return set()
    done: set[tuple[str, str]] = set()
    for line in output_path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            done.add((row["question_id"], row["architecture"]))
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def run_taxonomy_batch(
    rows: Iterable[dict],
    output_path: Path,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    progress: bool = True,
) -> dict:
    """Label a batch of wrong-answer questions; stream JSONL to `output_path`.

    Each input row dict must carry:
      - `question_id`, `architecture`, `question`, `options` (dict),
        `chunks` (list of dicts or strings), `pred_letter`, `pred_text`,
        `gold_letter`.

    Returns a summary dict including parse-failure count.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _load_completed_keys(output_path)
    if completed:
        print(f"[taxonomy] resuming — {len(completed)} (qid, arch) rows already done")

    rows_list = list(rows)
    n_written = 0
    n_skipped = 0
    n_parse_failures = 0
    n_cached = 0
    t_start = time.time()

    iterator = tqdm(rows_list, desc="taxonomy", disable=not progress)
    with output_path.open("a", encoding="utf-8") as f:
        for row in iterator:
            key = (row["question_id"], row["architecture"])
            if key in completed:
                n_skipped += 1
                continue
            label = classify_one(
                question_id=row["question_id"],
                architecture=row["architecture"],
                question=row["question"],
                options=row["options"],
                chunks=row["chunks"],
                pred_letter=row["pred_letter"],
                pred_text=row["pred_text"],
                gold_letter=row["gold_letter"],
                model=model,
                temperature=temperature,
            )
            f.write(json.dumps(label.to_jsonl_row()) + "\n")
            f.flush()
            n_written += 1
            if not label.parse_ok:
                n_parse_failures += 1
            if label.was_cached:
                n_cached += 1

    return {
        "output_path": str(output_path),
        "model": model,
        "n_rows_written_this_run": n_written,
        "n_rows_already_done": n_skipped,
        "n_parse_failures": n_parse_failures,
        "n_cache_hits_this_run": n_cached,
        "wall_time_s_this_run": time.time() - t_start,
    }
