"""OpenAI client wrapper with disk-cache.

Mirrors `src/generation/groq_client.py` for the OpenAI API. Used by the
`gpt-4o` variant of the golden-set pilot ([`notebooks/04_golden_dataset_gpt4o.ipynb`])
so we can A/B the open-weights constructor (`gpt-oss-120b` via Groq) against
the original locked plan (`gpt-4o` via OpenAI).

Cache key = sha256(provider + model + temperature + prompt) — separate provider
namespace from Groq, so identical prompts are cached independently per backend.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.utils.cache import DEFAULT_CACHE_DIR, cache_get, cache_put

DEFAULT_CONSTRUCTOR_MODEL = "gpt-4o"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 2048
PROVIDER = "openai"


_client_singleton: OpenAI | None = None


def _client() -> OpenAI:
    global _client_singleton
    if _client_singleton is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to .env (repo root) and run "
                "`from dotenv import load_dotenv; load_dotenv()` before calling this."
            )
        _client_singleton = OpenAI(api_key=api_key)
    return _client_singleton


def openai_complete_full(
    prompt: str,
    model: str = DEFAULT_CONSTRUCTOR_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
) -> tuple[dict, bool]:
    """Run a single OpenAI chat completion and return the full payload.

    Returned dict shape (mirrors `groq_complete_full`):
        {"text": str, "latency_s": float, "model": str, "temperature": float,
         "usage": {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}}
    """
    if use_cache:
        cached = cache_get(PROVIDER, model, temperature, prompt, cache_dir=cache_dir)
        if cached is not None:
            return cached, True

    t0 = time.time()
    resp = _client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    latency = time.time() - t0
    text = resp.choices[0].message.content or ""

    payload: dict[str, Any] = {
        "text": text,
        "latency_s": latency,
        "model": model,
        "temperature": temperature,
        "usage": {
            "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
            "completion_tokens": getattr(resp.usage, "completion_tokens", None),
            "total_tokens": getattr(resp.usage, "total_tokens", None),
        },
    }
    if use_cache:
        cache_put(PROVIDER, model, temperature, prompt, payload, cache_dir=cache_dir)
    return payload, False
