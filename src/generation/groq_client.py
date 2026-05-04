"""Groq client wrapper with disk-cache.

Every call goes through `src/utils/cache.py` per AGENTS.md §2.3 — re-runs are free.
The default model is the locked thesis answerer (`llama-3.3-70b-versatile`).

Returns `(text, latency_s, was_cached)` so callers can log timing and cache-hit rate.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from groq import Groq

from src.utils.cache import DEFAULT_CACHE_DIR, cache_get, cache_put

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 256
PROVIDER = "groq"


_client_singleton: Groq | None = None


def _client() -> Groq:
    global _client_singleton
    if _client_singleton is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to .env (repo root) and run "
                "`from dotenv import load_dotenv; load_dotenv()` before calling this."
            )
        _client_singleton = Groq(api_key=api_key)
    return _client_singleton


def groq_complete(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
) -> tuple[str, float, bool]:
    """Run a single Groq chat completion. Returns (text, latency_s, was_cached).

    Cache key = sha256(provider + model + temperature + prompt). The cached payload
    stores the full response dict plus the original latency for transparency.
    """
    if use_cache:
        cached = cache_get(PROVIDER, model, temperature, prompt, cache_dir=cache_dir)
        if cached is not None:
            return cached["text"], cached.get("latency_s", 0.0), True

    t0 = time.time()
    resp = _client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    latency = time.time() - t0
    text = resp.choices[0].message.content or ""

    if use_cache:
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
        cache_put(PROVIDER, model, temperature, prompt, payload, cache_dir=cache_dir)

    return text, latency, False
