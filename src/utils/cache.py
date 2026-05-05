"""Disk-backed cache for LLM calls.

Cache key = sha256(provider + model + temperature + prompt) — see AGENTS.md §2.3.
Stored as one JSON file per key under `data/cache/<provider>/<key[:2]>/<key>.json`.

Resuming after a Groq rate-limit pause is free; re-running an experiment after a
bug fix only pays for the prompts that actually changed.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

# Anchor the cache directory to the repo root rather than the caller's cwd.
# Without this, a Jupyter kernel running with cwd = notebooks/ silently writes
# the cache to `notebooks/data/cache/...` and breaks resumability.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = Path(os.environ.get("THESIS_CACHE_DIR", str(_REPO_ROOT / "data" / "cache")))


def _key(provider: str, model: str, temperature: float, prompt: str) -> str:
    payload = f"{provider}\n{model}\n{temperature:.4f}\n{prompt}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _path(cache_dir: Path, provider: str, key: str) -> Path:
    return cache_dir / provider / key[:2] / f"{key}.json"


def cache_get(
    provider: str,
    model: str,
    temperature: float,
    prompt: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> dict[str, Any] | None:
    """Return cached payload dict if present, else None."""
    p = _path(Path(cache_dir), provider, _key(provider, model, temperature, prompt))
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def cache_put(
    provider: str,
    model: str,
    temperature: float,
    prompt: str,
    payload: dict[str, Any],
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Path:
    """Write payload to disk. Returns the file path."""
    key = _key(provider, model, temperature, prompt)
    p = _path(Path(cache_dir), provider, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p
