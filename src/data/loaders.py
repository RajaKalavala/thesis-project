"""Loaders for the canonical processed datasets.

Three callers, three datasets:

- `load_medqa_4opt()` returns the full **12,723-row** evaluation surface used by
  every Phase-4 experiment. The on-disk parquet stores `options` as a JSON
  string (`options_json`); this loader parses it back to a dict and assigns a
  stable `question_id` of the form `medqa_NNNNN` from the row index.

- `load_golden(accepted_only=True)` returns the 234-row golden RAGAS subset
  produced by Phase 3 (`data/processed/golden_ragas_300.jsonl`). Golden rows
  carry their own `question_id` (the stratified-sample index 0..299) which is
  **NOT** the medqa_4opt row index — `question` text is the join key. The
  loader exposes both `question_id` (golden's own) and `question` so callers
  can match either way.

- `load_chunks()` returns the 67,599-chunk corpus from `chunks.parquet`,
  unchanged from Notebook 01.

Per `docs/dataset.md` §2.2: never join 4-opt and 5-opt by `answer_idx` (the
letter is re-assigned during option reduction). Always join by `question` text.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# Anchor data paths to the repo root computed from this file's location, not
# the caller's cwd. Jupyter sets cwd = notebooks/ which would otherwise break
# every load. `parents[2]` from `<repo>/src/data/loaders.py` is `<repo>/`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = _REPO_ROOT / "data" / "processed"
MEDQA_4OPT_PATH = PROCESSED_DIR / "medqa_4opt.parquet"
GOLDEN_PATH = PROCESSED_DIR / "golden_ragas_300.jsonl"
CHUNKS_PATH = PROCESSED_DIR / "chunks.parquet"


def load_medqa_4opt(path: Path = MEDQA_4OPT_PATH) -> pd.DataFrame:
    """Load the 12,723-row 4-option MedQA dataset with `options` parsed and a
    stable `question_id` derived from the row index.

    Returned columns: `question_id`, `question`, `answer`, `answer_idx`,
    `options` (dict), `meta_info`, `split`, `n_metamap_phrases`,
    `metamap_phrases` (list[str]).
    """
    df = pd.read_parquet(path)
    df = df.reset_index(drop=True).copy()
    df["question_id"] = [f"medqa_{i:05d}" for i in range(len(df))]
    df["options"] = df["options_json"].map(json.loads)
    df["metamap_phrases"] = df["metamap_phrases_json"].map(json.loads)
    return df[
        [
            "question_id",
            "question",
            "answer",
            "answer_idx",
            "options",
            "meta_info",
            "split",
            "n_metamap_phrases",
            "metamap_phrases",
        ]
    ]


def load_golden(
    path: Path = GOLDEN_PATH,
    accepted_only: bool = True,
) -> list[dict]:
    """Load the golden RAGAS subset.

    `accepted_only=True` is a no-op in practice — the canonical file at
    `data/processed/golden_ragas_300.jsonl` already contains only the 234
    accepted rows (the 53 needs_review and 13 dropped live in
    `data/processed/golden/`). The flag is kept for callers that point this
    loader at the staged `golden_validated.jsonl` instead.
    """
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if accepted_only:
        rows = [r for r in rows if r.get("final_status") == "accepted"]
    return rows


def load_chunks(path: Path = CHUNKS_PATH) -> pd.DataFrame:
    """Load the 67,599-row chunked textbook corpus from Notebook 01."""
    return pd.read_parquet(path)
