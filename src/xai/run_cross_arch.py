"""Cross-architecture Phase 6 extension — run LIME + SHAP + agreement on the
retrieval-changed subsets of Naive, Sparse, and Hybrid (Multi-Hop already done).

Each architecture's retrieval-changed subset = questions on test_1273 where
the architecture's predicted letter differs from EXP_01 No-RAG's predicted
letter. This is the same "retrieval-changed" surface used for Multi-Hop in
EXP_10/11/12 — the only surface where chunks demonstrably changed the LLM's
answer (memorisation cases have nothing to attribute).

The script reuses the existing `src/xai/{lime_passage,shap_passage,agreement}.py`
modules verbatim. The Multi-Hop wiring (notebook 06) is cloned in `main()`.

Output paths (parallel to Multi-Hop's `stage_b_retrievalchanged_mhop.jsonl`):
    results/exp_10_lime_passage/stage_b_retrievalchanged_<arch>.jsonl
    results/exp_11_shap_passage/stage_b_retrievalchanged_<arch>.jsonl
    results/exp_12_agreement/stage_b_retrievalchanged_<arch>.jsonl

Usage:
    .venv/bin/python -m src.xai.run_cross_arch --pilot
        → Stage A: 5 Naive questions only (smoke). ~2 min Groq.

    .venv/bin/python -m src.xai.run_cross_arch
        → Stage B: full retrieval-changed subsets for Naive (186 Q),
          Sparse (136 Q), Hybrid (184 Q). ~15 min Groq total.

    .venv/bin/python -m src.xai.run_cross_arch --archs naive
        → Run a single architecture (resumable; safe to re-invoke).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

import pandas as pd

from src.generation.groq_client import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    groq_complete,
)
from src.retrieval.base import Chunk
from src.xai.agreement import run_agreement_batch
from src.xai.lime_passage import run_subset_lime_batch
from src.xai.shap_passage import run_shap_from_lime_batch


# Per-architecture config. Each row carries the result-folder prefix used in
# `results/exp_*__test_1273/`, plus the short label used in output filenames
# (matches the `<arch>` suffix in stage_b_retrievalchanged_<arch>.jsonl).
@dataclass(frozen=True)
class ArchSpec:
    short: str           # filename suffix (matches Multi-Hop's "mhop")
    prefix: str          # `results/<prefix>__test_1273/`
    label: str           # the `architecture` field written into each JSONL row

NAIVE  = ArchSpec("naive",  "exp_02_naive_rag",  "exp_02_naive_rag")
SPARSE = ArchSpec("sparse", "exp_03_sparse_rag", "exp_03_sparse_rag")
HYBRID = ArchSpec("hybrid", "exp_04_hybrid_rag", "exp_04_hybrid_rag")
ARCHS_BY_NAME = {"naive": NAIVE, "sparse": SPARSE, "hybrid": HYBRID}


def _load_predictions(prefix: str) -> dict[str, str]:
    """{question_id: pred_letter} from a test_1273 predictions.jsonl."""
    path = REPO_ROOT / "results" / f"{prefix}__test_1273" / "predictions.jsonl"
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        out[r["question_id"]] = r.get("pred_letter")
    return out


def _load_retrieval(prefix: str) -> dict[str, list[tuple[str, float]]]:
    """{question_id: [(chunk_id, score), ...]} from a test_1273 retrieval.jsonl."""
    path = REPO_ROOT / "results" / f"{prefix}__test_1273" / "retrieval.jsonl"
    out: dict[str, list[tuple[str, float]]] = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        out[r["question_id"]] = list(zip(r["retrieved_chunk_ids"], r["retrieved_chunk_scores"]))
    return out


def _retrieval_changed_qids(arch_preds: dict[str, str], norag_preds: dict[str, str]) -> list[str]:
    """Question IDs where the architecture's predicted letter differs from No-RAG's.

    Same definition used for Multi-Hop in notebook 06. Sorted for reproducibility.
    """
    common = set(arch_preds) & set(norag_preds)
    return sorted(qid for qid in common if arch_preds[qid] != norag_preds[qid])


def _build_rows(
    arch: ArchSpec,
    qids: Iterable[str],
    medqa_by_qid: dict[str, dict],
    chunks_by_id: pd.DataFrame,
    retrieval_by_qid: dict[str, list[tuple[str, float]]],
) -> list[dict]:
    """Build the row-dict format `run_subset_lime_batch` expects."""
    rows: list[dict] = []
    for qid in qids:
        md = medqa_by_qid.get(qid)
        if md is None:
            continue
        ret = retrieval_by_qid.get(qid, [])
        chunks: list[Chunk] = []
        for chunk_id, score in ret:
            try:
                cr = chunks_by_id.loc[chunk_id]
            except KeyError:
                continue
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    book_name=str(cr["book_name"]),
                    text=str(cr["text"]),
                    score=float(score),
                )
            )
        if not chunks:
            continue
        rows.append({
            "question_id": qid,
            "question": md["question"],
            "options": md["options"],
            "gold_letter": md["gold_letter"],
            "chunks": chunks,
            "architecture": arch.label,
        })
    return rows


def _load_medqa_test() -> dict[str, dict]:
    """{question_id: {question, options, gold_letter}} for the 1,273 test rows."""
    md = pd.read_parquet(REPO_ROOT / "data/processed/medqa_4opt.parquet")
    md = md.reset_index(drop=False).rename(columns={"index": "row_idx"})
    md["question_id"] = "medqa_" + md["row_idx"].astype(str)
    md = md[md["split"] == "test"].copy()
    out: dict[str, dict] = {}
    for _, r in md.iterrows():
        out[r["question_id"]] = {
            "question": r["question"],
            "options": json.loads(r["options_json"]),
            "gold_letter": r["answer_idx"],
        }
    return out


def run_arch(arch: ArchSpec, *, limit: int | None = None) -> dict:
    """Run LIME + SHAP + agreement for one architecture's retrieval-changed subset.

    Returns a small summary dict per stage with counts + paths.
    """
    print(f"\n{'='*72}\n[{arch.short}] Phase 6 extension — LIME + SHAP + agreement")
    print(f"{'='*72}")

    # --- Stage 0: identify retrieval-changed subset ---------------------------
    norag_preds = _load_predictions("exp_01_base_llm")
    arch_preds = _load_predictions(arch.prefix)
    qids = _retrieval_changed_qids(arch_preds, norag_preds)
    print(f"[{arch.short}] retrieval-changed = {len(qids)} questions (of {len(arch_preds):,} total)")
    if limit is not None:
        qids = qids[:limit]
        print(f"[{arch.short}] LIMIT applied → running first {len(qids)} questions only")

    medqa_by_qid = _load_medqa_test()
    chunks_by_id = pd.read_parquet(REPO_ROOT / "data/processed/chunks.parquet").set_index("chunk_id")
    retrieval_by_qid = _load_retrieval(arch.prefix)
    rows = _build_rows(arch, qids, medqa_by_qid, chunks_by_id, retrieval_by_qid)
    print(f"[{arch.short}] built {len(rows)} row dicts with non-empty chunk lists")

    # --- Stage 1: LIME subset-sampling ---------------------------------------
    lime_out = REPO_ROOT / "results/exp_10_lime_passage" / f"stage_b_retrievalchanged_{arch.short}.jsonl"
    llm_callable = partial(
        groq_complete,
        model=DEFAULT_MODEL,
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS,
    )

    def _llm(prompt: str) -> tuple[str, float, bool]:
        # groq_complete returns (text, latency, was_cached); pass-through
        return llm_callable(prompt)

    t0 = time.time()
    lime_summary = run_subset_lime_batch(
        rows=rows,
        output_path=lime_out,
        llm_callable=_llm,
        n_samples=16,
        seed=42,
        alpha=0.1,
        progress=True,
    )
    t_lime = time.time() - t0
    print(f"[{arch.short}] LIME done in {t_lime:.1f}s → {lime_out}")

    # --- Stage 2: SHAP on LIME output (with No-RAG anchor) -------------------
    no_rag_pred_map: dict[str, str] = {qid: norag_preds[qid] for qid in qids if qid in norag_preds}
    shap_out = REPO_ROOT / "results/exp_11_shap_passage" / f"stage_b_retrievalchanged_{arch.short}.jsonl"

    t0 = time.time()
    shap_summary = run_shap_from_lime_batch(
        lime_jsonl=lime_out,
        output_path=shap_out,
        no_rag_pred_map=no_rag_pred_map,
        progress=True,
    )
    t_shap = time.time() - t0
    print(f"[{arch.short}] SHAP done in {t_shap:.2f}s → {shap_out}")

    # --- Stage 3: LIME ↔ SHAP agreement --------------------------------------
    agr_out = REPO_ROOT / "results/exp_12_agreement" / f"stage_b_retrievalchanged_{arch.short}.jsonl"

    t0 = time.time()
    agr_summary = run_agreement_batch(
        lime_jsonl=lime_out,
        shap_jsonl=shap_out,
        output_path=agr_out,
        progress=True,
    )
    t_agr = time.time() - t0
    print(f"[{arch.short}] agreement done in {t_agr:.2f}s → {agr_out}")

    return {
        "arch": arch.short,
        "n_retrieval_changed": len(qids),
        "n_rows_built": len(rows),
        "lime_summary": lime_summary,
        "shap_summary": shap_summary,
        "agr_summary": agr_summary,
        "wall_time_s": {"lime": t_lime, "shap": t_shap, "agreement": t_agr},
        "outputs": {
            "lime":  str(lime_out.relative_to(REPO_ROOT)),
            "shap":  str(shap_out.relative_to(REPO_ROOT)),
            "agreement": str(agr_out.relative_to(REPO_ROOT)),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archs",
        default="naive,sparse,hybrid",
        help="Comma-separated arch list (any of naive,sparse,hybrid). Default: all three.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N retrieval-changed questions per arch (smoke test).",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Pilot mode = --archs naive --limit 5.",
    )
    args = parser.parse_args()

    if args.pilot:
        args.archs = "naive"
        args.limit = 5

    archs = [ARCHS_BY_NAME[a.strip()] for a in args.archs.split(",") if a.strip()]
    if not archs:
        raise SystemExit("No architectures selected.")

    summaries = []
    for arch in archs:
        s = run_arch(arch, limit=args.limit)
        summaries.append(s)

    # Print a short JSON summary at the end for downstream parsing.
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for s in summaries:
        print(json.dumps(s, indent=2, default=str))


if __name__ == "__main__":
    main()
