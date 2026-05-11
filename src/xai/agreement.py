"""LIME ↔ SHAP per-passage agreement (EXP_12).

For each `(question_id, architecture)` row present in both the LIME-subset
output and the SHAP output, compute three agreement scores between the
two methods' per-passage rankings:

- **`top1`**: 1 if argmax(|LIME coef|) == argmax(|SHAP coef|), else 0.
- **`top3_overlap`**: `|top3(LIME) ∩ top3(SHAP)| / 3` ∈ [0, 1].
- **`spearman_rho`**: Spearman rank correlation between |LIME| and |SHAP|
  per-passage magnitudes across all chunks of the question (sign-aware
  Spearman of the signed values — useful when sign carries causality info).

We compute these for both score signals — `correctness` and `sameletter` —
since they capture different things (gold-vs-pred vs same-letter-as-full).

Special cases:
- If either method reports `top_chunk_by_X = None` for a question (zero
  variance → no attribution defined), we emit `top1 = NaN`. EXP_12
  aggregate stats exclude NaN rows.
- If `n_passages = 0` (No-RAG), we skip the row.

The resulting per-question agreement scores feed Phase 7 confidence-aware
rejection — high LIME/SHAP agreement on a question is one of the signals
in the confidence vector.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from tqdm.auto import tqdm


@dataclass
class AgreementResult:
    question_id: str
    architecture: str
    n_passages: int
    # correctness-signal agreement
    correctness_top1: float  # 0/1 or NaN (when either method has no top-1)
    correctness_top3_overlap: float
    correctness_spearman: float  # ranges [-1, 1]; NaN if degenerate
    # sameletter-signal agreement
    sameletter_top1: float
    sameletter_top3_overlap: float
    sameletter_spearman: float

    def to_jsonl_row(self) -> dict:
        return asdict(self)


def _top_n_chunk_ids(passages: list[dict], coef_key: str, n: int) -> list[str]:
    """Return the top-`n` chunk_ids by |coef|. Stable order: ties broken by rank."""
    ranked = sorted(passages, key=lambda p: (-abs(p[coef_key]), p["rank"]))
    return [p["chunk_id"] for p in ranked[:n]]


def _top1_score(lime_top: str | None, shap_top: str | None) -> float:
    if lime_top is None or shap_top is None:
        return float("nan")
    return 1.0 if lime_top == shap_top else 0.0


def _topn_overlap(lime_chunks: list[str], shap_chunks: list[str], n: int) -> float:
    """Fraction of overlap between two top-n lists. NaN if either is empty."""
    if not lime_chunks or not shap_chunks:
        return float("nan")
    return float(len(set(lime_chunks[:n]) & set(shap_chunks[:n]))) / float(n)


def _spearman_safe(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation; returns NaN on degenerate input."""
    if a.size < 2 or b.size < 2:
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    rho, _ = spearmanr(a, b)
    return float(rho) if np.isfinite(rho) else float("nan")


def agreement_from_records(
    lime_record: dict, shap_record: dict
) -> AgreementResult | None:
    """Compute per-question LIME ↔ SHAP agreement on both signals.

    Returns None on degenerate rows (n_passages == 0).
    """
    if lime_record["question_id"] != shap_record["question_id"]:
        raise ValueError(
            f"question_id mismatch: {lime_record['question_id']} vs {shap_record['question_id']}"
        )
    if lime_record["architecture"] != shap_record["architecture"]:
        raise ValueError(
            f"architecture mismatch on {lime_record['question_id']}: "
            f"{lime_record['architecture']} vs {shap_record['architecture']}"
        )

    n_passages = int(lime_record["n_passages"])
    if n_passages == 0:
        return None

    # Each method's per-passage coefs (correctness + sameletter)
    lime_passages = lime_record["passages"]
    shap_passages = shap_record["passages"]
    if len(lime_passages) != n_passages or len(shap_passages) != n_passages:
        raise ValueError(
            f"passage count mismatch on {lime_record['question_id']}: "
            f"LIME={len(lime_passages)} SHAP={len(shap_passages)} expected {n_passages}"
        )

    # Align by chunk_id (order may differ in principle; in practice both
    # follow retrieval rank order)
    lime_by_id = {p["chunk_id"]: p for p in lime_passages}
    shap_by_id = {p["chunk_id"]: p for p in shap_passages}
    chunk_ids = [p["chunk_id"] for p in lime_passages]
    if set(chunk_ids) != set(p["chunk_id"] for p in shap_passages):
        raise ValueError(f"chunk_id sets differ on {lime_record['question_id']}")

    def _signal_agreement(
        lime_key: str, shap_key: str, top_id_key: str
    ) -> tuple[float, float, float]:
        lime_top_n = _top_n_chunk_ids(lime_passages, lime_key, 3)
        shap_top_n = _top_n_chunk_ids(shap_passages, shap_key, 3)
        top1 = _top1_score(
            lime_record.get(top_id_key), shap_record.get(top_id_key)
        )
        top3 = _topn_overlap(lime_top_n, shap_top_n, 3)
        lime_vec = np.array([lime_by_id[cid][lime_key] for cid in chunk_ids])
        shap_vec = np.array([shap_by_id[cid][shap_key] for cid in chunk_ids])
        rho = _spearman_safe(lime_vec, shap_vec)
        return top1, top3, rho

    c_top1, c_top3, c_rho = _signal_agreement(
        "correctness_coef", "correctness_shap", "top_chunk_by_correctness"
    )
    s_top1, s_top3, s_rho = _signal_agreement(
        "sameletter_coef", "sameletter_shap", "top_chunk_by_sameletter"
    )

    return AgreementResult(
        question_id=lime_record["question_id"],
        architecture=lime_record["architecture"],
        n_passages=n_passages,
        correctness_top1=c_top1,
        correctness_top3_overlap=c_top3,
        correctness_spearman=c_rho,
        sameletter_top1=s_top1,
        sameletter_top3_overlap=s_top3,
        sameletter_spearman=s_rho,
    )


def run_agreement_batch(
    lime_jsonl: Path,
    shap_jsonl: Path,
    output_path: Path,
    *,
    progress: bool = True,
) -> dict:
    """Read paired LIME and SHAP JSONLs, compute per-question agreement,
    stream to `output_path`. Skips degenerate rows.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lime_rows = {
        (r["question_id"], r["architecture"]): r
        for r in (json.loads(l) for l in Path(lime_jsonl).read_text().splitlines())
    }
    shap_rows = {
        (r["question_id"], r["architecture"]): r
        for r in (json.loads(l) for l in Path(shap_jsonl).read_text().splitlines())
    }
    keys = sorted(set(lime_rows) & set(shap_rows))
    print(
        f"[agreement] LIME rows={len(lime_rows)}, SHAP rows={len(shap_rows)}, "
        f"common={len(keys)}"
    )

    n_written = 0
    n_skipped = 0
    t_start = time.time()
    with output_path.open("w", encoding="utf-8") as f:
        for key in tqdm(keys, desc="agreement", disable=not progress):
            try:
                result = agreement_from_records(lime_rows[key], shap_rows[key])
            except ValueError as e:
                print(f"  ⚠ skipping {key} due to {e}")
                n_skipped += 1
                continue
            if result is None:
                n_skipped += 1
                continue
            f.write(json.dumps(result.to_jsonl_row()) + "\n")
            n_written += 1

    return {
        "output_path": str(output_path),
        "method": "lime_shap_agreement",
        "n_rows_written": n_written,
        "n_rows_skipped": n_skipped,
        "n_lime_rows": len(lime_rows),
        "n_shap_rows": len(shap_rows),
        "n_common": len(keys),
        "wall_time_s": time.time() - t_start,
    }
