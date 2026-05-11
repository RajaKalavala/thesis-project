"""Passage-level SHAP via KernelSHAP on existing LIME sample data (EXP_11).

## Key efficiency property — no new Groq calls

KernelSHAP and LIME both fit a linear model on `(subset mask, prediction score)`
pairs — they differ only in how the samples are *weighted*:

- **LIME** (`src/xai/lime_passage.py`): uniform-weight ridge regression.
- **KernelSHAP** (this module): weighted least-squares with the SHAP kernel
  `w(S) = (k − 1) / [C(k, |S|) · |S| · (k − |S|)]`, which axiomatically
  yields Shapley values.

So we reuse EXP_10 Stage B's sample data (one JSONL row per question, each
containing 16 binary masks + the LLM's per-mask predictions) and compute
SHAP values via weighted regression — zero new LLM calls. EXP_12 then
compares the LIME and SHAP per-passage rankings on the same data.

## No-RAG anchor

If the No-RAG prediction is supplied for a question, we add it as a
synthetic all-zeros sample (large weight, forcing the constraint). This
gives KernelSHAP the missing endpoint of its convex combination and
sharpens the per-passage Shapley estimates.

## Sign convention

SHAP values use the same score functions as LIME-subset:
- `correctness_shap[i]`: positive = chunk i pushes the answer toward gold.
- `sameletter_shap[i]`: positive = chunk i anchors the LLM to its full-prompt prediction.

Top-1 passage = argmax(|shap_value|), ties broken by retrieval rank.

## Resumability

`run_shap_from_lime_batch` streams output to a new JSONL, skipping any
`(question_id, architecture)` already present. Safe to re-run.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from math import comb
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm.auto import tqdm

_HIGH_WEIGHT = 1e6  # force the all-ones / all-zeros constraints


def kernel_shap_weight(subset_size: int, k: int) -> float:
    """Standard SHAP kernel weight for a subset of given size.

    The all-zeros and all-ones masks get a "large" weight (1e6) which acts
    as a soft constraint forcing `f(∅) = intercept` and
    `f(all) = intercept + Σ shap_i`.
    """
    if k <= 0:
        return 0.0
    if subset_size == 0 or subset_size == k:
        return _HIGH_WEIGHT
    return float(k - 1) / float(comb(k, subset_size) * subset_size * (k - subset_size))


@dataclass
class ShapPassageAttribution:
    chunk_id: str
    rank: int
    correctness_shap: float
    sameletter_shap: float


@dataclass
class ShapResult:
    question_id: str
    architecture: str
    method: str  # "kernel_shap"
    gold_letter: str
    full_pred_letter: str | None
    full_correct: bool
    n_passages: int
    n_samples: int
    no_rag_anchor_used: bool
    correctness_intercept: float
    sameletter_intercept: float
    correctness_score_variance: float
    sameletter_score_variance: float
    passages: list[ShapPassageAttribution] = field(default_factory=list)
    top_chunk_by_correctness: str | None = None
    top_chunk_by_sameletter: str | None = None
    note: str = ""

    def to_jsonl_row(self) -> dict:
        return asdict(self)


def passage_shap_from_lime_record(
    lime_record: dict,
    no_rag_pred_letter: str | None = None,
    gold_letter: str | None = None,
) -> ShapResult:
    """Compute SHAP values for one question from an existing LIME-subset record.

    Parameters
    ----------
    lime_record
        One row from `stage_b_retrievalchanged_mhop.jsonl` (or any other
        `run_subset_lime_batch` output). Must contain `samples` (list of
        masks + predictions) and `passages` (per-chunk metadata).
    no_rag_pred_letter
        Optional. The LLM's prediction with NO chunks (i.e., No-RAG prompt).
        If supplied, we add a synthetic all-zeros sample with the
        corresponding scores, anchoring the SHAP regression at both endpoints.
        Pulled from `exp_01_base_llm__test_1273/predictions.jsonl`.
    gold_letter
        Required if `no_rag_pred_letter` is supplied (to compute the
        all-zeros sample's correctness score). Falls back to the LIME
        record's `gold_letter`.
    """
    k = lime_record["n_passages"]
    samples = lime_record.get("samples", [])
    chunks_meta = lime_record.get("passages", [])
    gold = gold_letter or lime_record.get("gold_letter")
    full_pred = lime_record.get("full_pred_letter")

    if k == 0 or not samples:
        return ShapResult(
            question_id=lime_record["question_id"],
            architecture=lime_record["architecture"],
            method="kernel_shap",
            gold_letter=gold or "",
            full_pred_letter=full_pred,
            full_correct=bool(lime_record.get("full_correct", False)),
            n_passages=0,
            n_samples=0,
            no_rag_anchor_used=False,
            correctness_intercept=0.0,
            sameletter_intercept=0.0,
            correctness_score_variance=0.0,
            sameletter_score_variance=0.0,
            passages=[],
            note="no_passages_skipped_shap",
        )

    # Build design matrix
    X = np.array([s["mask"] for s in samples], dtype=float)
    y_corr = np.array([s["correct"] for s in samples], dtype=float)
    y_same = np.array([s["sameletter"] for s in samples], dtype=float)
    subset_sizes = X.sum(axis=1).astype(int)
    weights = np.array(
        [kernel_shap_weight(int(sz), k) for sz in subset_sizes], dtype=float
    )

    # Optional No-RAG anchor: synthetic all-zeros sample
    no_rag_anchor_used = False
    if no_rag_pred_letter is not None and gold is not None and full_pred is not None:
        no_rag_correct = int(no_rag_pred_letter == gold)
        no_rag_sameletter = int(no_rag_pred_letter == full_pred)
        X = np.vstack([X, np.zeros(k)])
        y_corr = np.concatenate([y_corr, [float(no_rag_correct)]])
        y_same = np.concatenate([y_same, [float(no_rag_sameletter)]])
        weights = np.concatenate([weights, [_HIGH_WEIGHT]])
        no_rag_anchor_used = True

    var_corr = float(np.var(y_corr))
    var_same = float(np.var(y_same))

    def _fit(y: np.ndarray, var: float) -> tuple[list[float], float]:
        if var == 0:
            return [0.0] * k, float(y.mean()) if y.size else 0.0
        lr = LinearRegression().fit(X, y, sample_weight=weights)
        return [float(c) for c in lr.coef_], float(lr.intercept_)

    coef_corr, int_corr = _fit(y_corr, var_corr)
    coef_same, int_same = _fit(y_same, var_same)

    passages = []
    for i, meta in enumerate(chunks_meta):
        passages.append(
            ShapPassageAttribution(
                chunk_id=meta["chunk_id"],
                rank=meta["rank"],
                correctness_shap=float(coef_corr[i]),
                sameletter_shap=float(coef_same[i]),
            )
        )

    def _argmax_abs(coefs: list[float], variance: float) -> str | None:
        if variance == 0 or all(abs(c) < 1e-9 for c in coefs):
            return None
        i = int(np.argmax(np.abs(coefs)))
        return chunks_meta[i]["chunk_id"]

    return ShapResult(
        question_id=lime_record["question_id"],
        architecture=lime_record["architecture"],
        method="kernel_shap",
        gold_letter=gold or "",
        full_pred_letter=full_pred,
        full_correct=bool(lime_record.get("full_correct", False)),
        n_passages=k,
        n_samples=len(samples) + (1 if no_rag_anchor_used else 0),
        no_rag_anchor_used=no_rag_anchor_used,
        correctness_intercept=int_corr,
        sameletter_intercept=int_same,
        correctness_score_variance=var_corr,
        sameletter_score_variance=var_same,
        passages=passages,
        top_chunk_by_correctness=_argmax_abs(coef_corr, var_corr),
        top_chunk_by_sameletter=_argmax_abs(coef_same, var_same),
    )


def _load_completed_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    done: set[tuple[str, str]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            done.add((row["question_id"], row["architecture"]))
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def run_shap_from_lime_batch(
    lime_jsonl: Path,
    output_path: Path,
    no_rag_pred_map: dict[str, str] | None = None,
    *,
    progress: bool = True,
) -> dict:
    """Read each LIME-subset record, compute SHAP via weighted regression on
    the same `(mask, score)` samples, and stream the SHAP results to
    `output_path`.

    Parameters
    ----------
    lime_jsonl
        Path to a `run_subset_lime_batch` output (e.g. the EXP_10 Stage B
        canonical file).
    output_path
        Where to write SHAP results (one JSONL row per question).
    no_rag_pred_map
        Optional. `{question_id: no_rag_pred_letter}` from EXP_01. If given,
        each question's SHAP gets a synthetic all-zeros anchor sample.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _load_completed_keys(output_path)
    if completed:
        print(f"[shap] resuming — {len(completed)} (qid, arch) rows already done")

    lime_jsonl = Path(lime_jsonl)
    lime_records = [json.loads(line) for line in lime_jsonl.read_text().splitlines()]

    n_written = 0
    n_skipped = 0
    n_anchored = 0
    t_start = time.time()
    iterator = tqdm(lime_records, desc="KernelSHAP", disable=not progress)
    with output_path.open("a", encoding="utf-8") as f:
        for rec in iterator:
            key = (rec["question_id"], rec["architecture"])
            if key in completed:
                n_skipped += 1
                continue
            no_rag_pred = (
                no_rag_pred_map.get(rec["question_id"]) if no_rag_pred_map else None
            )
            result = passage_shap_from_lime_record(
                lime_record=rec,
                no_rag_pred_letter=no_rag_pred,
                gold_letter=rec.get("gold_letter"),
            )
            f.write(json.dumps(result.to_jsonl_row()) + "\n")
            f.flush()
            n_written += 1
            if result.no_rag_anchor_used:
                n_anchored += 1

    return {
        "output_path": str(output_path),
        "method": "kernel_shap",
        "n_rows_written_this_run": n_written,
        "n_rows_already_done": n_skipped,
        "n_with_no_rag_anchor": n_anchored,
        "wall_time_s_this_run": time.time() - t_start,
    }
