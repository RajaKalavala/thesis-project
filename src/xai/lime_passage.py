"""Passage-level LIME — two implementations (EXP_10).

## Two methods, kept side-by-side

### 1. Leave-one-out (LOO) — `lime_passage_loo`

For each question with `k` retrieved passages:

- Run the full prompt + `k` LOO prompts (remove one passage at a time).
- Per-passage attribution = `score_full − score_loo_i` (binary correctness)
  or `1 if loo_pred ≠ full_pred else 0` (same-letter).

**Discovered limitation (2026-05-11 smoke)**: LOO produces all-zero
attribution on the smoke when chunks carry distributed (not concentrated)
signal. Removing 1 of k chunks rarely flips the LLM's answer because the
remaining `k-1` chunks anchor the prediction. Kept here for the
methodology-comparison row in the thesis writeup.

### 2. Subset-sampling LIME — `passage_subset_lime`

For each question with `k` retrieved passages:

- Generate `N` random binary masks (each chunk in/out with `p=0.5`),
  always including the all-ones mask so the full prediction anchors the fit.
- Run the LLM on each subset; record per-sample `correctness` and
  `sameletter` scores.
- Fit a **ridge regression** (`score ≈ Σ w_i · mask_i + b`) per signal.
  The fitted coefficients `w_i` are the per-passage attributions; their
  signs distinguish "supports the answer" (positive) from "pulls away from
  the answer" (negative).
- **Top-1 passage** = argmax(|w_i|).

This captures **distributed grounding**: a chunk that consistently appears
in subsets where the answer is correct (or where the answer stays the same
as the full prediction) will get a positive coefficient, even if removing
it alone in a LOO setup wouldn't flip the answer. The ridge regularisation
(`alpha=0.1`) stabilises fits when `N < k` (e.g., k=15 with N=16 samples is
borderline-underdetermined).

Cost per question: `N` LLM calls, regardless of `k`. With `N=16`, that's 16
calls — identical to k=15 LOO and ~3× more than k=5 LOO. Standard LIME
methodology; defensible in the methodology chapter.

## Cost / cache

Every Groq call goes through `groq_complete` which is disk-cached by
`(model, temp, prompt) hash`. The **full prompt** call hits cache from the
underlying experiment (EXP_02/04/05) → free. The other LOO / subset prompts
are new (different chunk subset → different prompt) → fresh calls.

## Resumability

The `run_lime_passage_batch` and `run_subset_lime_batch` runners both write
one JSONL row per `(question_id, architecture)` to disk as they go. Re-runs
skip rows already in the output file.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.linear_model import Ridge
from tqdm.auto import tqdm

from src.generation.prompts import (
    build_evidence_grounded_prompt,
    build_no_rag_prompt,
    parse_letter,
)
from src.retrieval.base import Chunk


@dataclass
class PassageAttribution:
    """Per-passage row in a LimeResult."""

    chunk_id: str
    rank: int  # 0-indexed retrieval rank
    loo_pred_letter: str | None
    loo_correct: bool
    correctness_attribution: int  # +1 helped, -1 hurt, 0 no effect
    sameletter_attribution: int  # 1 if LOO changed the letter, else 0
    loo_was_cached: bool
    loo_latency_s: float


@dataclass
class LimeResult:
    """One question × one architecture's LIME-LOO record.

    Stored as a JSONL row by `run_lime_passage_batch`."""

    question_id: str
    architecture: str
    gold_letter: str
    full_pred_letter: str | None
    full_correct: bool
    full_was_cached: bool
    full_latency_s: float
    n_passages: int
    passages: list[PassageAttribution] = field(default_factory=list)
    top_chunk_by_correctness: str | None = None
    top_chunk_by_sameletter: str | None = None
    note: str = ""  # set if e.g. n_passages == 0 (No-RAG) so we skipped LOO

    def to_jsonl_row(self) -> dict:
        d = asdict(self)
        # asdict already converts dataclasses nested in lists
        return d


def _score_correctness(pred_letter: str | None, gold_letter: str) -> int:
    return 1 if (pred_letter == gold_letter) else 0


def lime_passage_loo(
    question_id: str,
    question: str,
    options: dict[str, str],
    gold_letter: str,
    chunks: list[Chunk],
    architecture: str,
    llm_callable: Callable[[str], tuple[str, float, bool]],
) -> LimeResult:
    """Compute LOO passage attribution for one question.

    Parameters
    ----------
    question_id, question, options, gold_letter
        Question payload — same shape as the runner's row parser produces.
    chunks
        Retrieved chunks in retrieval rank order. May be empty (e.g.
        No-RAG architecture) — in that case the result holds a `note` and
        no per-passage rows; the caller treats it as a degenerate case.
    architecture
        Free-form label that goes into the result row (e.g. "exp_05_multi_hop_rag").
    llm_callable
        `(prompt: str) -> (text, latency_s, was_cached)`. Caller wires up
        `groq_complete` with model/temp/max_tokens defaults.

    Returns
    -------
    LimeResult with per-passage attributions.
    """
    n_passages = len(chunks)
    if n_passages == 0:
        # No-RAG: build the no-rag prompt, get the answer, return early.
        prompt = build_no_rag_prompt(question, options)
        text, latency, was_cached = llm_callable(prompt)
        pred = parse_letter(text)
        return LimeResult(
            question_id=question_id,
            architecture=architecture,
            gold_letter=gold_letter,
            full_pred_letter=pred,
            full_correct=(pred == gold_letter),
            full_was_cached=was_cached,
            full_latency_s=latency,
            n_passages=0,
            passages=[],
            top_chunk_by_correctness=None,
            top_chunk_by_sameletter=None,
            note="no_passages_skipped_loo",
        )

    # Full prompt — same as the underlying experiment's prompt for this question
    # (should hit cache from EXP_02/04/05).
    chunk_texts = [c.text for c in chunks]
    full_prompt = build_evidence_grounded_prompt(question, options, chunk_texts)
    full_text, full_latency, full_was_cached = llm_callable(full_prompt)
    full_pred = parse_letter(full_text)
    score_full = _score_correctness(full_pred, gold_letter)

    passage_rows: list[PassageAttribution] = []
    for i, chunk in enumerate(chunks):
        loo_texts = [t for j, t in enumerate(chunk_texts) if j != i]
        loo_prompt = build_evidence_grounded_prompt(question, options, loo_texts)
        loo_text, loo_latency, loo_was_cached = llm_callable(loo_prompt)
        loo_pred = parse_letter(loo_text)
        score_loo = _score_correctness(loo_pred, gold_letter)
        passage_rows.append(
            PassageAttribution(
                chunk_id=chunk.chunk_id,
                rank=i,
                loo_pred_letter=loo_pred,
                loo_correct=(loo_pred == gold_letter),
                correctness_attribution=score_full - score_loo,
                sameletter_attribution=int(loo_pred != full_pred),
                loo_was_cached=loo_was_cached,
                loo_latency_s=loo_latency,
            )
        )

    # Top-1 selection (ties broken by retrieval rank, i.e. lower-rank wins).
    def _pick_top(key: Callable[[PassageAttribution], int]) -> str | None:
        best_score = max((key(p) for p in passage_rows), default=None)
        if best_score is None or best_score <= 0:
            # No passage helped (correctness) / changed the letter (sameletter).
            return None
        for p in passage_rows:  # ranks are ascending; first hit wins ties
            if key(p) == best_score:
                return p.chunk_id
        return None

    top_correct = _pick_top(lambda p: p.correctness_attribution)
    top_sameletter = _pick_top(lambda p: p.sameletter_attribution)

    return LimeResult(
        question_id=question_id,
        architecture=architecture,
        gold_letter=gold_letter,
        full_pred_letter=full_pred,
        full_correct=(full_pred == gold_letter),
        full_was_cached=full_was_cached,
        full_latency_s=full_latency,
        n_passages=n_passages,
        passages=passage_rows,
        top_chunk_by_correctness=top_correct,
        top_chunk_by_sameletter=top_sameletter,
    )


def _load_completed_keys(output_path: Path) -> set[tuple[str, str]]:
    """Read existing JSONL and return `{(question_id, architecture)}` pairs
    so the batch runner can resume mid-stream."""
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


def run_lime_passage_batch(
    rows: list[dict],
    output_path: Path,
    llm_callable: Callable[[str], tuple[str, float, bool]],
    *,
    progress: bool = True,
) -> dict:
    """Run LIME-LOO over a batch and stream results to disk.

    Each `row` dict must carry:
        - ``question_id``        (str)
        - ``question``           (str)
        - ``options``            (dict[str, str])
        - ``gold_letter``        (str)
        - ``chunks``             (list[Chunk]; may be empty)
        - ``architecture``       (str)

    Returns a small summary dict (n_rows, n_skipped, wall_time_s, ...).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _load_completed_keys(output_path)
    if completed:
        print(f"[lime] resuming — {len(completed)} (qid, arch) rows already done")

    n_written = 0
    n_skipped = 0
    t_start = time.time()
    iterator = tqdm(rows, desc="LIME-LOO", disable=not progress)
    with output_path.open("a", encoding="utf-8") as f:
        for row in iterator:
            key = (row["question_id"], row["architecture"])
            if key in completed:
                n_skipped += 1
                continue
            result = lime_passage_loo(
                question_id=row["question_id"],
                question=row["question"],
                options=row["options"],
                gold_letter=row["gold_letter"],
                chunks=row["chunks"],
                architecture=row["architecture"],
                llm_callable=llm_callable,
            )
            f.write(json.dumps(result.to_jsonl_row()) + "\n")
            f.flush()
            n_written += 1

    return {
        "output_path": str(output_path),
        "n_rows_written_this_run": n_written,
        "n_rows_already_done": n_skipped,
        "wall_time_s_this_run": time.time() - t_start,
    }


# ---------------------------------------------------------------------------
#  Subset-sampling LIME (method 2)
# ---------------------------------------------------------------------------


@dataclass
class SubsetPassageAttribution:
    """Per-passage row in a SubsetLimeResult — coefficient + magnitude."""

    chunk_id: str
    rank: int
    correctness_coef: float
    sameletter_coef: float


@dataclass
class SubsetSample:
    """One random-subset sample's record — kept for audit + later SHAP reuse."""

    mask: list[int]
    n_chunks_included: int
    pred_letter: str | None
    correct: int
    sameletter: int
    was_cached: bool
    latency_s: float


@dataclass
class SubsetLimeResult:
    """One (question × architecture) result for subset-sampling LIME."""

    question_id: str
    architecture: str
    method: str  # "subset_lime"
    gold_letter: str
    full_pred_letter: str | None
    full_correct: bool
    full_was_cached: bool
    full_latency_s: float
    n_passages: int
    n_samples: int
    seed: int
    alpha: float
    correctness_intercept: float
    sameletter_intercept: float
    correctness_score_variance: float  # bookkeeping: 0 ⇒ ridge has nothing to learn
    sameletter_score_variance: float
    passages: list[SubsetPassageAttribution] = field(default_factory=list)
    samples: list[SubsetSample] = field(default_factory=list)
    top_chunk_by_correctness: str | None = None
    top_chunk_by_sameletter: str | None = None
    note: str = ""

    def to_jsonl_row(self) -> dict:
        return asdict(self)


def _generate_masks(k: int, n_samples: int, seed: int) -> list[list[int]]:
    """Return `n_samples` binary masks of length `k`.

    The first mask is always all-ones (so the full prediction anchors the
    regression). Remaining masks are drawn with each bit ~ Bernoulli(0.5),
    rejecting all-zero masks (which would degenerate to the No-RAG prompt)
    and the all-ones mask (already included).
    """
    rng = np.random.default_rng(seed)
    masks: list[list[int]] = [[1] * k]  # all-ones first
    attempts = 0
    while len(masks) < n_samples and attempts < n_samples * 20:
        attempts += 1
        m = rng.binomial(1, 0.5, size=k).tolist()
        s = sum(m)
        if s == 0 or s == k:  # skip degenerate masks
            continue
        masks.append(m)
    # If we couldn't fill (very small k), pad with random partial masks allowing repeats
    while len(masks) < n_samples:
        m = rng.binomial(1, 0.5, size=k).tolist()
        if sum(m) == 0:
            m[0] = 1  # avoid empty
        masks.append(m)
    return masks


def passage_subset_lime(
    question_id: str,
    question: str,
    options: dict[str, str],
    gold_letter: str,
    chunks: list[Chunk],
    architecture: str,
    llm_callable: Callable[[str], tuple[str, float, bool]],
    *,
    n_samples: int = 16,
    seed: int = 42,
    alpha: float = 0.1,
) -> SubsetLimeResult:
    """Subset-sampling LIME for one (question, architecture) record.

    Algorithm:
      1. Run the full prompt (cached). Record `full_pred_letter`.
      2. Generate `n_samples` masks (first is all-ones; rest random ~ Bern(0.5),
         excluding all-zero / all-ones repeats).
      3. For each mask, run the LLM on the subset prompt; record `correct`
         (vs gold) and `sameletter` (vs full prediction) as 0/1.
      4. Fit two ridge regressions (`score ~ X · w + b`) on the binary mask
         design matrix; `correctness_coef[i]` and `sameletter_coef[i]` are the
         per-passage attributions for each signal.
      5. Top-1 passage = argmax(|coef|). If the target variance is 0 (e.g.
         all `n_samples` predictions are the same letter), the top is None and
         coefficients are reported anyway (will be ~zero).

    The full prediction is included as the all-ones row of the regression
    matrix, anchoring the fit. Same-letter score for that row is 1 by
    construction.
    """
    n_passages = len(chunks)
    if n_passages == 0:
        # No-RAG path: same convention as the LOO module.
        prompt = build_no_rag_prompt(question, options)
        text, latency, was_cached = llm_callable(prompt)
        pred = parse_letter(text)
        return SubsetLimeResult(
            question_id=question_id,
            architecture=architecture,
            method="subset_lime",
            gold_letter=gold_letter,
            full_pred_letter=pred,
            full_correct=(pred == gold_letter),
            full_was_cached=was_cached,
            full_latency_s=latency,
            n_passages=0,
            n_samples=0,
            seed=seed,
            alpha=alpha,
            correctness_intercept=0.0,
            sameletter_intercept=0.0,
            correctness_score_variance=0.0,
            sameletter_score_variance=0.0,
            passages=[],
            samples=[],
            top_chunk_by_correctness=None,
            top_chunk_by_sameletter=None,
            note="no_passages_skipped_subset",
        )

    chunk_texts = [c.text for c in chunks]
    masks = _generate_masks(n_passages, n_samples, seed)

    samples: list[SubsetSample] = []
    full_pred: str | None = None
    full_latency = 0.0
    full_was_cached = False

    for mask in masks:
        if sum(mask) == n_passages:
            # All-ones — the full prediction. Call once; cache should hit.
            prompt = build_evidence_grounded_prompt(question, options, chunk_texts)
        else:
            included = [chunk_texts[i] for i in range(n_passages) if mask[i] == 1]
            prompt = build_evidence_grounded_prompt(question, options, included)
        text, latency, was_cached = llm_callable(prompt)
        pred = parse_letter(text)
        if sum(mask) == n_passages and full_pred is None:
            full_pred = pred
            full_latency = latency
            full_was_cached = was_cached
        samples.append(
            SubsetSample(
                mask=mask,
                n_chunks_included=sum(mask),
                pred_letter=pred,
                correct=int(pred == gold_letter) if pred is not None else 0,
                sameletter=int(pred == full_pred) if (pred is not None and full_pred is not None) else 0,
                was_cached=was_cached,
                latency_s=latency,
            )
        )

    # After the loop, full_pred is set; update sameletter for any sample that
    # ran before we saw the all-ones row (shouldn't happen since all-ones is first,
    # but be defensive).
    if full_pred is not None:
        for s in samples:
            s.sameletter = int(s.pred_letter == full_pred) if s.pred_letter is not None else 0

    # Build design matrix and fit ridge
    X = np.array([s.mask for s in samples], dtype=float)  # (N, k)
    y_correct = np.array([s.correct for s in samples], dtype=float)
    y_sameletter = np.array([s.sameletter for s in samples], dtype=float)

    var_correct = float(np.var(y_correct))
    var_sameletter = float(np.var(y_sameletter))

    if var_correct > 0:
        rc = Ridge(alpha=alpha, fit_intercept=True).fit(X, y_correct)
        coef_correct = rc.coef_.tolist()
        intercept_correct = float(rc.intercept_)
    else:
        coef_correct = [0.0] * n_passages
        intercept_correct = float(y_correct.mean()) if len(y_correct) else 0.0

    if var_sameletter > 0:
        rs = Ridge(alpha=alpha, fit_intercept=True).fit(X, y_sameletter)
        coef_sameletter = rs.coef_.tolist()
        intercept_sameletter = float(rs.intercept_)
    else:
        coef_sameletter = [0.0] * n_passages
        intercept_sameletter = float(y_sameletter.mean()) if len(y_sameletter) else 0.0

    passage_rows = [
        SubsetPassageAttribution(
            chunk_id=chunks[i].chunk_id,
            rank=i,
            correctness_coef=float(coef_correct[i]),
            sameletter_coef=float(coef_sameletter[i]),
        )
        for i in range(n_passages)
    ]

    def _argmax_abs(coefs: list[float], variance: float) -> str | None:
        if variance == 0:
            return None
        if all(abs(c) < 1e-9 for c in coefs):
            return None
        return chunks[int(np.argmax(np.abs(coefs)))].chunk_id

    return SubsetLimeResult(
        question_id=question_id,
        architecture=architecture,
        method="subset_lime",
        gold_letter=gold_letter,
        full_pred_letter=full_pred,
        full_correct=(full_pred == gold_letter),
        full_was_cached=full_was_cached,
        full_latency_s=full_latency,
        n_passages=n_passages,
        n_samples=n_samples,
        seed=seed,
        alpha=alpha,
        correctness_intercept=intercept_correct,
        sameletter_intercept=intercept_sameletter,
        correctness_score_variance=var_correct,
        sameletter_score_variance=var_sameletter,
        passages=passage_rows,
        samples=samples,
        top_chunk_by_correctness=_argmax_abs(coef_correct, var_correct),
        top_chunk_by_sameletter=_argmax_abs(coef_sameletter, var_sameletter),
    )


def run_subset_lime_batch(
    rows: list[dict],
    output_path: Path,
    llm_callable: Callable[[str], tuple[str, float, bool]],
    *,
    n_samples: int = 16,
    seed: int = 42,
    alpha: float = 0.1,
    progress: bool = True,
) -> dict:
    """Run subset-sampling LIME over a batch and stream results to disk.

    Same row shape and resumability contract as `run_lime_passage_batch`.
    Each row's per-question seed is `seed + hash(question_id) mod 2**32` so
    runs are reproducible per-question without all questions sharing the
    same random masks.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _load_completed_keys(output_path)
    if completed:
        print(f"[subset_lime] resuming — {len(completed)} (qid, arch) rows already done")

    n_written = 0
    n_skipped = 0
    t_start = time.time()
    iterator = tqdm(rows, desc="LIME-subset", disable=not progress)
    with output_path.open("a", encoding="utf-8") as f:
        for row in iterator:
            key = (row["question_id"], row["architecture"])
            if key in completed:
                n_skipped += 1
                continue
            per_q_seed = (seed + (hash(row["question_id"]) & 0x7FFFFFFF)) % (2**32)
            result = passage_subset_lime(
                question_id=row["question_id"],
                question=row["question"],
                options=row["options"],
                gold_letter=row["gold_letter"],
                chunks=row["chunks"],
                architecture=row["architecture"],
                llm_callable=llm_callable,
                n_samples=n_samples,
                seed=per_q_seed,
                alpha=alpha,
            )
            f.write(json.dumps(result.to_jsonl_row()) + "\n")
            f.flush()
            n_written += 1

    return {
        "output_path": str(output_path),
        "method": "subset_lime",
        "n_samples_per_question": n_samples,
        "alpha": alpha,
        "n_rows_written_this_run": n_written,
        "n_rows_already_done": n_skipped,
        "wall_time_s_this_run": time.time() - t_start,
    }
