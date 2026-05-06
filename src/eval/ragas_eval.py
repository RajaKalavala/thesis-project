"""RAGAS evaluation with Claude Sonnet 4.6 as judge.

Scores `predictions.jsonl` from any Phase-4 experiment against the golden 234.
The judge is Claude Sonnet 4.6 to satisfy the three-family-separation rule
(generator = LLaMA, constructor = gpt-4o, judge = Claude — `plan.md` §0 #11).
Locked 2026-05-06 (upgrade from `claude-3-5-sonnet-20241022`).

Embeddings (used by Answer Relevancy + Answer Correctness) are local
sentence-transformers wrapping `BAAI/bge-large-en-v1.5` — the same retrieval
embedder, intentionally reused so we don't introduce a second embedder family.

**Metric applicability is auto-detected from the data**:

| Metric | Needs context? | EXP_01 (No-RAG) | EXP_02–EXP_05 (RAG) |
|---|---|---|---|
| Faithfulness | yes | skip | run |
| Context Precision | yes | skip | run |
| Context Recall | yes | skip | run |
| Answer Relevancy | no | run | run |
| Answer Correctness | no | run | run |

For No-RAG, the four context-dependent fields are written as `null` in
`summary.json` (Option A — N/A by design, defended in
`docs/output_notes/04a_exp01_output.md` §3 and `docs/results_schema.md`).

**Resumability** is file-level: per-row scores are written to
`ragas_scores.csv` and `summary.json` is updated atomically. Re-running with
the same predictions detects an existing scores CSV and skips judging.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]

JUDGE_MODEL = "claude-sonnet-4-6"
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 1024
EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"

# RunConfig — passed to RAGAS `evaluate()`. Tightened from defaults
# (`max_workers=16`) after EXP_01's 2026-05-06 run produced ~40 % NaN scores
# on Anthropic-side rate-limit throttles. Lowering concurrency to 4 keeps the
# request rate under Anthropic's per-minute cap; bumping `max_wait` to 120 s
# gives sustained-throttle waits more headroom; default `max_retries=10` and
# `timeout=180` are unchanged.
RAGAS_MAX_WORKERS = 4
RAGAS_MAX_WAIT = 120
RAGAS_MAX_RETRIES = 10
RAGAS_TIMEOUT = 180

# Metric set — names map to the RAGAS legacy lowercase singletons. RAGAS 0.4
# has two parallel metric hierarchies: `ragas.metrics.collections.*`
# (capitalized; the modern InstructorLLM path) and the legacy lowercase
# pre-built singletons in `ragas.metrics` that pass `evaluate()`'s isinstance
# check. The collections classes only work via direct `await metric.single_turn_ascore(sample)`
# calls — they are NOT recognised by `evaluate()` (different `BaseMetric` tree).
# We use the legacy path here because (a) `evaluate()` is the cleanest batch
# orchestrator and (b) the deprecation removal is scheduled for RAGAS v1.0,
# well after thesis submission.
CONTEXT_DEPENDENT = {"faithfulness", "context_precision", "context_recall"}
CONTEXT_INDEPENDENT = {"answer_relevancy", "answer_correctness"}
ALL_METRIC_NAMES = list(CONTEXT_DEPENDENT) + list(CONTEXT_INDEPENDENT)


def _get_metric_singleton(name: str):
    """Return the legacy pre-built RAGAS metric singleton for `name`.
    `evaluate()` injects `llm` + `embeddings` into these at run time.
    Lazy import to keep this module loadable when RAGAS isn't installed
    (e.g. during the unit tests that only exercise dataset construction)."""
    import warnings

    with warnings.catch_warnings():
        # The lowercase imports trip a deprecation warning on every call;
        # silence it locally — the migration target (`metrics.collections`)
        # is incompatible with `evaluate()`, so the deprecation isn't
        # actionable for us until RAGAS ships an `evaluate_collections()`.
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas.metrics import (
            answer_correctness,
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

    return {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "answer_correctness": answer_correctness,
    }[name]


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


def _load_predictions(predictions_path: Path) -> pd.DataFrame:
    return pd.DataFrame(
        json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()
    )


def _load_retrieval(retrieval_path: Path) -> pd.DataFrame:
    return pd.DataFrame(
        json.loads(line) for line in retrieval_path.read_text(encoding="utf-8").splitlines() if line.strip()
    )


def _load_golden_by_qid(golden_path: Path) -> dict[int, dict]:
    """Index the golden file by its integer `question_id` (0..299)."""
    rows = [json.loads(line) for line in golden_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {int(r["question_id"]): r for r in rows if r.get("final_status") == "accepted"}


def _golden_qid_from_pred(pred_qid: str) -> int | None:
    """`golden_NNN` → `NNN`. Returns None for non-golden ids."""
    if not pred_qid.startswith("golden_"):
        return None
    try:
        return int(pred_qid.split("_", 1)[1])
    except (ValueError, IndexError):
        return None


def build_ragas_rows(
    predictions: pd.DataFrame,
    retrieval: pd.DataFrame,
    golden_by_qid: dict[int, dict],
    chunks_by_id: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Join predictions + retrieval + golden into one row per question.

    Output columns (RAGAS-ready + bookkeeping):
        question_id, user_input, response, retrieved_contexts (list[str]),
        reference, _pred_letter, _gold_letter, _is_correct, _has_context.
    """
    chunks_by_id = chunks_by_id or {}
    ret_by_qid = dict(zip(retrieval["question_id"], retrieval["retrieved_chunk_ids"]))

    rows: list[dict[str, Any]] = []
    for _, p in predictions.iterrows():
        qid = p["question_id"]
        gold_qid = _golden_qid_from_pred(qid)
        if gold_qid is None or gold_qid not in golden_by_qid:
            # Skip rows without a corresponding golden entry — RAGAS can't score
            # them (no `reference` text available for Answer Correctness).
            continue
        g = golden_by_qid[gold_qid]
        # Convert the predicted letter to its full option text. Comparing
        # letters against the prose `gold_answer_text` would tank Answer
        # Correctness for the wrong reason.
        pred_letter = p["pred_letter"]
        if pred_letter and pred_letter in g["options"]:
            response = g["options"][pred_letter]
        else:
            response = pred_letter or "(no answer parsed)"
        chunk_ids = ret_by_qid.get(qid, []) or []
        retrieved_contexts = [chunks_by_id.get(cid, "") for cid in chunk_ids]
        # RAGAS dataset validation rejects empty-list contexts; pass a single
        # empty-string placeholder for No-RAG. Context-dependent metrics will
        # be skipped at the metric-set level, so this placeholder is never
        # actually consumed.
        if not retrieved_contexts:
            retrieved_contexts = [""]
        rows.append(
            {
                "question_id": qid,
                "user_input": g["question"],
                "response": response,
                "retrieved_contexts": retrieved_contexts,
                "reference": g["gold_answer_text"],
                "_pred_letter": pred_letter,
                "_gold_letter": p["gold_letter"],
                "_is_correct": bool(p["is_correct"]),
                "_has_context": len(chunk_ids) > 0,
            }
        )
    return pd.DataFrame(rows)


def applicable_metrics(rows: pd.DataFrame) -> list[str]:
    """Choose the metric set based on whether any row has retrieved context.

    No-RAG (`_has_context` False everywhere) → only context-independent metrics.
    Any RAG (at least one row with context) → all five.
    """
    return ALL_METRIC_NAMES if rows["_has_context"].any() else sorted(CONTEXT_INDEPENDENT)


# ---------------------------------------------------------------------------
# Judge wiring
# ---------------------------------------------------------------------------


def build_judge_and_embeddings(
    judge_model: str = JUDGE_MODEL,
    embed_model: str = EMBED_MODEL_NAME,
):
    """Construct the LangChain Claude judge + LangChain-wrapped BGE embedder.

    `evaluate()` accepts metrics from the legacy `ragas.metrics.base.Metric`
    tree, which expects a `LangchainLLMWrapper`-compatible LLM (the modern
    `InstructorLLM` from `ragas.llms.llm_factory` is incompatible — collections
    metrics that take it aren't recognised by `evaluate()`). So we wrap
    `ChatAnthropic` and `HuggingFaceEmbeddings` (sentence-transformers BGE-large)
    via the LangChain bridges that RAGAS provides. File-level resumability via
    `ragas_scores.csv` is sufficient for our 234-row scale.

    Lazy imports keep this module loadable in environments without RAGAS or
    Anthropic installed (e.g. the unit tests that only exercise dataset
    construction).
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is missing. Add it to .env at the repo root and "
            "reload — RAGAS judging requires the Anthropic API."
        )
    from langchain_anthropic import ChatAnthropic
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    llm = LangchainLLMWrapper(
        ChatAnthropic(
            model=judge_model,
            temperature=JUDGE_TEMPERATURE,
            max_tokens=JUDGE_MAX_TOKENS,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=embed_model))
    return llm, embeddings


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def score_predictions(
    predictions_dir: Path,
    golden_path: Path,
    chunks_path: Path | None = None,
    *,
    n_smoke: int | None = None,
    judge_model: str = JUDGE_MODEL,
    embed_model: str = EMBED_MODEL_NAME,
    overwrite: bool = False,
    rescore_nans: bool = False,
) -> dict:
    """Run RAGAS on `<predictions_dir>/predictions.jsonl` and update
    `summary.json` with the aggregate scores.

    Side effects:
        - Writes `<predictions_dir>/ragas_scores.csv` (per-row scores +
          bookkeeping columns).
        - Updates `<predictions_dir>/summary.json` in place — adds
          `RAGAS_Faithfulness`, `RAGAS_Hallucination_Rate`,
          `RAGAS_Answer_Relevance`, `RAGAS_Context_Precision`,
          `RAGAS_Context_Recall`, `Answer_Correctness`, plus the
          `ragas_*` provenance keys.

    Three modes — pick one:

    - **Fresh run** (`overwrite=False`, `rescore_nans=False`, no CSV exists):
      score every joined row.
    - **Cache hit** (`overwrite=False`, `rescore_nans=False`, CSV exists):
      skip the judge, refresh `summary.json` from the existing CSV.
    - **Full re-run** (`overwrite=True`): re-score every row, replacing the
      existing CSV.
    - **NaN-only rescore** (`rescore_nans=True`): identify rows where any
      applicable metric is NaN in the existing CSV, re-score *only those rows*,
      merge the new scores back in (replace NaN cells, preserve good ones).
      Cheap recovery path for the rate-limit-induced NaN issue documented in
      `docs/output_notes/04a_exp01_output.md` §4.3.

    Parameters:
        n_smoke: if set, score only the first `n_smoke` rows (use for the
            ~10-row Stage A pilot before scaling).
        overwrite: full re-run, ignoring any existing `ragas_scores.csv`.
        rescore_nans: re-score only the NaN rows from an existing
            `ragas_scores.csv`. Mutually exclusive with `overwrite`.
    """
    if overwrite and rescore_nans:
        raise ValueError("`overwrite` and `rescore_nans` are mutually exclusive — pick one.")

    predictions_dir = Path(predictions_dir)
    pred_path = predictions_dir / "predictions.jsonl"
    ret_path = predictions_dir / "retrieval.jsonl"
    summary_path = predictions_dir / "summary.json"
    scores_path = predictions_dir / "ragas_scores.csv"

    # Build the join
    predictions = _load_predictions(pred_path)
    retrieval = _load_retrieval(ret_path)
    golden_by_qid = _load_golden_by_qid(golden_path)
    chunks_by_id: dict[str, str] = {}
    if chunks_path is not None and chunks_path.exists():
        chunks_df = pd.read_parquet(chunks_path)
        chunks_by_id = dict(zip(chunks_df["chunk_id"], chunks_df["text"]))
    rows = build_ragas_rows(predictions, retrieval, golden_by_qid, chunks_by_id)
    if rows.empty:
        raise RuntimeError(
            f"No rows joined — check that {pred_path} contains golden_NNN ids "
            f"and that {golden_path} is the canonical accepted-only file."
        )

    if n_smoke is not None:
        rows = rows.head(n_smoke).reset_index(drop=True)

    metric_names = applicable_metrics(rows)
    print(f"[ragas] joined {len(rows)} rows | metrics = {metric_names}")

    if rescore_nans:
        # NaN-only path — re-score just the rows whose existing scores are NaN
        if not scores_path.exists():
            raise RuntimeError(
                f"`rescore_nans=True` but {scores_path.name} doesn't exist — "
                f"there's nothing to rescore. Run with `rescore_nans=False` first."
            )
        existing = pd.read_csv(scores_path)
        nan_qids = _nan_question_ids(existing, metric_names)
        if not nan_qids:
            print("[ragas] no NaN rows to rescore — already complete")
            result_df = existing
        else:
            partial = rows[rows["question_id"].isin(nan_qids)].reset_index(drop=True)
            print(f"[ragas] rescoring {len(partial)} NaN rows ({len(nan_qids)} unique question_ids)")
            partial_result = _run_judge(partial, metric_names, judge_model, embed_model)
            result_df = _merge_partial_scores(existing, partial_result, metric_names)
            result_df.to_csv(scores_path, index=False)
            print(f"[ragas] merged + wrote {scores_path}")
    elif scores_path.exists() and not overwrite:
        # Resumability — file-level skip
        print(f"[ragas] {scores_path.name} already exists → skipping judge, refreshing summary only")
        result_df = pd.read_csv(scores_path)
    else:
        # Fresh run / full overwrite
        result_df = _run_judge(rows, metric_names, judge_model, embed_model)
        result_df.to_csv(scores_path, index=False)
        print(f"[ragas] wrote {scores_path}")

    summary = _refresh_summary(summary_path, result_df, metric_names, judge_model)
    return summary


def _nan_question_ids(existing: pd.DataFrame, metric_names: list[str]) -> set[str]:
    """Return the set of question_ids in `existing` whose row has NaN in any
    of the active metric columns."""
    cols = [m for m in metric_names if m in existing.columns]
    if not cols:
        return set()
    nan_mask = existing[cols].apply(pd.to_numeric, errors="coerce").isna().any(axis=1)
    return set(existing.loc[nan_mask, "question_id"].astype(str))


def _merge_partial_scores(
    existing: pd.DataFrame,
    partial: pd.DataFrame,
    metric_names: list[str],
) -> pd.DataFrame:
    """Update `existing` in place: for each row in `partial`, replace NaN
    cells in the matching `existing` row with the new score. Cells that
    already had a non-NaN score in `existing` are preserved (we don't
    overwrite a previously-good score with a re-judged one).

    Match key is `question_id`. Returns the merged DataFrame.
    """
    out = existing.copy()
    cols = [m for m in metric_names if m in partial.columns and m in out.columns]
    out_idx = {str(qid): i for i, qid in enumerate(out["question_id"])}
    n_updated = 0
    for _, prow in partial.iterrows():
        qid = str(prow["question_id"])
        if qid not in out_idx:
            continue
        i = out_idx[qid]
        for col in cols:
            old = pd.to_numeric(out.at[i, col], errors="coerce")
            new = pd.to_numeric(prow[col], errors="coerce")
            if pd.isna(old) and pd.notna(new):
                out.at[i, col] = new
                n_updated += 1
    print(f"[ragas] merge: updated {n_updated} cells across {len(cols)} metric columns")
    return out


def _run_judge(
    rows: pd.DataFrame,
    metric_names: list[str],
    judge_model: str,
    embed_model: str,
) -> pd.DataFrame:
    """The expensive bit — calls Claude. Caller wraps with file-level cache."""
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.run_config import RunConfig

    llm, embeddings = build_judge_and_embeddings(judge_model, embed_model)
    # Legacy lowercase singletons — `evaluate()` injects llm + embeddings into them.
    metrics = [_get_metric_singleton(name) for name in metric_names]

    samples = [
        SingleTurnSample(
            user_input=r["user_input"],
            response=r["response"],
            retrieved_contexts=r["retrieved_contexts"],
            reference=r["reference"],
        )
        for _, r in rows.iterrows()
    ]
    dataset = EvaluationDataset(samples=samples)

    # Conservative throughput config — see module-level RAGAS_* constants for rationale.
    run_config = RunConfig(
        timeout=RAGAS_TIMEOUT,
        max_retries=RAGAS_MAX_RETRIES,
        max_wait=RAGAS_MAX_WAIT,
        max_workers=RAGAS_MAX_WORKERS,
    )
    print(
        f"[ragas] RunConfig: max_workers={RAGAS_MAX_WORKERS}, "
        f"max_retries={RAGAS_MAX_RETRIES}, max_wait={RAGAS_MAX_WAIT}s, timeout={RAGAS_TIMEOUT}s"
    )

    t0 = time.time()
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
        raise_exceptions=False,
    )
    wall_s = time.time() - t0
    print(f"[ragas] judge wall-time {wall_s:.1f} s for {len(rows)} rows × {len(metric_names)} metrics")

    result_df = result.to_pandas()
    # Re-attach bookkeeping columns so the CSV is inspectable on its own
    for col in ["question_id", "_pred_letter", "_gold_letter", "_is_correct", "_has_context"]:
        result_df[col] = rows[col].values
    return result_df


def _refresh_summary(
    summary_path: Path,
    result_df: pd.DataFrame,
    metric_names: list[str],
    judge_model: str,
) -> dict:
    """Merge RAGAS aggregates into the existing summary.json."""
    if not summary_path.exists():
        raise RuntimeError(
            f"summary.json missing at {summary_path} — run the experiment notebook first."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    def _mean_or_none(col: str) -> float | None:
        if col not in result_df.columns:
            return None
        s = pd.to_numeric(result_df[col], errors="coerce").dropna()
        return float(s.mean()) if len(s) else None

    def _hallucination_rate() -> float | None:
        if "faithfulness" not in result_df.columns:
            return None
        s = pd.to_numeric(result_df["faithfulness"], errors="coerce").dropna()
        return float((s < 0.5).mean()) if len(s) else None

    summary["RAGAS_Faithfulness"] = _mean_or_none("faithfulness")
    summary["RAGAS_Hallucination_Rate"] = _hallucination_rate()
    summary["RAGAS_Answer_Relevance"] = _mean_or_none("answer_relevancy")
    summary["RAGAS_Context_Precision"] = _mean_or_none("context_precision")
    summary["RAGAS_Context_Recall"] = _mean_or_none("context_recall")
    summary["Answer_Correctness"] = _mean_or_none("answer_correctness")
    summary["ragas_metrics_run"] = metric_names
    summary["ragas_n_scored"] = int(len(result_df))
    summary["ragas_judge"] = judge_model
    summary["ragas_timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
