"""Generic experiment runner — one entry point for all of EXP_01–EXP_05.

`run_experiment(retriever, dataset_df, output_dir, ...)` loops over questions,
calls the retriever, builds the prompt (No-RAG when retrieval is empty,
evidence-grounded otherwise), calls Groq through the disk-cached client, parses
the predicted letter, and streams three artefacts:

- `predictions.jsonl` — one row per question (`question_id`, `gold_letter`,
  `pred_letter`, `raw_response`, `latency_s`, `was_cached`, `is_correct`).
- `retrieval.jsonl`   — one row per question (`question_id`,
  `retrieved_chunk_ids`, `retrieved_chunk_scores`). Empty list for No-RAG.
- `summary.json`      — paste-into-Excel row. Schema locked in
  `docs/results_schema.md`; RAGAS keys ship as `null` until the judge runs.

**Resumability** is the load-bearing property: every long Groq run dies
eventually (rate limit, network, laptop sleep). The runner reads any existing
`predictions.jsonl` at startup and skips question_ids it already processed.
Combined with the `groq_complete` disk cache, restarts are free.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
from tqdm.auto import tqdm

from src.eval.non_llm_metrics import exact_match
from src.generation.groq_client import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    groq_complete,
)
from src.generation.prompts import (
    build_evidence_grounded_prompt,
    build_no_rag_prompt,
    parse_letter,
)
from src.retrieval.base import Retriever


# Workbook-aligned EXP_01 spec (Excel: Experiments Guide row "EXP_01_BASE_LLM"):
# temperature=0, max_tokens=700. The same defaults apply to EXP_02–EXP_05.
EXP_DEFAULT_MAX_TOKENS = 700
EXP_DEFAULT_TEMPERATURE = 0.0


@dataclass
class _RowOutcome:
    question_id: str
    gold_letter: str
    pred_letter: str | None
    raw_response: str
    latency_s: float
    was_cached: bool
    is_correct: bool
    retrieved_chunk_ids: list[str]
    retrieved_chunk_scores: list[float]


def _row_to_inputs(row: pd.Series | dict, dataset_label: str) -> tuple[str, str, dict[str, str], str]:
    """Pull (question_id, question_text, options_dict, gold_letter) from either
    a medqa_4opt row or a golden-set row.
    """
    if dataset_label.startswith("golden"):
        # Golden rows: question_id is int (0..299), options is already a dict,
        # gold answer is `gold_answer_letter`.
        qid = f"golden_{row['question_id']:03d}"
        question = row["question"]
        options = row["options"]
        gold = row["gold_answer_letter"]
    else:
        # medqa_4opt rows from load_medqa_4opt: question_id is "medqa_NNNNN",
        # options is a dict (parsed from options_json), gold is `answer_idx`.
        qid = row["question_id"]
        question = row["question"]
        options = row["options"]
        gold = row["answer_idx"]
    return qid, question, options, gold


def _load_completed_ids(predictions_path: Path) -> set[str]:
    """Return question_ids already present in `predictions.jsonl` so the
    runner can resume mid-stream. Tolerates a partially-written final line."""
    if not predictions_path.exists():
        return set()
    completed: set[str] = set()
    for line in predictions_path.read_text(encoding="utf-8").splitlines():
        try:
            completed.add(json.loads(line)["question_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return completed


def run_experiment(
    retriever: Retriever,
    dataset: pd.DataFrame | list[dict],
    output_dir: Path,
    experiment_id: str,
    dataset_label: str,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = EXP_DEFAULT_TEMPERATURE,
    max_tokens: int = EXP_DEFAULT_MAX_TOKENS,
    k: int = 5,
    progress: bool = True,
) -> dict:
    """Run an experiment end-to-end and return the `summary.json` dict.

    Parameters
    ----------
    retriever
        Any `Retriever`. `NoRetrieval` ⇒ No-RAG prompt path.
    dataset
        Either a `pd.DataFrame` from `load_medqa_4opt()` or a `list[dict]`
        from `load_golden()`. `dataset_label` controls the row-shape parser.
    output_dir
        Created if missing. Three files written: `predictions.jsonl`,
        `retrieval.jsonl`, `summary.json`.
    experiment_id
        Free-form (e.g. `"EXP_01_BASE_LLM"`). Echoed into `summary.json`.
    dataset_label
        One of `"smoke_50"`, `"golden_234"`, `"full_12723"` — distinguishes
        which evaluation surface is in play and selects the row parser.
        Strings starting with `"golden"` use the golden-row path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    retrieval_path = output_dir / "retrieval.jsonl"
    summary_path = output_dir / "summary.json"

    # iterate uniformly whether dataset is a DataFrame or a list of dicts
    rows: list = list(dataset.to_dict(orient="records")) if isinstance(dataset, pd.DataFrame) else list(dataset)
    completed = _load_completed_ids(predictions_path)
    if completed:
        print(f"[runner] resuming: {len(completed)} of {len(rows)} already done")

    outcomes: list[_RowOutcome] = []
    cache_hits = 0
    n_calls = 0
    t_start = time.time()

    iterator = tqdm(rows, desc=experiment_id, disable=not progress)
    with predictions_path.open("a", encoding="utf-8") as f_pred, retrieval_path.open("a", encoding="utf-8") as f_ret:
        for row in iterator:
            qid, question, options, gold = _row_to_inputs(row, dataset_label)
            if qid in completed:
                continue

            chunks = retriever.retrieve(question, k=k)
            if chunks:
                prompt = build_evidence_grounded_prompt(question, options, [c.text for c in chunks])
            else:
                prompt = build_no_rag_prompt(question, options)

            text, latency, was_cached = groq_complete(
                prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            n_calls += 1
            if was_cached:
                cache_hits += 1

            pred = parse_letter(text)
            outcome = _RowOutcome(
                question_id=qid,
                gold_letter=gold,
                pred_letter=pred,
                raw_response=text,
                latency_s=latency,
                was_cached=was_cached,
                is_correct=exact_match(pred, gold),
                retrieved_chunk_ids=[c.chunk_id for c in chunks],
                retrieved_chunk_scores=[c.score for c in chunks],
            )
            outcomes.append(outcome)

            f_pred.write(
                json.dumps(
                    {
                        "question_id": outcome.question_id,
                        "gold_letter": outcome.gold_letter,
                        "pred_letter": outcome.pred_letter,
                        "raw_response": outcome.raw_response,
                        "latency_s": outcome.latency_s,
                        "was_cached": outcome.was_cached,
                        "is_correct": outcome.is_correct,
                    }
                )
                + "\n"
            )
            f_ret.write(
                json.dumps(
                    {
                        "question_id": outcome.question_id,
                        "retrieved_chunk_ids": outcome.retrieved_chunk_ids,
                        "retrieved_chunk_scores": outcome.retrieved_chunk_scores,
                    }
                )
                + "\n"
            )

    wall_time_s = time.time() - t_start

    # Re-load full predictions stream for the summary so resumed runs roll up
    # correctly (in-memory `outcomes` only covers this invocation).
    all_preds = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    n = len(all_preds)
    n_correct = sum(1 for r in all_preds if r.get("is_correct"))
    summary = {
        # identity
        "experiment_id": experiment_id,
        "dataset": dataset_label,
        "n_questions": n,
        # answerer config (Excel "Generator Model" + temp/max_tokens cells)
        "Generator_Model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        # answer-side metrics — Excel column names mirrored verbatim, with `Acuuracy` typo preserved
        "Acuuracy": n_correct / n if n else 0.0,
        "Exact_Match": n_correct / n if n else 0.0,
        "n_correct": n_correct,
        # retrieval-side metrics (filled by EXP_02+; null for EXP_01)
        "Recall@3": None,
        "Recall@5": None,
        "Recall@10": None,
        "MRR": None,
        # RAGAS suite (filled by src/eval/ragas_eval.py once it lands)
        "RAGAS_Faithfulness": None,
        "RAGAS_Hallucination_Rate": None,
        "RAGAS_Answer_Relevance": None,
        "RAGAS_Context_Precision": None,
        "RAGAS_Context_Recall": None,
        "Answer_Correctness": None,
        # operations / observability
        "mean_latency_s": (sum(r["latency_s"] for r in all_preds) / n) if n else 0.0,
        "wall_time_s_this_run": wall_time_s,
        "n_calls_this_run": n_calls,
        "cache_hits_this_run": cache_hits,
        "cache_hit_rate_this_run": (cache_hits / n_calls) if n_calls else 0.0,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
