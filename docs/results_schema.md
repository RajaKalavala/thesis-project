# Results Schema — `summary.json` Lock

> **Purpose.** Pin the shape of `results/exp_*/summary.json` once, here, so EXP_01 → EXP_16 all write the same keys and the optional Streamlit demo (Phase 10) can read every experiment uniformly. Per [`plan.md` §13](../plan.md#13-the-12-results-tables-—-coverage-checklist) and [`plan.md` §10.5 Stage A](../plan.md#10·-phase-10-—-demo-ui-optional-parallel-track), locking this schema *before* expensive runs prevents rework later.
>
> **Companion files:** [`plan.md`](../plan.md) (locked decisions) · [`docs/architecture.md` §3](architecture.md) (`src/eval/runner.py` contract) · `docs/thesis-files/Raja Kalavala Final Thesis Project Sheet.xlsx` (the 12 results tables).

---

## 1. The three artefacts every experiment writes

```
results/<experiment_id>__<dataset_label>/
├── predictions.jsonl   ← one row per question
├── retrieval.jsonl     ← one row per question (empty `retrieved_chunk_ids` for No-RAG)
└── summary.json        ← one paste-into-Excel row
```

The runner (`src/eval/runner.py::run_experiment`) writes all three. Each line in `predictions.jsonl` and `retrieval.jsonl` is **independent** so partial runs are still inspectable.

---

## 2. `summary.json` — the canonical row

### 2.1 Field-by-field

| Field | Type | Source / how filled | Excel column |
|---|---|---|---|
| `experiment_id` | str | Caller (e.g. `"EXP_01_BASE_LLM"`) | row identity |
| `dataset` | str | `"smoke_50"` / `"golden_234"` / `"full_12723"` | row identity |
| `n_questions` | int | rows actually scored | — |
| `Generator_Model` | str | `"llama-3.3-70b-versatile"` per [`plan.md` §0 #1](../plan.md) | Table 1 *Generator Model* |
| `temperature` | float | `0.0` per Excel `Experiments Guide` | — |
| `max_tokens` | int | `700` per Excel `Experiments Guide` | — |
| `Acuuracy` | float ∈ [0, 1] | `n_correct / n_questions` | Table 1 *Acuuracy* (Excel typo preserved) |
| `Exact_Match` | float ∈ [0, 1] | same numerator as `Acuuracy` for MCQ; kept separate to mirror the workbook | Table 1 *Exact Match* |
| `n_correct` | int | sum of `is_correct` in `predictions.jsonl` | — |
| `Recall@3` / `Recall@5` / `Recall@10` | float ∈ [0, 1] \| null | mean over golden rows of `recall_at_k(retrieved, gold_chunks, k)` from `non_llm_metrics.py`. `null` for EXP_01 (no retrieval) and for non-golden surfaces (no chunk-level ground truth). | Table 1 / Table 8 |
| `MRR` | float ∈ [0, 1] \| null | mean over golden rows | Table 1 / Table 8 |
| `RAGAS_Faithfulness` | float ∈ [0, 1] \| null | filled by `src/eval/ragas_eval.py` (Claude Sonnet 4.6 judge, golden-234 only) — **null for EXP_01 No-RAG** (see §2.3) | Table 1 |
| `RAGAS_Hallucination_Rate` | float ∈ [0, 1] \| null | fraction of golden rows with `RAGAS_Faithfulness < 0.5`. null whenever Faithfulness is null | Table 1 |
| `RAGAS_Answer_Relevance` | float ∈ [0, 1] \| null | RAGAS judge — context-independent, runs for every architecture | Table 1 |
| `RAGAS_Context_Precision` | float ∈ [0, 1] \| null | RAGAS judge (golden-only) — **null for EXP_01 No-RAG** (see §2.3) | Table 1 |
| `RAGAS_Context_Recall` | float ∈ [0, 1] \| null | RAGAS judge (golden-only) — **null for EXP_01 No-RAG** (see §2.3) | Table 1 |
| `Answer_Correctness` | float ∈ [0, 1] \| null | RAGAS judge — context-independent, runs for every architecture | Table 1 |
| `ragas_metrics_run` | list[str] \| absent | which metrics were actually scored (e.g. `["answer_relevancy", "answer_correctness"]` for EXP_01) | provenance |
| `ragas_n_scored` | int \| absent | number of golden rows that received scores | provenance |
| `ragas_judge` | str \| absent | judge model id (e.g. `"claude-sonnet-4-6"`) | provenance |
| `ragas_timestamp_utc` | str (ISO-8601) \| absent | when judging finished | provenance |
| `mean_latency_s` | float | mean of per-row `latency_s` across the **whole `predictions.jsonl`** (including resumed sessions) | Table 1 *Latency* |
| `wall_time_s_this_run` | float | wall time for the *current* invocation; resumed sessions reset this | — |
| `n_calls_this_run` | int | API calls this invocation made (excludes resume-skipped rows) | — |
| `cache_hits_this_run` | int | of those, how many hit the disk cache | — |
| `cache_hit_rate_this_run` | float ∈ [0, 1] | `cache_hits_this_run / n_calls_this_run` | — |
| `timestamp_utc` | str (ISO-8601) | `time.gmtime()` at `run_experiment` end | — |

### 2.2 The `null` policy

A field is `null` (not `0`, not `"N/A"`) when the experiment didn't compute it. Three reasons a field can legitimately be `null`:

1. **Architecturally inapplicable** — e.g. `Recall@K` for EXP_01 (no retrieval).
2. **Surface inapplicable** — e.g. `Recall@K` on `full_12723` (no chunk-level ground truth; only golden has it).
3. **Module not yet built** — e.g. all `RAGAS_*` keys until `src/eval/ragas_eval.py` lands.

`null` survives the Excel paste as a literal blank cell, which is the right signal for "we didn't measure this." `0.0` would falsely imply "we measured this and it was zero."

### 2.3 RAGAS metric applicability — per architecture

Three of the five RAGAS metrics depend on `retrieved_contexts`. EXP_01 is the No-RAG baseline — `retrieved_contexts = []` for every row — so context-dependent metrics are **not measured** rather than **set to zero**. This is *Option A* in the EXP_01 design ([discussion in `docs/output_notes/04a_exp01_output.md`](output_notes/04a_exp01_output.md)).

| Metric | Needs context? | EXP_01 (No-RAG) | EXP_02 Naive · EXP_03 Sparse · EXP_04 Hybrid · EXP_05 Multi-Hop |
|---|---|---|---|
| `RAGAS_Faithfulness` | yes | **null** (undefined) | computed |
| `RAGAS_Hallucination_Rate` | derived from Faithfulness | **null** | computed (fraction with Faithfulness < 0.5) |
| `RAGAS_Context_Precision` | yes | **null** | computed |
| `RAGAS_Context_Recall` | yes | **null** | computed |
| `RAGAS_Answer_Relevance` | no | computed | computed |
| `Answer_Correctness` | no | computed | computed |

The runner detects applicability automatically — `src/eval/ragas_eval.py::applicable_metrics()` returns the context-independent subset when `_has_context` is `False` for every row, the full set otherwise. No per-experiment configuration needed.

**Methodology defence**: assigning `RAGAS_Faithfulness = 0` to No-RAG would assert "100 % hallucination" without ever running the judge, which is harder to defend in a viva than reporting `null` and explaining that *"context-grounded faithfulness is undefined when no context is retrieved"*. The thesis discussion still gets to make the *"RAG enables faithfulness scoring at all"* point — just framed as architectural, not numerical.

### 2.4 Excel-column-name fidelity

Field names mirror Excel headers verbatim, **including the `Acuuracy` typo**. Reason: the workbook is the source of truth for table headers; renaming on our side guarantees a headers mismatch when pasting. If the workbook header is later corrected, this schema follows in lockstep.

---

## 3. `predictions.jsonl` — one row per question

```json
{
  "question_id": "medqa_00000",
  "gold_letter": "D",
  "pred_letter": "D",
  "raw_response": "D",
  "latency_s": 1.31,
  "was_cached": false,
  "is_correct": true
}
```

`question_id` formats by surface:

| Surface | Format | Source |
|---|---|---|
| `full_12723` / `smoke_*` (medqa-derived) | `medqa_NNNNN` | row index in `data/processed/medqa_4opt.parquet` |
| `golden_234` | `golden_NNN` | `question_id` integer in `data/processed/golden_ragas_300.jsonl` (the stratified-sample index 0..299) |

The two id namespaces are deliberately distinct because **golden's `question_id` is NOT the medqa_4opt row index** (per [`docs/dataset.md` §2.2](dataset.md)). To bridge the two, join by `question` text — `predictions.jsonl` doesn't carry the question text to keep file size bounded; load it from the source dataset.

---

## 4. `retrieval.jsonl` — one row per question

```json
{
  "question_id": "medqa_00000",
  "retrieved_chunk_ids": [],
  "retrieved_chunk_scores": []
}
```

Both lists are empty for EXP_01 (No-RAG) by design — keeping the file present means downstream code (e.g. UI loaders) doesn't need a special case for "this experiment has no retrieval."

For EXP_02–EXP_05, lists carry up to `k` (default 5) chunk IDs ranked best-first. Score semantics differ per retriever (cosine for dense, BM25 for sparse, RRF for hybrid) — comparable *within* an experiment but not *across* experiments.

---

## 5. Versioning policy

| Change to schema | Action |
|---|---|
| Add a new field | Bump nothing; document the field here with its source. Existing experiments' `summary.json` files just lack the field. |
| Rename or remove a field | Forbidden once a single experiment has shipped to `results/`. Add a new field instead and deprecate the old one with a comment. |
| Change a field's type | Treat as rename + remove. |

The runner does **not** hard-validate the schema on write — the goal is observability, not enforcement. If the Streamlit demo (Phase 10) is built, Stage A's CI-style assert will validate at read time.

---

*Last refreshed: 2026-05-05. When this drifts from `src/eval/runner.py`, **the runner wins** and this doc gets resynced — but every drift should be deliberate and reflected in [`docs/todo.md` decision log](todo.md).*
