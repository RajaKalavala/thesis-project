# Thesis Working TODO

> **Purpose.** Single source of truth for *what's done, what's next, what's blocked*. Update as you go. Aligned 1:1 with [plan.md](../plan.md) phases — but more granular and actionable.
>
> **Convention.**
> - `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked / needs decision
> - Each step has a **deliverable** so you can tell when it's actually done
> - When you finish a step, mark `[x]` and note the date in `(YYYY-MM-DD)` or commit hash if useful

---

## 0 · Setup & locked decisions

- [x] Python 3.12 venv at `.venv/`, requirements installed, Jupyter kernel `thesis-rag` registered
- [x] `.gitignore` covers venv, processed data, indices, results, secrets
- [x] All 11 stack decisions locked in [plan.md §0](../plan.md#0-locked-decisions)
- [ ] `.env` populated with **two required** API keys (recalibrated 2026-05-04): `GROQ_API_KEY` (generator + constructor) and `ANTHROPIC_API_KEY` (RAGAS judge). `OPENAI_API_KEY` is now optional — only needed if the open-weights constructor pilot fails.
  - Deliverable: `.venv/bin/python -c "import os; from dotenv import load_dotenv; load_dotenv(); print({k: '✓' if os.getenv(k) else '✗' for k in ['GROQ_API_KEY','ANTHROPIC_API_KEY']})"` shows both ✓

---

## 1 · Data processing & EDA — ✅ COMPLETE

- [x] `notebooks/00_data_processing_and_eda.ipynb` runs end-to-end inside the venv
- [x] `data/processed/medqa_5opt.parquet` — 12,723 rows
- [x] `data/processed/medqa_4opt.parquet` — 12,723 rows + metamap_phrases
- [x] `data/processed/textbook_stats.parquet` — 18 books
- [x] `data/processed/eda_summary.json` — headline numbers
- [x] [docs/dataset.md](dataset.md) — field-level reference

---

## 2 · Shared infrastructure (chunking + dual embedding + indices)

### 2.1 Notebook 01 — `notebooks/01_chunking_and_corpus_prep.ipynb` ✅ COMPLETE (2026-05-03)

- [x] Load all 18 textbooks from `medqa-data/textbooks/en/*.txt` (2026-05-03)
- [x] Recursive 400-token chunker, **80-token overlap**, tiktoken cl100k_base for token counting (2026-05-03)
- [x] Drop chunks <30 tokens; assign deterministic `chunk_id` (`<book>_chunk_<5-digit>`) (2026-05-03)
- [x] Save `data/processed/chunks.parquet` with columns: `chunk_id`, `book_name`, `text`, `n_tokens`, `n_chars` (2026-05-03)
- [x] Acceptance (recalibrated 2026-05-03): **~50k–75k** total chunks (the original "~32k–40k" was mathematically unreachable for a 12.85 M-word corpus with 400/80 config); mean **300–380** tokens; max ≤ 450; Harrison's still ~25% of chunks (2026-05-03 — actual: **67,599 chunks, mean 323.9 tokens, max ≤ 400, Harrison's 24.66 %**)
- **Deliverable:** `chunks.parquet` exists (~30 MB, 67,599 rows); per-book chunk-count bar chart + token-distribution histogram printed in the notebook ✓

### 2.2 Notebook 02 — `notebooks/02_embeddings_and_indices.ipynb` ✅ COMPLETE (2026-05-04)

- [x] Embed all chunks with **`BAAI/bge-large-en-v1.5`** (sentence-transformers; batch 32; **actual: ~355 min on M1 Pro MPS**, far above the original `MPS ≈ 12 min` / recalibrated `~22 min` estimates — sustained-load thermal throttling and/or partial-MPS coverage on BGE-large's 1024-d attention) (2026-05-03)
  - Saves `data/processed/embeddings.npy` — 67,599 × 1,024 float32, **276.9 MB**, all rows L2-normalised (norm = 1.0000)
- [x] Build ChromaDB collection `medqa_textbooks_bge_400` at `data/indices/chroma_textbooks/` (cosine HNSW, metadata = `{book_name, n_tokens}`) — 67,599 vectors, **1,124.2 MB on disk**, build wall-time 1m 39s (2026-05-04)
- [x] Build BM25 index at `data/indices/bm25.pkl` — **105.8 MB**, 67,599 chunk_ids, build wall-time 8 s (2026-05-04)
- [x] Smoke query *"What is the first-line treatment for community-acquired pneumonia?"* → top-3 from both indices clearly relate to pneumonia/antibiotics (2026-05-04)
  - Dense (ChromaDB) → Harrison's empirical-therapy + macrolide/β-lactam combo + Novak ATS guidelines
  - Sparse (BM25) → Pharmacology_Katzung's CAP case study with **ceftriaxone + azithromycin** (literal first-line) + Harrison's M. pneumoniae and Legionella chapters
  - The two retrievers find *different but equally relevant* content ⇒ Hybrid RAG (EXP_04) has real signal to fuse
- **Deliverable:** ✓ all three indices on disk; count parity verified — `chunks.parquet (67,599) == embeddings.npy (67,599) == ChromaDB.count() (67,599) == bm25.chunk_ids (67,599)`

### 2.3 Notebook 03 — `notebooks/03_smoke_test_pipeline.ipynb` ✅ COMPLETE (2026-05-04)

- [x] End-to-end on 3 dev questions: query → BGE retrieval → prompt construction → Groq → parse predicted letter (2026-05-04)
- [x] Print question, top-5 retrieved chunks, generated answer side by side (2026-05-04)
- [x] Time per question; verify Groq quota (2026-05-04 — Groq free tier comfortably handles the load; mean latency 1.38 s)
- **Deliverable:** ✓ 3 / 3 letters parsed cleanly (`'B'`, `'B'`, `'D'`); mean latency **1.38 s** (≪ 5 s budget); cache works (Q0 hit cache from §6 in §7)
  - Q0 (gallstone ileus → distal ileum): truth=B, predicted=**B** ✓
  - Q1 (avoidant personality disorder vignette): truth=A, predicted=**B** Schizoid ✗ — **informative retrieval miss**: top-5 returned general personality-disorder content but no Cluster C / Avoidant-specific chunks. Demonstrates why Hybrid RAG (EXP_04) is needed when answer labels aren't named in the question stem. Pipeline is correct; this is a real retrieval limitation worth recording in methodology.
  - Q2 (HIV gp120 binding): truth=D, predicted=**D** ✓

---

## 3 · `src/` skeleton modules (build before any experiment)

Build in this dependency order. Add a 3-question unit test against `chunks.parquet` for each module.

- [x] `src/data/loaders.py` — `load_medqa_4opt`, `load_golden`, `load_chunks`; parses `options_json`, assigns stable `medqa_NNNNN` ids (2026-05-05 — built alongside §5 EXP_01 prep)
- [x] `src/data/chunker.py` — recursive 400/80 chunker (called by Notebook 01) (2026-05-03 — built alongside §2.1)
- [x] `src/data/embedder.py` — BGE-large wrapper (`load_bge`, `embed_passages`, `embed_queries` with prefix, MPS auto-detect) (2026-05-04 — built alongside §2.2)
- [x] `src/data/indices.py` — `build_chroma`/`load_chroma` + `build_bm25`/`load_bm25` + `bm25_top_k` (2026-05-04 — built alongside §2.2; loader role from todo plus the build role for the one-time §2.2 run)
- [x] `src/retrieval/base.py` — `Retriever` ABC + `Chunk` dataclass (2026-05-05 — built alongside §5 EXP_01 prep)
- [x] `src/retrieval/none.py` — returns `[]` (for EXP_01) (2026-05-05 — built alongside §5 EXP_01 prep)
- [x] `src/retrieval/naive.py` — `NaiveRetriever(chroma_collection, embedder_model)` conforms to the `Retriever` ABC; ChromaDB top-k with BGE query prefix; returns `Chunk` objects with cosine *similarity* scores (1 − distance). 3-row real-data test in `tests/test_naive_retriever.py` (3 / 3 passing). (2026-05-06 — built alongside §5 EXP_02 prep)
- [x] `src/retrieval/sparse.py` — `SparseRetriever(bm25_payload, chunks_df)` conforms to the `Retriever` ABC; wraps the existing `bm25_top_k` helper from `src/data/indices.py`; populates `Chunk` with raw BM25 score + chunk text + book name. 4-row real-data test in `tests/test_sparse_retriever.py` (4 / 4 passing, includes a "rare-term recovery" check — *cisplatin* keyword appears in top-5 for a mechanism-of-action question). (2026-05-07 — built alongside §5 EXP_03 prep)
- [x] `src/retrieval/hybrid.py` — `HybridRetriever(embedder, chroma, bm25, chunks_df)` conforms to `Retriever` ABC; wraps the existing `hybrid_top_k` function (kept for backwards compatibility with Notebook 04 golden construction); RRF fusion k=60. 4-row real-data test in `tests/test_hybrid_retriever.py` (4 / 4 passing, including a backwards-compat check on the function API). (2026-05-07 — built alongside §5 EXP_04 prep)
- [x] `src/retrieval/multi_hop.py` — `MultiHopRetriever(embedder, chroma, max_hops=3, per_hop_k=5)` conforms to `Retriever` ABC; iterative dense retrieval with sub-query generation via Groq; dedup across hops; early stop on no-progress. Plus `build_multi_hop_subquery_prompt` in `src/generation/prompts.py`. 5-row test in `tests/test_multi_hop_retriever.py` (5 / 5 passing, includes pure-Python `_clean_subquery` regression test + real-data hop dedup + k truncation + constructor input validation). (2026-05-07 — built alongside §5 EXP_05 prep)
- [x] `src/generation/groq_client.py` — Groq wrapper with disk cache (key = sha256 of `provider + model + temp + prompt`); returns `(text, latency_s, was_cached)` (2026-05-04 — built alongside §2.3, exercised in 3 real Groq calls)
- [x] `src/generation/prompts.py` — `build_evidence_grounded_prompt` + `build_no_rag_prompt` + permissive `parse_letter` (2026-05-04 — built alongside §2.3; multi-hop variant deferred until EXP_05)
- [x] `src/eval/non_llm_metrics.py` — `exact_match`, `accuracy`, `recall_at_k`, `mrr`, `ndcg_at_k` (2026-05-05 — answer-side used in EXP_01; retrieval-side ready for EXP_02)
- [x] `src/eval/ragas_eval.py` — wraps RAGAS 0.4.3 with **Claude Sonnet 4.6** judge. Uses the **legacy `evaluate()` path** (lowercase pre-built singletons from `ragas.metrics`, wrapped LLM via `LangchainLLMWrapper(ChatAnthropic(...))`, embeddings via `LangchainEmbeddingsWrapper(HuggingFaceEmbeddings)` for BGE-large). The modern `ragas.metrics.collections` API was tried during the 2026-05-06 upgrade but proved incompatible with `evaluate()` — the collections classes inherit from a different `BaseMetric` tree and are only callable via `await metric.single_turn_ascore(sample)`. Legacy lowercase singletons are scheduled for removal at RAGAS v1.0 (well after thesis submission). Auto-skips context-dependent metrics for No-RAG (Option A — see [results_schema.md §2.3](results_schema.md)); writes `ragas_scores.csv` + updates `summary.json` in place. **Resilience layer added 2026-05-06**: `RunConfig(max_workers=4, max_retries=10, max_wait=120)` keeps on-first-pass NaN rate low; `score_predictions(..., rescore_nans=True)` re-judges only the NaN rows and merges scores back via `_merge_partial_scores` (preserves already-good cells, replaces NaN ones). 4-mode idempotency contract (fresh / cache hit / full rerun / NaN rescore) — see [results_schema.md §5.2](results_schema.md). (2026-05-05 built; 2026-05-06 judge upgraded 3.5 → 4.6 + multi-iteration API debugging + NaN-resilience layer, see decision log)
- [x] `src/eval/runner.py` — `run_experiment(retriever, dataset, output_dir, experiment_id, dataset_label)` writes `predictions.jsonl`, `retrieval.jsonl`, `summary.json`. Resumable: skips `question_id`s already in `predictions.jsonl`. (2026-05-05 — built alongside §5 EXP_01 prep)
- [x] `src/utils/cache.py` — disk cache for all LLM calls (Groq, OpenAI, Anthropic) — JSON files at `data/cache/<provider>/<key[:2]>/<key>.json` (2026-05-04 — built alongside §2.3, exercised by `groq_complete`)
- **Deliverable:** every module has at least one passing 3-row test against real data — `tests/test_exp01_modules.py` passes 5 / 5 (loaders + NoRetrieval + non_llm_metrics + runner end-to-end with resume + golden-row shape) (2026-05-05)

---

## 4 · Phase 3 — Golden RAGAS dataset (built from scratch)

### Notebook 04 — `notebooks/04_golden_main_gpt4o.ipynb` ✅ COMPLETE (2026-05-04)

> The original `04_golden_ragas_dataset.ipynb` was split into A/B variants (`04_golden_dataset_gptoss.ipynb` + `04_golden_dataset_gpt4o.ipynb`) for the constructor comparison; the production build is `04_golden_main_gpt4o.ipynb` with new prompts, staged JSONL pipeline, and `STAGE = pilot/production` flag.

- [x] **Stage 0 (smoke pilot, mandatory)** — Sample **50** stratified questions, run all stages A–G end-to-end (2026-05-04). Pilot results: 27/50 accepted, multi-hop 8 %, JSON malformation 0 %, Pass-1 sufficiency 94 %; one validator overshoot (`check_points >= 3`) caught + fixed before scaling.
- [x] **Stage A** — Stratified sample 300 from 12,723 (4-option). 60 long-vignettes forced. Seed 42. (2026-05-04)
- [x] **Stage B** — Hybrid retrieval (BGE + BM25 + RRF k=60) → top-10 candidates per question. Construction-time bias: prepend correct answer to query. Saved to `data/processed/golden/golden_candidates.jsonl` (5.3 MB, 300 rows). (2026-05-04)
- [x] **Stage C — Pass 1** — `gpt-4o` (T=0) evidence selection: structured `selected_chunks` (chunk_id, book_name, support_level, reason), `best_gold_context` (verbatim concat), `evidence_keywords`, `is_evidence_sufficient`. Saved to `golden_evidence_selected.jsonl` (5.9 MB). 294/300 ok, 6 schema/error. (2026-05-04)
- [x] **Stage D — Pass 2** — `gpt-4o` (T=0.2) reference answer + explanation + tightened `requires_multihop` (definition pinned in prompt). Saved to `golden_with_references.jsonl` (7.0 MB). 294/300 ok, 6 skipped (Pass-1 errors). (2026-05-04)
- [x] **Stage E — Pass 3** — `gpt-4o` (T=0) validation: `answer_match` boolean + 0–5 scores + `final_status`. Saved to `golden_validated.jsonl` (7.0 MB). 238 accepted / 49 needs_review / 7 rejected / 6 skipped. (2026-05-04)
- [x] **Stage F — Audit** — Pure-Python: gold answer in reference_answer; evidence_keywords in gold_context; chunk_ids resolvable; `answer_match` true. Audit downgraded 4 accepted → needs_review. Final: **234 accepted, 53 needs_review, 13 dropped.** (2026-05-04)
- [x] **Stage G — Save** — `data/processed/golden_ragas_300.jsonl` (234 accepted, 1.0 MB). Plus staged files in `data/processed/golden/` for full-pipeline transparency. (2026-05-04)
- [ ] Manually spot-check 30 accepted rows for grounding quality
- [x] Multi-hop label audit — production rate is **6 %** (gate: < 60 %), well within budget after Pass-2 prompt tightening. (2026-05-04)
- **Deliverable:** ✅ **234 accepted rows of 300** (≥ 220 target met) · **`requires_multihop=yes` rate 6 %** (< 50 % target met) · **total construction cost $6.61** measured on `gpt-4o` (locked after A/B vs `gpt-oss-120b` produced lower-quality output for $0.40).

---

## 5 · Phase 4 — Group A baseline experiments (EXP_01 → EXP_05)

For each experiment: write `notebooks/04*_exp0*_*.ipynb`, run on full **12,723** for accuracy/retrieval metrics, run on golden **1,000** for full RAGAS metrics. Cache every LLM call.

### EXP_01 — No-RAG baseline ✅ COMPLETE (2026-05-05)

- [x] Notebook `04a_exp01_base_llm.ipynb` — three gated stages: smoke 50 → golden 234 → **test 1,273** (locked 2026-05-06 per [plan.md §0 #8](../plan.md); originally ran on full 12,723, now derived from that artifact). (2026-05-05)
- [x] Run Stage A (smoke 50, **Acuuracy = 0.940**), Stage B (golden 234, **0.902**), Stage C (full 12,723, **0.869**) — full-12,723 run preserved as the contamination-evidence anchor (2026-05-05)
- [x] **Canonical headline = `exp_01_base_llm__test_1273/`** (derived 2026-05-06 from the full-12,723 run by filtering `split == 'test'` rows): **`Acuuracy = 0.7738`** over 1,273 rows — the contamination-clean baseline number for cross-architecture comparison.
- [x] Outputs to `results/exp_01_base_llm__{smoke_50,golden_234,test_1273,full_12723}/` (canonical = `test_1273`; `full_12723` retained as legacy contamination evidence) — `predictions.jsonl` + `retrieval.jsonl` + `summary.json` for each (2026-05-05/06)
- [x] **Methodology finding (contamination empirically validated):** train + dev = 0.880, **test = 0.774** ⇒ 10.6 pp gap. Aligns with literature LLaMA-class No-RAG ceiling (~75–78 %) on the test split; confirms [plan.md §15](../plan.md) contamination risk. **Drove the 2026-05-06 lock** to evaluate all subsequent experiments on the test split only — cleaner methodology, MedRAG/MIRAGE-comparable, ~10× faster wall time. See [output_notes/04a_exp01_output.md](output_notes/04a_exp01_output.md).
- [x] `src/eval/ragas_eval.py` + `notebooks/04a_exp01_ragas.ipynb` built (2026-05-05) and **RAN to completion 2026-05-06** — Stage A smoke (10 rows, 0 NaN, signal validated) then Stage B full 234. **Headline numbers** in `results/exp_01_base_llm__golden_234/summary.json`: `Answer_Correctness = 0.8738` (over 137 non-NaN), `RAGAS_Answer_Relevance = 0.5977` (over 174 non-NaN), `Acuuracy` unchanged at 0.9017. Judge correctness gap: AC=0.93 on correct rows vs AC=0.31 on wrong rows ⇒ **+62 pp** (calibrated). For EXP_01 specifically: `RAGAS_Faithfulness`, `RAGAS_Hallucination_Rate`, `RAGAS_Context_Precision`, `RAGAS_Context_Recall` stay `null` by design (Option A — undefined without retrieved context). EXP_02–EXP_05 will compute all five (~$30–40 each).
  - **⚠️ Open issue**: ~40 % of rows came back NaN on first pass (97 / 234 on AnswerCorrectness, 60 / 234 on AnswerRelevancy). Diagnosis confirmed transient API failures (uniform NaN rate across correct/wrong/length/topic strata; not content-related). See [`docs/output_notes/04a_exp01_output.md` §4.3](output_notes/04a_exp01_output.md). Two fixes recommended **before EXP_02**: (a) add `RunConfig(max_retries=5, max_wait=60)` to `_run_judge` to auto-retry transient errors; (b) add a `rescore_nans()` mode to `score_predictions` that re-runs only NaN rows (~$5 to complete EXP_01, saves ~$12–16 per future architecture).
- **Tables touched:** Table 1 (row 1 — `Acuuracy`, `Exact Match`, `Generator_Model`, `mean_latency_s`), Table 7 (row 1), Table 9 (col 1)
- **Schema lock:** `summary.json` shape pinned in [docs/results_schema.md](results_schema.md) (2026-05-05); RAGAS keys ship as `null` until the judge runs.
- **Operational note:** wall time for full 12,723 was **58 min** (vs the 5 h projection in [plan.md §14](../plan.md)). All five Group A experiments will land well inside the originally-budgeted Groq window.

### EXP_02 — Naive RAG (dense ChromaDB top-k) ✅ COMPLETE (2026-05-07)

- [x] Notebook `04b_exp02_naive_rag.ipynb` — three gated stages: smoke 50 → golden 234 → test 1,273. (2026-05-06 built · 2026-05-07 user ran all stages)
- [x] Notebook `04b_exp02_ragas.ipynb` — three gated stages (smoke 10 → full 234 → NaN rescore), all 5 RAGAS metrics. (2026-05-06 built · 2026-05-07 user ran)
- [x] **Canonical: `exp_02_naive_rag__test_1273/`** — `Acuuracy = 0.7573` (964 / 1,273), mean_latency = 0.444 s, wall_time = 11.7 min. **−1.65 pp vs EXP_01 No-RAG.** First architecture data point: Naive Dense RAG with BGE-large is *worse* than No-RAG on the contamination-clean test split.
- [x] Outputs to `results/exp_02_naive_rag__{smoke_50,golden_234,test_1273}/` + `golden_234_ragas_smoke/` + legacy `full_12723/` (abandoned partial)
- [x] **RAGAS results on golden_234** (Claude Sonnet 4.6, n=234, NaN rate <2.1 % across all 5 metrics — resilience layer worked): `RAGAS_Faithfulness = 0.1308` · `Hallucination_Rate = 0.8957` · `RAGAS_Context_Precision = 0.3285` · `RAGAS_Context_Recall = 0.4124` · `Answer_Relevance = 0.5961` · `Answer_Correctness = 0.8376`. **88.3 % of correct answers were ungrounded** (Faithfulness < 0.5) — LLaMA produced the right MCQ option from pre-training memorisation, NOT from retrieved chunks. The thesis's central novelty (confidence-aware rejection) is now empirically motivated. See [`docs/output_notes/04b_exp02_output.md`](output_notes/04b_exp02_output.md).
- [x] **Cost**: ~$11.50 (smoke $0.50 + full ~$11) — well under the recalibrated $11–12 projection. Cumulative thesis API spend: $4.50 (EXP_01 RAGAS) + $11.50 (EXP_02 RAGAS) + $6.61 (Phase 3 golden) = **~$22.61** of the projected ~$70–80 total.
- **Tables touched:** Table 1 (row 2 — all metrics filled) · Table 8 (Context Precision/Recall row 1) · Table 9 (Naive RAG column)
- **Methodology hypothesis for EXP_03/04/05**: Hybrid RAG must demonstrate Context Precision ≥ 0.50 to "earn its keep" over Naive's 0.33; Multi-Hop must achieve Faithfulness > 0 on `requires_multihop=yes` rows where Naive scored 0.000. Falsifiable, anchored in EXP_02's data.

### EXP_03 — Sparse RAG (BM25) ✅ FULLY COMPLETE (baseline 2026-05-07 · RAGAS 2026-05-10)

- [x] Notebook `04c_exp03_sparse_rag.ipynb` — three gated stages: smoke 50 → golden 234 → test 1,273. (2026-05-07 built · 2026-05-07 user ran all stages)
- [x] Notebook `04c_exp03_ragas.ipynb` — three gated stages, all 5 RAGAS metrics. (2026-05-07 built · 2026-05-10 user ran)
- [x] **Canonical: `exp_03_sparse_rag__test_1273/`** — `Acuuracy = 0.7581` (965 / 1,273), mean_latency = 0.435 s, wall_time = **97 min**. **−1.57 pp vs EXP_01 No-RAG**, **+0.08 pp vs EXP_02 Naive Dense** (essentially tied with dense).
- [x] **RAGAS results on golden_234** (Claude Sonnet 4.6, n=234, NaN rate <1 % per metric): `RAGAS_Faithfulness = 0.0401` · `Hallucination_Rate = 0.9657` · `RAGAS_Context_Precision = 0.0811` · `RAGAS_Context_Recall = 0.1073` · `Answer_Relevance = 0.5971` · `Answer_Correctness = 0.8384`. **Falsifiable hypothesis FALSIFIED**: sparse CP = 0.081 vs the > 0.33 threshold (worse than dense, not better). **The strongest single piece of evidence in Phase 4 for the memorisation thesis**: CP collapsed 4× yet accuracy is unchanged (LLM ignores the bad chunks). Full analysis in [`output_notes/04c_exp03_output.md`](output_notes/04c_exp03_output.md).
- [x] Outputs to `results/exp_03_sparse_rag__{smoke_50,golden_234,test_1273,golden_234_ragas_smoke}/`
- [x] **Two empirical findings from baseline that informed EXP_04 design**:
  - **Complementarity**: 153 of 1,273 test questions disagree between dense & sparse, with 50/50 right/wrong split — anchored the RRF hypothesis that turned out to be a measurement artefact (see EXP_04 conclusions).
  - **USMLE-step pattern**: sparse beats dense on step2&3 (+1.68 pp), dense beats sparse on step1 (+1.33 pp).
- [x] **Cost**: ~$11 RAGAS Sonnet 4.6.
- **Tables touched:** Table 1 (row 3 — all metrics filled), Table 8 (row 2 — filled), Table 9 (Sparse-RAG column)

### EXP_04 — Hybrid RAG (RRF) ✅ FULLY COMPLETE (baseline 2026-05-07 · RAGAS 2026-05-10)

- [x] Notebook `04d_exp04_hybrid_rag.ipynb` — three gated stages (smoke 50 → golden 234 → test 1,273), uses refactored `HybridRetriever` (BGE + ChromaDB + BM25 fused via RRF k=60). (2026-05-07 built · 2026-05-07 user ran all stages)
- [x] Notebook `04d_exp04_ragas.ipynb` — three gated stages, all 5 RAGAS metrics. (2026-05-07 built · 2026-05-10 user ran)
- [x] **Canonical: `exp_04_hybrid_rag__test_1273/`** — `Acuuracy = 0.7659` (975 / 1,273), mean_latency = 0.443 s, wall_time = **123 min** (Hybrid is the slowest architecture because RRF carries `rank-bm25`'s O(N=67k) Python scoring). **+0.86 pp vs EXP_02 Naive Dense**, **−0.79 pp vs EXP_01 No-RAG**.
- [x] **RAGAS results on golden_234** (Claude Sonnet 4.6, n=234, NaN rate <2 % per metric): `RAGAS_Faithfulness = 0.0944` · `Hallucination_Rate = 0.9174` · `RAGAS_Context_Precision = 0.2797` · `RAGAS_Context_Recall = 0.3483` · `Answer_Relevance = 0.5966` · `Answer_Correctness = 0.8273`. **Falsifiable hypothesis verdicts**: ✓ Hybrid Acuuracy > 0.76 SUPPORTED (just clears at 0.7659); ✗ Hybrid CP ≥ 0.50 FALSIFIED (got 0.280, *worse* than Naive's 0.329 — RRF fusion with sparse's near-random CP=0.081 contaminates the union). **Publishable counter-result for the discussion chapter**: RRF requires both retrievers to clear a precision floor. Full analysis in [`output_notes/04d_exp04_output.md`](output_notes/04d_exp04_output.md).
- [x] Outputs to `results/exp_04_hybrid_rag__{smoke_50,golden_234,test_1273,golden_234_ragas_smoke}/`
- [x] **Cost**: ~$11 RAGAS Sonnet 4.6.
- **Tables touched:** Table 1 (row 4 — all metrics filled), Table 8 (row 3 — filled), Table 9 (Hybrid-RAG column)

### EXP_05 — Multi-Hop RAG ✅ FULLY COMPLETE (baseline 2026-05-07 · RAGAS 2026-05-10) — **THE HEADLINE FINDING**

- [x] Notebook `04e_exp05_multi_hop_rag.ipynb` — three gated stages, uses new `MultiHopRetriever` (3-hop iterative dense, sub-query generation via Groq, dedup across hops, early stop on no-progress, returns up to 15 chunks per question). (2026-05-07 built · 2026-05-07 user ran all stages)
- [x] Notebook `04e_exp05_ragas.ipynb` — three gated stages, all 5 RAGAS metrics. (2026-05-07 built · 2026-05-10 user ran)
- [x] **Canonical: `exp_05_multi_hop_rag__test_1273/`** — `Acuuracy = 0.7958` (1,013 / 1,273), mean_latency = 0.660 s, wall_time = **57 min**. **+2.20 pp vs EXP_01 No-RAG** (the only RAG architecture to beat No-RAG), **+3.85 pp vs EXP_02 Naive Dense**, **+2.99 pp vs EXP_04 Hybrid**.
- [x] **RAGAS results on golden_234** (Claude Sonnet 4.6, n=234, NaN rate <4 % per metric): `RAGAS_Faithfulness = 0.2833` (2.2× Naive's 0.131; median 0.250 — the only architecture to lift median F above zero) · `Hallucination_Rate = 0.7371` (lowest) · `RAGAS_Context_Precision = 0.3737` (highest) · `RAGAS_Context_Recall = 0.7115` (+30 pp vs Naive — iterative hops doubling recall) · `Answer_Relevance = 0.5955` · `Answer_Correctness = 0.8685` (matches No-RAG's 0.874). **Falsifiable hypothesis verdicts**: ✓ Multi-Hop Acuuracy ≥ 0.7573 SUPPORTED (got 0.7958, +3.85 pp); ✓ Multi-Hop F > 0.05 on multi-hop subset SUPPORTED (got 0.229, vs Naive's 0.000 on the same 13 rows). Full analysis in [`output_notes/04e_exp05_output.md`](output_notes/04e_exp05_output.md).
- [x] Outputs to `results/exp_05_multi_hop_rag__{smoke_50,golden_234,test_1273,golden_234_ragas_smoke}/`
- [x] Hop budget = 3, k=5 per hop, terminate on no-new-chunks (per [plan.md §0 #7](../plan.md), implemented in `src/retrieval/multi_hop.py`)
- [x] **Regression analysis vs No-RAG (test_1273)**: 101 fixes, 73 regressions = **net +28 questions** (the only architecture with a positive net; EXP_02 −21, EXP_03 −20, EXP_04 −10).
- [x] **Grounded-correct fraction**: 28.4 % of correct answers reach F ≥ 0.5 (vs 11.6 % Naive, 4.1 % Sparse, 8.7 % Hybrid). **The only architecture where Faithfulness is a useable confidence signal for Phase 7.**
- [x] **Cost**: ~$13 RAGAS Sonnet 4.6 (slightly higher than EXP_02–04 due to bigger context per question, up to 15 chunks vs 5).
- **Tables touched:** Table 1 (row 5 — all metrics filled), Table 8 (row 4 — filled), Table 9 (Multi-Hop column)

### After Group A finishes ✅ COMPLETE 2026-05-10

- [x] **Phase 4 close-out summary** with cross-architecture table + falsifiable hypothesis verdicts → [`plan.md §6.1`](../plan.md). Tables 1, 8, 9 ready to populate from `summary.json` files.
- [ ] Document the BGE-large embedder choice in the methodology section (one paragraph; cite TREC-COVID nDCG@10 benchmark; flag domain-fine-tuned ablation as future work) — pending thesis writeup phase
- **Deliverable:** ✅ Tables 1, 8, 9 fully populated for all 5 architectures (EXP_01 No-RAG + EXP_02–05 RAG)

---

## 6 · Phase 5 — Group B: Adaptive Routing (EXP_06, EXP_07)

### EXP_06 — Question complexity labelling ✅ FULLY COMPLETE (2026-05-10)

- [x] `src/retrieval/complexity.py` — rule-based classifier using length (`n_words`), `metamap_phrases` count, and cue-word presence (complex-decision + factoid-mechanism). Thresholds anchored to corpus-wide 33rd/67th percentiles: `WORDS_P33=93 / WORDS_P67=133 / PHRASES_P33=28 / PHRASES_P67=41`. (2026-05-10 built)
- [x] `tests/test_complexity.py` — 7/7 tests passing (4 synthetic + 1 real-data + 2 invariants)
- [x] Notebook `05_exp06_complexity_labels.ipynb` — 18 cells; applies to 12,723 questions; writes `data/processed/complexity_labels.parquet` (123 KB); prints per-bucket accuracy attribution from existing EXP_01–05 predictions; samples 100 stratified rows for manual review. (2026-05-10 built · 2026-05-10 user ran)
- [x] **Distribution** (test_1273 split): Simple 366 (28.7 %) · Moderate 394 (30.9 %) · Complex 513 (40.3 %). Overall on 12,723: 29.5 / 32.7 / 37.7 %. Step1 skews Simple (46 %); Step2&3 skews Complex (63 %). All buckets clear the 15–55 % acceptance band.
- [x] **Manual review (100 stratified, seed=42)**: **1 disagreement** (`medqa_8198`, multi-system multiple myeloma work-up, w=134 p=39, labelled Moderate but reviewer would lean Complex — boundary case). Far below the ≤20/100 acceptance gate.
- [x] **Methodology footnote anchored**: "MedQA is overwhelmingly clinical vignettes; the 'Simple' bucket is honestly *shortest-third-vignette*, not pure factoid. The proposal terminology is preserved for plan alignment; the rule is a length + entity-density + cue-word proxy for complexity."
- **Tables touched:** Table 3 (all 3 rows + thresholds). Full analysis in [`docs/output_notes/05_exp06_output.md`](output_notes/05_exp06_output.md).

### EXP_07 — Adaptive RAG controller ✅ FULLY COMPLETE (2026-05-11)

- [x] `src/retrieval/adaptive.py` — `AdaptiveRetriever` conforms to `Retriever` ABC; takes a `question_to_bucket` lookup + a `bucket_to_retriever` routing table; exposes `dispatch_counts` and `unknown_question_count` for audit. (2026-05-10 built)
- [x] `tests/test_adaptive.py` — 6/6 tests passing (dispatch correctness, default fallback, NoRetrieval pass-through, construction-time validation, peek-doesn't-dispatch, real-data lookup join)
- [x] Notebook `05_exp07_adaptive_rag.ipynb` — 24 cells; 5 stages (smoke A → smoke B → full A → full B → RAGAS score-join). (2026-05-10 built · 2026-05-11 user ran)
- [x] **Both routing variants tested**:
  - **Variant A** (proposal): Simple → Naive, Moderate → Hybrid, Complex → Multi-Hop → **Acuuracy 0.7863** (1001/1273), 1.806 Groq calls/Q, wall 30 min, 40 % cache hits (k=15 wrinkle, see methodology footnote).
  - **Variant B** (data-driven binary): Simple → No-RAG, Moderate → Multi-Hop, Complex → Multi-Hop → **Acuuracy 0.7832** (997/1273), 2.425 calls/Q, wall 2.3 min, 99 % cache hits.
- [x] **Pareto frontier on test_1273**: NoRAG (0.7738 / 1.00) → **Variant A (0.7863 / 1.81)** → Multi-Hop (0.7958 / 3.00). All three on frontier. Naive/Sparse/Hybrid/Variant_B all DOMINATED.
- [x] **Hypothesis verdicts (all SUPPORTED)**: H1 Variant A on Pareto frontier ✓; H2 Variant A dominates Variant B ✓ (higher acc AND fewer calls); H3 Variant A 2.0× more marginally efficient than Multi-Hop on top of A ✓ (0.0156 vs 0.0079 acc/extra-call).
- [x] **RAGAS score-join** (golden_234, no new judge calls): Variant A F=0.197 / CR=0.571 / AC=0.847; Variant B F=0.276 / CR=0.754 / AC=0.867. Variant B has better grounding metrics at the cost of more compute — a publishable two-axis trade-off.
- [x] **Regression analysis vs No-RAG**: Variant A net **+16** questions (82 fixes / 66 regressions); Variant B net +12. First single-shot architecture between No-RAG and Multi-Hop with positive net (EXP_02 −21, EXP_03 −20, EXP_04 −10, EXP_05 +28).
- [x] **Methodology footnote anchored** (k=15 vs k=5 chunk fan-out): runner used uniform k=15; Naive (Variant A's Simple lane) and Hybrid (Variant A's Moderate lane) baselines used k=5. Variant A simulator (k=5 predictions) = 0.7895 vs actual (k=15) = 0.7863 (Δ=−0.31 pp). Variant B's gap ≈ 0 (uses Multi-Hop which natively returns k=15). Both numbers reported; actual is canonical.
- **Tables touched:** Table 1 (row 6 — both variants), Table 2 (per-bucket accuracy), Table 10 (adaptive-vs-fixed). Full analysis in [`docs/output_notes/05_exp07_output.md`](output_notes/05_exp07_output.md).

---

## 7 · Phase 6 — Group C support: Explainability (EXP_10, EXP_11, EXP_12)

> Run **before** EXP_08/09 — confidence signals consume LIME-SHAP agreement.

### EXP_10 — Passage-level LIME

- [ ] `src/xai/lime_passage.py` — perturb retrieved passages, regenerate, score change, fit local linear surrogate
- [ ] Notebook `06_exp10_lime_passage.ipynb` — 200 questions × 5 architectures (sample to bound cost)
- **Tables touched:** Table 6

### EXP_11 — Passage-level SHAP

- [ ] `src/xai/shap_passage.py` — sample passage subsets, estimate marginal contribution
- [ ] Notebook `06_exp11_shap_passage.ipynb` — same 200 questions × 5 architectures
- **Tables touched:** Table 6

### EXP_12 — LIME ↔ SHAP agreement

- [ ] `src/xai/agreement.py` — top-1 / top-3 overlap; correlate with Faithfulness
- [ ] Notebook `06_exp12_xai_agreement.ipynb`
- **Tables touched:** Table 6

---

## 8 · Phase 7 — Group C: Confidence-Aware Rejection (EXP_08, EXP_09)

### EXP_08 — Confidence signal extraction

- [ ] `src/confidence/signals.py` — assemble per-question vector: `[retrieval_score_mean, retrieval_score_var, faithfulness, context_precision, answer_relevancy, lime_top_score, shap_top_score, lime_shap_agreement]` normalised to [0, 1]
- [ ] Notebook `07_exp08_confidence_signals.ipynb` — saves `data/processed/confidence_features.parquet`
- **Tables touched:** Table 5

### EXP_09 — Threshold tuning + rejection

- [ ] `src/confidence/rejection.py` — weighted formula: `0.30·retrieval + 0.30·faithfulness + 0.20·relevancy + 0.20·agreement`
- [ ] Sweep thresholds {0.5, 0.6, 0.7, 0.8, 0.9} on a 200-row validation slice
- [ ] Pick threshold that **minimises hallucination rate while keeping ≥ 70% accept rate**
- [ ] Apply locked threshold to **test 1,273** across all 5 architectures (per [plan.md §0 #8](../plan.md))
- [ ] When `confidence < threshold`, emit *"Evidence is insufficient for a reliable answer."*
- **Tables touched:** Table 4, Table 11

---

## 9 · Phase 8 — Group D: Hallucination Error-Type Taxonomy (EXP_13, EXP_14, EXP_15)

### EXP_13 — Define categories

- [ ] `src/taxonomy/categories.py` — 6 categories: `unsupported_diagnosis`, `unsupported_treatment`, `wrong_reasoning_chain`, `partial_evidence_misuse`, `option_mismatch`, `context_omission`
- [ ] Annotation guideline doc with 1–2 examples per category
- **Tables touched:** Table 7 (header)

### EXP_14 — Label low-faithfulness outputs

- [ ] Filter outputs with `Faithfulness < 0.6`
- [ ] Manually annotate ~150 across architectures (~3 h)
- [ ] If ≥ 100 manual labels, train a logistic-regression classifier and label the rest; validate on held-out 30
- **Tables touched:** Table 7

### EXP_15 — Architecture-level error patterns

- [ ] Notebook `08_exp15_arch_error_analysis.ipynb` — error type × architecture cross-tab
- [ ] Confirm/refute hypotheses: Naive→context_omission, Sparse→option_mismatch, Hybrid→fewer retrieval errors, Multi-Hop→reasoning_chain errors
- **Tables touched:** Table 7

---

## 10 · Phase 9 — EXP_16: Final Synthesis

- [ ] Notebook `09_exp16_final_ranking.ipynb`
- [ ] Aggregate every architecture's metrics into one row
- [ ] Compute `final_score = 0.25·Acc + 0.25·Faithfulness + 0.20·Retrieval + 0.15·Safety + 0.10·Explainability + 0.05·Latency` (all [0,1] normalised)
- [ ] Rank
- [ ] Map ranks → use-case recommendations (low-cost / high-accuracy / low-hallucination / balanced)
- **Tables touched:** Table 10, Table 12

---

## 10.5 · Phase 10 — Demo UI (**optional, parallel track**)

> **Status:** optional. Cached-mode only — the UI reads from `results/exp_*/` artifacts, never makes a live LLM call. Four tabs: **Architecture Battle · Explainability · Confidence & Safety · Results Dashboard**. Build only if time permits — the thesis defends without it.

> **Why optional matters:** if at the end of Phase 9 the experiment results are weak or incomplete, **drop the UI** and put that time into the thesis writeup. Plan.md §15 lists this as a top risk.

### Stage A — Scaffolding (~2 days · do right after Phase 3)

- [ ] Create `app/main.py` — Streamlit entry point with tab router
- [ ] Create `app/tabs/{battle, explainability, confidence, dashboard}.py` — empty stubs that render placeholders
- [ ] Create `app/utils.py` — cached-data loader functions with mock-data fallback
- [ ] Create `app/_mock_data.py` — placeholder data built from the legacy 65-row golden so the UI renders
- [ ] Create `streamlit_app.py` at repo root (Streamlit Cloud entry point) → imports `app/main.py`
- [ ] **Lock `summary.json` schema** in `docs/results_schema.md` — every experiment writes to this exact shape
- [ ] Add a CI-style assert in `src/eval/runner.py` that the produced `summary.json` matches the schema
- [ ] `pip install streamlit` + add to `requirements.txt`
- **Acceptance:** `.venv/bin/streamlit run app/main.py` opens a browser, all 4 tabs render with mock data, no errors. `docs/results_schema.md` exists and the runner asserts against it.

### Stage B — Incremental wire-up (~0 extra days · parallel to Phases 4–9)

- [ ] After EXP_01 done → wire **Tab 1** to read `results/exp_01_base_llm/predictions.jsonl`
- [ ] After EXP_02–EXP_05 done → Tab 1 shows the 4-RAG side-by-side comparison
- [ ] After EXP_07 (Adaptive) done → Tab 1 gets a 5th pane
- [ ] After EXP_10–EXP_12 done → Tab 2 (Explainability) reads real LIME/SHAP outputs
- [ ] After EXP_08–EXP_09 done → Tab 3 (Confidence & Safety) reads real signal/threshold data
- [ ] After EXP_16 done → Tab 4 (Dashboard) reads all `results/exp_*/summary.json`
- **Acceptance:** mock data is gone; every tab shows real experiment outputs; `app/_mock_data.py` deleted.

### Stage C — Polish & demo prep (~3 days · after Phase 9)

- [ ] Styling pass — consistent colour palette, readable fonts, no Streamlit dev warnings
- [ ] Add help / about pages explaining each tab
- [ ] Capture ≥6 screenshots for the thesis report (results chapter)
- [ ] Deploy to **Streamlit Cloud free tier** (one-click GitHub integration)
- [ ] Verify the deployed app on a phone browser
- [ ] Record a 3-minute screencast walking through the 4 tabs
- [ ] Add the Streamlit Cloud URL + screencast link to the thesis report
- **Acceptance:** public URL accessible; screencast at `docs/demo.mp4` (or linked from a static host); thesis results chapter contains the screenshots.

---

## 11 · Excel workbook — paste-the-results pass

When all 16 experiments are done:

- [ ] Table 1 — Overall Architecture Performance (6 rows)
- [ ] Table 2 — Adaptive Retrieval by Complexity (4 rows)
- [ ] Table 3 — Question Complexity Labelling Summary (3 rows)
- [ ] Table 4 — Confidence-Aware Rejection (5 rows)
- [ ] Table 5 — Confidence Signal Breakdown (5 rows)
- [ ] Table 6 — LIME / SHAP Explainability (5 rows)
- [ ] Table 7 — Hallucination Error-Type Taxonomy (6 rows)
- [ ] Table 8 — Retrieval Quality (5 rows)
- [ ] Table 9 — Before / After RAG (8 metric rows × 6 architecture cols)
- [ ] Table 10 — Adaptive vs Best Fixed (8 rows)
- [ ] Table 11 — Confidence Threshold Tuning (5 rows)
- [ ] Table 12 — Final Weighted Ranking (6 rows)
- **Deliverable:** every cell either filled or marked "N/A"

---

## 12 · Thesis writing

- [ ] Methodology chapter — lift from `plan.md`, `docs/thesis_understanding.md`, `docs/dataset.md`, `docs/tech_stack.md`
- [ ] Results chapter — driven by the 12 results tables, with per-stratum breakdowns (`meta_info`, `question_type`, `requires_multihop`)
- [ ] Discussion — what each architecture's failure mode means, the LIME/SHAP agreement story, the hallucination taxonomy as a contribution
- [ ] Limitations — corpus bias (Harrison's 25%), single-LLM, English-only, 18-textbook KB
- [ ] References — keep BibTeX synced as you write
- [ ] Final thesis draft → supervisor review → revisions → submission

---

## Decision log (anchor for the viva)

| Date | Decision | Locked in | Driver |
|---|---|---|---|
| 2026-04-25 (pilot) | LLaMA 3.3 70B as answerer | plan.md §0 #1 | Proposal §7.5.1 + pilot |
| 2026-04-25 (pilot) | ChromaDB over FAISS | plan.md §0 #3 | Pilot retrieval-quality parity |
| 2026-05-03 | BGE-large-en-v1.5 as primary embedder | plan.md §0 #2 | +28 nDCG@10 over MiniLM on TREC-COVID |
| 2026-05-03 | 400-token chunks (initially 40 overlap) | plan.md §0 #5 | BGE 512-token max headroom |
| 2026-05-03 | Evaluate full 12,723 (not 1,500 subset) | plan.md §0 #8 | EDA showed canonical MedQA size |
| 2026-05-03 | **80-token (20%) overlap** instead of 40 | plan.md §0 #5 | 2024–25 medical-RAG standard, boundary-loss protection |
| 2026-05-03 | ~~MedEmbed-large as ablation embedder~~ → **DROPPED, BGE-large only** | plan.md §0 #2 | **Reversed same day.** Single-embedder design saves ~24 h Groq + ~$8 RAGAS judging. Methodology defends BGE-large via TREC-COVID nDCG@10 benchmark; identifies medical-fine-tuned ablation as future work. |
| 2026-05-03 | **GPT-4o (full) for golden constructor** | plan.md §0 #10 | 3-pass JSON-strict reliability |
| 2026-05-03 | **Claude 3.5 Sonnet as RAGAS judge** | plan.md §0 #11 | Different family from generator + constructor |
| 2026-05-03 | **Phase 10 — Demo UI accepted as optional parallel track** | plan.md §12 (Phase 10) | Cached-only mode, 4 tabs, Streamlit. Build only if time permits; drop if experiments slip. Schema-lock in Stage A protects against rework. |
| 2026-05-03 | **Golden RAGAS dataset sized at 300** (was originally 1,000) | plan.md §0 #9, §5 | Cost-efficiency. 300 preserves ≥ 60 rows per `question_type` bucket for stratified analysis; below-the-floor risk at 200 was avoided. Built in two stages: 50-row pilot ($2) → 250 production ($10). |
| 2026-05-03 | **Chunk-count expectation recalibrated**: ~36k → **~67k** chunks (acceptance band 50k–75k) | plan.md §0 #5 estimate, §4 acceptance, §15 risk; docs/todo.md §2.1 | Math: corpus = 16.7 M cl100k tokens; 400-token chunks with 80-token overlap advance ≤ 320 unique tokens, giving a floor of ~52k chunks. `RecursiveCharacterTextSplitter` fills to ~80% of the cap (mean ≈ 324 tokens), so realistic count is ~67k. The 400/80 config itself stays locked. Notebook 01 produced **67,599 chunks** at mean 323.9 tokens, Harrison's 24.66% — all within the new band. Knock-on: BGE embed time ≈ 45–50 min CPU (was estimated 25 min); index size ≈ 274 MB (was 131 MB). |
| 2026-05-04 | **BGE-large embed time recalibrated again**: ~22 min MPS → **~355 min MPS** measured | plan.md §0 #2 + §4 + §14; docs/architecture.md §8; docs/tech_stack.md; memory/project_thesis_overview.md | Notebook 02 §6 actual wall-time 354.9 min on M1 Pro MPS for 67,599 chunks at batch 32. First-batch timing extrapolated to ~118 min, but sustained throughput degraded to ~16 s/batch over 6 hours — thermal throttling and/or partial-MPS coverage on BGE-large's 1024-d attention. **Output is correct** (norms = 1.0, shape (67599,1024), retrieval works on the smoke query). The cost is paid once: `embeddings.npy` is now on disk and §6 is resumable. No re-embed needed; just update plan estimates so future-you doesn't believe the 22-min figure. |
| 2026-05-04 | **`chromadb>=0.5,<0.6` + `transformers>=4.46,<5.0` pinned** in `requirements.txt` | requirements.txt | chromadb 0.5 transitively requires `tokenizers<0.21`, which is incompatible with `transformers 5.x` (needs `tokenizers>=0.22`). Pinning transformers to the 4.x line keeps both libraries co-installable. Both work at runtime despite a cosmetic pip-resolver warning. Telemetry warnings silenced via `Settings(anonymized_telemetry=False)` + `ANONYMIZED_TELEMETRY=False` env var. |
| 2026-05-04 | **Phase 2 fully complete** (Notebooks 01 + 02 + 03 all run end-to-end with verified outputs) | docs/todo.md §2.1, §2.2, §2.3; memory/project_thesis_overview.md | All four shared infrastructure artefacts on disk: `chunks.parquet` (67,599) · `embeddings.npy` (67,599 × 1024) · `chroma_textbooks/` (cosine HNSW, count parity ✓) · `bm25.pkl`. Notebook 03 smoke test: 3 / 3 letters parsed, mean latency 1.38 s, disk cache operational. Q1's wrong answer (Avoidant labelled as Schizoid) was an *informative retrieval miss* — top-5 returned general personality-disorder content but no Cluster-C specifics. Methodology finding worth recording: Naive RAG insufficient when answer label isn't named in the question stem; motivates Hybrid RAG (EXP_04). |
| 2026-05-04 | **Constructor swap: GPT-4o → `openai/gpt-oss-120b` via Groq** (~$12 → $0) | plan.md §0 #10, §5 stages C–E + cost paragraph + §14 budget + §16; docs/tech_stack.md §3 + §4; docs/architecture.md §0 + §3.4 + §8.1; docs/todo.md §0 + §4 stages; memory/project_thesis_overview.md | Path #2 from the cost-vs-methodology trade discussion (2026-05-04). OpenAI's open-weights 120B model preserves the 3-family separation (Meta gen / OpenAI cons / Anthropic judge) at $0 because Groq hosts it free. Judge stays Claude 3.5 Sonnet ($10–15) — Faithfulness scoring is too sensitive to use an open-weights judge without methodological doubt. **50-row pilot is now the quality gate**: accept rate ≥ 80 %, JSON malformation < 5 %. Fallback ladder if pilot fails: GPT-4o-mini ($3) → GPT-4o full ($12). `OPENAI_API_KEY` becomes optional. |
| 2026-05-04 | **gpt-oss-120b pilot ran: 42 % accept rate (gate ✗) at $0.09 measured cost** | docs/output_notes/04_notebook_output.md (pending); decision-log here | 50-row pilot results: accept 21/50 = 42 %, malformed JSON 0 % ✓, Pass-1 sufficiency 60 % (gate ≥ 90 % ✗), `requires_multihop` 62 % (gate < 60 % ✗). JSON output is rock solid; failures are calibration (over-conservative sufficiency, over-labelled multi-hop) not capability. Cost was 94 % cheaper than GPT-4o would have been. **Decision deferred:** A/B comparison run with `gpt-4o` on the same 50 questions (notebook `04_golden_dataset_gpt4o.ipynb`) before locking constructor. |
| 2026-05-04 | **A/B comparison strategy adopted for constructor choice** | notebooks/04_golden_dataset_gptoss.ipynb + 04_golden_dataset_gpt4o.ipynb; src/generation/openai_client.py | Same 50 questions (seed 42), same prompts, same retrieval — only constructor swaps. Decision rule: pick the model whose `(accept_rate × estimated_quality)` divided by `cost_for_300_rows` is highest. Defensible in viva: "we tested both empirically and chose X because Y, Z." Adds ~$1–2 cost to the GPT-4o pilot, ~30 lines for `openai_client.py`. |
| 2026-05-04 | **Constructor lock REVERTED back to `gpt-4o`** (gpt-oss-120b A/B comparison concluded) | docs/output_notes/04_notebook_output.md (A/B writeup) | gpt-4o produced 78 % salvageable vs 64 % for gpt-oss-120b on identical 50 questions; 0 loop errors vs 11; better per-example faithfulness (Pass-3 score 5/5 vs 3/5 on common smoke question). Cost difference for 300 rows ($6.68 vs $0.40) is a rounding error against the thesis budget; quality lift compounds into every Faithfulness / Context Precision / Answer Correctness number. **plan.md §0 #10 is to be reverted** when the production run completes — see "lock pending" entry below. |
| 2026-05-04 | **New 3-pass prompts adopted + multi-hop tightening locked** | src/generation/golden_prompts.py; notebooks/04_golden_main_gpt4o.ipynb | User-provided new prompts replace the original three pass templates: structured `selected_chunks` list (with support_level + reason), verbatim `best_gold_context` concatenation by Pass 1, `answer_match` boolean in Pass 3, staged-JSONL pipeline. Added one-sentence multi-hop tightening to Pass 2: *"yes ONLY when answering requires combining ≥2 distinct facts from ≥2 different gold passages AND the answer cannot be inferred from any single passage alone."* **Empirical validation 2026-05-04 pilot:** multi-hop rate dropped from 66 % → 8 % (58 pp reduction). |
| 2026-05-04 | **`hallucination_check_points` validator relaxed from `>= 3` to `>= 1`** | src/generation/golden_prompts.py `validate_pass2` | Pilot-run diagnosis showed all 12 Pass-2 schema failures had exactly 2 check_points (perfectly reasonable atomic claims; the model was right, the validator was wrong). The original prompt did not specify a minimum count. Lesson recorded in 04_golden_main_output.md §3.3: validators should mirror the prompt's literal contract, not impose extra structure. Recovers ~12 rows for ~$0.06 added Pass-3 cost. |
| 2026-05-04 | **Phase 3 pilot success (50-row run, gpt-4o + new prompts)** | docs/output_notes/04_golden_main_output.md | 4 of 5 quality gates passing. Salvageable rate 74 %, multi-hop 8 %, JSON malformation 0 %, sufficiency 94 %. Only failing gate is accept rate (54 %) — root cause was the now-relaxed `check_points >= 3` validator. **Production run unblocked**: change `STAGE = "production"` in notebook §1, Restart Kernel & Run All; cache covers first 50, only 250 net new at ~$5.40 added cost. |
| 2026-05-04 | **Phase 3 PRODUCTION COMPLETE — 234/300 accepted** | docs/output_notes/04_golden_main_output.md §5; data/processed/golden_ragas_300.jsonl | Production run on 300 questions: **234 accepted, 53 needs_review, 13 dropped, salvageable 95.7 %.** Cost **$6.61** measured (matches the original locked plan). 4/5 gates pass — accept rate 78 % is 2 pp below the 80 % gate but the actual deliverable target (≥ 220 accepted) is exceeded. Multi-hop rate held at 6 % at scale; JSON malformation 0.11 %. Wall time ~80 min. **Constructor swap from gpt-4o → gpt-oss-120b is now formally REVERTED** across plan.md / tech_stack.md / architecture.md / dataset.md / beginners_guide.md / thesis_understanding.md / README.md / memory; gpt-oss-120b A/B preserved as historical record in `notebooks/04_golden_dataset_gptoss.ipynb` and `docs/output_notes/04_notebook_output.md`. |
| 2026-05-05 | **Phase 4 EXP_01 implementation COMPLETE — runner skeleton + No-RAG retriever + notebook ready** | src/data/loaders.py · src/retrieval/{base,none}.py · src/eval/{non_llm_metrics,runner}.py · tests/test_exp01_modules.py · docs/results_schema.md · notebooks/04a_exp01_base_llm.ipynb | Built only what EXP_01 needs to run: 5 new src modules + 1 notebook + 1 test file + 1 schema-lock doc. **Schema lock**: `summary.json` keys mirror Excel `Results Table` headers verbatim (`Acuuracy` typo preserved; RAGAS keys ship as `null` until judge lands) so EXP_02–EXP_05 paste into the workbook without rework. **Resumability** built into `runner.run_experiment` — skips `question_id`s already in `predictions.jsonl`, so a 5-h Groq run that dies mid-stream restarts free of charge. **Question_id is dataset-local**: `medqa_NNNNN` for the full surface (row index in `medqa_4opt.parquet`), `golden_NNN` for the golden 234 (the stratified-sample index 0..299 from Phase 3 — NOT the medqa_4opt row index; the two surfaces join by `question` text per dataset.md §2.2). All 5 real-data tests pass; 3-question runner smoke shows mean latency 1.31 s and resume making 0 net Groq calls. **User runs the experiment manually**; this entry covers the implementation step only. **Deferred to next session**: `src/eval/ragas_eval.py` (Claude 3.5 Sonnet judge, fills the four `RAGAS_*` columns + `Answer_Correctness` for EXP_01's `golden_234` predictions). EXP_02→EXP_05 just swap the retriever and reuse this scaffolding. |
| 2026-05-05 | **Path-resolution bug fixed in `src/data/loaders.py` + `src/utils/cache.py`** | src/data/loaders.py · src/utils/cache.py | Both modules used relative paths (`Path("data/processed")`, `Path("data/cache")`) for their defaults. Worked when CWD = repo root (script + tests) but broke under Jupyter where CWD = `notebooks/`: `FileNotFoundError: data/processed/medqa_4opt.parquet`. The cache bug was *silent* — would have silently written to `notebooks/data/cache/...` and broken resumability without any error. Fix: anchor defaults to `Path(__file__).resolve().parents[2]` so paths resolve to repo root regardless of caller's CWD. All 5 tests still pass; cache module preserves `THESIS_CACHE_DIR` env-var override. |
| 2026-05-05 | **EXP_01 RUN COMPLETE — full 12,723 + golden 234 + smoke 50** + **dataset contamination empirically validated** | results/exp_01_base_llm__{smoke_50,golden_234,full_12723}/ · docs/output_notes/04a_exp01_output.md | Full run: **Acuuracy = 0.8693** (11,060 / 12,723), mean_latency = 0.279 s, wall_time = 58 min, p95 latency = 0.466 s, 0 parse failures across 13,007 calls (smoke + golden + full). **The methodology-critical finding**: split-stratified accuracy gap is **train + dev = 0.880 vs test = 0.774 (10.61 pp)**. Aligns with literature LLaMA No-RAG ceiling on MedQA-US (~75–78 %), confirming the [plan.md §15](../plan.md) contamination risk. Long-vignette accuracy (n=518) is **0.853** vs short = **0.870** — only 1.7 pp gap, which raises the bar for EXP_05 Multi-Hop ("must beat the rest by enough to justify 3× compute"). Golden 234 split mix is 188 train / 28 dev / 18 test — heavily contaminated, so its 0.902 number is *expected* given the underlying split mix. Implication: every subsequent architecture reports paired (full / test-only) numbers; thesis discussion pivots from "RAG raises accuracy" to "RAG reduces hallucination on Faithfulness" via the confidence-aware-rejection novelty layer. Wall-time projection for Phase 4 revised down from ~30 h Groq to ~5 h total for all 5 baseline experiments combined. |
| 2026-05-05 | **`src/eval/ragas_eval.py` + `notebooks/04a_exp01_ragas.ipynb` BUILT (RAGAS judge wired)** | src/eval/ragas_eval.py · notebooks/04a_exp01_ragas.ipynb · tests/test_ragas_eval.py · docs/results_schema.md §2.3 · requirements.txt | RAGAS 0.4.3 + LangChain `ChatAnthropic` + local BGE-large embeddings (no OpenAI dependency for embeddings → preserves the three-family-separation methodology). **Option A locked for No-RAG metric applicability**: Faithfulness, Context Precision, Context Recall return `null` for EXP_01 because they're undefined without retrieved context — defended in [results_schema.md §2.3](results_schema.md) as cleaner than asserting 0.0/100% without measurement. Auto-detects applicability per row via `_has_context` flag — EXP_02→EXP_05 will run all five metrics with zero per-experiment config. **Resumability**: `ragas_scores.csv` is the cache barrier — re-runs skip the judge entirely. Smoke (10 rows ~$0.30) → full (234 rows ~$3–5) gating in the notebook. 6 / 6 structure tests pass without making any Anthropic calls; 5 / 5 EXP_01 tests still pass. **Pending user action**: add `ANTHROPIC_API_KEY` to `.env`, then run notebook smoke + full. Methodology shift: EXP_01 contributes Answer Relevancy + Answer Correctness only; the Faithfulness baseline lands with EXP_02 onward. |
| 2026-05-06 | **RAGAS judge upgraded `claude-3-5-sonnet-20241022` → `claude-sonnet-4-6`** + **Phase 4 RAGAS cost recalibrated $10–15 → $140–160** | plan.md §0 #11 + §5 + §6 + §14 + §15 · docs/tech_stack.md §2.4 + §3.4 + §4 · docs/architecture.md §3 + §4 · docs/results_schema.md §2.1 · docs/todo.md §3 + §5 · AGENTS.md §2.2 · README.md · docs/thesis_understanding.md · docs/output_notes/04a_exp01_output.md · src/eval/ragas_eval.py · notebooks/04a_exp01_ragas.ipynb + 04a_exp01_base_llm.ipynb · requirements.txt · memory/project_thesis_overview.md | **Why upgrade**: Sonnet 4.6 has materially better structured-output adherence + sub-statement claim verification — exactly the workload RAGAS Faithfulness needs. Pricing is identical ($3/M input · $15/M output across Sonnet generations) so the swap costs $0 in pricing differential; the win is fewer NaN scores + better hallucination detection. Defensibility-wise, defaulting to a Q4-2024 model in 2026 invites the viva question *"why didn't you use the current best judge?"* — the upgrade closes that. **Why cost recalibrated**: walked through the realistic per-metric call structure (Faithfulness 2–4 calls/row, Context Precision k=5 calls/row, AnswerCorrectness 2–3 calls/row) at ~3000 input + 200 output tokens each. Plan.md's original $10–15 figure was ~10× optimistic. New estimate: EXP_01 ~$8–12 (2 metrics, 234 rows), EXP_02–EXP_05 ~$30–40 each (5 metrics × 234), Phase 4 RAGAS total ~$140–160. Total thesis API spend revised $25–35 → ~$150–170 (still well within an MSc budget). **API path note (debugged on 2026-05-06)**: RAGAS 0.4.3 has two parallel + incompatible metric hierarchies. The capitalized `ragas.metrics.collections.*` classes (e.g. `Faithfulness(llm=...)`) require `ragas.llms.llm_factory`'s `InstructorLLM` and inherit from `SimpleBaseMetric` — they are **not** recognised by `evaluate()` (different `BaseMetric` tree). The legacy lowercase singletons (`from ragas.metrics import faithfulness, ...`) inherit from `Metric → MetricWithLLM`, are accepted by `evaluate()`, and require a `LangchainLLMWrapper`-compatible LLM. We use the legacy `evaluate()` path — clean batch orchestration, file-level resumability via `ragas_scores.csv`, scheduled for removal only at RAGAS v1.0 (well after thesis submission). The deprecation warnings on import are silenced inside `_get_metric_singleton`. **Critical timing**: the upgrade was made BEFORE any RAGAS run landed, so no architectures will be judged with mixed models. Switching mid-Phase 4 would have invalidated cross-architecture comparison. |
| 2026-05-06 | **EXP_01 RAGAS run COMPLETE — `Answer_Correctness = 0.8738`, `RAGAS_Answer_Relevance = 0.5977`** + **NaN issue surfaced** | results/exp_01_base_llm__golden_234/{ragas_scores.csv, summary.json} · docs/output_notes/04a_exp01_output.md §4 | Stage A smoke (10/10 scored, 0 NaN, AC gap 0.88 vs 0.45 = 43 pp on correct/wrong) → Stage B full 234. **Calibrated judge confirmed**: AC mean 0.93 on the 124 correct rows that scored, 0.31 on the 13 wrong rows that scored ⇒ 62 pp gap. Median AC on correct = 1.000, 75th pct = 1.000 — judge gives perfect scores when LLaMA's prose response exactly matches the reference (211 / 234 rows). Answer Relevancy ~0.60 across all strata is *expected* RAGAS behaviour for short MCQ-style answers (response ≈ option text vs full clinical-vignette user_input — embedding cosine geometry is bounded). **By MedQA split**: AC train=0.872, dev=0.895, test=0.855 — within ±2 pp, demonstrating the judge is robust to the MedQA-pretraining-contamination signal that drove the 12 pp gap on Exact Match. **By question_type**: management is hardest (Acuuracy 0.82, AC 0.83 — both lowest of 4 buckets) — concrete hypothesis for EXP_02–EXP_04 to refute or confirm. **NaN issue (~40 % of rows)**: 97/234 NaN on AC, 60/234 NaN on AR. Diagnosis confirmed transient API failures (uniform NaN rate across all data slices ⇒ not content-related). RAGAS's `raise_exceptions=False` swallows transient errors at the ~1,400-call scale of a full RAGAS pass. **Blocking action before EXP_02**: configure `RunConfig(max_retries, max_wait)` to auto-retry, and add `rescore_nans()` mode to fill in missing rows cheaply (~$5 to complete EXP_01, ~$12–16 saved per architecture afterward). Doesn't invalidate EXP_01's signal — n=137 with a 62 pp gap is far above significance — but the writeup wants completeness. |
| 2026-05-06 | **NaN-resilience layer ADDED to `src/eval/ragas_eval.py`** (RunConfig tightening + `rescore_nans` mode) | src/eval/ragas_eval.py · tests/test_ragas_eval.py · notebooks/04a_exp01_ragas.ipynb · docs/results_schema.md §5 | **Two related fixes for the EXP_01 NaN issue, both no-LLM-cost preventatives**: (1) `_run_judge` now passes `RunConfig(max_workers=4, max_retries=10, max_wait=120, timeout=180)` to RAGAS's `evaluate()`. The default `max_workers=16` was overwhelming Anthropic's per-minute rate cap with concurrent calls; lowering to 4 (with longer max_wait for sustained-throttle backoffs) should drop on-first-pass NaN from ~40 % to a single-digit %. (2) Added `score_predictions(..., rescore_nans=True)` mode + `_merge_partial_scores` helper. Loads existing `ragas_scores.csv`, identifies rows whose active-metric cells are NaN, re-judges only those, merges new scores back (preserves already-good cells via `pd.isna(old) and pd.notna(new)` gate). Mutually exclusive with `overwrite=True`. **4-mode idempotency contract** documented in [results_schema.md §5.2](results_schema.md): fresh / cache hit / full rerun / NaN rescore. Notebook gained Stage C (`RUN_RESCORE_NANS` flag) sandwiched between Stage B and Inspect — single cell, ~$5 to complete EXP_01. **10/10 RAGAS structure tests pass** (4 new: `_nan_question_ids` correctness + missing-column tolerance, `_merge_partial_scores` NaN-only replacement + bookkeeping-column preservation). All free, no API calls. **Pending user action**: flip `RUN_RESCORE_NANS = True` and re-run notebook to complete EXP_01's RAGAS row before EXP_02 starts. |
| 2026-05-06 | **EXP_02 build COMPLETE — `src/retrieval/naive.py` + two notebooks ready** | src/retrieval/naive.py · tests/test_naive_retriever.py · notebooks/04b_exp02_naive_rag.ipynb · notebooks/04b_exp02_ragas.ipynb · docs/todo.md §3 + §5 | First non-trivial retriever — `NaiveRetriever(chroma_collection, embedder_model)` conforms to the `Retriever` ABC; the runner swaps it in for `NoRetrieval` with zero other changes. Score semantics: `Chunk.score = 1 − cosine_distance` (cosine similarity in [0,1]; higher = better). Notebooks mirror the EXP_01 layout with three gated stages each (smoke 50 → golden 234 → full 12,723 for the baseline; smoke 10 → full 234 → NaN rescore for RAGAS). RAGAS notebook fires **all 5 metrics** because retrieval is non-empty (auto-detected via `_has_context`); the resilience layer (`RunConfig(max_workers=4, ...)`) is on by default. **Wall-time projection**: baseline ~1–2 h on full 12,723 (BGE encode + Chroma + Groq per row); RAGAS ~10–20 min. **Cost**: $0 baseline (Groq) + ~$30–40 RAGAS Sonnet 4.6. **Pending user action**: run both notebooks. EXP_03 (Sparse RAG) reuses this same pair of notebooks template — only `src/retrieval/sparse.py` needs new code; copy/swap the notebook path. |
| 2026-05-07 | **EXP_04 Hybrid + EXP_05 Multi-Hop builds COMPLETE — all 4 RAG retrievers + 8 notebooks ready for batch run** | src/retrieval/{hybrid.py refactor, multi_hop.py NEW} · src/generation/prompts.py (multi-hop subquery template) · tests/{test_hybrid_retriever.py, test_multi_hop_retriever.py} · notebooks/{04d_exp04_hybrid_rag, 04d_exp04_ragas, 04e_exp05_multi_hop_rag, 04e_exp05_ragas}.ipynb · docs/todo.md §3 + §5 | Two new retriever modules + one prompt + 4 notebooks built in one session, per user's "build all baselines and RAGAS notebooks together so I can batch-run" plan. **`HybridRetriever`**: refactor of `src/retrieval/hybrid.py` — adds the `Retriever` ABC subclass while keeping the existing `hybrid_top_k` function intact for Notebook 04 golden-construction backwards compatibility. **`MultiHopRetriever`**: new module — 3-hop iterative dense retrieval; each non-first hop generates a follow-up search query via Groq using `build_multi_hop_subquery_prompt` (added to `src/generation/prompts.py`); chunks deduped across hops; early-stop on no-progress; returns up to 15 chunks per question (vs 5 for EXP_02–04). Per-question cost: 1 final answer + 2 sub-query Groq calls = 3× EXP_02 (~30–45 min wall time on test_1273; cost still $0 on Groq free tier). **All 4 retrievers + 8 notebooks tested**: 13 / 13 real-data tests pass (4 hybrid + 5 multi-hop + 4 sparse from earlier). **Notebook hypothesis-check cells**: each baseline + RAGAS notebook auto-loads earlier architectures' summary.json and prints a side-by-side comparison + verdict on the falsifiable hypothesis (Hybrid Acuuracy > 0.76 / Hybrid Context Precision ≥ 0.50 / Multi-Hop Faithfulness > 0.05 on multi-hop rows). **Pending user action**: run all 4 baselines (~3 h total, $0) → then run all 4 RAGAS notebooks ($45–48 total, ~1 h). After that Phase 4 is fully populated and ready for cross-architecture writeup. |
| 2026-05-07 | **EXP_03 Sparse RAG baseline RUN COMPLETE — `Acuuracy = 0.7581` (test_1273)** + **complementarity-with-dense empirical signal identified** | results/exp_03_sparse_rag__{smoke_50,golden_234,test_1273}/ · docs/output_notes/04c_exp03_output.md · docs/todo.md §5 EXP_03 · memory/project_thesis_overview.md | All 3 baseline surfaces written. **Headline**: EXP_03 0.7581 ≈ EXP_02 0.7573 (within 0.1 pp on test_1273); both ~1.6 pp below EXP_01 No-RAG 0.7738. **Complementarity finding (key for EXP_04)**: 153 of 1,273 test questions disagree between dense and sparse; 76 dense-right vs 77 sparse-right (essentially 50/50). Strong evidence the two retrievers catch orthogonal subsets — RRF fusion should help. **Per-USMLE-step pattern**: sparse beats dense on step2&3 (+1.68 pp; clinical-decision vocabulary matches BM25 keyword strength), dense beats sparse on step1 (+1.33 pp; basic-science semantic concepts match BGE-large strength). **Operational anomaly**: EXP_03 wall_time = 97 min for test_1273 vs EXP_02's 12 min — `rank-bm25.get_scores` is O(N=67k) per query (~4 s/query) vs ChromaDB HNSW's ~0.1 s/query. Cost is $0 either way (Groq free tier); this is a known characteristic of `rank-bm25` not having an inverted index — would warrant Pyserini / Elasticsearch in production. **RAGAS deferred**: user is batching all 4 RAG architectures' RAGAS evaluations together (run after EXP_04 + EXP_05 baselines done) instead of judging architecture-by-architecture. Saves nothing on cost; does mean the EXP_03 narrative is incomplete pending RAGAS Context Precision number. **Sharpened hypothesis for EXP_04**: Hybrid Acuuracy on test_1273 should land above 0.76 (above both 0.7573 and 0.7581) if RRF correctly fuses the complementary lanes. Falsified if Hybrid ≤ 0.76. |
| 2026-05-07 | **EXP_03 build COMPLETE — `src/retrieval/sparse.py` + two notebooks ready** | src/retrieval/sparse.py · tests/test_sparse_retriever.py · notebooks/04c_exp03_sparse_rag.ipynb · notebooks/04c_exp03_ragas.ipynb · docs/todo.md §3 + §5 | Second non-trivial retriever — `SparseRetriever(bm25_payload, chunks_df)` conforms to the `Retriever` ABC; wraps the existing `bm25_top_k` helper from `src/data/indices.py`. 4 / 4 real-data tests pass (loaders + score-ordering + k=0 boundary + rare-term *cisplatin* keyword recovery). Notebooks mirror the EXP_02 layout — same stratified smoke seed (42) so smoke surfaces are directly comparable across architectures; baseline + RAGAS notebooks gated identically. RAGAS notebook's inspect cell auto-compares Context Precision / Faithfulness against the now-locked EXP_02 reference numbers (CP=0.33, F=0.13) to test the falsifiable hypothesis. **Wall-time projection**: ~12 min Groq baseline (BM25 retrieval is faster than BGE encode + Chroma) + ~10–20 min RAGAS. **Cost**: $0 baseline + ~$11–12 RAGAS Sonnet 4.6. **Pending user action**: run both notebooks. EXP_04 Hybrid (next) will refactor existing `src/retrieval/hybrid.py` under the `Retriever` ABC and reuse this same notebook template. |
| 2026-05-07 | **EXP_02 Naive RAG run COMPLETE — `Acuuracy = 0.7573` (test_1273), `Faithfulness = 0.131`, Hallucination_Rate = 0.896** + **central thesis-narrative anchor identified** | results/exp_02_naive_rag__{test_1273,golden_234}/ · docs/output_notes/04b_exp02_output.md · docs/todo.md §5 EXP_02 · memory/project_thesis_overview.md | All 4 surfaces written (smoke 50 / golden 234 / test 1,273 / legacy partial). RAGAS judging ran clean (NaN rate <2 % across all 5 metrics — resilience layer dropped failure rate 95%+ from EXP_01's 25–42 %). **Headline finding: Naive RAG underperforms No-RAG.** Test_1273: 0.7573 vs 0.7738 (−1.65 pp); Golden_234: 0.8504 vs 0.9017 (−5.13 pp); Answer_Correctness: 0.838 vs 0.874 (−3.62 pp). **Mechanism (RAGAS exposes it):** Context Precision = 0.33 → 2/3 of retrieved chunks are noise; Faithfulness = 0.13 → LLaMA isn't grounding in retrieved chunks. **Smoking gun**: 88.3 % of correct answers are ungrounded (Faithfulness < 0.5) — LLaMA answers from memorisation while retrieved chunks act as distractors. **Per-stratum patterns**: multi-hop questions (n=13) get Faithfulness=0.000 (Naive cannot stitch evidence across hops); Step 2&3 questions lose 3.54 pp accuracy on test_1273 vs Step 1 unchanged; treatment-type questions hardest (Acuuracy 0.735 vs diagnosis 0.897); regression analysis on test_1273 shows Naive fixed 64 EXP_01 errors but introduced 85 new ones (net −21). **Thesis impact**: this is the central narrative anchor for the discussion chapter — *naive retrieval doesn't help on a contaminated model unless retrieval quality is high*; motivates Hybrid (EXP_04) + Multi-Hop (EXP_05) + the confidence-aware rejection layer (Phase 7) as the central novelty contribution. **Concrete falsifiable hypotheses generated for EXP_03–EXP_05**: see [output_notes/04b_exp02_output.md §4](output_notes/04b_exp02_output.md). **Cost**: $11.50 (within the recalibrated $11–12 projection from EXP_02 smoke); cumulative spend ≈ $22.61. |
| 2026-05-06 | **Evaluation surface narrowed: full 12,723 → test split (1,273)** | plan.md §0 #8 + §3 + §6 + §14 + §16 · docs/dataset.md §4 · docs/tech_stack.md §2.5 + §6 · docs/architecture.md §7.3 + §8.1 + §10 · AGENTS.md §2.2 · README.md · docs/todo.md §5 EXP_01–EXP_05 + §6 EXP_07 + §7 EXP_09 · memory/project_thesis_overview.md · notebooks/04a_exp01_base_llm.ipynb · notebooks/04b_exp02_naive_rag.ipynb · results/exp_01_base_llm__test_1273/ (derived) · results/exp_01_base_llm__full_12723/README_LEGACY.md · results/exp_02_naive_rag__full_12723/README_LEGACY.md | **Why narrow**: EXP_01's full-12,723 run revealed a **10.6 pp accuracy gap** between train+dev (0.880) and test (0.774) — strong evidence of LLaMA pretraining contamination on train+dev. The test split (n=1,273) is the only contamination-clean slice and matches MedRAG / MIRAGE's primary reporting surface, giving direct apples-to-apples comparison with published baselines. Statistical power is sound (95 % CI ±2.4 pp at p≈0.85), wall time per architecture drops 10× (~6–10 min vs ~1–2 h). Methodology framing is *cleaner* not weaker — we report on the un-contaminated subset by design. **What was preserved**: EXP_01's full-12,723 run (kept as `results/exp_01_base_llm__full_12723/` with a README_LEGACY.md marker) is the empirical anchor for the contamination-vs-clean comparison that drove this lock — it is NOT deleted. **What was abandoned**: EXP_02's partial full-12,723 run (4,844/12,723 rows, all train, no test rows touched) — zero cache value for the new surface; left on disk as `results/exp_02_naive_rag__full_12723/` with a legacy marker. **What stays unchanged**: golden 234 RAGAS surface (independent of accuracy surface; train-skewed by design but RAGAS measures answer-vs-reference grounding which is robust to memorisation — judge AC train=0.872 vs test=0.855, only 1.7 pp gap). **Migration**: EXP_01 `test_1273/` was DERIVED on 2026-05-06 by filtering full_12723 predictions to `split == 'test'` rows and recomputing summary.json aggregates — no new Groq calls, $0 cost. EXP_02 needs a fresh test_1273 baseline run (~10 min, $0). EXP_03/04/05/07 will run directly on test_1273 from the start. **Cost re-recalibrated**: EXP_01 done @ $4.50, EXP_02 RAGAS in progress, total Phase 4 RAGAS ≈ $50–70 (5 architectures × $11–12 each + safety) — empirically anchored from EXP_02 smoke @ $0.50. |
| 2026-05-06 | **Stage C rescore SKIPPED for EXP_01 — uneven-sample-size methodology footnote accepted** | docs/output_notes/04a_exp01_output.md · methodology chapter (future) | User decided not to rescore EXP_01's NaN rows (~$5 + 5 min of work, would have grown AC sample 137 → ~230). **Implication**: EXP_01 RAGAS `Answer_Correctness = 0.8738` is the locked headline figure, computed over n=137 non-NaN. EXP_02–EXP_05 will run with the new `RunConfig(max_workers=4, ...)` resilience layer in place from the start, expected on-first-pass NaN rate < 5 % → effective sample sizes ~225–234. **Methodology footnote required in the writeup** (anchor here so future-you doesn't have to reverse-engineer this): *"EXP_01 (No-RAG) RAGAS scoring was run before the throughput-resilience layer was added to the runner; ~40 % of rows returned NaN due to Anthropic rate-limit throttles absorbed by `evaluate(raise_exceptions=False)`. Reported `Answer_Correctness` is the mean over the 137 non-NaN rows. EXP_02–EXP_05 used the resilient `RunConfig(max_workers=4, max_retries=10, max_wait=120)` from the start and report means over near-complete samples (~95 %+ coverage). The per-metric correctness gap (AC = 0.93 on rows where LLaMA was correct vs 0.31 where it was wrong) holds at n=137 with effect size far above any reasonable significance threshold."* **Per-stratum analysis caveat**: EXP_01 cells like `requires_multihop=yes` (n=13 attempted, ~7 scored) and `test`-split (n=18 attempted, ~11 scored) are too thin for per-stratum claims; reporting will combine across multi-hop status and split for EXP_01 only. **Why skip**: $5/5min has marginal benefit when the 62 pp gap is already overwhelmingly significant; the methodology footnote is shorter than re-running, and EXP_02 build is the higher-leverage next move. Resilience-layer code stays in place — only the EXP_01 backfill is skipped. |
| 2026-05-11 | **PHASE 5 COMPLETE — EXP_06 complexity labels + EXP_07 adaptive routing both done; Variant A on Pareto frontier validates proposal's three-way table** | src/retrieval/{complexity,adaptive}.py · tests/test_{complexity,adaptive}.py · notebooks/05_exp06_complexity_labels.ipynb · notebooks/05_exp07_adaptive_rag.ipynb · data/processed/complexity_labels.parquet · results/exp_07_adaptive_variant_{a,b}__{smoke_50,test_1273}/ · docs/output_notes/05_exp06_output.md · docs/output_notes/05_exp07_output.md · plan.md §7.1 + §15 · memory/project_thesis_overview.md | **EXP_06** (2026-05-10): rule-based 3-bucket classifier (Simple / Moderate / Complex) over 12,723 MedQA questions; thresholds anchored to corpus 33rd/67th percentiles (words 93/133; phrases 28/41) + complex-decision/factoid-mechanism cue words. Output 29.5 / 32.7 / 37.7 % distribution; step1 skews Simple (46 %), step2&3 skews Complex (63 %). Manual review 1/100 disagreement (≤20 acceptance gate). **EXP_07** (2026-05-11): both routing variants run on test_1273. Variant A (proposal: Simple→Naive, Moderate→Hybrid, Complex→Multi-Hop) **Acuuracy 0.7863 at 1.806 Groq calls/Q**; Variant B (data-driven binary: Simple→NoRAG, Moderate/Complex→Multi-Hop) 0.7832 at 2.425 calls/Q. **All three hypotheses SUPPORTED**: (H1) Variant A on Pareto frontier between No-RAG (0.7738/1.00) and Multi-Hop (0.7958/3.00); (H2) Variant A dominates Variant B (higher acc + fewer calls); (H3) Variant A's marginal acc/extra-call (0.0156 vs No-RAG) is 2.0× higher than Multi-Hop's marginal gain on top of A (0.0079). **Pareto frontier**: NoRAG → Variant A → Multi-Hop. All single-shot RAGs (Naive, Sparse, Hybrid) and Variant B DOMINATED. **RAGAS score-joined from underlying golden_234 (no new judge calls, $0)**: Variant A F=0.197 / CR=0.571 / AC=0.847; Variant B F=0.276 / CR=0.754 / AC=0.867 — two-axis trade-off (A wins on Acc/cost, B wins on grounding). **Regression vs No-RAG on test_1273**: Variant A net +16 (82 fixes / 66 regressions); first single-shot architecture between No-RAG and Multi-Hop with positive net (EXP_02 −21, EXP_03 −20, EXP_04 −10, EXP_05 +28). **Methodology footnote anchored (k=15 vs k=5 wrinkle)**: runner used uniform k=15 fan-out; Variant A's Naive/Hybrid lanes ran at k=15 instead of EXP_02/04's k=5; simulator (k=5 predictions) = 0.7895 vs actual = 0.7863 (Δ=−0.31 pp); Variant B's gap ≈ 0 because Multi-Hop natively returns 15 chunks. Implication: routing is not sensitive to chunk fan-out within [5, 15]; both numbers reported, actual is canonical. **Discussion-chapter Act 4**: adaptive routing captures most of Multi-Hop's accuracy gain (84 %) at a fraction of the compute (60 %); proposal's three-way table is the data-defensible Pareto-frontier choice. **Cost**: $0 (all Groq). Cumulative project spend unchanged at ~$60. **Tables 1 row 6, 2, 3, 10 ready to populate.** Next: Phase 6 LIME/SHAP (~$0 Groq, 12–24 h wall). |
| 2026-05-10 | **PHASE 4 COMPLETE — all 5 architectures' baselines + RAGAS done; EXP_05 Multi-Hop is the headline architecture** | docs/output_notes/04c_exp03_output.md (RAGAS section added) · 04d_exp04_output.md (NEW) · 04e_exp05_output.md (NEW) · plan.md §6.1 + §14 + §15 · memory/project_thesis_overview.md · docs/todo.md §5 EXP_03/04/05 | User ran EXP_04 + EXP_05 baselines (2026-05-07, $0 Groq) + EXP_03/04/05 RAGAS (2026-05-10, ~$35 Sonnet 4.6 across the 3 batched). **Headline**: EXP_05 Multi-Hop is the only RAG architecture in the comparison to beat No-RAG on the contamination-clean test split (Acuuracy 0.7958 vs 0.7738; +2.20 pp). It dominates every RAGAS metric: F=0.283 (2.2× Naive's 0.131, median 0.250 vs 0.000 elsewhere), CR=0.711 (+30 pp vs Naive), CP=0.374, Hallucination_Rate=0.737, AC=0.869 (matches No-RAG). **Falsifiable hypotheses verdicts**: ✗ Sparse CP > 0.33 FALSIFIED (got 0.081); ✓ Hybrid Acuuracy > 0.76 SUPPORTED (got 0.7659, just clears); ✗ Hybrid CP ≥ 0.50 FALSIFIED (got 0.280, *worse* than Naive's 0.329 — RRF fusion with sparse's near-random CP contaminates the union — publishable counter-result for the discussion); ✓ Multi-Hop Acuuracy ≥ 0.7573 SUPPORTED (got 0.7958, +3.85 pp); ✓ Multi-Hop F > 0.05 on multi-hop subset SUPPORTED (got 0.229 vs Naive's 0.000 on same 13 rows). **Strongest single piece of evidence in Phase 4 for the memorisation thesis**: EXP_03 Sparse CP=0.081 (4× lower than Naive) yet Acuuracy=0.7581 (within 0.08 pp of Naive) — direct empirical decoupling of retrieval quality from accuracy. **Regression vs No-RAG on test_1273**: EXP_05 net +28 (101 fixes / 73 regressions; the only architecture with positive net); EXP_02/03/04 all net negative (−21/−20/−10). **Oracle ceiling on No-RAG ∪ Multi-Hop alone = 0.8531** (+5.7 pp over standalone Multi-Hop) — meaningful headroom for Phase 5 EXP_07 adaptive routing. **Implications for Phase 5**: the proposal's three-way Simple/Moderate/Complex → Naive/Hybrid/Multi-Hop split is now under review. A binary No-RAG / Multi-Hop router may capture most of the gain since Hybrid (CP=0.280, net −10) does not justify its place between Naive and Multi-Hop. EXP_07 should test both routing-table variants. **Implications for Phase 7**: only Multi-Hop has a graded Faithfulness distribution (median 0.25; 28.4 % of correct answers grounded at F≥0.5). On Naive/Sparse/Hybrid, median F=0.000 — thresholding is unstable; the threshold sweep {0.5, 0.6, 0.7, 0.8, 0.9} should run on Multi-Hop primarily. **Discussion-chapter narrative** is now three acts: (1) EXP_01→02 naive dense retrieval *hurts* a memorisation-strong LLM; (2) EXP_03→04 single-shot retrieval (sparse / hybrid / RRF) does not solve the retrieval-quality problem; (3) EXP_05 iterative multi-hop retrieval is what delivers grounded improvement. **Cost reconciled**: Phase 4 RAGAS came in at ~$50 measured (EXP_01 $4.50 + EXP_02–05 ≈ $11–13 each), 3× under the 2026-05-06 $140–160 ceiling. Total project spend now ~$60 of an MSc budget (was projected $150–170). **Tables 1, 8, 9 ready to populate** for all 5 architectures. |

---

*Last refreshed: 2026-05-11. When this file drifts from `plan.md`, `plan.md` wins — sync this file to it.*
