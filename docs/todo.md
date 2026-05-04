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

- [ ] `src/data/loaders.py` — load parquet + golden JSONL
- [x] `src/data/chunker.py` — recursive 400/80 chunker (called by Notebook 01) (2026-05-03 — built alongside §2.1)
- [x] `src/data/embedder.py` — BGE-large wrapper (`load_bge`, `embed_passages`, `embed_queries` with prefix, MPS auto-detect) (2026-05-04 — built alongside §2.2)
- [x] `src/data/indices.py` — `build_chroma`/`load_chroma` + `build_bm25`/`load_bm25` + `bm25_top_k` (2026-05-04 — built alongside §2.2; loader role from todo plus the build role for the one-time §2.2 run)
- [ ] `src/retrieval/base.py` — `Retriever` ABC: `retrieve(q: str, k: int) -> list[Chunk]`
- [ ] `src/retrieval/none.py` — returns `[]` (for EXP_01)
- [ ] `src/retrieval/naive.py` — ChromaDB top-k with BGE query prefix (Notebook 03 inlined this; refactor into `src/retrieval/` before EXP_02)
- [ ] `src/retrieval/sparse.py` — BM25 top-k
- [ ] `src/retrieval/hybrid.py` — RRF fusion (k=60)
- [ ] `src/retrieval/multi_hop.py` — decompose → 1–3 hops → accumulate
- [x] `src/generation/groq_client.py` — Groq wrapper with disk cache (key = sha256 of `provider + model + temp + prompt`); returns `(text, latency_s, was_cached)` (2026-05-04 — built alongside §2.3, exercised in 3 real Groq calls)
- [x] `src/generation/prompts.py` — `build_evidence_grounded_prompt` + `build_no_rag_prompt` + permissive `parse_letter` (2026-05-04 — built alongside §2.3; multi-hop variant deferred until EXP_05)
- [ ] `src/eval/non_llm_metrics.py` — Exact Match, Retrieval Recall@K, MRR, nDCG@K, latency
- [ ] `src/eval/ragas_eval.py` — wrap RAGAS with **Claude 3.5 Sonnet** as judge (Anthropic key)
- [ ] `src/eval/runner.py` — `run_experiment(retriever, dataset, output_dir)` writes `predictions.jsonl`, `retrieval.jsonl`, `summary.json`
- [x] `src/utils/cache.py` — disk cache for all LLM calls (Groq, OpenAI, Anthropic) — JSON files at `data/cache/<provider>/<key[:2]>/<key>.json` (2026-05-04 — built alongside §2.3, exercised by `groq_complete`)
- **Deliverable:** every module has at least one passing 3-row test against real data

---

## 4 · Phase 3 — Golden RAGAS dataset (built from scratch)

### Notebook 04 — `notebooks/04_golden_ragas_dataset.ipynb`

- [ ] **Stage 0 (smoke pilot, mandatory)** — Sample **50** stratified questions, run all stages A–G end-to-end on this subset, save to `golden_ragas_50_pilot.jsonl`. **Cost: $0** (Groq free tier with `gpt-oss-120b`, recalibrated 2026-05-04 from ~$2 GPT-4o). **Pilot quality gate:** accept rate ≥ 80 %, JSON malformation < 5 %, Pass-1 sufficiency rate ≥ 90 %, all `chunk_id`s resolve, `requires_multihop` rate < 60 %. If any gate fails → fall back to GPT-4o-mini (~$1 for the pilot, ~$3 for full 300).
- [ ] **Stage A** — Stratified sample 300 total of 12,723 (4-option). 50 from the pilot + 250 fresh. Stratify by `meta_info` × length bucket; force 60 long-vignettes. Seed 42.
- [ ] **Stage B** — Hybrid retrieval (BGE + BM25 + RRF k=60) → top-10 candidates per question. Construction-time only: include gold answer text + first 8 metamap phrases in the search query.
- [ ] **Stage C — Pass 1** — `openai/gpt-oss-120b` (via Groq) evidence selection (temp 0, JSON mode) → 1–3 strongest passages, 3–8 medical keywords, `is_evidence_sufficient`
- [ ] **Stage D — Pass 2** — `openai/gpt-oss-120b` reference answer + explanation (temp 0.2, JSON mode) → `reference_answer`, `reference_explanation`, `why_other_options_are_less_suitable`, `hallucination_check_points`, `question_type`, `requires_multihop`
- [ ] **Stage E — Pass 3** — `openai/gpt-oss-120b` validation (temp 0, JSON mode) → 0–5 scores + `final_status`
- [ ] **Stage F — Audit** — Pure-Python: gold answer in reference_answer; evidence_keywords in gold_context; chunk_ids resolvable in chunks.parquet
- [ ] **Stage G — Save** — `data/processed/golden_ragas_300.jsonl` (accepted) + companion `_needs_review.jsonl` + `_dropped.jsonl`
- [ ] Manually spot-check 30 accepted rows for grounding quality
- [ ] Multi-hop label audit: spot-check 10 `requires_multihop=yes` rows; if rate > 50%, tighten the Pass-2 prompt definition and re-run those rows
- **Deliverable:** ≥ 220 accepted rows out of 300; rate of `requires_multihop=yes` < 50%; total construction cost **$0** on the locked plan (`gpt-oss-120b` via Groq free tier; recalibrated 2026-05-04 from ≤ $14 GPT-4o). Fallback ladder if pilot fails: GPT-4o-mini ($3) → GPT-4o full ($12).

---

## 5 · Phase 4 — Group A baseline experiments (EXP_01 → EXP_05)

For each experiment: write `notebooks/04*_exp0*_*.ipynb`, run on full **12,723** for accuracy/retrieval metrics, run on golden **1,000** for full RAGAS metrics. Cache every LLM call.

### EXP_01 — No-RAG baseline

- [ ] Notebook `04a_exp01_base_llm.ipynb`
- [ ] Run on full 12,723 — direct LLM answer, no retrieval
- [ ] Outputs to `results/exp_01_base_llm/`
- [ ] Metrics: Exact Match Accuracy, Answer Correctness, Hallucination Rate (Faithfulness < 0.5)
- **Tables touched:** Table 1 (row 1), Table 7 (row 1), Table 9 (col 1)

### EXP_02 — Naive RAG (dense ChromaDB top-k)

- [ ] Notebook `04b_exp02_naive_rag.ipynb`
- [ ] Run on full 12,723 with the BGE-large ChromaDB collection
- [ ] Outputs to `results/exp_02_naive_rag/`
- [ ] Metrics: Exact Match Accuracy + RAGAS suite (on golden 300) + Retrieval Recall@K + latency
- **Tables touched:** Table 1, Table 8, Table 9

### EXP_03 — Sparse RAG (BM25)

- [ ] Notebook `04c_exp03_sparse_rag.ipynb`
- [ ] Run on full 12,723 with the BM25 index
- **Tables touched:** Table 1, Table 8, Table 9

### EXP_04 — Hybrid RAG (RRF)

- [ ] Notebook `04d_exp04_hybrid_rag.ipynb`
- [ ] Run on full 12,723 — BGE + BM25 fused with RRF k=60
- **Tables touched:** Table 1, Table 8, Table 9

### EXP_05 — Multi-Hop RAG

- [ ] Notebook `04e_exp05_multi_hop_rag.ipynb`
- [ ] Run on full 12,723 — BGE-large dense retrieval per hop
- [ ] Hop budget = 3, k=5 per hop, terminate on no-new-chunks
- **Tables touched:** Table 1, Table 8, Table 9

### After Group A finishes

- [ ] Document the BGE-large embedder choice in the methodology section (one paragraph; cite TREC-COVID nDCG@10 benchmark; flag domain-fine-tuned ablation as future work)
- **Deliverable:** Tables 1, 8, 9 fully populated for the 4 baseline RAGs + No-RAG

---

## 6 · Phase 5 — Group B: Adaptive Routing (EXP_06, EXP_07)

### EXP_06 — Question complexity labelling

- [ ] `src/retrieval/complexity.py` — rule-based classifier using length, option features, metamap phrase count, cue words
- [ ] Notebook `05_exp06_complexity_labels.ipynb` — apply to 12,723 → `data/processed/complexity_labels.parquet`
- [ ] Manually review 100 random labels; iterate the rule until ≥ 80% rater agreement
- **Tables touched:** Table 3

### EXP_07 — Adaptive RAG controller

- [ ] `src/retrieval/adaptive.py` — dispatches Simple→Naive, Moderate→Hybrid, Complex→Multi-Hop using the winning embedder
- [ ] Notebook `05_exp07_adaptive_rag.ipynb` — full 12,723
- **Tables touched:** Table 1 (row 6), Table 2, Table 10

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
- [ ] Apply locked threshold to full 12,723 across all 5 architectures
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

---

*Last refreshed: 2026-05-03. When this file drifts from `plan.md`, `plan.md` wins — sync this file to it.*
