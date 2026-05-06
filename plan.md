# Thesis Execution Plan — From Scratch

> **Title:** *Systematic Comparison of Multiple Retrieval-Augmented Generative AI Architectures for Evidence-Based Medical Question Answering with Explainability and Hallucination Control*
>
> **Author:** Raja Kalavala (PN1196988) · MSc AI/ML, LJMU · Submission **March 2026**
>
> **Plan philosophy.** Build everything from zero. The pre-existing 65-row golden subset and the previous Notebook 04 pipeline are reference material — the work below stands on its own. Every decision named below is **locked** unless an explicit signal forces a change.

---

## 0. Locked decisions

| # | Decision | Value | Why |
|---|---|---|---|
| 1 | LLM (generator + reasoner) | **`llama-3.3-70b-versatile`** via Groq Cloud | Proposal §7.5.1; pilot validated; 131k context; free/cheap inference. |
| 2 | Embedding model | **`BAAI/bge-large-en-v1.5`** (1024-d, 335M, 512-token max) | Strong general SOTA. Top-tier on MTEB; ~75 nDCG@10 on TREC-COVID (medical-IR benchmark). Open weights, sentence-transformers compatible. **A medical-fine-tuned ablation (e.g. MedEmbed-large) was scoped out for compute budget — identified as future work in the writeup.** |
| 3 | Vector database | **ChromaDB** (one persistent collection) | Built-in persistence + metadata filtering. |
| 4 | Sparse index | **`rank-bm25`** (Okapi BM25) | Standard medical-IR baseline; pairs with ChromaDB for Hybrid RAG. |
| 5 | Chunking | **Recursive 400-token chunks, 80-token overlap** (20%) | 20% overlap is the standard in 2024–25 medical-RAG papers; protects against boundary loss. ~67k chunks (recalibrated 2026-05-03; original "~36k" estimate was unreachable for a 12.85 M-word corpus with this config — see Phase 2 for math). |
| 6 | Hybrid fusion | Reciprocal Rank Fusion, **k=60** | Proposal §7.6.3; standard. |
| 7 | Multi-Hop budget | **3 hops max** | Proposal §7.6.4. |
| 8 | Evaluation surface (full) | **All 12,723 MedQA US questions** | Train + dev + test combined — canonical benchmark scope per the corrected workbook. |
| 9 | Golden RAGAS reference subset | **Stratified 300 questions** built from scratch in two stages: 50-row smoke pilot → 250 more for production (= 300 total) | Drives Faithfulness, Context Recall, Context Precision, Answer Correctness on 300 rows; the remaining 12,423 still get exact-match accuracy and retrieval recall. Sized for cost-efficiency while preserving per-stratum analysis room (≥ 60 rows per `question_type` bucket). |
| 10 | Golden-set **constructor** LLM | **`gpt-4o`** via OpenAI API — three-pass JSON pipeline with new prompts (structured `selected_chunks`, verbatim `best_gold_context`, `answer_match` boolean, multi-hop tightening) | **Locked 2026-05-04 after empirical A/B comparison** (see `docs/output_notes/04_notebook_output.md`): gpt-4o produced 78 % salvageable rows vs 64 % for `openai/gpt-oss-120b` on identical 50 questions, 0 loop errors vs 11, and 5/5 vs 3/5 faithfulness on the smoke question. Production run on 300 questions: **234 accepted, 53 needs_review, 13 dropped** at $6.61 total. Multi-hop tightening in Pass 2 dropped over-labelling from 66 % → 6 %. Cost is small in absolute terms ($6.61 for the whole golden set); the cost saving from gpt-oss-120b ($0.40 for 300) was not worth the calibration loss. |
| 11 | RAGAS **judge** LLM | **`claude-sonnet-4-6`** (Anthropic) — different family from generator AND constructor | **Locked 2026-05-06**, upgraded from `claude-3-5-sonnet-20241022`. Same per-token pricing ($3/M input, $15/M output) but materially better structured-output adherence + sub-statement claim verification — exactly the workload RAGAS Faithfulness needs. Realistic cost ~$140–160 across 234 golden × applicable metrics × 5 architectures (recalibrated 2026-05-06; original $10–15 estimate was ~10× optimistic on per-row call counts). |
| 12 | Top-k passed to LLM | **k=5** | Tune in EXP_02 if Context Recall is the bottleneck. |

---

## 1. The big picture

You will run **16 experiments** in 5 groups. All 16 share infrastructure: same chunked corpus, same ChromaDB collection, same BM25 index, same prompt template, same LLM, same evaluator. Only the experiment-specific component (retrieval strategy, complexity router, confidence layer, taxonomy labeller) changes. Outputs roll up into the **12 results tables** in the Excel workbook.

```
                      RAW DATA (medqa-data/)
                              │
                              ▼
  Phase 1 ──── Data processing & EDA       (Notebook 00 ✓ DONE)
                              │
                              ▼
  Phase 2 ──── Build shared infrastructure (Notebooks 01-03)
                  · text chunking
                  · BGE-large embeddings
                  · ChromaDB + BM25 indices
                              │
                              ├──────────────┐
                              ▼              ▼
  Phase 3 ─ Golden RAGAS dataset    Phase 4 ─ Baseline RAG (EXP_01-05)
            (Notebook 04)                    (Notebooks 04a-04e)
                              │              │
                              └─────┬────────┘
                                    ▼
  Phase 5 ──── Adaptive routing               (EXP_06-07)
  Phase 6 ──── Explainability (LIME + SHAP)   (EXP_10-12)
  Phase 7 ──── Confidence-aware rejection     (EXP_08-09)
  Phase 8 ──── Hallucination taxonomy         (EXP_13-15)
  Phase 9 ──── Final synthesis                (EXP_16)
                              │
                              ▼
                     12 results tables → thesis chapters

  Phase 10 ─── Streamlit demo UI (OPTIONAL, parallel track)
              Stage A scaffolding (after Phase 3)  →
              Stage B wire-up (during Phases 4-9)  →
              Stage C polish + deploy (after Phase 9)
```

---

## 2. Repository structure (target)

```
thesis-project/
├── data/                              ← all derived artefacts (gitignored)
│   ├── processed/
│   │   ├── medqa_5opt.parquet         ✓ from Notebook 00
│   │   ├── medqa_4opt.parquet         ✓ from Notebook 00
│   │   ├── textbook_stats.parquet     ✓ from Notebook 00
│   │   ├── eda_summary.json           ✓ from Notebook 00
│   │   ├── chunks.parquet             ← from Notebook 01 (~67k rows)
│   │   ├── embeddings.npy             ← from Notebook 02 (BGE-large, ~67k × 1024 float32)
│   │   ├── golden_ragas_50_pilot.jsonl   ← from Notebook 04 stage 1 (smoke pilot, ~$2)
│   │   └── golden_ragas_300.jsonl        ← from Notebook 04 stage 2 (production, +~$10)
│   └── indices/
│       ├── chroma_textbooks/          ← ChromaDB collection (BGE-large embeddings)
│       └── bm25.pkl                   ← pickled rank-bm25 index
├── src/                               ← reusable Python modules
│   ├── data/                          (loaders, chunker)
│   ├── retrieval/                     (naive, sparse, hybrid, multi_hop, adaptive, complexity)
│   ├── generation/                    (groq_client, prompts)
│   ├── eval/                          (ragas_eval, non_llm_metrics, runner)
│   ├── xai/                           (lime_passage, shap_passage, agreement)
│   ├── confidence/                    (signals, rejection)
│   ├── taxonomy/                      (categories, labeller, analysis)
│   └── utils/                         (config, cache, retry)
├── notebooks/
│   ├── 00_data_processing_and_eda.ipynb           ✓ DONE
│   ├── 01_chunking_and_corpus_prep.ipynb
│   ├── 02_embeddings_and_indices.ipynb
│   ├── 03_smoke_test_pipeline.ipynb
│   ├── 04_golden_ragas_dataset.ipynb
│   ├── 04a_exp01_base_llm.ipynb
│   ├── 04b_exp02_naive_rag.ipynb
│   ├── 04c_exp03_sparse_rag.ipynb
│   ├── 04d_exp04_hybrid_rag.ipynb
│   ├── 04e_exp05_multi_hop_rag.ipynb
│   ├── 05_exp06_complexity_labels.ipynb
│   ├── 05_exp07_adaptive_rag.ipynb
│   ├── 06_exp10_lime_passage.ipynb
│   ├── 06_exp11_shap_passage.ipynb
│   ├── 06_exp12_xai_agreement.ipynb
│   ├── 07_exp08_confidence_signals.ipynb
│   ├── 07_exp09_confidence_rejection.ipynb
│   ├── 08_exp13_taxonomy_setup.ipynb
│   ├── 08_exp14_error_labels.ipynb
│   ├── 08_exp15_arch_error_analysis.ipynb
│   └── 09_exp16_final_ranking.ipynb
├── results/
│   └── exp_XX_<name>/
│       ├── predictions.jsonl
│       ├── retrieval.jsonl
│       ├── ragas_scores.csv
│       └── summary.json               ← row that pastes into the Excel
├── configs/
│   ├── base.yaml                      (model, chunking, top-k, prompt template path)
│   └── exp_XX.yaml                    (per-experiment overrides)
├── docs/
│   ├── README.md                      ✓ (docs index)
│   ├── beginners_guide.md             ✓
│   ├── thesis_understanding.md        ✓
│   ├── tech_stack.md                  ✓
│   ├── dataset.md                     ✓
│   ├── architecture.md                ✓
│   ├── todo.md                        ✓
│   └── thesis-files/                  (proposal PDF + experiment workbook)
├── plan.md                            ← this file
├── requirements.txt                   ✓
├── .venv/                             ✓
├── .gitignore                         ✓
└── .env                               ← GROQ_API_KEY (generator + constructor), ANTHROPIC_API_KEY (judge)
```

---

## 3. Phase 1 — Data processing & EDA  ✅ done

**Deliverables (all present):**
- `data/processed/medqa_5opt.parquet` — 12,723 rows
- `data/processed/medqa_4opt.parquet` — 12,723 rows + metamap_phrases (use this everywhere downstream)
- `data/processed/textbook_stats.parquet` — 18 rows
- `data/processed/eda_summary.json` — headline numbers
- [docs/dataset.md](docs/dataset.md) — field-level reference for the data

**Headline numbers locked in:** 12,723 questions (10,178 train / 1,272 dev / 1,273 test); 18 books / 12.85M words; 4.07% long-vignette base rate; Harrison's = 24.95% of corpus by word count.

---

## 4. Phase 2 — Build shared infrastructure (3 notebooks)

This phase builds the artefacts every later notebook *loads*. Build them once and never touch them again.

### Notebook 01 — `01_chunking_and_corpus_prep.ipynb`

**Goal.** Split the 18-textbook corpus into ~67k recursively-chunked passages (recalibrated from the original "~32k" — see acceptance-check note below).

| Step | Action |
|---|---|
| 1 | Load all 18 `medqa-data/textbooks/en/*.txt` into a single concatenated stream, tagging each chunk with its source book name. |
| 2 | Use `langchain_text_splitters.RecursiveCharacterTextSplitter`: `chunk_size=400` tokens, **`chunk_overlap=80` tokens** (20% overlap — the 2024–25 standard for medical-RAG, protects against boundary loss). Use `tiktoken` (cl100k_base) for token counting so chunks size correctly for both BGE input and the LLaMA generator. |
| 3 | For each chunk, store: `chunk_id` (deterministic, e.g. `Pharmacology_Katzung_chunk_00421`), `book_name`, `text`, `n_tokens`, `n_chars`. |
| 4 | Drop chunks with fewer than 30 tokens (boilerplate / table residue). |
| 5 | Save `data/processed/chunks.parquet`. Print per-book chunk count + token-distribution histogram to verify chunking is sensible. |

**Acceptance check (recalibrated 2026-05-03):** ~50k–75k chunks total, mean 300–380 tokens, no chunk >450 tokens, Harrison's still ~25% of chunks (corpus-frequency bias preserved per `docs/dataset.md` §3.1). The corpus has ≈ 16.7 M cl100k tokens (12.85 M words × 1.30 tok/word). With chunk_size = 400 and overlap = 80, each chunk advances ≤ 320 unique tokens, so the floor is ~52k chunks even with perfect packing; `RecursiveCharacterTextSplitter` honours paragraph/sentence boundaries and fills to ~80% of the cap, landing around 65–70k chunks at mean ≈ 320–340 tokens. The original `plan.md §0 #5` estimate of ~36k was mathematically unreachable for this corpus + config; the 400/80 config itself stays locked (BGE 512-token max ⇒ chunk_size ≤ 400).

### Notebook 02 — `02_embeddings_and_indices.ipynb`

**Goal.** Build the dense (BGE-large + ChromaDB) and sparse (BM25) indices that all four retrieval architectures will share.

| Step | Action |
|---|---|
| 1 | Load `chunks.parquet`. |
| 2 | Load `BAAI/bge-large-en-v1.5` via `sentence-transformers`. **Important:** prepend BGE's recommended retrieval-passage prefix (no prefix for passages in v1.5; the query prefix `"Represent this sentence for searching relevant passages: "` is applied at *query* time inside `src/retrieval/`). |
| 3 | Embed all chunks in batches of 32. **Measured wall-time on M1 Pro MPS for 67,599 chunks: ~355 min (≈ 6 h)** — far above the originally-projected `~22 min MPS / ~45–50 min CPU` figures. First-batch timing extrapolates to ~118 min, but sustained throughput degrades to ~16 s/batch over 6 hours (thermal throttling and/or partial-MPS coverage on BGE-large's 1024-d attention). Output is correct (norms = 1.0, shape (67599, 1024)); cost is paid once because §6 is resumable from `data/processed/embeddings.npy` (float32, ~67k × 1024 ≈ 277 MB). |
| 4 | Initialise **persistent** ChromaDB: `chromadb.PersistentClient(path="data/indices/chroma_textbooks")`. Create collection `medqa_textbooks_bge_400` with `metadata={"hnsw:space": "cosine"}`. Add chunks in batches of 1,000 — pass `ids`, `embeddings`, `documents`, `metadatas={book_name, n_tokens}`. |
| 5 | Build BM25 index over the same chunk text (lowercase + simple word-split). Save as `data/indices/bm25.pkl` alongside the chunk-id ordering. |
| 6 | Sanity check: query *"What is the first-line treatment for community-acquired pneumonia?"* on both indices (ChromaDB dense + BM25 sparse). Top-3 from each should clearly relate to pneumonia/antibiotics; if not, debug before proceeding. |

**Acceptance check:** ChromaDB collection's `count()` matches `len(chunks_df)`. BM25 returns sensible top-k. Total disk footprint ≈ 200–400 MB.

### Notebook 03 — `03_smoke_test_pipeline.ipynb`

**Goal.** End-to-end sanity check before any experiment touches the LLM at scale.

| Step | Action |
|---|---|
| 1 | Pick 3 MedQA dev questions. |
| 2 | For each: embed query (with BGE query prefix) → ChromaDB top-5 → build evidence-grounded prompt → Groq call → parse predicted option (A/B/C/D). |
| 3 | Print question, retrieved chunks, generated answer side by side. Verify: retrieval is on-topic, prompt format renders cleanly, LLM returns a valid letter. |
| 4 | Time the round-trip per question. Confirm Groq quota / rate-limit headroom. |

If anything is off here, fix it before Phase 3 — the same code path runs 12,723 × 4 times in Phase 4.

---

## 5. Phase 3 — Golden RAGAS dataset (built from scratch)

The full RAGAS suite needs **reference contexts** and **reference answers** that don't exist in raw MedQA. This phase builds them for a stratified **300-question subset** in two stages: a **50-row smoke pilot first** (~$2) to flush pipeline bugs cheaply, then **250 more for production** (~$10). The 300 size is the budget-efficient floor that preserves clean per-stratum analysis (≥ 60 rows per `question_type` bucket); see decision rationale in `docs/todo.md` decision log.

### Notebook 04 — `04_golden_ragas_dataset.ipynb`

**Goal.** Produce `data/processed/golden_ragas_300.jsonl` containing per-question `gold_context`, `reference_answer`, `reference_explanation`, `hallucination_check_points`.

**Build order:**
1. **Stage 0 (mandatory smoke pilot):** sample 50 questions, run all stages A–G end-to-end, save to `golden_ragas_50_pilot.jsonl`. Verify: Pass-1 sufficiency rate ≥ 90 %, every cited `chunk_id` resolves, `requires_multihop` rate < 60 %. If any check fails, fix the prompts or pipeline before scaling. Cost: ~$2.
2. **Stage 1 (production build):** sample 250 *additional* questions (different seed offset), run the same pipeline, merge into `golden_ragas_300.jsonl`. Cost: ~$10.

| Stage | Action |
|---|---|
| **A. Stratified sampling — 300 of 12,723** | From the 4-option dataset. Stratify across `meta_info` (Step 1 / Step 2&3) × length bucket (≤120 words / 121–200 / >200). Force 60 long-vignette rows so the multi-hop architecture has a fair test surface. Random seed = 42. |
| **B. Hybrid retrieval per question** | For each sampled row: BGE-large query (with prefix) + BM25 + RRF k=60 → top-10 candidate chunks. Construction-time only: include the gold answer text in the search query (`question + " " + answer + " " + " ".join(metamap_phrases[:8])`). The bias toward retrieving answer-supporting chunks is intentional during golden-set construction. |
| **C. Constructor evidence selection (Pass 1)** | Prompt **`gpt-4o`** (instructed JSON, T=0) with question + correct answer + 10 candidate passages. Returns: structured `selected_chunks` (chunk_id, book_name, support_level∈{strong,moderate,weak}, reason), `best_gold_context` (verbatim concatenation of selected passages), `evidence_keywords`, `is_evidence_sufficient`, `review_note`. |
| **D. Constructor reference answer (Pass 2)** | Prompt **`gpt-4o`** (T=0.2) with question + correct answer + Pass-1 `best_gold_context`. Returns: `reference_answer`, `reference_explanation`, `why_other_options_are_less_suitable`, `hallucination_check_points`, `question_type` ∈ {diagnosis, treatment, mechanism, management, other}, `requires_multihop` (yes/no). The `requires_multihop` field includes a tightened definition: *"yes ONLY when answering requires combining ≥2 distinct facts from ≥2 different gold passages AND the answer cannot be inferred from any single passage alone"* — empirically dropped over-labelling from 66 % to 6 %. |
| **E. Constructor validation (Pass 3)** | Prompt **`gpt-4o`** (T=0) to score 0–5 on evidence relevance, faithfulness, explanation quality. Adds an explicit `answer_match: bool` check. Assesses hallucination risk (low/medium/high). Decides `final_status` ∈ {accepted, needs_review, rejected}. |
| **F. Automated audit** | Pure-Python checks (no LLM): (i) gold answer text appears in `reference_answer`; (ii) all `evidence_keywords` appear in `gold_context`; (iii) every cited `chunk_id` exists in `chunks.parquet`. Any failure ⇒ `quality_status = "needs_review"`. |
| **G. Save** | Accepted rows → `golden_ragas_300.jsonl` (target ≥ 220 accepted; needs_review and rejected to companion files). |

**Cost (measured, production run 2026-05-04):** **$6.61** total for the full 300-question build (50-row pilot $1.08 + 250-row scale-up $5.53). Wall time ~80 min total (50-row pilot 12 min, production scale-up 68 min). 887 LLM calls; 138 cache hits.

**Why `gpt-4o` for the constructor:** validated empirically against `openai/gpt-oss-120b` (a free open-weights alternative on Groq) — same 50 questions, same prompts, same retrieval. Result: gpt-4o produced 78 % salvageable rows vs 64 %, 0 loop errors vs 11, and 5/5 faithfulness vs 3/5 on the common smoke question. The cost difference for 300 rows ($6.61 vs $0.40) is small relative to the total thesis budget; the quality difference compounds into every Faithfulness / Context Precision / Answer Correctness number in chapters 4 and 5. **Full A/B comparison preserved** in `docs/output_notes/04_notebook_output.md`.

**Why a different family for the judge:** the constructor is OpenAI/`gpt-4o`. The answerer (Phase 4) is LLaMA. The RAGAS judge in Phase 4 is **Claude Sonnet 4.6** — a third family — to kill evaluator-on-evaluator bias on metrics where the reference answer (constructor output) is consumed by the judge (Context Recall, Context Precision, Answer Correctness). The judge stays on the paid API because RAGAS metrics like Faithfulness require sub-statement-level hallucination detection where Claude/GPT-4-class judges are validated against human raters far more thoroughly than open-weights alternatives.

**Why build from scratch:** earlier exploratory work used MiniLM + 200-token chunks; those artefacts have been deleted. Building fresh against the new BGE-large + 400/80-token index ensures every `chunk_id` reference is valid.

**Acceptance check:** ≥ 220 accepted rows out of 300, mean per-row gold context ≥ 200 words, every `chunk_id` resolvable in `chunks.parquet`, `requires_multihop = "yes"` rate < 50% (earlier exploratory runs over-labelled multi-hop at ~77% — tighten the Pass-2 prompt definition).

**Methodology caveat to write up.** *"The golden RAGAS reference subset was set to 300 stratified questions for cost-efficiency. This sample preserves clean per-stratum analysis (≥ 60 rows per `question_type` bucket) while keeping construction cost under $15. Statistical-significance tests on architecture-level RAGAS scores are reported with this sample size acknowledged."* — saves you from a viva surprise.

---

## 6. Phase 4 — Group A: Baseline RAG (EXP_01–EXP_05)

**These five experiments produce the bulk of Tables 1, 8, 9.**

Build the `src/` modules in this dependency order before running any of EXP_01–EXP_05:

1. `src/data/loaders.py` — load parquet + golden JSONL into pandas
2. `src/retrieval/base.py` — `Retriever` ABC: `retrieve(question: str, k: int) -> list[Chunk]`
3. `src/retrieval/none.py` — returns `[]` (for EXP_01)
4. `src/retrieval/naive.py` — ChromaDB top-k with BGE query prefix
5. `src/retrieval/sparse.py` — BM25 top-k
6. `src/retrieval/hybrid.py` — RRF fusion of naive + sparse
7. `src/retrieval/multi_hop.py` — decompose → 1–3 hops → accumulate
8. `src/generation/groq_client.py` — Groq call wrapper with disk cache (key = sha256 of `model + temperature + prompt`) so repeats are free
9. `src/generation/prompts.py` — base evidence-grounded prompt + No-RAG variant + Multi-Hop prompt
10. `src/eval/non_llm_metrics.py` — Exact Match, Retrieval Recall@K (against golden `gold_chunks`), MRR, nDCG@K, latency
11. `src/eval/ragas_eval.py` — wrap the 5 RAGAS metrics with **`claude-sonnet-4-6`** as judge; runs only on the 234-row accepted golden subset
12. `src/eval/runner.py` — `run_experiment(retriever, dataset_df, output_dir)` writes `predictions.jsonl`, `retrieval.jsonl`, `summary.json`

### Run order

| # | Notebook | Retriever module | Dataset slice | Time estimate |
|---|---|---|---|---|
| EXP_01 | `04a_exp01_base_llm.ipynb` | `none.py` | full 12,723 | ~6 h Groq |
| EXP_02 | `04b_exp02_naive_rag.ipynb` | `naive.py` (k=5) | full 12,723 | ~6 h Groq |
| EXP_03 | `04c_exp03_sparse_rag.ipynb` | `sparse.py` (k=5) | full 12,723 | ~5 h Groq |
| EXP_04 | `04d_exp04_hybrid_rag.ipynb` | `hybrid.py` (k=5) | full 12,723 | ~6 h Groq |
| EXP_05 | `04e_exp05_multi_hop_rag.ipynb` | `multi_hop.py` (≤3 hops, k=5/hop) | full 12,723 | ~12–18 h Groq (3× the calls) |

**Single embedder for the whole thesis.** All five experiments use the same BGE-large ChromaDB collection. EXP_03 (Sparse / BM25) doesn't use the embedder; EXP_01 (No-RAG) doesn't retrieve. The methodology paragraph in the writeup will justify BGE-large based on its TREC-COVID nDCG@10 benchmark and identify a domain-fine-tuned-embedder ablation as future work.

For each architecture: also score against the 300 golden rows with full RAGAS. That's 5 architectures × 300 RAGAS rows × 5 metrics ≈ 7,500 Claude Sonnet judge calls ≈ **$10–15**.

**Tables filled after Phase 4:**
- **Table 1** Overall Architecture Performance (5 of 6 architecture rows)
- **Table 8** Retrieval Quality (4 RAG rows)
- **Table 9** Before/After RAG Comparison

---

## 7. Phase 5 — Group B: Adaptive Retrieval (EXP_06, EXP_07)

Adds the **complexity-routed** architecture. Fills Tables 2, 3.

### EXP_06 — Question complexity labelling

`src/retrieval/complexity.py` — rule-based labeller using:
- question word count
- option length variance
- `metamap_phrases` count
- presence of cue words: *"most likely"*, *"best next step"*, *"mechanism"*, *"except"*, *"initial management"*, *"diagnosis"*, *"treatment"*

Output: `complexity` ∈ {Simple, Moderate, Complex}, saved to `data/processed/complexity_labels.parquet`. **Manually review 100 rows for label quality before running EXP_07.** Target: ≥ 80% rater agreement on the manual sample.

### EXP_07 — Adaptive RAG controller

`src/retrieval/adaptive.py` — at query time, look up `complexity` and dispatch:
- Simple → `naive.py`
- Moderate → `hybrid.py`
- Complex → `multi_hop.py`

Run on full 12,723. Time estimate: weighted average of EXP_02 / EXP_04 / EXP_05.

**Tables filled:** Table 2, Table 3, Table 1 row 6, Table 10.

---

## 8. Phase 6 — Group C support: Explainability (EXP_10, EXP_11, EXP_12)

Run **before** confidence-aware rejection because EXP_08's confidence vector consumes LIME-SHAP agreement as a signal.

| Exp | Method | Module |
|---|---|---|
| EXP_10 | Passage-level LIME — perturb each retrieved passage (mask one, re-prompt LLM, score change). Local surrogate ranks passages by influence. | `src/xai/lime_passage.py` |
| EXP_11 | Passage-level SHAP — sample passage subsets, estimate marginal contribution per passage. | `src/xai/shap_passage.py` |
| EXP_12 | Top-1 / Top-3 overlap between LIME and SHAP rankings, per question. Correlate with Faithfulness and accuracy. | `src/xai/agreement.py` |

**Cost-control:** LIME / SHAP need many LLM calls per question. Run on a **sampled 200 questions per architecture** (5 archs × 200 = 1,000 rows × ~10 perturbed prompts = 10k Groq calls). Document the sampling in the methodology section.

**Tables filled:** Table 6.

---

## 9. Phase 7 — Group C: Confidence-Aware Rejection (EXP_08, EXP_09)

### EXP_08 — Confidence signal extraction

For each question (re-using EXP_02–EXP_07 outputs + EXP_10–EXP_12 outputs), compute:

```python
signals = {
    "retrieval_score_mean":   normalised mean of top-k similarity scores,
    "retrieval_score_var":    variance of top-k scores (low = retriever is confident),
    "ragas_faithfulness":     from src/eval/ragas_eval.py,
    "ragas_context_precision":...,
    "ragas_answer_relevancy": ...,
    "lime_top_passage_score": from EXP_10,
    "shap_top_passage_score": from EXP_11,
    "lime_shap_agreement":    from EXP_12,
}
```

All normalised to [0, 1]. Saved as `confidence_features.parquet`.

### EXP_09 — Threshold tuning + rejection

Weighted formula (Excel default — tune on a 200-row validation slice):

```
confidence = 0.30·retrieval + 0.30·faithfulness + 0.20·relevancy + 0.20·agreement
```

Sweep thresholds {0.5, 0.6, 0.7, 0.8, 0.9}. Pick the threshold that **minimises hallucination rate while keeping ≥ 70% accept rate** on the validation slice. Lock that threshold and run on full 12,723.

When `confidence < threshold`, output: *"Evidence is insufficient for a reliable answer."*

**Tables filled:** Tables 4, 5, 11.

---

## 10. Phase 8 — Group D: Hallucination Error-Type Taxonomy (EXP_13, EXP_14, EXP_15)

### EXP_13 — Define categories

`src/taxonomy/categories.py` codifies the 6 categories from the workbook:
- `unsupported_diagnosis`
- `unsupported_treatment`
- `wrong_reasoning_chain`
- `partial_evidence_misuse`
- `option_mismatch`
- `context_omission`

Plus an annotation guideline (1–2 example outputs per category).

### EXP_14 — Label low-faithfulness outputs

Filter the per-architecture outputs where `Faithfulness < 0.6`. For each:
1. Manually annotate ~150 across architectures (~3 h work).
2. If ≥ 100 manual labels, train a logistic-regression classifier on `[answer_embedding, context_overlap, option_mismatch_flag, ragas_signals]` and label the remainder. Validate on a held-out 30 manual labels.
3. Else: label everything manually (smaller working set).

### EXP_15 — Architecture-level error patterns

Cross-tab error type × architecture. Confirm or refute the workbook hypotheses:
- Naive → context omission?
- Sparse → semantic mismatch / option mismatch?
- Hybrid → fewer retrieval errors?
- Multi-Hop → reasoning-chain errors?

**Tables filled:** Table 7.

---

## 11. Phase 9 — EXP_16: Final Synthesis

Aggregate every architecture into a single row across:

```
final_score = 0.25·Accuracy + 0.25·Faithfulness + 0.20·Retrieval
            + 0.15·Safety   + 0.10·Explainability + 0.05·Latency
```

(All component scores normalised to [0, 1].)

Rank. Map ranks → deployment recommendation:
- Lowest cost → `Naive`
- Highest accuracy → whoever wins the accuracy column
- Lowest hallucination → whoever wins post-rejection accuracy
- Highest explainability → highest LIME-SHAP-agreement architecture
- Best balanced → `Adaptive` (the proposal's expected winner)

**Tables filled:** Tables 10, 12.

---

## 12. Phase 10 — Demo UI (**optional, parallel track**)

> **Status:** optional. Marked here so it has a real slot, not a vague *"I'll get to it."* Build only if time permits — the thesis defends without it; with it, the central claim becomes *visible* in 5 seconds during the viva.

**Scope guardrails.** This is a **research-presentation tool**, not a clinical deployment. The proposal §6 explicitly excludes *"Real-time clinical deployment, patient-facing testing"* — the demo UI is neither. Document it in the methodology section as: *"A Streamlit demonstration application was developed to enable interactive examination of retrieval and generation behaviour across architectures. The application reads cached experiment outputs and is a research artefact, not intended for clinical use; all answers display the system's confidence score and a safety-rejection indicator."*

**Mode.** **Cached-only** — the UI reads from `results/exp_*/predictions.jsonl` and `retrieval.jsonl`, never makes a live LLM call. No Groq/OpenAI/Anthropic dependency at demo time, no demo-day failures, fully offline-capable. (Live mode — paste a new question → real Groq call — may be added at the end of the project as a stretch goal.)

### 12.1 Stack

- **Streamlit** (Python-native, lives in the same `.venv`, runs on M1 Pro fine)
- One file per tab in `app/` directory
- Shared utilities in `app/utils.py` (cached-data loaders, plot helpers)
- Deploy to **Streamlit Cloud free tier** for the viva (one-click GitHub deploy)

### 12.2 The four tabs

| # | Tab | Reads from | Shows |
|---|---|---|---|
| 1 | **Architecture Battle** *(crown jewel)* | `results/exp_0[2-5]/predictions.jsonl` + `retrieval.jsonl` | Pick a MedQA question → 4 panes side-by-side (Naive / Sparse / Hybrid / Multi-Hop) → each pane: retrieved chunks (collapsible snippets), generated answer, predicted option, ground-truth match badge, RAGAS Faithfulness, latency. Optional 5th pane for Adaptive RAG once EXP_07 is done. |
| 2 | **Explainability View** | `results/exp_10/lime.jsonl` + `results/exp_11/shap.jsonl` + `results/exp_12/agreement.parquet` | Pick any answered question + architecture → show LIME and SHAP rankings of which retrieved passages drove the answer, with the relevant text highlighted. Show LIME-SHAP top-1 / top-3 agreement. |
| 3 | **Confidence & Safety** | `results/exp_08/confidence_features.parquet` + `results/exp_09/threshold_sweep.csv` | Slider for the rejection threshold (0.5–0.9) → live update of accept rate, accuracy, hallucination rate. Sample list of rejected questions with the signal that triggered the reject. |
| 4 | **Results Dashboard** | All `results/exp_*/summary.json` | Interactive versions of the 12 Excel tables — filter by `complexity`, `question_type`, `requires_multihop`. Drill down into any cell to see the underlying questions. |

### 12.3 Three-stage build

| Stage | When | Effort | Deliverable | Acceptance |
|---|---|---|---|---|
| **A — Scaffolding** | After Phase 3 (golden dataset done), before Phase 4 baseline runs | ~2 days | Empty Streamlit app at `app/main.py` with all 4 tabs. **Hardcoded placeholder data** in `app/_mock_data.py` (built from the legacy 65-row golden as a stand-in). The output schema for `results/exp_*/summary.json` is **frozen** during this stage — see Stage A acceptance below. | The app runs (`streamlit run app/main.py`) and displays mock content in all 4 tabs without errors. The output-schema spec is documented in `docs/results_schema.md`. **Critically: this freezes `summary.json` shape *before* you spend 30+ hours of Groq running experiments.** |
| **B — Incremental wire-up** | Continuous, parallel to Phases 4–9 | ~0 extra days | As each experiment writes to `results/exp_XX/`, swap the corresponding mock loader in `app/utils.py` for the real loader. No per-experiment UI work if Stage A's schema is right. | The app's tab N stops showing mock data and shows the real experiment's data after that experiment completes. |
| **C — Polish & demo prep** | After Phase 9 (all experiments done), before thesis writing crunch | ~3 days | Styling pass (consistent fonts, colour palette, no dev-mode warnings); screenshots for the thesis report (PNG export); deploy to Streamlit Cloud; record a 3-minute demo screencast. | Public Streamlit Cloud URL works on a phone browser; ≥6 screenshots embedded into the thesis results chapter; screencast saved to `docs/demo.mp4` (or linked from a static host). |

### 12.4 Repository additions

```
thesis-project/
├── app/                                ← NEW (only if Phase 10 is taken on)
│   ├── main.py                         (Streamlit entry point with tab router)
│   ├── tabs/
│   │   ├── battle.py                   (Tab 1 — Architecture Battle)
│   │   ├── explainability.py           (Tab 2 — LIME / SHAP view)
│   │   ├── confidence.py               (Tab 3 — Threshold slider)
│   │   └── dashboard.py                (Tab 4 — Results tables)
│   ├── utils.py                        (cached-data loaders, plot helpers)
│   ├── _mock_data.py                   (Stage A placeholder, deleted after Stage B)
│   └── assets/
│       └── style.css                   (optional Stage C polish)
├── docs/
│   └── results_schema.md               ← NEW (locks summary.json shape)
└── streamlit_app.py                    ← NEW (Streamlit Cloud entry point, imports app/main.py)
```

### 12.5 Hardware impact

Negligible. Streamlit runs on M1 Pro CPU only — no GPU, no embedding model in memory at demo time. Memory footprint at runtime is ~300–500 MB. Streamlit Cloud free tier is plenty for a viva demo (no concurrent-user pressure).

---

## 13. The 12 results tables — coverage checklist

| Table | Title | Filled by | Notebook |
|---|---|---|---|
| 1 | Overall Architecture Performance | EXP_01–EXP_07 | each baseline + adaptive |
| 2 | Adaptive Retrieval by Complexity | EXP_06–EXP_07 | `05_exp07` |
| 3 | Question Complexity Labelling Summary | EXP_06 | `05_exp06` |
| 4 | Confidence-Aware Rejection Results | EXP_08–EXP_09 | `07_exp09` |
| 5 | Confidence Signal Breakdown | EXP_08, EXP_10–EXP_12 | `07_exp08` |
| 6 | LIME / SHAP Explainability | EXP_10–EXP_12 | `06_exp12` |
| 7 | Hallucination Error-Type Taxonomy | EXP_13–EXP_15 | `08_exp15` |
| 8 | Retrieval Quality | EXP_02–EXP_07 | `04d_exp04` (or summary nb) |
| 9 | Before / After RAG | EXP_01–EXP_05 | `04e_exp05` |
| 10 | Adaptive vs Best Fixed | EXP_02 / 04 / 05 / 07 | `09_exp16` |
| 11 | Confidence Threshold Tuning | EXP_08–EXP_09 | `07_exp09` |
| 12 | Final Weighted Ranking | EXP_16 | `09_exp16` |

If every experiment writes its `summary.json` with **exactly the column names from the Excel workbook**, populating the workbook is a paste step at the end. The same `summary.json` schema also feeds the demo UI's Tab 4 (§12.2) — locking the schema in Stage A of Phase 10 saves rework.

---

## 14. Cost & runtime budget

| Phase | Compute | API cost |
|---|---|---|
| Phase 1 (EDA) ✓ | 5 min CPU | $0 |
| Phase 2 (chunk + embed once + ChromaDB + BM25) | **~6 h MPS measured** (chunk ~1 min · embed ~355 min · Chroma 1m 39s · BM25 8 s — recalibrated 2026-05-04 from the original ~25 min estimate) | $0 |
| Phase 3 (golden 300, staged 50 pilot + 250 production) — `gpt-4o` via OpenAI API | ~80 min measured | **$6.61** measured 2026-05-04 (matches the original locked plan; gpt-oss-120b A/B alternative produced lower-quality output for $0.40 — see `docs/output_notes/04_notebook_output.md`) |
| Phase 4 (Group A · 5 experiments × 12,723) | ~30–40 h Groq | small, Groq is cheap |
| Phase 4 RAGAS judge (5 archs × 234 × applicable metrics) | ~3–4 h | **~$140–160 Claude Sonnet 4.6** (recalibrated 2026-05-06; original $10–15 estimate was ~10× optimistic on per-row call counts — Faithfulness alone makes 2–4 calls/row, Context Precision makes k=5 calls/row, etc.) |
| Phase 5 (adaptive) | ~10 h Groq | small |
| Phase 6 (LIME/SHAP, sampled to 200/arch) | ~6–10 h Groq | small |
| Phase 7 (confidence) | <1 h compute (re-uses prior outputs) | $0 |
| Phase 8 (taxonomy) | 3 h human + 30 min compute | ~$3 GPT-4o-mini if classifier-assisted |
| Phase 9 (synthesis) | 30 min | $0 |
| **Phase 10 (demo UI, optional)** | ~5 days human time, spread A/B/C | $0 (Streamlit Cloud free tier) |
| **Total** | **~3–4 weeks elapsed (+ ~5 days if Phase 10 taken)** | **~$150–170** (recalibrated 2026-05-06: Phase 3 $6.61 + Phase 4 RAGAS ~$140–160 + Phase 8 taxonomy ~$3) |

The dominant cost is **wall-clock**, not money — Phase 4 RAGAS is the one paid item that meaningfully affects the budget. Disk-cache every Groq + Claude + GPT-4o response by `(experiment_id, question_id, prompt_hash)` so resuming after a rate-limit pause is free, and use file-level `ragas_scores.csv` resumability so a failed Claude run can pick up where it stopped without re-spending.

**API key inventory (locked 2026-05-06):** `GROQ_API_KEY` (LLaMA 3.3 70B generator), `OPENAI_API_KEY` (`gpt-4o` golden-set constructor + optional `gpt-4o-mini` taxonomy classifier), `ANTHROPIC_API_KEY` (Claude Sonnet 4.6 RAGAS judge). Three keys, three families. Demo UI in cached mode needs **none** of them.

---

## 15. Risk register

| Risk | Probability | Mitigation |
|---|---|---|
| Groq rate limits stall a long run | High | Disk-cache every response; resumable runners; back off + retry on 429s. |
| Golden-set construction labels `requires_multihop = yes` too aggressively (legacy 77%) | Medium | In Pass 2 prompt, define multi-hop as *"requires combining ≥2 distinct facts from ≥2 distinct passages"*. Spot-check 30 rows. |
| BGE-large index doesn't fit in 16 GB | Low | 67k × 1024 × 4 bytes ≈ 274 MB. Plenty of headroom on a 16 GB box. ChromaDB on disk. |
| Same-LLM-family bias (LLaMA generates, LLaMA judges) | High if not mitigated | RAGAS judge = **`claude-sonnet-4-6`** (different family from both generator and constructor). |
| Multi-Hop RAG loops indefinitely | Medium | Hard cap 3 hops, hard cap tokens/hop, stop early if no new chunks retrieved. |
| LIME/SHAP perturbation cost explodes | Medium | Sample 200 Q/arch (documented in methodology). |
| Manual hallucination labels are subjective | Medium | Have a second annotator label 30 cases; report inter-annotator agreement (Cohen's κ). |
| Dataset contamination — LLaMA already saw MedQA in pretraining | Medium-High | Compare EXP_01 No-RAG accuracy vs EXP_02 Naive RAG; if No-RAG is already > 75%, hallucination is the more interesting story than accuracy. |
| Index disagrees between sessions (different embedding model versions) | Low | Pin `sentence-transformers==3.0.x` and BGE model revision in requirements.txt. |
| **(Phase 10) Demo UI schema drift** — `summary.json` shape changes mid-project, breaks the UI | Medium | Lock the schema in Stage A; document in `docs/results_schema.md`; CI-style assert in `src/eval/runner.py` that every produced `summary.json` matches the schema. |
| **(Phase 10) UI consumes time better spent on experiments** | Medium-High | Phase 10 is **optional**. If at the end of Phase 9 the experiment results are weak or incomplete, drop the UI entirely and write up the thesis without it. |

---

## 16. Sequenced next moves

You're at the end of Phase 1. Phase 2 unblocks everything else.

1. **Now → next session:** build Notebook 01 (chunking) and Notebook 02 (BGE-large embeddings + ChromaDB + BM25). 1 day of work end-to-end including the smoke test in Notebook 03.
2. **Following session:** build the `src/` skeleton modules (`retrieval/`, `generation/`, `eval/runner.py`).
3. **Then:** Notebook 04 — golden RAGAS dataset construction (✅ DONE 2026-05-04). 234 accepted out of 300 attempted. `gpt-4o` via OpenAI API, $6.61 total. Canonical deliverable at `data/processed/golden_ragas_300.jsonl`.
4. **(Optional) Right after Phase 3:** Phase 10 Stage A — Streamlit scaffolding with mock data. **The key reason to do Stage A here**: it forces you to lock `summary.json` shape before any expensive experiment runs, saving rework later.
5. **Then:** Phase 4 — run EXP_01 → EXP_05 sequentially. Tables 1, 8, 9 fill from these. (UI Stage B wires up incrementally as outputs arrive.)
6. **Then:** Phase 5 → 6 → 7 → 8 → 9 in order. Each one consumes outputs of the prior.
7. **(Optional) After Phase 9:** Phase 10 Stage C — UI polish, screenshots, Streamlit Cloud deploy.
8. **Final:** thesis writing — methodology + results + discussion chapters. Most of the methodology lifts directly from this `plan.md`, the dataset README, and the Excel workbook.

If anything in the locked decisions (§0) blocks progress, surface it before sinking time into a long Groq run — the cost of pausing and re-deciding is one hour; the cost of running 30 hours on a wrong setting is 30 hours.

If the demo UI starts eating time that should go into experiments, drop it. The thesis defends without it; it does NOT defend without complete experimental results.
