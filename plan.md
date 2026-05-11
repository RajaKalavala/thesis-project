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
| 8 | Evaluation surface | **MedQA US `test` split: 1,273 questions** | **Locked 2026-05-06**, narrowed from "all 12,723" (train+dev+test combined). EXP_01's full-12,723 run revealed a 10.6 pp accuracy gap between train+dev (0.880) and test (0.774) — strong evidence of LLaMA pretraining contamination on train+dev. The test split is the only contamination-clean slice of MedQA-US and matches MedRAG / MIRAGE's primary reporting surface, giving direct apples-to-apples comparison with published baselines. n=1,273 is statistically sound for headline accuracy (CI ±2.4 pp at p≈0.85), wall time is ~6 min per architecture (vs ~1–2 h on full 12,723), and the methodology framing is *cleaner* not weaker — we're reporting on the un-contaminated subset by design. The EXP_01 full-12,723 run is preserved for the train-vs-test contamination breakdown that drove this lock. RAGAS scoring still runs on the golden 234 (independent surface). |
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

If anything is off here, fix it before Phase 3 — the same code path runs 1,273 × 5 times in Phase 4 (test split × 5 architectures, per the locked evaluation surface §0 #8).

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
| EXP_01 | `04a_exp01_base_llm.ipynb` | `none.py` | **test 1,273** (locked §0 #8) | ~6 min Groq |
| EXP_02 | `04b_exp02_naive_rag.ipynb` | `naive.py` (k=5) | **test 1,273** | ~10 min Groq |
| EXP_03 | `04c_exp03_sparse_rag.ipynb` | `sparse.py` (k=5) | **test 1,273** | ~10 min Groq |
| EXP_04 | `04d_exp04_hybrid_rag.ipynb` | `hybrid.py` (k=5) | **test 1,273** | ~10 min Groq |
| EXP_05 | `04e_exp05_multi_hop_rag.ipynb` | `multi_hop.py` (≤3 hops, k=5/hop) | **test 1,273** | ~30 min Groq (3× the calls per question) |

**Single embedder for the whole thesis.** All five experiments use the same BGE-large ChromaDB collection. EXP_03 (Sparse / BM25) doesn't use the embedder; EXP_01 (No-RAG) doesn't retrieve. The methodology paragraph in the writeup will justify BGE-large based on its TREC-COVID nDCG@10 benchmark and identify a domain-fine-tuned-embedder ablation as future work.

For each architecture: also score against the 300 golden rows with full RAGAS. That's 5 architectures × 300 RAGAS rows × 5 metrics ≈ 7,500 Claude Sonnet judge calls ≈ **$10–15**.

**Tables filled after Phase 4:**
- **Table 1** Overall Architecture Performance (5 of 6 architecture rows)
- **Table 8** Retrieval Quality (4 RAG rows)
- **Table 9** Before/After RAG Comparison

### 6.1 Phase 4 close-out (2026-05-10) — Group A baseline results

All 5 architectures' baselines + RAGAS are complete. Cross-architecture table:

| Architecture | acc_test | acc_golden | F | Hall | CP | CR | AC | AR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EXP_01 No-RAG | **0.7738** | 0.9017 | n/a | n/a | n/a | n/a | 0.8738 | 0.598 |
| EXP_02 Naive Dense | 0.7573 | 0.8504 | 0.131 | 0.896 | 0.329 | 0.412 | 0.838 | 0.596 |
| EXP_03 Sparse BM25 | 0.7581 | 0.8376 | 0.040 | 0.966 | 0.081 | 0.107 | 0.838 | 0.597 |
| EXP_04 Hybrid RRF | 0.7659 | 0.8376 | 0.094 | 0.917 | 0.280 | 0.348 | 0.827 | 0.597 |
| **EXP_05 Multi-Hop** | **0.7958** | **0.9017** | **0.283** | **0.737** | **0.374** | **0.711** | **0.869** | 0.595 |

**Headline (one-liner)**: EXP_05 Multi-Hop is the only RAG architecture in the comparison to beat No-RAG on the contamination-clean test split (+2.20 pp), and it dominates every RAGAS metric. Single-shot retrieval (Naive, Sparse, Hybrid) all underperform No-RAG by 0.79–1.65 pp.

**Falsifiable hypothesis verdicts** (anchored in EXP_02 §4 + EXP_03 §4):

| # | Hypothesis | Verdict | Anchor | Result |
|---|---|---|---|---|
| 1 | Sparse Context Precision > 0.33 | ✗ **FALSIFIED** | [`04b_exp02_output.md` §4](docs/output_notes/04b_exp02_output.md) | got 0.081 (worse than dense) |
| 2 | Hybrid Acuuracy on test_1273 > 0.76 | ✓ **SUPPORTED** | [`04c_exp03_output.md` §4](docs/output_notes/04c_exp03_output.md) | got 0.7659 (just clears) |
| 3 | Hybrid Context Precision ≥ 0.50 | ✗ **FALSIFIED** | [`04b_exp02_output.md` §4](docs/output_notes/04b_exp02_output.md) | got 0.280 (worse than Naive's 0.329) |
| 4 | Multi-Hop Acuuracy ≥ 0.7573 | ✓ **SUPPORTED** | [`04b_exp02_output.md` §3.5](docs/output_notes/04b_exp02_output.md) | got 0.7958 (+3.85 pp) |
| 5 | Multi-Hop Faithfulness > 0.05 on multi-hop subset | ✓ **SUPPORTED** | [`04b_exp02_output.md` §3.5](docs/output_notes/04b_exp02_output.md) | got 0.229 (+18 pp threshold margin) |

**Implications for downstream phases**:

- **Phase 5 EXP_07 adaptive routing**: the proposal's three-way Simple/Moderate/Complex → Naive/Hybrid/Multi-Hop split is now under review. Oracle ceiling on No-RAG ∪ Multi-Hop alone = 0.8531 on test_1273; full 5-arch oracle = 0.8696. **A binary No-RAG / Multi-Hop router may capture most of the gain** without the Hybrid lane (which has CP < Naive). EXP_07 should test both routing-table variants.
- **Phase 7 confidence-aware rejection**: Multi-Hop is the only architecture with a graded Faithfulness distribution (median 0.25). On Naive/Sparse/Hybrid, median F = 0.000 — thresholding is unstable. The threshold sweep {0.5, 0.6, 0.7, 0.8, 0.9} should be run on Multi-Hop primarily; Naive/Hybrid may need a {0.1, 0.2, 0.3} regime.
- **Discussion chapter narrative** is now three acts: (1) EXP_01→02 *naive dense retrieval hurts on a memorisation-strong LLM*; (2) EXP_03→04 *single-shot retrieval — sparse, hybrid, RRF-fused — does not solve the retrieval-quality problem*; (3) EXP_05 *iterative multi-hop retrieval is what delivers grounded improvement.* The strongest single piece of evidence is EXP_03's CP=0.081 with unchanged accuracy — direct empirical decoupling of retrieval quality from accuracy on a contaminated benchmark.

Full details in the per-experiment output notes:
- [`04a_exp01_output.md`](docs/output_notes/04a_exp01_output.md) (No-RAG + RAGAS AR/AC)
- [`04b_exp02_output.md`](docs/output_notes/04b_exp02_output.md) (Naive Dense + 5 RAGAS)
- [`04c_exp03_output.md`](docs/output_notes/04c_exp03_output.md) (Sparse BM25 + 5 RAGAS, RAGAS section added 2026-05-10)
- [`04d_exp04_output.md`](docs/output_notes/04d_exp04_output.md) (Hybrid RRF + 5 RAGAS)
- [`04e_exp05_output.md`](docs/output_notes/04e_exp05_output.md) (Multi-Hop + 5 RAGAS — **the headline finding**)

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

Run on **test 1,273** (per §0 #8). Time estimate: weighted average of EXP_02 / EXP_04 / EXP_05 ≈ 10–15 min Groq.

**Tables filled:** Table 2, Table 3, Table 1 row 6, Table 10.

### 7.1 Phase 5 close-out (2026-05-11) — Group B routing results

Both Phase 5 experiments are complete. EXP_06 produced 12,723 complexity labels at the 33rd/67th percentile-anchored thresholds with 1/100 rater disagreement on manual review. EXP_07 ran two routing-table variants on test_1273.

**Cross-architecture Pareto frontier (test_1273)**:

| Strategy | Acuuracy | calls/Q | Status |
|---|---:|---:|---|
| EXP_01 No-RAG | 0.7738 | 1.000 | **Pareto frontier** (cheapest) |
| EXP_02 Naive | 0.7573 | 1.000 | dominated |
| EXP_03 Sparse | 0.7581 | 1.000 | dominated |
| EXP_04 Hybrid | 0.7659 | 1.000 | dominated |
| **EXP_07 Variant A** (proposal) | **0.7863** | **1.806** | **Pareto frontier** (middle) |
| EXP_07 Variant B (binary) | 0.7832 | 2.425 | dominated by Variant A |
| EXP_05 Multi-Hop | 0.7958 | 3.000 | **Pareto frontier** (top) |

**Headline**: Variant A (the proposal's Simple → Naive / Moderate → Hybrid / Complex → Multi-Hop split) is the cost-efficient Pareto-frontier point. It captures **84 %** of Multi-Hop's accuracy gain over No-RAG at **60 %** of Multi-Hop's compute. Variant A's marginal accuracy gain (0.0156 acc per extra Groq call vs No-RAG) is **2.0× more efficient** than Multi-Hop's marginal gain on top of Variant A (0.0079 acc/call).

**Falsifiable hypothesis verdicts** (all SUPPORTED):

| # | Hypothesis | Verdict | Result |
|---|---|---|---|
| H1 | Variant A on Pareto frontier between No-RAG and Multi-Hop | ✓ SUPPORTED | dominates Naive/Sparse/Hybrid; sits between No-RAG and Multi-Hop on the frontier |
| H2 | Variant A dominates Variant B (higher acc + fewer calls) | ✓ SUPPORTED | 0.7863 > 0.7832, 1.806 < 2.425 |
| H3 | Variant A marginally more efficient than Multi-Hop on top of A | ✓ SUPPORTED | 0.0156 vs 0.0079 acc/call (2.0× ratio) |

**RAGAS score-join (golden_234, route per Q)**:

| Metric | Variant A | Variant B | Multi-Hop alone |
|---|---:|---:|---:|
| Faithfulness | 0.197 | **0.276** | 0.283 |
| Context Precision | 0.360 | 0.379 | 0.374 |
| Context Recall | 0.571 | **0.754** | 0.711 |
| Answer Correctness | 0.847 | 0.867 | 0.869 |

Variant B has **better grounding metrics** than Variant A (because it routes Moderate→Multi-Hop instead of →Hybrid) at higher compute cost. Two-axis trade-off:
- **Best Acc/cost**: Variant A.
- **Best grounding among adaptive**: Variant B.
- **Strictly best on every metric**: Multi-Hop alone (3× compute).

**Methodology footnote** (k=15 vs k=5 wrinkle): EXP_07's runner used uniform `k=15` chunk fan-out matching Multi-Hop's max chunk return. For Naive (Variant A's Simple lane) and Hybrid (Variant A's Moderate lane), this exceeds the k=5 used in EXP_02/EXP_04 baselines, producing different chunk sets. Variant A simulator (using k=5 underlying predictions) estimates 0.7895; actual run at k=15 lands at 0.7863 (Δ=−0.31 pp). Variant B's gap is negligible (Δ=−0.08 pp) because its lanes use Multi-Hop (which natively returns 15 chunks). The choice keeps the runner `k` parameter uniform across variants; the small accuracy cost confirms adaptive routing is not sensitive to chunk fan-out within [5, 15]. Both numbers reported for transparency; the actual run is canonical.

**Implications for downstream phases**:

- **Phase 6 LIME/SHAP**: Variant A is the deployment-realistic explainability target (on the cost-quality frontier). LIME/SHAP at the chunk level should explain *why* Naive-retrieved chunks help (or don't) on the Simple bucket — that's the Phase 4 ungrounded-correct mystery viewed through the routing lens.
- **Phase 7 confidence-aware rejection**: Variant B's higher Faithfulness (0.276 vs 0.197 for A) makes it the more useful surface for the threshold sweep. Multi-Hop alone remains the primary surface; Variant B is the secondary "routing-with-good-grounding" comparison.
- **Discussion-chapter narrative**: the Phase 4 three-act structure now extends to a fourth: *(Act 4) Adaptive routing captures most of Multi-Hop's accuracy gain at a fraction of the compute cost; the proposal's three-way table is the data-defensible Pareto-frontier choice.*

Full details in:
- [`docs/output_notes/05_exp06_output.md`](docs/output_notes/05_exp06_output.md) (complexity labels)
- [`docs/output_notes/05_exp07_output.md`](docs/output_notes/05_exp07_output.md) (adaptive routing — **the Phase 5 headline**)

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

### 8.1 Phase 6 close-out (2026-05-11) — passage-level XAI complete

Phase 6 ran with **two methodology pivots** documented in full at [`docs/output_notes/06_exp10_11_12_output.md`](docs/output_notes/06_exp10_11_12_output.md):

1. **LIME-LOO → subset-sampling LIME**. Leave-one-out ablation produced all-zero attribution on the smoke (chunks carry distributed grounding, removing 1 of k doesn't flip the answer). Pivoted to N=16 random binary masks + ridge regression. Both methods kept side-by-side for audit; subset-LIME is the canonical method.
2. **Random 200-sample → retrieval-changed sample**. LIME has nothing to attribute on questions where chunks don't drive the answer (memorisation cases). Restricted to the 205 retrieval-changed Multi-Hop questions on test_1273 (101 fixes + 73 breaks + 31 both-wrong-different-letters).

**EXP_10 canonical results** (`stage_b_retrievalchanged_mhop.jsonl`, 205 questions × Multi-Hop):
- **Signal density**: 65.4 % correctness, 78.5 % same-letter. Attribution magnitudes span [−0.5, +0.5].
- **Coefficient signs match causality**: on fix questions, 80 % of top-1 chunks have positive correctness coefs (chunks support gold); on break questions, 67 % have negative coefs (chunks distract). The flip across subsets is the empirical proof the method captures real causality.
- **Retrieval-rank vs LLM-influence decoupling**: top-influence chunk is rank-0 only 13 % of the time; mean rank of top-influence chunk = 5.05. **Publishable counter-result**: BGE/RRF retrieval rank doesn't predict which chunk drives the LLM's answer.

**EXP_11 KernelSHAP** (reuses Stage B data + No-RAG anchor; $0 new Groq):
- Signal density 90.2 % (correctness) / 100 % (same-letter) — the No-RAG anchor materially helps.
- Wall time: 0.1 sec total.

**EXP_12 LIME ↔ SHAP agreement**:
- Top-1 agreement: 51.5 % (correctness, n=134).
- Top-3 overlap: mean 0.556, median 0.667.
- **Spearman ρ: mean 0.632, median 0.706. 51 % of questions have strong agreement (ρ > 0.7).**
- Agreement is highest on "break" questions (sameletter ρ = 0.732) — both methods agree most strongly on which chunks distracted the LLM.

**Thesis-defensible methodology footnotes** anchored in [`docs/output_notes/06_exp10_11_12_output.md`](docs/output_notes/06_exp10_11_12_output.md):
- LIME signal is well-defined only on retrieval-changed questions (§2.2 there).
- LIME-LOO is structurally inadequate for distributed grounding (§2.1).
- The No-RAG anchor raises SHAP signal density 65 → 90 % (§2.6).
- LIME-SHAP strong rank correlation with moderate top-1 divergence — two complementary point-estimates feeding Phase 7 (§2.7).

**Cost**: $0 (Groq only; LIME = subset perturbations on free tier; SHAP + agreement reuse LIME data with no new LLM calls). Wall time: 24 min Stage B + 0.1 sec SHAP + 0.05 sec agreement = ~24 min total. Cumulative project spend unchanged at ~$60.

**Implications for Phase 7** (next):
- Per-question agreement scores ((top-1, top-3, Spearman ρ) × (correctness, sameletter)) feed Phase 7's confidence vector alongside Faithfulness and retrieval scores.
- On Multi-Hop, 51 % of retrieval-changed questions have strong LIME-SHAP agreement → these are the questions where the chunk-level attribution is reliable and the confidence signal is trustworthy.

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

Sweep thresholds {0.5, 0.6, 0.7, 0.8, 0.9}. Pick the threshold that **minimises hallucination rate while keeping ≥ 70% accept rate** on the validation slice. Lock that threshold and run on **test 1,273** (per §0 #8).

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
| Phase 4 (Group A · 5 experiments × 1,273 test split) | ~1 h Groq total (recalibrated 2026-05-06 from ~30–40 h on full 12,723) | $0 (Groq free tier) |
| Phase 4 RAGAS judge (5 archs × 234 × applicable metrics) ✅ | **~6 h measured** | **~$50 measured** (EXP_01 $4.50 + EXP_02–05 ≈ $11–13 each = ~$50; recalibrated again 2026-05-10 from the 2026-05-06 estimate of $140–160. Empirical per-row cost was ~$0.05 across 5 metrics — closer to the *original* $10–15 estimate than the 2026-05-06 over-correction. The full Phase 4 RAGAS spend came in 3× *under* the 2026-05-06 ceiling.) |
| Phase 5 (adaptive) | ~10 h Groq | small |
| Phase 6 (LIME/SHAP, sampled to 200/arch) | ~6–10 h Groq | small |
| Phase 7 (confidence) | <1 h compute (re-uses prior outputs) | $0 |
| Phase 8 (taxonomy) | 3 h human + 30 min compute | ~$3 GPT-4o-mini if classifier-assisted |
| Phase 9 (synthesis) | 30 min | $0 |
| **Phase 10 (demo UI, optional)** | ~5 days human time, spread A/B/C | $0 (Streamlit Cloud free tier) |
| **Total** | **~3–4 weeks elapsed (+ ~5 days if Phase 10 taken)** | **~$60 reconciled 2026-05-10** (Phase 3 $6.61 + Phase 4 RAGAS ~$50 measured + Phase 8 taxonomy ~$3 pending). Comfortably under any MSc thesis budget; the 2026-05-06 over-correction to $140–160 has been retired. |

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
| **(Phase 4 close-out 2026-05-10) Hybrid RRF underperforms its hypothesis** — CP=0.280 (< 0.50 threshold and < Naive's 0.329) | Resolved as a finding | Documented as a publishable counter-result in [`docs/output_notes/04d_exp04_output.md` §4](docs/output_notes/04d_exp04_output.md): RRF requires both retrievers to clear a precision floor; sparse's CP=0.081 contaminates the fused union. Phase 5 EXP_07 routing-table design should test a binary No-RAG/Multi-Hop split before assuming Naive/Hybrid/Multi-Hop. |
| **(Phase 4 close-out 2026-05-10) Naive/Sparse/Hybrid all underperform No-RAG by 0.8–1.7 pp on test_1273** | Resolved as central thesis finding | The contamination story (LLaMA pretrained on MedQA → 0.7738 No-RAG ceiling) plus single-shot retrieval's failure to ground the LLM (88 % of correct answers ungrounded on Naive) is the discussion-chapter anchor. Multi-Hop is the architecture that breaks the pattern (Acuuracy 0.7958, +2.20 pp; F=0.283). |
| **(Phase 5 close-out 2026-05-11) Adaptive routing does not beat Multi-Hop on raw accuracy** | Reframed: routing wins on cost-adjusted Pareto frontier | EXP_07 Variant A acc=0.7863 vs Multi-Hop 0.7958 (−0.94 pp), but at 1.806 vs 3.0 Groq calls/Q. Variant A captures 84 % of Multi-Hop's gain over No-RAG at 60 % of the compute. The honest framing is the Pareto frontier (No-RAG → Variant A → Multi-Hop); on that axis, Variant A is the cost-efficient deployment-realistic point. Avoids the proposal's premise that routing would beat fixed architectures on accuracy (which the data did not support). |
| **(Phase 5 close-out 2026-05-11) k=15 vs k=5 chunk-fan-out wrinkle in Variant A** | Documented as methodology footnote | EXP_07 runner used uniform k=15 fan-out (matching Multi-Hop's max chunk return). Naive/Hybrid in Variant A's lanes ran at k=15 instead of EXP_02/04's k=5. Empirical effect: Variant A actual 0.7863 vs simulator (k=5 underlying) 0.7895 (Δ=−0.31 pp). Both numbers reported; actual is canonical. Demonstrates routing is not sensitive to chunk fan-out within [5, 15] range. |

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
