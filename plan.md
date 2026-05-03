# Thesis Experiment Plan — Step-by-Step Execution Guide

> **Purpose.** Concrete, ordered execution plan for the thesis *"Systematic Comparison of Multiple Retrieval-Augmented Generative AI Architectures for Evidence-Based Medical Question Answering with Explainability and Hallucination Control."*
>
> Reads the original proposal ([docs/thesis-files/RajaKalavala_PN1196988_OriginalProposal.pdf](docs/thesis-files/RajaKalavala_PN1196988_OriginalProposal.pdf)), the experiment workbook ([docs/thesis-files/Raja Kalavala Final Thesis Project Sheet.xlsx](docs/thesis-files/Raja%20Kalavala%20Final%20Thesis%20Project%20Sheet.xlsx)), the existing golden-data construction ([docs/golden-data/](docs/golden-data/)), and the available data ([medqa-data/](medqa-data/)) into a single sequenced plan.
>
> Companion: [docs/THESIS_UNDERSTANDING.md](docs/THESIS_UNDERSTANDING.md) — the conceptual frame.

---

## 0. The big picture in one paragraph

You will run **16 experiments** (EXP_01 → EXP_16) across **5 groups**, all on the **same** evaluation set (the *golden* dataset), the **same** LLM (LLaMA 3.3 70B via Groq), the **same** prompt template, the **same** chunking + embedding + index pipeline. Only the experiment-specific component varies. Outputs feed **12 results tables** that together prove four claims: (1) RAG beats no-RAG, (2) some retrieval architectures beat others, (3) **adaptive routing** by question complexity beats any fixed architecture, (4) **confidence-aware rejection** + **hallucination-type taxonomy** make the chosen architecture safer than RAGAS-faithfulness-only thresholding.

---

## 1. Mental model — what changed since the proposal

The proposal defines the **baseline frame** (4 architectures + RAGAS + LIME/SHAP + 3-layer hallucination control). The Excel workbook **layers three novelties on top**:

| Layer | Source | What it does |
|---|---|---|
| **Baseline** (Group A) | Proposal | EXP_01 No-RAG · EXP_02 Naive · EXP_03 Sparse · EXP_04 Hybrid · EXP_05 Multi-Hop |
| **Novelty 1: Adaptive Retrieval** (Group B) | Excel | EXP_06 label question complexity (Simple/Moderate/Complex) → EXP_07 route Simple→Naive, Moderate→Hybrid, Complex→Multi-Hop |
| **Novelty 2: Confidence-Aware Rejection** (Group C) | Excel | EXP_08 build multi-signal confidence score (retrieval + faithfulness + relevancy + LIME-SHAP agreement) → EXP_09 reject below threshold; supported by EXP_10/11/12 (LIME, SHAP, agreement) |
| **Novelty 3: Hallucination Error-Type Taxonomy** (Group D) | Excel | EXP_13 define error categories → EXP_14 label low-faithfulness outputs → EXP_15 architecture-level error pattern analysis |
| **Final synthesis** (Group E) | Excel | EXP_16 weighted ranking + deployment recommendation framework |

The Excel **supersedes** the simpler "Three-Layer Hallucination Control" in §7.7 of the proposal. Instead of a single `Faithfulness < 0.5` reject rule, Novelty 2 builds a multi-signal confidence vector and tunes the threshold (Table 11: 0.5 / 0.6 / 0.7 / 0.8 / 0.9).

---

## 2. What's already in the repo

| Asset | Location | Status |
|---|---|---|
| Original proposal | `docs/thesis-files/RajaKalavala_PN1196988_OriginalProposal.pdf` | Reference |
| Experiment workbook (the source of truth for what to run) | `docs/thesis-files/Raja Kalavala Final Thesis Project Sheet.xlsx` | Reference |
| Golden-data construction docs | `docs/golden-data/{README,methodology,schema,analysis,roadmap}.md` | Done |
| Conceptual reference | `docs/THESIS_UNDERSTANDING.md` | Done |
| MedQA US questions (5-option) | `medqa-data/questions/US/{train,dev,test}.jsonl` | 10,178 / 1,272 / 1,273 |
| MedQA US questions (**4-option**, with metamap phrases) | `medqa-data/questions/US/4_options/phrases_no_exclude_{train,dev,test}.jsonl` | Same splits, options reduced to A–D |
| MedQA US qbank (full pool) | `medqa-data/questions/US/US_qbank.jsonl` | 14,369 |
| Medical textbook corpus | `medqa-data/textbooks/en/*.txt` | 18 English books · ~12.85M words · ~88 MB |
| Chinese / Taiwan data | `medqa-data/textbooks/zh_*` · `medqa-data/questions/{Mainland,Taiwan}/` | **Out of scope** per proposal §6 |
| Pilot golden dataset (100-seed, 65 accepted) | `golden-data/medqa_ragas_golden.jsonl` | Done — needs scale-up |
| Pilot intermediate artifacts | `golden-data/golden_{seed_100, candidates, evidence_selected, with_references, validated, audited}.jsonl` | Done |
| Pilot Vector DB comparison results (sample size 20) | Excel sheet "Vector DB comparision" | Done — see §3 |

---

## 3. Decisions already made in the pilot (do not redo)

The Vector DB sheet records a 20-question pilot run on all four architectures with **all-MiniLM-L6-v2** embeddings × **{FAISS, ChromaDB, Qdrant, Pinecone}**. Use those as locked-in defaults unless evidence forces a change:

| Component | Pilot choice | Status |
|---|---|---|
| Generator LLM | `llama-3.3-70b-versatile` via Groq | **Locked** |
| Vector DB | **FAISS** (matches proposal §7.4.3) | **Locked** |
| Embedding | `all-MiniLM-L6-v2` (384-d, 22M params) | **Pilot default** — proposal still lists BGE-large-en-v1.5 / MedCPT as candidates; the **"Embedding Comparision" sheet is empty** so this is the next decision to make (see §6.0) |
| Chunking | Recursive 200-token chunks, 20-token overlap (per Notebook 04) | **Pilot default** |
| Reranker / fusion | RRF k=60 for Hybrid | **Locked** (proposal §7.6.3, Notebook 04) |
| Top-K | top-10 candidates → top-k passed to LLM (k≈4–10 depending on experiment) | Tune in EXP_02 |

---

## 4. The blocker before any of EXP_01–EXP_16 can run at full scale

The Excel results tables ("Detail Guide" R3 + "Results Table" R2) explicitly say *"the same 1,500 golden MedQA questions"* across all experiments. Today you have **65 accepted rows** in `golden-data/medqa_ragas_golden.jsonl`.

You have three options. **Pick one before writing any experiment code**:

| Option | What it costs | What you get | Recommendation |
|---|---|---|---|
| **A. Scale Notebook 04 from 100 → 1,500** | ~$45 GPT-4o + ~7 hours Colab T4 (linear scale of methodology.md numbers) | A 1,500-question RAGAS-ready golden set with reference answers, gold contexts, hallucination check-points. Matches the Excel exactly. | **Do this** if the thesis writeup will quote the Excel's 1,500-question numbers. |
| **B. Run experiments on the 65-accepted set** | $0 extra | Statistically thinner per-architecture cells (~16 questions per stratified slice for a 4-arch × question_type cross-tab). Excel's "1,500" rows would need rewording in writeup. | Acceptable for early-validation runs; not ideal for the final thesis tables. |
| **C. Hybrid: 65 for full RAGAS evaluation, full 1,273 test split for accuracy-only metrics** | $0 extra | Two reporting tables: "RAGAS metrics on 65 golden" + "Exact-match accuracy on 1,273 test". Plays to each dataset's strength. | Reasonable middle path. Means Tables 1, 9 split into two views. |

**Action item:** decide A / B / C **before §6**. The rest of the plan assumes A (the cleanest match to the Excel) but works for B/C with mechanical row-count substitutions.

---

## 5. Repository layout to add

Right now everything lives in docs + data folders. To run 16 experiments cleanly, propose this layout (additions only — nothing existing is moved):

```
thesis-project/
├── data/                          # NEW — derived artifacts (gitignored except small files)
│   ├── processed/                 # cleaned & deduped MedQA + textbook chunks
│   │   ├── medqa_clean.parquet
│   │   ├── textbook_chunks.parquet
│   │   └── golden_medqa_1500.jsonl   # output of §6.A — the new evaluation set
│   └── indices/                   # built once, reused by every experiment
│       ├── faiss.index
│       ├── faiss_meta.parquet
│       └── bm25.pkl
├── src/                           # NEW — reusable Python
│   ├── data/                      # loaders, cleaners, chunker
│   ├── retrieval/                 # naive / sparse / hybrid / multi_hop / adaptive
│   ├── generation/                # Groq client, prompt templates
│   ├── eval/                      # RAGAS wrapper, exact-match, retrieval recall, latency
│   ├── xai/                       # passage-level LIME, passage-level SHAP, agreement
│   ├── confidence/                # signal extraction + threshold tuning
│   ├── taxonomy/                  # error-type definitions + labeller
│   └── utils/                     # config, logging, caching, Groq retry
├── notebooks/                     # NEW — one per experiment, thin orchestration only
│   ├── 01_base_llm.ipynb
│   ├── 02_naive_rag.ipynb
│   ├── 03_sparse_rag.ipynb
│   ├── 04_hybrid_rag.ipynb
│   ├── 05_multi_hop_rag.ipynb
│   ├── 06_complexity_labeling.ipynb
│   ├── 07_adaptive_rag.ipynb
│   ├── 08_confidence_signals.ipynb
│   ├── 09_confidence_rejection.ipynb
│   ├── 10_lime_passage.ipynb
│   ├── 11_shap_passage.ipynb
│   ├── 12_xai_agreement.ipynb
│   ├── 13_error_taxonomy_setup.ipynb
│   ├── 14_error_type_labeling.ipynb
│   ├── 15_architecture_error_analysis.ipynb
│   └── 16_final_comparative_analysis.ipynb
├── results/                       # NEW — per-experiment outputs
│   └── exp_XX_<name>/
│       ├── predictions.jsonl      # one row per question
│       ├── retrieval.jsonl        # retrieved chunks + scores per question
│       ├── ragas_scores.csv       # per-question RAGAS metric values
│       └── summary.json           # aggregate row that gets pasted into the Excel
├── configs/                       # NEW — experiment hyperparams as YAML/JSON
│   ├── base.yaml                  # shared: model, embedding, chunking, top-k
│   └── exp_XX.yaml                # per-experiment overrides
├── plan.md                        # this file
├── pyproject.toml / requirements.txt
└── .env.example                   # GROQ_API_KEY, OPENAI_API_KEY (for golden-set construction only)
```

**Why this layout works:** every experiment is a thin notebook that imports `src/` modules. The same retriever module powers EXP_02 and EXP_07's "Naive" branch. Same RAGAS evaluator runs across all 16. No code is duplicated, and a change in the prompt template propagates everywhere automatically.

---

## 6. Step-by-step execution order

Execute in this order — each step depends on the previous outputs.

### Step 0 — Environment & infrastructure (1–2 days)

1. Create Python env (Python 3.11 recommended; the proposal's 3.13 + numpy 2.3 combo is bleeding-edge — pick a known-good combo unless you want to fight library compat).
2. Add `requirements.txt`: `langchain`, `sentence-transformers`, `faiss-cpu`, `rank-bm25`, `ragas`, `lime`, `shap`, `groq`, `openai` (golden construction only), `pandas`, `pyarrow`, `tqdm`, `python-dotenv`, `pytest`, `matplotlib`, `seaborn`, `scikit-learn`.
3. `.env` with `GROQ_API_KEY` and (for §6.A only) `OPENAI_API_KEY`.
4. Add a 3-question smoke test that runs end-to-end: load 3 MedQA Qs → embed → retrieve → call Groq → score with RAGAS. This catches API/auth issues before any long run.

### Step 1 — Decide the embedding model ("Embedding Comparision" sheet — currently empty)

The proposal lists three candidates: `BGE-large-en-v1.5` (1024-d, 335M), `MedCPT` (768-d, 110M), `all-MiniLM-L6-v2` (384-d, 22M). Pilot used MiniLM. Run a small bake-off:

- On the existing 65 golden rows, build three FAISS indices (one per embedding) over the same chunked corpus.
- For each, run Naive RAG and record: Retrieval Recall@10, Context Precision, RAGAS Faithfulness, latency, index build time, memory footprint.
- Pick the winner. Fill the empty "Embedding Comparision" sheet.
- **Then freeze it** for the rest of the experiments.

### Step 2 — Build (or scale) the golden dataset → §4 decision

If §4 = A, scale Notebook 04's pipeline to 1,500. Concretely:
- Stratify across {`step1`, `step2&3`} × {`question_type` proxy from question length / cue words}.
- Re-use the cached `data/indices/` so retrieval doesn't repeat.
- Three GPT-4o passes (selection → reference → validation) + automated audit, identical to the 100-question run.
- Output: `data/processed/golden_medqa_1500.jsonl`.

If §4 = B or C, save current `golden-data/medqa_ragas_golden.jsonl` as the canonical eval set under the same path.

### Step 3 — Build shared infrastructure (`src/`)

Build the modules in this dependency order. Each module has a *test against 3 golden rows* before moving on.

1. `src/data/loaders.py` — load MedQA JSONL, golden JSONL, textbook txt files. Returns dataframes/dicts.
2. `src/data/chunker.py` — recursive 200-token / 20-token-overlap splitter (matches Notebook 04). Outputs `chunks.parquet` once.
3. `src/data/embeddings.py` — wrap chosen embedding model (from Step 1). Outputs `embeddings.npy` once.
4. `src/data/indices.py` — build & load FAISS + BM25 indices. Idempotent (loads cached if present).
5. `src/generation/groq_client.py` — Groq call wrapper with retry, rate-limit handling, response caching to disk (key: question_id + experiment_id). **Cache hits will save you ~30% of total run cost.**
6. `src/generation/prompts.py` — one shared evidence-grounded prompt template + variant for No-RAG (EXP_01) and Multi-Hop (EXP_05).
7. `src/eval/ragas_eval.py` — wrap the 5 RAGAS metrics; use a *different* judge LLM family than LLaMA 3.3 to avoid evaluator-on-evaluator bias (per `roadmap.md`). Suggest GPT-4o-mini or Claude.
8. `src/eval/non_llm_metrics.py` — Exact Match, Retrieval Recall@K, MRR, nDCG@K, latency, token counts.
9. `src/eval/runner.py` — given `(questions, retriever_fn, generator_fn) → results.jsonl + summary.json`. Used by every experiment.

### Step 4 — Run Group A: Baseline experiments (EXP_01–EXP_05)

These produce the bulk of Tables 1, 8, 9. Run sequentially — each takes a few hours of Groq calls.

| Exp | Module needed | Expected output |
|---|---|---|
| **EXP_01 No-RAG** | `src/retrieval/none.py` (returns `[]`) + base prompt | `results/exp_01_base_llm/{predictions, summary}` |
| **EXP_02 Naive RAG** | `src/retrieval/naive.py` (FAISS top-k) | `results/exp_02_naive_rag/...` |
| **EXP_03 Sparse RAG** | `src/retrieval/sparse.py` (BM25 top-k) | `results/exp_03_sparse_rag/...` |
| **EXP_04 Hybrid RAG** | `src/retrieval/hybrid.py` (FAISS + BM25 + RRF k=60) | `results/exp_04_hybrid_rag/...` |
| **EXP_05 Multi-Hop RAG** | `src/retrieval/multi_hop.py` (decompose → 1–3 hops → accumulate) | `results/exp_05_multi_hop_rag/...` |

For each: same `src/eval/runner.py`, same prompt template, same LLM, same indices. Only the retriever changes.

After this step you can fill **Table 1** (Overall Architecture Performance), **Table 8** (Retrieval Quality), **Table 9** (Before/After RAG).

### Step 5 — Run Group B: Adaptive Retrieval (EXP_06, EXP_07) → fills Tables 2, 3

- **EXP_06** — `src/retrieval/complexity.py`: rule-based labeller using question length, option length, metamap phrase count, presence of cues like *"most likely"*, *"best next step"*, *"mechanism"*, *"except"*, *"initial management"*. Output: `complexity_labels.parquet` keyed by `question_id`. Manually review 10% for label-quality.
- **EXP_07** — `src/retrieval/adaptive.py`: at query time, look up complexity → dispatch to Naive / Hybrid / Multi-Hop retriever (reuse Step 4 modules). Same prompt + LLM. Output: `results/exp_07_adaptive_rag/...`.

Fills **Table 2** (Adaptive Results by complexity) and **Table 3** (Complexity Labelling Summary).

### Step 6 — Run Explainability (EXP_10, EXP_11, EXP_12) → fills Table 6

- **EXP_10** — `src/xai/lime_passage.py`: passage-level perturbation (mask one passage at a time, regenerate, score change). Local surrogate model ranks passages by influence.
- **EXP_11** — `src/xai/shap_passage.py`: passage-subset sampling, marginal contribution per passage.
- **EXP_12** — `src/xai/agreement.py`: top-1 / top-3 overlap between LIME and SHAP rankings; correlate with Faithfulness and accuracy.

Run on **all five generating architectures** (Naive, Sparse, Hybrid, Multi-Hop, Adaptive). Note: LIME/SHAP at passage level requires re-calling the LLM many times per question — budget accordingly. Sample a subset (e.g. 200 questions per architecture) if cost is a concern.

Fills **Table 6** (LIME/SHAP Explainability) and feeds the LIME-SHAP-agreement signal into Step 7.

### Step 7 — Run Confidence-Aware Rejection (EXP_08, EXP_09, plus Threshold Tuning Table 11) → fills Tables 4, 5, 11

- **EXP_08** — `src/confidence/signals.py`: per-question vector with `[retrieval_score_mean, retrieval_score_variance, ragas_faithfulness, ragas_context_precision, ragas_answer_relevancy, lime_top_passage_score, shap_top_passage_score, lime_shap_agreement]`. Normalise to [0,1].
- **EXP_09** — `src/confidence/rejection.py`: weighted formula (Excel suggestion: `0.30·retrieval + 0.30·faithfulness + 0.20·relevancy + 0.20·explanation_agreement`). Tune threshold on the validation slice across {0.5, 0.6, 0.7, 0.8, 0.9}. Reject below threshold; emit *"Evidence is insufficient for a reliable answer."*

Compute pre- vs post-rejection accuracy, hallucination rate, rejection rate. Fills **Table 4** (Rejection Results), **Table 5** (Signal Breakdown), **Table 11** (Threshold Tuning).

### Step 8 — Run Hallucination Error-Type Taxonomy (EXP_13, EXP_14, EXP_15) → fills Table 7

- **EXP_13** — `src/taxonomy/categories.py`: codify the six error categories listed in the workbook (`unsupported_diagnosis`, `unsupported_treatment`, `wrong_reasoning_chain`, `partial_evidence_misuse`, `option_mismatch`, `context_omission`). Write the annotation guideline.
- **EXP_14** — `src/taxonomy/labeller.py`: filter outputs where `Faithfulness < threshold`. Manually annotate ~150 across architectures (or use GPT-4o as a classifier if cost-prohibitive). If you label ≥100 samples, train a small classifier (logistic regression on `[answer_text_embedding, context_overlap, option_mismatch_flag, ragas_signals]`) to label the rest.
- **EXP_15** — `src/taxonomy/analysis.py`: cross-tab error type × architecture. Confirm or refute the Excel hypotheses (Naive→context omission, Sparse→semantic mismatch, Hybrid→reduces retrieval errors, Multi-Hop→reasoning chain errors).

Fills **Table 7** (Error-Type Taxonomy).

### Step 9 — Final Comparative Analysis (EXP_16) → fills Tables 10, 12

- Aggregate every architecture's metrics into a single row.
- Compute the **weighted score** (per Excel: `0.25·Accuracy + 0.25·Faithfulness + 0.20·Retrieval + 0.15·Safety + 0.10·Explainability + 0.05·Latency`).
- Rank.
- Map ranks to deployment recommendations (low-cost / high-accuracy / low-hallucination / high-explainability).

Fills **Table 10** (Adaptive vs best fixed) and **Table 12** (Final Weighted Ranking).

---

## 7. The 12 results tables — which experiments fill them

| Table | Title | Filled by |
|---|---|---|
| 1 | Overall Architecture Performance | EXP_01–EXP_07 |
| 2 | Adaptive Retrieval Results by Complexity | EXP_06, EXP_07 |
| 3 | Question Complexity Labelling Summary | EXP_06 |
| 4 | Confidence-Aware Rejection Results | EXP_08, EXP_09 |
| 5 | Confidence Signal Breakdown | EXP_08, EXP_10–EXP_12 |
| 6 | LIME and SHAP Explainability Results | EXP_10–EXP_12 |
| 7 | Hallucination Error-Type Taxonomy Results | EXP_13–EXP_15 |
| 8 | Retrieval Quality Comparison | EXP_02–EXP_07 (retrieval logs) |
| 9 | Before and After RAG Comparison | EXP_01–EXP_05 |
| 10 | Adaptive RAG vs Best Fixed Architecture | EXP_02, EXP_04, EXP_05, EXP_07 |
| 11 | Confidence Threshold Tuning | EXP_08, EXP_09 |
| 12 | Final Weighted Ranking | EXP_16 |

If you stamp `summary.json` per experiment with **exactly the column names from the Excel**, populating tables becomes a row-paste step at the end.

---

## 8. Cost & runtime estimate

| Phase | Model calls | Wall time | API cost (approx) |
|---|---|---|---|
| Step 1 (embedding bake-off) | 65 Q × 3 embeddings × 1 retrieval pass | ~1 h | $0 (Groq free tier covers it) |
| Step 2A (golden 1,500) | 1,500 × 3 GPT-4o passes | ~7 h Colab T4 | ~$45 GPT-4o |
| Step 4 (Group A, 5 experiments) | 5 × 1,500 = 7,500 Groq calls | ~6 h | small (Groq is fast & cheap) |
| Step 5 (Adaptive) | 1,500 Groq calls | ~1.5 h | small |
| Step 6 (LIME/SHAP) | high — depends on perturbations per question | many hours | sample 200 Q/arch to bound cost |
| Step 7 (Confidence) | re-uses prior outputs | <1 h compute | $0 |
| Step 8 (Taxonomy) | manual labels + optional GPT-4o classifier | 1–2 days human + ~$5 GPT-4o | small |
| RAGAS judge | 5 metrics × ~7,500 outputs × judge model | several hours | ~$15–30 with GPT-4o-mini |
| **Total** | | **~3–4 weeks** | **~$70–100** |

---

## 9. Risks & how to mitigate (extends proposal §9.2)

| Risk | Why it matters | Mitigation |
|---|---|---|
| Golden 1,500 takes too long / costs too much | Blocks every Group | Start with 65 + topup batch of 500 first; only scale to 1,500 once pipeline is verified |
| Groq rate limits during long runs | Mid-experiment failure | Disk-cache every Groq response keyed by `(question_id, experiment_id, prompt_hash)`; resume from cache on restart |
| Same-LLM-family bias (LLaMA generates, LLaMA judges) | RAGAS scores artificially high | Use GPT-4o-mini or Claude as RAGAS judge — *different family from generator* |
| Multi-Hop loops indefinitely | Wastes Groq calls | Hard cap at 3 hops, hard cap on tokens per hop, stop early if no new chunks retrieved |
| LIME/SHAP perturbation cost explodes | Step 6 dominates budget | Sample ≤200 questions per architecture, or use approximate Shapley with 50 samples |
| Complexity labels for EXP_06 are noisy | Adaptive RAG comparison becomes unfair | Manually review 100 labels (~1 hour); if >20% disagreement, refine the rule before EXP_07 |
| Hallucination taxonomy labels are subjective | EXP_15 conclusions look weak | Have a second annotator label 50 cases; report inter-annotator agreement |
| Vector DB / index doesn't fit in 16 GB RAM | EXP_02+ all fail | Chunked corpus is small (~88 MB raw). Even at 1024-d float32 embeddings × 70k chunks ≈ 280 MB. Fits comfortably. |
| Golden set's `requires_multihop=yes` rate is 77% (analysis.md §6 Finding 1) | Inflates Multi-Hop RAG win | Apply the stricter `requires_chained_retrieval` re-label from `analysis.md §11.3` before EXP_05 stratified analysis |

---

## 10. Open decisions to make before coding starts

You have to decide these — I cannot decide them for you:

1. **Golden dataset size** — A (1,500), B (65), or C (hybrid) per §4.
2. **Embedding model** — confirm pilot's MiniLM, or run §6.1 bake-off and pick BGE-large-en-v1.5 / MedCPT.
3. **RAGAS judge model** — different family from LLaMA. GPT-4o-mini (cheap, OpenAI) or Claude Haiku (cheap, Anthropic) are the practical picks.
4. **Top-k for retrieval** — Excel doesn't pin it. Try k=4 and k=8; pick whichever wins on Naive RAG Faithfulness vs latency trade-off.
5. **Multi-Hop hop budget** — fix at 3 (proposal §7.6.4). Confirm or change before EXP_05.
6. **Confidence formula weights** — Excel suggests `0.30/0.30/0.20/0.20`. Consider tuning by grid search on the validation slice rather than fixing them.
7. **Hallucination annotation method** — pure manual (slow, subjective), GPT-4o classifier (cheap, evaluator-bias risk), or hybrid (manual for 100, classifier for the rest)?

---

## 11. Suggested next move

If you tell me which of the §10 decisions you want to lock in, I can start with **Step 0 + Step 3 (shared infrastructure modules)** — those don't depend on any of the open questions. The blocker for Step 4 onwards is the §4 golden-dataset decision.

If you want my recommendation: **C (hybrid evaluation: 65-row golden for full RAGAS, full 1,273-question test split for accuracy-only)**, **MiniLM** (pilot already validated it; embedding bake-off is a side-quest you can run in parallel later), **GPT-4o-mini** as RAGAS judge, **k=5** retrieved chunks, **3 hops** for Multi-Hop, **Excel weights** for confidence, **hybrid annotation** (100 manual + GPT-4o classifier for the rest).
