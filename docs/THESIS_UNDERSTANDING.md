# Thesis Understanding — Working Reference

> **Title:** Systematic Comparison of Multiple Retrieval-Augmented Generative AI Architectures for Evidence-Based Medical Question Answering with Explainability and Hallucination Control
> **Author:** Raja Kalavala (PN1196988) — MSc AI & ML, LJMU
> **Supervisor:** Dr. K Lokeshwaran
> **Submission:** March 2026

---

## 1. The One-Line Pitch

Build four RAG architectures (Naive, Sparse, Hybrid, Multi-Hop Explainable), run them on the **same** medical dataset with the **same** LLM, embedding model, prompt template and retrieval infrastructure — and measure which retrieval strategy produces the **most accurate, least hallucinated, and most explainable** answers for clinical question answering.

The retrieval strategy is the **only** independent variable. Everything else is held constant.

---

## 2. The Problem Being Solved

LLMs are now used clinically (physician adoption: 38% in 2023 → 66% in 2024), but they still confidently fabricate medical information — a recent oncology study found ~2-in-5 AI medical responses contained incorrect or unsupported statements. In medicine, that risks misdiagnosis, wrong treatment, patient harm.

RAG was introduced to ground answers in real evidence, but **different RAG designs retrieve evidence differently**. The literature gap:

- Most studies evaluate **one** RAG setup in isolation.
- Comparisons that exist often use general-domain (finance, e-commerce) datasets.
- Hallucination behavior across architectures is rarely measured systematically.
- Explainability — tracing answers back to specific evidence passages — is mostly absent from RAG evaluations.

**Nobody has done a controlled side-by-side comparison of multiple RAG architectures on the same medical benchmark covering accuracy + hallucination + explainability together.** This thesis fills that gap.

---

## 3. Research Questions

1. What is the relative impact of retrieval architecture (Naive / Sparse / Hybrid / Multi-Hop) on factual accuracy and hallucination rate when LLM and data are held constant?
2. Which architecture scores highest on RAGAS Faithfulness and Context Relevance across the 12,723 MedQA USMLE questions?
3. To what extent can LIME and SHAP attribute generated medical answers back to specific retrieved evidence passages across the four architectures?
4. Does multi-hop retrieval produce statistically lower hallucination rates than single-pass retrieval?

---

## 4. Aim & Objectives

**Aim:** Compare the four RAG architectures under controlled conditions to determine which delivers the most accurate, least hallucinated, and most explainable responses on MedQA USMLE.

**Objectives:**
1. Implement & validate all four architectures on a shared LLM/embedding/retrieval stack.
2. Compare factual accuracy across all 12,723 USMLE-style questions.
3. Measure hallucination behavior using five RAGAS metrics.
4. Apply LIME and SHAP at passage level for explainability tracing.
5. Produce a comparison framework mapping clinical requirements (accuracy / explainability / hallucination tolerance) to the most suitable RAG architecture.

---

## 5. The Four RAG Architectures

| # | Architecture | Retrieval Strategy | Notes |
|---|---|---|---|
| 1 | **Naive RAG** | Single-step dense semantic retrieval (FAISS top-k) | Baseline. Question → embed → vector search → top-k → prompt → LLM. |
| 2 | **Sparse RAG** | BM25 keyword retrieval only | Tokenize question → BM25 score against textbook chunks → top-k → prompt → LLM. |
| 3 | **Hybrid RAG** | Dense (FAISS) + Sparse (BM25) fused with **Reciprocal Rank Fusion** (k=60) | Two parallel retrieval paths merged by RRF, then LLM. |
| 4 | **Multi-Hop Explainable RAG** | Iterative, up to **3 hops**: question decomposition → initial retrieval → reasoning over gaps → refined query → targeted retrieval | Hop 1: initial. Hop 2: identify gaps, refine. Hop 3: fill remaining gaps. Terminates when sub-questions all supported or hop budget exhausted. LIME/SHAP applied post-hoc. |

### Reciprocal Rank Fusion formula (Hybrid)

```
RRF(d) = Σ_{r ∈ R} 1 / (k + rank_r(d))     with k = 60
```

### BM25 formula (Sparse)

```
BM25(q,d) = Σ_{t ∈ q} IDF(t) · [ f(t,d)·(k1+1) / ( f(t,d) + k1·(1 - b + b·|d|/avgdl) ) ]
```

---

## 6. Shared Stack (held constant across all 4 architectures)

| Component | Choice | Notes |
|---|---|---|
| **LLM** | LLaMA 3.3 70B Versatile via **Groq API** | 131k context window. Same model for all 4 architectures. |
| **Embedding model** | **BGE-large-en-v1.5** (primary candidate, 1024-dim, 335M params) | Alternatives: MedCPT (NCBI, 768-dim, SOTA on 3/5 BEIR), all-MiniLM-L6-v2 (384-dim, 22M params). Final pick after MedQA validation. |
| **Vector DB** | **FAISS** | In-memory, free, GPU-capable, 0.34ms query latency. Used in original RAG paper. |
| **Sparse index** | rank-bm25 | BM25 keyword index built over the same chunks. |
| **Prompt template** | Same evidence-grounded template across all four | Final prompting technique chosen during validation (zero-shot / few-shot / CoT / self-consistency / tree-of-thought / instruction-with-evidence-grounding). |
| **Chunking** | Recursive (primary candidate) | Alternatives evaluated: fixed-size, overlapping (10–20%), semantic. |

---

## 7. Dataset

### MedQA USMLE
- 12,723 English multiple-choice questions (USMLE-style).
- Source: Jin et al. (2020) — `arxiv.org/abs/2009.13081`.
- License: CC-BY-4.0, no PHI.
- Schema:

| Field | Description | Example | Type |
|---|---|---|---|
| `id` | Unique ID | `usmle_001` | string |
| `question` | Clinical scenario | `"A 45-year-old male presents with chest pain..."` | string |
| `opa` / `opb` / `opc` / `opd` | Options A–D | `"Myocardial infarction"` | string |
| `correct` | Index of correct answer (A=0, B=1, C=2, D=3) | `1` | integer |
| `subject` | Medical subject | `"Cardiology"` | string |

### Knowledge Base — 18 English Medical Textbooks
Used as the retrieval corpus across all four architectures. Covers anatomy, pathology, pharmacology and other clinical subjects. **Data leakage prevention:** test questions and their exact contexts must not appear in the retrieval corpus.

### Preprocessing pipeline (Figure 7.1)

```
MedQA JSONL ──► parse ──► dedupe ──► validate options ──► standardize encoding ──┐
                                                                                  ├─► chunking ──► embedding ──┬─► FAISS index
Textbook corpus ──► extract raw text ──► clean ──► normalize terminology ────────┘                            └─► BM25 index
```

---

## 8. Evaluation Framework

### RAGAS metrics (five core)

| Metric | What it measures | Formula sketch |
|---|---|---|
| **Faithfulness** | Are generated claims supported by the retrieved context? | `(1/M) · Σ 𝟙[claim_i supported by context]` |
| **Answer Correctness** | Overall correctness vs ground truth | `0.75 · F1(TP,FP,FN) + 0.25 · cos(a, g)` |
| **Context Precision** | Are relevant chunks ranked above irrelevant ones? | `(1/K) · Σ Precision@k · Rel_k` |
| **Context Recall** | Does retrieved context contain all needed info? | `(1/G) · Σ 𝟙[gt_claim_j ∈ context]` |
| **Answer Relevancy** | Does the answer actually address the question? | `(1/N) · Σ cos(q_i, q)` |

### Non-LLM metrics (objective deterministic cross-checks)

1. **Exact Match Accuracy** — string compare predicted option (A/B/C/D) vs ground truth, % across 12,723 questions per architecture.
2. **Retrieval Recall** — % of questions where ≥1 of the top-k chunks contains info relevant to the correct answer. Distinguishes retrieval failure from generation failure.

---

## 9. Hallucination Control — Three-Layer Strategy

| Layer | Strategy | How it works |
|---|---|---|
| **L1** | Evidence-grounded prompt design | Prompt forces model to answer ONLY from retrieved evidence; flag insufficient evidence rather than guess. |
| **L2a** | Retrieval optimization | The 4-way comparison itself reveals which retrieval is most complete/relevant, reducing info gaps. |
| **L2b** | Multi-hop validation | Each retrieval step validated before next hop — catches unsupported claims early in reasoning chain. |
| **L3** | Answer rejection | If RAGAS Faithfulness < **0.5**, flag answer as potentially unreliable rather than presenting as confident. |

### Answer acceptance flow (Figure 7.6)
```
generated answer + retrieved context
        │
        ▼
RAGAS Faithfulness score
        │
   ┌────┴────┐
   ▼         ▼
score ≥0.5   score <0.5
   │             │
accept &     flag as
LIME/SHAP    unreliable,
analysis     log for review
```

---

## 10. Explainability — LIME & SHAP at passage level

| Feature | LIME | SHAP |
|---|---|---|
| Approach | Simplified local model explaining a single prediction | Shapley value computation across all feature combinations |
| Scope | Local explanations per query | Local **and** global explanations |
| Computation cost | Moderate | Higher |
| Application | Adapted to **passage level** — perturb retrieved passages, observe answer change. Output: ranked list of passages by influence. | Each retrieved chunk treated as a feature; Shapley value = its marginal contribution. |

The clinical use case: a doctor can see *which textbook passage(s)* the system relied on to form its answer.

---

## 11. End-to-End Experimental Workflow (Figure 7.7)

```
 ┌─────────────┐   ┌──────────────────┐   ┌───────────┐   ┌────────────┐   ┌─────────────────┐
 │ MedQA       │──►│ Data Cleaning    │──►│ Chunking  │──►│ Embedding  │──►│ Vector Store    │
 │ (12,723 Q)  │   │ & Preprocessing  │   │           │   │ (BGE-large)│   │ (FAISS + BM25)  │
 │ + Textbooks │   │                  │   │           │   │            │   │                 │
 └─────────────┘   └──────────────────┘   └───────────┘   └────────────┘   └────────┬────────┘
                                                                                     │
                          ┌──────────────────────────────────────────────────────────┤
                          │                          │                  │            │
                          ▼                          ▼                  ▼            ▼
                    ┌───────────┐            ┌────────────┐     ┌────────────┐  ┌───────────┐
                    │ Naive RAG │            │ Sparse RAG │     │ Hybrid RAG │  │ Multi-Hop │
                    └─────┬─────┘            └──────┬─────┘     └──────┬─────┘  └──────┬────┘
                          └─────────────┬───────────┴──────────────────┴───────────────┘
                                        ▼
                            ┌─────────────────────────────┐
                            │ LLaMA 3.3 70B via Groq API  │
                            └────────────────┬────────────┘
                                             ▼
                              50,892 generated answers
                              (12,723 × 4 architectures)
                                             │
              ┌──────────────────────────────┼──────────────────────────────┐
              ▼                              ▼                              ▼
     ┌────────────────┐            ┌──────────────────┐         ┌────────────────────┐
     │ RAGAS metrics  │            │ Explainability   │         │ Hallucination      │
     │                │            │ (LIME + SHAP)    │         │ Detection          │
     └────────┬───────┘            └─────────┬────────┘         └─────────┬──────────┘
              └──────────────────────────────┼──────────────────────────────┘
                                             ▼
                            ┌──────────────────────────────────┐
                            │ Comparative analysis +            │
                            │ statistical significance testing  │
                            └────────────────┬─────────────────┘
                                             ▼
                            ┌──────────────────────────────────┐
                            │ Optimal RAG framework for         │
                            │ Medical QA — deployment guidance  │
                            └──────────────────────────────────┘
```

**Total generations:** 12,723 questions × 4 architectures = **50,892 answers** to evaluate.

---

## 12. Scope

### In-scope
- Implementation & evaluation of the four RAG architectures.
- MedQA USMLE (English, 12,723 MCQs) as the benchmark.
- FAISS for dense, BM25 for sparse retrieval.
- Five RAGAS metrics + LIME/SHAP explainability + 3-layer hallucination control.
- Statistical significance testing across architectures.

### Out-of-scope
- Fine-tuning / retraining LLMs on medical data.
- Non-English datasets, clinical notes, EHR integration.
- Knowledge graphs, graph-based retrieval, agentic RAG.
- Real-time clinical deployment, patient-facing testing.
- New explainability methods beyond LIME/SHAP.
- Human expert / clinical panel evaluation.
- Comparison with non-RAG baselines (standalone LLMs, prompt-only, fine-tuned).

---

## 13. Resources

### Hardware
- MacBook Pro, Apple M1 Pro, 16 GB unified memory, 1 TB SSD.
- No local GPU needed (LLM inference via Groq Cloud API).

### Software stack
| Category | Tools | Purpose |
|---|---|---|
| Dev environment | Google Colab | Interactive runs, environment management |
| Lang & data | Python 3.13, Pandas 2.2.4, NumPy 2.3.2 | Preprocessing, chunk handling, results analysis |
| LLM & prompts | LangChain, Sentence-Transformers | Prompt flows, retrieval chains, dense embeddings |
| Retrieval | FAISS, rank-bm25 | Dense + sparse retrieval |
| Inference | LLaMA 3.3 70B via Groq | Answer generation across all 4 RAGs |
| Eval & XAI | RAGAS, LIME, SHAP | Metrics + explainability |
| Viz | Matplotlib 3.10.5, Seaborn 0.13.2 | Plots & comparison charts |
| VC | Git, GitHub | Repo management |

---

## 14. Risks & Mitigations

| Risk | P | Mitigation |
|---|---|---|
| Groq API downtime / rate limits | Med | Fall back to Together AI / HF Inference with Qwen 2.5 72B |
| Poor vector DB retrieval quality | Med | Test multiple chunking strategies, tune size & overlap during validation |
| LLaMA 3.3 hallucinations | High | Evidence-grounded prompts + 3-layer control + Faithfulness threshold 0.5 |
| Dataset contamination (pretraining overlap) | Med | Compare LLM with vs without retrieval to detect memorization |
| Hardware failure / data loss | Low | Backups on Google Drive + GitHub, version-controlled outputs |
| RAGAS inconsistency | Low | Cross-check with non-LLM metrics |

---

## 15. Limitations Acknowledged Upfront

- MedQA covers USMLE-style only — may not generalize to clinical notes, patient conversations, real-time diagnostics.
- Results are specific to LLaMA 3.3 70B; different LLMs may produce different rankings.
- Knowledge base = 18 textbooks; excludes clinical guidelines, case reports, recent literature.

---

## 16. Timeline (March → June 2026, 15 weeks)

Key phases from the Gantt chart:
- W1: Proposal submission, lit review (✓ done at W1–W2)
- W1–W3: Environment setup + data cleaning/preprocessing
- W3–W7: Implement Naive → Sparse → Hybrid → Multi-Hop (sequentially, ~2 weeks each)
- W7–W8: RAGAS evaluation across all architectures
- W8–W9: LIME/SHAP explainability + error analysis & hallucination mitigation
- W9: Results write-up v1
- W5–W7: Interim report (parallel)
- W9: Interim report review, then rework
- W10–W12: Results v2 + final thesis draft
- W13: Video presentation
- W14: **Final thesis submission**

---

## 17. What This Means for the Code

When we start building, the implementation will need:

1. **Shared infrastructure** — one set of preprocessing, chunking, embedding, FAISS index, BM25 index, LLM client, prompt template, RAGAS evaluator, LIME/SHAP analyzers — used by all four architectures.
2. **Four retriever implementations** — each conforming to a common interface (`question → list[passages]`), so the rest of the pipeline is identical.
3. **One evaluation harness** — runs all four architectures over 12,723 questions, captures answers + retrieved passages + RAGAS scores + explainability outputs into a structured results store.
4. **Statistical comparison** — significance tests across architectures, plus the deployment-guidance framework (which architecture wins under which clinical priority: accuracy / explainability / hallucination tolerance).

The cleanest starting point is the shared infrastructure (data → chunks → embeddings → indices), then Naive RAG end-to-end as the baseline, then layer in the other three architectures one at a time.
