# Notebook 04c — EXP_03 Sparse RAG · Output Notes (baseline-only, RAGAS pending)

> **Notebook (baseline):** [`notebooks/04c_exp03_sparse_rag.ipynb`](../../notebooks/04c_exp03_sparse_rag.ipynb)
> **Notebook (RAGAS):** [`notebooks/04c_exp03_ragas.ipynb`](../../notebooks/04c_exp03_ragas.ipynb) — built but **not yet run**; RAGAS evaluation deferred until all 4 RAG baselines (EXP_02, 03, 04, 05) complete, then judge them as a batch
> **Run on:** 2026-05-07 (baseline smoke + golden + test_1273)
> **Phase:** 4 — Group A baseline experiments (third of five)
> **Architecture:** Sparse keyword retrieval + LLM. `SparseRetriever` (BM25 top-5 over chunk text) + `llama-3.3-70b-versatile` answerer · T=0 · k=5 · max_tokens=700.
> **Companion:** [`docs/output_notes/04a_exp01_output.md`](04a_exp01_output.md) (No-RAG baseline) · [`docs/output_notes/04b_exp02_output.md`](04b_exp02_output.md) (Naive Dense baseline)

---

## 1. Output

| Surface | Rows | Acuuracy | Wall time | Notes |
|---|---:|---:|---:|---|
| `exp_03_sparse_rag__smoke_50` | 50 | **0.9000** | 2.9 min | smoke validation passed |
| `exp_03_sparse_rag__golden_234` | 234 | **0.8376** | 2.6 min (mostly cache hits) | RAGAS pending |
| **`exp_03_sparse_rag__test_1273`** | **1,273** | **0.7581** | **97 min** | **CANONICAL — Table 1 row 3** |

RAGAS metrics for golden_234 are `null` in `summary.json` until the RAGAS notebook runs.

---

## 2. Headline finding — *Sparse and Dense are essentially tied on accuracy, but they disagree on which questions they get right*

### 2.1 Test split (n=1,273) — three architectures side-by-side

| Architecture | Acuuracy | Δ vs No-RAG |
|---|---:|---:|
| EXP_01 No-RAG | **0.7738** | — |
| EXP_02 Naive Dense (BGE-large) | 0.7573 | −1.65 pp |
| **EXP_03 Sparse BM25** | **0.7581** | **−1.57 pp** |

Sparse and dense end up **within 0.1 pp of each other** on the headline accuracy. Both lose to No-RAG by ~1.6 pp. This is the second confirmation (after EXP_02) that **naive retrieval with k=5 chunks does not help on a contamination-clean test split** for this LLaMA / corpus / chunking configuration.

### 2.2 The complementarity finding — they disagree on 153 questions, 50/50 right/wrong

Paired per-question comparison on test_1273:

| | Both right | Both wrong | E2 right, E3 wrong | E2 wrong, E3 right | Total disagree |
|---|---:|---:|---:|---:|---:|
| EXP_02 vs EXP_03 | 888 | 232 | 76 | 77 | **153 (12.0 %)** |

**Of the 153 questions where dense and sparse disagree, dense is right on 76 and sparse is right on 77.** They're not redundant — they catch *complementary* sets of questions. This is the first piece of empirical evidence that **EXP_04 Hybrid (RRF fusion of dense + sparse) should help**: the fusion can pick up the right answer in either lane, and there's no dominant lane.

If Hybrid simply *picked the dense answer* for every question, it would land at EXP_02's number (0.7573). If it picked the sparse answer for every question, it would land at EXP_03's (0.7581). If RRF correctly recovers each retriever's strengths, it should land somewhere above both — potentially close to or above No-RAG's 0.7738 if fusion + better retrieval quality (Context Precision) lifts the LLM's grounding.

### 2.3 Per-USMLE-step pattern — sparse wins step2&3, dense wins step1

| meta_info | n | EXP_01 No-RAG | EXP_02 Naive Dense | EXP_03 Sparse BM25 |
|---|---:|---:|---:|---:|
| step1 (basic science) | 679 | 0.7585 | **0.7585** (tied with No-RAG) | 0.7452 (worst) |
| step2&3 (clinical decision) | 594 | **0.7912** (best) | 0.7559 | **0.7727** |

**Sparse retrieval helps step2&3 (+1.68 pp vs dense) and hurts step1 (−1.33 pp vs dense)**. Plausible mechanism: step2&3 questions describe a clinical scenario in vocabulary that matches textbook chapters literally (drug names, eponyms, anatomy) — exactly where BM25's keyword matching wins. Step1 questions are more about underlying mechanisms, where dense semantic embeddings find the right concept better than literal matches.

This is a methodologically interesting finding for the thesis discussion: *"Dense and sparse retrievers have orthogonal strengths along the USMLE step axis. The Hybrid architecture (EXP_04) should test whether RRF can capture both."*

### 2.4 Regression analysis vs No-RAG (test_1273)

| Architecture | Both right with No-RAG | Both wrong | No-RAG right, RAG wrong (regression) | No-RAG wrong, RAG right (fix) | Net |
|---|---:|---:|---:|---:|---:|
| EXP_02 Naive Dense | 900 | 224 | 85 | 64 | **−21** |
| EXP_03 Sparse BM25 | 923 | 246 | 62 | 42 | **−20** |

Both retrievers introduce more new errors than they fix vs. No-RAG. **Sparse is slightly more conservative** (62 regressions vs dense's 85, but also fewer fixes — 42 vs 64). The total impact on accuracy is essentially identical.

### 2.5 Golden 234 — a small accuracy gap appears

| Architecture | golden_234 Acuuracy |
|---|---:|
| EXP_01 No-RAG | 0.9017 |
| EXP_02 Naive Dense | 0.8504 |
| **EXP_03 Sparse BM25** | **0.8376** (lowest) |

On the train-skewed golden surface, sparse is 1.3 pp below dense and 6.4 pp below No-RAG. The pattern is consistent with the test-split finding — RAG hurts vs No-RAG on questions LLaMA already memorised. RAGAS scoring (when run) will tell us *why*: is sparse Context Precision higher than dense's 0.33 (the falsifiable hypothesis from EXP_02), and does that translate to higher Faithfulness?

---

## 3. Operational notes

### 3.1 Wall time — BM25 is ~40× slower per query than ChromaDB HNSW

| Surface | EXP_02 Naive (Chroma) | EXP_03 Sparse (BM25) |
|---|---:|---:|
| smoke_50 | 30 s | 175 s |
| golden_234 | 132 s | 158 s (44 fresh + cached rest) |
| **test_1273** | **699 s (~12 min)** | **5827 s (~97 min)** |

Per-question cost breakdown on test_1273:
- EXP_02 ChromaDB lookup: ~0.1 s/query
- EXP_03 BM25 `get_scores`: ~4.0 s/query (iterates over all 67,599 chunks in pure Python)

This is a known characteristic of `rank-bm25` — it doesn't have an inverted index, so every query is O(N_chunks). For a thesis-scale eval (1,273 questions × few minutes overnight) it's fine. For production it would warrant a switch to a sparse vector store like Elasticsearch or Pyserini. Cost is still $0 (Groq), so this is operationally tolerable.

### 3.2 Parse health

0 null `pred_letter` values across 1,557 generation calls (smoke + golden + test). Mean `latency_s` = 0.435 s — same as EXP_02 (Groq performance is stable). All `retrieval.jsonl` rows have 5 chunk IDs with non-zero BM25 scores.

---

## 4. Conclusions (baseline only)

1. **EXP_03 baseline is COMPLETE.** Three surfaces written; RAGAS judging pending until all 4 RAG baselines run, then evaluated as a batch.

2. **Headline numbers for Excel Table 1 row 3** (canonical = `test_1273`):

   | Cell | Value | Source |
   |---|---|---|
   | `Acuuracy` | **0.7581** | `test_1273` |
   | `Exact Match` | **0.7581** | same |
   | `Generator Model` | `llama-3.3-70b-versatile` | locked |
   | `mean_latency_s` | 0.435 | `test_1273` |
   | `RAGAS_*` columns | **pending** | will fire when 04c_exp03_ragas notebook runs |

3. **Two empirical findings to anchor the EXP_04 Hybrid hypothesis**:
   - **Complementarity**: 153 of 1,273 test questions get different answers from dense vs sparse, with a 50/50 right/wrong split. The two retrievers catch orthogonal subsets — fusion should help.
   - **USMLE-step pattern**: sparse wins step2&3 (+1.68 pp), dense wins step1 (+1.33 pp). RRF should recover the right retriever per question type.

4. **Sharpened falsifiable hypothesis for EXP_04 Hybrid**: *Hybrid Acuuracy on test_1273 should land **above 0.76** (i.e., above both EXP_02's 0.7573 and EXP_03's 0.7581). If RRF works as advertised, it should also lift Context Precision above EXP_02's 0.33. Falsified if Hybrid Acuuracy ≤ 0.76 (i.e., RRF doesn't capture the complementarity).*

5. **The contamination-defining narrative continues**: like EXP_02, Sparse RAG underperforms No-RAG on test_1273. Together, EXP_02 + EXP_03 establish that **single-strategy retrieval (k=5) does not help LLaMA on this benchmark** when the model already has the answer from pretraining. The thesis discussion can now anchor the central claim: *"Naive retrieval — whether dense or sparse — provides retrieval-distractor noise that exceeds any retrieval-grounding signal on a memorisation-strong baseline. Improvements need to come from retrieval-quality fusion (Hybrid, EXP_04) and iterative refinement (Multi-Hop, EXP_05)."*

6. **What we still don't know** (RAGAS pending):
   - Is sparse Context Precision > 0.33 (the EXP_02 anchor)?
   - Does sparse Faithfulness exceed dense's 0.131?
   - Does the per-USMLE-step pattern hold in the RAGAS metrics?

   These will be answered when `04c_exp03_ragas.ipynb` runs (after EXP_04 + EXP_05 baselines are also done — per the user's batch-baselines-then-batch-RAGAS strategy).

---

## 5. Next steps

1. **Build EXP_04 Hybrid baseline + RAGAS notebooks** — refactor `src/retrieval/hybrid.py` under the `Retriever` ABC, then duplicate the EXP_03 notebook pair as `04d_*` with `HybridRetriever` swapped in. The existing `hybrid_top_k` function (used by Notebook 04 golden construction) stays intact for backwards compatibility.
2. **Build EXP_05 Multi-Hop baseline + RAGAS notebooks** — new `src/retrieval/multi_hop.py` (3-hop iterative dense retrieval with sub-query generation via Groq, dedup across hops, early stop on no-progress). New prompt template in `src/generation/prompts.py` for the sub-query generation. Notebook pair `04e_*`.
3. **User runs all 4 baselines** (EXP_02 + EXP_03 are done; EXP_04 + EXP_05 are next) — total wall time ~3 h, $0.
4. **User runs all 4 RAGAS evaluations** — total cost ~$45–48, total wall time ~1 h.
5. **Final thesis-cross-architecture analysis** — write up the headline Table 1 with all 4 RAG architectures filled, plus the per-stratum patterns this notebook surfaced.

---

## 6. Files produced

```
results/
├── exp_03_sparse_rag__smoke_50/
│   ├── predictions.jsonl   ← 50 rows
│   ├── retrieval.jsonl     ← 50 rows × 5 chunk_ids + raw BM25 scores
│   └── summary.json
├── exp_03_sparse_rag__golden_234/
│   ├── predictions.jsonl   ← 234 rows
│   ├── retrieval.jsonl     ← 234 rows × 5 chunk_ids
│   └── summary.json        ← RAGAS columns pending
└── exp_03_sparse_rag__test_1273/
    ├── predictions.jsonl   ← 1,273 rows · canonical headline run
    ├── retrieval.jsonl     ← 1,273 rows × 5 chunk_ids
    └── summary.json        ← Acuuracy = 0.7581
```
