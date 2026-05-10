# Notebook 04c — EXP_03 Sparse RAG · Output Notes

> **Notebooks:** [`notebooks/04c_exp03_sparse_rag.ipynb`](../../notebooks/04c_exp03_sparse_rag.ipynb) (baseline) + [`notebooks/04c_exp03_ragas.ipynb`](../../notebooks/04c_exp03_ragas.ipynb) (judging)
> **Run on:** 2026-05-07 (baseline smoke + golden + test) → 2026-05-10 (RAGAS smoke + full)
> **Phase:** 4 — Group A baseline experiments (third of five)
> **Architecture:** Sparse keyword retrieval + LLM. `SparseRetriever` (BM25 top-5 over chunk text) + `llama-3.3-70b-versatile` answerer · T=0 · k=5 · max_tokens=700.
> **Judge:** `claude-sonnet-4-6` via Anthropic (RAGAS 0.4.3, all 5 metrics, golden_234 surface)
> **Companion:** [`docs/output_notes/04a_exp01_output.md`](04a_exp01_output.md) (No-RAG baseline) · [`docs/output_notes/04b_exp02_output.md`](04b_exp02_output.md) (Naive Dense baseline)

---

## 1. Output

| Surface | Rows | Acuuracy | Wall time | Notes |
|---|---:|---:|---:|---|
| `exp_03_sparse_rag__smoke_50` | 50 | **0.9000** | 2.9 min | smoke validation passed |
| `exp_03_sparse_rag__golden_234` | 234 | **0.8376** | 2.6 min (mostly cache hits) | RAGAS pending |
| **`exp_03_sparse_rag__test_1273`** | **1,273** | **0.7581** | **97 min** | **CANONICAL — Table 1 row 3** |

RAGAS aggregates on `golden_234` (Claude Sonnet 4.6, ran 2026-05-10, 234 rows scored, NaN rate <1 % per metric):

| Metric | Value | Comment |
|---|---:|---|
| `RAGAS_Faithfulness` | **0.0401** | lowest of all 5 architectures (3× lower than Naive's 0.131) |
| `RAGAS_Hallucination_Rate` | 0.9657 | highest (97 %) |
| `RAGAS_Answer_Relevance` | 0.5971 | flat across architectures (~0.59–0.60) |
| `RAGAS_Context_Precision` | **0.0811** | **catastrophic** — 4× lower than Naive's 0.329 |
| `RAGAS_Context_Recall` | 0.1073 | 4× lower than Naive's 0.412 |
| `Answer_Correctness` | 0.8384 | within 0.001 of Naive — judge agrees the *answers* are equivalent |

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

### 2.5 RAGAS reveals catastrophic retrieval quality — yet accuracy is *unaffected*

The single most striking RAGAS finding (added 2026-05-10):

| Metric | EXP_02 Naive Dense | EXP_03 Sparse BM25 | Δ |
|---|---:|---:|---:|
| Context Precision | 0.329 | **0.081** | **−24.8 pp** |
| Context Recall | 0.412 | **0.107** | **−30.5 pp** |
| Faithfulness | 0.131 | **0.040** | −9.1 pp |
| Hallucination Rate | 0.896 | **0.966** | +7.0 pp |
| **`Acuuracy` (test_1273)** | 0.7573 | **0.7581** | **+0.08 pp** (essentially identical) |
| **Answer Correctness** | 0.838 | **0.838** | **+0.00 pp** (identical) |

**The sparse retriever surfaces near-random chunks (CP = 0.081 means 8 % of retrieved chunks are relevant) — and accuracy is unchanged.** This is the strongest single piece of evidence in the entire Phase 4 dataset that **the LLM is answering from pre-training memorisation, not from retrieved context**. If retrieval quality genuinely drove answer quality, accuracy should collapse on EXP_03; instead it ties EXP_02 to within a single question.

**Falsifiable hypothesis verdict** (anchored in [`docs/output_notes/04b_exp02_output.md` §4](04b_exp02_output.md)): *"Sparse Context Precision will improve on questions with rare medical terms. Falsified if sparse Context Precision ≤ 0.33."* → **❌ FALSIFIED.** Sparse CP = 0.081 is *worse* than dense, not better. The mechanism: BM25 retrieval on a 67k-chunk corpus does not concentrate top-5 hits on rare-term-bearing chunks; it surfaces high-frequency keyword matches that are usually irrelevant.

This finding reframes the thesis discussion: **single-shot retrieval — whether dense or sparse — is incapable of meaningfully grounding a memorisation-strong LLM on this benchmark.** It motivates Multi-Hop (EXP_05, see [`output_notes/04e_exp05_output.md`](04e_exp05_output.md)) as the architecture that actually delivers grounded improvement.

### 2.6 Per-stratum RAGAS breakdown

By **USMLE step** (golden_234, n=234):

| meta_info | n | EXP_02 F | EXP_03 F | EXP_02 CP | EXP_03 CP | EXP_02 CR | EXP_03 CR |
|---|---:|---:|---:|---:|---:|---:|---:|
| step1 | 128 | 0.183 | **0.063** | 0.363 | **0.099** | 0.449 | **0.109** |
| step2&3 | 106 | 0.067 | **0.012** | 0.286 | **0.060** | 0.368 | **0.105** |

Sparse Faithfulness collapses on step2&3 (F = 0.012) — clinical-decision questions are where the BM25 keyword bag is least helpful even if a relevant chunk exists.

By **multi-hop subset** (n=13):

| | EXP_02 Naive | EXP_03 Sparse |
|---|---:|---:|
| Faithfulness | 0.000 | 0.013 |
| Context Recall | 0.000 | 0.000 |
| Acuuracy | 0.769 | 0.769 |

Both single-shot retrievers fail multi-hop questions equally. Neither can stitch evidence; both rely on the LLM's memorisation. Only Multi-Hop (EXP_05) breaks this pattern (F = 0.229 on the same subset).

### 2.7 Grounded-correct fraction — sparse is the worst

| Architecture | n correct | grounded (F≥0.5) | % grounded |
|---|---:|---:|---:|
| EXP_02 Naive Dense | 199 | 23 | 11.6 % |
| **EXP_03 Sparse BM25** | **196** | **8** | **4.1 %** |
| EXP_04 Hybrid RRF | 196 | 17 | 8.7 % |
| EXP_05 Multi-Hop | 211 | 60 | 28.4 % |

**Only 4.1 % of EXP_03's correct answers are grounded — the lowest of any architecture.** The remaining 95.9 % are correct because LLaMA memorised the answer, not because the retrieved chunks support it. For the confidence-aware rejection layer (Phase 7), this means EXP_03's Faithfulness is essentially un-thresholdable: median F = 0.000, mean = 0.040.

### 2.8 Golden 234 — a small accuracy gap appears

| Architecture | golden_234 Acuuracy |
|---|---:|
| EXP_01 No-RAG | 0.9017 |
| EXP_02 Naive Dense | 0.8504 |
| **EXP_03 Sparse BM25** | **0.8376** (lowest) |

On the train-skewed golden surface, sparse is 1.3 pp below dense and 6.4 pp below No-RAG. The pattern is consistent with the test-split finding — RAG hurts vs No-RAG on questions LLaMA already memorised. **RAGAS scoring (now complete, see §2.5) confirms the *why*: sparse Context Precision is 0.081 — far below the 0.33 threshold and far below dense's 0.329. The falsifiable hypothesis is FALSIFIED.**

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

## 4. Conclusions

1. **EXP_03 is COMPLETE.** Three baseline surfaces + RAGAS aggregates written; both notebooks ran end-to-end; RAGAS judge cleared with <1 % NaN per metric. RAGAS cost: ~$11 Sonnet 4.6.

2. **Headline numbers for Excel Table 1 row 3** (canonical = `test_1273`, RAGAS from `golden_234`):

   | Cell | Value | Source |
   |---|---|---|
   | `Acuuracy` | **0.7581** | `test_1273` |
   | `Exact Match` | **0.7581** | same |
   | `Generator Model` | `llama-3.3-70b-versatile` | locked |
   | `mean_latency_s` | 0.435 | `test_1273` |
   | `RAGAS_Faithfulness` | **0.0401** | `golden_234` |
   | `RAGAS_Hallucination_Rate` | **0.9657** | derived |
   | `RAGAS_Answer_Relevance` | 0.5971 | `golden_234` |
   | `RAGAS_Context_Precision` | **0.0811** | `golden_234` |
   | `RAGAS_Context_Recall` | **0.1073** | `golden_234` |
   | `Answer_Correctness` | 0.8384 | `golden_234` |

3. **Falsifiable hypothesis verdict** (anchored in [`output_notes/04b_exp02_output.md` §4](04b_exp02_output.md)):
   - ✗ **FALSIFIED**: Sparse Context Precision > 0.33 — got **0.081**. BM25 retrieval on the chunked corpus does not surface relevant chunks at all; precision is *worse* than dense, not better.

4. **The strongest single piece of evidence in Phase 4 for the memorisation thesis** (added 2026-05-10): EXP_03's CP = 0.081 (near-random retrieval) yet Acuuracy = 0.7581 (within 0.08 pp of EXP_02 Naive's 0.7573). **If retrieval quality drove answer quality, accuracy should collapse on EXP_03; instead it ties EXP_02 to within a single question.** This decouples retrieval quality from accuracy on a memorisation-strong LLM and motivates the thesis-central novelty: confidence-aware rejection (Phase 7) needs a retrieval-quality signal that the LLM's accuracy alone cannot provide.

5. **The contamination-defining narrative continues**: like EXP_02, Sparse RAG underperforms No-RAG on test_1273. Together with EXP_04 Hybrid (also under-performs), **single-shot retrieval — whether dense, sparse, or RRF-fused — does not help a memorisation-strong LLM on this benchmark.** Only EXP_05 Multi-Hop breaks the pattern (see [`output_notes/04e_exp05_output.md`](04e_exp05_output.md)).

6. **Cross-reference**: EXP_03's "complementarity with dense" finding (153 questions disagree with 50/50 right/wrong split) was the original anchor for the EXP_04 Hybrid hypothesis. The EXP_04 result (see [`output_notes/04d_exp04_output.md`](04d_exp04_output.md)) shows the complementarity does not survive RRF fusion — sparse's near-random CP pollutes the fused top-5. The pairwise-disagreement signal turns out to be a measurement artefact of two retrievers each producing mostly-noise output.

---

## 5. Next steps

1. **Phase 4 is COMPLETE.** All 5 architectures' baselines + RAGAS done. See [`output_notes/04d_exp04_output.md`](04d_exp04_output.md) (Hybrid) and [`output_notes/04e_exp05_output.md`](04e_exp05_output.md) (Multi-Hop, the headline finding).

2. **Methodology paragraph for the thesis writeup** — anchor the EXP_03 finding here so it's not improvised later: *"EXP_03 (Sparse RAG with BM25 top-5 over the 67k-chunk Harrison's-dominant corpus) achieved Acuuracy 0.7581 on the contamination-clean test split — essentially identical to EXP_02 Naive Dense (0.7573). RAGAS judging revealed a striking decoupling: sparse Context Precision was 0.081 (vs 0.329 for dense) and Context Recall was 0.107 (vs 0.412), yet Acuuracy was unchanged. **The LLM produced correct MCQ options at the same rate when retrieval quality dropped four-fold — direct empirical evidence that on a memorisation-strong benchmark, single-shot retrieval does not contribute meaningfully to accuracy.** The hypothesis that BM25 keyword matching would recover rare-medical-term cases (where embedding similarity might miss literal drug names) was falsified: sparse CP is **lower** than dense CP, not higher. This finding motivates the iterative multi-hop architecture (EXP_05) as the only retrieval strategy in this comparison that actually grounds the LLM's answers."*

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
│   ├── ragas_scores.csv    ← 234 rows × 5 metrics (NaN <1 % per col)
│   └── summary.json        ← all 5 RAGAS aggregates filled (2026-05-10)
├── exp_03_sparse_rag__test_1273/
│   ├── predictions.jsonl   ← 1,273 rows · canonical headline run
│   ├── retrieval.jsonl     ← 1,273 rows × 5 chunk_ids
│   └── summary.json        ← Acuuracy = 0.7581
└── exp_03_sparse_rag__golden_234_ragas_smoke/
    ├── ragas_scores.csv    ← Stage A pilot (10 rows)
    └── summary.json
```

The `golden_234/summary.json` was rewritten 2026-05-10 to fill the five RAGAS columns; `ragas_scores.csv` (per-row metrics, 234 rows × 5) is also at `golden_234/`.
