# Notebook 04d — EXP_04 Hybrid RAG · Output Notes

> **Notebooks:** [`notebooks/04d_exp04_hybrid_rag.ipynb`](../../notebooks/04d_exp04_hybrid_rag.ipynb) (baseline) + [`notebooks/04d_exp04_ragas.ipynb`](../../notebooks/04d_exp04_ragas.ipynb) (judging)
> **Run on:** 2026-05-07 (baseline smoke + golden + test) → 2026-05-10 (RAGAS smoke + full)
> **Phase:** 4 — Group A baseline experiments (fourth of five)
> **Architecture:** Reciprocal Rank Fusion of dense (BGE-large + ChromaDB) and sparse (BM25) top-5 retrievers. `HybridRetriever` (RRF k=60) + `llama-3.3-70b-versatile` answerer · T=0 · k=5 fused chunks · max_tokens=700.
> **Judge:** `claude-sonnet-4-6` via Anthropic (RAGAS 0.4.3, all 5 metrics, golden_234 surface)
> **Companion:** [`docs/output_notes/04b_exp02_output.md`](04b_exp02_output.md) (Naive Dense baseline) · [`04c_exp03_output.md`](04c_exp03_output.md) (Sparse BM25 baseline)

---

## 1. Output

| Surface | Rows | Acuuracy | Wall time | Notes |
|---|---:|---:|---:|---|
| `exp_04_hybrid_rag__smoke_50` | 50 | 0.9000 | ~2 min | smoke validation passed |
| `exp_04_hybrid_rag__golden_234` | 234 | **0.8376** | 18 min | RAGAS surface |
| **`exp_04_hybrid_rag__test_1273`** | **1,273** | **0.7659** | **123 min** | **CANONICAL — Table 1 row 4** |

RAGAS aggregates on `golden_234` (Claude Sonnet 4.6, 234 rows scored, NaN rate <2 % per metric):

| Metric | Value |
|---|---:|
| `RAGAS_Faithfulness` | 0.0944 |
| `RAGAS_Hallucination_Rate` | 0.9174 |
| `RAGAS_Answer_Relevance` | 0.5966 |
| `RAGAS_Context_Precision` | 0.2797 |
| `RAGAS_Context_Recall` | 0.3483 |
| `Answer_Correctness` | 0.8273 |

---

## 2. Headline finding — *Hybrid clears the accuracy hypothesis but FALSIFIES the Context Precision hypothesis*

| Hypothesis | Threshold | Got | Verdict |
|---|---|---:|:---:|
| Hybrid Acuuracy on test_1273 > 0.76 (above both EXP_02 and EXP_03) | > 0.76 | **0.7659** | ✓ **SUPPORTED** (just clears) |
| Hybrid Context Precision ≥ 0.50 (above EXP_02's 0.33) | ≥ 0.50 | **0.2797** | ✗ **FALSIFIED** (worse than Naive's 0.329) |

**The two findings together are intelligible — and counter-intuitive.** Hybrid RRF marginally improves accuracy by surfacing complementary chunks the dense or sparse retriever each missed alone. But the *retrieval quality* (CP) is **worse than dense alone**, not better. Sparse BM25's catastrophic CP (0.081) pollutes the fused top-5; the Reciprocal Rank Fusion includes sparse-ranked chunks that dense correctly de-prioritised, lowering precision of the union vs dense alone.

**This is a publishable counter-result for the discussion chapter.** The proposal hypothesised that combining dense + sparse via RRF would lift retrieval quality. The empirical finding is the opposite: **RRF with a weak partner retriever degrades precision relative to the strong partner alone, even when accuracy nudges up.**

---

## 3. Meaning of the outputs

### 3.1 Cross-architecture headline (golden_234 + test_1273)

| Architecture | acc_test | acc_golden | F | Hall | CP | CR | AC |
|---|---:|---:|---:|---:|---:|---:|---:|
| EXP_01 No-RAG | **0.7738** | 0.9017 | n/a | n/a | n/a | n/a | 0.8738 |
| EXP_02 Naive Dense | 0.7573 | 0.8504 | 0.131 | 0.896 | **0.329** | 0.412 | 0.838 |
| EXP_03 Sparse BM25 | 0.7581 | 0.8376 | 0.040 | 0.966 | 0.081 | 0.107 | 0.838 |
| **EXP_04 Hybrid RRF** | **0.7659** | 0.8376 | 0.094 | 0.917 | 0.280 | 0.348 | 0.827 |
| EXP_05 Multi-Hop | 0.7958 | 0.9017 | 0.283 | 0.737 | 0.374 | 0.711 | 0.869 |

**Hybrid sits between Naive and Multi-Hop on accuracy, but underneath both Naive and Multi-Hop on Context Precision and Faithfulness.** The accuracy lift is real but small (+0.86 pp vs Naive); the retrieval-quality regression is real and meaningful (−4.9 pp CP vs Naive).

### 3.2 RRF mechanism — why fusion *lowers* Context Precision

RRF combines two ranked lists by reciprocal rank: `score(chunk) = Σ 1/(k + rank_i)` over retrievers. With k=60 and top-5 from each:
- A chunk that ranks #1 dense and #1 sparse contributes 2/61 ≈ 0.033.
- A chunk that ranks #1 dense only contributes 1/61 ≈ 0.0164.
- A chunk that ranks #5 sparse only contributes 1/65 ≈ 0.0154.

**Critical**: a dense-only top-1 chunk (high quality) and a sparse-only top-5 chunk (likely noise — sparse CP is 0.081) score within 6 % of each other. With sparse retrieval being near-random on the chunked corpus, RRF promotes sparse-noise into the fused top-5 — chunks that dense correctly placed below position 5 because they're irrelevant.

The proposal's intuition was that fusion would recover dense-misses by adding sparse-rights. The empirical finding is that **with this corpus and this BM25 implementation, sparse contributes very few "rights" and many "noise" — the fusion is net-negative on precision.**

### 3.3 Per-USMLE-step — Hybrid wins step1, loses step2&3 (mirrors dense)

| meta_info | n | EXP_01 No-RAG | EXP_02 Naive | EXP_03 Sparse | **EXP_04 Hybrid** | EXP_05 Multi-Hop |
|---|---:|---:|---:|---:|---:|---:|
| step1 (basic science) | 679 | 0.7585 | 0.7585 | 0.7452 | **0.7688** | 0.7997 |
| step2&3 (clinical decision) | 594 | **0.7912** | 0.7559 | 0.7727 | 0.7626 | 0.7912 |

**Step 1**: Hybrid is +1.0 pp over both Naive and Sparse — RRF does help on basic-science questions where dense and sparse genuinely catch different chunks.

**Step 2&3**: Hybrid is −2.9 pp vs No-RAG, slightly *worse* than Sparse alone — the sparse half's CP=0.08 dilutes dense's contribution on long clinical vignettes where embedding similarity outperforms keyword overlap.

The orthogonal-strengths story from EXP_03 ([`output_notes/04c_exp03_output.md` §2.3](04c_exp03_output.md)) does not generalise: **RRF doesn't recover sparse's step2&3 win because the rest of sparse's top-5 is noise that crowds out dense's better chunks.**

### 3.4 Per-question-type breakdown on golden_234

| question_type | n | EXP_02 Naive | EXP_03 Sparse | **EXP_04 Hybrid** | EXP_05 Multi-Hop |
|---|---:|---:|---:|---:|---:|
| diagnosis | 126 | 0.897 | 0.905 | 0.889 | 0.937 |
| mechanism | 36 | 0.806 | 0.806 | **0.833** | 0.917 |
| treatment | 34 | 0.735 | **0.794** | 0.706 | 0.824 |
| management | 34 | 0.824 | 0.676 | 0.794 | 0.853 |

**Hybrid is *worse* than Sparse alone on treatment (0.706 vs 0.794)** — the question type where Sparse's keyword matching (drug names, dosing schedules) was its specific strength. Fusion drowns that signal. **Hybrid wins on mechanism (0.833)** — basic-science questions where dense + sparse catch different concept-vs-keyword chunks.

### 3.5 Regression analysis vs No-RAG (test_1273)

| Architecture | both right | both wrong | NoRAG right, RAG wrong | NoRAG wrong, RAG right | net |
|---|---:|---:|---:|---:|---:|
| EXP_02 Naive | 900 | 224 | 85 | 64 | −21 |
| EXP_03 Sparse | 923 | 246 | 62 | 42 | −20 |
| **EXP_04 Hybrid** | 903 | 216 | 82 | **72** | **−10** |
| EXP_05 Multi-Hop | 912 | 187 | 73 | 101 | +28 |

**Hybrid has the lowest regression count of the three single-shot RAG architectures (−10 vs −21/−20).** It fixes 72 No-RAG errors (more than Naive's 64 or Sparse's 42) — RRF *does* surface complementary correct chunks on a meaningful subset of questions. But it still introduces 82 new errors that No-RAG had right; the noise-vs-signal ratio is closer to 1:1 than for any other single-shot RAG.

### 3.6 Pairwise disagreement with EXP_05 Multi-Hop

| Pair | Differ | A right, B wrong | B right, A wrong |
|---|---:|---:|---:|
| EXP_04 Hybrid vs EXP_05 Multi-Hop | 161 | 48 | 86 |
| EXP_04 Hybrid vs EXP_02 Naive | 126 | 53 | 42 |

**Hybrid agrees with Multi-Hop on 87.4 % of test questions but Multi-Hop wins the disagreements 86–48.** When Hybrid and Multi-Hop disagree, Multi-Hop is right ~64 % of the time. **Hybrid is best understood as "Naive + a noisy second opinion" — it fixes a few of Naive's errors at the cost of preserving most of them, while Multi-Hop's iterative re-querying surfaces a substantively different (better) chunk distribution.**

### 3.7 Faithfulness on correct vs wrong rows — judge calibration confirmed

| Architecture | F (correct) | F (wrong) | AC (correct) | AC (wrong) |
|---|---:|---:|---:|---:|
| EXP_04 Hybrid | 0.102 | 0.054 | 0.925 | 0.336 |

Mean Faithfulness on correct rows is 1.9× higher than on wrong rows — the judge correctly assigns slightly higher F when the answer happens to align with retrieved chunks. But the absolute floor (0.102 on correct rows) is below the 0.5 hallucination cutoff. **Hybrid does not meaningfully ground the LLM's correct answers — only 8.7 % of correct answers reach F ≥ 0.5** (vs 11.6 % for Naive, 28.4 % for Multi-Hop).

### 3.8 Operational health

- **Parse failures**: 0 across 1,557 generation calls.
- **Wall time**: 123 min for test_1273 — *slowest* of the 5 architectures because Hybrid carries `rank-bm25`'s O(N=67k) Python scoring (4 s/query) on top of ChromaDB. Multi-Hop's 3-hop ChromaDB-only iteration is 2× faster.
- **Mean latency** 0.443 s per question (Groq generation only). Cost: $0.

---

## 4. Conclusions

1. **EXP_04 is COMPLETE.** All three baseline surfaces written; both notebooks ran end-to-end; RAGAS judge cleared with <2 % NaN per metric. Real cost: ~$11 RAGAS Sonnet 4.6.

2. **Headline numbers for Excel Table 1 row 4** (canonical = `test_1273`):

   | Cell | Value | Source |
   |---|---|---|
   | `Acuuracy` | **0.7659** | `test_1273` |
   | `Exact_Match` | **0.7659** | `test_1273` |
   | `Generator_Model` | `llama-3.3-70b-versatile` | locked |
   | `mean_latency_s` | 0.443 | `test_1273` |
   | `RAGAS_Faithfulness` | 0.0944 | `golden_234` |
   | `RAGAS_Hallucination_Rate` | 0.9174 | derived |
   | `RAGAS_Answer_Relevance` | 0.5966 | `golden_234` |
   | `RAGAS_Context_Precision` | 0.2797 | `golden_234` |
   | `RAGAS_Context_Recall` | 0.3483 | `golden_234` |
   | `Answer_Correctness` | 0.8273 | `golden_234` |

3. **Falsifiable hypotheses verdicts**:
   - ✓ **SUPPORTED**: Hybrid Acuuracy on test_1273 > 0.76 — got **0.7659** (just clears).
   - ✗ **FALSIFIED**: Hybrid Context Precision ≥ 0.50 — got **0.2797** (worse than Naive's 0.329).

4. **Counter-result for the discussion chapter**: *"Reciprocal Rank Fusion of dense and sparse top-k retrieval did not improve retrieval quality on this corpus and benchmark; in fact, fused Context Precision (0.280) was lower than dense-alone (0.329). Mechanism: BM25 retrieval on the chunked Harrison's-dominant corpus has CP = 0.081, contributing more noise than signal to the RRF union; the fused top-5 includes sparse-ranked chunks that dense correctly de-prioritised. Accuracy lift is real but small (+0.86 pp vs Naive on test_1273) — RRF does occasionally recover dense-misses via sparse-rights, but the precision cost dominates the recall gain on aggregated metrics. The orthogonal-strengths intuition (sparse for clinical-decision keywords, dense for basic-science semantics) holds in pairwise comparison (EXP_02 vs EXP_03 disagreed on 153 questions with 50/50 right/wrong split) but does not survive RRF fusion when one of the lanes has near-random precision."*

5. **The thesis-architecture decision space narrows**: Hybrid does not justify its place between Naive and Multi-Hop. **For Phase 5 EXP_07 adaptive routing, the routing table should test a binary No-RAG / Multi-Hop split before assuming the proposal's Naive / Hybrid / Multi-Hop trichotomy.** Hybrid's net regression of −10 vs No-RAG is the best of the single-shot RAG architectures but still net-negative; Multi-Hop is the only architecture with a positive net (+28).

6. **Phase 7 confidence-aware rejection**: Hybrid's Faithfulness distribution is bimodal-near-zero (median 0; only 8.7 % of correct answers reach F ≥ 0.5), so thresholding it as a confidence signal is unstable. **Multi-Hop is the architecture where the confidence layer has signal to gate.**

7. **Operational health**: 0 parse failures; mean latency 0.44 s; RAGAS NaN rate <2 %. Wall time on test_1273 (123 min) is dominated by `rank-bm25`'s O(N) Python scoring inside RRF — would warrant a switch to Pyserini or Elasticsearch for production but is operationally tolerable for thesis-scale evaluation.

---

## 5. Next steps

1. **EXP_05 Multi-Hop** is now also complete — see [`output_notes/04e_exp05_output.md`](04e_exp05_output.md) for the headline finding (Multi-Hop is the only architecture that beats No-RAG).

2. **Phase 4 cross-architecture writeup** for the thesis discussion chapter — central narrative: single-shot retrieval (Naive, Sparse, Hybrid) all underperform No-RAG on the contamination-clean test split because retrieval quality is too low to ground a memorisation-strong LLM; **iterative multi-hop retrieval is what delivers**.

3. **Phase 5 EXP_06/07 adaptive routing** — design EXP_07 to test routing-table variants (binary No-RAG / Multi-Hop vs the proposal's Naive / Hybrid / Multi-Hop) since the empirical evidence above suggests Hybrid may not earn its place.

4. **Methodology paragraph for the thesis writeup** — anchor the EXP_04 finding here so it's not improvised later: *"EXP_04 (Hybrid RAG via Reciprocal Rank Fusion of dense BGE-large and sparse BM25 top-5 retrievers) achieved Acuuracy 0.7659 on the contamination-clean test split — a marginal +0.86 pp lift over Naive Dense (EXP_02) but still −0.79 pp below No-RAG (EXP_01). RAGAS judging exposed an unexpected mechanism: fused Context Precision (0.280) was lower than dense-alone (0.329) because BM25's catastrophic standalone precision (CP = 0.081) contributed more noise than signal to the RRF union. The orthogonal-strengths story (153 of 1,273 test questions disagree between dense and sparse, 50/50 split, suggesting RRF should help) holds in pairwise comparison but does not survive fusion when one retriever is near-random. The accuracy lift is real but small; the precision regression is meaningful. This is a publishable counter-result for the medical-RAG literature: RRF requires both retrievers to clear a precision floor — a weak partner degrades the fused output."*

---

## 6. Files produced

```
results/
├── exp_04_hybrid_rag__smoke_50/
│   ├── predictions.jsonl   ← 50 rows
│   ├── retrieval.jsonl     ← 50 rows × 5 chunk_ids (RRF-fused)
│   └── summary.json
├── exp_04_hybrid_rag__golden_234/
│   ├── predictions.jsonl   ← 234 rows
│   ├── retrieval.jsonl     ← 234 rows × 5 chunk_ids
│   ├── ragas_scores.csv    ← 234 rows × 5 metrics (NaN <2 % per col)
│   └── summary.json        ← all 5 RAGAS aggregates filled
├── exp_04_hybrid_rag__golden_234_ragas_smoke/
│   ├── ragas_scores.csv    ← Stage A pilot (10 rows)
│   └── summary.json
└── exp_04_hybrid_rag__test_1273/
    ├── predictions.jsonl   ← 1,273 rows · canonical headline run
    ├── retrieval.jsonl     ← 1,273 rows × 5 chunk_ids
    └── summary.json        ← Acuuracy = 0.7659
```

Schema in [`docs/results_schema.md`](../results_schema.md). The canonical headline number for Table 1 row 4 lives in `test_1273/summary.json`; RAGAS aggregates live in `golden_234/summary.json` (per the two-surface separation locked 2026-05-06).
