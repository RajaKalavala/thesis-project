# Notebook 04b — EXP_02 Naive RAG · Output Notes

> **Notebooks:** [`notebooks/04b_exp02_naive_rag.ipynb`](../../notebooks/04b_exp02_naive_rag.ipynb) (baseline) + [`notebooks/04b_exp02_ragas.ipynb`](../../notebooks/04b_exp02_ragas.ipynb) (judging)
> **Run on:** 2026-05-06 (baseline smoke + golden + test) → 2026-05-07 (RAGAS smoke + full)
> **Phase:** 4 — Group A baseline experiments (second of five)
> **Architecture:** Dense retrieval + LLM. `NaiveRetriever` (BGE-large query embedding → ChromaDB top-5 over BAAI/bge-large-en-v1.5 indices) + `llama-3.3-70b-versatile` answerer · T=0 · k=5 · max_tokens=700.
> **Judge:** `claude-sonnet-4-6` via Anthropic (RAGAS 0.4.3, all 5 metrics, golden_234 surface)
> **Companion:** [`docs/results_schema.md`](../results_schema.md) · [`docs/output_notes/04a_exp01_output.md`](04a_exp01_output.md) (No-RAG baseline this experiment is compared against)

---

## 1. Output

Four result directories under `results/`:

| Surface | Rows | Accuracy | RAGAS run? | Notes |
|---|---:|---:|---|---|
| `exp_02_naive_rag__smoke_50` | 50 | **0.7800** | no | smoke validation |
| `exp_02_naive_rag__golden_234` | 234 | **0.8504** | ✅ all 5 metrics | RAGAS judging surface |
| **`exp_02_naive_rag__test_1273`** | **1,273** | **0.7573** | (RAGAS surface is golden) | **CANONICAL — Table 1 row 2** |
| `exp_02_naive_rag__full_12723` | 4,844 (partial) | n/a | no | LEGACY — abandoned 2026-05-06 under the test-split lock; see `README_LEGACY.md` |

The `test_1273` and `golden_234` directories are the canonical artifacts. The full-12,723 partial is preserved for audit trail only.

---

## 2. Headline finding — *RAG hurts accuracy on a contamination-clean surface*

| Metric | EXP_01 No-RAG | EXP_02 Naive RAG | Δ |
|---|---:|---:|---:|
| `Acuuracy` (test_1273, canonical) | **0.7738** | **0.7573** | **−1.65 pp** |
| `Acuuracy` (golden_234) | 0.9017 | 0.8504 | −5.13 pp |
| `Answer_Correctness` (golden_234) | 0.8738 | 0.8376 | −3.62 pp |
| `RAGAS_Answer_Relevance` | 0.5977 | 0.5961 | −0.16 pp (no change) |
| **`RAGAS_Faithfulness`** | `null` (No-RAG) | **0.1308** | first measured |
| **`RAGAS_Hallucination_Rate`** | `null` (No-RAG) | **0.8957** | first measured |
| `RAGAS_Context_Precision` | `null` | 0.3285 | first measured |
| `RAGAS_Context_Recall` | `null` | 0.4124 | first measured |

**Naive RAG with BGE-large dense retrieval is *worse* than No-RAG on the contamination-clean test split.** This is unexpected at face value but methodologically clean once you read the RAGAS numbers — the LLM is *largely ignoring* the retrieved context, so retrieval mostly adds noise without adding signal.

---

## 3. Meaning of the outputs

### 3.1 The "memorised, not grounded" smoking gun

Cross-tab on the golden 234 — was the answer correct? was it grounded in the retrieved chunks (Faithfulness ≥ 0.5)?

| | Ungrounded (F < 0.5) | Grounded (F ≥ 0.5) | Total |
|---|---:|---:|---:|
| Wrong answer | 34 | 1 | 35 |
| Correct answer | **176** | 23 | 199 |
| Total | 210 | 24 | 234 |

**88.3 % of correct answers were ungrounded** — LLaMA produced the right option without the retrieved chunks supporting it. The confidence-aware-rejection layer (Phase 7) is what this thesis builds to address exactly this case: an answer that's correct *by accident* (memorisation) is operationally indistinguishable from a confident hallucination, and only Faithfulness scoring exposes the difference.

Mean Faithfulness:
- Correct answers: **0.1477**
- Wrong answers: **0.0333**

The judge correctly assigns slightly higher Faithfulness when the answer happens to align with the retrieved chunks, but in absolute terms both are far below the 0.5 hallucination cutoff. **Naive RAG is not actually grounding.**

### 3.2 Why retrieval quality is the issue — Context Precision = 0.33

The Faithfulness collapse has a clean upstream cause: **only 33 % of retrieved chunks are actually relevant to the question** (Context Precision = 0.3285 over 234 rows). Two-thirds of what BGE-large dense retrieval surfaces is noise.

Mean Context Recall = 0.4124 — only 41 % of the reference answer's claims are recoverable from the retrieved chunks. **The right evidence is being missed almost as often as it's being found.**

This is the diagnostic the thesis was built to surface. **Hybrid RAG (EXP_04) and Multi-Hop (EXP_05) need to demonstrably lift Context Precision and Context Recall to "earn their keep".** If they don't, the discussion chapter has a clean *"naive dense retrieval is insufficient for medical QA on USMLE-style vignettes"* finding.

### 3.3 By MedQA split — golden 234 is heavily train-skewed

| Split | n | Acuuracy | Faithfulness | Context Precision | Context Recall | Answer Correctness |
|---|---:|---:|---:|---:|---:|---:|
| train | 188 | 0.846 | 0.137 | 0.335 | 0.399 | 0.842 |
| dev | 28 | 0.857 | 0.077 | 0.260 | 0.375 | 0.796 |
| **test** | **18** | **0.889** | **0.145** | 0.367 | **0.611** | 0.858 |

The 18-row test slice within golden has the **highest** Context Recall (0.611 vs 0.4 elsewhere) and Answer Correctness — but n=18 is too small to stake a thesis claim on. The proper test-split evaluation is on `test_1273` (where RAGAS isn't run because there's no golden reference for those rows).

### 3.4 By question_type — *treatment* is where Naive RAG hurts most

| Type | n | Acuuracy | Faithfulness | Answer Correctness |
|---|---:|---:|---:|---:|
| diagnosis | 126 | **0.897** (best) | 0.134 | 0.849 |
| management | 34 | 0.824 | **0.040** (worst) | 0.822 |
| mechanism | 36 | 0.806 | **0.198** (best) | 0.843 |
| treatment | 34 | **0.735** (worst) | 0.153 | 0.789 |
| other | 4 | 1.000 | 0.000 | 0.969 |

*Treatment* questions ("what's the next step in management?") are where Naive RAG falls apart on accuracy. *Management* questions have the lowest Faithfulness (4 %), suggesting management answers are most reliant on the LLM's medical-decision-making memory rather than on retrieved evidence. **Concrete EXP_03/04/05 hypothesis**: hybrid or multi-hop retrieval should narrow the treatment-vs-diagnosis gap by surfacing the relevant clinical-guideline chunks.

### 3.5 Multi-hop questions — Naive RAG completely fails

| `requires_multihop` | n | Acuuracy | Faithfulness | Answer Correctness |
|---|---:|---:|---:|---:|
| no | 221 | 0.855 | 0.139 | 0.838 |
| **yes** | **13** | **0.769** | **0.000** | 0.836 |

**Faithfulness = 0.000 on every multi-hop question** — Naive RAG retrieves a single batch of 5 chunks and cannot stitch together evidence across multiple sources. Acuuracy drops 8.6 pp on multi-hop too. This sets up a clean falsifiable hypothesis for **EXP_05 Multi-Hop RAG**: *"the multi-hop architecture should achieve Faithfulness > 0 on multi-hop questions by accumulating evidence across ≤3 retrieval rounds."* If EXP_05 fails this, multi-hop doesn't earn its 3× compute cost.

### 3.6 Test split (n=1,273) — by USMLE-step breakdown

| meta_info | n | EXP_01 No-RAG | EXP_02 Naive RAG | Δ |
|---|---:|---:|---:|---:|
| step1 | 679 | 0.7585 | 0.7585 | **0.00 pp** |
| step2&3 | 594 | 0.7912 | 0.7559 | **−3.54 pp** |

**Naive RAG is neutral on Step 1 questions and hurts Step 2&3 questions.** Step 2&3 are the longer, more clinically-decision-oriented vignettes — exactly where retrieval *should* help most. Instead it distracts. Strong evidence the noise-rate of Naive dense retrieval (Context Precision 0.33) is what's driving the regression.

### 3.7 Regression analysis — net loss of 21 questions

Question-by-question outcome on `test_1273`:

| EXP_01 outcome | EXP_02 outcome | n |
|---|---|---:|
| ✅ correct | ✅ correct | 900 |
| ❌ wrong | ❌ wrong | 224 |
| ✅ correct | ❌ wrong (regression) | **85** |
| ❌ wrong | ✅ correct (fix) | **64** |

**Naive RAG fixed 64 questions LLaMA was wrong on, but introduced 85 new errors LLaMA had been right on. Net: −21 questions.** The thesis discussion can frame this as: *"Naive RAG provides modest evidence-grounded correction (64 fixes) but at the cost of more frequent retrieval-distractor errors (85 regressions). Aggregated across the test surface, the noise-to-signal ratio is unfavourable."*

### 3.8 Resilience layer worked — NaN rate dropped from 25–42 % (EXP_01) to <2 % (EXP_02)

| Metric | EXP_01 NaN rate (no resilience layer) | EXP_02 NaN rate (RunConfig + retries) |
|---|---:|---:|
| Faithfulness | n/a (No-RAG) | 4 / 234 = **1.7 %** |
| Context Precision | n/a | 4 / 234 = **1.7 %** |
| Context Recall | n/a | 0 / 234 = **0.0 %** |
| Answer Relevancy | 60 / 234 = 25.6 % | 2 / 234 = **0.9 %** |
| Answer Correctness | 97 / 234 = 41.5 % | 5 / 234 = **2.1 %** |

The 2026-05-06 fix (`RunConfig(max_workers=4, max_retries=10, max_wait=120)`) **reduced Anthropic-side rate-limit-induced NaN by 95 %+**. EXP_02's RAGAS scores are computed over near-complete samples (~98 % coverage on every metric). Methodology footnote required for EXP_01's lower coverage; EXP_02 onward is uniformly high-coverage.

---

## 4. Conclusions

1. **EXP_02 is COMPLETE.** All four surfaces written; both notebooks run end-to-end; RAGAS judge ran cleanly with <2 % NaN rate. Real cost ≈ $11.50 ($0.50 smoke + ~$11 full).

2. **Headline numbers for Excel Table 1 row 2:**

   | Cell | Value | Source |
   |---|---|---|
   | `Acuuracy` | **0.7573** | `test_1273` |
   | `Exact Match` | **0.7573** | `test_1273` |
   | `Generator Model` | `llama-3.3-70b-versatile` | locked |
   | `mean_latency_s` | 0.444 | `test_1273` |
   | `RAGAS_Faithfulness` | **0.1308** | `golden_234` (n=230 non-NaN) |
   | `RAGAS_Hallucination_Rate` | **0.8957** | derived from Faithfulness < 0.5 |
   | `RAGAS_Answer_Relevance` | 0.5961 | golden_234 |
   | `RAGAS_Context_Precision` | 0.3285 | golden_234 |
   | `RAGAS_Context_Recall` | 0.4124 | golden_234 |
   | `Answer_Correctness` | 0.8376 | golden_234 |

3. **The thesis discussion now has its central narrative anchor.** EXP_01 set up the contamination story (LLaMA pre-training on MedQA → 0.77 baseline on test). EXP_02 reveals that **naive retrieval doesn't help on a contaminated model — and it actively hurts on Step 2&3 + treatment + multi-hop questions** because Context Precision is only 0.33. The whole thesis-distinguishing question — "*does retrieval improve grounding even when the LLM already knows the answer?*" — now has a measurable answer: **No, with naive dense retrieval. 88 % of correct answers are still ungrounded.** The remaining experiments (EXP_03 sparse, EXP_04 hybrid, EXP_05 multi-hop, EXP_07 adaptive) need to demonstrate they fix the retrieval-quality problem.

4. **Concrete falsifiable hypotheses for EXP_03–EXP_05** generated from the EXP_02 data:
   - **EXP_03 Sparse RAG (BM25)**: hypothesis — Context Precision will *improve* on questions with rare medical terms (e.g. drug names, anatomical structures) where keyword matching beats embedding similarity. Falsified if sparse Context Precision ≤ 0.33.
   - **EXP_04 Hybrid RAG (RRF)**: hypothesis — Context Precision should be ≥ 0.50 by combining dense + sparse top-k. Falsified if Hybrid Context Precision ≤ Naive's 0.33.
   - **EXP_05 Multi-Hop RAG**: hypothesis — multi-hop architecture achieves Faithfulness > 0 on the 13 multi-hop golden rows where Naive scored 0.000. Falsified if Multi-Hop Faithfulness on `requires_multihop=yes` rows is ≤ 0.05.

5. **The confidence-aware rejection layer (Phase 7) is the central novelty contribution.** With Hallucination_Rate = 0.90 on Naive RAG, even a moderately accurate confidence signal (retrieval score + Faithfulness + LIME-SHAP agreement) can dramatically improve safety. This is now the strongest claim in the thesis.

6. **Operational health**: 0 parse failures across 1,557 generation calls (test_1273 + golden_234 + smoke_50). Mean latency 0.44 s vs Notebook 03's 1.38 s — Groq is fast for this workload. Resilience-layer NaN rate <2 % across all 5 RAGAS metrics. Total Phase 4 RAGAS cost so far: $5 (EXP_01 $4.50 + EXP_02 $0.50 smoke + ~$11 full) ≈ $16.

---

## 5. Next steps

1. **Build EXP_03 Sparse RAG** — `src/retrieval/sparse.py` (BM25 wrapper around the existing `bm25_top_k` helper) ≈ 30 lines + duplicate the two notebook templates with `04c_` prefix and `exp_03_sparse_rag` paths. Test the *"sparse beats dense on rare-term questions"* hypothesis from §4 above.

2. **After EXP_03**: EXP_04 Hybrid RAG (refactor existing `src/retrieval/hybrid.py` under the `Retriever` ABC), then EXP_05 Multi-Hop RAG (new module). Each follows the same notebook template; only the retriever class swaps.

3. **Methodology paragraph for the thesis writeup** — anchor the EXP_02 finding here so it's not improvised later: *"EXP_02 (Naive Dense RAG with BGE-large) was empirically inferior to EXP_01 (No-RAG) on the contamination-clean test split (-1.65 pp accuracy). RAGAS judging exposes the mechanism: Context Precision = 0.33 (only 1/3 of retrieved chunks were relevant) and Faithfulness = 0.13 (mean across all answers; median = 0.000). On 88 % of questions LLaMA answered correctly, the answer was un-grounded in the retrieved evidence (Faithfulness < 0.5), demonstrating that naive retrieval did not contribute to the LLM's correct decisions on this benchmark — the LLM was answering from pre-training memorisation, with retrieved chunks acting as noise rather than signal. This finding motivates the subsequent retrieval-quality improvements (Hybrid RAG, Multi-Hop RAG) and the confidence-aware rejection layer (Phase 7) that distinguishes memorised-correct from grounded-correct answers."*

---

## 6. Files produced

```
results/
├── exp_02_naive_rag__smoke_50/
│   ├── predictions.jsonl   ← 50 rows
│   ├── retrieval.jsonl     ← 50 rows × 5 chunk_ids
│   └── summary.json
├── exp_02_naive_rag__golden_234/
│   ├── predictions.jsonl   ← 234 rows
│   ├── retrieval.jsonl     ← 234 rows × 5 chunk_ids
│   ├── ragas_scores.csv    ← 234 rows × 5 metrics (NaN rate <2 % per col)
│   └── summary.json        ← updated 2026-05-07 with all 5 RAGAS aggregates
├── exp_02_naive_rag__golden_234_ragas_smoke/
│   ├── ragas_scores.csv    ← Stage A pilot (10 rows; the validation record)
│   └── summary.json
├── exp_02_naive_rag__test_1273/
│   ├── predictions.jsonl   ← 1,273 rows · canonical headline run
│   ├── retrieval.jsonl     ← 1,273 rows × 5 chunk_ids
│   └── summary.json        ← Acuuracy = 0.7573
└── exp_02_naive_rag__full_12723/
    ├── predictions.jsonl   ← 4,844 partial rows (LEGACY — abandoned)
    ├── retrieval.jsonl     ← 4,844 partial rows
    └── README_LEGACY.md
```

Schema in [`docs/results_schema.md`](../results_schema.md). The canonical headline number for Table 1 row 2 lives in `test_1273/summary.json`; RAGAS aggregates live in `golden_234/summary.json` (per the two-surface separation locked 2026-05-06).
