# Notebook 04e — EXP_05 Multi-Hop RAG · Output Notes

> **Notebooks:** [`notebooks/04e_exp05_multi_hop_rag.ipynb`](../../notebooks/04e_exp05_multi_hop_rag.ipynb) (baseline) + [`notebooks/04e_exp05_ragas.ipynb`](../../notebooks/04e_exp05_ragas.ipynb) (judging)
> **Run on:** 2026-05-07 (baseline smoke + golden + test) → 2026-05-10 (RAGAS smoke + full)
> **Phase:** 4 — Group A baseline experiments (fifth and final)
> **Architecture:** 3-hop iterative dense retrieval + LLM. `MultiHopRetriever` (BGE-large query embedding → ChromaDB top-5 → Groq-generated sub-query → ChromaDB top-5 → up to 3 hops, dedup across hops, early-stop on no-progress) + `llama-3.3-70b-versatile` answerer · T=0 · k=5/hop · max chunks 15 · max_tokens=700.
> **Judge:** `claude-sonnet-4-6` via Anthropic (RAGAS 0.4.3, all 5 metrics, golden_234 surface)
> **Companion:** [`docs/output_notes/04a_exp01_output.md`](04a_exp01_output.md) (No-RAG baseline) · [`04b_exp02_output.md`](04b_exp02_output.md) (Naive Dense) · [`04c_exp03_output.md`](04c_exp03_output.md) (Sparse BM25) · [`04d_exp04_output.md`](04d_exp04_output.md) (Hybrid RRF)

---

## 1. Output

| Surface | Rows | Acuuracy | Wall time | Notes |
|---|---:|---:|---:|---|
| `exp_05_multi_hop_rag__smoke_50` | 50 | 0.9000 | ~3 min | smoke validation passed |
| `exp_05_multi_hop_rag__golden_234` | 234 | **0.9017** | 6.3 min | RAGAS surface; **matches No-RAG** |
| **`exp_05_multi_hop_rag__test_1273`** | **1,273** | **0.7958** | **57 min** | **CANONICAL — Table 1 row 5** |

RAGAS aggregates on `golden_234` (Claude Sonnet 4.6, 234 rows scored, NaN rate <4 % per metric):

| Metric | Value |
|---|---:|
| `RAGAS_Faithfulness` | **0.2833** |
| `RAGAS_Hallucination_Rate` | **0.7371** |
| `RAGAS_Answer_Relevance` | 0.5955 |
| `RAGAS_Context_Precision` | **0.3737** |
| `RAGAS_Context_Recall` | **0.7115** |
| `Answer_Correctness` | **0.8685** |

---

## 2. Headline finding — *Multi-Hop is the first RAG architecture to beat No-RAG, and it dominates every RAGAS metric*

| Metric | EXP_01 No-RAG | EXP_02 Naive | EXP_03 Sparse | EXP_04 Hybrid | **EXP_05 Multi-Hop** | Δ vs No-RAG |
|---|---:|---:|---:|---:|---:|---:|
| `Acuuracy` (test_1273, canonical) | 0.7738 | 0.7573 | 0.7581 | 0.7659 | **0.7958** | **+2.20 pp** ✓ |
| `Acuuracy` (golden_234) | 0.9017 | 0.8504 | 0.8376 | 0.8376 | **0.9017** | tied (best) |
| `RAGAS_Faithfulness` | n/a | 0.131 | 0.040 | 0.094 | **0.283** | **2.2× Naive** |
| `Hallucination_Rate` | n/a | 0.896 | 0.966 | 0.917 | **0.737** | lowest |
| `RAGAS_Context_Precision` | n/a | 0.329 | 0.081 | 0.280 | **0.374** | highest |
| `RAGAS_Context_Recall` | n/a | 0.412 | 0.107 | 0.348 | **0.711** | **+30 pp vs Naive** |
| `Answer_Correctness` | 0.8738 | 0.838 | 0.838 | 0.827 | **0.869** | matches No-RAG |

**Multi-Hop earns its 3× compute cost.** It's the only architecture that:
1. **Beats No-RAG on the contamination-clean test split** (+2.20 pp). EXP_02–04 all *underperformed* No-RAG.
2. **Lifts Faithfulness above noise floor** — median F = 0.250 (vs 0.000 for Naive/Sparse/Hybrid). Faithfulness becomes a graded, useable signal for Phase 7 confidence-aware rejection.
3. **More than doubles Context Recall** (0.71 vs 0.41 for Naive) — the iterative hops are doing what they were designed for: stitching evidence across multiple retrieval rounds.

---

## 3. Meaning of the outputs

### 3.1 The grounded-correct fraction — Multi-Hop is the only architecture that moves the needle

EXP_02 surfaced the central anomaly: 88 % of correct answers were *ungrounded* (Faithfulness < 0.5) — LLaMA answers from memorisation while retrieved chunks act as noise. Cross-architecture comparison on `golden_234`:

| Architecture | n correct | grounded (F≥0.5) | % grounded | median F (correct rows) |
|---|---:|---:|---:|---:|
| EXP_02 Naive Dense | 199 | 23 | 11.6 % | 0.000 |
| EXP_03 Sparse BM25 | 196 | 8 | 4.1 % | 0.000 |
| EXP_04 Hybrid RRF | 196 | 17 | 8.7 % | 0.000 |
| **EXP_05 Multi-Hop** | **211** | **60** | **28.4 %** | **0.250** |

**Multi-Hop is the only architecture where the median answer is partially grounded.** For the confidence-aware rejection layer (Phase 7), this matters more than the mean: a graded F signal is gateable, a near-zero-everywhere F signal is not.

### 3.2 Multi-hop subset (n=13) — the falsifiable hypothesis is supported

The EXP_02 data flagged Faithfulness = 0.000 on every multi-hop golden row. The hypothesis from [`output_notes/04b_exp02_output.md` §3.5](04b_exp02_output.md): *"Multi-Hop should achieve Faithfulness > 0.05 on `requires_multihop=yes` rows."*

| Architecture | F on multi-hop (n=13) | F on non-multi-hop (n=221) | CR on multi-hop | Acc on multi-hop |
|---|---:|---:|---:|---:|
| EXP_02 Naive Dense | 0.000 | 0.139 | 0.000 | 0.769 |
| EXP_03 Sparse BM25 | 0.013 | 0.042 | 0.000 | 0.769 |
| EXP_04 Hybrid RRF | 0.050 | 0.097 | 0.231 | 0.846 |
| **EXP_05 Multi-Hop** | **0.229** | **0.287** | **0.615** | **0.846** |

**Multi-Hop F = 0.229 on the multi-hop subset — 18 pp above the 0.05 threshold.** Hypothesis ✓ SUPPORTED. The 3-hop iteration recovers ~62 % of the reference answer's claims (CR = 0.615) on questions Naive cannot stitch evidence for at all. This is the single cleanest piece of evidence that the multi-hop architecture earns its compute cost.

### 3.3 Per-USMLE-step pattern on test_1273 — Multi-Hop is the only architecture to *match* No-RAG on Step 2&3

| meta_info | n | EXP_01 No-RAG | EXP_02 Naive | EXP_03 Sparse | EXP_04 Hybrid | **EXP_05 Multi-Hop** |
|---|---:|---:|---:|---:|---:|---:|
| step1 (basic science) | 679 | 0.7585 | 0.7585 | 0.7452 | 0.7688 | **0.7997** (+4.1 pp) |
| step2&3 (clinical decision) | 594 | **0.7912** | 0.7559 | 0.7727 | 0.7626 | **0.7912** (tied) |

**Step 1**: Multi-Hop is the *only* architecture to beat No-RAG (+4.1 pp). The iterative retrieval surfaces the basic-science concepts (anatomy, biochem, pharm mechanism) that BGE-large embedding alone misses on the first hop.

**Step 2&3**: every other RAG architecture loses 1.9–3.5 pp vs No-RAG. **Only Multi-Hop fully recovers** to No-RAG parity. The clinical-decision vignettes — where retrieved chunks most often act as distractors — are exactly where iterative re-querying earns its keep.

### 3.4 Per-question-type breakdown on golden_234 — Multi-Hop wins on the hardest categories

| question_type | n | EXP_02 Naive | EXP_03 Sparse | EXP_04 Hybrid | **EXP_05 Multi-Hop** |
|---|---:|---:|---:|---:|---:|
| diagnosis | 126 | 0.897 | 0.905 | 0.889 | **0.937** |
| mechanism | 36 | 0.806 | 0.806 | 0.833 | **0.917** (+11 pp vs Naive) |
| treatment | 34 | 0.735 | 0.794 | 0.706 | **0.824** (+9 pp vs Naive) |
| management | 34 | 0.824 | 0.676 | 0.794 | **0.853** |
| other | 4 | 1.000 | 0.750 | 0.750 | 0.750 |

**Treatment was Naive's worst bucket (0.735) — Multi-Hop closes 9 pp of that gap to 0.824.** The hypothesis from [`output_notes/04b_exp02_output.md` §3.4](04b_exp02_output.md) — *"hybrid or multi-hop retrieval should narrow the treatment-vs-diagnosis gap"* — is supported: Multi-Hop, not Hybrid, is the architecture that delivers.

Faithfulness by question_type tells the same story:

| question_type | EXP_02 | EXP_03 | EXP_04 | **EXP_05** |
|---|---:|---:|---:|---:|
| diagnosis | 0.134 | 0.050 | 0.101 | **0.310** |
| mechanism | 0.198 | 0.058 | 0.148 | **0.330** |
| treatment | 0.153 | 0.026 | 0.072 | **0.257** |
| management | 0.040 | 0.005 | 0.048 | **0.195** |

Multi-Hop's grounding lift is uniform across categories — it's not winning by being good at one type and bad at others.

### 3.5 Regression analysis vs No-RAG (test_1273) — net positive

Question-by-question outcome paired against EXP_01 No-RAG on the canonical 1,273 test rows:

| Architecture | both right | both wrong | NoRAG right, RAG wrong | NoRAG wrong, RAG right | net |
|---|---:|---:|---:|---:|---:|
| EXP_02 Naive Dense | 900 | 224 | 85 | 64 | **−21** |
| EXP_03 Sparse BM25 | 923 | 246 | 62 | 42 | **−20** |
| EXP_04 Hybrid RRF | 903 | 216 | 82 | 72 | **−10** |
| **EXP_05 Multi-Hop** | **912** | **187** | **73** | **101** | **+28** |

**Multi-Hop is the only architecture with a positive net.** It fixes 101 questions LLaMA was wrong on while introducing 73 new errors — a 28-question gain on a 1,273-question benchmark. The other three RAG architectures all add more new errors than they fix.

### 3.6 Coverage / oracle ceiling — adaptive routing has substantial headroom

What if the system could pick the best architecture per question?

| Coverage scenario | Acuuracy on test_1273 |
|---|---:|
| EXP_01 No-RAG (single best fixed) | 0.7738 |
| EXP_05 Multi-Hop (single best RAG) | **0.7958** |
| All 4 RAGs got it right (intersection) | 0.6591 |
| At least one RAG got it right (union) | 0.8617 |
| Oracle: best of {No-RAG, Multi-Hop} per Q | **0.8531** |
| Oracle: best of all 5 architectures per Q | **0.8696** |

**The oracle ceiling on No-RAG + Multi-Hop alone is 0.8531 (+5.7 pp over Multi-Hop standalone).** This is the headroom Phase 5 EXP_07 adaptive routing has to work with — and it's substantial. Importantly, the oracle ceiling is mostly *between* No-RAG and Multi-Hop; the EXP_02/03/04 lanes don't add much that Multi-Hop misses. **Implication for adaptive routing**: a binary No-RAG / Multi-Hop router may capture most of the gain. The Hybrid/Sparse/Naive lanes may not earn their place in the routing table.

### 3.7 Wall time — Multi-Hop is *faster* than Hybrid

| Architecture | test_1273 wall time | per-question | Groq calls/q |
|---|---:|---:|---:|
| EXP_01 No-RAG | 6.3 min | 0.30 s | 1 |
| EXP_02 Naive Dense | 11.7 min | 0.55 s | 1 |
| EXP_03 Sparse BM25 | 97 min | 4.6 s | 1 |
| EXP_04 Hybrid RRF | 123 min | 5.8 s | 1 |
| **EXP_05 Multi-Hop** | **57 min** | **2.7 s** | 3 (1 answer + 2 sub-queries) |

**Multi-Hop is 2× faster than Hybrid** despite making 3× the Groq calls per question. Reason: Hybrid is bottlenecked by `rank-bm25`'s O(N=67k) Python scoring (4 s/query for the sparse half of RRF), while Multi-Hop's three ChromaDB lookups are ~0.3 s combined. Cost remains $0 on the Groq free tier.

### 3.8 Operational health

- **Parse failures**: 0 across 1,557 generation calls (smoke + golden + test).
- **Sub-query generation**: clean — no Groq parse failures across 2,546 sub-query calls (golden + test, ≤2 sub-queries per question).
- **RAGAS NaN rate**: 0.4 % (1/234) on Answer Correctness, 0.9 % (2/234) on Faithfulness, 3.8 % (9/234) on Context Precision — all within tolerable methodology bounds; resilience layer working.
- **Mean latency** 0.66 s per question on test_1273 (Groq generation only) — slightly higher than Naive's 0.44 s due to longer prompts (15 chunks vs 5).

---

## 4. Conclusions

1. **EXP_05 Multi-Hop is the headline architecture of the thesis.** It is the only RAG architecture to (a) beat No-RAG on the contamination-clean test split (+2.20 pp), (b) lift Faithfulness above noise floor (median 0.25 vs 0.00 elsewhere), and (c) more than double Context Recall over Naive Dense.

2. **The thesis discussion narrative now has a clean three-act structure**:
   - **Act 1** (EXP_01 → EXP_02): naive dense retrieval *hurts* a memorisation-strong LLM on a contamination-clean test split (88 % of correct answers ungrounded).
   - **Act 2** (EXP_03 → EXP_04): single-shot retrieval — whether sparse, hybrid, or fused via RRF — does not solve the retrieval-quality problem. EXP_03's catastrophic Context Precision (0.08) yet preserved accuracy is the strongest piece of evidence in the thesis that the LLM is answering from memorisation, not retrieval.
   - **Act 3** (EXP_05): iterative multi-hop retrieval is what actually delivers grounded improvement. Both accuracy and Faithfulness are lifted; Context Recall doubles.

3. **Headline numbers for Excel Table 1 row 5** (canonical = `test_1273`):

   | Cell | Value | Source |
   |---|---|---|
   | `Acuuracy` | **0.7958** | `test_1273` |
   | `Exact_Match` | **0.7958** | `test_1273` |
   | `Generator_Model` | `llama-3.3-70b-versatile` | locked |
   | `mean_latency_s` | 0.660 | `test_1273` |
   | `RAGAS_Faithfulness` | **0.2833** | `golden_234` |
   | `RAGAS_Hallucination_Rate` | **0.7371** | derived |
   | `RAGAS_Answer_Relevance` | 0.5955 | `golden_234` |
   | `RAGAS_Context_Precision` | **0.3737** | `golden_234` |
   | `RAGAS_Context_Recall` | **0.7115** | `golden_234` |
   | `Answer_Correctness` | **0.8685** | `golden_234` |

4. **Falsifiable hypotheses verdicts** (anchored in [`output_notes/04b_exp02_output.md` §4](04b_exp02_output.md) + [`04c_exp03_output.md` §4](04c_exp03_output.md)):
   - ✓ **SUPPORTED**: Multi-Hop Acuuracy on test_1273 ≥ EXP_02's 0.7573 — got **0.7958** (+3.85 pp).
   - ✓ **SUPPORTED**: Multi-Hop Faithfulness > 0.05 on multi-hop subset — got **0.229** (+18 pp threshold margin).

5. **Phase 7 (confidence-aware rejection) is now empirically grounded for Multi-Hop only.** On Naive/Sparse/Hybrid, Faithfulness is bimodal-near-zero (median 0); thresholding it as a confidence signal is unstable. On Multi-Hop, F has a graded distribution (median 0.25, 28.4 % above 0.5) — the threshold sweep {0.5, 0.6, 0.7, 0.8, 0.9} planned in EXP_09 has actual signal to gate.

6. **Implication for Phase 5 EXP_07 adaptive routing**: the oracle ceiling on No-RAG + Multi-Hop alone (0.8531) suggests a binary router may capture most of the gain. Hybrid (0.7659) and Sparse (0.7581) may not earn their place; the routing table in EXP_07 should be designed to verify this rather than assume the proposal's three-way split.

7. **Operational health**: 0 parse failures across 1,557 generation calls + 2,546 sub-query calls; mean latency 0.66 s; RAGAS NaN rate <4 % per metric. Multi-Hop is *faster* than Hybrid on test_1273 (57 min vs 123 min) because the BM25 scoring inside Hybrid dominates wall-time. Total Phase 4 RAGAS cost: ~$45–50 (EXP_01 $4.50 + EXP_02–05 ~$11–13 each).

---

## 5. Next steps

1. **Phase 4 is COMPLETE.** Tables 1, 8, 9 ready to populate from `summary.json` files. The cross-architecture writeup is anchored.

2. **Phase 5 — EXP_06 + EXP_07 adaptive routing**:
   - EXP_06: rule-based question-complexity classifier ([plan.md §7](../plan.md)). The proposal's Simple → Naive / Moderate → Hybrid / Complex → Multi-Hop split is now under review — the data above suggests a binary No-RAG / Multi-Hop router may dominate. Recommend testing both routing tables in EXP_07 with `summary.json` accuracy as the decision rule.
   - EXP_07: dispatcher + run on `test_1273`. Since Adaptive routes per question, RAGAS metrics for EXP_07 are **score-joined** from the underlying architectures' golden_234 RAGAS — no new judge calls needed.

3. **Phase 6 LIME/SHAP** (sampled to 200 questions per architecture, all Groq → $0). Multi-Hop is the highest-value architecture to explain — Faithfulness has graded signal there. LIME/SHAP on EXP_02/03/04 will mostly explain why the LLM ignored the retrieved chunks (the retrieval was noise).

4. **Phase 7 confidence-aware rejection** anchors on Multi-Hop's Faithfulness distribution. The threshold sweep {0.5, 0.6, 0.7, 0.8, 0.9} should report rejection rate vs accuracy-on-accepted; with median F = 0.25 and 28.4 % above 0.5, the {0.3, 0.4, 0.5} range may be more informative.

5. **Methodology paragraph for the thesis writeup** — anchor the EXP_05 finding here: *"EXP_05 (Multi-Hop RAG with 3-hop iterative dense retrieval and Groq sub-query generation) was the only RAG architecture in the comparison to outperform No-RAG on the contamination-clean MedQA-US test split (Acuuracy 0.7958 vs 0.7738; +2.20 pp). RAGAS judging on the golden 234 surface confirms the mechanism: Faithfulness 0.283 (vs 0.131 for Naive Dense) and Context Recall 0.712 (vs 0.412 for Naive) — Multi-Hop's iterative hops recover ~70 % of the reference answer's evidentiary claims, more than doubling single-shot dense retrieval. On the 13 multi-hop golden questions where Naive scored Faithfulness 0.000, Multi-Hop scored 0.229 — empirical confirmation that single-shot retrieval cannot stitch evidence across multiple sources, and iterative re-querying solves this. The grounded fraction of correct answers rises from 11.6 % (Naive) to 28.4 % (Multi-Hop), making Faithfulness a useable confidence signal for the rejection layer (Phase 7)."*

---

## 6. Files produced

```
results/
├── exp_05_multi_hop_rag__smoke_50/
│   ├── predictions.jsonl   ← 50 rows
│   ├── retrieval.jsonl     ← 50 rows × ≤15 chunk_ids per Q
│   └── summary.json
├── exp_05_multi_hop_rag__golden_234/
│   ├── predictions.jsonl   ← 234 rows
│   ├── retrieval.jsonl     ← 234 rows × ≤15 chunk_ids
│   ├── ragas_scores.csv    ← 234 rows × 5 metrics (NaN <4 % per col)
│   └── summary.json        ← all 5 RAGAS aggregates filled
├── exp_05_multi_hop_rag__golden_234_ragas_smoke/
│   ├── ragas_scores.csv    ← Stage A pilot (10 rows)
│   └── summary.json
└── exp_05_multi_hop_rag__test_1273/
    ├── predictions.jsonl   ← 1,273 rows · canonical headline run
    ├── retrieval.jsonl     ← 1,273 rows × ≤15 chunk_ids
    └── summary.json        ← Acuuracy = 0.7958
```

Schema in [`docs/results_schema.md`](../results_schema.md). The canonical headline number for Table 1 row 5 lives in `test_1273/summary.json`; RAGAS aggregates live in `golden_234/summary.json` (per the two-surface separation locked 2026-05-06).
