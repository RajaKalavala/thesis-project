# Notebook 08 — EXP_13 / EXP_14 / EXP_15 · Phase 8 Output Notes

> **Notebook:** [`notebooks/08_exp13_14_15_taxonomy.ipynb`](../../notebooks/08_exp13_14_15_taxonomy.ipynb) (16 cells, 3 gated stages)
> **Modules:** [`src/taxonomy/categories.py`](../../src/taxonomy/categories.py) · [`src/taxonomy/labeller.py`](../../src/taxonomy/labeller.py) · [`src/taxonomy/analysis.py`](../../src/taxonomy/analysis.py)
> **Tests:** 17 / 17 passing
> **Run on:** 2026-05-11 (built + ran in one session, three gated stages: smoke → Multi-Hop only → all 5 archs)
> **Phase:** 8 — Group D: hallucination error-type taxonomy (cross-architecture)
> **Architecture / surface:** all 5 architectures on `golden_234`, wrong-answer rows only (157 total question-arch labels)
> **Labeller:** `gpt-4o-mini-2024-07-18` via OpenAI JSON-mode, disk-cached
> **Companion:** [`07_exp08_exp09_output.md`](07_exp08_exp09_output.md) (Phase 7 confidence-aware rejection — the previous step)

---

## 1. Output

```
results/
├── exp_14_taxonomy_labels/
│   ├── smoke_3_multihop.jsonl           ← 3 wrong Multi-Hop questions (Stage A)
│   ├── stage_b_multihop_wrong.jsonl     ← 23 wrong Multi-Hop questions (Stage B — full Multi-Hop)
│   ├── stage_c_norag_wrong.jsonl        ← 23 wrong NoRAG (Stage C)
│   ├── stage_c_naive_wrong.jsonl        ← 35 wrong Naive
│   ├── stage_c_sparse_wrong.jsonl       ← 38 wrong Sparse
│   ├── stage_c_hybrid_wrong.jsonl       ← 38 wrong Hybrid
│   └── stage_c_multihop_wrong.jsonl     ← 23 cached from Stage B (re-run hit cache)
└── exp_15_taxonomy_analysis/
    ├── table7_counts.csv                ← 6 × 5 category × architecture counts
    └── table7_proportions.csv           ← Same but normalised within-architecture
```

**Cost**: **~$3.20** for 157 question-arch labels (gpt-4o-mini at ~$0.02/call). Wall time: ~9 min (Stage A 20 s + Stage B 73 s + Stage C 419 s; the 23 Multi-Hop labels in Stage C all hit cache from Stage B). **0 parse failures across all 157 calls.**

---

## 2. Headline finding — the 6-category taxonomy is empirically a 4-category taxonomy on this benchmark

| Category | Total across 5 archs | % of all wrong | Used by |
|---|---:|---:|---|
| **wrong_reasoning_chain** | 71 | **45.2 %** | All 4 RAG architectures |
| **context_omission** | 23 | **14.6 %** | NoRAG only (100 % of its wrong cases) |
| **unsupported_treatment** | 47 | **29.9 %** | All 4 RAG architectures |
| **unsupported_diagnosis** | 16 | **10.2 %** | All 4 RAG architectures |
| partial_evidence_misuse | 0 | 0.0 % | never fires |
| option_mismatch | 0 | 0.0 % | never fires |

Two of the six proposal categories never fire across the 157 labelled cases:

- **`option_mismatch`** is structurally inaccessible: the generator prompt ends with *"Output exactly one letter (A, B, C, D, or E). Nothing else."* The predicted response text is therefore almost always a single letter — there's no prose argument to mismatch against the final letter.
- **`partial_evidence_misuse`** is conceptually distinct from `wrong_reasoning_chain` but the distinction is subtle. gpt-4o-mini folds these cases into `wrong_reasoning_chain` rather than calling them out separately. This is a **labeller-level finding**, not a generator-level finding: there *may* be partial-evidence-misuse cases hiding inside `wrong_reasoning_chain`, but the classifier doesn't separate them at the prompt-design level.

**Methodology footnote anchored**: *"The proposal's §7.8 6-category taxonomy was empirically a 4-category taxonomy on this benchmark. `option_mismatch` is structurally inaccessible because the generator prompt forces single-letter output. `partial_evidence_misuse` is folded into `wrong_reasoning_chain` by gpt-4o-mini-2024-07-18 in its current calibration. The remaining 4 categories — `wrong_reasoning_chain`, `unsupported_treatment`, `unsupported_diagnosis`, `context_omission` — fire 0–60 % per architecture and produce the cross-architecture taxonomy described in §3."*

---

## 3. Meaning of the outputs

### 3.1 Table 7 — counts (paste-target for Excel Table 7)

| Category | NoRAG | Naive | Sparse | Hybrid | MultiHop | Σ |
|---|---:|---:|---:|---:|---:|---:|
| unsupported_diagnosis | 0 | 6 | 4 | 4 | 2 | 16 |
| unsupported_treatment | 0 | 8 | 12 | 17 | 10 | 47 |
| wrong_reasoning_chain | 0 | 21 | 22 | 17 | 11 | 71 |
| partial_evidence_misuse | 0 | 0 | 0 | 0 | 0 | 0 |
| option_mismatch | 0 | 0 | 0 | 0 | 0 | 0 |
| **context_omission** | **23** | 0 | 0 | 0 | 0 | 23 |
| **Total wrong** | **23** | **35** | **38** | **38** | **23** | **157** |

### 3.2 Table 7 — within-architecture proportions

| Category | NoRAG | Naive | Sparse | Hybrid | MultiHop |
|---|---:|---:|---:|---:|---:|
| unsupported_diagnosis | 0.0 % | 17.1 % | 10.5 % | 10.5 % | 8.7 % |
| unsupported_treatment | 0.0 % | 22.9 % | 31.6 % | 44.7 % | 43.5 % |
| wrong_reasoning_chain | 0.0 % | 60.0 % | 57.9 % | 44.7 % | 47.8 % |
| context_omission | **100.0 %** | 0.0 % | 0.0 % | 0.0 % | 0.0 % |

### 3.3 Five thesis-publishable findings

**Finding 1: NoRAG is 100 % `context_omission` — a clean sanity check passed**.

The labeller correctly assigns `context_omission` to all 23 NoRAG wrong cases (NoRAG retrieves no chunks → every wrong answer has missing context by construction) and to **zero** RAG wrong cases. This validates the labelling methodology: gpt-4o-mini correctly distinguishes "no chunks available" from "chunks present but the model ignored/misused them".

**Finding 2: RAG-architecture wrong-answer distributions are dominated by 2 categories — `wrong_reasoning_chain` + `unsupported_treatment`**.

For all 4 RAG architectures, these two categories account for 83–91 % of wrong answers:

| Architecture | (wrong_reasoning + unsupported_treatment) % |
|---|---:|
| Naive | 82.9 % |
| Sparse | 89.5 % |
| Hybrid | 89.5 % |
| **Multi-Hop** | **91.3 %** |

**Both categories blame the generator, not retrieval**: in both, chunks were present. The model either reasoned wrong from them (chunks support gold but LLM picked otherwise) or picked an option with no chunk support (LLM ignored the chunks). This is the **meta-message of the thesis**: retrieval brings the evidence; the generator is where most errors live.

**Finding 3: Multi-Hop has the LOWEST `unsupported_diagnosis` proportion (8.7 %)** — directly consistent with Phase 4's grounding-improvement narrative.

| Architecture | unsupported_diagnosis % |
|---|---:|
| Naive | 17.1 % |
| Sparse | 10.5 % |
| Hybrid | 10.5 % |
| **Multi-Hop** | **8.7 %** ⭐ |

Multi-Hop's iterative retrieval most effectively reduces the fraction of diagnoses with no chunk support. Aligns with the Phase 4 finding that Multi-Hop is the architecture with graded Faithfulness (median 0.25 vs 0.000 elsewhere).

**Finding 4: Multi-Hop + Hybrid CONCENTRATE in `unsupported_treatment` (43–45 %) — a counter-intuitive distributional shift**.

| Architecture | unsupported_treatment % | wrong_reasoning_chain % |
|---|---:|---:|
| Naive | 22.9 % | 60.0 % |
| Sparse | 31.6 % | 57.9 % |
| Multi-Hop | 43.5 % | 47.8 % |
| Hybrid | **44.7 %** | 44.7 % |

The shift is striking: as we move from single-shot dense (Naive) to multi-hop iterative retrieval, wrong-answer mass migrates *from* `wrong_reasoning_chain` *to* `unsupported_treatment`. Interpretation: better retrieval doesn't fix treatment questions — it just gives the model more wrong treatment options to pick from. The Phase 4 EXP_02 §3.4 finding ("treatment is the hardest question type for RAG") is now anchored at category-level resolution: **treatment errors are predominantly about the model picking an unsupported treatment, not about reasoning wrong from supported chunks.**

**Finding 5: Naive + Sparse have the HIGHEST `wrong_reasoning_chain` (58–60 %) — they retrieve relevant chunks but reason wrong from them**.

Naive and Sparse are above Hybrid and Multi-Hop on `wrong_reasoning_chain`. The labeller's reasoning: at k=5 chunks, the retrieved set often does include gold-supporting evidence, but the LLM picks a different letter. As k grows (Hybrid, Multi-Hop) the LLM has more chunks to draw on and the failure mode shifts toward picking an unsupported option from a richer set. **This makes Naive/Sparse the "model-reasoning-failure" architectures and Hybrid/Multi-Hop the "model-option-selection-failure" architectures** — different root causes, different mitigations.

### 3.4 Per-architecture top categories

| Architecture | 1st category (count) | 2nd category (count) |
|---|---|---|
| **NoRAG** | context_omission (23 / 100 %) | — |
| **Naive** | wrong_reasoning_chain (21 / 60 %) | unsupported_treatment (8 / 23 %) |
| **Sparse** | wrong_reasoning_chain (22 / 58 %) | unsupported_treatment (12 / 32 %) |
| **Hybrid** | unsupported_treatment / wrong_reasoning_chain tie (17 / 45 % each) | unsupported_diagnosis (4 / 11 %) |
| **Multi-Hop** | wrong_reasoning_chain (11 / 48 %) | unsupported_treatment (10 / 43 %) |

The shift from `wrong_reasoning_chain` dominance (Naive/Sparse) to `unsupported_treatment` near-tie (Multi-Hop/Hybrid) is the key cross-arch finding.

### 3.5 Operational properties

- **0 parse failures** across all 157 OpenAI JSON-mode calls.
- **All rationales cite chunk numbers** (e.g., "chunk [3] supports B") or quote response text — defensible audit trail.
- **Cache works perfectly**: Multi-Hop's 23 Stage-B labels hit cache during Stage C (wall time 0.02 sec vs 73 sec for the fresh Stage B run).
- **Latency profile**: ~3.6 sec/call (Stage C aggregate). gpt-4o-mini is faster than gpt-4o on this task.

---

## 4. Conclusions

1. **Phase 8 is COMPLETE.** 157 question-arch labels on disk; 0 parse failures; Table 7 paste-ready CSVs at `results/exp_15_taxonomy_analysis/`.

2. **Headline numbers for Excel Table 7** — both raw counts and within-architecture percentages are written to CSV. The cross-tab structure (6 categories × 5 architectures, with totals row) is the final paste-target.

3. **Three thesis-publishable findings from Phase 8**:
   - **The proposal's 6-category taxonomy is empirically a 4-category taxonomy** on this benchmark (option_mismatch structurally inaccessible due to the generator prompt; partial_evidence_misuse folded into wrong_reasoning_chain by gpt-4o-mini). Methodology footnote required.
   - **NoRAG vs RAG distributions are categorically distinct**: NoRAG is 100 % `context_omission`; RAG is 0 % `context_omission`. A clean validation of the labelling pipeline.
   - **The wrong-answer mass shifts from `wrong_reasoning_chain` (single-shot retrieval) to `unsupported_treatment` (Hybrid / Multi-Hop)** as retrieval quality improves. Better retrieval doesn't fix treatment errors — it just gives the model more wrong-treatment options. This is the category-level resolution of the Phase 4 "treatment is hardest" finding.

4. **Discussion-chapter narrative now has SIX acts** (Phases 4–8 together):
   - Act 1 (EXP_01 → 02): Naive dense retrieval *hurts* a memorisation-strong LLM.
   - Act 2 (EXP_03 → 04): Single-shot retrieval (sparse / hybrid / RRF-fused) does not solve the retrieval-quality problem.
   - Act 3 (EXP_05): Iterative multi-hop retrieval delivers grounded improvement.
   - Act 4 (EXP_07): Adaptive routing captures most of Multi-Hop's gain at 60 % of the compute.
   - Act 5 (EXP_08 / 09): Confidence-aware rejection over RAGAS converts grounded improvement into safety-grade clinical deployment.
   - **Act 6 (EXP_13 / 14 / 15)**: **The cross-architecture error taxonomy shows that wrong-answer mass migrates from "model reasoning failures" (Naive / Sparse) to "model option-selection failures" (Hybrid / Multi-Hop) as retrieval quality improves. The remaining error budget is dominated by `wrong_reasoning_chain` and `unsupported_treatment` — both generator-level failures that better retrieval alone cannot fix.**

5. **Cost**: **~$3.20** (157 × ~$0.02 with gpt-4o-mini). Cumulative project spend now **~$63 / ~$80 ceiling**.

---

## 5. Next steps

### 5.1 Phase 9 — EXP_16 final synthesis

The next (and last experimental) step. Per [`plan.md §11`](../../plan.md):

- **Weighted ranking** of all architectures + the adaptive variant + the confidence-aware-rejected variant.
- Default weighting (from the proposal): 0.25 Accuracy + 0.25 Faithfulness + 0.20 Retrieval Recall + 0.15 Safety (Hallucination Rate or rejection layer's accuracy-on-accepted) + 0.10 Explainability (LIME-SHAP agreement) + 0.05 Latency.
- All inputs are on disk from Phases 4–8; **pure aggregation, no LLM calls**.
- Output: Table 12 in the Excel workbook (Final Weighted Ranking, 6 rows).
- Cost: $0. Wall time: ~30 min.

### 5.2 Methodology paragraph for the thesis writeup

Anchor here so it's not improvised later:

> *"EXP_13 codifies the 6-category hallucination taxonomy from the proposal §7.8. EXP_14 uses gpt-4o-mini-2024-07-18 (JSON-mode, disk-cached) to label each wrong-answer question on the golden_234 surface across all 5 architectures (157 question-arch labels, 0 parse failures, ~$3.20 OpenAI spend). EXP_15 aggregates into a category × architecture cross-tab. The empirical finding: the 6-category taxonomy collapses to 4 categories on this benchmark — `option_mismatch` is structurally inaccessible because the generator prompt enforces single-letter output, and `partial_evidence_misuse` is folded into `wrong_reasoning_chain` by gpt-4o-mini's calibration. Across all 4 RAG architectures, wrong-answer mass concentrates in `wrong_reasoning_chain` (45–60 %) and `unsupported_treatment` (23–45 %), with `unsupported_diagnosis` accounting for 9–17 %. NoRAG's wrong cases are 100 % `context_omission` by construction, validating the labeller's discrimination between retrieval-level and generation-level failure modes. The thesis-relevant cross-architecture pattern: as retrieval quality rises from Naive (k=5 dense) through Hybrid (RRF) to Multi-Hop (3-hop iterative), the wrong-answer distribution shifts from `wrong_reasoning_chain` dominance (60 % on Naive) to `unsupported_treatment` near-parity (45 % on Hybrid, 44 % on Multi-Hop). Better retrieval doesn't fix treatment errors — it shifts them from reasoning failures to option-selection failures."*

### 5.3 Optional Phase 10 — Streamlit demo UI

Per `plan.md §12`. Optional, parallel track. Cached-mode reads of the JSONL files we now have. ~5 days human time; deploys to Streamlit Cloud free tier.

---

## 6. Files produced

```
src/taxonomy/
├── __init__.py
├── categories.py         ← 6 CategoryDef + format helpers (135 LOC)
├── labeller.py           ← gpt-4o-mini JSON-mode classifier + resumable batch (240 LOC)
└── analysis.py           ← crosstab + cohens_kappa + headline_table (115 LOC)

tests/
└── test_taxonomy.py      ← 17 tests · all pass

notebooks/
└── 08_exp13_14_15_taxonomy.ipynb  ← 16 cells · 3 gated stages · ran in ~9 min

results/
├── exp_14_taxonomy_labels/        ← 7 JSONL files (smoke + Stage B + 5 × Stage C)
└── exp_15_taxonomy_analysis/
    ├── table7_counts.csv          ← 6 × 5 paste-ready for Excel Table 7
    └── table7_proportions.csv     ← within-arch percentages
```
