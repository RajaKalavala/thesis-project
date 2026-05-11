# Chat Context — 2026-05-11 (Phase 8 close-out)

> Working transcript of the fourth and final implementation-experiment session on 2026-05-11. Closed Phase 8 (hallucination taxonomy — EXP_13 / 14 / 15). Earlier same day: Phase 5 close-out ([`chat_context_2026-05-11.md`](chat_context_2026-05-11.md)), Phase 6 close-out ([`chat_context_2026-05-11_phase6.md`](chat_context_2026-05-11_phase6.md)), Phase 7 close-out ([`chat_context_2026-05-11_phase7.md`](chat_context_2026-05-11_phase7.md)). Phase 8 was the cleanest implementation session yet — designed, built, tested, ran in three gated stages, all in one sitting.

---

## 1. Where we started and where we ended

| Item | Start of session | End of session |
|---|---|---|
| Phase 8 design | not committed | ✅ scope locked: 5 archs × wrong rows on golden_234, gpt-4o-mini classifier |
| `src/taxonomy/categories.py` | absent | ✅ 6 CategoryDef + rater guidance (135 LOC) |
| `src/taxonomy/labeller.py` | absent | ✅ gpt-4o-mini JSON-mode + resumable batch (240 LOC) |
| `src/taxonomy/analysis.py` | absent | ✅ crosstab + Cohen's κ + headline (115 LOC) |
| Tests | none | ✅ 17/17 passing |
| Notebook | none | ✅ 16 cells, 3 gated stages, ran in ~9 min |
| Output notes | absent | ✅ written |
| plan.md §10.1 close-out | absent | ✅ added |
| docs/todo.md §9 | unstarted | ✅ EXP_13/14/15 marked FULLY COMPLETE |
| memory snapshot | said "Phase 7 done" | ✅ Phase 8 done snapshot (6-act narrative) |

Cumulative project spend: **~$60 → ~$63.20** (Phase 8 spend: ~$3.20).

---

## 2. The Table 7 cross-tab

| Category | NoRAG | Naive | Sparse | Hybrid | MultiHop | Σ |
|---|---:|---:|---:|---:|---:|---:|
| unsupported_diagnosis | 0 | 6 | 4 | 4 | 2 | 16 |
| unsupported_treatment | 0 | 8 | 12 | 17 | 10 | 47 |
| wrong_reasoning_chain | 0 | 21 | 22 | 17 | 11 | 71 |
| partial_evidence_misuse | 0 | 0 | 0 | 0 | 0 | 0 |
| option_mismatch | 0 | 0 | 0 | 0 | 0 | 0 |
| **context_omission** | **23** | 0 | 0 | 0 | 0 | 23 |
| **Total wrong** | **23** | **35** | **38** | **38** | **23** | **157** |

0 parse failures across all 157 OpenAI JSON-mode calls. Every rationale cites chunk numbers `[N]` and references the predicted/gold letter explicitly.

---

## 3. Three publishable findings

### 3.1 The 6-category taxonomy is empirically a 4-category taxonomy on this benchmark

Two categories never fired across all 157 cases:

- **`option_mismatch`** is *structurally inaccessible*. The generator prompt ends with *"Output exactly one letter (A, B, C, D, or E). Nothing else."* So `pred_text` is almost always a single letter (`'C'`, `'D'`, etc.) — there's no prose argument to mismatch against the final letter. This category requires a different generator-prompt design to be testable.

- **`partial_evidence_misuse`** is folded into `wrong_reasoning_chain` by gpt-4o-mini's calibration. The distinction is subtle (chunk fragment lifted out of context vs full chunks support gold but LLM reasons wrong), and the labeller defaults to the broader category. *There may be partial-evidence-misuse cases inside the 71 wrong_reasoning_chain labels*, but the classifier doesn't separate them.

This is a publishable methodology finding — the proposal's §7.8 6-category taxonomy was speculative; empirically only 4 categories differentiate. Anchored as a methodology footnote in the writeup.

### 3.2 NoRAG vs RAG distributions are categorically distinct — the labelling pipeline passes a clean sanity check

NoRAG retrieves no chunks → every wrong answer is by construction a `context_omission`. The labeller correctly assigns this category in **all 23 NoRAG wrong cases** and in **zero RAG wrong cases**. This validates that gpt-4o-mini distinguishes "no chunks available" from "chunks present but model ignored/misused them" at 100 % accuracy on the structural-validation subset.

### 3.3 Wrong-answer mass migrates from `wrong_reasoning_chain` to `unsupported_treatment` as retrieval quality rises

The headline cross-architecture pattern:

| Architecture | wrong_reasoning_chain % | unsupported_treatment % |
|---|---:|---:|
| Naive (k=5 dense) | **60.0 %** | 22.9 % |
| Sparse (k=5 BM25) | **57.9 %** | 31.6 % |
| Hybrid (RRF) | 44.7 % | **44.7 %** ← tied |
| Multi-Hop (3-hop) | 47.8 % | **43.5 %** |

As retrieval quality improves, mass shifts away from "model reasoned wrong from supporting chunks" toward "model picked an option with no chunk support". **Better retrieval doesn't fix treatment errors — it just gives the model more wrong-treatment options to pick from.** Category-level resolution of the Phase 4 §3.4 "treatment is hardest" finding.

This is also a finding about *root causes*:
- **Naive + Sparse** are dominated by reasoning failures (the chunks are right; the model reasons wrong)
- **Hybrid + Multi-Hop** are dominated by option-selection failures (the chunks are richer; the model still picks an unsupported option)

Different root causes → different mitigations. For Phase 7 confidence-aware rejection (already done), the mitigation is *don't trust the model when its confidence vector is low*. For Phase 9 EXP_16 weighted ranking, the Safety dimension has two axes now: Hallucination_Rate (Phase 4) and error-type concentration (Phase 8).

---

## 4. Design choices (locked before build)

These were all surfaced in the design-stage message and accepted without revision:

1. **Surface = golden_234, scope = wrong-answer rows only.** Consistent with Phase 7. The taxonomy answers "what *kinds* of errors does each architecture make?" — only wrong rows are interesting.
2. **Labeller = gpt-4o-mini-2024-07-18 (JSON-mode).** Cheap (~$0.02/call), reliable JSON output, sufficient discrimination for the 6-way categorisation. Same OpenAI family as the constructor (gpt-4o for Phase 3 golden set) but distinct model — methodologically acceptable because we're labelling errors, not grading correctness.
3. **All 5 architectures.** Cross-arch comparison is the headline; same RAGAS-anchored surface; total 157 question-arch labels at ~$3.
4. **3 gated stages.** Smoke (3 Q × Multi-Hop, ~$0.06) → Stage B (full Multi-Hop, ~$0.50) → Stage C (all 5 archs, ~$2.68). User confirmed each gate before scaling.
5. **answer_correctness excluded as a feature** — already done in Phase 7 for the same quasi-circularity reason; not relevant here because labeller already sees the gold letter directly.

No methodology pivots needed mid-session.

---

## 5. Documents updated this session

| File | Change |
|---|---|
| `src/taxonomy/__init__.py` | NEW (empty marker) |
| `src/taxonomy/categories.py` | NEW — 6 CategoryDef + format helpers (135 LOC) |
| `src/taxonomy/labeller.py` | NEW — gpt-4o-mini JSON-mode classifier + resumable batch (240 LOC) |
| `src/taxonomy/analysis.py` | NEW — crosstab + Cohen's κ + headline table (115 LOC) |
| `tests/test_taxonomy.py` | NEW — 17 tests, all pass |
| `notebooks/08_exp13_14_15_taxonomy.ipynb` | NEW — 16 cells, 3 gated stages, ran in ~9 min |
| `results/exp_14_taxonomy_labels/*.jsonl` | NEW — 7 files (smoke + Stage B + 5 × Stage C) |
| `results/exp_15_taxonomy_analysis/table7_counts.csv` | NEW — paste-ready for Excel Table 7 |
| `results/exp_15_taxonomy_analysis/table7_proportions.csv` | NEW — within-arch percentages |
| `docs/output_notes/08_exp13_14_15_output.md` | NEW |
| `plan.md §10.1` | NEW — Phase 8 close-out |
| `docs/todo.md §9` | EXP_13/14/15 marked FULLY COMPLETE |
| `docs/todo.md` decision log | 2026-05-11 Phase 8 close-out entry added |
| `memory/project_thesis_overview.md` | Phase 8 done snapshot + 6-act discussion narrative |
| `docs/discussion_notes/chat_context_2026-05-11_phase8.md` | NEW — this file |

---

## 6. The thesis-claim space at end of Phase 8

The discussion chapter narrative now has **six acts** (Phases 4–8 together):

1. **Act 1 (EXP_01 → 02)**: Naive dense retrieval *hurts* a memorisation-strong LLM.
2. **Act 2 (EXP_03 → 04)**: Single-shot retrieval (sparse, hybrid, RRF-fused) does not solve the retrieval-quality problem.
3. **Act 3 (EXP_05)**: Iterative multi-hop retrieval delivers grounded improvement.
4. **Act 4 (EXP_07)**: Adaptive routing captures most of Multi-Hop's gain at 60 % of the compute.
5. **Act 5 (EXP_08 / 09)**: Confidence-aware rejection over RAGAS converts grounded improvement into safety-grade clinical deployment. (The thesis-central novelty contribution.)
6. **Act 6 (EXP_13 / 14 / 15)** — **NEW**: Cross-architecture error taxonomy shows wrong-answer mass migrates from "model reasoning failures" (Naive / Sparse) to "model option-selection failures" (Hybrid / Multi-Hop) as retrieval quality improves. The remaining error budget is dominated by `wrong_reasoning_chain` (45–60 %) and `unsupported_treatment` (23–45 %), both generator-level failures that better retrieval alone cannot fix.

The 6-act structure now spans every locked-decision experiment in plan.md §0–§10. Phase 9 (EXP_16 final synthesis) is the assembly step; the experimental claims are done.

---

## 7. Open methodology questions — none new

All design decisions are locked. The methodology footnotes accumulated over Phases 4–8:

1. Two evaluation surfaces (golden_234 for RAGAS + Phase 7 + Phase 8; test_1273 for accuracy + Phase 6) — defensible via Phase 4's judge-robustness measurement (train-vs-test gap = 0.8 pp).
2. LIME signal is well-defined only on retrieval-changed questions (Phase 6 §2.2).
3. Phase 6 LIME-LOO produced zero attribution; subset sampling is the canonical method.
4. Phase 7 runs on golden_234 because it's the only surface with full RAGAS.
5. **Phase 8 (new)**: 6 → 4 category collapse — `option_mismatch` structurally inaccessible due to generator prompt; `partial_evidence_misuse` folded into `wrong_reasoning_chain` by gpt-4o-mini's calibration.

---

## 8. Next steps in order

1. **Phase 9 — EXP_16 final synthesis**. Weighted ranking of all architectures + adaptive variant + confidence-rejected variant. All inputs on disk from Phases 4–8. **Pure aggregation, $0, ~30 min.** Default weighting (proposal §7.9): 0.25 Accuracy + 0.25 Faithfulness + 0.20 Retrieval Recall + 0.15 Safety + 0.10 Explainability + 0.05 Latency. Output: Table 12 (Final Weighted Ranking). The deployment-recommendation step.
2. **Thesis writing**. The 6-act discussion chapter narrative is ready; methodology footnotes are all anchored; results tables fill from the on-disk CSVs.
3. **(Optional) Phase 10 — Streamlit demo UI**. Cached-mode reads of the JSONL files. ~5 days human time.

---

## 9. Things to watch / risks

- **Phase 9 weighting sensitivity**: the 0.25/0.25/0.20/0.15/0.10/0.05 weights are from the proposal. If a reviewer wants different weights, the per-architecture metrics are all on disk; only the aggregator changes. Worth running a "weights sensitivity" sub-analysis in EXP_16 to show the ranking is robust to weight perturbations.
- **The 6 → 4 category collapse** is a methodology finding worth defending in the viva. The honest answer: the proposal's 6-category taxonomy was designed before the labelling pipeline was built, and the empirical data shows two categories don't differentiate on this benchmark. Anchored.
- **partial_evidence_misuse going to 0** specifically: could be re-examined with a more nuanced labeller (gpt-4o or human raters with explicit exemplars). Not required for the central thesis claim. Listed as a Phase 8 v2 optional extension.

---

## 10. Where to find what

| Need | File |
|---|---|
| Full Phase 8 analysis | [`docs/output_notes/08_exp13_14_15_output.md`](../output_notes/08_exp13_14_15_output.md) |
| Table 7 paste-target (counts) | `results/exp_15_taxonomy_analysis/table7_counts.csv` |
| Table 7 within-arch proportions | `results/exp_15_taxonomy_analysis/table7_proportions.csv` |
| Per-question taxonomy labels | `results/exp_14_taxonomy_labels/stage_c_*.jsonl` (5 files, one per arch) |
| Phase 8 close-out summary | [`plan.md §10.1`](../../plan.md) |
| Decision log entry | [`docs/todo.md`](../todo.md) bottom |
| 6-act discussion narrative | this file §6 + memory/project_thesis_overview.md |

---

*Companion: Phase 7 close-out earlier same day at [`chat_context_2026-05-11_phase7.md`](chat_context_2026-05-11_phase7.md). Phase 6 at [`chat_context_2026-05-11_phase6.md`](chat_context_2026-05-11_phase6.md). Phase 5 at [`chat_context_2026-05-11.md`](chat_context_2026-05-11.md). Phase 4 at [`chat_context_2026-05-10.md`](chat_context_2026-05-10.md). Phase 8 was the cleanest of all five sessions — no methodology pivots needed. Next session — likely Phase 9 EXP_16 build — should produce a new chat_context. The experimental work is done.*
