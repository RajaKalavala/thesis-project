# Chat Context — 2026-05-10 (Phase 4 close-out)

> Working transcript of the session that closed Phase 4. Captures the cross-architecture analysis the user ran on EXP_03/04/05 RAGAS results, the falsifiable-hypothesis verdicts, and what the implications are for Phases 5–7.

---

## 1. Where we started this session and where we ended

| Item | Start of session | End of session |
|---|---|---|
| EXP_03 RAGAS | pending | ✅ ran (~$11) |
| EXP_04 baseline | ran 2026-05-07 | unchanged |
| EXP_04 RAGAS | pending | ✅ ran (~$11) |
| EXP_05 baseline | ran 2026-05-07 | unchanged |
| EXP_05 RAGAS | pending | ✅ ran (~$13) |
| Output notes 04c (RAGAS section) | absent | ✅ added |
| Output notes 04d_exp04_output.md | absent | ✅ written |
| Output notes 04e_exp05_output.md | absent | ✅ written (headline finding) |
| plan.md §6.1 close-out | absent | ✅ added |
| plan.md §14 cost line | $140–160 estimate | reconciled to ~$50 measured |
| plan.md §15 risk register | no Phase 4 results entries | added two |
| docs/todo.md decision log | last entry 2026-05-07 | added 2026-05-10 close-out |
| memory/project_thesis_overview.md | said "EXP_03 baseline only, EXP_04/05 pending" | updated to full Phase 4 done |

Cumulative project spend: **~$60** of an MSc budget (was projected $150–170; the 2026-05-06 cost over-correction has been retired).

---

## 2. The cross-architecture headline (what the user asked for the analysis on)

| Architecture | acc_test | acc_golden | F | Hall | CP | CR | AC |
|---|---:|---:|---:|---:|---:|---:|---:|
| EXP_01 No-RAG | **0.7738** | 0.9017 | n/a | n/a | n/a | n/a | 0.8738 |
| EXP_02 Naive Dense | 0.7573 | 0.8504 | 0.131 | 0.896 | 0.329 | 0.412 | 0.838 |
| EXP_03 Sparse BM25 | 0.7581 | 0.8376 | 0.040 | 0.966 | 0.081 | 0.107 | 0.838 |
| EXP_04 Hybrid RRF | 0.7659 | 0.8376 | 0.094 | 0.917 | 0.280 | 0.348 | 0.827 |
| **EXP_05 Multi-Hop** | **0.7958** | **0.9017** | **0.283** | **0.737** | **0.374** | **0.711** | **0.869** |

**One-liner**: EXP_05 Multi-Hop is the only RAG architecture in the comparison to beat No-RAG on the contamination-clean test split (+2.20 pp). Single-shot retrieval (Naive, Sparse, Hybrid) all underperform No-RAG by 0.79–1.65 pp.

---

## 3. Falsifiable hypotheses — final verdicts

These were the five hypotheses anchored across the EXP_02 and EXP_03 output notes, with thresholds locked before EXP_04/05 RAGAS ran:

| # | Hypothesis | Verdict | Result | Anchor |
|---|---|---|---|---|
| 1 | Sparse Context Precision > 0.33 (rare-term recovery) | ✗ **FALSIFIED** | 0.081 (worse than dense) | [`04b §4`](../output_notes/04b_exp02_output.md) |
| 2 | Hybrid Acuuracy on test_1273 > 0.76 | ✓ **SUPPORTED** | 0.7659 (just clears) | [`04c §4`](../output_notes/04c_exp03_output.md) |
| 3 | Hybrid Context Precision ≥ 0.50 | ✗ **FALSIFIED** | 0.280 (worse than Naive's 0.329) | [`04b §4`](../output_notes/04b_exp02_output.md) |
| 4 | Multi-Hop Acuuracy ≥ Naive's 0.7573 | ✓ **SUPPORTED** | 0.7958 (+3.85 pp) | [`04b §3.5`](../output_notes/04b_exp02_output.md) |
| 5 | Multi-Hop F > 0.05 on multi-hop subset | ✓ **SUPPORTED** | 0.229 (+18 pp threshold margin) | [`04b §3.5`](../output_notes/04b_exp02_output.md) |

**Net: 3 supported, 2 falsified.** The two falsifications are publishable findings, not failures.

---

## 4. The three findings that move the thesis

### 4.1 EXP_05 Multi-Hop is the headline architecture

Beats No-RAG on accuracy. Doubles Context Recall over Naive (0.71 vs 0.41). Lifts median Faithfulness above zero (median 0.25 vs 0.000 for Naive/Sparse/Hybrid). 28.4 % of correct answers reach F ≥ 0.5 (vs 4–12 % elsewhere). Net +28 questions vs No-RAG on test_1273 — the only architecture with a positive net (EXP_02/03/04 all net negative). On the 13 multi-hop golden questions where Naive scored F=0.000, Multi-Hop scored 0.229.

Multi-Hop also wins on the hardest categories: treatment +9 pp vs Naive (0.824 vs 0.735), mechanism +11 pp (0.917 vs 0.806). It's *faster* than Hybrid on test_1273 (57 min vs 123 min) because the BM25 scoring inside Hybrid dominates wall-time.

### 4.2 The decoupling result — EXP_03 Sparse

EXP_03 Sparse retrieval has **CP=0.081** (4× lower than Naive's 0.329) yet Acuuracy=0.7581 (within 0.08 pp of Naive). **If retrieval quality drove answer quality, accuracy should collapse on EXP_03; instead it ties EXP_02 to within a single question.** This is the strongest single piece of evidence in Phase 4 that the LLM is answering from pre-training memorisation, not from retrieved context.

This reframes the central thesis claim. The contamination story (EXP_01) plus the decoupling result (EXP_03) plus Multi-Hop's grounded improvement (EXP_05) gives a clean three-act discussion-chapter narrative.

### 4.3 The Hybrid RRF counter-result

EXP_04 Hybrid clears the accuracy hypothesis (0.7659 > 0.76 threshold) but **falsifies** the Context Precision hypothesis (0.280 < 0.50, and *worse* than Naive's 0.329). Mechanism: RRF fuses dense (CP=0.33) with sparse (CP=0.08); sparse's near-random precision contaminates the fused top-5. This is a publishable counter-result for the medical-RAG literature: **RRF requires both retrievers to clear a precision floor** — a weak partner degrades the fused output.

The "complementarity finding" from EXP_03 (153 of 1,273 test questions disagree between dense and sparse, 50/50 right/wrong) turns out to be a measurement artefact: two retrievers each producing mostly-noise output will look "complementary" on disagreement counts.

---

## 5. Implications for downstream phases

### 5.1 Phase 5 EXP_07 adaptive routing — proposal split is under review

Coverage analysis on test_1273:
- Best single fixed: EXP_05 Multi-Hop = 0.7958
- Oracle on No-RAG ∪ Multi-Hop alone = **0.8531** (+5.7 pp over standalone)
- Oracle on all 5 architectures = 0.8696

**The Hybrid lane does not justify its place in the routing table.** EXP_07 should test:
- **Routing variant A** (proposal): Simple → Naive, Moderate → Hybrid, Complex → Multi-Hop.
- **Routing variant B** (data-driven): Simple → No-RAG, Complex → Multi-Hop. Skip Hybrid.

A binary router may capture most of the oracle gain at lower compute. Decision rule: pick whichever variant minimises wall-time × cost while clearing 0.8 on test_1273.

### 5.2 Phase 7 confidence-aware rejection — Multi-Hop is the only architecture with usable F signal

| Architecture | median F (correct rows) | % grounded (F≥0.5) |
|---|---:|---:|
| EXP_02 Naive | 0.000 | 11.6 % |
| EXP_03 Sparse | 0.000 | 4.1 % |
| EXP_04 Hybrid | 0.000 | 8.7 % |
| **EXP_05 Multi-Hop** | **0.250** | **28.4 %** |

The threshold sweep planned in EXP_09 ({0.5, 0.6, 0.7, 0.8, 0.9}) has signal on Multi-Hop. On the others, median F=0 means thresholding is unstable; if the rejection layer is run on Naive/Sparse/Hybrid at all, a {0.1, 0.2, 0.3} regime is more informative.

### 5.3 Phase 6 LIME / SHAP

Multi-Hop is the highest-value architecture to explain — Faithfulness has graded signal there. LIME / SHAP on Naive/Sparse/Hybrid will mostly explain why the LLM ignored the retrieved chunks (the retrieval was noise) — interesting but already covered by the Faithfulness distribution analysis above.

---

## 6. Documents updated in this session

| File | Change |
|---|---|
| `docs/output_notes/04c_exp03_output.md` | Added §2.5–2.7 RAGAS analysis (catastrophic CP=0.08 yet unchanged accuracy), §4 conclusions rewritten for full RAGAS, file-produced section updated |
| `docs/output_notes/04d_exp04_output.md` | NEW — full EXP_04 analysis including the falsified CP≥0.50 hypothesis as a publishable counter-result |
| `docs/output_notes/04e_exp05_output.md` | NEW — full EXP_05 analysis as the headline finding (only RAG to beat No-RAG, dominates every RAGAS metric) |
| `plan.md §6.1` | Added Phase 4 close-out with cross-architecture table + hypothesis verdicts + downstream implications |
| `plan.md §14` | Cost line for Phase 4 RAGAS reconciled $140–160 → ~$50 measured; total project ~$150–170 → ~$60 |
| `plan.md §15` | Two risk-register entries added marking Hybrid CP failure and the 1.6 pp single-shot-retrieval underperformance as resolved findings |
| `docs/todo.md §5` | EXP_03/04/05 marked fully complete with RAGAS results inline |
| `docs/todo.md` decision log | Added 2026-05-10 entry summarising the entire close-out |
| `memory/project_thesis_overview.md` | Updated Phase 4 status; added cross-arch table, hypothesis verdicts, downstream implications, cost reconciliation |
| `docs/discussion_notes/chat_context_2026-05-10.md` | NEW — this file |

---

## 7. Open methodology questions — none new

The two open questions from [`chat_context_2026-05-07.md`](chat_context_2026-05-07.md) §3 are still open, both unaffected by Phase 4 close-out:

1. **Two evaluation surfaces** (test_1273 accuracy + golden_234 RAGAS). The judge-AC train-vs-test gap (Δ=1.6 pp) is well below the contamination signal (Exact Match Δ=10.6 pp) → judge is contamination-robust → defensible with footnote (Option A). Building a 150-row test-only golden subset (Option B, ~$15) remains the gold standard if a reviewer pushes harder.

2. **LIME / SHAP cost** (Phase 6). Confirmed $0 via Groq for the perturbation re-generations (~42k Groq calls, bounded by free tier). The SHAP-with-Faithfulness-as-score variant ($320) is explicitly avoided.

---

## 8. Next steps in order

1. **User commits** the Phase 4 close-out edits (output notes + plan.md + todo.md + memory + this context file).
2. **Phase 5 EXP_06 — rule-based question complexity classifier** ([plan.md §7](../../plan.md)). Output: `data/processed/complexity_labels.parquet`. Manual review of 100 random labels to validate ≥ 80 % rater agreement.
3. **Phase 5 EXP_07 — adaptive routing**. Implement both routing-table variants (proposal three-way + binary No-RAG/Multi-Hop) and run on test_1273. Score-join RAGAS metrics from the underlying architectures' golden_234 — no new judge calls.
4. **Phase 6 EXP_10/11/12 — LIME, SHAP, agreement** at the passage level. Sample 200 questions per architecture; all Groq → $0 (~12–24 h wall-time bounded by free tier). Multi-Hop is the highest-value architecture to explain.
5. **Phase 7 EXP_08/09 — confidence signals + rejection threshold sweep**. Pure aggregation over Phase 4 + Phase 6 outputs; no LLM calls. Threshold sweep on Multi-Hop primarily.
6. **Phase 8 EXP_13/14/15 — hallucination taxonomy** (~$3 GPT-4o-mini classifier-assisted).
7. **Phase 9 EXP_16 — final synthesis**.
8. **(Optional) Phase 10 — Streamlit demo UI**.

---

## 9. Where to find what (Phase 4 done snapshot)

| Need | File |
|---|---|
| Headline thesis numbers | [`plan.md §6.1`](../../plan.md) cross-architecture table |
| EXP_05 Multi-Hop full analysis (the headline) | [`docs/output_notes/04e_exp05_output.md`](../output_notes/04e_exp05_output.md) |
| Hybrid RRF counter-result | [`docs/output_notes/04d_exp04_output.md`](../output_notes/04d_exp04_output.md) |
| Sparse decoupling result | [`docs/output_notes/04c_exp03_output.md` §2.5](../output_notes/04c_exp03_output.md) |
| Memorisation smoking-gun (88 % ungrounded correct) | [`docs/output_notes/04b_exp02_output.md` §3.1](../output_notes/04b_exp02_output.md) |
| Contamination evidence (10.6 pp train-vs-test gap) | [`docs/output_notes/04a_exp01_output.md`](../output_notes/04a_exp01_output.md) + `results/exp_01_base_llm__full_12723/README_LEGACY.md` |
| Falsifiable hypothesis verdicts | [`plan.md §6.1`](../../plan.md) verdict table |
| Decision log | [`docs/todo.md`](../todo.md) bottom |
| Cost reconciliation | [`plan.md §14`](../../plan.md) row "Phase 4 RAGAS judge" |

---

*Companion: previous transcript at [`chat_context_2026-05-07.md`](chat_context_2026-05-07.md). Next session — likely Phase 5 EXP_06 build — should produce `chat_context_2026-05-XX.md` with the complexity-classifier design + initial label distribution.*
