# Chat Context — 2026-05-11 (Phase 6 close-out)

> Working transcript of the second session on 2026-05-11, which closed Phase 6 (passage-level explainability — EXP_10 / EXP_11 / EXP_12). The Phase 5 close-out from earlier the same day is at [`chat_context_2026-05-11.md`](chat_context_2026-05-11.md). Captures the two methodology pivots, the signal-density results, and the LIME-SHAP agreement findings.

---

## 1. Where we started and where we ended

| Item | Start of session | End of session |
|---|---|---|
| EXP_10 LIME (LOO) | not started | ✅ built + smoke ran (all-zero — STRUCTURAL verdict) |
| EXP_10 LIME (subset-sampling) | not designed | ✅ canonical method, 78.5 % signal density on retrieval-changed subset |
| EXP_11 SHAP | not started | ✅ ran on Stage B data with No-RAG anchor (90.2 % signal density), $0 |
| EXP_12 agreement | not started | ✅ ρ = 0.63 across 134 retrieval-changed questions, $0 |
| `src/xai/lime_passage.py` | absent | ✅ 450 LOC, 10/10 tests pass |
| `src/xai/shap_passage.py` | absent | ✅ 220 LOC, 7/7 tests pass |
| `src/xai/agreement.py` | absent | ✅ 175 LOC, 9/9 tests pass |
| Output notes 06_exp10_11_12_output.md | absent | ✅ written |
| plan.md §8.1 close-out | absent | ✅ added |
| docs/todo.md §7 EXP_10/11/12 | unstarted placeholders | ✅ marked FULLY COMPLETE |
| memory/project_thesis_overview.md | said "Phase 5 done" | ✅ Phase 6 done snapshot |

Cumulative project spend unchanged at **~$60** (Phase 6 was $0 — all Groq free tier).

---

## 2. The two methodology pivots (the hard-won learnings)

### Pivot 1: LIME-LOO → subset-sampling LIME

**The setup**: original `passage_loo_lime` ran leave-one-out ablation. For each question with k chunks, build the full prompt + k LOO prompts (remove one chunk each). Per-passage attribution = score_full − score_loo_i. Standard LIME at first order.

**What broke**: 3-question smoke on Multi-Hop (k=15) showed **all-zero attribution**. Every LOO produced the same letter as the full prediction. No signal to attribute.

**Diagnostic** (Stage A2): same 3 questions on Naive (k=5). With each chunk carrying ~20 % of the evidence (vs Multi-Hop's ~7 %), if Multi-Hop's redundancy was the cause, Naive should show signal. **It didn't — Naive also all-zero on these 3 questions.** Verdict: STRUCTURAL, not Multi-Hop-specific.

**Why**: Multi-Hop's RAGAS Faithfulness median is 0.25 (from EXP_05). Most "correct" rows are *partially* grounded across multiple chunks — no single chunk is essential. LOO can't attribute distributed signal.

**Fix**: `passage_subset_lime` with N=16 random binary masks (each chunk in/out with p=0.5) + ridge regression on the mask matrix. A chunk that's consistently present in subsets where the LLM picks the full prediction gets a positive ridge coefficient, even if no single LOO flips the answer.

**Both methods kept side-by-side** in `src/xai/lime_passage.py` for the methodology-comparison row in the thesis writeup.

### Pivot 2: random 200-sample → retrieval-changed 205

**The setup**: subset-sampling smoke (Stage A3) on the same 3 random questions. Still all zeros.

**Cross-architecture check**: looked up each smoke question's prediction across all 5 architectures.
- Q1 (medqa_11615): every architecture predicts B (gold=B). **Memorisation-only** — chunks didn't drive the answer.
- Q2 (medqa_12517): same, all C correct. **Memorisation-only**.
- Q3 (medqa_11531): No-RAG=A correct; *every* RAG=C wrong. **Retrieval-distractor consensus** — chunks drove the answer to C but no single chunk dominated.

**Insight**: LIME has nothing to attribute on memorisation or distractor-consensus cases. The right test subset is questions where retrieval *demonstrably* changed the LLM's answer (No-RAG_pred ≠ MultiHop_pred). Phase 4 identified 174 such questions on test_1273 (101 fixes + 73 breaks). I added 31 "both-wrong-different-letters" to get 205 total.

**Stage A4** (3 retrieval-fixed questions): **signal found**, coefficients in [−0.5, +0.5], directional sense respected. Methodology validated.

**Stage B canonical run**: 205 retrieval-changed Multi-Hop questions × 16 subset samples = 3,280 Groq calls (mostly fresh, 15.2 % cache hits). 24 min wall time, $0.

---

## 3. The Stage B headline results

**Signal density**: 65.4 % correctness / 78.5 % same-letter. Mean top |coef| = 0.595. Attribution coefficients span [−0.5, +0.5].

**Coefficient signs respect causality** (the empirical proof of correctness):

| Subset | n signal | Top coef positive | Top coef negative |
|---|---:|---:|---:|
| Fixes (chunks help) | 72 | **58 (80 %)** | 14 (19 %) |
| Breaks (chunks hurt) | 51 | 17 (33 %) | **34 (67 %)** |

The sign flips between fix and break subsets exactly as a real causal attribution should. Noise would give symmetric distributions.

**Retrieval rank vs LLM influence — publishable counter-result**:
- Top-influence chunk's mean rank = **5.05** (out of 11.8 mean chunks).
- Rank-0 (highest BGE/RRF retrieval score) is the top-influence chunk only **13.4 %** of the time.
- **The chunk the retriever ranks first is not the chunk that drives the LLM's answer on Multi-Hop.** Counter to the standard "trust the top retrieved chunk" assumption in medical RAG.

---

## 4. EXP_11 + EXP_12 — $0 sequels

**Key efficiency property**: KernelSHAP and LIME both fit linear models on `(mask, score)` pairs. They differ only in *how the samples are weighted*. So EXP_11 runs on the *exact same data* as EXP_10 Stage B, no new Groq calls.

**EXP_11 KernelSHAP**:
- SHAP kernel weights: `w(S) = (k-1) / [C(k,|S|) · |S| · (k-|S|)]`
- **No-RAG anchor**: synthetic all-zeros sample with EXP_01's prediction + 1e6 weight (forces the intercept constraint).
- Result: signal density **90.2 % correctness / 100 % same-letter**. Lifts LIME's 65/78 % via the anchor.
- Wall time: 0.1 sec total.

**EXP_12 LIME ↔ SHAP agreement**:
- Per-question: top-1 (same argmax), top-3 overlap, Spearman ρ on both signals.
- **Top-1 agreement**: 51.5 % (correctness, n=134) / 47.2 % (sameletter, n=161).
- **Top-3 overlap**: mean 0.556 / 0.504.
- **Spearman ρ**: mean **0.632 / 0.653**; median 0.706 / 0.734.
- **Distribution (correctness signal)**: 68 strong (>0.7), 46 moderate, 18 weak, 2 anti-correlation.

**Interpretation**: LIME and SHAP agree strongly at the rank level (ρ ≈ 0.63) but disagree on the *single* top-1 chunk ~50 % of the time. Both are noisy point-estimates of the same underlying causal signal. Useful as a **combined** confidence signal for Phase 7, not perfectly interchangeable.

**Stratified by change-type**:

| Type | n | Corr top-1 | Corr ρ | Same top-1 | Same ρ |
|---|---:|---:|---:|---:|---:|
| Fix | 101 | 0.514 | 0.605 | 0.514 | 0.605 |
| Break | 73 | 0.569 | 0.656 | 0.475 | **0.732** |
| Both wrong | 31 | 0.273 | **0.707** | 0.357 | 0.602 |

**Agreement is highest on "break" questions** — both methods most strongly agree about which chunks distracted the LLM. This is exactly where Phase 7 confidence-aware rejection wants the strongest signal (to flag confidently-wrong answers).

---

## 5. Documents updated this session

| File | Change |
|---|---|
| `src/xai/__init__.py` | NEW (empty marker) |
| `src/xai/lime_passage.py` | NEW — LOO + subset-sampling LIME (450 LOC) |
| `src/xai/shap_passage.py` | NEW — KernelSHAP from LIME data (220 LOC) |
| `src/xai/agreement.py` | NEW — top-1 / top-3 / Spearman (175 LOC) |
| `tests/test_lime_passage.py` | NEW — 10 tests, all pass |
| `tests/test_shap_passage.py` | NEW — 7 tests, all pass |
| `tests/test_agreement.py` | NEW — 9 tests, all pass |
| `notebooks/06_exp10_lime_passage.ipynb` | NEW — 22 cells, 4 smoke stages + Stage B canonical + obsolete Stage C |
| `notebooks/06_exp11_exp12_shap_agreement.ipynb` | NEW — 18 cells, SHAP + agreement (no Groq) |
| `results/exp_10_lime_passage/` | NEW dir, 5 JSONLs (smoke audit trail + canonical Stage B) |
| `results/exp_11_shap_passage/` | NEW dir, canonical SHAP output |
| `results/exp_12_agreement/` | NEW dir, canonical agreement output |
| `docs/output_notes/06_exp10_11_12_output.md` | NEW — combined Phase 6 output notes |
| `plan.md §8.1` | NEW — Phase 6 close-out |
| `docs/todo.md §7` | EXP_10/11/12 marked FULLY COMPLETE with method-pivot history |
| `docs/todo.md` decision log | 2026-05-11 Phase 6 close-out entry added |
| `memory/project_thesis_overview.md` | Phase 6 done snapshot |
| `docs/discussion_notes/chat_context_2026-05-11_phase6.md` | NEW — this file |

---

## 6. What changed in the thesis claim space

The Phase 6 close-out **adds three thesis-publishable findings** to the discussion chapter:

1. **LIME-LOO is structurally inadequate for distributed grounding** (methodology contribution). Single-chunk LOO ablation produces all-zero attribution when retrieved chunks collectively support the answer rather than depending on any one. The subset-sampling pivot is the methodologically correct alternative.

2. **LIME signal is well-defined only on retrieval-changed questions** (methodology contribution). On memorisation-only cases (where the LLM gets the answer without chunks) and retrieval-distractor consensus cases (where all chunks agree on the wrong answer), per-chunk attribution is necessarily zero. Reporting LIME on the full test set without this filter wastes compute and inflates the "no-signal" denominator.

3. **Retrieval rank decouples from LLM influence** (substantive medical-RAG counter-result). The chunk the retriever ranks first is the top-influence chunk only 13 % of the time on Multi-Hop. Medical-RAG systems that surface only the top retrieved chunk for grounding are reading the wrong signal.

Plus a **methodological proof-of-correctness** (the coefficient-sign asymmetry across fix vs break subsets) and a **complementarity result** (LIME and SHAP agree at the rank level but not on top-1, supporting their use as combined confidence signals).

---

## 7. Next steps in order

1. **(Done this session)** Phase 6 EXP_10/11/12 complete; output notes + plan.md + todo.md + memory all synced.
2. **Phase 7 — Confidence-aware rejection** ([`07_exp08_confidence_signals.ipynb` + `07_exp09_confidence_rejection.ipynb`](../../notebooks/)).
   - Assemble per-question confidence vector: Faithfulness (Phase 4) + retrieval-quality (Phase 4) + LIME-SHAP-agreement (Phase 6).
   - Sweep rejection thresholds {0.5, 0.6, 0.7, 0.8, 0.9}.
   - Report accuracy-on-accepted vs rejection-rate trade-off.
   - Pure aggregation; no LLM calls. Cost: $0.
3. **Phase 8 — Hallucination taxonomy** (~$3 GPT-4o-mini classifier-assisted).
4. **Phase 9 — Final synthesis** (EXP_16 weighted ranking).
5. **(Optional) Phase 10** — Streamlit demo UI.

---

## 8. Things to watch / risks

- **Phase 6 only ran on Multi-Hop**. Extending to Naive (k=5, 149 retrieval-changed Q) and Hybrid (k=5, 154 retrieval-changed Q) is straightforward (~5 min Groq each) but not strictly needed for the central thesis claim. Multi-Hop is the high-value architecture (the only one on the Pareto frontier above No-RAG). If a reviewer asks for cross-architecture XAI, this is the easy extension.
- **The k=15 vs k=5 wrinkle from EXP_07** is unrelated to Phase 6 — EXP_10 uses each architecture's own retrieved chunks directly, no resizing.
- **Phase 7 needs to handle missing agreement scores cleanly**. Of the 205 retrieval-changed questions, 134 have correctness agreement and 161 have sameletter agreement (the rest are zero-variance for that signal). The confidence vector must use NaN-safe aggregation.

---

## 9. Where to find what

| Need | File |
|---|---|
| Full Phase 6 analysis | [`docs/output_notes/06_exp10_11_12_output.md`](../output_notes/06_exp10_11_12_output.md) |
| Methodology pivot history | this file §2 + EXP_10 notebook smoke-stages audit trail |
| Stage B canonical output | `results/exp_10_lime_passage/stage_b_retrievalchanged_mhop.jsonl` |
| SHAP canonical output | `results/exp_11_shap_passage/stage_b_retrievalchanged_mhop.jsonl` |
| Agreement canonical output | `results/exp_12_agreement/stage_b_retrievalchanged_mhop.jsonl` |
| Phase 6 close-out summary | [`plan.md §8.1`](../../plan.md) |
| Decision log entry | [`docs/todo.md`](../todo.md) bottom |
| Per-question agreement scores (for Phase 7) | `results/exp_12_agreement/stage_b_retrievalchanged_mhop.jsonl` |

---

*Companion: Phase 5 close-out earlier same day at [`chat_context_2026-05-11.md`](chat_context_2026-05-11.md). Previous: [`chat_context_2026-05-10.md`](chat_context_2026-05-10.md) (Phase 4 close-out). Next session — likely Phase 7 confidence-rejection build — should produce a new chat_context.*
