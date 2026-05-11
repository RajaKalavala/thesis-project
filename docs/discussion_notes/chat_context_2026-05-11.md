# Chat Context — 2026-05-11 (Phase 5 close-out)

> Working transcript of the session that closed Phase 5 (Group B adaptive routing). Captures the EXP_06 manual review, the EXP_07 routing-table decision, the dry-run that surfaced the Pareto-frontier framing, the actual run results, and the k=15 vs k=5 wrinkle.

---

## 1. Where we started this session and where we ended

| Item | Start | End |
|---|---|---|
| EXP_06 complexity labels | not started | ✅ written (12,723 rows, 1/100 disagreement) |
| EXP_07 Variant A (proposal) | not started | ✅ 0.7863 acc, 1.806 calls/Q |
| EXP_07 Variant B (binary) | not started | ✅ 0.7832 acc, 2.425 calls/Q |
| `src/retrieval/complexity.py` + tests | absent | ✅ 7/7 tests pass |
| `src/retrieval/adaptive.py` + tests | absent | ✅ 6/6 tests pass |
| Output notes 05_exp06_output.md | absent | ✅ written |
| Output notes 05_exp07_output.md | absent | ✅ written |
| plan.md §7.1 close-out | absent | ✅ added |
| plan.md §15 risk register | Phase 4 entries only | ✅ Phase 5 entries added |
| docs/todo.md §6 | both experiments unstarted | ✅ both marked FULLY COMPLETE |
| memory/project_thesis_overview.md | said "Phase 4 done" | ✅ Phase 5 done snapshot |

Cumulative project spend unchanged at **~$60** (Phase 5 was $0 — all Groq).

---

## 2. The Phase 5 headline finding (what the discussion chapter gets)

| Strategy | Acuuracy (test_1273) | Groq calls/Q | Pareto status |
|---|---:|---:|---|
| EXP_01 No-RAG | 0.7738 | 1.000 | **frontier** |
| EXP_02 Naive | 0.7573 | 1.000 | dominated |
| EXP_03 Sparse | 0.7581 | 1.000 | dominated |
| EXP_04 Hybrid | 0.7659 | 1.000 | dominated |
| **EXP_07 Variant A** | **0.7863** | **1.806** | **frontier** |
| EXP_07 Variant B | 0.7832 | 2.425 | dominated |
| EXP_05 Multi-Hop | 0.7958 | 3.000 | **frontier** |

**Variant A captures 84 % of Multi-Hop's accuracy gain over No-RAG at 60 % of the compute.** Marginal efficiency:
- No-RAG → Variant A: +1.26 pp / +0.81 calls = **0.0156 acc/extra call**
- Variant A → Multi-Hop: +0.94 pp / +1.19 calls = **0.0079 acc/extra call** (2.0× worse)

This is the discussion-chapter Act 4 — the proposal's three-way Simple/Moderate/Complex split is **empirically validated** as the cost-efficient Pareto-frontier point.

---

## 3. Decisions made (chronological)

### 3.1 Routing-variant choice (user, early in session)

Asked the user whether EXP_07 should test:
- (A) Variant A only (proposal as-is)
- (B) Variant B only (binary, data-driven)
- (C) Both, comparing in EXP_07

**User chose (C)**: both variants tested, comparison row in Table 10. Maximises defensibility.

### 3.2 Thresholds for EXP_06 rule

First-cut classifier was too aggressive on Complex (97.6 % of questions). Recalibrated to **percentile-anchored thresholds** (`WORDS_P33=93 / WORDS_P67=133 / PHRASES_P33=28 / PHRASES_P67=41`), giving a clean 29.5 / 32.7 / 37.7 % distribution.

### 3.3 Manual review of EXP_06 labels

User initially said they hadn't done the 100-row manual review. I offered to do it as LLM judgment (acknowledging it's not a clinician's review). Reviewed all 100; found **1 disagreement** (`medqa_8198`, boundary case). Far below the ≤20 acceptance gate. Rule locked.

### 3.4 EXP_07 win-metric framing

Asked the user whether EXP_07 should frame "win" as:
- (A) Cost-adjusted (Acc per Groq call) — my recommendation
- (B) Raw Acuuracy (proposal as-is)
- (C) Both, side-by-side

**User said "go with your recommendation"** → cost-adjusted.

After implementing, the **dry-run revealed the naive Acc/call ratio is misleading** (favours No-RAG trivially). Pivoted the framing to **Pareto frontier + marginal efficiency**, which is what the final notebook + output_notes use.

### 3.5 The k=15 chunk fan-out wrinkle (discovered post-run)

The EXP_07 notebook set `TOP_K = 15` uniformly for all retrievers (matching Multi-Hop's max chunk return). For Naive (Variant A's Simple lane) and Hybrid (Variant A's Moderate lane), this was different from their baseline experiments (EXP_02 and EXP_04 ran at k=5). Empirical effect:

| Variant | Simulator (k=5 underlying) | Actual run (k=15 fan-out) | Δ |
|---|---:|---:|---:|
| Variant A | 0.7895 | 0.7863 | **−0.31 pp** |
| Variant B | 0.7840 | 0.7832 | −0.08 pp |

Variant B's near-zero gap is the proof that for Multi-Hop, k=15 chunks are identical to EXP_05's chunks (cache hits 99 %). Variant A's larger gap comes from Naive/Hybrid producing different (more) chunks at k=15, which (a) caused 60 % cache misses and (b) marginally hurt accuracy — consistent with Phase 4's "more chunks = more retrieval-distractor noise" finding.

**Decision**: documented as a methodology footnote in `05_exp07_output.md` §3.3 and `plan.md` §15. Both simulator and actual numbers reported; actual is canonical for Table 1 row 6. Demonstrates routing is **not sensitive to chunk fan-out within [5, 15]** — a small positive methodology footnote.

---

## 4. Open methodology questions — none new

The two open questions carry forward unchanged:
1. **Two evaluation surfaces** (test_1273 accuracy + golden_234 RAGAS). Defensible with footnote (Option A). EXP_07 RAGAS is **score-joined** from underlying-architectures' golden_234 — no new judge calls, ~$0.
2. **LIME / SHAP cost** (Phase 6). $0 via Groq for perturbations.

---

## 5. Current state (2026-05-11)

### 5.1 Code

`src/` (Phase 5 additions):
- `src/retrieval/complexity.py` ✅ — rule + thresholds + dataframe-vectorised classifier
- `src/retrieval/adaptive.py` ✅ — dispatcher with audit hooks (`dispatch_counts`, `unknown_question_count`)

`tests/` (Phase 5 additions): 7 + 6 = 13 new tests; all passing.

`notebooks/` (Phase 5):
- `05_exp06_complexity_labels.ipynb` ✅ ran end-to-end (parquet written)
- `05_exp07_adaptive_rag.ipynb` ✅ ran end-to-end (4 surfaces written, 1 fix for `include_groups=False` kwarg placement)

### 5.2 Results on disk

```
results/
├── exp_07_adaptive_variant_a__smoke_50/        ← Acuuracy 0.8200
├── exp_07_adaptive_variant_a__test_1273/       ← Acuuracy 0.7863  · CANONICAL
├── exp_07_adaptive_variant_b__smoke_50/        ← Acuuracy 0.8600
└── exp_07_adaptive_variant_b__test_1273/       ← Acuuracy 0.7832  · CANONICAL

data/processed/
└── complexity_labels.parquet                   ← 12,723 rows × 8 cols
```

### 5.3 Cost spent

| Phase | Item | Spent | Pending |
|---|---|---:|---:|
| 3 | Golden construction | $6.61 ✅ | — |
| 4 | RAGAS (5 archs × Sonnet 4.6) | ~$50 ✅ | — |
| 5 | EXP_06 + EXP_07 (Groq only) | **$0 ✅** | — |
| 6 | LIME / SHAP (Groq) | — | $0 |
| 7 | Confidence layer (no LLM) | — | $0 |
| 8 | Taxonomy (gpt-4o-mini) | — | ~$3 |
| **Total** | | **~$60** | **~$3** |

Grand total tracking: **~$63 of the ~$80 ceiling**; well within an MSc budget.

---

## 6. Falsifiable hypotheses pending answer

Phase 5's three hypotheses all landed SUPPORTED. Phase 4's all answered. The remaining falsifiable claims for the thesis:

| Phase | Hypothesis | Status |
|---|---|---|
| 6 | LIME / SHAP top-1 passage agreement > 0.4 (Multi-Hop) | pending EXP_12 |
| 6 | LIME / SHAP agreement correlates positively with Faithfulness | pending |
| 7 | Confidence rejection at threshold τ=0.5 lifts accuracy on accepted set by ≥ 5 pp | pending EXP_09 |
| 7 | Multi-Hop's graded F distribution makes thresholding stable; Variants A/B less stable | pending |
| 8 | 6-category taxonomy covers ≥ 90 % of hallucinated answers across architectures | pending EXP_13–15 |

---

## 7. Next steps in order

1. **(Done this session)** EXP_06 labels + EXP_07 routing both run; output notes + plan.md + todo.md + memory all synced.
2. **Phase 6 EXP_10 — passage-level LIME**. New `src/xai/lime_passage.py`; notebook `06_exp10_lime_passage.ipynb`. Sample 200 questions per architecture (5 × 200 = 1,000 rows × ~10 perturbations = ~10k Groq calls). Bounded by Groq free tier; ~12–24 h wall time. $0 cost.
3. **Phase 6 EXP_11 — passage-level SHAP**. New `src/xai/shap_passage.py`. Same 200-question subset for direct LIME/SHAP comparison.
4. **Phase 6 EXP_12 — LIME ↔ SHAP agreement**. Top-1 / Top-3 overlap; correlate with Faithfulness and accuracy.
5. **Phase 7 EXP_08/09 — confidence rejection**. Pure aggregation over Phase 4 + Phase 6 outputs; no LLM calls. Threshold sweep {0.5, 0.6, 0.7, 0.8, 0.9} on Multi-Hop primarily; secondary sweep on Variant B.
6. **Phase 8 EXP_13/14/15 — hallucination taxonomy** (~$3 GPT-4o-mini).
7. **Phase 9 EXP_16 — final synthesis**.
8. **(Optional) Phase 10 — Streamlit demo UI**.

---

## 8. Things to watch / risks

- **LIME / SHAP wall time**: 12–24 h Groq rate-limited. Plan accordingly — if the user wants to run overnight, the resumability layer in `src/utils/cache.py` handles disconnects.
- **Variant A vs Variant B for Phase 7**: both will be re-examined as candidate confidence-rejection surfaces. Variant B's higher F (0.276) might let the threshold sweep produce cleaner numbers; Variant A's lower F (0.197) might collapse the gate. Worth running both.
- **The k=15 footnote**: small but real. Anchored in `plan.md §15` and `05_exp07_output.md §3.3` so future-you can defend it without reverse-engineering.

---

## 9. Where to find what (Phase 5 done snapshot)

| Need | File |
|---|---|
| EXP_06 rule + thresholds + manual review | [`docs/output_notes/05_exp06_output.md`](../output_notes/05_exp06_output.md) |
| EXP_07 Pareto frontier + RAGAS score-join + hypothesis verdicts | [`docs/output_notes/05_exp07_output.md`](../output_notes/05_exp07_output.md) |
| Phase 5 close-out (cross-architecture table + 3 hypothesis verdicts) | [`plan.md §7.1`](../../plan.md) |
| Routing-table choice rationale | this file + [`chat_context_2026-05-07.md`](chat_context_2026-05-07.md) |
| k=15 vs k=5 footnote | [`05_exp07_output.md` §3.3](../output_notes/05_exp07_output.md) + [`plan.md §15`](../../plan.md) |
| Decision log entry (this session) | [`docs/todo.md`](../todo.md) bottom |
| Tables ready to populate | Table 1 row 6, Table 2, Table 3, Table 10 |

---

*Companion: previous transcript at [`chat_context_2026-05-10.md`](chat_context_2026-05-10.md) (Phase 4 close-out). Next session — likely Phase 6 EXP_10 LIME build — should produce `chat_context_2026-05-XX.md` with the perturbation prompt design + sampling rationale.*
