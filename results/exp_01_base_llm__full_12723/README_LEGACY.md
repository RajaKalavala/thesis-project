# LEGACY — full-12,723 EXP_01 run

**Status**: archived 2026-05-06. **Not** the canonical artifact for EXP_01.

## Why kept

This directory holds EXP_01's run on the full 12,723-question MedQA US corpus (train + dev + test). It's preserved because it's the **empirical anchor for the contamination story** that drove the 2026-05-06 evaluation-surface narrowing — *"train + dev = 0.880 vs test = 0.774, 10.6 pp gap"*. Without this run the methodology footnote ("evaluation narrowed to test split because the LLM memorised train+dev") has no observed evidence behind it.

## What replaces it

The canonical artifact for EXP_01 is now [`../exp_01_base_llm__test_1273/`](../exp_01_base_llm__test_1273/) — derived 2026-05-06 by filtering this directory's predictions to `split == 'test'` rows. Same predictions, different aggregate. **Headline `Acuuracy = 0.7738`** over 1,273 rows.

## Provenance

- Run date: 2026-05-05
- Wall time: 58 min on M1 Pro · Groq (LLaMA 3.3 70B)
- Cost: $0 (Groq free tier)
- Acuuracy: 0.8693 (11,060 / 12,723) — *the contaminated number*

## Don't

- Don't paste this `summary.json`'s Acuuracy into the Excel `Results Table` — that surface is no longer canonical.
- Don't run RAGAS against this directory's `predictions.jsonl` — RAGAS surface is `golden_234`, independent of this.
- Don't delete unless you also rewrite the methodology chapter footnote that cites the 10.6 pp contamination gap.

## See also

- [plan.md §0 #8](../../plan.md#0-locked-decisions) — the locked evaluation surface decision
- [docs/dataset.md §4](../../docs/dataset.md#4-two-evaluation-surfaces--test-split--golden-subset) — the why
- [docs/output_notes/04a_exp01_output.md §4.4](../../docs/output_notes/04a_exp01_output.md) — stratified breakdown that surfaced the contamination
- [docs/todo.md decision log](../../docs/todo.md) — 2026-05-06 entry covering the narrow-to-test-split decision
