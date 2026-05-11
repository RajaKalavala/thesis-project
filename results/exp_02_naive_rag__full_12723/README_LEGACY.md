# LEGACY — incomplete EXP_02 partial full-12,723 run

**Status**: abandoned 2026-05-06. **Not** the canonical artifact for EXP_02.

## What's here

A partial run of EXP_02 Naive RAG on the full 12,723 corpus that was **stopped at row 4,844 / 12,723** (~38 % completion). No `summary.json` was written because the run didn't finish.

## Why abandoned

Per the 2026-05-06 evaluation-surface lock ([plan.md §0 #8](../../plan.md#0-locked-decisions)), Phase 4 evaluates on the **test split (1,273)**, not the full corpus. Continuing this run is no longer methodologically aligned.

**Cache value to the new test-split run: zero.** MedQA's 4-option parquet is segregated as train (rows 0–10,177) → dev (10,178–11,449) → test (11,450–12,722). The first 4,844 rows hit by this run are 100 % train rows; no test rows were reached. So when EXP_02's canonical `test_1273/` run fires, the Groq cache won't hit any of the work done here.

## What replaces it

The canonical artifact for EXP_02 will be [`../exp_02_naive_rag__test_1273/`](../) once the user runs `notebooks/04b_exp02_naive_rag.ipynb` Stage C (target: test 1,273). That run will be ~10 min wall time, $0 cost.

## Don't

- Don't try to resume this directory — the runner won't because it uses a new directory name (`test_1273`), and even if forced, all the wasted work is on rows we no longer evaluate.
- Don't delete prematurely; keep until EXP_02 baseline canonical run is complete and verified.

## See also

- [plan.md §0 #8](../../plan.md#0-locked-decisions) — the locked evaluation surface
- [notebooks/04b_exp02_naive_rag.ipynb](../../notebooks/04b_exp02_naive_rag.ipynb) — the canonical run notebook
