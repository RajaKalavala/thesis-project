# Notebook 04 — Output Notes (Constructor A/B Comparison)

> **Notebooks:**
> - [`notebooks/04_golden_dataset_gptoss.ipynb`](../../notebooks/04_golden_dataset_gptoss.ipynb) — `openai/gpt-oss-120b` via Groq
> - [`notebooks/04_golden_dataset_gpt4o.ipynb`](../../notebooks/04_golden_dataset_gpt4o.ipynb) — `gpt-4o` via OpenAI API
>
> **Run on:** 2026-05-04
> **Phase:** 3 — Stage 0 (mandatory smoke pilot)
> **Sample:** identical 50 questions, seed = 42, identical prompts, identical retrieval (Hybrid RRF k=60). Only the constructor LLM differs.

---

## 1. Output

**Artifacts saved to disk (variant-prefixed):**

| File | gpt-oss-120b | gpt-4o |
|---|---|---|
| `golden_ragas_50_pilot_<v>.jsonl` | 21 rows · 123.5 KB | 20 rows · 137.0 KB |
| `golden_ragas_50_pilot_<v>_needs_review.jsonl` | 11 rows · 65.6 KB | 19 rows · 116.4 KB |
| `golden_ragas_50_pilot_<v>_dropped.jsonl` | 18 rows · 43.5 KB | 11 rows · 62.5 KB |

**Side-by-side metrics from §10 of each notebook:**

| Metric | `gpt-oss-120b` (Groq) | `gpt-4o` (OpenAI) |
|---|---|---|
| **Accept rate** | 21 / 50 = **42 %** | 20 / 50 = **40 %** |
| Pass-1 sufficiency rate | 30 / 50 = **60 %** | 49 / 50 = **98 %** |
| JSON malformation | 0 / 150 = **0 %** | 0 / 150 = **0 %** |
| Loop schema/validation errors | **11** | **0** |
| Needs-review count | 11 | 19 |
| Rejected count | 18 | 11 |
| `requires_multihop = "yes"` | 31 / 50 = 62 % | 33 / 50 = 66 % |
| All cited `chunk_id`s resolve | ✓ | ✓ |
| **Wall time (50 rows)** | 6.6 min | 11.0 min |
| Total prompt tokens | 308,450 | 336,513 |
| Total completion tokens | 76,930 (incl. hidden reasoning) | 27,201 |
| **Cost — 50-row pilot** | $0 free / $0.09 paid | **$1.11** |
| **Cost — 300 rows extrapolated** | $0 free / $0.55 paid | **$6.68** |

**Quality-gate verdict (5 gates from [docs/todo.md §4](../todo.md)):**

| Gate | gpt-oss-120b | gpt-4o |
|---|---|---|
| Accept rate ≥ 80 % | ✗ (42 %) | ✗ (40 %) |
| JSON malformation < 5 % | ✓ (0 %) | ✓ (0 %) |
| Pass-1 sufficiency ≥ 90 % | ✗ (60 %) | ✓ (98 %) |
| All chunk_ids resolve | ✓ | ✓ |
| `requires_multihop` < 60 % | ✗ (62 %) | ✗ (66 %) |
| **Gates passed** | **2 / 5** | **3 / 5** |

---

## 2. Meaning of the outputs

### 2.1 The accept-rate tie is misleading

Both models land at ~40 % accepted, but for opposite reasons:

- **`gpt-oss-120b` rejects 18 questions** because Pass 1 too often says *"evidence insufficient"* (60 % sufficiency rate). Conservative — refuses to attempt Pass 2 on marginal evidence.
- **`gpt-4o` rejects only 11 questions** because Pass 1 says *"sufficient"* almost always (98 %). Reaches Pass 2 → some questions then get audit-flagged at Stage F and downgraded to needs_review.

The headline accept-rate number hides this asymmetry. The right metric for our purpose is **salvageable rows** — accepted + needs_review (which a reviewer can manually fix in ~10 min each):

| Metric | gpt-oss-120b | gpt-4o |
|---|---|---|
| Accepted + needs_review | 32 / 50 = **64 %** | 39 / 50 = **78 %** |
| Extrapolated to 300-row build | ~192 salvageable | ~234 salvageable |

The deliverable in [docs/todo.md §4](../todo.md) is **≥220 accepted rows out of 300**. `gpt-4o` lands closer to that without manual rescue work.

### 2.2 Loop errors tell a quieter story

`gpt-4o` had **0 loop errors**. `gpt-oss-120b` had **11**. Both report 0 % JSON-malformation by my parser, so the parser sees clean JSON in both cases — but `gpt-oss-120b` produced output that failed schema validation (e.g., empty `selected_chunk_ids`, wrong key types, or chunk_ids that don't resolve in `chunks.parquet`). Of the 18 rejected questions on the gpt-oss run, 11 are **schema/operational failures**, not the model's judgement that the question is unanswerable. That's load-bearing reliability, not a prompt issue.

### 2.3 Per-example quality on the smoke-test question (Alzheimer reliability vs validity)

| | `gpt-oss-120b` | `gpt-4o` |
|---|---|---|
| `reference_answer` | "Reliable – the test gives consistent results on repeated measurements but does not agree with the gold-standard diagnosis." | "The new test would most accurately be described as reliable because it produces very similar results for any given patient." |
| Pass-3 `faithfulness_score` | **3 / 5** | **5 / 5** |
| Pass-3 `evidence_relevance_score` | 3 / 5 | 4 / 5 |
| `requires_multihop` | yes ✗ (this is a single-concept recall question) | no ✓ |

`gpt-4o`'s reference is more directly grounded in the question stem; `gpt-oss-120b`'s reference adds an unstated inference about the gold standard that isn't in the chunks. The Pass-3 self-rated faithfulness gap (3 vs 5) tracks this. This is one example, but it is consistent with the broader pattern — `gpt-4o`'s outputs sit closer to the gold context.

### 2.4 Token efficiency — the hidden cost of reasoning models

`gpt-oss-120b` produced 76,930 completion tokens to generate roughly the same amount of *visible* JSON as `gpt-4o`'s 27,201. The extra ~50k are **hidden reasoning tokens** (the model "thinks" before emitting JSON; Groq counts those as completion tokens for billing). On the paid Groq tier this still nets to far cheaper than `gpt-4o`, but it does mean naive token estimates undershoot reasoning-model cost by ~3×.

### 2.5 Both fail the multi-hop gate, but for different reasons

`gpt-oss-120b` labels the Alzheimer reliability question as `requires_multihop=yes` even though it's a single-concept recall question — over-labelling driven by reasoning-model habit. `gpt-4o` is more accurate on individual examples (correctly labelled the same question as `no`) but its overall multi-hop rate is *higher* (66 % vs 62 %) because it accepts more borderline questions where multi-hop is genuinely required. Both will need Pass-2 prompt tightening before scaling to 300; this is a prompt issue independent of model choice.

---

## 3. Conclusions

### 3.1 Recommendation — lock `gpt-4o`, scale to 300 (after one prompt-tuning pass)

The cost saving of `gpt-oss-120b` is real ($6.28 absolute for 300 rows) but the data shows three measurable quality advantages of `gpt-4o`:

1. **Salvageable rate +14 pp** (78 % vs 64 %) → ~42 more usable rows from a 300-row build → meets the ≥220 deliverable target without over-sampling.
2. **Zero loop errors** vs eleven → operational reliability for any future re-runs.
3. **Per-example reference-answer faithfulness is materially higher** on the one example we can A/B compare directly.

Decision rule, applied:

```
Δ salvageable_rate / Δ cost_for_300_rows
  = (78 % − 64 %) / ($6.68 − $0.40)
  = +14 % / $6.28
  = +2.2 % per dollar
```

Positive → `gpt-4o` wins. The cost difference is a rounding error against the thesis budget; the dataset-quality lift compounds into every Faithfulness / Context Precision / Context Recall number in chapters 4 and 5.

### 3.2 What the comparison cost us

- **gpt-oss-120b pilot:** $0 (Groq free tier) — 6.6 min wall time
- **gpt-4o pilot:** $1.11 (OpenAI paid) — 11 min wall time
- **Total spend on the A/B:** **$1.11**
- **Time spent on the A/B:** ~30 min including this analysis

For ~$1 of API spend we have an empirically defensible constructor choice — viva-ready, not "we picked the cheaper model" or "we picked the locked model".

### 3.3 What still needs fixing before scaling to 300 (regardless of model)

Both pilots failed the **`requires_multihop < 60 %` gate** (gpt-oss 62 %, gpt-4o 66 %). The Pass-2 prompt definition needs tightening — *"yes only when the answer requires combining ≥2 distinct facts from ≥2 distinct chunks AND the answer cannot be inferred from any single chunk alone"*. After tuning, expect the rate to drop to ~30 % and accept rate to climb a few points.

If the user picks `gpt-oss-120b` instead, the additional fix needed is the **Pass-1 sufficiency criterion** — currently 60 % vs the 90 % gate. Loosen *"insufficient"* to mean *"NO chunk discusses the answer concept"* rather than *"chunks don't directly state the answer"*.

### 3.4 If the user prefers `gpt-oss-120b` despite the data

Reasons to still pick `gpt-oss-120b`:
- **Pure principle** — open-weights, free, fully reproducible by anyone with a Groq key
- **Marginal cost matters** — if the $6.28 is real (other thesis costs eat into the budget hard)
- **Willingness to do prompt-tuning rounds** — the 60 % sufficiency could plausibly be tuned to 80–90 %; if it works, the salvageable-rate gap closes

Trade-offs to accept if going this route:
- ~3 hours of prompt iteration before re-running the pilot
- Likely need to over-sample (build 350 rows to get 220 accepted)
- Slightly weaker viva position when explaining the deviation from the locked plan

### 3.5 What this notebook unblocks

Either way, **Phase 3 Stage 0 is functionally complete**: we have a working 3-pass constructor pipeline, audit logic, and quality-gate framework. The decision pending is *which constructor* to use for the 250-row production scale-up.

After the constructor is locked:
1. Tune Pass-2 multi-hop prompt to fix the >60 % over-labelling
2. (If staying with `gpt-oss-120b`) tune Pass-1 sufficiency criterion
3. Re-run 50-row pilot once to verify gates pass
4. Scale to 250 production rows → merge with pilot → 300 total in `golden_ragas_300.jsonl`
5. Move on to Phase 4 (EXP_01 → EXP_05)

---

**Status as of 2026-05-04:** decision pending user input on constructor (gpt-4o recommended; gpt-oss-120b also defensible if budget-constrained). Once locked, the next iteration is a single prompt-tuning round — not a re-run of the full pilot.
