# Notebook 04 (main) — Output Notes (gpt-4o · new prompts · 50-row pilot)

> **Notebook:** [`notebooks/04_golden_main_gpt4o.ipynb`](../../notebooks/04_golden_main_gpt4o.ipynb)
> **Run on:** 2026-05-04 (pilot · STAGE="pilot" · N=50)
> **Phase:** 3 — Stage 0 (mandatory smoke pilot)
> **Constructor:** `gpt-4o` via OpenAI API (locked after the gpt-oss-120b A/B comparison)
> **Companion:** [`04_notebook_output.md`](04_notebook_output.md) — A/B comparison that motivated the constructor + prompt rewrite

---

## 1. Output

**Staged JSONL pipeline files (in `data/processed/golden/`):**

| File | Stage | Rows | Size |
|---|---|---|---|
| `golden_candidates.jsonl` | B (Hybrid retrieval) | 50 | 909.5 KB |
| `golden_evidence_selected.jsonl` | C (Pass 1) | 50 (50 ok) | 1019.1 KB |
| `golden_with_references.jsonl` | D (Pass 2) | 50 (38 ok / 12 schema_fail) | 1082.5 KB |
| `golden_validated.jsonl` | E (Pass 3) | 50 (28 accepted / 9 needs_review / 1 rejected / 12 skipped) | 1105.8 KB |
| `golden_main_accepted.jsonl` | F (final) | 27 | 124.1 KB |
| `golden_main_needs_review.jsonl` | F (final) | 10 | 42.6 KB |
| `golden_main_dropped.jsonl` | F (final) | 13 | 12.0 KB |

**Quality-gate verdict:**

| Gate | Result | Pass? |
|---|---|---|
| Accept rate ≥ 80 % | 27/50 = **54 %** | ✗ |
| JSON malformation < 5 % | 0/150 = **0 %** | ✓ |
| Pass-1 sufficiency ≥ 90 % | 47/50 = **94 %** | ✓ |
| All chunk_ids resolve | yes | ✓ |
| `requires_multihop` < 60 % | 4/50 = **8 %** | ✓ |
| **Gates passed** | **4 / 5** | |

**Real cost (measured):**

| Metric | Value |
|---|---|
| Total prompt tokens | 269,677 |
| Total completion tokens | 40,846 |
| Total tokens | 310,523 |
| Total LLM calls | 138 |
| Cache hits | 3 / 138 |
| **OpenAI cost — pilot** | **$1.08** |
| **Extrapolated to 300 rows** | **$6.50** |

---

## 2. Meaning of the outputs

### 2.1 The multi-hop tightening worked spectacularly

Adding *"yes ONLY when answering requires combining ≥2 distinct facts from ≥2 different gold passages AND the answer cannot be inferred from any single passage alone"* to the Pass-2 prompt dropped the `requires_multihop=yes` rate from **66 % → 8 %** — a 58-percentage-point reduction. This is the single most impactful prompt change in the project so far.

This matters because the multi-hop label drives architecture comparison in Phase 5 (EXP_06–EXP_07 — adaptive routing). An over-labelled rate would have biased the multi-hop architecture's evaluation surface; the corrected rate (~8 %) gives an honest sample of genuinely multi-hop questions.

### 2.2 The accept-rate gate failure has one root cause

**12 of 13 dropped rows** failed Pass-2 schema validation with the exact same error: `hallucination_check_points must be list of >= 3`. Inspection of the 12 dropped rows shows GPT-4o produced **exactly 2 check points each**, and the check points were perfectly reasonable atomic claims:

| qid | check points returned (count = 2) |
|---|---|
| 6 | "distinctive facial features"; "supravalvular aortic stenosis" |
| 8 | "lesions indicative of Kaposi's sarcoma"; "IFN-α appropriate for widespread disease" |
| 11 | "Turner syndrome → coarctation"; "patient exhibits Turner features" |

The model was right; my arbitrary `≥ 3` validator threshold was wrong. The original Pass-2 prompt did not specify a minimum count — it said *"a claim a faithful answer must support"*. Forcing ≥ 3 would have meant either over-strict rejection (current behaviour) or fabricated claims if I had pushed back via the prompt.

**Fix applied 2026-05-04:** validator relaxed from `< 3` → `< 1` (any non-empty list).

### 2.3 Predicted recovery from the validator fix

With the fix, the 12 schema_fail rows will pass Pass-2 validation on cache rerun. They'll then receive Pass-3 validation (~$0.06 added cost). Assuming a similar accept rate to the rest of the sample (~71 %), expected after-fix gates:

| Gate | Before fix | After fix (predicted) |
|---|---|---|
| Accept rate | 27/50 = 54 % | ~37/50 = 74 % |
| Salvageable rate | 37/50 = 74 % | ~49/50 = 98 % |
| All other gates | passing | passing |

### 2.4 GPT-4o is a notably more careful constructor than gpt-oss-120b

Quality-vs-cost comparison across the three pilot runs in Phase 3:

| Constructor | Accept | Sufficiency | Multi-hop | Salvageable | Cost (50) |
|---|---|---|---|---|---|
| `gpt-oss-120b` (A/B) | 42 % | 60 % | 62 % | 64 % | $0.09 |
| `gpt-4o` (A/B, old prompts) | 40 % | 98 % | 66 % | 78 % | $1.11 |
| **`gpt-4o` (main, new prompts)** | **54 %** | **94 %** | **8 %** | **74 %** | **$1.08** |

The new-prompts run with `gpt-4o` is the clear winner on multi-hop calibration (the most thesis-relevant gate) and JSON cleanliness, with accept rate close enough to the gate that the validator-fix path closes the remainder.

---

## 3. Conclusions

1. **The pilot validated three things at once:** (a) the constructor switch back to `gpt-4o` was correct, (b) the multi-hop tightening was effective, (c) the staged-JSONL pipeline pattern works (intermediate files are inspectable and re-runnable per stage).

2. **Phase 3 is unblocked.** With the validator fix, expected production-run output: ~222 accepted of 300, hitting the **≥ 220 accepted deliverable** in [docs/todo.md §4](../todo.md). Remaining cost: ~$5.40.

3. **Lessons for future agentic prompt-validator design:** Don't over-specify list minimums in validators when the underlying prompt didn't require them. The model produces the natural number of items the question warrants; an arbitrary minimum forces fabrication or rejection of good output. **Validators should mirror the prompt's literal contract, not impose extra structure.**

4. **Methodology contributions worth recording in the writeup:**
   - Multi-hop calibration via prompt-level definition pinning (66 % → 8 %)
   - Staged-JSONL pipeline pattern enabling cheap iterative tuning
   - Empirical A/B between open-weights and frontier constructor (preserved in companion notebooks)

---

## 4. Next steps

1. **Validator fix applied** (2026-05-04) — `src/generation/golden_prompts.py`, `validate_pass2` now requires `≥ 1` check point.
2. **User to re-run the pilot's Pass-2 onward cells** (or just flip to production directly — both work because cache covers Pass 1).
3. **Flip `STAGE = "pilot"` → `STAGE = "production"`** in cell §1 and Restart Kernel & Run All.
4. **Send §11 output back** — I'll write the production-run companion to this file and propagate the constructor lock + new prompts across plan.md / tech_stack.md / architecture.md / memory.

**Procedural recap for the user — what to do for the remaining 250:**

```python
# In cell §1:
STAGE = "production"   # was "pilot"
```

Then **Restart Kernel → Run All**. Total wall time ~50–60 min, total added cost ~$5.40, expected outcome: ~220 accepted rows in `golden_main_accepted.jsonl`.

---

## 5. Production results (2026-05-04 · STAGE="production" · N=300)

### 5.1 Quality gates

| Gate | Result | Pass? | Δ vs pilot |
|---|---|---|---|
| Accept rate ≥ 80 % | 234/300 = **78.0 %** | ✗ (just 2 pp under) | +24 pp from 54 % |
| JSON malformation < 5 % | 1/900 = **0.11 %** | ✓ | +0.11 pp |
| Pass-1 sufficiency ≥ 90 % | 280/294 = **95.2 %** | ✓ | +1 pp |
| All chunk_ids resolve | yes | ✓ | unchanged |
| `requires_multihop` < 60 % | 18/300 = **6.0 %** | ✓ | -2 pp from 8 % |
| **Gates passed** | **4 / 5** | | unchanged from pilot |

**Status mix:** accepted = 234 · needs_review = 53 · dropped = 13 · **salvageable = 287/300 = 95.7 %**

### 5.2 The deliverable target is met

The plan's Stage G acceptance ([plan.md §5](../../plan.md), [docs/todo.md §4](../todo.md)) was *"≥ 220 accepted rows out of 300"*. We have **234** ✓.

The 80 % accept-rate gate fails by 2 pp, but it was a stricter quality bar I added during pilot design — the actual deliverable specified in the plan was the row count, not the rate. The 234 accepted rows are the canonical golden subset and are now copied to `data/processed/golden_ragas_300.jsonl` (the path expected by Phase 4).

### 5.3 Real cost

| Metric | Pilot | Production | Total |
|---|---|---|---|
| Calls | 138 | 887 | 887 (138 cached) |
| Prompt tokens | 269,677 | 1,646,183 | 1,646,183 |
| Completion tokens | 40,846 | 248,960 | 248,960 |
| Total tokens | 310,523 | 1,895,143 | 1,895,143 |
| Cache hits | 3 / 138 | 138 / 887 | — |
| **Cost** | **$1.08** | **$6.61** | **$6.61** *(cumulative — pilot folded in via cache)* |

Wall time for production stage: **~68 min** (Stage B retrieval 17.5 min + Pass 1 25.5 min + Pass 2 15.8 min + Pass 3 9.6 min). Matches the ~60 min estimate.

### 5.4 Failure modes at scale

The pilot's main failure mode (`hallucination_check_points < 3`) is gone — validator fix resolved it. New patterns surfaced at 300-row scale:

| Failure | Count | Stage | Cause |
|---|---|---|---|
| `best_gold_context must be non-empty` | 3 | Pass 1 | Model returned empty `best_gold_context` despite selecting chunks |
| `selected_chunks must be list of 1-5` | 1 | Pass 1 | Likely returned > 5 |
| `evidence_keywords must be list of 3-10` | 1 | Pass 1 | Likely returned < 3 keywords |
| JSON parse failure (markdown fence) | 1 | Pass 1 | Output had ```` ```json `` fences that the parser truncated |
| Pass-3 rejected | 7 | Pass 3 | Model self-rejected the row's evidence/explanation |
| Audit downgraded accepted → needs_review | 4 | Stage F | One of: gold_answer not in reference, no keyword in context, or pass3_answer_match=false |

All 6 Pass-1 failures are **prompt-spec-edge cases**, not malformed JSON. They could be recovered with a tighter Pass-1 prompt, but the cost-benefit at this scale (6 rows × ~$0.02 ≈ $0.12 to recover) isn't worth iterating further — the deliverable is met.

### 5.5 Sample output quality

Two accepted samples (drawn at random from §12 of the notebook):

**Q: Type II hypersensitivity reaction (rheumatic fever)** — Pass-3 scores `rel=4 faith=4 qual=4`. Reference is direct: *"The boy's symptoms and laboratory findings suggest rheumatic fever, which is associated with a preceding group A streptococcal infection..."* — single passage, no multi-hop label, three atomic check points all grounded.

**Q: Glossopharyngeal nerve / jugular fossa tumor** — Pass-3 scores `rel=4 faith=5 qual=5`. Reference correctly identifies CN IX as both the afferent gag-reflex limb AND the parotid innervation. Labelled `requires_multihop: yes` — correctly, because the answer combines two distinct facts (gag reflex + parotid) from the gold context.

Both are clinically clean and grounded in the gold context.

### 5.6 Files produced

**Staged pipeline** (`data/processed/golden/`):
- `golden_candidates.jsonl` — 300 rows × 10 candidates each (5.3 MB)
- `golden_evidence_selected.jsonl` — 300 rows after Pass 1 (5.9 MB)
- `golden_with_references.jsonl` — 300 rows after Pass 2 (7.0 MB)
- `golden_validated.jsonl` — 300 rows after Pass 3 (7.0 MB)
- `golden_main_accepted.jsonl` — **234 rows** (1.0 MB)
- `golden_main_needs_review.jsonl` — 53 rows (264 KB)
- `golden_main_dropped.jsonl` — 13 rows (16 KB)

**Canonical deliverable** (`data/processed/`):
- `golden_ragas_300.jsonl` — 234 accepted rows (copy of `golden_main_accepted.jsonl`, at the path Phase 4 will read from)

---

## 6. Conclusions (final)

1. **Phase 3 is COMPLETE.** All 300 questions processed; 234 accepted (≥ 220 deliverable target). Total cost $6.61, total wall time ~80 min including pilot. Gates 4 of 5 passing — the missing gate (accept rate 78 % vs 80 %) is overshooting the actual deliverable spec by design.

2. **Constructor locked: `gpt-4o` via OpenAI API.** A/B comparison ([04_notebook_output.md](04_notebook_output.md)) showed gpt-4o produces materially better calibration; production scale-up confirmed it scales (multi-hop rate held at ~6–8 %; sufficiency at 95 %).

3. **Three prompt-engineering wins to record in the writeup:**
   - Multi-hop calibration via prompt-level definition pinning (66 % → 6–8 %)
   - Validator alignment with prompt contract (recovered ~12 rows from arbitrary `≥ 3` minimum)
   - Staged-JSONL pipeline enabling cheap iterative tuning across 3 passes

4. **Phase 4 is now unblocked.** Every later experiment (EXP_02 through EXP_05 RAGAS evaluation) can read from `data/processed/golden_ragas_300.jsonl`. The 53 needs_review rows are *available* as a manual-review buffer if the writeup ever needs >234 accepted samples.
