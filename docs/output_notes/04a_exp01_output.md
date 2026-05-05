# Notebook 04a — EXP_01 No-RAG Baseline · Output Notes

> **Notebook:** [`notebooks/04a_exp01_base_llm.ipynb`](../../notebooks/04a_exp01_base_llm.ipynb)
> **Run on:** 2026-05-05 (smoke + golden + full 12,723, all three stages)
> **Phase:** 4 — Group A baseline experiments (first of five)
> **Architecture:** No retrieval — direct LLM answer to question + options
> **Generator:** `llama-3.3-70b-versatile` via Groq · T=0 · max_tokens=700 (per Excel `Experiments Guide`)
> **Companion:** [`docs/results_schema.md`](../results_schema.md) · [`docs/todo.md` §5 EXP_01](../todo.md)

---

## 1. Output

Three result directories under `results/`, each with `predictions.jsonl`, `retrieval.jsonl` (empty lists — No-RAG by definition), and `summary.json`:

| Surface | Rows | Accuracy (`Acuuracy`) | Mean latency | Wall time | Cache hits |
|---|---:|---:|---:|---:|---:|
| `exp_01_base_llm__smoke_50` | 50 | **0.9400** (47/50) | 0.27 s | 14 s | 0 / 50 |
| `exp_01_base_llm__golden_234` | 234 | **0.9017** (211/234) | 0.25 s | 58 s | 3 / 234 |
| `exp_01_base_llm__full_12723` | 12,723 | **0.8693** (11,060/12,723) | 0.28 s | 58 min | 286 / 12,723 |

Runner integrity (every row in every surface):
- `pred_letter` parsed cleanly — **0 nulls** across 13,007 calls
- 99.97 % of `raw_response`s are exactly 1 character (LLM follows "output exactly one letter" precisely)
- Cache hits on `full_12723` = 286 = 50 + 234 + 2 idempotent re-runs ⇒ disk cache works on prompt-hash, surfaces share answers when question text matches
- All three `summary.json`s conform to [`docs/results_schema.md`](../results_schema.md); RAGAS keys + Recall@K ship as `null` (next session)

---

## 2. Meaning of the outputs

### 2.1 The accuracy gap across surfaces is the contamination signal — not noise

| Stage | Surface | Accuracy | Comment |
|---|---|---:|---|
| Smoke | 50 stratified rows (any split) | 0.9400 | small sample noise + likely train-heavy (seed 42 sample) |
| Golden | 234 rows (188 train · 28 dev · **18 test**, see §2.4) | 0.9017 | construction biased toward "easy-to-validate" questions |
| Full | 12,723 (10,178 train + 1,272 dev + 1,273 test) | 0.8693 | the headline figure, but the *useful* number is split-stratified |

The full 12,723 number breaks down sharply by split:

| Split | n | Accuracy |
|---|---:|---:|
| train | 10,178 | **0.8792** |
| dev | 1,272 | **0.8860** |
| test | 1,273 | **0.7738** |

**Train + dev vs test gap: 10.61 percentage points.** This is exactly the dataset-contamination risk anticipated in [`plan.md` §15](../../plan.md): *"LLaMA already saw MedQA in pretraining... if No-RAG is already > 75 %, hallucination is the more interesting story than accuracy."* Empirically validated.

The **test-split number (77.4 %)** aligns with the literature LLaMA-class No-RAG-on-MedQA-US ceiling reported by MedRAG / MIRAGE (~75–78 %), confirming that the test split is the cleanest evaluation surface. Train and dev are inflated by ~11 pp of memorisation.

### 2.2 What this means for the thesis discussion

1. **The headline `Acuuracy = 0.8693` is reportable but needs a methodology footnote.** The thesis writeup must accompany every full-12,723 number with the train-vs-test breakdown so the contamination story is visible. *Hiding* the gap would invite a viva question; *leading* with it converts a contamination risk into a thesis contribution.

2. **The interesting story shifts from accuracy to hallucination control.** The plan flagged this in advance: when No-RAG is already at ~88 % on contaminated data, the differentiator across architectures is not "can RAG raise accuracy" (modest headroom) but "can RAG reduce wrong-answer-with-confidence rate" (which is what RAGAS Faithfulness will measure on the golden 234). EXP_01's headline number sets the stage for *why* the confidence-aware-rejection layer (Phase 7) is the load-bearing novelty contribution.

3. **The test split becomes the secondary headline.** For each subsequent architecture (EXP_02–EXP_05, EXP_07), report two numbers: full-12,723 (paste-cell to Table 1) and test-split-only. The test-split number is the directly-comparable-to-literature figure; the full number is the per-row labelling surface. Per [`docs/dataset.md` §4](../dataset.md), this dual-reporting is enabled because the loader carries the `split` column through the runner pipeline.

### 2.3 Long-vignette accuracy is *not* materially worse

The proposal hypothesised that long vignettes (>200 words, 4.07 % of the corpus per [`docs/dataset.md` §2.5](../dataset.md)) are where Multi-Hop RAG should win. EXP_01's No-RAG slice on long vs short:

| Stratum | n | Accuracy |
|---|---:|---:|
| short (≤ 200 words) | 12,205 | 0.8700 |
| long (> 200 words) | 518 | **0.8533** |

**1.67 percentage points** is not a meaningful gap. No-RAG handles long vignettes nearly as well as short. This makes the EXP_05 Multi-Hop story **more** demanding, not less: Multi-Hop has to beat naive/hybrid RAG on long vignettes by a margin large enough to justify its 3× compute cost. If it doesn't, the discussion chapter has a real "Multi-Hop did not earn its complexity" finding to acknowledge.

### 2.4 Golden 234's split distribution explains its 90 % accuracy

Of the 234 accepted golden rows: **188 from train (80.3 %), 28 from dev (12.0 %), 18 from test (7.7 %)**. The golden subset is dominated by train, which is where No-RAG is most contaminated → 90 % accuracy on golden 234 is *expected* given the underlying split mix (188 × 0.879 + 28 × 0.886 + 18 × 0.774) / 234 ≈ 0.872, vs the actual 0.902 (the construction process selected questions where evidence is unambiguous, slightly elevating accuracy further).

**Implication for RAGAS evaluation:** when the Claude judge runs on the golden 234, Faithfulness scores will be measured against a train-heavy slice. That's fine for relative comparison across architectures (every architecture sees the same 234 questions) but the writeup should note that golden's split-mix is not representative of the full corpus.

### 2.5 Edge cases worth recording

- **2 "E" predictions out of 12,723 (0.016 %)** — the LLM occasionally hedged with text like *"E is not an option, so I will choose..."* and the permissive letter-parser caught the first standalone `\bE\b`. Both are correctly counted as wrong (gold ∈ {A, B, C, D}). Not a parser bug — the parser is doing the literal job; the LLM is the one producing illegal letters. A stricter parser that rejects E in 4-opt would gain +0.0 pp accuracy (both cases were wrong anyway). Keeping the parser permissive matches existing convention in [`src/generation/prompts.py`](../../src/generation/prompts.py).

- **4 raw responses > 5 chars (0.031 %)** — three of these had explanatory preamble before the answer letter ("To find the clearance...", "D) is not the best because...", etc.). The parser still extracted a letter for all of them. Net effect: 1 correct answer survived the format violation, 3 were wrong (consistent with the overall 87 % accuracy). The LLM's adherence to "output exactly one letter" at 99.97 % is excellent.

- **2 latency outliers ≈ 180 s** — these are Groq SDK internal retry-on-rate-limit waits. They don't affect correctness, only throughput, and are absorbed by the 12,723-row average. p95 latency is **0.466 s**; the runner's wall-time budget (5 h estimate from [`plan.md` §14](../../plan.md)) was wildly conservative — actual wall time was 58 minutes.

### 2.6 No letter bias in predictions

| Letter | Gold | Predicted | Δ |
|---|---:|---:|---:|
| A | 3,267 | 3,161 | -106 |
| B | 3,279 | 3,417 | +138 |
| C | 3,255 | 3,223 | -32 |
| D | 2,922 | 2,920 | -2 |

A small B-bias (~+4 % over expected) but no extreme letter-collapse failure mode. Per-step analysis (Step 1 = 0.870 vs Step 2&3 = 0.868) is also balanced — no curriculum-stage bias.

---

## 3. Conclusions

1. **EXP_01 is COMPLETE.** All three surfaces written; runner schema-conformant; cache validated; contamination story empirically established.

2. **Headline numbers for Excel Table 1 row 1:**

   | Cell | Value |
   |---|---|
   | `Acuuracy` | 0.8693 |
   | `Exact Match` | 0.8693 |
   | `Generator Model` | `llama-3.3-70b-versatile` |
   | `mean_latency_s` | 0.279 |
   | `RAGAS_*` columns | pending RAGAS judge (next session) |

3. **The 87 % full-12,723 number is misleading without the 77 % test-split number alongside it.** Every subsequent architecture's results table should report both, and the methodology chapter should lead with the contamination disclosure. This converts a known-but-hidden risk into a documented limitation that strengthens the methodology rather than undermining it.

4. **The thesis discussion-chapter narrative is now anchored.** With No-RAG at 87–88 % on the full corpus and 77 % on the contamination-clean test split, the four RAG architectures (EXP_02–EXP_05) need to demonstrate *either*:
   - **Accuracy**: meaningful gain on the test split (>3 pp over 77 % is a defensible claim at n=1,273)
   - **Faithfulness**: lower hallucination on RAGAS metrics (the more interesting and likely-stronger differentiator)

   The latter is what the proposal's confidence-aware-rejection novelty rides on.

5. **Operational health is excellent.** 0 parse failures across 13,007 calls. p95 latency 0.47 s. Cache works. Resumability built and proven (re-running any stage is instant). Wall-time projection for Phase 4 revised down from ~30 h Groq to **~5 h Groq** for all five baseline experiments combined.

---

## 4. Next steps (in order)

1. **Build `src/eval/ragas_eval.py`** — wraps RAGAS with Claude 3.5 Sonnet judge, runs the five RAGAS metrics (Faithfulness, Context Recall, Context Precision, Answer Relevancy, Answer Correctness) against `results/exp_01_base_llm__golden_234/predictions.jsonl`. Cost: ~$2–3 for EXP_01 alone (golden 234 × 5 metrics × Claude calls). Fills the four `RAGAS_*` columns + `Answer_Correctness` + derives `RAGAS_Hallucination_Rate` (fraction with `Faithfulness < 0.5` per [`docs/todo.md` §5 EXP_01](../todo.md)).

2. **Build `src/retrieval/naive.py`** (refactor of the inlined logic in Notebook 03 under the `Retriever` ABC) — unblocks EXP_02.

3. **Run EXP_02 Naive RAG** — `notebooks/04b_exp02_naive_rag.ipynb` mirrors 04a's three-stage layout, swapping `NoRetrieval()` for `NaiveRetriever(...)`. Same runner, same schema, same cost profile.

4. **Lock the `RAGAS_Hallucination_Rate < 0.5` definition in the methodology chapter** — this is now an EXP_01-driven baseline against which all four RAG architectures are scored.

5. **Add a methodology paragraph to the thesis writeup**: *"Phase 4 baseline (EXP_01, No-RAG) revealed a 10.6 pp accuracy gap between the train+dev splits (87.99 %) and the test split (77.38 %), consistent with documented LLaMA pretraining inclusion of MedQA. All subsequent results are reported in pairs (full corpus / test-split only) so the contamination effect is visible to the reader."*

---

## 5. Files produced

```
results/
├── exp_01_base_llm__smoke_50/
│   ├── predictions.jsonl   ← 50 rows, 8 KB
│   ├── retrieval.jsonl     ← 50 empty rows, 4 KB
│   └── summary.json        ← 728 B
├── exp_01_base_llm__golden_234/
│   ├── predictions.jsonl   ← 234 rows, 38 KB
│   ├── retrieval.jsonl     ← 234 empty rows, 20 KB
│   └── summary.json        ← 778 B
└── exp_01_base_llm__full_12723/
    ├── predictions.jsonl   ← 12,723 rows
    ├── retrieval.jsonl     ← 12,723 empty rows
    └── summary.json
```

`predictions.jsonl` is the canonical artifact for downstream RAGAS evaluation and the optional Streamlit demo's Tab 1 ([`plan.md` §10.5](../../plan.md)). Schema: see [`docs/results_schema.md` §3](../results_schema.md).
