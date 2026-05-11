# Notebook 09 — EXP_16 · Phase 9 Output Notes

> **Notebook:** [`notebooks/09_exp16_final_ranking.ipynb`](../../notebooks/09_exp16_final_ranking.ipynb)
> **Module:** [`src/synthesis/`](../../src/synthesis/) (`aggregator.py` · `normaliser.py` · `ranker.py` · `recommender.py`)
> **Tests:** [`tests/test_synthesis.py`](../../tests/test_synthesis.py) (11 / 11 passing)
> **Run on:** 2026-05-11 (built + ran in one session; no LLM calls)
> **Phase:** 9 — Final synthesis
> **Cost:** **$0** (pure aggregation over Phases 4–8 outputs)
> **Wall time:** **<10 s** end-to-end

---

## 1. Output

```
results/exp_16_final_synthesis/
├── table12_final_ranking.csv          ← Excel Table 12 paste-target
├── table10_adaptive_vs_fixed.csv      ← Excel Table 10 paste-target
├── component_scores_raw.csv           ← per-arch raw metrics (intermediate)
├── component_scores_normalised.csv    ← per-arch [0,1] components (intermediate)
├── pareto_status.csv                  ← (accuracy, calls/Q) frontier sanity check
├── recommendations.csv                ← use-case → architecture mapping
├── sensitivity_ranks.csv              ← rank stability under 3 alt. weight regimes
└── summary.json                       ← paste-into-Excel anchor + headlines
```

---

## 2. Headline finding — Multi-Hop wins the locked weighted ranking; the proposal's "Adaptive should be best balanced" hypothesis is **falsified**

Plan §11 weights: `0.25·Accuracy + 0.25·Faithfulness + 0.20·Retrieval + 0.15·Safety + 0.10·Explainability + 0.05·Latency`.

**Table 12 — Final Weighted Ranking**:

| Rank | Architecture | Accuracy | Faithfulness | Retrieval | Safety | Explainability | Latency | **final_score** |
|:-:|---|---:|---:|---:|---:|---:|---:|---:|
| **1** | **MultiHop** | 0.7958 | 0.2833 | 0.5426 | 0.2629 | 0.6325 | 0.0909 | **0.4855** |
| 2 | Adaptive_B | 0.7832 | 0.2756 | 0.5668 | 0.2471 | 0.4506 | 0.3054 | 0.4755 |
| 3 | Adaptive_A | 0.7863 | 0.1966 | 0.4652 | 0.1631 | 0.2549 | 0.0000 | 0.3887 |
| 4 | Naive | 0.7573 | 0.1308 | 0.3705 | 0.1043 | 0.0000 | 0.6306 | 0.3433 |
| 5 | Hybrid | 0.7659 | 0.0944 | 0.3140 | 0.0826 | 0.0000 | 0.6316 | 0.3218 |
| 6 | Sparse | 0.7581 | 0.0401 | 0.0942 | 0.0343 | 0.0000 | 0.6512 | 0.2561 |
| 7 | NoRAG | 0.7738 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.2434 |

**Multi-Hop ranks #1 by 0.01 over Adaptive Variant B.** The proposal's expected winner — Adaptive RAG — is not the best-balanced architecture under the locked weights; **Adaptive Variant A places #3** behind even the simpler Variant B.

### 2.1 Why Adaptive A loses to Adaptive B in the synthesis

Variant A routes Moderate questions to **Hybrid** (Faithfulness 0.094); Variant B routes Moderate to **Multi-Hop** (Faithfulness 0.283). Plan §11 puts a 0.25 weight on Faithfulness — by the time you score the weighted column, that 0.19-point Faithfulness gap between A and B costs A ≈ 0.05 final-score points, easily enough to drop A from rank 2 to rank 3. Variant A is still the *cost-efficient* point on the accuracy-vs-Groq-calls Pareto frontier (notebook 05 §9), but the multi-dimension synthesis trades cheapness for groundedness.

### 2.2 Why NoRAG ranks last despite high accuracy

NoRAG scores **second** on the raw `accuracy_test_1273` column (0.7738) — only beaten by Multi-Hop (0.7958) and the two Adaptive variants. But it scores **zero** on Faithfulness, Retrieval, Safety, and Explainability because none of those metrics are defined without retrieved chunks (NaN → 0 floor by the normaliser's conservative default). This is the central thesis-claim mechanism made quantitative: a memorised-correct architecture without grounding takes a structural penalty in any framework that values evidence-supported answers. The 0.05 latency advantage is not enough to climb back.

---

## 3. Meaning of the outputs

### 3.1 Component-score interpretation

| Component | Definition (per `src/synthesis/normaliser.py`) | Why this choice |
|---|---|---|
| Accuracy | Raw `accuracy_test_1273` (already in [0,1]) | Direct interpretation; comparable across the seven archs on a single 1,273-Q surface. |
| Faithfulness | Raw `RAGAS_Faithfulness` golden_234 | Native [0,1]; Adaptive variants via score-join from underlying-arch ragas_scores.csv. |
| Retrieval | mean(Context Precision, Context Recall) golden_234 | Composite of RAGAS's two retrieval-quality axes — neither alone captures the full picture; mean preserves the [0,1] range. |
| Safety | 1 − Hallucination_Rate golden_234 | Hallucination_Rate is the fraction of rows with F < 0.5 (matches per-arch output notes convention). NoRAG = 0 because the metric is structurally undefined without chunks. |
| Explainability | Raw mean LIME-SHAP Spearman ρ (EXP_12) | Measured only on Multi-Hop's retrieval-changed 205-Q surface; Adaptive variants inherit by routing share (Variant A: Complex-bucket→Multi-Hop = 40.3 % → score 0.255; Variant B: Moderate+Complex→Multi-Hop = 71.3 % → score 0.451). Naive/Sparse/Hybrid have no measurement → 0. |
| Latency | min-max-inverted `mean_latency_s_test_1273` | Lower is better, so the column is rescaled to [0,1] and then inverted in-set. Fastest = 1.0 (NoRAG), slowest = 0.0 (Adaptive_A at 0.696 s). |

### 3.2 Pareto frontier sanity check — three architectures stand alone

| Architecture | Acc | Calls/Q | Status |
|---|---:|---:|---|
| NoRAG | 0.7738 | 1.000 | **frontier** |
| Naive | 0.7573 | 1.000 | dominated |
| Sparse | 0.7581 | 1.000 | dominated |
| Hybrid | 0.7659 | 1.000 | dominated |
| MultiHop | 0.7958 | 3.000 | **frontier** |
| Adaptive_A | 0.7863 | 1.806 | **frontier** |
| Adaptive_B | 0.7832 | 2.425 | dominated (by Adaptive_A) |

Same conclusion as the Phase 5 close-out (notebook 05 §9). Naive/Sparse/Hybrid are dominated by NoRAG on the (accuracy, compute) plane; Variant B is dominated by Variant A. The three-point frontier (NoRAG → Adaptive_A → Multi-Hop) is the deployment-relevant menu of cost-quality trade-offs.

### 3.3 Sensitivity — Multi-Hop's top rank survives 3 of 4 weight regimes

`sensitivity_ranks.csv`:

| Architecture | plan_default | accuracy_heavy | safety_heavy | **compute_heavy** |
|---|:-:|:-:|:-:|:-:|
| **MultiHop** | **1** | **1** | **1** | 2 |
| Adaptive_B | 2 | 2 | 2 | **1** |
| Adaptive_A | 3 | 3 | 3 | 6 |
| Naive | 4 | 4 | 4 | 3 |
| Hybrid | 5 | 5 | 5 | 4 |
| Sparse | 6 | 7 | 6 | 7 |
| NoRAG | 7 | 6 | 7 | 5 |

**Multi-Hop is rank-1 under plan_default, accuracy_heavy (0.50 acc weight), and safety_heavy (0.30 safety + 0.30 faithfulness)**. The single regime that flips the top spot is **compute_heavy** (0.20 latency weight), under which Adaptive Variant B takes #1 — its 13 % latency advantage and 65 % of the Groq-calls cost makes the difference once compute is heavily weighted. Adaptive_A drops to rank 6 in the compute-heavy regime because *its* mean_latency_s on test_1273 (0.696 s — the highest of any architecture) is the worst raw value; the min-max inversion lands it at exactly 0.

**Methodology footnote for the writeup**: *"The plan §11 weights are the locked default; rank stability was tested under three alternative regimes (accuracy-heavy, safety-heavy, compute-heavy). Multi-Hop remained rank-1 under three of four regimes, with Adaptive Variant B taking the top spot only under a compute-heavy weighting (0.20 latency · 0.20 accuracy · 0.20 faithfulness · ...). The locked ranking is therefore not an artefact of a fragile weight choice — it inverts only when latency/compute is upweighted by 4× over the plan default."*

### 3.4 Use-case recommendations

Mapping the data to the five categories in plan §11:

| Use case | Pick | Source metric | Value |
|---|---|---|---:|
| Lowest cost | **NoRAG** | Groq calls / Q | 1.0 |
| Highest accuracy | **MultiHop** | accuracy_test_1273 | 0.7958 |
| Lowest hallucination | **MultiHop** | Hallucination_Rate (lower is better) | 0.7371 |
| Highest explainability | **MultiHop** | LIME-SHAP Spearman ρ | 0.6325 |
| **Best balanced** | **MultiHop** | final_score (plan §11) | 0.4855 |

Multi-Hop wins **four of five** use cases. The proposal's framing — *"Best balanced → Adaptive (the proposal's expected winner)"* — is not supported by the data on the locked weights. The honest framing in the discussion chapter is:

> *"Across the six-dimension weighted synthesis, Multi-Hop RAG is the best balanced architecture for safety-critical deployment. Adaptive routing remains a deployment choice for cost-bounded scenarios — Variant B wins under a compute-heavy weighting (0.20 latency), and Variant A is the cost-efficient Pareto-frontier point. The thesis's central novelty contribution remains the **confidence-aware rejection layer over Multi-Hop**, which lifts accuracy to 1.000 at 60 % rejection (Phase 7) — not the routing layer itself, which is empirically the cost-optimisation knob, not the quality-optimisation knob."*

This is a Phase 9 reframing of the proposal, anchored in data. Reversed expectation, defended by sensitivity analysis.

### 3.5 Table 10 — Adaptive vs Best Fixed (Multi-Hop)

Δ columns show (Adaptive − Multi-Hop), with sign chosen so positive = adaptive wins:

| Metric | Multi-Hop | Adaptive_A | Adaptive_B | Δ A vs MH | Δ B vs MH |
|---|---:|---:|---:|---:|---:|
| Accuracy (test_1273) | 0.7958 | 0.7863 | 0.7832 | −0.0094 | −0.0126 |
| Faithfulness (golden_234) | 0.2833 | 0.1966 | 0.2756 | −0.0868 | −0.0077 |
| Context Precision | 0.3737 | 0.3598 | 0.3792 | −0.0139 | +0.0055 |
| Context Recall | 0.7115 | 0.5705 | 0.7544 | −0.1410 | **+0.0428** |
| Hallucination Rate (↓) | 0.7371 | 0.8369 | 0.7529 | −0.0998 | −0.0159 |
| LIME-SHAP Spearman ρ | 0.6325 | 0.2549 | 0.4506 | −0.3776 | −0.1818 |
| Mean latency (s) (↓) | 0.660 | 0.696 | 0.574 | −0.036 | **+0.086** |
| Groq calls/Q (↓) | 3.000 | 1.806 | 2.425 | **+1.194** | **+0.575** |
| **final_score** | **0.4855** | 0.3887 | 0.4755 | −0.0968 | −0.0101 |

**Variant B's lonely Δ-positive wins**: Context Precision (+0.6 pp), Context Recall (+4.3 pp), Mean latency (−13 %), and Groq calls/Q (−19 %). **Variant A's only Δ-positive win**: Groq calls/Q (−40 %). On every grounding metric, both Adaptive variants are weaker than Multi-Hop; Adaptive_B's Faithfulness is just 0.008 below Multi-Hop's, but the rest of the metrics keep Multi-Hop on top.

---

## 4. Conclusions

1. **Phase 9 is COMPLETE.** One module suite (4 files, 11 unit tests passing), one notebook, eight CSV/JSON output files. Excel Tables 10 + 12 paste-ready. Cumulative project spend remains ~$63 / $80 ceiling (Phase 9 added $0).

2. **Headline numbers for Excel Tables 10 + 12**:

   | Cell | Value | Source |
   |---|---|---|
   | Top-ranked architecture | **MultiHop** | EXP_16 weighted synthesis |
   | Top-ranked final_score | **0.4855** | plan §11 weights |
   | Runner-up | Adaptive_B (0.4755) | rank 2 by 0.01 |
   | Pareto frontier | NoRAG · Adaptive_A · MultiHop | (accuracy, calls/Q) plane |
   | Lowest cost recommendation | NoRAG (1.0 calls/Q) | matches proposal §11 |
   | Best balanced recommendation | **MultiHop** | **falsifies** proposal expectation of Adaptive |

3. **Three thesis-publishable findings from Phase 9**:
   - **The proposal's "Adaptive should be best balanced" expectation is falsified on the locked weights.** Adaptive_A places #3; Multi-Hop wins. The data-honest reframing is that adaptive routing is a *cost-optimisation* knob (Pareto-frontier middle point), not the quality-optimisation winner.
   - **The headline does not survive a compute-heavy reweighting** — under 0.20 latency weight, Adaptive_B takes rank 1. This is methodologically the right note for the methodology section: the rank order is stable for plan-default / accuracy-heavy / safety-heavy weights, but inverts under compute-heavy weighting. The deployment recommendation is therefore conditional on the operator's compute budget.
   - **NoRAG's last-place ranking** is the quantitative version of the central thesis claim. Contamination-driven memorisation gives NoRAG the second-highest raw accuracy, but the structural zero on Faithfulness / Retrieval / Safety / Explainability reflects that the answer is correct *by coincidence*, not by grounded reasoning. The weighting scheme operationalises the thesis's safety-first stance.

4. **Implications for the discussion chapter — Act 7 (the closing)**:
   - *Act 7: across the six-dimension weighted synthesis (Accuracy, Faithfulness, Retrieval, Safety, Explainability, Latency), Multi-Hop RAG ranks first under the plan-locked weights and under 2 of 3 alternative weight regimes. Adaptive routing — the proposal's expected winner on the balanced category — places second (Variant B) and third (Variant A). The data inverts the proposal's premise: adaptive routing earns its place as the cost-efficient Pareto-frontier point, not the balanced winner. The confidence-aware rejection layer (Phase 7), which lifts Multi-Hop's accuracy to 1.000 at 60 % rejection, is the operational mechanism that translates Multi-Hop's grounding lead into a safety-grade clinical-deployment system.*

5. **Operational health**: 11 / 11 unit tests pass; the synthesis reproduces Variant A/B score-joined RAGAS values to within 1e-4 of the numbers reported in [`05_exp07_output.md` §3.4](05_exp07_output.md) (re-derived from disk, not hand-copied); 0 NaN warnings; deterministic re-run.

---

## 5. Next steps

### 5.1 Thesis writeup (the natural sequel)

Phase 9 closes the experiment programme. The thesis chapter structure now has every data anchor it needs:

- **Methodology chapter** — every locked decision in [`plan.md §0`](../../plan.md) is data-defensible; the four caveats (k=5 vs k=15 for adaptive, golden_234 contamination skew, taxonomy 6→4 collapse, plan-§11-weight sensitivity) all have methodology footnotes drafted in the Phase 4–9 output notes.
- **Results chapter** — Tables 1–12 from the [`Raja Kalavala Final Thesis Project Sheet.xlsx`](../thesis-files/) are paste-ready from `results/exp_*/summary.json` or `results/exp_*/table*.csv`.
- **Discussion chapter** — the seven-act narrative (No-RAG contamination → single-shot RAG failure mode → Multi-Hop grounding → adaptive routing as the cost-efficient Pareto point → confidence-aware rejection as the safety mechanism → hallucination taxonomy as the error-mode resolution → weighted synthesis as the deployment recommendation) is anchored act-by-act in the Phase 4–9 output notes.

### 5.2 (Optional) Phase 10 — Demo UI Stage C

Plan §12 marks Phase 10 as optional, parallel-track. With Phase 9 complete and the experiment story landed, the question is whether to invest the ~3 days for Stage C (styling, screenshots, Streamlit Cloud deploy). [Plan §15 risk #11](../../plan.md#15-risk-register) says: *"if at the end of Phase 9 the experiment results are weak or incomplete, drop the UI entirely and write up the thesis without it."* — the results are **not** weak: every table has a publishable finding. The UI is now an upside, not a salvage attempt. Defer the call to the user; the thesis defends without it.

### 5.3 Methodology paragraph for the writeup

Anchor here so it's not improvised later:

> *"EXP_16 (Final Synthesis) aggregates the seven evaluated architectures (No-RAG, Naive Dense, Sparse BM25, Hybrid RRF, Multi-Hop, Adaptive Variant A, Adaptive Variant B) into a single weighted ranking across six dimensions: Accuracy (test_1273 exact-match), Faithfulness (RAGAS golden_234), Retrieval (mean of Context Precision and Context Recall), Safety (1 − Hallucination_Rate), Explainability (LIME-SHAP Spearman ρ from EXP_12), and Latency (mean per-question latency on test_1273). Component weights are 0.25 / 0.25 / 0.20 / 0.15 / 0.10 / 0.05 per the locked proposal in plan §11. RAGAS-style metrics enter at their native [0,1] value; latency is min-max-inverted in-set. Architectures with structurally undefined measurements (No-RAG on Faithfulness/Retrieval/Safety/Explainability; non-Multi-Hop fixed architectures on Explainability) take a conservative zero floor. Adaptive variants' RAGAS values are derived by per-question score-join from the underlying-architecture ragas_scores.csv files (no new judge calls). On the locked weights, Multi-Hop RAG ranked first (final_score = 0.4855), with Adaptive Variant B second (0.4755) and Adaptive Variant A third (0.3887). The ranking is stable under accuracy-heavy and safety-heavy alternative weights but inverts under a compute-heavy weighting, where Adaptive Variant B takes the top spot — a sensitivity finding documented in the methodology section. The use-case recommendations from the data are: lowest cost = No-RAG (1 Groq call/Q), highest accuracy = Multi-Hop, lowest hallucination = Multi-Hop, highest explainability = Multi-Hop, best balanced = Multi-Hop. The proposal's expected winner on the balanced category (Adaptive RAG) is not supported by the synthesis under the locked weights; the data-honest reframing positions adaptive routing as the cost-efficient Pareto-frontier choice rather than the quality-optimisation winner."*

---

## 6. Files produced

```
src/synthesis/
├── __init__.py
├── aggregator.py            ← collect_architecture_metrics + adaptive score-join (270 LOC)
├── normaliser.py            ← raw_with_minmax_latency scheme (95 LOC)
├── ranker.py                ← weighted_score + pareto_frontier (75 LOC)
└── recommender.py           ← use_case_recommendations (85 LOC)

tests/
└── test_synthesis.py        ← 11 tests · all pass

notebooks/
└── 09_exp16_final_ranking.ipynb  ← 10 sections · no LLM calls

results/exp_16_final_synthesis/
├── table12_final_ranking.csv          ← Excel Table 12 paste-target
├── table10_adaptive_vs_fixed.csv      ← Excel Table 10 paste-target
├── component_scores_raw.csv           ← per-arch raw metrics (intermediate)
├── component_scores_normalised.csv    ← per-arch [0,1] components (intermediate)
├── pareto_status.csv                  ← (accuracy, calls/Q) frontier sanity check
├── recommendations.csv                ← use-case → architecture mapping
├── sensitivity_ranks.csv              ← rank stability under 3 alt. weight regimes
└── summary.json                       ← paste-into-Excel anchor + headlines
```
