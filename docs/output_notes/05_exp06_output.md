# Notebook 05 — EXP_06 Question Complexity Labels · Output Notes

> **Notebook:** [`notebooks/05_exp06_complexity_labels.ipynb`](../../notebooks/05_exp06_complexity_labels.ipynb)
> **Module:** [`src/retrieval/complexity.py`](../../src/retrieval/complexity.py)
> **Tests:** [`tests/test_complexity.py`](../../tests/test_complexity.py) (7/7 passing)
> **Run on:** 2026-05-10 (label generation + manual review)
> **Phase:** 5 — Group B adaptive routing (input artefact for EXP_07)
> **Output:** [`data/processed/complexity_labels.parquet`](../../data/processed/complexity_labels.parquet) — 12,723 rows × 8 cols, 123 KB

---

## 1. The rule

A rule-based classifier assigns each MedQA question one of `Simple` / `Moderate` / `Complex` using four features available at notebook time (no LLM calls):

1. **`n_words`** — word count of the question stem
2. **`n_phrases`** — count of MetaMap medical phrases (precomputed in `medqa_4opt.parquet`)
3. **`has_complex_cue`** — clinical-decision cue phrases like *"best next step"*, *"initial management"*, *"most appropriate next"*
4. **`has_simple_cue`** — factoid / mechanism cues like *"mechanism of action"*, *"rate-limiting"*, *"derived from"*

Thresholds are anchored to the **33rd / 67th percentiles** of the full 12,723-question corpus (computed 2026-05-10):

```
n_words   p33 = 93   p67 = 133
n_phrases p33 = 28   p67 = 41
```

Rule order:
- `Complex` if `has_complex_cue` OR `(n_words ≥ 133 AND n_phrases ≥ 41)`
- `Simple`  if `(n_words ≤ 93 AND n_phrases ≤ 28)` OR `(has_simple_cue AND n_words ≤ 93)`
- `Moderate` otherwise

**Why a rule and not a learned classifier**: transparency. A learned classifier would need its own training data, introduce a model-of-a-model dependency, and make the routing decision opaque to a viva examiner. A length + entity-density + cue-word rule is defensible end-to-end and auditable from the parquet.

---

## 2. Output distribution

| Bucket | n | % | step1 | step2&3 |
|---|---:|---:|---:|---:|
| Simple | 3,759 | **29.5 %** | 46.4 % | 8.9 % |
| Moderate | 4,163 | **32.7 %** | 36.5 % | 28.1 % |
| Complex | 4,801 | **37.7 %** | 17.1 % | 63.0 % |

By split (sanity check — rule is content-only, not split-aware):

| split | Simple | Moderate | Complex |
|---|---:|---:|---:|
| train | 3,031 | 3,343 | 3,804 |
| dev | 362 | 426 | 484 |
| test | 366 | 394 | 513 |

The step-stratified pattern is exactly what the rule was designed to produce — basic-science (step1) skews short / Simple, clinical-decision (step2&3) skews Complex. All three buckets sit comfortably in the 15–55 % gate band.

---

## 3. Manual review — 100 stratified rows (seed=42, 33/33/34)

**Rater disagreement count: 1 / 100** ✓ (gate was ≤ 20)

The single disagreement: `medqa_8198` (#54), labelled `Moderate`, w=134 p=39:

> 55yo man with 3-day decreased urine output + bilateral pedal edema + 4-month back pain + hypercholesterolemia → asks for next diagnostic test (multiple myeloma work-up — bone marrow biopsy).

Multi-system clinical-decision question. Falls just under both Complex thresholds (134 words ≥ 133 ✓ but 39 phrases < 41 ✗, so the `(long_stem AND high_entity)` rule fails) and has no cue word. Mild rater preference for Complex; rule says Moderate. Boundary case — not worth re-tuning the rule.

---

## 4. Patterns surfaced by the review

- **Cue-word logic works.** Every short-stem-with-`has_complex_cue` row is correctly routed Complex — they're "best next step" / "initial management" / acute-decision questions where Multi-Hop's iterative retrieval should help. The rule correctly overrides length-based bucketing in these cases.
- **Simple bucket lives up to its routing role.** Most Simple rows are short basic-science recall questions (mechanism, pathology, biochem, embryology, statistics) — exactly the questions where Naive's k=5 dense retrieval should be sufficient, or where No-RAG can rely on LLaMA's pretraining memorisation.
- **Moderate is the right "single-vignette diagnosis" bucket.** Most Moderate rows are 100–130 word single-system diagnosis or treatment vignettes.
- **Complex bucket structurally captures step2&3 long vignettes** (63 % of step2&3 land Complex) — the questions that need multi-hop's evidence-stitching.

---

## 5. Methodology footnote (required for the thesis writeup)

> **"MedQA is overwhelmingly clinical vignettes — even the 'Simple' bucket is mostly *short-vignette* questions, not pure factoids. The proposal's terminology (Simple / Moderate / Complex) is preserved for plan-alignment, but the rule is honestly a *length + entity density + cue-word* proxy for complexity, not a deep-semantic measure. The bucket structure is validated empirically by manual review (1/100 rater disagreement) and by the per-bucket accuracy stratification — the chosen architecture wins its assigned bucket in 8 of 9 cases (see [`05_exp07_output.md` §3.4](05_exp07_output.md))."**

---

## 6. Pre-EXP_07 per-bucket accuracy (the routing-relevance check)

Before locking the routing decisions, the EXP_06 notebook reports per-bucket accuracy of each underlying architecture from existing EXP_01–05 predictions:

| Bucket | n | No-RAG | Naive | Sparse | Hybrid | Multi-Hop |
|---|---:|---:|---:|---:|---:|---:|
| Simple | 366 | 0.7951 | 0.8169 | 0.7760 | 0.8060 | **0.8361** |
| Moderate | 394 | 0.7563 | 0.7437 | 0.7589 | 0.7690 | **0.7716** |
| Complex | 513 | 0.7719 | 0.7251 | 0.7446 | 0.7349 | **0.7856** |

**Multi-Hop wins every bucket on raw accuracy** — so the proposal's "Simple → Naive, Moderate → Hybrid, Complex → Multi-Hop" routing trades accuracy for compute, not for accuracy. This anchored the EXP_07 framing as **cost-adjusted Pareto frontier**, not raw accuracy.

The data also fed two empirical insights that shaped EXP_07's hypothesis design:
1. On Complex (n=513), every single-shot RAG architecture (Naive/Sparse/Hybrid) is *worse* than No-RAG — only Multi-Hop earns its place. This validates EXP_05's finding that iterative retrieval is what delivers grounded improvement.
2. On Simple, Multi-Hop wins by +1.9 pp over Naive but at 3× the Groq calls — so the routing question is whether the +1.9 pp is worth 2 extra Groq calls per question. (Empirically in EXP_07: no, for cost-bounded deployments.)

---

## 7. Conclusions

1. **EXP_06 is COMPLETE.** 12,723 labels written to `data/processed/complexity_labels.parquet`. Bucket distribution 30 / 33 / 38 with a clean USMLE-step stratification. Manual review: 1/100 disagreement.

2. **Headline tables touched**: Table 3 (Question Complexity Labelling Summary, 3 rows). The Excel cells are:

   | Cell | Value |
   |---|---|
   | n_Simple | 3,759 (29.5 %) |
   | n_Moderate | 4,163 (32.7 %) |
   | n_Complex | 4,801 (37.7 %) |
   | rater_disagreement | 1/100 |
   | rule | "n_words p33/p67 + n_phrases p33/p67 + cue-word matching" |
   | thresholds | "WORDS_P33=93, WORDS_P67=133, PHRASES_P33=28, PHRASES_P67=41" |

3. **Locked decision**: 3-bucket rule with the percentile-anchored thresholds above is the EXP_06 deliverable for the rest of Phase 5/6/7/8/9. Re-tuning would only be triggered by a thesis-supervisor review finding.

4. **Methodology paragraph for the thesis writeup** — anchor here so it's not improvised later: *"EXP_06 builds a rule-based 3-class complexity classifier (Simple / Moderate / Complex) over the full 12,723-question MedQA-US 4-option corpus. The rule uses four features (question word count, MetaMap medical-phrase count, clinical-decision cue-word presence, factoid-mechanism cue-word presence) with thresholds anchored to the 33rd and 67th percentiles of the full corpus. The classifier produces a 29.5 / 32.7 / 37.7 % distribution across the three buckets; step1 questions skew Simple (46 %) and step2&3 skew Complex (63 %). Manual review of 100 stratified rows (seed=42) yielded 1 rater disagreement, far below the 20 % acceptance threshold. Caveat: MedQA is overwhelmingly clinical vignettes; the 'Simple' bucket is honestly *shortest-third-vignette*, not pure factoid. The classifier is the input artefact for EXP_07 adaptive routing."*

---

## 8. Files produced

```
data/processed/
└── complexity_labels.parquet   ← 12,723 rows × 8 cols, 123 KB
                                  cols: question_id, complexity (ordered cat),
                                        n_words, n_phrases, has_complex_cue,
                                        has_simple_cue, meta_info, split

src/retrieval/complexity.py     ← rule implementation (~135 LOC)
tests/test_complexity.py        ← 7 tests (4 synthetic + 1 real-data + 2 invariants)
```

The parquet is consumed by [`notebooks/05_exp07_adaptive_rag.ipynb`](../../notebooks/05_exp07_adaptive_rag.ipynb) via a question-text → bucket lookup dict built by joining on `question_id`.
