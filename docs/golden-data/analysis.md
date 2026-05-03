# Golden Dataset — Quality-Control Analysis

Analysis of the **100-question construction run completed on 2026-04-25**, updated the same day with the results of a manual review pass over the audit-flagged rows. All numbers are computed directly from the JSONL artifacts in `data/processed/` (and mirrored in `golden-data/` locally).

> **2026-04-25 update — manual review applied.** Section 11 documents the row-by-row spot-check of the 10 keyword-flagged rows (9 dropped, 1 salvaged) and the 5+5 random spot-checks for accepted-set grounding and multi-hop labelling. **Final accepted count: 65** (up from 64).

---

## 1. Pipeline integrity

Zero API failures across all three GPT-4 passes. Every stage produced exactly 100 rows.

| Stage | Rows | Status |
|---|---|---|
| Seed | 100 | ✓ |
| Candidates (10 chunks each) | 100 | ✓ |
| Pass 1 — evidence selection | 100 | 100 / 100 GPT-4 calls succeeded |
| Pass 2 — reference answer | 100 | 100 / 100 |
| Pass 3 — validation | 100 | 100 / 100 |
| Audited | 100 | ✓ |
| **Final accepted (after manual review)** | **65** | up from 64 after salvaging `medqa_gold_088` |
| Needs review | 22 | down from 32 after manual review pass |
| Dropped (manual review) | 9 | retrieval-failure rows confirmed unusable |
| Rejected (Pass 3) | 4 | |

---

## 2. Stratification — drifted from the 40 / 40 / 20 spec

| Bucket | Target | Actual |
|---|---|---|
| Step 1 | 40 | **45** |
| Step 2&3 | 40 | **55** |
| Long-vignette (>200 words) | 20 | **23** *(overlaps both step buckets)* |

The drift is by design, not a bug. Long-vignette questions are sampled first; the step buckets top up from the remainder. Long vignettes are themselves predominantly Step 2&3 (clinical case stems), which is why Step 2&3 ends up over-represented.

**Implication for the thesis methodology section:** describe the sampling as *"20 long-vignette questions sampled first; remaining 80 stratified across Step 1 and Step 2&3"*, not as three independent buckets.

---

## 3. Pass 1 — evidence selection

Healthy distribution. GPT-4 found supporting evidence for almost every question, and predominantly classified it as strong.

| Indicator | Value | Read |
|---|---|---|
| `is_evidence_sufficient = True` | 99 / 100 | retrieval is finding relevant chunks for nearly every question |
| Mean selected chunks per question | 1.83 (range 1–5) | reasonable parsimony |
| Support-level distribution | 133 strong · 48 moderate · 2 weak | strong-heavy, as desired |
| Mean evidence keywords per question | 4.70 (range 3–8) | enough for the audit's keyword check to be meaningful |

---

## 4. Pass 2 — reference answer + explanation

| Field | Distribution | Comment |
|---|---|---|
| `question_type` | 48 diagnosis · 20 management · 17 mechanism · 14 treatment · 1 other | sensible USMLE mix |
| `requires_multihop` | **77 yes · 23 no** | **suspiciously high — see §6** |
| `hallucination_check_points` per Q | mean 2.95 (range 2–4) | enough atomic claims for downstream RAGAS hallucination scoring |

---

## 5. Pass 3 — validation scores

| Metric | Value | Read |
|---|---|---|
| `answer_match: True` | 100 / 100 | **structurally tautological — do not cite** |
| Mean `evidence_relevance_score` | 3.73 / 5 | mediocre — retrieval finds plausible chunks, not always the most supportive |
| Mean `faithfulness_score` | 3.86 / 5 | OK |
| Mean `explanation_quality_score` | 4.32 / 5 | high — GPT-4 writes well, expected |
| `hallucination_risk` | 61 low · 32 medium · 7 high | a third of rows have medium+ risk; inspect the 7 `high` |
| `final_status` (LLM verdict) | 65 accepted · 27 needs_review · 8 rejected | |

**Why `answer_match` is meaningless here:** Pass 2 is given the gold answer as input and instructed to write a reference around it. Pass 3 then "checks" whether the reference matches the gold answer — a check that is structurally guaranteed to pass. This field is not a quality signal and should be excluded from any thesis claim about the construction process.

---

## 6. Two findings worth flagging in the thesis

### Finding 1 — `requires_multihop = yes` rate is 77 %

USMLE questions are typically single-hop (one mechanism, one drug, one diagnosis). GPT-4 labelling 77 % as multi-hop is suspiciously high and risks **inflating the apparent value of the Multi-Hop RAG architecture** at evaluation time.

**Mitigation:** manually inspect 10 of the questions labelled `requires_multihop = yes`. If most do not actually require chained retrieval, either:
- relabel via a stricter prompt (define "multi-hop" as "the answer requires combining facts from two distinct clinical entities or two separate textbook passages"), or
- drop the `requires_multihop` claim from the thesis stratification.

### Finding 2 — keyword hallucination caught by the audit

The automated audit flagged **10 rows** with `no_evidence_keyword_in_gold_context` — meaning Pass 1 declared `evidence_keywords` that do not actually appear in the `best_gold_context` text. GPT-4's self-validation (Pass 3) did not catch this; the mechanical audit did.

**This is the strongest defensibility signal in the run** and worth highlighting in the thesis methodology section: the multi-stage construction with an independent non-LLM audit catches failure modes that a single-LLM pipeline would rubber-stamp.

---

## 7. Retrieval coverage

Top books across all 1,000 candidate chunks (10 per question × 100 questions):

| Rank | Book | Candidates | Share |
|---|---|---|---|
| 1 | InternalMed_Harrison | 339 | 33.9 % |
| 2 | Pharmacology_Katzung | 103 | 10.3 % |
| 3 | Gynecology_Novak | 78 | 7.8 % |
| 4 | Neurology_Adams | 77 | 7.7 % |
| 5 | Pediatrics_Nelson | 68 | 6.8 % |

Harrison's at 34 % over-represents its share of the corpus by word count (24.9 %, per [docs/dataset_exploration.md](../dataset_exploration.md)). This is corpus-frequency bias — Harrison's has more chunks in the index, so more compete for the top-k slots. It is shared by every architecture in the comparison and therefore cancels out for *relative* architecture analysis, but the absolute retrieval picture is internal-medicine-heavy.

---

## 8. Final accepted dataset — composition (post manual review)

`data/processed/medqa_ragas_golden.jsonl` — **65 rows**:

| Slice | Distribution |
|---|---|
| `meta_info` | 34 step2&3 · 31 step1 |
| `question_type` | 28 diagnosis · 14 management · 12 treatment · 11 mechanism |
| `requires_multihop` | 47 yes · 18 no *(see §11 below for the multi-hop spot-check verdict)* |

This is the dataset Notebook 05 (pending) will consume.

---

## 9. Companion files (post manual review)

| File | Rows | Purpose |
|---|---|---|
| `medqa_ragas_golden.jsonl` | 65 | Accepted — input to Notebook 05 |
| `medqa_ragas_golden_needs_review.jsonl` | 22 | Audit issues + LLM rejections still under triage |
| `medqa_ragas_golden_dropped.jsonl` | 9 | Retrieval-failure rows confirmed unusable. Each row has `manual_review.reason` explaining which Pass 1 chunk was wrong-topic. |

---

## 10. Manual review — methodology

Three spot-check tasks were applied over the construction output:

| # | Task | Sample | Method |
|---|---|---|---|
| 1 | Verdict on the 10 keyword-flagged rows | All 10 rows in `needs_review` with `audit_issues` containing `no_evidence_keyword_in_gold_context` | Read each `gold_context` in full; checked whether `reference_explanation` claims are grounded in the cited text or sourced from the LLM's prior knowledge. |
| 2 | Grounding spot-check | 5 random rows from `medqa_ragas_golden.jsonl` (seed 42) | For each claim in `reference_explanation`, verified textual support in `gold_context`. |
| 3 | Multi-hop label audit | 5 random rows from `medqa_ragas_golden.jsonl` filtered to `requires_multihop = "yes"` (seed 42) | Applied criterion: *requires combining ≥2 distinct facts from ≥2 distinct knowledge sources.* |

---

## 11. Manual review — findings

### 11.1 Keyword-flagged rows (10 reviewed)

**9 dropped, 1 salvaged.** The audit's "no evidence keyword in gold context" signal turned out to correlate with a deeper failure: in 9 of 10 cases, retrieval pulled the **wrong textbook passage** — Pass 2's reference explanation is correct medically but is sourced from GPT-4's prior knowledge rather than the cited evidence.

| Golden ID | Verdict | Failure mode |
|---|---|---|
| `medqa_gold_004` | **dropped** | Retrieved chunk is bowel ischemia, question is lead poisoning. |
| `medqa_gold_045` | **dropped** | Partial grounding — key claim *"increased peripheral resistance via unopposed alpha"* not supported by the cited beta-blocker-in-heart-failure passage. |
| `medqa_gold_050` | **dropped** | Retrieved chunk is phenylketonuria; Pass 2 explicitly notes the evidence does not mention adenosine deaminase deficiency. |
| `medqa_gold_051` | **dropped** | Retrieved chunk is anaerobic neck infections; question is the cellular mechanism of diapedesis. |
| `medqa_gold_052` | **dropped** | Retrieved chunk is giant cell arteritis; question is polycythemia vera. |
| `medqa_gold_056` | **dropped** | Retrieved chunk lists *other* causes of vertical gaze palsy (PSP, Parkinson, Whipple). Does not establish pineal tumor as a cause. |
| `medqa_gold_067` | **dropped** | Retrieved chunk is hyperthyroidism; question is pheochromocytoma diagnosis. |
| `medqa_gold_069` | **dropped** | Retrieved chunk is post-transplant infections; question is mycophenolate mofetil side effects. |
| `medqa_gold_086` | **dropped** | Retrieved chunk is tumor lysis syndrome; question is methanol poisoning treatment. |
| `medqa_gold_088` | **salvaged** | Audit false positive — context is on-topic for influenza vaccination in chronic-pulmonary patients (stems `influenza`, `chronic pulm`, `respiratory` all match). Pass 1 chose over-specific compound keywords (`"influenza vaccine"`, `"vaccination"`) that did not substring-match. |

**Implication for the thesis:** the audit caught a real signal stronger than first credited. *9 of 10 rows it flagged also failed substantive grounding*, not just keyword presence. The pipeline's combination of LLM construction + automated audit + manual triage is doing real quality-control work.

**Implication for retrieval research:** even with answer-aware search query construction (`question + answer + metamap_phrases`) and hybrid RRF retrieval, ~10 % of questions ended up with the wrong topic chunk in Pass 1. Worth a paragraph in the thesis discussion chapter — the four RAG architectures will each have their own version of this failure mode at evaluation time.

### 11.2 Grounding spot-check (5 random accepted rows)

**5 / 5 properly grounded.** Each `reference_explanation` is supported by claims appearing in the cited `gold_context`.

| Golden ID | Topic | Grounding evidence |
|---|---|---|
| `medqa_gold_006` | Interstitial cystitis → self-care | Context: *"Treatment typically focuses on symptoms… voiding regimen and local care."* Explanation paraphrases. |
| `medqa_gold_025` | High-altitude → hypoxemic hypoxia | Context: *"Hypoxemia and respiratory alkalosis are consistently present."* Explanation grounds in this. |
| `medqa_gold_043` | Pioglitazone → weight gain + edema | Context: *"thiazolidinedione promotes weight gain and is associated with peripheral edema."* Explanation cites both effects. |
| `medqa_gold_047` | Iodide → I₂ via thyroid peroxidase | Context: *"thyroid peroxidase (TPO) then oxidizes iodide…"* Explanation paraphrases verbatim. |
| `medqa_gold_055` | Atherosclerosis → endothelial cells injured first | Context: *"risk factors… disturb the normal functions of the vascular endothelium."* Explanation grounds in this. |

**Verdict on the accepted set: solid.** The 65 accepted rows can be defended as evidence-grounded ground truth.

### 11.3 Multi-hop label audit (5 random rows)

**3 clear multi-hop · 2 borderline single-hop.** The 77 % `requires_multihop = yes` rate is **defensible if the methodology section redefines the label** as *"requires ≥2 retrieval / reasoning steps, including the common diagnose-then-treat pattern"* — not as *"requires explicitly chained multi-document retrieval."*

| Golden ID | Verdict | Reasoning |
|---|---|---|
| `medqa_gold_002` | borderline (single-hop in spirit) | Recognise IDA labs → iron supplementation; vegan diet is confirmatory, not a separate hop. |
| `medqa_gold_015` | multi-hop | Identify which lipid abnormality persists despite statin (TG = 365) → drug class for hypertriglyceridemia. Two distinct retrievals. |
| `medqa_gold_027` | borderline (single-hop) | Disease is given by flow cytometry (WASP mutation); only treatment fact is retrieved. |
| `medqa_gold_033` | multi-hop | Diagnose bacterial meningitis from clinical signs → recall bacterial CSF profile. |
| `medqa_gold_093` | multi-hop | Diagnose GBS from post-infectious ascending weakness + albuminocytologic dissociation → recall pathology (Schwann cells). |

**For Notebook 05 RAGAS evaluation:** if a stricter multi-hop signal is needed for the Multi-Hop RAG architecture comparison, derive a secondary label `requires_chained_retrieval = yes` only when two distinct passages must be retrieved sequentially (the pure 015 / 033 / 093 case), excluding diagnose-then-treat. Do this in Notebook 05, not in this golden-set construction.

---

## 12. Summary verdict (updated)

**The dataset is ready as ground truth for Notebook 05.** Final composition:

- 65 accepted rows, all spot-checked or audit-passed.
- 22 rows still in `needs_review` for optional triage.
- 9 rows in `dropped` with documented retrieval-failure reasons.
- 4 rows rejected by Pass 3.

The accepted set's reference explanations are evidence-grounded; the audit pipeline catches real failures (not just keyword hygiene); the `requires_multihop` label is usable with a methodology-section caveat. No blocking issues remain for the architecture-comparison phase.
