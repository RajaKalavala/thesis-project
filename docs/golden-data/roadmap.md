# Golden Dataset — Roadmap

What has been completed, what remains, and the sequenced next steps for getting from "dataset built" to "thesis ready for write-up".

Last updated: **2026-04-25**.

---

## Done — golden dataset construction

| # | Task | Output |
|---|---|---|
| 1 | Notebook 04 implemented (10-stage pipeline) | [notebooks/colab/04_golden_dataset_construction.ipynb](../../notebooks/colab/04_golden_dataset_construction.ipynb) |
| 2 | Stratified sampling of 100 questions from MedQA dev | `data/processed/golden_seed_100.parquet` |
| 3 | Persistent index built (chunks + embeddings + ChromaDB) | `data/indices/{chunks.parquet, embeddings.npy, chroma_textbooks/}` |
| 4 | Hybrid retrieval (RRF) — top-10 candidates per question | `data/processed/golden_candidates.jsonl` |
| 5 | GPT-4 Pass 1 — evidence selection | `data/processed/golden_evidence_selected.jsonl` |
| 6 | GPT-4 Pass 2 — reference answer + explanation | `data/processed/golden_with_references.jsonl` |
| 7 | GPT-4 Pass 3 — validation scores | `data/processed/golden_validated.jsonl` |
| 8 | Automated audit (3 mechanical checks) | `data/processed/golden_audited.jsonl` |
| 9 | Filter & save final dataset | `data/processed/medqa_ragas_golden.jsonl` (initially 64 accepted rows) |
| 10 | Quality-control analysis report | [analysis.md](analysis.md) |
| 11 | Documentation folder created | `docs/golden-data/` |
| 12 | **P1 — manual review of 10 keyword-flagged rows** | 9 dropped → `medqa_ragas_golden_dropped.jsonl`; 1 salvaged into accepted. Detail in [analysis.md §11.1](analysis.md). |
| 13 | **P2 — grounding spot-check on 5 random accepted rows** | 5 / 5 properly grounded. [analysis.md §11.2](analysis.md). |
| 14 | **P3 — multi-hop label audit on 5 random rows** | 3 clear · 2 borderline. Label kept with methodology-section caveat. [analysis.md §11.3](analysis.md). |
| 15 | Final accepted dataset count post-review | **65 rows** — `data/processed/medqa_ragas_golden.jsonl` |

---

## Optional — additional triage (not blocking)

The 22 rows still in `medqa_ragas_golden_needs_review.jsonl` could potentially be recovered by re-running Pass 2 with a stricter prompt that requires the explanation to **quote** specific phrases from `gold_context`. This would lift the accepted count further.

| | |
|---|---|
| Effort | ~20 min Colab time + ~$0.50 GPT-4 cost |
| Source | the 22 needs_review rows |
| Method | Re-run Pass 2 on those rows with `temperature=0` and an explicit "ground every claim in the supplied evidence — quote the relevant phrase in parentheses" instruction |
| Expected outcome | 5–12 additional rows recovered into accepted (final accepted count 70–77) |
| Required if | The 65 accepted rows yield insufficient stratified slices for the per-architecture comparison |
| Skip if | 65 is comfortable for the comparison (likely yes for 4 architectures × 65 = 260 evaluation cells) |

---

## Pending — downstream of the golden dataset

These are the steps that consume `medqa_ragas_golden.jsonl` and produce the thesis comparison results.

### N1 — Build Notebook 05: RAGAS evaluation harness

| | |
|---|---|
| Goal | A reusable evaluator that runs one RAG architecture against the golden dataset and produces a metrics row. |
| Inputs | `medqa_ragas_golden.jsonl` + a function that, given a question, returns `(generated_answer, retrieved_contexts, retrieved_chunk_ids)` |
| Outputs | One CSV per architecture in `results/{architecture}/eval_run_{date}.csv` |
| RAGAS metrics | Faithfulness · Answer Relevancy · Context Precision · Context Recall · Answer Correctness |
| Beyond-RAGAS metrics | Per the proposal's 20-metric framework — Accuracy · F1 · Token-F1 · ROUGE-1/L · BERTScore · Recall@3/5/10 · MRR · Hallucination check-point coverage |
| LLM judge for RAGAS | Use a different model family from Pass 1–3 (e.g. Claude or LLaMA via Groq) to avoid evaluator-on-evaluator bias |

### N2 — Run the four RAG architectures

| Architecture | Retrieval | Status |
|---|---|---|
| Naive RAG | dense (ChromaDB) | not yet run on golden |
| Sparse RAG | BM25 | not yet run on golden |
| Hybrid RAG | dense + BM25, RRF k=60 | not yet run on golden |
| Multi-Hop Explainable RAG | iterative, up to 3 hops | not yet run on golden |

All four share: `gpt-4o` (or LLaMA 3.3-70B via Groq) as the answerer, `all-MiniLM-L6-v2` embeddings, the same prompt template. Only the retrieval strategy varies.

### N3 — Cross-tabulate results

| Architecture | Accuracy | Faithfulness | Context Precision | Context Recall | Answer Correctness | … |
|---|---|---|---|---|---|---|
| Naive | | | | | | |
| Sparse | | | | | | |
| Hybrid | | | | | | |
| Multi-Hop | | | | | | |

Stratified slices: `meta_info` (Step 1 vs Step 2&3), `question_type`, `requires_multihop`.

### N4 — Thesis writeup

- Methodology chapter — sections lifted from [methodology.md](methodology.md) and [analysis.md](analysis.md).
- Results chapter — built from the cross-tab.
- Discussion — what the per-stratum results say about each architecture.

---

## Suggested order of operations

1. **Today (1–2 hours):** complete P1, P2, P3 on the golden dataset.
2. **Next session:** start N1 (Notebook 05). Use the persistent index from `data/indices/`.
3. **Following session:** run N2 — all four architectures over the golden 100. Expected runtime ~1 hour, expected API cost ~$5–8 with `gpt-4o`.
4. **After that:** N3 cross-tab and analysis.
5. **Then:** thesis writing.

---

## Risks worth tracking

| Risk | Why it matters | Mitigation |
|---|---|---|
| Multi-hop label is unreliable | Inflates apparent Multi-Hop RAG win | P3 above resolves this |
| Same LLM family for construction *and* RAGAS judging | Evaluator-on-evaluator bias | Use a different model family for RAGAS in N1 |
| Corpus skew toward Harrison's | Affects absolute retrieval picture, not relative | Acknowledge in discussion; no fix needed for comparison |
| 64 accepted rows may shrink with stricter manual review | Sample size for stratified analysis | If the count drops below ~40, consider running another batch from dev to top up |

---

*See also: [README.md](README.md) (overview) · [methodology.md](methodology.md) · [schema.md](schema.md) · [analysis.md](analysis.md)*
