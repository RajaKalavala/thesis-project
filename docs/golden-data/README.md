# Golden Dataset for RAGAS Evaluation

A 100-question RAGAS-ready ground-truth dataset built from the **MedQA dev split** and the **18-textbook medical knowledge base**, used to evaluate the four RAG architectures in this thesis (Naive · Sparse · Hybrid · Multi-Hop Explainable).

---

## 1. Why this dataset exists

The thesis topic — *Systematic Comparison of Multiple Retrieval-Augmented Generative AI Architectures for Evidence-Based Medical Question Answering with Explainability and Hallucination Control* — cannot be evaluated using only MedQA's `correct_answer` field.

RAGAS metrics (Faithfulness, Context Recall, Context Precision, Answer Correctness, Answer Relevance) require:

- a **reference answer** in full sentence form,
- supporting **textbook evidence** with chunk-level traceability,
- explicit **claims** that a faithful generation must cover (for hallucination scoring).

MedQA only ships `(question, options, correct_answer, answer_idx)`. This folder documents the *enriched* golden dataset that adds the missing fields.

---

## 2. Project context — where this fits

| Stage in the thesis | Status | Artifact |
|---|---|---|
| Notebook 01 — preprocessing & EDA | Done | [docs/dataset_exploration.md](../dataset_exploration.md) |
| Notebook 02 / 02a — VectorDB & component experiments | Done | `notebooks/colab/02*` |
| Notebook 03 — embedding experiment | Done | `notebooks/colab/03_embedding_experiment_colab.ipynb` |
| **Notebook 04 — golden dataset construction** | **Done** *(this folder)* | [notebooks/colab/04_golden_dataset_construction.ipynb](../../notebooks/colab/04_golden_dataset_construction.ipynb) |
| Notebook 05 — RAGAS evaluation harness | **Pending** | — |
| 4 RAG architecture comparison run | **Pending** | — |
| 20-metric evaluation table | **Pending** | `experiments/ThesisExperiment.xlsx` |
| Thesis writeup | **Pending** | `docs/` |

---

## 3. Documents in this folder

| File | Read this when you want to know… |
|---|---|
| [methodology.md](methodology.md) | How the dataset was built — the 10-stage pipeline from sampling to filtering. |
| [schema.md](schema.md) | What each field in the final JSONL means and where it came from. |
| [analysis.md](analysis.md) | What the run produced — quality-control report on the 100-question batch. |
| [roadmap.md](roadmap.md) | What's done, what's pending, and the sequenced next steps. |

---

## 4. Quick stats — current run (2026-04-25)

| | |
|---|---|
| Source | MedQA dev split (1,272 questions) |
| Sample size | 100 questions, stratified (45 Step 1 · 55 Step 2&3 · 23 long-vignette overlap) |
| Knowledge base | 18 textbooks · ~12.85M words → **~70k chunks** at 200 tokens |
| Embedding | `all-MiniLM-L6-v2` (384-d, normalized) |
| Retrieval | Hybrid · ChromaDB (dense) + BM25 (sparse) · RRF fusion (k=60) · top-10 candidates |
| LLM judge | OpenAI **GPT-4** (`gpt-4o`) — 3 passes per question |
| **Final accepted rows (post manual review)** | **65 / 100** |
| Needs human review | 22 / 100 |
| Dropped (manual review) | 9 / 100 |
| Rejected (Pass 3) | 4 / 100 |
| Build time | ~30–45 min on T4 GPU |
| API cost | ~$3 |

---

## 5. Files produced

### Construction outputs — `data/processed/`

```
golden_seed_100.parquet                  # the 100 sampled questions (raw seed)
golden_candidates.jsonl                  # top-10 retrieved candidates per question
golden_evidence_selected.jsonl           # GPT-4 Pass 1 (evidence selection)
golden_with_references.jsonl             # GPT-4 Pass 2 (reference answers)
golden_validated.jsonl                   # GPT-4 Pass 3 (validation scores)
golden_audited.jsonl                     # full record + automated audit verdicts
medqa_ragas_golden.jsonl                 # FINAL — 65 accepted samples ←
medqa_ragas_golden_needs_review.jsonl    # 22 rows awaiting optional triage
medqa_ragas_golden_dropped.jsonl         # 9 retrieval-failure rows dropped after manual review
```

### Reusable index — `data/indices/`

```
chunks.parquet                           # ~70k chunks with book_name + chunk_id + text
embeddings.npy                           # 70k × 384 float32 sentence-transformer embeddings
chroma_textbooks/                        # persistent ChromaDB collection
```

The index is built once and reused by every downstream notebook — turning a ~10 min build into a ~30 s load.

---

## 6. Source notebook

[notebooks/colab/04_golden_dataset_construction.ipynb](../../notebooks/colab/04_golden_dataset_construction.ipynb)

Run on Colab with a T4 GPU. Requires:
- `OPENAI_API_KEY` (Colab Secrets or inline)
- `medqa_dev.parquet` and `textbook_corpus.json` in `/content/drive/MyDrive/MedQA-Thesis/data/processed/`

---

*Companion to: [docs/dataset.md](../dataset.md) (structural reference) · [docs/dataset_exploration.md](../dataset_exploration.md) (EDA narrative)*
