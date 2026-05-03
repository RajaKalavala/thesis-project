# Golden Dataset — Construction Methodology

Step-by-step description of the pipeline implemented in [notebooks/colab/04_golden_dataset_construction.ipynb](../../notebooks/colab/04_golden_dataset_construction.ipynb). Every stage checkpoints to disk so failed stages can be re-run without redoing prior work.

---

## Design principles

1. **Reuse, don't reinvent.** Chunking (200-token semantic), embedding (`all-MiniLM-L6-v2`), and retrieval config match Notebook 02a exactly so the golden-set retrieval is *identical* to what the four RAG architectures will use at evaluation time.
2. **Three separate LLM passes**, not one mega-prompt. Different temperatures, different cognitive tasks, harder to rubber-stamp.
3. **Independent automated audit** after the LLM is done. Mechanical checks (string presence, ID validity) catch failure modes the LLM self-validation cannot.
4. **Idempotent.** The expensive index build (chunks + embeddings + ChromaDB) is cached in `data/indices/` and reused on every subsequent run.

---

## Stage 1 — Stratified sampling

| | |
|---|---|
| Input | `data/processed/medqa_dev.parquet` (1,272 questions) |
| Output | `data/processed/golden_seed_100.parquet` |
| Logic | (1) Sample 20 long-vignette questions (>200 words) first across both step buckets. (2) From the remainder, sample 40 Step 1. (3) From what's left, sample 40 Step 2&3. Result: 100 unique questions. |
| Random seed | 42 (reproducible) |

**Why dev, not test:** the test split (1,273 questions) is reserved for the final architecture comparison. Dev is the fair location for ground-truth construction.

**Why 100, not 1,272:** RAGAS evaluation needs evidence + reference per row, and creating those at LLM cost ($0.03/row) for the full dev split would be ~$40 with no statistical benefit over a stratified sample.

---

## Stage 2 — Build retrieval index (cached)

| | |
|---|---|
| Input | `data/processed/textbook_corpus.json` (18 books, ~12.85M words) |
| Output | `data/indices/{chunks.parquet, embeddings.npy, chroma_textbooks/}` |
| Behavior | Idempotent — loads cached artifacts if present, rebuilds otherwise. |

**Chunking:** `RecursiveCharacterTextSplitter` with `chunk_size=200 tokens`, `overlap=20 tokens`. Separators ordered from coarse paragraph breaks down to spaces. Drops chunks <10 words.

**Embedding:** `all-MiniLM-L6-v2` (384 dimensions, normalized). Batch size 256, runs in ~5 min on T4 GPU.

**Indices built:** ChromaDB (dense, cosine), BM25Okapi (sparse). The same indices serve this notebook and every downstream RAG-architecture notebook.

---

## Stage 3 — Retrieve top-10 candidate evidence

| | |
|---|---|
| Input | seed parquet + indices |
| Output | `data/processed/golden_candidates.jsonl` |
| Per question | 10 candidate textbook chunks + metadata |

**Search query construction:**
```
query = question + correct_answer + metamap_phrases (joined)
```
Including the gold answer in the query is allowed here — this is golden-set construction, not system evaluation. The bias toward retrieving answer-supporting chunks is the point.

**Retrieval:**
1. Dense retrieval (ChromaDB, top-30).
2. Sparse retrieval (BM25, top-30).
3. Reciprocal Rank Fusion (k=60) over the two ranked lists.
4. Keep top-10 fused chunks.

---

## Stage 4 — Pass 1: GPT-4 evidence selection

| | |
|---|---|
| Input | candidates JSONL |
| Output | `data/processed/golden_evidence_selected.jsonl` |
| Model | `gpt-4o` · `temperature=0` · JSON mode |

**Prompt summary:** "You will receive a question + correct answer + 10 candidate passages. Identify which passages directly support the correct answer. Reject irrelevant or insufficient ones. Return strict JSON."

**Produced fields:** `selected_chunks` (with `support_level` ∈ {strong, moderate, weak} and `reason`), `best_gold_context` (verbatim concatenation of the strongest 1–3 chunks), `evidence_keywords` (medical terms that should appear in the evidence), `is_evidence_sufficient` (boolean), `review_note`.

---

## Stage 5 — Pass 2: GPT-4 reference answer + explanation

| | |
|---|---|
| Input | evidence-selected JSONL |
| Output | `data/processed/golden_with_references.jsonl` |
| Model | `gpt-4o` · `temperature=0.2` · JSON mode |

Slightly higher temperature here because Pass 2 is the only *generation* task in the pipeline (not a judging task).

**Prompt summary:** "Given the question, correct answer, and gold evidence, write a single-sentence reference answer + 3–6 sentence explanation grounded in the evidence. Do not introduce claims the evidence does not support."

**Produced fields:** `reference_answer`, `reference_explanation`, `why_other_options_are_less_suitable`, `hallucination_check_points` (claims a faithful generation must support), `question_type` ∈ {diagnosis, treatment, mechanism, management, other}, `requires_multihop` ∈ {yes, no}.

---

## Stage 6 — Pass 3: GPT-4 validation

| | |
|---|---|
| Input | references JSONL |
| Output | `data/processed/golden_validated.jsonl` |
| Model | `gpt-4o` · `temperature=0` · JSON mode |

**Prompt summary:** "Validate this golden sample. Score evidence relevance, faithfulness, explanation quality (each 0–5). Assess hallucination risk (low/medium/high). Decide final_status: accepted / needs_review / rejected."

**Caveat — the `answer_match` field is structurally tautological.** Pass 2 is given the correct answer as input, so Pass 3 always reports `answer_match: True`. Do not cite this as a quality signal in the thesis. Use the per-axis 0–5 scores and `hallucination_risk` instead.

---

## Stage 7 — Automated audit

| | |
|---|---|
| Input | validated JSONL + `chunk_metadata` from Stage 2 |
| Output | `data/processed/golden_audited.jsonl` |
| Model | None — pure Python checks |

Three mechanical checks the LLM cannot rubber-stamp:

1. **`correct_answer_not_in_reference_answer`** — does the `reference_answer` text contain the MedQA gold-answer string? Catches Gemini/GPT-4 disagreeing with the verified label.
2. **`no_evidence_keyword_in_gold_context`** — do the declared `evidence_keywords` actually appear in `best_gold_context`? Catches keyword hallucination by Pass 1.
3. **`invalid_chunk_ids:N`** — are the cited `chunk_id`s real entries in the index? Catches LLMs inventing chunk identifiers.

**Final `quality_status` rule:**
- Any audit issue ⇒ `needs_review` regardless of LLM verdict.
- No audit issues + LLM verdict `accepted` ⇒ `accepted`.
- No audit issues + LLM verdict `rejected` ⇒ `rejected`.
- Otherwise ⇒ `needs_review`.

---

## Stage 8 — Save final dataset

| | |
|---|---|
| Input | audited JSONL |
| Output | `data/processed/medqa_ragas_golden.jsonl` (accepted only) + `medqa_ragas_golden_needs_review.jsonl` |

The accepted file contains a flat per-row schema described in [schema.md](schema.md). It is the input to the (still-pending) Notebook 05 RAGAS evaluation harness.

---

## Cost & runtime profile (gpt-4o, 100 questions)

| Phase | Wall time (T4 GPU) | API cost |
|---|---|---|
| Stage 2 (index build, first run) | ~10 min | $0 |
| Stage 2 (index load, subsequent runs) | ~30 s | $0 |
| Stage 3 (retrieval) | ~1 min | $0 |
| Stage 4 (Pass 1) | ~10 min | ~$1.70 |
| Stage 5 (Pass 2) | ~8 min | ~$0.85 |
| Stage 6 (Pass 3) | ~6 min | ~$0.45 |
| Stages 7–8 | <5 s | $0 |
| **Total (cold)** | **~30–45 min** | **~$3.00** |
| Total (warm) | ~25 min | ~$3.00 |

A 5-question smoke test runs end-to-end in ~15 min for ~$0.15.

---

*Source code: [notebooks/colab/04_golden_dataset_construction.ipynb](../../notebooks/colab/04_golden_dataset_construction.ipynb)*
