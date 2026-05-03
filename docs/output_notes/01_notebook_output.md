# Notebook 01 — Output Notes

> **Notebook:** [`notebooks/01_chunking_and_corpus_prep.ipynb`](../../notebooks/01_chunking_and_corpus_prep.ipynb)
> **Run on:** 2026-05-03
> **Phase:** 2 — Shared infrastructure (chunking)

---

## 1. Output

**Artifact saved to disk:**

| File | Path | Size | Rows |
|---|---|---|---|
| `chunks.parquet` | `data/processed/chunks.parquet` | ~30 MB | **67,599** |

**Schema of `chunks.parquet`:**

| Column | Type | Example |
|---|---|---|
| `chunk_id` | string | `InternalMed_Harrison_chunk_06494` |
| `book_name` | string | `InternalMed_Harrison` |
| `text` | string | the actual passage content (≤ 400 tokens, ≥ 30 tokens) |
| `n_tokens` | int64 | `388` |
| `n_chars` | int64 | `1,899` |

**Inline outputs printed in the notebook:**

| Section | What it shows |
|---|---|
| §2 Inventory | All 18 textbook files with sizes in MB (sorted) |
| §3 Smoke test | First_Aid_Step1 → 547 chunks, mean 359 tokens, max 399 |
| §4 Full corpus | 67,599 chunks, mean 323.9 tokens, median ~330, min 30, max 400 |
| §5 Acceptance | 5 assertions all ✓ (count band, mean, max, Harrison's, ID uniqueness) |
| §6 Bar chart | Per-book chunk count, horizontal bars |
| §6 Histogram | Token-count distribution across all 67,599 chunks |

---

## 2. Meaning of the outputs

- **`chunk_id`** — the **primary key** for every chunk. Format `<book_stem>_chunk_<NNNNN>`. Deterministic: same input + same chunker config ⇒ same IDs forever. Every later notebook references chunks *by this ID*, so stability matters.

- **`book_name`** — source attribution. Tells us which of the 18 textbooks a chunk came from. Used by per-book bar charts, the corpus-bias check, and the methodology's coverage analysis.

- **`text`** — the actual passage content. This is what BGE-large will embed (Notebook 02) and what the LLM eventually reads inside the prompt (Phase 4 experiments).

- **`n_tokens`** — cl100k token count, used downstream to confirm `chunk + question + prompt template` fits inside LLaMA's 131k context window. Also used by RAGAS to compute context length statistics.

- **`n_chars`** — character count. UI/logging only — never used for ML decisions.

- **Mean ≈ 324 tokens (vs the 400-token cap)** — `RecursiveCharacterTextSplitter` fills to ~80 % of the cap because it respects paragraph/sentence boundaries *before* reaching 400 tokens. Normal and expected; smaller chunks slightly favour **retrieval precision** over recall.

- **Harrison's share = 24.66 %** — Harrison's contributes 16,669 of 67,599 chunks. Compared to its raw-corpus word-count share of 24.95 % (from Notebook 00), the gap is just 0.3 pp ⇒ chunking treated all 18 books uniformly. No book got over- or under-sampled.

- **All acceptance assertions ✓** — green light to proceed to Notebook 02. If any had failed, embedding wastes 45 min of CPU on broken chunks.

---

## 3. Conclusions

1. **The chunk corpus is built and audited.** 67,599 deterministic chunks, no boilerplate (drops < 30 tokens), no oversize chunks (max ≤ 400).

2. **Corpus mix preserved.** Per-book chunk-count shares track word-count shares to within 0.3 pp ⇒ chunking introduced no bias. The Harrison's dominance (≈ 25 %) is a known *corpus* property — flagged in `plan.md §15` and the thesis Limitations chapter.

3. **Original "~36k chunks" estimate was mathematically unreachable** for a 12.85 M-token corpus with 400/80 config — formal recalibration to 50k–75k is now reflected in `plan.md §0 #5`, `docs/architecture.md`, `docs/tech_stack.md`, `docs/todo.md` decision log, and project memory. Actual count of 67,599 sits in the middle of the new band.

4. **Single source of truth on disk.** All later steps read from this one file:
   - Notebook 02 → BGE-large embeddings + ChromaDB + BM25
   - EXP_01–EXP_05 → 5 baseline RAG experiments
   - Notebook 04 → golden-set audit (verifies every cited `chunk_id` resolves here)
   - EXP_10–EXP_12 → LIME / SHAP passage attribution

5. **Knock-on effects of the higher chunk count.** BGE embedding takes ≈ 45–50 min on M1 Pro CPU (or ≈ 22 min on Apple MPS), not the originally predicted 25/12 min. Embedding array on disk will be ≈ 274 MB float32, not 131 MB. Both fit comfortably on the 16 GB M1 Pro; documented in `plan.md §15` risk register.

---

**Next:** Notebook 02 turns these chunks into searchable indices — BGE-large dense embeddings → ChromaDB collection, plus a parallel BM25 sparse index. After that, Notebook 03 smoke-tests the end-to-end retrieve → generate path on 3 dev questions before any Phase 4 experiment touches the full 12,723.
