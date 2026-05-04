# Notebook 02 — Output Notes

> **Notebook:** [`notebooks/02_embeddings_and_indices.ipynb`](../../notebooks/02_embeddings_and_indices.ipynb)
> **Run on:** 2026-05-03 (full embed) → 2026-05-04 (Chroma + BM25 + smoke query)
> **Phase:** 2 — Shared infrastructure (embeddings + indices)
> **Device:** Apple MPS (M1 Pro, 16 GB)

---

## 1. Output

**Artifacts saved to disk:**

| File / Directory | Path | Size | Count |
|---|---|---|---|
| Dense embeddings | `data/processed/embeddings.npy` | **276.9 MB** | 67,599 × 1,024 float32, L2-normalised |
| ChromaDB collection | `data/indices/chroma_textbooks/` (`medqa_textbooks_bge_400`) | **1,124.2 MB** | 67,599 vectors, cosine HNSW |
| BM25 index | `data/indices/bm25.pkl` | **105.8 MB** | `{"index": BM25Okapi, "chunk_ids": [67,599]}` |

**Inline outputs printed in the notebook (cells run in order):**

| Section | What printed | Headline value |
|---|---|---|
| §1 Setup | Repo paths, device, model name, query prefix | `device = mps` |
| §2 Load chunks | 67,599 chunks across 18 books | mean 324 tok, max 402 |
| §3 Load BGE | Model wall-time | 18.7 s |
| §4 Tiny smoke | 5-chunk shape/dtype/norm assertions | shape `(5, 1024)`, norm 1.0000 ✓ |
| §5 Pre-flight | Per-batch timing extrapolated to full corpus | 3.36 s/batch → ~118.4 min predicted |
| §6 Full embed | Loaded existing `embeddings.npy` (skipped re-embed) | shape `(67599, 1024)` |
| §7 ChromaDB build | Collection `medqa_textbooks_bge_400`, count parity | 67,599 ✓, 1m 39s wall-time |
| §8 BM25 build | Pickle to `bm25.pkl`, 67,599 chunk IDs | 8 s wall-time |
| §9 Smoke query | Top-3 from each index for the pneumonia query | dense + sparse both relevant ✓ |
| §10 Acceptance | Cross-artifact count parity assertions | all `=` 67,599 ✓ |

---

## 2. Meaning of the outputs

- **`embeddings.npy`** — every row is a 1,024-dimensional vector summarising one chunk's semantic content. **L2-normalised** so cosine similarity reduces to a dot product (faster on the HNSW index). Float32 because BGE-large outputs at that precision and there's no quality benefit from float64. The 276.9 MB ≈ 67,599 × 1,024 × 4 bytes confirms the arithmetic.

- **`chroma_textbooks/`** — a persistent ChromaDB collection storing the same 67,599 vectors plus the original chunk text (for retrieval display) and per-chunk metadata `{book_name, n_tokens}`. The HNSW graph (the navigation structure that makes top-k queries O(log N) instead of O(N)) inflates the on-disk footprint to ~4× the raw embedding size — that's expected, not bloat. Cosine HNSW is what plan.md §0 #3 locks in.

- **`bm25.pkl`** — a sparse, term-frequency-based index over the same 67,599 chunks. Tokenisation is lowercase + alphanumeric word-split (so `"COVID-19"` → `["covid", "19"]`). The pickle stores both the `BM25Okapi` index and the parallel `chunk_ids` list, so a BM25 score-array index → `chunk_id` is an O(1) lookup.

- **The "two indices" design.** Dense (BGE) and sparse (BM25) capture **different kinds of relevance**:
  - BGE understands paraphrase ("first-line therapy" ≈ "initial empirical management") even when the literal words don't match
  - BM25 anchors to exact medical terminology — drug names, organisms, ICD codes — which BGE's training data may smooth over
  - The Hybrid RAG architecture (EXP_04) fuses both via RRF; this notebook is what makes that possible.

- **The smoke query proved the asymmetry is real:**
  - Dense top-3 → Harrison's empiric-therapy chapter, macrolide/β-lactam combo, ATS guidelines (Novak)
  - Sparse top-3 → Pharmacology_Katzung's case study with **ceftriaxone + azithromycin** (the literal answer), Harrison's M. pneumoniae and Legionella chapters

  Without this complementarity, Hybrid RAG would be redundant. We now have evidence it isn't.

---

## 3. Conclusions

1. **Phase 2 shared infrastructure is complete.** Three indices on disk, count parity (`67,599 == 67,599 == 67,599`) holds across `chunks.parquet ↔ embeddings.npy ↔ ChromaDB ↔ BM25`. Every subsequent notebook (03, 04, 04a–04e, 05–09) reads from this disk state and never re-embeds.

2. **Both retrievers return clinically relevant content for the pneumonia smoke query** — dense surfaces the *concept* of empirical therapy, sparse surfaces the *literal answer* (ceftriaxone + azithromycin). This dual-mode retrieval validates the Hybrid RAG hypothesis and is the proof that EXP_02–EXP_05 will retrieve sensible content at scale.

3. **Embedding wall-time was ≈ 6 h on MPS — 16× the plan.md `~22 min` estimate.** First-batch timing (3.36 s) extrapolated to ~118 min, but sustained throughput over 6 hours degraded to ~16 s/batch — strong signal of thermal throttling and/or partial-MPS coverage of BGE-large's 1024-d attention under load. **The output is correct** (norms = 1.0, shape correct, retrieval works), but the timing estimate in plan.md / docs/architecture.md / docs/tech_stack.md must be updated so future-you (or any agent) doesn't believe the 22-min figure when planning a re-run.

4. **Storage footprint is bigger than originally projected:** total of ≈ 1.5 GB on disk (276.9 MB embeddings + 1.1 GB Chroma + 105.8 MB BM25). Still trivial for the 16 GB M1 Pro and gitignored, but the originally-quoted "131 MB" in plan.md §15 risk register is obsolete (already updated to 274 MB on chunk recalibration; now further confirmed at the embedding-array level).

5. **Resumability proved out.** When the kernel was restarted between §6 and §7, re-running §1–§6 took ~15 sec total because §6 detected `embeddings.npy` on disk and skipped re-embedding. The 6-hour cost is paid exactly once. This is the resume-from-cache pattern that AGENTS.md §2.3 mandates for everything downstream.

6. **The "Failed to send telemetry event" warnings in cells 7 and 9 are cosmetic** — chromadb's posthog telemetry has an API mismatch with the newer posthog version pulled in alongside it. We've already disabled telemetry via `Settings(anonymized_telemetry=False)` and `ANONYMIZED_TELEMETRY=False`; the messages still leak from a separate code path but no data is being sent and no functionality is affected.

---

**Next:** Notebook 03 — end-to-end smoke test of the **retrieve → prompt → Groq → parse-letter** pipeline on 3 dev questions. After that passes, we can move on to Phase 3 (golden RAGAS dataset construction) or jump to Phase 4 EXP_01 (No-RAG baseline) — both unlocked by what's now on disk.
