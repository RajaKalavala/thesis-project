# Raja's Notes

> Personal study notes for the thesis — short, plain-language recaps of the decisions I'll need to defend in the viva. Grows entry by entry as the project moves forward.

---

## Chunking — Recursive vs Semantic Passage

- **What I used: `RecursiveCharacterTextSplitter` (400 tokens, 80-token overlap, tiktoken cl100k_base).** It walks down a separator list — paragraphs (`\n\n`) → sentences (`. `) → words (` `) → characters — and cuts at the strongest natural break that keeps the piece under 400 tokens. Free, fully deterministic (same input ⇒ same chunks every time), reproducible across machines and years, and the dominant configuration in 2024–25 medical-RAG papers (MIRAGE, MedRAG, BioRAG).

- **Why not Semantic Passage chunking.** It would have to embed every sentence to detect topic shifts, then cut where similarity drops — an expensive embedding pass _just to decide where to cut_ (~doubles compute), and the boundaries would shift any time the embedding model changes, invalidating every cached `chunk_id`. Textbook prose already has clean paragraph structure, so the recursive splitter captures the same natural boundaries for free.

- **Output of Notebook 01 (the conclusion).** Produced **67,599 chunks** across all 18 textbooks (mean ≈ 324 tokens, max ≤ 400, fragments < 30 tokens dropped), saved as 5 columns — `chunk_id`, `book_name`, `text`, `n_tokens`, `n_chars` — to `data/processed/chunks.parquet`. Harrison's share landed at **24.66%**, matching the EDA word-count share of 24.95% to within 0.3 pp — confirms chunking did **not** bias the corpus mix. This single file is the source of truth feeding every later step: embeddings (Notebook 02), BM25 (Notebook 02), all 5 baseline RAG experiments (EXP_01–05), the golden-set audit (Notebook 04), and the LIME/SHAP attribution (EXP_10–12).

### Corpus Bias — Harrison's Share

- **Harrison's share = the % of all chunks that come from `InternalMed_Harrison.txt`** (_Harrison's Principles of Internal Medicine_). It's the largest textbook in the corpus — 22.4 MB / ~3.2 M words, vs the next biggest at 11.5 MB — so it contributes **24.95 %** of the raw corpus by word count and **24.66 %** by chunk count (16,669 of 67,599). The 0.3 pp gap between the two is the green light: chunking treated all 18 books uniformly, no book got over- or under-sampled. The bias itself is **real and acknowledged**, not fixed — at retrieval time a random top-5 pulls ~1.25 chunks from Harrison's on average, so a USMLE question whose answer sits in Harrison's looks "easier" because the retriever has ~4× the surface area to find it (e.g., ~30 PE-related chunks in Harrison's vs ~2 in First_Aid_Step1). Flagged in the thesis **Limitations** section so the examiner can't surprise me with it.

---
