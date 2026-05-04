# Notebook 03 — Output Notes

> **Notebook:** [`notebooks/03_smoke_test_pipeline.ipynb`](../../notebooks/03_smoke_test_pipeline.ipynb)
> **Run on:** 2026-05-04
> **Phase:** 2 — gate before any Phase-4 experiment touches the full 12,723

---

## 1. Output

**No new files saved to disk.** This notebook is the end-to-end smoke test of the Phase-2 pipeline; its purpose is verification, not artifact production. Side effect: 2 fresh Groq responses cached to `data/cache/groq/<key[:2]>/<key>.json` (Q1 + Q2; Q0 was cached during the §6 smoke and re-used in §7).

**Pipeline tested end-to-end:**

```
question (MedQA dev)
  └─► embed_queries (BGE prefix)   ← src/data/embedder.py
        └─► ChromaDB top-5         ← src/data/indices.py
              └─► build_evidence_grounded_prompt   ← src/generation/prompts.py
                    └─► groq_complete (cached)     ← src/generation/groq_client.py
                          └─► parse_letter         ← src/generation/prompts.py
```

**Per-question results:**

| # | Topic | Truth | Predicted | Correct? | Latency | Cached? |
|---|---|---|---|---|---|---|
| 0 | Gallstone ileus → bowel obstruction location | **B** Distal ileum | **B** | ✓ | 0.35 s | True (from §6 smoke) |
| 1 | Personality disorder vignette | **A** Avoidant | **B** Schizoid | ✗ | 3.40 s | False |
| 2 | HIV gp120 binding step | **D** Attachment to host CD4 | **D** | ✓ | 0.38 s | False |

**Aggregate:**
- 3 / 3 letters parsed cleanly (raw responses: `'B'`, `'B'`, `'D'`)
- Mean latency 1.38 s (≪ 5 s budget) — Groq's free tier comfortably handles 12,723 questions × 5 architectures
- Cache hit rate: 1 / 3 on this run (Q0 reused §6's smoke cache); will be 100 % on every subsequent re-run of cells §6–§7

---

## 2. Meaning of the outputs

- **The 3 raw response strings — `'B'`, `'B'`, `'D'` — are themselves a result.** LLaMA 3.3 70B obeyed the instruction *"Output exactly one letter (A, B, C, D, or E). Nothing else."* on all three calls. No preamble, no "The answer is…", no explanations. That's the simplest possible parsing case. The permissive `parse_letter` regex (which would handle "The answer is C." or "A) something") wasn't needed here, but it stays in the code as belt-and-suspenders for Phase 4 when 12,723 questions × stochastic generation will produce some chatty responses.

- **Latency profile (0.35 / 3.40 / 0.38 s)** shows expected variance:
  - The two **non-cached** calls (Q1, Q2) took 3.40 s and 0.38 s — Groq's response time depends on prompt length and current load. Q1's prompt happens to be longer (~2,800 chars of question + retrieved chunks), Q2's prompt is denser but shorter.
  - Q0's 0.35 s in §7 is *not* the actual cache-lookup time — `groq_complete` returns the **original call's latency** when cached, so timing is comparable across runs. Cache lookup itself is microseconds.

- **Cache key works as designed.** `sha256(provider + model + temperature + prompt)` = a 64-character hex string. Same prompt + model + temp at any point in the future returns the cached response instantly. Re-running the notebook — even after a kernel restart, even tomorrow — will produce identical results at zero Groq cost. AGENTS.md §2.3 satisfied.

- **The Q1 wrong answer reveals a real retrieval limitation, not a pipeline bug.** Top-5 for Q1 returned: borderline personality disorder · borderline mood/relationships · referral framing · DSM cluster organization · interview technique. **None of the five chunks specifically discuss "Avoidant" or Cluster C content.** BGE's semantic search pulled *general* personality-disorder material rather than *label-specific* content because the question describes symptoms but never names the answer label. The LLM, given only those five chunks, picked the most plausible disorder *that was mentioned in the chunks* (Schizoid in Cluster A) rather than the correct one (Avoidant in Cluster C). This is precisely the failure mode that motivates Hybrid RAG — a BM25 search on terms like "avoidant", "social inhibition", "embarrassment" might surface the Cluster C chapter directly.

- **Q0 and Q2 retrieval was clearly on-topic.** Q0's top-5 are bowel-obstruction and gallstone-ileus chunks from Harrison's, First_Aid, Surgery_Schwartz; Q2's top-5 are HIV/gp120 chunks from Histology_Ross, Immunology_Janeway, Pathology_Robbins, Harrison's. Dense retrieval is doing the right thing for questions where the answer concepts are in the question stem.

---

## 3. Conclusions

1. **The full retrieve → prompt → Groq → parse pipeline works end-to-end** on real MedQA dev questions. All Phase 4 experiments now have a tested code path; nothing is hypothetical.

2. **Acceptance ([docs/todo.md §2.3](../todo.md)) passed:** 3 / 3 letters parsed cleanly, mean latency 1.38 s ≪ 5 s budget, side-by-side display renders cleanly.

3. **Disk cache is operational.** Re-running the entire notebook (after the first run) costs nothing on Groq. This is the foundation for resumable Phase 4 experiments — a 12,723-question Groq run that hits a rate limit at hour 5 can be restarted from where it stopped, paying only for the questions that hadn't been cached yet.

4. **The Q1 retrieval miss is a methodology finding worth recording in the thesis writeup.** It demonstrates concretely why Naive RAG (dense top-k alone) is insufficient: when the answer label isn't named in the question stem, semantic similarity can pull general-topic content while missing the specific clinical label. This is the failure mode Hybrid RAG (EXP_04) is hypothesised to fix.

5. **Phase 2 is now fully complete.** All three notebooks (01 chunking, 02 embeddings + indices, 03 smoke test) have run end-to-end with verified outputs:
   - `chunks.parquet` (67,599 rows) ✓
   - `embeddings.npy` (67,599 × 1,024) ✓
   - `chroma_textbooks/` (67,599 vectors, cosine HNSW) ✓
   - `bm25.pkl` (67,599 chunk_ids) ✓
   - `data/cache/groq/` (3 cached responses) ✓
   - `src/` skeleton — 5 modules tested in production code paths

6. **Two cosmetic notes for future reference.** (a) The "Failed to send telemetry event" warnings in cells using ChromaDB are harmless — chromadb's posthog telemetry has a method-signature mismatch with the newer posthog client; we've already disabled telemetry, the warnings still print but no data leaves the machine. (b) BGE on MPS for **query encoding** (single query at a time) is fast and not affected by the sustained-load thermal throttling that slowed Notebook 02's full embed pass.

---

**What's now safe to do:**
- **Phase 3** — Notebook 04, golden RAGAS dataset construction (needs `OPENAI_API_KEY`)
- **Phase 4** — EXP_01 → EXP_05 baseline RAG experiments on the full 12,723 (needs `ANTHROPIC_API_KEY` for the RAGAS judge)

The pipeline plumbing is verified. Whatever happens next is a question of LLM behaviour and retrieval quality at scale — the infrastructure is no longer a variable.
