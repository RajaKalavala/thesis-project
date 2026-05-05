# Chat Context Handoff — 2026-05-05

> **Purpose.** Self-contained context dump so a new chat session can pick up the project at the right point without re-reading the full prior conversation.
> **Coverage.** Phase 1 (✅) → Phase 2 (✅) → Phase 3 (✅). Phase 4 not started.
> **Last working session ended:** Phase 3 production-run complete; all docs synchronised; ready to start Phase 4 (EXP_01 → EXP_05).

---

## 1. Where the project stands (quick orientation)

**Thesis:** *"Systematic Comparison of Multiple Retrieval-Augmented Generative AI Architectures for Evidence-Based Medical Question Answering with Explainability and Hallucination Control."*
MSc AI/ML, LJMU. Submission target **March 2026**.

**Phase status:**

| Phase | Status | Notes |
|---|---|---|
| 1 — Data processing & EDA (Notebook 00) | ✅ DONE | 12,723 MedQA US questions; 18 textbooks; Harrison's = 24.95 % of corpus |
| 2 — Shared infrastructure (Notebooks 01 + 02 + 03) | ✅ DONE | 67,599 chunks · BGE-large embeddings · ChromaDB + BM25 · pipeline smoke test passed |
| 3 — Golden RAGAS dataset (Notebook 04 main) | ✅ DONE | 234 accepted of 300 attempted at $6.61 |
| 4 — Group A baseline experiments (EXP_01–EXP_05) | ⬜ NEXT | Build `src/retrieval/` modules + 5 experiment notebooks |
| 5 — Adaptive routing (EXP_06–EXP_07) | ⬜ | |
| 6 — Explainability (EXP_10–EXP_12) | ⬜ | |
| 7 — Confidence-aware rejection (EXP_08–EXP_09) | ⬜ | |
| 8 — Hallucination taxonomy (EXP_13–EXP_15) | ⬜ | |
| 9 — Final synthesis (EXP_16) | ⬜ | |
| 10 — Streamlit demo UI (optional) | ⬜ | parallel track if time permits |

---

## 2. Locked decisions (current snapshot — verified post-revert)

These are **load-bearing** for the methodology defence. Do not re-debate without explicit user input.

| # | Decision | Value |
|---|---|---|
| 1 | LLM (generator/answerer) | `llama-3.3-70b-versatile` via Groq |
| 2 | Embedding model | `BAAI/bge-large-en-v1.5` (1024-d, 335M, 512-token max) |
| 3 | Vector DB | ChromaDB persistent, cosine HNSW |
| 4 | Sparse index | rank-bm25 (Okapi) |
| 5 | Chunking | recursive 400-token chunks, 80-token overlap (20 %) |
| 6 | Hybrid fusion | RRF k=60 |
| 7 | Multi-hop budget | 3 hops max |
| 8 | Evaluation surface | full 12,723 MedQA US |
| 9 | Golden RAGAS subset | 300 stratified; staged 50-pilot + 250-production. **234 accepted** in `data/processed/golden_ragas_300.jsonl` |
| 10 | Golden-set constructor | **`gpt-4o`** via OpenAI API (locked 2026-05-04 after A/B vs `gpt-oss-120b`) — three-pass JSON pipeline, new prompts |
| 11 | RAGAS judge | `claude-3-5-sonnet-20241022` via Anthropic |
| 12 | Top-k passed to LLM | k=5 |

**Three-family separation (load-bearing methodology):** Generator = Meta (LLaMA), Constructor = OpenAI (gpt-4o), Judge = Anthropic (Claude). Kills evaluator-on-evaluator bias.

**API keys in `.env` (all populated):** `GROQ_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`.

---

## 3. Notable discussions and decisions (chronological)

### 3.1 Notebook 01 — chunking recalibration
- Original plan estimated "~36k chunks" for 12.85 M-word corpus.
- Math showed this was unreachable: with 400-token chunks and 80-token overlap, advance ≤ 320 unique tokens per chunk → floor of ~52k chunks even with perfect packing.
- `RecursiveCharacterTextSplitter` fills to ~80 % of the 400-token cap (mean ≈ 324 tokens) → realistic count is ~67k.
- **Actual result: 67,599 chunks**, mean 323.9 tokens, Harrison's 24.66 % (vs 24.95 % word-share — chunking didn't bias the mix).
- All affected docs (plan.md §0 #5, §4, §15; tech_stack §3; architecture §8; todo §2.1; memory) were updated in lockstep.

### 3.2 Notebook 02 — embedding wall-time recalibration
- Original plan said ~22 min on Apple MPS.
- **Actual: ~355 min (~6 hours)** — 16× overshoot.
- Cause: thermal throttling and/or partial-MPS coverage of BGE-large's 1024-d attention under sustained load. First batch took 3.36 s; sustained throughput degraded to ~16 s/batch.
- Output is correct (norms = 1.0, shape (67599, 1024)). Cell §6 is resumable via `embeddings.npy` — cost paid exactly once.
- ChromaDB on-disk size: **1,124 MB** (HNSW graph adds ~4× the raw embedding size — expected, not bloat).
- BM25 size: 105.8 MB.
- Smoke query *"first-line treatment for community-acquired pneumonia"* validated dense ↔ sparse asymmetry: dense → Harrison's empirical-therapy chapter; sparse → Pharmacology_Katzung's literal `ceftriaxone + azithromycin` answer. **Confirms Hybrid RAG (EXP_04) has real signal to fuse.**

### 3.3 Notebook 03 — pipeline smoke test
- 3 dev questions through retrieve → prompt → Groq → parse path.
- **2/3 correct, mean latency 1.38 s, all letters parsed cleanly.**
- **Q1 (avoidant personality vignette) wrong answer is methodologically informative**, not a pipeline bug. Top-5 retrieved general personality-disorder content but no Cluster-C / Avoidant specifics. Demonstrates why Naive RAG fails when answer label isn't named in the question stem; motivates Hybrid RAG.
- All 4 src modules (`utils/cache.py`, `generation/{prompts,groq_client}.py`) verified in production code path.

### 3.4 Phase 3 saga — the golden-dataset construction journey

This was the longest discussion arc. Key chapters:

**3.4.1 Cost-vs-methodology debate (constructor swap proposal).**
- User asked whether to use open-source models instead of OpenAI/Claude to save money.
- Discussion landed on path #2: **swap gpt-4o → gpt-oss-120b on Groq** (free), keep Claude as judge. Preserved 3-family separation.
- All docs updated to reflect the swap.

**3.4.2 Cost reality-check.**
- User asked "Will Groq not cost me money to run this with 300 questions?"
- Honest answer: free tier handles it (slow, ~30 RPM rate limit) OR ~$0.57 paid for the full 300-row build. Updated estimates to "$0–$1" instead of strict "$0".
- Reasoning-token caveat surfaced: gpt-oss-120b is a reasoning model; hidden reasoning tokens inflate `completion_tokens` ~3× beyond visible JSON.

**3.4.3 First pilot run with gpt-oss-120b.**
- 50 questions, 3 passes each, 150 LLM calls.
- **Result: 21/50 accepted (42 %), 0 % JSON malformation, 60 % sufficiency, 62 % multi-hop, $0.09 measured.**
- 18 dropped (~11 from Pass-1/2/3 schema fails, ~7 from Pass-3 self-rejection).
- Three failure modes diagnosed: Pass-1 over-rejection ("evidence insufficient" too often), `requires_multihop` over-labelling, Pass-3 over-rejection.
- Quality gates: 2/5 passing.

**3.4.4 A/B comparison strategy.**
- User: "How about a 50-question evaluation with gpt-4o, then decide?"
- Built `04_golden_dataset_gpt4o.ipynb` (mirror of gptoss notebook with `openai_complete_full` swapped in).
- Built `src/generation/openai_client.py` (mirror of `groq_client.py`).
- Same 50 questions (seed 42), same prompts, same retrieval — only constructor changed.

**3.4.5 gpt-4o A/B pilot result.**
- **20/50 accepted (40 %), 0 % malformation, 98 % sufficiency, 66 % multi-hop, $1.11.**
- Headline accept-rate is essentially tied with gpt-oss (40 % vs 42 %), but key differences:
  - **gpt-4o: 0 loop errors vs gpt-oss-120b's 11.**
  - **gpt-4o salvageable rate (accepted + needs_review) = 78 % vs 64 % for gpt-oss.**
  - On the common smoke question, gpt-4o Pass-3 self-rated faithfulness 5/5 vs 3/5 for gpt-oss.
  - gpt-4o correctly labelled the Alzheimer reliability question as `requires_multihop=no`; gpt-oss labelled it `yes` (wrong).
- Recommendation: gpt-4o. Cost difference for 300 rows ($6.68 vs $0.40) is small; quality lift compounds.

**3.4.6 New prompts proposed by user.**
- User shared improved Pass 1, 2, 3 templates with three concrete improvements:
  1. Pass 1 returns structured `selected_chunks` (with support_level + reason), `best_gold_context` (verbatim concatenation by the constructor — eliminates a chunk-id lookup in our code).
  2. Staged-JSONL pipeline pattern (each pass reads previous file, writes next file in `data/processed/golden/`). Better debuggability + selective re-run.
  3. Pass 3 has explicit `answer_match: bool` boolean.
- I added one tightening sentence to Pass 2: *"requires_multihop yes ONLY when answering requires combining ≥2 distinct facts from ≥2 different gold passages AND the answer cannot be inferred from any single passage alone."*

**3.4.7 Main production notebook built.**
- Created `notebooks/04_golden_main_gpt4o.ipynb` with:
  - `STAGE = "pilot" | "production"` flag (pilot N=50, production N=300; pilot is strict subset).
  - Staged JSONL pipeline (Stage B → C → D → E → F → G).
  - Per-pass usage tracking for cost reporting.
- Old A/B notebooks (`04_golden_dataset_gptoss.ipynb`, `04_golden_dataset_gpt4o.ipynb`) **kept as frozen historical record** — not meant to re-run.

**3.4.8 Pilot run with new prompts.**
- 27/50 accepted (54 %), 0 % malformation, 94 % sufficiency, **8 % multi-hop** (down from 66 %!), $1.08.
- 4/5 gates passing. Multi-hop tightening **worked spectacularly** — 58 pp reduction.
- 12 rows failed Pass-2 schema validation: all returned exactly 2 `hallucination_check_points`; my validator threshold was `≥ 3`. The model was right; the validator was wrong.
- **Validator relaxed from `≥ 3` to `≥ 1`** (matches the prompt's literal contract).

**3.4.9 Production run.**
- STAGE flipped to "production"; cache covered the first 50 rows.
- **Result: 234/300 accepted (78 %), 53 needs_review, 13 dropped.**
- Salvageable rate **95.7 %**.
- Cost $6.61 measured (matches estimate).
- Wall time ~80 min total (Stage B retrieval 17.5 min + Pass 1 25.5 min + Pass 2 15.8 min + Pass 3 9.6 min).
- Multi-hop rate **6 %** at scale (gate < 60 %).
- Constructor formally locked back to gpt-4o; all docs synchronised.
- Canonical deliverable: `data/processed/golden_ragas_300.jsonl` (234 rows).

---

## 4. User preferences & working style (from memory + observed patterns)

These shape every interaction:

1. **One step per session** — finish, verify, then move on. Don't batch.
2. **Smoke-test before scaling** — propose 50-row pilots before 300-row builds, 3-question smoke before 12,723-question runs.
3. **Lead with *why* → *how* → *example*** — beginner-level explanations preferred. Use analogies.
4. **Don't waste money** — propose cheaper alternatives proactively for non-trivial cost.
5. **Be honest, push back on weak ideas** — user has reversed three decisions (1000→300 golden, MedEmbed→dropped, gpt-oss-120b→gpt-4o) after honest pushback and appreciates it.
6. **Update docs when decisions change** — sync plan.md §0, tech_stack §3, architecture, todo log, memory in lockstep.
7. **DO NOT auto-execute notebooks** — user runs Jupyter cells manually. Agent edits `.ipynb` files only; never invokes `nbconvert`/`jupyter execute`. Recorded in `feedback_no_auto_execute_notebooks.md`.
8. **Output notes pattern** — every notebook's results get written up in `docs/output_notes/<NN>_notebook_output.md` with three sections (Output / Meaning / Conclusions).
9. **Personal viva-prep file** — `docs/raja_notes.md` accumulates plain-language one-paragraph entries per topic.

---

## 5. Codebase state (current files)

### 5.1 `src/` modules (8 of ~14 built)

```
src/
├── data/
│   ├── chunker.py        # recursive 400/80-token chunker
│   ├── embedder.py       # BGE-large wrapper (load_bge, embed_passages, embed_queries with prefix)
│   └── indices.py        # build/load Chroma + build/load BM25 + bm25_top_k
├── utils/
│   └── cache.py          # sha256-keyed disk cache
├── generation/
│   ├── prompts.py        # evidence-grounded + No-RAG + parse_letter (Phase 4 prompts)
│   ├── golden_prompts.py # 3-pass golden-set construction prompts (Phase 3)
│   ├── groq_client.py    # groq_complete + groq_complete_full
│   └── openai_client.py  # openai_complete_full
└── retrieval/
    └── hybrid.py         # RRF fusion (k=60) of dense + sparse
```

**Remaining to build before Phase 4:**
- `src/retrieval/{base,none,naive,sparse,multi_hop,adaptive,complexity}.py`
- `src/eval/{non_llm_metrics,ragas_eval,runner}.py`
- `src/xai/{lime_passage,shap_passage,agreement}.py` (Phase 6)
- `src/confidence/{signals,rejection}.py` (Phase 7)
- `src/taxonomy/categories.py` (Phase 8)

### 5.2 Notebooks

```
notebooks/
├── 00_data_processing_and_eda.ipynb       ✅ DONE (Phase 1)
├── 01_chunking_and_corpus_prep.ipynb      ✅ DONE — 67,599 chunks
├── 02_embeddings_and_indices.ipynb        ✅ DONE — embeddings + Chroma + BM25
├── 03_smoke_test_pipeline.ipynb           ✅ DONE — pipeline gate
├── 04_golden_dataset_gptoss.ipynb         📦 FROZEN — A/B record (not meant to re-run)
├── 04_golden_dataset_gpt4o.ipynb          📦 FROZEN — A/B record (not meant to re-run)
└── 04_golden_main_gpt4o.ipynb             ✅ DONE — production build, 234 accepted
```

### 5.3 Data artefacts

```
data/processed/
├── medqa_4opt.parquet              ✅ from Notebook 00
├── medqa_5opt.parquet              ✅ from Notebook 00
├── textbook_stats.parquet          ✅ from Notebook 00
├── eda_summary.json                ✅ from Notebook 00
├── chunks.parquet                  ✅ 67,599 rows · 30 MB
├── embeddings.npy                  ✅ (67599, 1024) float32 · 277 MB
├── golden_ragas_300.jsonl          ✅ 234 accepted rows · 1.0 MB (canonical Phase 4 input)
└── golden/
    ├── golden_candidates.jsonl     staged: Hybrid retrieval output (300 × 10 candidates)
    ├── golden_evidence_selected.jsonl  staged: Pass 1 output
    ├── golden_with_references.jsonl    staged: Pass 2 output
    ├── golden_validated.jsonl          staged: Pass 3 output
    ├── golden_main_accepted.jsonl      234 rows
    ├── golden_main_needs_review.jsonl  53 rows
    └── golden_main_dropped.jsonl       13 rows

data/indices/
├── chroma_textbooks/                ✅ collection medqa_textbooks_bge_400, 67,599 vectors, 1.1 GB
└── bm25.pkl                         ✅ 67,599 chunk_ids, 105.8 MB

data/cache/
├── groq/                            ✅ 3 cached responses from Notebook 03
└── openai/                          ✅ ~880 cached responses from Notebook 04
```

### 5.4 Documentation

```
docs/
├── README.md                       index with reverse-lookup table
├── beginners_guide.md              project explained in plain English
├── thesis_understanding.md         proposal-to-implementation mapping
├── tech_stack.md                   locked decisions + rejected alternatives
├── dataset.md                      data shapes + Harrison's bias note
├── architecture.md                 src/ layout, hardware split, M1 Pro vs Colab
├── todo.md                         working state + decision log (kept in sync with plan.md)
├── raja_notes.md                   user's personal viva-prep notes
├── output_notes/
│   ├── 01_notebook_output.md       chunking results + meaning + conclusions
│   ├── 02_notebook_output.md       embeddings + indices results
│   ├── 03_notebook_output.md       pipeline smoke test results
│   ├── 04_notebook_output.md       gpt-oss-120b vs gpt-4o A/B comparison
│   └── 04_golden_main_output.md    Phase 3 production results (234 accepted)
└── discussion_notes/
    └── chat_context_2026-05-05.md  THIS FILE
```

---

## 6. What's next — Phase 4 roadmap

**Phase 4 = Group A baseline experiments (EXP_01 → EXP_05).** Five experiments, each running on the full 12,723 MedQA questions:

| Experiment | Architecture | Notebook | Time | Cost |
|---|---|---|---|---|
| EXP_01 | No-RAG baseline | `04a_exp01_base_llm.ipynb` | ~6 h Groq | $0 |
| EXP_02 | Naive RAG (Chroma top-5) | `04b_exp02_naive_rag.ipynb` | ~6 h Groq | $0 |
| EXP_03 | Sparse RAG (BM25 top-5) | `04c_exp03_sparse_rag.ipynb` | ~5 h Groq | $0 |
| EXP_04 | Hybrid RAG (RRF k=60) | `04d_exp04_hybrid_rag.ipynb` | ~6 h Groq | $0 |
| EXP_05 | Multi-Hop RAG (≤3 hops) | `04e_exp05_multi_hop_rag.ipynb` | ~12–18 h Groq | $0 |

**RAGAS judging across all 5 architectures × 234 golden rows × 5 metrics ≈ 5,850 Claude-Sonnet calls ≈ $10–15.**

**Build order before any EXP runs:**
1. `src/retrieval/base.py` — `Retriever` ABC: `retrieve(q: str, k: int) -> list[Chunk]`
2. `src/retrieval/none.py` — returns `[]` (for EXP_01)
3. `src/retrieval/naive.py` — Chroma top-k with BGE query prefix
4. `src/retrieval/sparse.py` — BM25 top-k
5. `src/retrieval/hybrid.py` — already built ✓
6. `src/retrieval/multi_hop.py` — decompose → 1–3 hops → accumulate
7. `src/eval/non_llm_metrics.py` — Exact Match, Recall@K, MRR, nDCG@K, latency
8. `src/eval/ragas_eval.py` — wrap RAGAS with Claude as judge
9. `src/eval/runner.py` — `run_experiment(retriever, dataset, output_dir)` writes `predictions.jsonl`, `retrieval.jsonl`, `summary.json`

**Tables that get filled by Phase 4:** Table 1 (Overall Architecture Performance), Table 8 (Retrieval Quality), Table 9 (Before/After RAG).

---

## 7. Critical files for a new chat to read first

In this exact order:

1. **[plan.md](../../plan.md)** — locked decisions + 16-experiment programme. Most up-to-date snapshot.
2. **[docs/todo.md](../todo.md)** — current working state + decision log (every reversal recorded with date).
3. **[docs/output_notes/04_golden_main_output.md](../output_notes/04_golden_main_output.md)** — Phase 3 production results (most recent meaningful work).
4. **[docs/output_notes/04_notebook_output.md](../output_notes/04_notebook_output.md)** — gpt-oss-120b vs gpt-4o A/B comparison (explains why constructor is gpt-4o not gpt-oss-120b — preserves the rationale).
5. **[memory/project_thesis_overview.md](file:///Users/rajak/.claude/projects/-Users-rajak-Workstation-Projects-myGitHub-thesis-project/memory/project_thesis_overview.md)** — locked-decisions snapshot in agent memory.
6. **[memory/feedback_step_by_step.md](file:///Users/rajak/.claude/projects/-Users-rajak-Workstation-Projects-myGitHub-thesis-project/memory/feedback_step_by_step.md)** — user's working-style preferences.
7. **[memory/feedback_no_auto_execute_notebooks.md](file:///Users/rajak/.claude/projects/-Users-rajak-Workstation-Projects-myGitHub-thesis-project/memory/feedback_no_auto_execute_notebooks.md)** — DO NOT auto-execute notebooks.
8. **[CLAUDE.md](../../CLAUDE.md)** — Claude-Code-specific rules.
9. **[AGENTS.md](../../AGENTS.md)** — canonical agent rules; load-bearing §2 hard rules.

---

## 8. Open methodology threads worth recording in the writeup

These came out of the work and should appear in the thesis methodology / discussion:

1. **Recursive vs semantic chunking** — chose recursive 400/80 for determinism, cost, and viva-defensibility (medical-RAG 2024-25 standard). Semantic chunking would have doubled compute and made `chunk_id`s sensitive to embedder version.

2. **Harrison's corpus bias** — Harrison's = 24.95 % of word count, 24.66 % of chunks. Real bias; documented in Limitations.

3. **BGE-large MPS sustained-load throttling** — measured ~6 h to embed 67k chunks vs predicted ~22 min. Worth noting in implementation chapter as a hardware caveat.

4. **Dense ↔ sparse retrieval asymmetry validates Hybrid RAG** — pneumonia smoke query showed dense retrieves clinical context, sparse retrieves literal answer terms. Hybrid via RRF combines both. **Without this asymmetry, Hybrid RAG would be redundant.**

5. **Naive RAG fails on label-discrimination questions** — Notebook 03 Q1 (avoidant personality) shows when the answer label isn't named in the question stem, semantic similarity pulls general-topic content instead of label-specific content. Motivates Hybrid RAG.

6. **Multi-hop calibration via prompt-level definition pinning** — added one sentence to Pass 2 prompt; `requires_multihop=yes` rate dropped from 66 % → 6 %. **Worth a paragraph in the methodology section.**

7. **Validator alignment with prompt contract** — `hallucination_check_points ≥ 3` validator was too strict; relaxed to ≥ 1 to match the model's natural output. Lesson: validators should mirror the prompt's literal contract, not impose extra structure.

8. **Empirical A/B for constructor choice** — gpt-4o vs gpt-oss-120b comparison is fully documented and viva-defensible. Examiner can see the data, not just the conclusion.

---

## 9. Things to watch out for (gotchas)

1. **`gpt-oss-120b` strict JSON mode breaks on Groq** (`response_format={"type":"json_object"}` returns `json_validate_failed`). Cause: model emits internal reasoning tokens before the JSON, validator chokes. Use **instructed JSON** with the reasoning-leak parser instead.

2. **Groq SDK URL-encodes slashes in `models.retrieve("openai/gpt-oss-120b")`** → 404. Workaround: use `models.list()` and check membership.

3. **chromadb pinned to 0.5.x** with `transformers<5.0` — chromadb's transitive `tokenizers<0.21` constraint is incompatible with `transformers 5.x` (which needs `tokenizers>=0.22`). Documented in `requirements.txt`.

4. **chromadb posthog telemetry warnings are cosmetic** — `Failed to send telemetry event ClientStartEvent: capture() takes 1 positional argument but 3 were given`. Disabled via `Settings(anonymized_telemetry=False)` + env var, but the warnings still leak from a separate code path. Harmless.

5. **MPS thermal throttling on long embedding runs** — first batch fast (~3 s), sustained throughput ~5× slower. Don't trust pre-flight extrapolation; use measured wall time from first 10 % of corpus.

6. **`golden_ragas_300.jsonl` actually contains 234 rows** — the "300" in the filename refers to the *sample size attempted*, not the *accepted count*. Phase 4 should read this file and treat it as the canonical golden subset.

---

## 10. Open items for next session

- [ ] Manual spot-check of 30 accepted golden rows for grounding quality (todo.md §4)
- [ ] Build `src/retrieval/{base,none,naive,sparse,multi_hop}.py`
- [ ] Build `src/eval/{non_llm_metrics,ragas_eval,runner}.py`
- [ ] Notebook `04a_exp01_base_llm.ipynb` (No-RAG baseline)
- [ ] Run EXP_01 on 12,723 questions (~6 h Groq, free)
- [ ] After EXP_01 completes: write `docs/output_notes/05_exp01_output.md`, mark §5 EXP_01 done in todo.md

**Ready to start Phase 4 immediately when user confirms.**
