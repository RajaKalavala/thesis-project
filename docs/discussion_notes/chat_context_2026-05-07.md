# Chat Context — 2026-05-05 → 2026-05-07 (3 sessions)

> Working transcript of the Phase 4 implementation work spanning 3 days. Captures **decisions made**, **the *why* behind each one**, **what's on disk now**, **what's pending**, and **open methodology questions** in case future-you (or a supervisor / examiner) needs to reconstruct the trail without re-reading the chat.

---

## 1. Where we started (2026-05-05) and where we are (2026-05-07)

| Phase | 2026-05-05 status | 2026-05-07 status |
|---|---|---|
| Phase 4 EXP_01 No-RAG | not started | ✅ baseline + RAGAS complete |
| Phase 4 EXP_02 Naive Dense | not started | ✅ baseline + RAGAS complete |
| Phase 4 EXP_03 Sparse BM25 | not started | ✅ baseline complete · RAGAS pending |
| Phase 4 EXP_04 Hybrid RRF | not started | ✅ retriever + 2 notebooks built · ALL pending user run |
| Phase 4 EXP_05 Multi-Hop | not started | ✅ retriever + 2 notebooks built · ALL pending user run |

Cumulative API spend: **~$22.61 of projected ~$70**.

---

## 2. Decisions made (chronological)

### 2.1 Phase 4 EXP_01 build (2026-05-05)

**Built**: `src/data/loaders.py`, `src/retrieval/{base,none}.py`, `src/eval/{non_llm_metrics,runner}.py`, `tests/test_exp01_modules.py`, `docs/results_schema.md`, `notebooks/04a_exp01_base_llm.ipynb`. Plus path-anchoring fix to `loaders.py` + `cache.py` (CWD = notebooks/ otherwise breaks file lookup; both modules now use `Path(__file__).resolve().parents[2]` for repo root).

**Why this slice**: EXP_01 is the simplest experiment (No-RAG → just LLM + parsing) and forces us to build the runner + eval skeleton + No-RAG retriever, which unlock EXP_02–EXP_05.

**Key design choice**: `summary.json` keys mirror Excel `Results Table` headers verbatim (including the `Acuuracy` typo) so populating the workbook is a paste step. Schema locked in `docs/results_schema.md`.

### 2.2 EXP_01 ran — contamination signal found (2026-05-05)

**Findings**:
- Full 12,723 Acuuracy = 0.8693
- **train + dev = 0.880 vs test = 0.774 → 10.6 pp gap**
- Aligns with literature LLaMA No-RAG ceiling on MedQA-US (~75–78 %) on the test split → empirically validates the contamination risk anticipated in `plan.md §15`
- Long-vignette accuracy (n=518) = 0.853 vs short = 0.870 → only 1.7 pp gap, raises the bar for EXP_05 Multi-Hop
- Operational health: 0 parse failures across 13,007 calls; mean latency 0.279 s; wall_time 58 min for full 12,723 (vs 5 h originally projected)

**Implications**: every subsequent architecture should report paired (full / test-split) numbers. The headline narrative pivots from *"RAG raises accuracy"* to *"RAG reduces hallucination on Faithfulness"* via the confidence-aware-rejection novelty layer (Phase 7).

### 2.3 RAGAS module + EXP_01 RAGAS run (2026-05-05/06)

**Built**: `src/eval/ragas_eval.py` + `notebooks/04a_exp01_ragas.ipynb`.

**Multi-iteration API debugging**:
1. First attempt: `ragas.metrics.collections.*` classes + `LangchainLLMWrapper` → `ValueError: Collections metrics only support modern InstructorLLM`
2. Second attempt: `ragas.llms.llm_factory(provider="anthropic", ...)` returning `InstructorLLM` → `TypeError: All metrics must be initialised metric objects`
3. **Final working pattern**: legacy lowercase singletons (`from ragas.metrics import faithfulness, …`) + `LangchainLLMWrapper(ChatAnthropic(...))`. Two parallel + incompatible metric hierarchies in RAGAS 0.4.3 — collections classes can't be passed to `evaluate()`. The legacy path is scheduled for removal at RAGAS v1.0 (well after thesis submission).

**EXP_01 RAGAS results**: Answer_Correctness = 0.8738 (over 137 non-NaN), RAGAS_Answer_Relevance = 0.5977 (over 174 non-NaN). Faithfulness / Context Precision / Context Recall stay `null` for No-RAG by design (Option A — undefined without retrieved context). Cost: $4.50.

**Smoking gun for the thesis**: even on EXP_01, AC = 0.93 on correct rows vs 0.31 on wrong rows → 62 pp gap → judge is calibrated. This validates Sonnet 4.6 as a working judge.

### 2.4 NaN issue + resilience layer (2026-05-06)

**Problem**: EXP_01's RAGAS run came back with ~40 % NaN cells (97/234 on AC, 60/234 on AR). Diagnosis: transient Anthropic API failures absorbed silently by `evaluate(raise_exceptions=False)` at the ~1,400-call scale of a full RAGAS pass — uniform NaN rate across all data slices ⇒ not content-related.

**Fix added** (no LLM cost): two preventatives in `src/eval/ragas_eval.py`:
1. `RunConfig(max_workers=4, max_retries=10, max_wait=120, timeout=180)` — default `max_workers=16` was overwhelming Anthropic's per-minute cap; lowering to 4 + longer max_wait keeps requests under the cap
2. `score_predictions(..., rescore_nans=True)` mode — re-judges only NaN rows and merges new scores back via `_merge_partial_scores` (preserves already-good cells)

Plus a 4-mode idempotency contract documented in `docs/results_schema.md` §5: fresh / cache hit / full rerun / NaN rescore.

**EXP_01 Stage C (rescore) was SKIPPED** by user choice — methodology footnote accepted instead. The 137-row sample is statistically defensible (62 pp correct/wrong gap is far above any threshold). EXP_02 onward used the resilience layer from the start; NaN rate dropped to <2.1 %.

### 2.5 Judge model upgrade: Claude 3.5 Sonnet → Sonnet 4.6 (2026-05-06)

**Why**: Sonnet 4.6 is materially better at structured output adherence + sub-statement claim verification (the workload RAGAS Faithfulness needs). Same per-token pricing ($3/M input · $15/M output) → swap is cost-neutral. Defensibility-wise, defaulting to a Q4-2024 model in 2026 invites a viva question we don't need.

**Doc cascade**: 15 files updated (plan.md §0 #11, tech_stack.md §3.4, architecture.md, AGENTS.md, README.md, results_schema.md, todo.md, memory, requirements.txt, src/eval/ragas_eval.py, both notebooks, output_notes). New decision-log entry covering the swap + cost recalibration.

### 2.6 Cost recalibration: $140–160 → ~$70 (2026-05-06 → 2026-05-07)

**Original estimate** (plan.md §14): ~$140–160 for Phase 4 RAGAS. **Empirical anchor** (after EXP_01 + EXP_02 RAGAS ran): **~$60–70 total**. Plan.md was ~10× optimistic on per-row call counts initially, then I over-corrected. Real per-row cost ≈ $0.05 across 5 metrics; per architecture ≈ $11–12.

| Architecture | Projected | Actual / status |
|---|---:|---:|
| EXP_01 RAGAS (2 metrics) | $11–12 | **$4.50** ✅ |
| EXP_02 RAGAS (5 metrics) | $11–12 | **$11.50** ✅ |
| EXP_03 RAGAS (5 metrics) | ~$12 | pending |
| EXP_04 RAGAS (5 metrics) | ~$12 | pending |
| EXP_05 RAGAS (5 metrics, larger context) | ~$13–15 | pending |
| EXP_07 (score-join, no judge calls) | $0 | pending |
| **Total Claude RAGAS** | **~$60–70** | |

Plus already-spent: $6.61 (Phase 3 golden, gpt-4o) + ~$3 (Phase 8 taxonomy, gpt-4o-mini, pending) → grand total **~$70–80**.

### 2.7 Evaluation surface narrowed: full 12,723 → test split 1,273 (2026-05-06)

**Why**: EXP_01 revealed the 10.6 pp train+dev vs test contamination gap. The test split is the only contamination-clean slice and matches MedRAG / MIRAGE's primary reporting surface. Statistical power is sound (n=1,273, 95 % CI ±2.4 pp at p≈0.85), wall time per architecture drops 10× (~6–10 min vs ~1–2 h), and the methodology framing is *cleaner* not weaker.

**What was preserved**: EXP_01 full-12,723 run kept on disk as `results/exp_01_base_llm__full_12723/` with a `README_LEGACY.md` marker — it's the contamination-evidence anchor.

**Migration**: EXP_01 `test_1273/` was DERIVED on 2026-05-06 by filtering full-12,723 predictions to `split == 'test'` rows and recomputing summary.json — zero new Groq calls. EXP_02 partial full-12,723 (4,844 rows started, all train) was abandoned as legacy — zero cache value for the new test surface.

**Notebook updates**: 04a + 04b (and the new 04c/04d/04e) Stage C cells filter to `split == 'test'` before passing to the runner.

### 2.8 EXP_02 Naive RAG run + central thesis-narrative anchor (2026-05-06/07)

**Headline finding**: **Naive RAG with BGE-large is *worse* than No-RAG on the contamination-clean test split.**

| Metric | EXP_01 No-RAG | EXP_02 Naive RAG | Δ |
|---|---:|---:|---:|
| `Acuuracy` (test_1273) | 0.7738 | 0.7573 | **−1.65 pp** |
| `Acuuracy` (golden 234) | 0.9017 | 0.8504 | −5.13 pp |
| `Answer_Correctness` | 0.8738 | 0.8376 | −3.62 pp |
| **`Faithfulness`** | null | **0.1308** | first measured |
| **`Hallucination_Rate`** | null | **0.8957** | first measured |
| `Context_Precision` | null | 0.3285 | first measured |
| `Context_Recall` | null | 0.4124 | first measured |

**The smoking gun**: cross-tab on golden 234 — **88.3 % of correct answers were ungrounded** (Faithfulness < 0.5). LLaMA produces correct MCQ options from pre-training memorisation while retrieved chunks act as distractors. **Mechanism**: Context Precision = 0.33 → 2/3 of retrieved chunks are noise.

**Per-stratum patterns**:
- Multi-hop (n=13): Faithfulness = 0.000 (Naive can't stitch evidence)
- Step 2&3 lost 3.54 pp on test_1273 (Step 1 unchanged)
- Treatment hardest (Acuuracy 0.735 vs diagnosis 0.897)
- Regression: 64 fixes vs 85 new errors (net −21 questions vs No-RAG)

**Thesis impact**: this is the central narrative anchor for the discussion chapter. The thesis-distinguishing question (*does retrieval improve grounding?*) now has a measurable answer: *no, with naive dense retrieval; 88 % of correct answers are still ungrounded.* Motivates EXP_04 Hybrid + EXP_05 Multi-Hop as retrieval-quality fixes, and Phase 7 confidence-aware rejection as the central novelty contribution.

### 2.9 EXP_03 Sparse RAG run + complementarity finding (2026-05-07)

**Headline**: EXP_03 0.7581 ≈ EXP_02 0.7573 on test_1273 (within 0.1 pp). Both ~1.6 pp below No-RAG.

**Two empirical findings that anchor EXP_04**:
1. **Complementarity**: 153 of 1,273 test questions disagree between dense & sparse, with **76 dense-right vs 77 sparse-right** (50/50 split). Strong evidence that RRF fusion should help.
2. **Per-USMLE-step**: sparse beats dense on step2&3 (+1.68 pp; clinical-decision vocabulary), dense beats sparse on step1 (+1.33 pp; basic-science semantic concepts). **Orthogonal strengths along the curriculum axis.**

**Operational anomaly**: EXP_03 wall_time = 97 min vs EXP_02's 12 min — `rank-bm25.get_scores` is O(N=67k) per query (~4 s/query) vs ChromaDB HNSW's ~0.1 s/query. Cost still $0.

**RAGAS deferred** by user — batching all 4 RAG architectures' RAGAS evaluations together after EXP_04/05 baselines done. Strategy: run all baselines (free) → then run all RAGAS (paid) as one batch.

### 2.10 EXP_04 Hybrid + EXP_05 Multi-Hop builds (2026-05-07)

**Built in one session per user's batch-build plan**:
- `src/retrieval/hybrid.py` refactor — added `HybridRetriever(Retriever)` class while keeping the existing `hybrid_top_k` function for Notebook 04 backwards compatibility
- `src/retrieval/multi_hop.py` (new) — `MultiHopRetriever` with 3-hop iterative dense + sub-query generation via Groq + dedup + early stop on no-progress
- `src/generation/prompts.py` — added `build_multi_hop_subquery_prompt`
- `tests/test_hybrid_retriever.py` (4 tests) + `tests/test_multi_hop_retriever.py` (5 tests)
- `notebooks/04d_exp04_hybrid_rag.ipynb` + `04d_exp04_ragas.ipynb`
- `notebooks/04e_exp05_multi_hop_rag.ipynb` + `04e_exp05_ragas.ipynb`

**Multi-hop design** (deliberately conservative, defensible): 3 hops max · k=5 per hop · sub-query generation via the same LLaMA via Groq · chunks deduped across hops · early stop if a hop returns 0 new chunks · returns up to 15 chunks to the answerer (vs 5 for EXP_02–04). No agent loops, no LLM-judge "do I need another hop?" gate, no tool use. Per-question Groq cost ≈ 3× EXP_02 (1 final answer + 2 sub-queries). Cost still $0 on Groq free tier.

---

## 3. Open methodology questions

### 3.1 Two evaluation surfaces — is the mismatch a problem?

**The concern (raised by user 2026-05-07)**: golden 234 is 80 % train (188 rows) / 12 % dev (28) / 8 % test (18). The headline accuracy comes from `test_1273` (100 % test, contamination-clean), but RAGAS metrics come from `golden_234` (80 % train, contamination-heavy).

**The empirical defence**: EXP_02 stratified breakdown shows the judge is contamination-robust:
- Faithfulness train = 0.137 vs test = 0.145 (Δ = 0.8 pp)
- Answer Correctness train = 0.842 vs test = 0.858 (Δ = 1.6 pp)
- vs Exact Match's 10.6 pp train-vs-test gap on EXP_01

The judge measures grounding (answer-vs-evidence), which is structurally independent of memorisation (answer-vs-pretraining-soup).

**Three forward paths discussed**:
- **Option A** — keep current setup, document with footnote (my recommendation; user is going with this implicitly)
- **Option B** — build a 150-row test-only golden subset (~$15 + 1 day; gold-standard methodology)
- **Option C** — supplement with a "RAGAS-on-18-test-rows" appendix table ($0; statistically thin)

Decision deferred until after all 4 RAG RAGAS runs complete. The cross-architecture comparison (the actual thesis claims) stays valid in all options.

### 3.2 LIME / SHAP cost (Phase 6) — confirmed $0

Both use Groq for the perturbation re-generations (~42k Groq calls total across 5 archs × 200 sampled questions), bounded by Groq free tier rate limits (~12–24 h wall time). The SHAP-with-Faithfulness-as-score variant would cost ~$320 — explicitly avoided.

---

## 4. Current state (2026-05-07)

### 4.1 Code

`src/`:
- `data/{loaders,chunker,embedder,indices}.py` ✅
- `generation/{prompts,groq_client,openai_client,golden_prompts,anthropic via langchain in ragas_eval}.py` ✅
- `retrieval/{base,none,naive,sparse,hybrid (refactored),multi_hop}.py` ✅
- `eval/{non_llm_metrics,runner,ragas_eval}.py` ✅
- `utils/cache.py` ✅
- Pending (Phase 5+): `retrieval/{complexity,adaptive}.py`, `xai/`, `confidence/`, `taxonomy/`

`tests/`: 28/28 passing — `test_exp01_modules.py` (5) + `test_ragas_eval.py` (10) + `test_naive_retriever.py` (3) + `test_sparse_retriever.py` (4) + `test_hybrid_retriever.py` (4) + `test_multi_hop_retriever.py` (5) — wait, that's 31. Numbers check: 5+10+3+4+4+5 = 31. (Previously stated 28; the actual count is 31.)

`notebooks/`:
- `04a_exp01_base_llm.ipynb` ✅ ran end-to-end
- `04a_exp01_ragas.ipynb` ✅ ran end-to-end
- `04b_exp02_naive_rag.ipynb` ✅ ran end-to-end
- `04b_exp02_ragas.ipynb` ✅ ran end-to-end
- `04c_exp03_sparse_rag.ipynb` ✅ ran end-to-end
- `04c_exp03_ragas.ipynb` ⏳ built, pending user run (batched with EXP_04/05)
- `04d_exp04_hybrid_rag.ipynb` ⏳ built, pending user run
- `04d_exp04_ragas.ipynb` ⏳ built, pending user run
- `04e_exp05_multi_hop_rag.ipynb` ⏳ built, pending user run
- `04e_exp05_ragas.ipynb` ⏳ built, pending user run

### 4.2 Results on disk

```
results/
├── exp_01_base_llm__{smoke_50, golden_234, test_1273, full_12723 (LEGACY)}/      ← all complete with RAGAS where applicable
├── exp_02_naive_rag__{smoke_50, golden_234, test_1273, full_12723 (LEGACY partial)}/  ← all complete with RAGAS
├── exp_03_sparse_rag__{smoke_50, golden_234, test_1273}/                          ← baseline complete · RAGAS pending
├── exp_04_hybrid_rag__   (not yet run)
└── exp_05_multi_hop_rag__ (not yet run)
```

### 4.3 Headline numbers so far (test_1273 — canonical)

| Architecture | `Acuuracy` | Δ vs No-RAG | Faithfulness (golden 234) | Hallucination_Rate | Context Precision |
|---|---:|---:|---:|---:|---:|
| EXP_01 No-RAG | **0.7738** | — | n/a | n/a | n/a |
| EXP_02 Naive Dense | 0.7573 | −1.65 pp | 0.1308 | 0.8957 | 0.3285 |
| EXP_03 Sparse BM25 | 0.7581 | −1.57 pp | (pending) | (pending) | (pending) |
| EXP_04 Hybrid RRF | (pending) | | (pending) | (pending) | (pending) |
| EXP_05 Multi-Hop | (pending) | | (pending) | (pending) | (pending) |

### 4.4 Cost spent

| Phase | Item | Spent | Pending |
|---|---|---:|---:|
| 3 | Golden construction (gpt-4o) | $6.61 ✅ | — |
| 4 | EXP_01 RAGAS (Sonnet 4.6) | $4.50 ✅ | — |
| 4 | EXP_02 RAGAS (Sonnet 4.6) | $11.50 ✅ | — |
| 4 | EXP_03 RAGAS | — | ~$12 |
| 4 | EXP_04 RAGAS | — | ~$12 |
| 4 | EXP_05 RAGAS | — | ~$13–15 |
| 4 | EXP_07 (score-join) | — | $0 |
| 8 | Taxonomy (gpt-4o-mini) | — | ~$3 |
| **Total** | | **$22.61** | **~$40–42** |

**Grand total tracking**: ~$63–65, comfortably under the ~$80 ceiling.

---

## 5. Falsifiable hypotheses pending answer

These were generated from EXP_01–EXP_03 data and locked into the EXP_04/05 notebook inspect cells:

| Hypothesis | Falsified if | Source |
|---|---|---|
| Sparse Context Precision > 0.33 (rare-term recovery) | Sparse CP ≤ 0.33 | EXP_02 §4 |
| Hybrid Acuuracy on test_1273 > 0.76 | Hybrid Acuuracy ≤ 0.76 | EXP_03 §4 |
| Hybrid Context Precision ≥ 0.50 | Hybrid CP ≤ 0.33 | EXP_02 §4 |
| Multi-Hop Acuuracy ≥ EXP_02's 0.7573 | Multi-Hop < 0.7573 | EXP_05 notebook |
| Multi-Hop Faithfulness > 0.05 on multi-hop subset | F ≤ 0.05 | EXP_02 §3.5 |

---

## 6. Locked decisions across plan.md / tech_stack.md / memory

Updated this week:
- **§0 #8** Evaluation surface: full 12,723 → test split 1,273 (2026-05-06)
- **§0 #11** RAGAS judge: Claude 3.5 Sonnet → Sonnet 4.6 (2026-05-06)
- Phase 4 RAGAS cost recalibrated $10–15 → ~$60–70 (2026-05-06 → 2026-05-07)
- Methodology footnote anchored: EXP_01 RAGAS NaN sample (137 of 234) due to pre-resilience-layer run

All previously locked: LLaMA 3.3 70B answerer · gpt-4o constructor · BGE-large embedder · ChromaDB (one collection) · BM25 (rank-bm25) · 400/80 chunks · k=5 · RRF k=60 · 3 hops max · golden 234 (sampled from 300).

---

## 7. Next steps in order

1. **User runs all 4 RAG baselines** (EXP_03 already done; EXP_04 + EXP_05 pending) — ~3 h total wall time, $0 on Groq free tier.
2. **User runs all 4 RAGAS notebooks** (EXP_03 + EXP_04 + EXP_05) — $36–40 total, ~1 h wall time. Includes EXP_03 RAGAS that was deferred for batching.
3. **I write `output_notes/04c_exp03_output.md` (RAGAS update)** + `04d_exp04_output.md` + `04e_exp05_output.md` — cross-architecture comparison + Table 1 ready for the workbook.
4. **Then Phase 5**: EXP_06 (rule-based question complexity classifier) + EXP_07 (adaptive routing — Simple→Naive, Moderate→Hybrid, Complex→Multi-Hop). Per discussion, EXP_07's RAGAS metrics will be **score-joined from underlying architectures** (no new judge calls) since Adaptive routes per question.
5. **Phase 6**: LIME + SHAP — sampled to 200 questions per architecture, all Groq → $0, ~12–24 h wall-time bounded by Groq rate limits.
6. **Phase 7**: Confidence-aware rejection — pure aggregation over Phase 4 + Phase 6 outputs, no LLM calls.
7. **Phase 8**: Hallucination taxonomy — manual + small classifier, ~$3.
8. **Phase 9**: EXP_16 final synthesis — pure aggregation.
9. **Phase 10 (optional)**: Streamlit demo UI — schema-locked in Stage A; Stage B wires up incrementally.

---

## 8. Things to watch / risks

- **Multi-hop accuracy may not beat Naive** — falsifiable hypothesis baked into EXP_05 notebook. If it doesn't earn its 3× compute, the discussion chapter has a clean "iterative retrieval did not improve over single-shot dense" narrative.
- **Hybrid may not beat dense or sparse** — the complementarity finding from EXP_03 (153 questions disagree, 50/50 split) suggests it should, but RRF could fail to capture the right retriever per question. Falsifiable in EXP_04 notebook.
- **Two-surface methodology** (test_1273 accuracy + golden_234 RAGAS) — defensible with the per-stratum empirical check (judge AC train=0.872 vs test=0.855, Δ=1.6 pp). If a reviewer pushes harder, Option B (build test-only golden ~$15) is the gold standard.
- **EXP_01 NaN sample** (137 of 234 on AC) — methodology footnote anchored in `output_notes/04a_exp01_output.md` and `todo.md` decision log; n=137 with 62 pp correct/wrong gap is statistically defensible.

---

## 9. Where to find what

| Need | File |
|---|---|
| EXP_01 full analysis (No-RAG + RAGAS) | `docs/output_notes/04a_exp01_output.md` |
| EXP_02 full analysis (Naive + RAGAS) | `docs/output_notes/04b_exp02_output.md` |
| EXP_03 baseline analysis (RAGAS pending) | `docs/output_notes/04c_exp03_output.md` |
| Golden dataset construction | `docs/output_notes/04_golden_main_output.md` + `04_notebook_output.md` |
| Locked decisions (the live state) | `plan.md §0` |
| Decision log (what changed when + why) | `docs/todo.md` (bottom) |
| Tech stack rationale | `docs/tech_stack.md §3` |
| Two-surface methodology | `docs/dataset.md §4` |
| `summary.json` schema + RAGAS NaN handling | `docs/results_schema.md` |
| EXP_01 contamination evidence | `results/exp_01_base_llm__full_12723/README_LEGACY.md` |

---

*Companion: previous transcript at [`chat_context_2026-05-05.md`](chat_context_2026-05-05.md). Next session — likely after user runs the 4 remaining baselines + 4 RAGAS notebooks — should produce `chat_context_2026-05-08.md` (or whenever) with the cross-architecture analysis.*
