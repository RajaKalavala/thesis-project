# Tech Stack — Canonical Reference

> Single source of truth for *every tool, model, and library used in this thesis* — and *why*. When [plan.md §0](../plan.md#0-locked-decisions) and this file disagree, plan.md wins.

---

## 1. The stack at a glance

```
┌─────────────────────────────────────────────────────────────────────┐
│  Hardware:  MacBook Pro · Apple M1 Pro · 16 GB unified memory       │
│  OS:        macOS · Python 3.12.7 (.venv at .venv/)                 │
│  Cloud:     none (no GPU rental, no Colab dependency)               │
└─────────────────────────────────────────────────────────────────────┘
                                  │
   ┌──────────────────────────────┼──────────────────────────────┐
   ▼                              ▼                              ▼
DATA & EDA                  RETRIEVAL                       GENERATION
pandas · numpy              langchain (chunker only)        groq (LLaMA generator)
pyarrow · matplotlib        sentence-transformers           openai (gpt-4o constructor)
seaborn · jupyter           ChromaDB · rank-bm25            anthropic (Claude judge)
                            tiktoken
   ▼                              ▼                              ▼
EVALUATION                  EXPLAINABILITY                  SAFETY & DEMO
RAGAS                       LIME · SHAP                     custom confidence
scikit-learn                                                Streamlit (optional)
```

---

## 2. Layer-by-layer breakdown

### 2.1 Environment

| Tool | Version | Purpose |
|---|---|---|
| **Python** | 3.12.7 | Runtime; chosen over 3.13 because faiss-cpu and several wheels still lag behind |
| **venv** (`.venv/`) | stdlib | Isolated Python environment; activate with `source .venv/bin/activate` |
| **Jupyter** + `ipykernel` | latest | Notebook interface; kernel registered as `thesis-rag` (display: *"Python 3.12 (thesis-rag)"*) |
| **Git + GitHub** | — | Version control |
| **VS Code** (or any editor) | — | Local IDE; user choice |

### 2.2 Data wrangling & EDA

| Tool | Version | Used for |
|---|---|---|
| **pandas** | 2.3.x | All tabular operations |
| **numpy** | 2.2.x | Embedding arrays, statistical ops |
| **pyarrow** | latest | Parquet read/write (efficient than CSV) |
| **scipy** | latest | Statistical-significance tests later |
| **matplotlib**, **seaborn** | latest | EDA plots, thesis figures |
| **openpyxl** | latest | Reading/writing the experiment workbook |

### 2.3 Chunking, embedding, indexing

| Tool | Why this one |
|---|---|
| **`langchain_text_splitters.RecursiveCharacterTextSplitter`** | Well-tested boundary handling for paragraphs/sentences/words. The *only* part of LangChain we use. |
| **`tiktoken`** (cl100k_base encoding) | Token counting. Aligns chunk sizing with both BGE-large input and the LLaMA generator. |
| **`sentence-transformers`** | Wraps the BGE-large embedder. Apple-MPS-friendly. |
| **`BAAI/bge-large-en-v1.5`** (1024-d, 335M, 512 tokens) | Strong general-purpose SOTA embedder. ~75 nDCG@10 on TREC-COVID (medical-IR benchmark). Locked. |
| **ChromaDB** (PersistentClient, single collection `medqa_textbooks_bge_400`) | File-based vector store. Built-in persistence + metadata filtering. Diverges from proposal §7.4.3 (which named FAISS) — methodology section documents the substitution. |
| **`rank-bm25`** (Okapi BM25) | Sparse keyword retrieval. Standard medical-IR baseline. |

### 2.4 LLM clients (three families, on purpose)

| Role | Provider | Model | API SDK | Why a different family |
|---|---|---|---|---|
| **Answerer** (the system under test) | Groq Cloud | `llama-3.3-70b-versatile` | `groq` | Fast LPU inference; 131k context; open-weights LLaMA reproducible |
| **Golden-set constructor** (Phase 3) | OpenAI | `gpt-4o` | `openai` | **Locked 2026-05-04** after empirical A/B against `openai/gpt-oss-120b` on Groq (78 % salvageable vs 64 %, 0 loop errors vs 11 — see `docs/output_notes/04_notebook_output.md`). Three-pass JSON pipeline with new prompts (structured `selected_chunks`, verbatim `best_gold_context`, `answer_match` boolean, multi-hop tightening). Production run 2026-05-04: 234 / 300 accepted at $6.61. |
| **RAGAS judge** (Phase 4) | Anthropic | `claude-sonnet-4-6` | `anthropic` + `langchain-anthropic` (wrapped via `ragas.llms.LangchainLLMWrapper(ChatAnthropic(...))` so `evaluate()` accepts it; the modern `ragas.llms.llm_factory` returns an `InstructorLLM` that the legacy `Metric` tree rejects, see [`docs/todo.md`](todo.md) decision log 2026-05-06) | Different family from BOTH generator (LLaMA) and constructor (`gpt-4o`) — kills evaluator-on-evaluator bias. **Upgraded 2026-05-06** from `claude-3-5-sonnet-20241022` — same per-token pricing ($3/M input · $15/M output), materially better structured-output adherence + sub-statement claim verification. Stays on the paid API: Faithfulness scoring requires sub-statement hallucination detection where Claude is validated against human raters far more thoroughly than open-weights judges. |

The three-family separation is **load-bearing** for the methodology defence: every metric the judge produces is from a model that was not involved in writing the references or generating the answers.

### 2.5 Evaluation

| Tool | Used for |
|---|---|
| **RAGAS** | Faithfulness, Context Precision, Context Recall, Answer Relevancy, Answer Correctness — runs only on the 300-row golden subset |
| **scikit-learn** | Exact-match accuracy, F1, ROUGE wrappers, the small classifier for hallucination-taxonomy labelling (Phase 8) |
| **Custom non-LLM metrics** (`src/eval/non_llm_metrics.py`) | Retrieval Recall@K, MRR, nDCG@K, latency — these run on the full 12,723 |

### 2.6 Explainability

| Tool | Used for |
|---|---|
| **LIME** (`lime` package) | Passage-level local explanations — perturb retrieved passages, observe answer change, fit local linear surrogate (EXP_10) |
| **SHAP** (`shap` package) | Passage-level Shapley value estimation via subset sampling (EXP_11) |
| **Custom agreement scorer** (`src/xai/agreement.py`) | Top-1 / top-3 overlap between LIME and SHAP rankings (EXP_12) |

### 2.7 Caching & infrastructure

| Tool | Used for |
|---|---|
| **Custom disk cache** (`src/utils/cache.py`) | Every LLM response keyed by `sha256(provider + model + temp + prompt)`. Resume after rate limits is free. |
| **`python-dotenv`** | Load API keys from `.env` |
| **`tqdm`** | Progress bars on long loops |
| **`requests`** | Anything that the SDKs don't cover |

### 2.8 Optional UI (Phase 10)

| Tool | Used for |
|---|---|
| **Streamlit** | Demo UI with 4 tabs (Architecture Battle, Explainability, Confidence & Safety, Results Dashboard). Cached-only mode — reads from `results/exp_*/` artifacts; no live LLM calls at demo time. |
| **Streamlit Cloud free tier** | Public deploy for the viva |

---

## 3. Decisions explicitly made — and what was rejected

This section records what we *thought* about and *didn't* pick, so the methodology section can defend the choices.

### 3.1 Embedder — **BGE-large only**

**Locked:** `BAAI/bge-large-en-v1.5`.

**Rejected (with reason):**
- `all-MiniLM-L6-v2` — smaller (384-d, 22M params), faster, but ~28 nDCG@10 lower than BGE on TREC-COVID. Too weak for medical text.
- `MedEmbed-large-v0.1` — medical fine-tune, would have given an ablation table, but adding it doubled the EXP_02/04/05 Groq runtime (~24 h extra) and ~$8 RAGAS cost. Scoped out for compute budget; identified as **future work** in the writeup.
- `MedCPT` (NCBI) — older, 768-d, less mature ecosystem.

### 3.2 Vector DB — **ChromaDB**

**Locked:** ChromaDB persistent collection at `data/indices/chroma_textbooks/`.

**Rejected (with reason):**
- **FAISS** (which the proposal named) — comparable retrieval quality on the pilot run, but lacks built-in persistence and metadata filtering. ChromaDB's ergonomics save engineering time. The methodology section documents this substitution.
- **Pinecone**, **Weaviate**, **Qdrant** — all viable, but require a service or Docker. ChromaDB stays in-process and file-based.

### 3.3 LLM (answerer) — **LLaMA 3.3 70B via Groq**

**Locked:** `llama-3.3-70b-versatile` via Groq Cloud.

**Rejected (with reason):**
- **GPT-4 / GPT-4o** as answerer — closed weights, less reproducible, more expensive at thesis scale.
- **Claude as answerer** — same closed-weights concern; also conflicts with using Claude as the RAGAS judge (would re-introduce same-family bias).
- **Self-hosted LLaMA on RunPod / Lambda** — ~$200 for 50 hours of A100 vs ~$0 on Groq. Not worth it.

### 3.4 RAGAS judge — **Claude Sonnet 4.6**

**Locked 2026-05-06:** `claude-sonnet-4-6` (upgraded from `claude-3-5-sonnet-20241022`).

**Why upgrade:**
- **Same pricing** — Anthropic held Sonnet at `$3/M input · $15/M output` across the 3.5 → 4.x line, so the upgrade is cost-neutral per token.
- **Better structured-output adherence** — RAGAS judges depend on multi-field JSON; Sonnet 4.6 produces fewer parse failures than 3.5 on this workload, meaning fewer NaN scores and fewer wasted calls.
- **Better sub-statement claim verification** — Faithfulness scoring decomposes answers into atomic claims and verifies each against retrieved context. Sonnet 4.6 catches subtler claim-context mismatches that 3.5 misses.
- **Defensibility in 2026** — defaulting to a Q4-2024 model when a 2025+ Sonnet exists invites the viva question *"why didn't you use the current best judge?"*. The 4.6 lock makes that a non-issue.

**Cost recalibrated 2026-05-06**: Phase 4 RAGAS budget is **~$140–160** (5 archs × 234 golden × applicable metrics). The earlier $10–15 estimate was ~10× optimistic — it didn't fully account for the multi-call structure of Faithfulness (2–4 calls/row) and Context Precision (k=5 calls/row).

**Rejected (with reason):**
- `gpt-4o-mini` as judge — same OpenAI family as the constructor → mild evaluator-on-evaluator bias on metrics that consume the reference (Context Recall/Precision, Answer Correctness). For ~$10–15, Claude removes that risk entirely.
- LLaMA 3.3 70B as judge — same family as the answerer. Forbidden by methodology principle.
- Open-weights judge (`gpt-oss-120b`, `qwen3-32b`) — RAGAS Faithfulness scoring requires sub-statement hallucination detection where open-weights models have weaker validation against human raters than Claude/GPT-4-class judges. Saving $10–15 here is not worth introducing methodological doubt on every Faithfulness number in the results chapter.

### 3.5 Golden-set constructor — **`gpt-4o`** (OpenAI API)

**Locked 2026-05-04** after empirical A/B comparison: same 50 questions, same prompts, same retrieval, only constructor swapped.

**A/B result:**

| Metric | `openai/gpt-oss-120b` (Groq, free) | `gpt-4o` (OpenAI, paid) |
|---|---|---|
| Salvageable rows | 64 % | **78 %** |
| Loop / schema errors | 11 | **0** |
| Smoke-question faithfulness | 3/5 | **5/5** |
| Cost (300 rows) | $0.40 | $6.61 |

**Why gpt-4o won.** Cost difference is small in absolute terms relative to the thesis budget; quality lift compounds into every Faithfulness / Context Precision / Context Recall number in chapters 4 and 5. Production run on 300 questions confirmed it scales: 234 accepted, 53 needs_review, 13 dropped at $6.61.

**Prompts (locked 2026-05-04):** new three-pass templates with structured `selected_chunks`, verbatim `best_gold_context` concatenation by Pass 1, `answer_match` boolean in Pass 3, and a tightened `requires_multihop` definition that empirically dropped over-labelling from 66 % to 6 %.

**Rejected (with reason):**
- `openai/gpt-oss-120b` via Groq — A/B-tested and underperformed; preserved in [`notebooks/04_golden_dataset_gptoss.ipynb`](../notebooks/04_golden_dataset_gptoss.ipynb) and [`docs/output_notes/04_notebook_output.md`](output_notes/04_notebook_output.md) as historical record.
- `gpt-4o-mini` — ~30× cheaper than full GPT-4o but less reliable on multi-pass strict JSON. The full GPT-4o is ~$6.61 for 300 rows vs ~$3 for mini; the cost gap is trivial against the quality lift.
- `qwen3-32b` (Alibaba, 32B) — fully independent family but smaller than gpt-oss-120b; expected to drop further under multi-pass JSON stress.

### 3.6 Frameworks/tools considered and **rejected**

| Tool | Why rejected |
|---|---|
| **LangChain** (broadly) | Adds abstraction layers that hide important details. We use only `RecursiveCharacterTextSplitter`. |
| **LangGraph** | Same — over-abstracts the multi-hop control flow we want to write directly. |
| **LangSmith** (observability) | Free-tier 5k-trace limit insufficient (we'll do 50k+ calls). Disk-cache + structured `predictions.jsonl` already give equivalent observability for free. |
| **RunPod / Lambda Labs** (GPU rental) | Almost everything is API-bound (Groq/OpenAI/Anthropic). The one CPU/GPU task (embedding ~67k chunks) measured **~6 h on Apple MPS** (2026-05-04) — but the cost is paid exactly once because the cell is resumable from `embeddings.npy` on disk. Rental setup overhead exceeds the saving. |
| **Colab** (free or Pro) | Same logic — embedding would be ~9 min on a T4 vs ~6 h on MPS, but with a 12 h session limit and the cost paid only once locally, M1 Pro still wins on operational simplicity. Colab is *worse* for the long Groq runs in Phase 4. |
| **vLLM / TGI** (self-hosted serving) | We're not self-hosting any model. Out of scope. |
| **Pinecone**, **Weaviate**, **Qdrant** | Viable vector DBs but require external services. ChromaDB is simpler. |

---

## 4. API key inventory

`.env` at repo root — **never committed** (in `.gitignore`):

```
GROQ_API_KEY=...        # Phase 4 onward — LLaMA 3.3 70B answerer
OPENAI_API_KEY=...      # Phase 3 — gpt-4o golden-set constructor
ANTHROPIC_API_KEY=...   # Phase 4 onward — Claude Sonnet 4.6 RAGAS judge
```

All three needed before experiments. Demo UI (Phase 10) in cached mode needs **none** of them.

---

## 5. requirements.txt — the actual pinned versions

The `.venv` was built from [`requirements.txt`](../requirements.txt). Highlights:

```
pandas>=2.2,<3.0
numpy>=1.26,<2.3
pyarrow>=15.0
sentence-transformers>=3.0
chromadb        (via langchain-community wrapper or direct)
rank-bm25>=0.2.2
langchain>=0.3              # only for RecursiveCharacterTextSplitter
groq>=0.11
openai>=1.40
anthropic                   # add if not present yet
ragas>=0.2
lime>=0.2
shap>=0.46
scikit-learn>=1.4
streamlit                   # add if Phase 10 is taken on
tiktoken>=0.7
python-dotenv>=1.0
tqdm>=4.66
```

When in doubt about an exact version, check `requirements.txt` — it's the authoritative file.

---

## 6. Hardware budget — the M1 Pro story

This is fully covered in [`architecture.md` §8](architecture.md#8-hardware-split--m1-pro-16-gb-vs-google-colab). Short version:

- **M1 Pro 16 GB does 100% of this thesis.** No GPU rental, no Colab dependency.
- The only heavy local task is **embedding** — measured **~6 h on Apple MPS** for 67,599 chunks (recalibrated 2026-05-04 from the original `~12 min MPS / ~25 min CPU` estimate; sustained-load thermal throttling + partial-MPS coverage on BGE-large). Cost is paid exactly once because the cell is resumable from `embeddings.npy`.
- All long-running experiments (12,723-question Groq runs) execute on the M1 Pro overnight as backgrounded scripts. Use `caffeinate -dimsu &` to disable sleep.
- Memory peak ~10 GB during embedding; comfortable within 16 GB.

---

## 7. When to update this document

- When a tool is **added** or **removed** from `requirements.txt` → update §2 and §5.
- When a **decision is locked or reversed** → update §3 with the rejected-alternative reasoning. Update [`docs/todo.md` decision log](todo.md) too.
- When **API keys change** → update §4.

When this drifts from `plan.md §0`, **plan.md wins** — sync this file to it.

---

*Companion docs: [`plan.md`](../plan.md) (locked decisions in §0) · [`architecture.md`](architecture.md) (how the code is laid out) · [`beginners_guide.md`](beginners_guide.md) (plain-English explanation)*
