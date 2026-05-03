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
pandas · numpy              langchain (chunker only)        groq (LLaMA)
pyarrow · matplotlib        sentence-transformers           openai (GPT-4o)
seaborn · jupyter           ChromaDB · rank-bm25            anthropic (Claude)
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
| **Golden-set constructor** (Phase 3) | OpenAI | `gpt-4o` (full, not mini) | `openai` | Strict-JSON 3-pass construction; mini drops more under structured-output stress |
| **RAGAS judge** (Phase 4) | Anthropic | `claude-3-5-sonnet-20241022` | `anthropic` | Different family from BOTH generator (LLaMA) and constructor (GPT-4o) — kills evaluator-on-evaluator bias |

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

### 3.4 RAGAS judge — **Claude 3.5 Sonnet**

**Locked:** `claude-3-5-sonnet-20241022`.

**Rejected (with reason):**
- `gpt-4o-mini` as judge — same family as GPT-4o constructor → mild evaluator-on-evaluator bias on metrics that consume the reference (Context Recall/Precision, Answer Correctness). For ~$50 extra, Claude removes that risk entirely.
- LLaMA 3.3 70B as judge — same family as the answerer. Forbidden by methodology principle.

### 3.5 Golden-set constructor — **GPT-4o (full)**

**Locked:** `gpt-4o`.

**Rejected (with reason):**
- `gpt-4o-mini` — ~30× cheaper but less reliable on strict-JSON multi-pass tasks. For 300 rows, full GPT-4o is ~$12 vs ~$1 for mini; the cost difference is trivial against the quality lift on Pass 2 (reference-explanation generation).

### 3.6 Frameworks/tools considered and **rejected**

| Tool | Why rejected |
|---|---|
| **LangChain** (broadly) | Adds abstraction layers that hide important details. We use only `RecursiveCharacterTextSplitter`. |
| **LangGraph** | Same — over-abstracts the multi-hop control flow we want to write directly. |
| **LangSmith** (observability) | Free-tier 5k-trace limit insufficient (we'll do 50k+ calls). Disk-cache + structured `predictions.jsonl` already give equivalent observability for free. |
| **RunPod / Lambda Labs** (GPU rental) | Almost everything is API-bound (Groq/OpenAI/Anthropic). The one CPU/GPU task (embedding ~67k chunks) takes ~45–50 min on M1 Pro CPU or ~22 min on Apple MPS. Not worth the rental setup overhead. |
| **Colab** (free or Pro) | Same logic — only useful for the one-time embedding sweep, which the M1 Pro handles in 22–50 min. Colab's 12-h session limit makes long Groq runs *worse* on Colab. |
| **vLLM / TGI** (self-hosted serving) | We're not self-hosting any model. Out of scope. |
| **Pinecone**, **Weaviate**, **Qdrant** | Viable vector DBs but require external services. ChromaDB is simpler. |

---

## 4. API key inventory

`.env` at repo root — **never committed** (in `.gitignore`):

```
GROQ_API_KEY=...        # Phase 4 onward — answerer
OPENAI_API_KEY=...      # Phase 3 — golden-set constructor (GPT-4o)
ANTHROPIC_API_KEY=...   # Phase 4 onward — RAGAS judge (Claude Sonnet)
```

All three needed before running experiments. Demo UI (Phase 10) in cached mode needs **none** of them.

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
- The only heavy local task is **embedding** (~25 min on CPU, ~12 min on Apple MPS).
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
