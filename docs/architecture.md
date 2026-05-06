# Project Architecture & Development Guide

> **Purpose.** How this project is structured, how to *develop* it day-to-day, and which work runs on the **M1 Pro / 16 GB MacBook** vs. **Google Colab**.
>
> Companions: [plan.md](../plan.md) · [todo.md](todo.md) · [thesis_understanding.md](thesis_understanding.md) · [dataset.md](dataset.md) · [tech_stack.md](tech_stack.md)

---

## 1. Architectural philosophy

Five rules. Everything else follows from these.

| # | Rule | Consequence |
|---|---|---|
| 1 | **Build infrastructure once, reuse it 16 times** | One chunked corpus, one set of indices, one LLM client, one evaluator — all 16 experiments are thin orchestration on top |
| 2 | **Notebooks orchestrate; `src/` does the work** | Code that's used twice belongs in `src/`. Notebooks are reproducible recipes, not where the algorithm lives |
| 3 | **Cache every API call to disk** | Resuming a stopped run is free. Re-running an experiment after a bug fix doesn't re-bill Groq/Anthropic |
| 4 | **One source of truth for each fact** | Locked decisions live in [plan.md §0](../plan.md#0-locked-decisions). Tech stack rationale in [tech_stack.md](tech_stack.md). Data shapes in [dataset.md](dataset.md). Working state in [todo.md](todo.md). Code follows. |
| 5 | **Idempotent pipelines** | Re-running notebook 02 with no changes should be a near-no-op (cached embeddings, cached indices). New chunk size? Bump a version flag, regenerate, everything downstream invalidates cleanly |

---

## 2. Layered view

```
┌──────────────────────────────────────────────────────────────────────┐
│  L5 — Output layer                                                   │
│  results/exp_XX/{predictions, retrieval, ragas, summary}             │
│  Excel workbook · thesis chapters · figures                          │
└──────────────────────────────────────────────────────────────────────┘
                                 ▲
┌──────────────────────────────────────────────────────────────────────┐
│  L4 — Orchestration layer (notebooks/)                               │
│  Thin recipes — import src/, set config, run pipeline, save outputs  │
│  00 EDA · 01 chunk · 02 index · 03 smoke · 04 golden                 │
│  04a-e EXP_01-05 · 05 EXP_06-07 · 06 EXP_10-12 · 07 EXP_08-09 · ...  │
└──────────────────────────────────────────────────────────────────────┘
                                 ▲
┌──────────────────────────────────────────────────────────────────────┐
│  L3 — Application layer (src/)                                       │
│  retrieval/ · generation/ · eval/ · xai/ · confidence/ · taxonomy/   │
│  Pure Python modules with stable interfaces, unit-testable           │
└──────────────────────────────────────────────────────────────────────┘
                                 ▲
┌──────────────────────────────────────────────────────────────────────┐
│  L2 — Infrastructure layer (built once, persisted on disk)           │
│  data/processed/{chunks,embeddings,golden_ragas_300}                 │
│  data/indices/{chroma_textbooks,bm25.pkl}                            │
│  data/cache/ — every LLM response keyed by (model, temp, prompt_hash)│
└──────────────────────────────────────────────────────────────────────┘
                                 ▲
┌──────────────────────────────────────────────────────────────────────┐
│  L1 — Raw data (read-only, version-controlled)                       │
│  medqa-data/questions/US/*.jsonl                                     │
│  medqa-data/textbooks/en/*.txt                                       │
└──────────────────────────────────────────────────────────────────────┘
```

**Direction of dependency:** higher layers depend on lower; never the reverse. A change to `chunks.parquet` (L2) can affect every notebook (L4). A change to a notebook never touches L1/L2/L3.

---

## 3. Module breakdown — `src/`

Build these in dependency order. Every module exports a small public API; everything else is private.

```
src/
├── config.py                  # frozen dataclass with all hyperparameters
├── data/
│   ├── loaders.py             # load_medqa(), load_golden(), load_chunks()
│   └── chunker.py             # recursive 400/80 chunker
├── retrieval/
│   ├── base.py                # Retriever ABC: retrieve(q, k) -> list[Chunk]
│   ├── none.py                # for EXP_01 No-RAG
│   ├── naive.py               # ChromaDB top-k (BGE-large)
│   ├── sparse.py              # BM25 top-k
│   ├── hybrid.py              # RRF fusion of naive + sparse
│   ├── multi_hop.py           # 3-hop iterative
│   ├── complexity.py          # rule-based labeller (EXP_06)
│   └── adaptive.py            # router using complexity labels (EXP_07)
├── generation/
│   ├── groq_client.py         # Groq client (LLaMA 3.3 70B answerer)
│   ├── openai_client.py       # OpenAI client (gpt-4o golden-set constructor)
│   ├── anthropic_client.py    # Claude Sonnet 4.6 for RAGAS judge
│   ├── prompts.py             # base evidence-grounded + No-RAG + Multi-Hop prompts (Phase 4)
│   └── golden_prompts.py      # 3-pass golden-set construction prompts (Phase 3)
├── eval/
│   ├── non_llm_metrics.py     # exact match, retrieval recall@k, MRR, nDCG@k
│   ├── ragas_eval.py          # RAGAS suite, Claude as judge
│   └── runner.py              # run_experiment(retriever, dataset, out_dir)
├── xai/
│   ├── lime_passage.py        # passage-level LIME (EXP_10)
│   ├── shap_passage.py        # passage-level SHAP (EXP_11)
│   └── agreement.py           # LIME ↔ SHAP overlap (EXP_12)
├── confidence/
│   ├── signals.py             # 8-dim confidence vector (EXP_08)
│   └── rejection.py           # threshold-based reject (EXP_09)
├── taxonomy/
│   ├── categories.py          # 6 hallucination types (EXP_13)
│   ├── labeller.py            # manual + classifier-assisted (EXP_14)
│   └── analysis.py            # arch × type cross-tab (EXP_15)
└── utils/
    ├── cache.py               # disk-cache decorator for LLM calls
    ├── logging.py             # structured logging to results/
    └── token_count.py         # tiktoken cl100k_base wrapper
```

**Public API contract per module — examples:**

```python
# src/retrieval/base.py
@dataclass
class Chunk:
    chunk_id: str
    book_name: str
    text: str
    score: float

class Retriever(ABC):
    @abstractmethod
    def retrieve(self, question: str, k: int) -> list[Chunk]: ...

# src/eval/runner.py
def run_experiment(
    retriever: Retriever,
    dataset: pd.DataFrame,
    output_dir: Path,
    experiment_id: str,
) -> dict:
    """Returns the summary dict that gets written to summary.json."""
```

**The interface stability matters.** EXP_07 Adaptive RAG calls EXP_02/04/05's retrievers under the hood — they all conform to `Retriever`. EXP_08 Confidence signals consume RAGAS scores from any architecture's runner output — same `summary.json` schema.

---

## 4. Configuration strategy

Two viable patterns. Pick **one** and stick with it.

### Option A — Frozen Python dataclass (recommended for thesis scope)

```python
# src/config.py
from dataclasses import dataclass, field

@dataclass(frozen=True)
class BaseConfig:
    # Locked decisions from plan.md §0
    answerer_model: str = "llama-3.3-70b-versatile"
    answerer_temp: float = 0.0
    answerer_max_tokens: int = 700

    embedder_model: str = "BAAI/bge-large-en-v1.5"
    chunk_size_tokens: int = 400
    chunk_overlap_tokens: int = 80

    chroma_path: Path = Path("data/indices/chroma_textbooks")
    bm25_path: Path = Path("data/indices/bm25.pkl")

    top_k_retrieval: int = 5
    rrf_k: int = 60
    multi_hop_budget: int = 3

    constructor_model: str = "gpt-4o"               # locked 2026-05-04 after A/B vs gpt-oss-120b
    judge_model: str = "claude-sonnet-4-6"

@dataclass(frozen=True)
class ExperimentConfig:
    base: BaseConfig
    experiment_id: str
    output_dir: Path
```

Pros: type-checked, autocomplete, refactor-safe, no string typos.
Cons: changing a value means editing Python.

### Option B — YAML configs

```
configs/base.yaml
configs/exp_02_naive.yaml
configs/exp_04_hybrid.yaml
...
```

Pros: declarative, easy to diff, can drop a new config without touching code.
Cons: no type checking, easy to typo a key.

**Recommendation: Option A** for an MSc-scope project. You won't have many config variants, and the dataclass makes the locked decisions inspectable from the IDE.

---

## 5. Data flow (end-to-end)

```
medqa-data/questions/US/4_options/*.jsonl ──► Notebook 00 ──► medqa_4opt.parquet
medqa-data/textbooks/en/*.txt              ──► Notebook 00 ──► textbook_stats.parquet
                                                                       │
                                                                       ▼
                                                          Notebook 01 ──► chunks.parquet (~67k rows)
                                                                       │
                                          ┌────────────────────────────┼─────────────────────────────┐
                                          ▼                            ▼                             ▼
                                Notebook 02 ──► embeddings.npy       ──► chroma_textbooks/  ──► bm25.pkl
                                                                       │
                                                                       ▼
                                                          Notebook 04 ──► golden_ragas_300.jsonl
                                                                       │
                                          ┌────────────────────────────┼─────────────────────────────┐
                                          ▼                            ▼                             ▼
                            Notebook 04a-e (EXP_01-05)    Notebook 05 (EXP_06-07)    Notebook 06-08 (EXP_10-15)
                                          │                            │                             │
                                          └────────────────────────────┼─────────────────────────────┘
                                                                       ▼
                                                            results/exp_XX/summary.json
                                                                       │
                                                                       ▼
                                                  Notebook 09 EXP_16 ──► final_ranking.json
                                                                       │
                                                                       ▼
                                                       Excel workbook (12 + 1 results tables)
                                                                       │
                                                                       ▼
                                                                 Thesis chapters
```

---

## 6. Caching strategy — the single most important pattern

Every LLM call must be cached. Here's the contract:

```python
# src/utils/cache.py
import hashlib, json
from pathlib import Path
from functools import wraps

CACHE_DIR = Path("data/cache")

def llm_cache(provider: str):
    def decorator(fn):
        @wraps(fn)
        def wrapped(model: str, prompt: str, temp: float, **kw):
            key_data = {"provider": provider, "model": model,
                        "prompt": prompt, "temp": temp, **kw}
            key = hashlib.sha256(
                json.dumps(key_data, sort_keys=True).encode()
            ).hexdigest()
            cache_file = CACHE_DIR / provider / f"{key}.json"
            if cache_file.exists():
                return json.loads(cache_file.read_text())
            result = fn(model, prompt, temp, **kw)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(result))
            return result
        return wrapped
    return decorator
```

**Why this matters:**
- **Resume after rate limits** — Groq's free tier rate-limits you mid-run; restart the notebook, cache hits, continue from where you stopped. Free.
- **Re-run after bug fixes** — Found a bug in your prompt template? Fix the bug, re-run. Cached responses for *unchanged* prompts return instantly; only the new prompt actually calls the API.
- **Compare architectures fairly** — EXP_02's LLaMA call for question Q with a particular prompt is *deterministic* under cache. EXP_04's LLaMA call for the same Q with different retrieved context is a different cache key. No accidental re-rolls.

**Disk footprint:** ~50k LLM calls × ~5 KB JSON each ≈ ~250 MB. Trivial.

**Cache invalidation:** if you change the prompt template, *do* delete `data/cache/groq/` for the affected experiment. Don't try to be clever — manual reset is fine for a thesis.

---

## 7. Development workflow — day-to-day

Three modes. Know which one you're in.

### 7.1 Module development (`src/`)

1. Open VS Code with the `thesis-rag` Jupyter kernel selected
2. Create or edit `src/<package>/<module>.py`
3. Write a tiny test in `tests/test_<module>.py` that exercises it on **3 real rows** from `chunks.parquet`
4. Run `pytest tests/test_<module>.py -xvs`
5. Iterate until green
6. Commit

**Why 3 real rows, not mocks:** the modules are tightly coupled to the data shape. Real-data tests catch schema drift; mocks just confirm your mock is right.

### 7.2 Notebook orchestration

1. New notebook `notebooks/XX_experiment.ipynb`
2. **Cell 1**: imports, repo path resolution, config load
   ```python
   from src.config import BaseConfig
   from src.retrieval.naive import NaiveRetriever
   from src.eval.runner import run_experiment
   cfg = BaseConfig()
   ```
3. **Cell 2**: load data
4. **Cell 3**: build the retriever
5. **Cell 4**: call `run_experiment(...)` — this writes `results/exp_XX/`
6. **Cell 5**: load `summary.json`, render a quick plot/table
7. **Cell 6**: print the row that pastes into the Excel workbook

Every notebook should be **runnable end-to-end with "Restart Kernel & Run All"** without manual intervention.

### 7.3 Long-running experiments

When you're about to launch a 6-hour Groq run:

1. **Disable Mac sleep**: `caffeinate -dimsu &` (or System Settings → Battery → never sleep on power)
2. **Plug in the laptop** — don't run from battery
3. **Stable network** — long Groq runs over coffee-shop Wi-Fi will fail
4. **Background-friendly** — run as a Python script, not a notebook cell, so you can `Ctrl+Z` + `bg`:
   ```bash
   .venv/bin/python -m src.eval.runner --config exp_02_naive_bge > results/exp_02/run.log 2>&1 &
   ```
5. **Tail the log periodically** to confirm it's still alive: `tail -f results/exp_02/run.log`
6. **The cache is your safety net** — if the run dies at question 850 of 1,273, the next launch starts at 851 (test-split evaluation per [`plan.md` §0 #8](../plan.md))

---

## 8. Hardware split — M1 Pro 16 GB vs Google Colab

> **TL;DR for hardware:** the **M1 Pro does everything** but the one heavy CPU/GPU task (embedding) is **slower than initial projections suggested**. Embedding the ~67k chunks once measured **≈ 355 min on Apple MPS** (recalibrated 2026-05-04 after Notebook 02 ran) — first-batch timing predicts ~118 min, but sustained throughput degrades to ~16 s/batch over 6 hours (thermal throttling + partial-MPS coverage on BGE-large). Colab T4 would do this in ~9 min, but the setup overhead + 12 h session limit + cost of restarting on disconnect still favours running locally — and the cost is paid exactly once because the cell is resumable from `embeddings.npy` on disk.

### 8.1 What runs on M1 Pro — and how

| Workload | Local feasibility | Notes |
|---|---|---|
| Phase 1 — EDA | ✅ Trivial | <5 min, ~500 MB peak. Already done. |
| Phase 2 — Chunking (Notebook 01) | ✅ Trivial | ~1–2 min CPU, <1 GB peak. |
| Phase 2 — **Dense embedding** (Notebook 02, BGE-large only) | ⚠️ Slow but feasible | **~355 min measured on Apple MPS** for 67,599 chunks at batch 32 (≈ 6 h, recalibrated 2026-05-04 from the original ~22 min projection). 1.3 GB model footprint. Run is resumable: `embeddings.npy` is read from disk on subsequent runs of cell §6. |
| Phase 2 — ChromaDB build | ✅ Trivial | Disk-based, ~1 min. |
| Phase 2 — BM25 index | ✅ Trivial | Pure Python, ~30 s. |
| Phase 2 — Smoke test (Notebook 03) | ✅ Trivial | 3 questions, ~30 s. |
| Phase 3 — **Golden RAGAS dataset** (Notebook 04) ✅ DONE 2026-05-04 | ✅ API-bound | All compute is OpenAI API calls (`gpt-4o` constructor — locked 2026-05-04 after A/B). M1 Pro just orchestrates. **Measured wall-time: ~80 min for 300 questions; cost $6.61.** Output: 234 accepted in `data/processed/golden_ragas_300.jsonl`. |
| Phase 4 — **Group A (5 experiments × 1,273 test split)** | ✅ API-bound (short) | All compute is Groq API. M1 Pro orchestrates + caches. **~1 h wall-clock total** (recalibrated 2026-05-06 from ~30–40 h on full 12,723; locked evaluation surface narrowed to test split per [`plan.md` §0 #8](../plan.md) for contamination-clean comparison). Each architecture is a coffee-break run. |
| Phase 4 — RAGAS judging | ✅ API-bound | Anthropic API. ~2 h wall-clock for all architectures × 300 golden × 5 metrics. |
| Phase 5 — Adaptive RAG | ✅ API-bound | ~10 h Groq. |
| Phase 6 — LIME / SHAP | ✅ Mostly API-bound | Local: small linear models, perturbation logic. Remote: Groq for re-prompting. ~6–10 h. |
| Phase 7 — Confidence | ✅ Trivial | Pure Python aggregation, no LLM calls. <30 min. |
| Phase 8 — Taxonomy | ✅ Mostly local | Manual labelling + small scikit-learn classifier. <1 h compute. Optional `gpt-4o-mini` classifier ~5 min (~$1). |
| Phase 9 — Final synthesis | ✅ Trivial | Pandas aggregation, plotting. <30 min. |
| Thesis writing | ✅ | Markdown + LaTeX + your favourite editor. |

### 8.2 Where Colab actually helps (and the trade-offs)

| Task | Local M1 Pro | Colab T4 (free) | Worth Colab? |
|---|---|---|---|
| BGE-large embed ~67k chunks (one-time) | **~355 min measured on MPS** (2026-05-04) | ~9 min on T4 | **No** — local run is resumable from `embeddings.npy`; the 6 h cost is paid exactly once |
| Anything else in this thesis | Already API-bound or trivially CPU-bound | Same speed (Colab CPU ≈ M1 Pro CPU); GPU unused for API calls | **No** |

**Colab pitfalls:**
- Free-tier session limit ~12 h, then disconnects. Long Groq runs DO NOT FIT.
- Free-tier GPU not guaranteed at peak times.
- Colab Pro is $10/month if you genuinely need it, but for this thesis **you don't**.
- Colab + Drive + your local repo = three places where files can drift out of sync. Sync discipline matters.

**My recommendation:** keep all work on the M1 Pro. Embedding is the slow part (≈ 6 h on MPS, recalibrated 2026-05-04 from the original `~12 min` estimate) but the cost is paid exactly once and the cell is resumable from `embeddings.npy` on disk — every later notebook reads it in <1 second.

### 8.3 Apple MPS (the M1 Pro GPU) — when to enable it

PyTorch / sentence-transformers support Apple's Metal Performance Shaders. Enable it for the embedding step:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="mps")
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
```

Speedup: ~2× over CPU for BGE-large. Memory: model uses ~1.3 GB unified memory; safe with 16 GB system as long as Chrome + Slack aren't gobbling 8 GB.

**Don't bother with MPS for:**
- Anything other than embedding (your other workloads are I/O / API-bound).
- LIME/SHAP — those are CPU-light orchestration, not heavy tensor work.
- LLaMA inference — that's on Groq, not local.

### 8.4 Memory budget on a 16 GB MacBook

Practical headroom during heavy work:

| Process | Memory | Notes |
|---|---|---|
| macOS + dock + Finder | ~3 GB | |
| VS Code + extensions + Cursor | ~1 GB | |
| Browser (Chrome/Safari, ~5 tabs) | ~1.5 GB | Close tabs you don't need |
| Jupyter kernel + pandas + chunks.parquet | ~1 GB | |
| Embedding model in RAM (BGE-large) | ~1.3 GB | Loaded once during Notebook 02 |
| ChromaDB collection in RAM | ~0.5 GB | |
| Headroom for batch tensors during encoding | ~2 GB | At batch size 32, should be safe |
| **Peak total** | **~10 GB** | **6 GB safety margin** |

Comfortable. One saving grace: don't keep a full 16 GB Chrome session open during embedding runs. Quit it.

If you find yourself swapping (Activity Monitor → Memory → Memory Pressure goes yellow/red), close apps. M1 Pro's swap is fast but it'll still slow embedding by ~3× under pressure.

---

## 9. Common gotchas (M1 Pro / venv specific)

| Gotcha | Symptom | Fix |
|---|---|---|
| `pip install faiss-cpu` on Python 3.13 | Wheel not available | Use Python 3.12 (already done in `.venv/`) |
| MPS device crashes on certain ops | `RuntimeError: MPS does not support ...` | Fall back to `device="cpu"` for that specific call; report to PyTorch |
| `sentence-transformers` re-downloads model on every run | Slow startup, 1.3 GB download each time | First-run cache lives at `~/.cache/huggingface/hub/`. Don't `rm -rf` it. |
| ChromaDB lock errors when two notebooks open same collection | `IOError: database is locked` | Only one notebook can hold a `PersistentClient` at a time; close one before opening the other |
| `caffeinate` not preventing sleep | Mac still sleeps | Use System Settings → Battery → "Prevent automatic sleeping when display is off" (on power adapter) |
| Groq rate limit hit mid-run | `429 Too Many Requests` | The cache decorator + retry logic handles it. Just restart the notebook. |
| API key missing | `KeyError: GROQ_API_KEY` | `cp .env.example .env`, fill in three keys, restart kernel |

---

## 10. Recommended development sequence (next 5 sessions)

| Session | Goal | Output | Hardware |
|---|---|---|---|
| 1 | Build `src/data/` + `src/utils/cache.py` + `src/config.py`. Notebook 01 (chunking). | `chunks.parquet` | M1 Pro |
| 2 | Build `src/data/indices.py`. Notebook 02 (dual embedding + ChromaDB + BM25). | `embeddings_*.npy`, `chroma_*/`, `bm25.pkl` | M1 Pro (or Colab for ~30 min embedding speedup) |
| 3 | Build `src/retrieval/{base,naive,sparse,hybrid}.py` + `src/generation/groq_client.py` + `src/generation/prompts.py`. Notebook 03 (smoke test). | First end-to-end answers on 3 dev questions | M1 Pro |
| 4 | Build `src/eval/{non_llm_metrics,runner}.py`. Run **EXP_01 No-RAG** on test split 1,273 (per [`plan.md` §0 #8](../plan.md)) — simplest experiment to flush bugs. | `results/exp_01_base_llm__test_1273/summary.json` | M1 Pro, ~6 min |
| 5 | Build `src/generation/openai_client.py` + Notebook 04 (golden RAGAS dataset, staged 50 → 300). | `golden_ragas_300.jsonl` | M1 Pro |

By session 5 you have *every reusable piece* of the pipeline. Sessions 6+ are mostly running experiments and writing.

---

## 11. The "build cleanly" checklist

Before you start any new piece of code, ask:

- [ ] Is this **shared infrastructure** (belongs in `src/`) or **experiment-specific** (belongs in a notebook)?
- [ ] Does it **read** from `data/` only, never write back to it (except via notebook 01/02/04)?
- [ ] Does it call the LLM through the **cache decorator**?
- [ ] Does it accept its **config** as a frozen dataclass, not as scattered globals?
- [ ] Does it have a **3-row test** I can run in ~10 seconds?
- [ ] Does its **output schema** match what the next consumer expects (Excel column names, RAGAS field names)?

If yes to all six, you're building something the future-you (and the examiner) can defend. If no to two or more, slow down and refactor.

---

*Last refreshed: 2026-05-03. When this drifts from `plan.md`, `plan.md` wins — sync this doc to it.*
