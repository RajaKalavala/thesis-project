# AGENTS.md — Instructions for AI Coding Agents

> Canonical instructions for any AI agent (Cursor, Aider, Claude Code, GitHub Copilot Workspace, etc.) working on this repo. Read this **first**, then read [`docs/README.md`](docs/README.md) to load the rest of the context.

---

## 1. What this project is

This is an **MSc thesis** (LJMU, submission March 2026): a controlled comparison of four RAG architectures on the MedQA US benchmark (12,723 USMLE questions), with three novelty layers (adaptive routing, confidence-aware rejection, hallucination taxonomy). It is **research code**, not production. Quality bar is "thesis-defensible" — every line should be explainable in a viva.

**Read [`docs/beginners_guide.md`](docs/beginners_guide.md) once for the project overview. Then keep [`plan.md`](plan.md) and [`docs/todo.md`](docs/todo.md) open as your working references.**

---

## 2. Hard rules (non-negotiable)

These rules are load-bearing. Violating them costs the user real time or money.

### 2.1 Never run an expensive task without a smoke test

**Rule.** Before any task that costs >$1 in API calls or >30 minutes wall time, propose a smoke-test version on small data first (3–50 rows). State the cost differential explicitly.

**Why.** The user is on a thesis budget; running a 6-hour Groq job that fails at hour 5 due to a prompt bug they could have caught in 30 seconds is the worst-case outcome.

**Concrete examples:**

| Big task | Smoke test first | Saves if buggy |
|---|---|---|
| Build 300-row golden dataset (~$12, ~2 h) | Build 50-row pilot first (~$2, 30 min) | $10 + 1.5 h |
| Run EXP_02 on 1,273 test questions (~10 min) | Run EXP_02 on 5 questions first (~1 min) | 10 min |
| Embed 36k chunks (~25 min on M1 Pro) | Embed 100 chunks first (~30 sec) | 25 min |
| LIME/SHAP on 200 questions (~3 h) | Try on 5 questions first (~10 min) | 3 h |

### 2.2 Never propose changes to locked decisions without explicit user input

The locked decisions are in [`plan.md` §0](plan.md). Specifically:
- Single embedder = `BAAI/bge-large-en-v1.5` (a dual-embedder ablation with MedEmbed was explicitly **rejected** by the user; do not re-propose without being asked).
- ChromaDB (one collection, not two).
- 400-token chunks with 80-token overlap.
- 300-row golden subset (was 1,000; user reduced for budget).
- LLaMA 3.3 70B answerer · GPT-4o constructor · Claude Sonnet 4.6 judge.
- 1,273-question MedQA US `test` split as the locked evaluation surface (narrowed from "all 12,723" 2026-05-06 — see plan.md §0 #8 for the contamination-clean methodology rationale; the EXP_01 full-12,723 run is preserved as the contamination-evidence anchor).

If you think a locked decision should change, **state your case clearly and ask the user** before editing files. Do not silently revise.

### 2.3 Cache every LLM call to disk

**Rule.** Every `groq.chat.completions.create()`, `openai.chat.completions.create()`, and `anthropic.messages.create()` goes through the disk-cache decorator (`src/utils/cache.py`). Cache key = `sha256(provider + model + temp + prompt)`.

**Why.** Resuming after a Groq rate limit must be free. Re-running an experiment after a bug fix must hit cache for unchanged prompts.

### 2.4 Notebooks orchestrate; `src/` does the work

**Rule.** Code that's used twice belongs in `src/`. Notebooks are reproducible recipes ("Restart Kernel & Run All" must work end-to-end), not where the algorithm lives.

### 2.5 One step per session with verification

The user explicitly wants **one implementation step at a time** with an acceptance check before moving on. Don't batch multiple major steps into a single PR/commit unless the user asks.

### 2.6 Lead with *why*, then *how*, then a *concrete example*

The user is a learner first, an implementer second. Beginner-level explanations are appreciated, never patronising. Avoid jargon-first; use analogies when introducing concepts.

---

## 3. Where to find context

When a question lands, here's where to look first:

| Question type | First file to read |
|---|---|
| *"What does the thesis claim?"* | [`docs/beginners_guide.md` §1](docs/beginners_guide.md) |
| *"What's locked vs negotiable?"* | [`plan.md` §0](plan.md) |
| *"What should I work on next?"* | [`docs/todo.md`](docs/todo.md) — find the next `[ ]` |
| *"Why was X chosen / Y rejected?"* | [`docs/tech_stack.md` §3](docs/tech_stack.md) |
| *"What does the data look like?"* | [`docs/dataset.md`](docs/dataset.md) |
| *"How is the code organised?"* | [`docs/architecture.md`](docs/architecture.md) |
| *"What's the cost / time budget?"* | [`plan.md` §14](plan.md) |
| *"What can fail?"* | [`plan.md` §15](plan.md) (risk register) |
| *"What did we decide and when?"* | [`docs/todo.md` decision log](docs/todo.md) |

Full doc index: [`docs/README.md`](docs/README.md).

---

## 4. Key environment facts

- **Python 3.12.7** in a local venv at `.venv/` (not Conda, not pyenv-managed).
- **Always invoke Python via `.venv/bin/python`** unless inside an activated venv. Never use system `python3` to install dependencies — that bypasses `requirements.txt`.
- **Jupyter kernel:** `thesis-rag` (display name *"Python 3.12 (thesis-rag)"*). When opening a notebook, ensure this kernel is selected.
- **Hardware:** MacBook M1 Pro, 16 GB. No GPU rental, no Colab dependency — see [`docs/architecture.md` §8](docs/architecture.md).
- **`.env`** at repo root holds `GROQ_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`. **Never commit it.** Always check it's in `.gitignore`.

### Common commands

```bash
# Activate the venv (zsh)
source .venv/bin/activate

# Run a notebook headless
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/00_data_processing_and_eda.ipynb

# Quick Python script (no activate)
.venv/bin/python -c "from src.data.loaders import load_medqa_full; print(load_medqa_full().shape)"

# Add a dependency
.venv/bin/pip install <package> && .venv/bin/pip freeze | grep <package> >> requirements.txt
```

---

## 5. Repository structure

```
thesis-project/
├── README.md                    ← project landing page
├── AGENTS.md                    ← THIS FILE
├── CLAUDE.md                    ← Claude-Code-specific notes; points here
├── plan.md                      ← execution master (kept at root for visibility)
├── requirements.txt
├── .gitignore  .venv/  .env
├── notebooks/
│   └── 00_data_processing_and_eda.ipynb  ← Phase 1 ✅
├── medqa-data/                  ← raw input (read-only)
├── data/                        ← derived artefacts (gitignored)
├── src/                         ← reusable Python modules (to build)
└── docs/
    ├── README.md                ← docs index
    ├── beginners_guide.md
    ├── thesis_understanding.md
    ├── tech_stack.md
    ├── dataset.md
    ├── architecture.md
    ├── todo.md
    └── thesis-files/            ← original proposal PDF + Excel workbook
```

---

## 6. Working style — what the user wants

The user has explicitly asked for these behaviours (recorded in memory at `~/.claude/projects/-Users-rajak-Workstation-Projects-myGitHub-thesis-project/memory/`):

1. **Implement one step at a time** with an acceptance check.
2. **Smoke-test before scaling** — see §2.1.
3. **Lead with *why* then *how* then *example*** — see §2.6.
4. **Don't waste money** — propose cheaper alternatives proactively when the cost is non-trivial.
5. **Be honest, not agreeable** — if the user proposes something that has a real downside, push back with the trade-off laid out clearly. The user has reversed three decisions on this project after pushback (golden 1000→300, MedEmbed→dropped, etc.) and appreciates the honesty.
6. **Update docs when decisions change** — `plan.md §0` and `docs/tech_stack.md §3` are the two places that must always reflect the live locked state. `docs/todo.md` decision log records *why* and *when*.

---

## 7. Source-of-truth hierarchy

When two docs disagree:

1. `plan.md §0` (locked decisions) wins for any technical-stack or scope decision.
2. `docs/tech_stack.md` mirrors §0; sync to plan.md when they drift.
3. `docs/todo.md` decision log is historical, not live state.
4. `README.md` is courtesy navigation, not authoritative.

If you change a locked decision, propagate in this order:
1. Update `plan.md §0` lock
2. Update `docs/tech_stack.md §3` (decisions + rejected alternatives)
3. Update `docs/architecture.md` if the code shape changes
4. Add an entry to `docs/todo.md` decision log
5. Update memory at `~/.claude/projects/.../memory/project_thesis_overview.md`

---

## 8. What "done" looks like for the project

The thesis defends when:

- All 12 results tables in [`docs/thesis-files/Raja Kalavala Final Thesis Project Sheet.xlsx`](docs/thesis-files/) are filled.
- Every claim in the thesis traces back to a specific row in a results table or a specific notebook output.
- The methodology section explains every locked decision in [`plan.md §0`](plan.md).
- (Optional) The Streamlit demo at `app/main.py` runs in cached mode for the viva.

Live progress: [`docs/todo.md`](docs/todo.md).

---

## 9. When in doubt

Ask the user. Especially before:
- Deleting files (other than scratch/cache files)
- Running a long Groq/OpenAI/Anthropic call without a smoke test
- Changing a `plan.md §0` locked decision
- Adding a new top-level dependency
- Pushing to a remote / opening a PR

This is a thesis. Reversibility matters more than speed.
