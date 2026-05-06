# Systematic Comparison of RAG Architectures for Medical QA

> **MSc thesis** — *Systematic Comparison of Multiple Retrieval-Augmented Generative AI Architectures for Evidence-Based Medical Question Answering with Explainability and Hallucination Control*
>
> Raja Kalavala (PN1196988) · MSc AI & ML, LJMU · Submission **March 2026**

---

## What this is

A controlled side-by-side comparison of **4 RAG architectures** (Naive · Sparse · Hybrid · Multi-Hop) on the **MedQA US** benchmark (12,723 USMLE-style clinical questions) with the same LLM, embedding model, and prompt template. Plus three novelty layers on top: adaptive routing by question complexity, multi-signal confidence-aware rejection, and a hallucination error-type taxonomy. **16 experiments → 12 + 1 results tables → thesis.**

Why it matters: nobody has done this comparison on a medical benchmark covering accuracy + hallucination + explainability *together*.

---

## 30-second tech stack

- **Python 3.12** in a local `.venv/` on a MacBook M1 Pro · 16 GB
- **LLaMA 3.3 70B** (via Groq) as the answerer · same model across all architectures
- **BGE-large-en-v1.5** embeddings · **ChromaDB** vector store · **rank-bm25** sparse index · 400/80-token chunks
- **GPT-4o** as golden-set constructor · **Claude Sonnet 4.6** as RAGAS judge (three-family separation kills evaluator bias)
- **RAGAS** + LIME + SHAP for evaluation and explainability
- **Streamlit** for an optional cached-mode demo UI (Phase 10)

Full details in [`docs/tech_stack.md`](docs/tech_stack.md).

---

## Quick start

```bash
git clone <this-repo> && cd thesis-project

# 1. Create the venv (Python 3.12)
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Add API keys
cat > .env <<EOF
GROQ_API_KEY=your-groq-key
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
EOF

# 3. Register the Jupyter kernel
python -m ipykernel install --user --name thesis-rag --display-name "Python 3.12 (thesis-rag)"

# 4. Run the EDA notebook end-to-end
jupyter lab notebooks/00_data_processing_and_eda.ipynb
# OR headless:
jupyter nbconvert --to notebook --execute --inplace notebooks/00_data_processing_and_eda.ipynb
```

---

## Where to find things

| What | Where |
|---|---|
| **What this thesis does (plain English)** | [`docs/beginners_guide.md`](docs/beginners_guide.md) |
| **The execution plan** (locked decisions, 16 experiments, costs, risks) | [`plan.md`](plan.md) |
| **What to do next** (working checklist + decision log) | [`docs/todo.md`](docs/todo.md) |
| **Tech stack** (every tool & why) | [`docs/tech_stack.md`](docs/tech_stack.md) |
| **The data** (MedQA + 18 textbooks) | [`docs/dataset.md`](docs/dataset.md) |
| **Code architecture** | [`docs/architecture.md`](docs/architecture.md) |
| **Original proposal mapping** | [`docs/thesis_understanding.md`](docs/thesis_understanding.md) |
| **The original proposal PDF + 12-table workbook** | [`docs/thesis-files/`](docs/thesis-files/) |
| **Full docs index** | [`docs/README.md`](docs/README.md) |

For AI agents working on this project: see [`AGENTS.md`](AGENTS.md) (canonical) and [`CLAUDE.md`](CLAUDE.md) (Claude-Code-specific).

---

## Status

| Phase | Description | Status |
|---|---|---|
| 1 | Data processing & EDA | ✅ done |
| 2 | Chunking + embedding + indices | ⏳ next |
| 3 | Golden RAGAS dataset (300 rows, staged) | ⏳ pending |
| 4 | Group A — baseline RAG (EXP_01–EXP_05) | ⏳ pending |
| 5 | Group B — adaptive routing (EXP_06–EXP_07) | ⏳ pending |
| 6 | Group C — explainability (EXP_10–EXP_12) | ⏳ pending |
| 7 | Group C — confidence rejection (EXP_08–EXP_09) | ⏳ pending |
| 8 | Group D — hallucination taxonomy (EXP_13–EXP_15) | ⏳ pending |
| 9 | Group E — final synthesis (EXP_16) | ⏳ pending |
| 10 | Demo UI (optional) | ⏳ optional |

Live state: see [`docs/todo.md`](docs/todo.md).

---

## License & data provenance

- **MedQA dataset**: Jin et al. (2020), CC-BY-4.0. No PHI.
- **Textbook corpus**: shipped alongside MedQA; used for retrieval grounding only, not redistributed.
- This research is **non-clinical** — no real patients, no PHI, no clinical deployment intended.

See proposal §7.11.1 for full ethical compliance statement.
