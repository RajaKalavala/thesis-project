# `docs/` — Documentation Index

> Central navigation for every documentation artefact in this repo. **If you're new here, start with [`beginners_guide.md`](beginners_guide.md).** If you're returning and need one specific thing, jump straight to the relevant doc using the table below.

---

## 1. Reading order — first time through

| # | Doc | Why read it | Time |
|---|---|---|---|
| 1 | [`beginners_guide.md`](beginners_guide.md) | The whole thesis explained in plain English with analogies. No jargon-first, lots of examples. | 20 min |
| 2 | [`thesis_understanding.md`](thesis_understanding.md) | Maps the original proposal (PDF) to the actual implementation. Useful when an examiner asks *"how does X relate to your proposal §Y?"* | 15 min |
| 3 | [`tech_stack.md`](tech_stack.md) | Every tool / model / library used and **why each one was picked** (and what was rejected). | 10 min |
| 4 | [`dataset.md`](dataset.md) | Field-level reference for the MedQA data + textbook corpus. EDA findings, schema, gotchas. | 15 min |
| 5 | [`architecture.md`](architecture.md) | How the code is structured (`src/` modules, notebooks, results), the dev workflow, hardware split. | 20 min |
| 6 | [`../plan.md`](../plan.md) | The **execution master**: locked decisions (§0), 16-experiment programme, cost & runtime budgets, risks. The single source of truth for *what to do next*. | 30 min |
| 7 | [`todo.md`](todo.md) | Working checklist mirroring `plan.md`'s phases — granular tasks with acceptance criteria + decision log. | 10 min |

**One-pass total: ~2 hours** to fully load the project context.

---

## 2. Reverse-lookup — *"I just need..."*

| If you're looking for… | Go to |
|---|---|
| The **central thesis claim** in plain English | [`beginners_guide.md` §1](beginners_guide.md) |
| The **proposal mapping** (PDF §X → implementation) | [`thesis_understanding.md`](thesis_understanding.md) |
| Why **BGE-large** was chosen and **MedEmbed dropped** | [`tech_stack.md` §3.1](tech_stack.md) |
| Why **ChromaDB** instead of FAISS | [`tech_stack.md` §3.2](tech_stack.md) |
| Why **Claude as judge** vs GPT-4o-mini | [`tech_stack.md` §3.4](tech_stack.md) |
| Whether to use **LangChain / LangSmith / RunPod / Colab** | [`tech_stack.md` §3.6](tech_stack.md) |
| The **MedQA dataset schema** | [`dataset.md` §2.2](dataset.md) |
| The **18-textbook corpus stats** + Harrison-bias note | [`dataset.md` §3](dataset.md) |
| The **`src/` module layout** + public API contract | [`architecture.md` §3](architecture.md) |
| The **caching strategy** (why it matters) | [`architecture.md` §6](architecture.md) |
| **What runs on M1 Pro vs Colab** | [`architecture.md` §8](architecture.md) |
| The **16 experiments** (EXP_01 → EXP_16) | [`../plan.md` §6–§11](../plan.md) |
| The **golden-dataset construction** workflow | [`../plan.md` §5](../plan.md) |
| The **cost budget** ($25–35 total) | [`../plan.md` §14](../plan.md) |
| The **risk register** | [`../plan.md` §15](../plan.md) |
| **What to do this session** | [`todo.md`](todo.md) (find the next `[ ]` checkbox) |
| The **decision log** for the viva | [`todo.md` bottom](todo.md) |
| The **Streamlit demo UI** spec (Phase 10, optional) | [`../plan.md` §12](../plan.md) |

---

## 3. File map

```
docs/
├── README.md              ← this file (you are here)
├── beginners_guide.md     ← plain-English thesis explanation
├── thesis_understanding.md ← proposal-to-implementation map
├── tech_stack.md          ← canonical tech reference
├── dataset.md             ← MedQA + textbook corpus reference
├── architecture.md        ← code structure + dev workflow + hardware
├── todo.md                ← working checklist + decision log
└── thesis-files/
    ├── RajaKalavala_PN1196988_OriginalProposal.pdf  (original proposal)
    └── Raja Kalavala Final Thesis Project Sheet.xlsx (12 results tables)
```

```
repo root /
├── README.md              ← short project landing page
├── AGENTS.md              ← canonical instructions for AI agents
├── CLAUDE.md              ← Claude Code-specific instructions
├── plan.md                ← THE execution master (kept at root for visibility)
└── ...
```

---

## 4. Source-of-truth hierarchy

When two docs disagree:

1. **`plan.md` §0 (locked decisions)** wins for any technical-stack or scope decision.
2. **`tech_stack.md`** is the consolidated reference — it should mirror `plan.md §0`. If they drift, sync `tech_stack.md` to `plan.md`.
3. **`todo.md` decision log** records *when* and *why* a decision was made — it's the historical reference, not the live state.
4. **`README.md`** at root is a courtesy landing page for anyone arriving cold; never the source of truth for technical detail.

If you change a locked decision, update in this order: `plan.md §0` → `tech_stack.md` → `architecture.md` (if the code shape changes) → `todo.md` decision log → memory (`~/.claude/.../memory/`).

---

## 5. What goes where (when adding new docs)

| If you're documenting… | Add it here |
|---|---|
| A new technical decision (model, library, hyperparameter) | `tech_stack.md` §3 + `plan.md §0` lock |
| A new data quirk or schema | `dataset.md` |
| A new code module's public API | `architecture.md` §3 |
| A new experiment phase or notebook | `plan.md` (new section) + `todo.md` (new checklist) |
| A new failure mode / risk | `plan.md §15` |
| A beginner-level explanation | `beginners_guide.md` |
| Source PDFs, supervisor letters, the experiment workbook | `thesis-files/` |

---

*This index lives at `docs/README.md`. The companion landing page at the repo root is just `README.md` — it points here.*
