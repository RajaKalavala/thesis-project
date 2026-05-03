# CLAUDE.md — Claude Code Instructions

> **Read [`AGENTS.md`](AGENTS.md) first.** It contains the canonical instructions that apply to any AI agent working on this repo. This file holds only the Claude-Code-specific bits on top.

---

## Where the context lives

Before doing any work, load the picture by reading these in order:

1. [`AGENTS.md`](AGENTS.md) — hard rules + working style
2. [`docs/README.md`](docs/README.md) — docs index with reading order
3. [`plan.md`](plan.md) — locked decisions (§0) + 16-experiment programme
4. [`docs/todo.md`](docs/todo.md) — current working state + decision log

When the user asks something specific, jump straight to the matching doc using the reverse-lookup table in [`docs/README.md` §2](docs/README.md).

---

## Memory layout

Persistent memory for this project lives at:

```
~/.claude/projects/-Users-rajak-Workstation-Projects-myGitHub-thesis-project/
├── MEMORY.md                              ← index (always loaded)
└── memory/
    ├── user_role.md                       ← MSc AI/ML student at LJMU
    ├── project_thesis_overview.md         ← locked decisions snapshot
    └── feedback_step_by_step.md           ← smoke-test discipline + style preferences
```

When a locked decision changes, update **all three** of: `plan.md §0` · `docs/tech_stack.md §3` · `memory/project_thesis_overview.md`. Sync order is in [`AGENTS.md` §7](AGENTS.md).

---

## Tool-use conventions specific to Claude Code

- **Use the Edit tool** for editing existing files; never overwrite with Write unless the diff would be larger than the file itself.
- **Use Bash via `.venv/bin/python`** for any Python invocation, not system `python3`.
- **Use the TodoWrite tool** for any task that has more than 2 distinct steps. The user explicitly asked for one-step-at-a-time work — TodoWrite makes the sequence visible.
- **Don't run `git commit` or `git push`** unless the user explicitly asks. The user reviews diffs in their editor before committing.
- **Don't `rm -rf` anything** without showing what's about to be deleted and getting confirmation, *except* for clearly-scoped temp files in `/tmp/`.
- **For long-running background tasks** (pip install, model downloads, embeddings runs), use `run_in_background: true` on Bash and notify when done.

---

## Known user preferences (also recorded in memory)

- **Smoke-test discipline** — propose small-data versions before any expensive run. Show the cost differential.
- **Lead with *why* → *how* → *example*** — beginner-level explanations preferred. Lots of analogies.
- **Be honest, push back on weak ideas** — the user has reversed three decisions on this project after pushback and appreciates it.
- **Don't waste API credits** — favour caching, resumability, and small validation runs.
- **One step per session** — finish, verify, then move on.

---

## Quick reference

| Need to… | Look here |
|---|---|
| Understand the thesis claim | [`docs/beginners_guide.md`](docs/beginners_guide.md) |
| See locked decisions | [`plan.md §0`](plan.md) |
| Find next task | [`docs/todo.md`](docs/todo.md) |
| Check tech stack | [`docs/tech_stack.md`](docs/tech_stack.md) |
| Reference data schema | [`docs/dataset.md`](docs/dataset.md) |
| Read code structure | [`docs/architecture.md`](docs/architecture.md) |

When in doubt, ask the user. This is a thesis — reversibility beats speed.
