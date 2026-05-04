# Beginner's Guide — The Whole Thesis in Plain English

> **Purpose.** Read this before any other doc when you need to remember **what the thesis is actually doing and why**. No jargon-first explanations. Lots of analogies. Lots of examples.
>
> Companions for when you're ready to dive in: [plan.md](../plan.md) (decisions + sequence) · [todo.md](todo.md) (working checklist) · [architecture.md](architecture.md) (how the code is organised) · [dataset.md](dataset.md) (the data) · [tech_stack.md](tech_stack.md) (every tool & why)

---

## 1. The 30-second elevator pitch

Doctors are starting to use AI to answer medical questions. The AI sometimes confidently gives **wrong** answers — that's called a *hallucination*. In medicine, a wrong answer can hurt or kill a patient.

There's a known fix: instead of asking the AI to remember everything, give it a stack of medical textbooks and tell it to **look things up** before answering. This is called **RAG** — *Retrieval-Augmented Generation*.

But there are **many different ways** to look things up. Nobody has done a careful, controlled comparison of these methods on medical questions.

**Your thesis fills that gap.** You will build 4 different "look-up methods", run them on the same 12,723 medical exam questions with the same AI, and measure which method:
- Gets the most answers right (**accuracy**)
- Sticks to the textbooks (**faithfulness** — no hallucinations)
- Can show *why* it gave each answer (**explainability**)

Plus three novelty ideas you add on top:
- An **adaptive** system that picks a different look-up method depending on how hard the question is
- A **safety layer** that refuses to answer when it's not sure
- A **classification of mistakes** — what kinds of errors does each method make

---

## 2. The cooking-competition analogy

Picture a cooking competition with **4 chefs** competing to make the same dish.

| In the analogy | In the thesis |
|---|---|
| The dish to cook | A medical question to answer |
| The kitchen (oven, knives, ingredients in the pantry) | The shared infrastructure (LLM, textbooks, prompt) |
| The 4 chefs | 4 retrieval architectures |
| What chefs are competing on | How they *shop for ingredients* (i.e. how they retrieve passages from the textbooks) |
| The judges | RAGAS metrics + LIME/SHAP explainers |
| The winner's prize | One paragraph in your thesis: *"For X clinical scenario, architecture Y is best."* |

**The clever part:** the only thing that's different between the 4 chefs is *how* they shop. Same ingredients in the pantry, same recipe, same oven, same dish. So if Chef Multi-Hop wins, we know it's because of *how it shops*, not because they got better tools.

That single-variable design is what makes this a thesis-quality experiment.

---

## 3. Meet the 4 chefs — what each retrieval method does

Imagine the question is:

> *"A 65-year-old man on warfarin presents with hematemesis. What's the next step?"*

(Translation: an old guy on a blood thinner is vomiting blood — what should the doctor do?)

### Chef 1 — Naive RAG ("the meaning matcher")

- **What it does:** turns the question into a numerical "fingerprint" (an embedding), looks for textbook passages with similar fingerprints, picks the top 5.
- **Strengths:** good at finding passages that are *about the same topic*, even if they don't share exact words.
- **Weaknesses:** can miss passages that are technically relevant but use different vocabulary.
- **Analogy:** searching Google by typing in plain English.

### Chef 2 — Sparse RAG ("the keyword hunter")

- **What it does:** breaks the question into individual words ("warfarin", "hematemesis"), looks for textbook passages containing those exact words, picks the top 5.
- **Strengths:** great when the question contains specific medical terms that *must* appear in the answer's evidence (drug names, lab values).
- **Weaknesses:** misses passages that paraphrase ("anticoagulant" vs "warfarin").
- **Analogy:** searching with quotation marks: `"warfarin reversal"`.

### Chef 3 — Hybrid RAG ("both")

- **What it does:** runs Chef 1 AND Chef 2, then combines their top results with a method called Reciprocal Rank Fusion (RRF). The passages that show up in both lists rise to the top.
- **Strengths:** catches what either method alone would miss.
- **Weaknesses:** slightly slower; can be confused if the two methods strongly disagree.
- **Analogy:** asking two librarians and trusting passages they both recommend.

### Chef 4 — Multi-Hop RAG ("the detective")

- **What it does:** doesn't try to answer in one shot. Breaks the question into sub-questions, looks each one up separately, then reasons across the results.

Walking through the warfarin example:
1. **Hop 1** — *"What is hematemesis?"* → finds passages about upper GI bleeding.
2. **Hop 2** — AI realises it needs to know about anticoagulant management → searches *"warfarin reversal"* → finds vitamin K, FFP, PCC info.
3. **Hop 3** — AI realises it needs the urgency level → searches *"active GI bleeding initial management"* → finds resuscitation info.
4. Combines all three searches → answers.

- **Strengths:** handles complex questions that span multiple medical concepts.
- **Weaknesses:** 3× slower (3 LLM calls per question); can wander off-topic.
- **Analogy:** a detective who finds one clue, then uses it to find the next clue, then the next.

### Plus the No-RAG baseline (EXP_01)

- **What it does:** no retrieval at all. Just asks the AI directly.
- **Why include it:** if Naive RAG only beats No-RAG by 2 points, RAG isn't really helping. If it beats by 20 points, RAG is the real story. This is the *control group*.

---

## 4. The full workflow — 8 steps in plain English

### Step 1 — Get and clean the data ✅ DONE

**What you did.** Loaded the 12,723 medical exam questions and 18 textbooks. Looked at how long the questions are, how the textbooks are distributed, etc.

**Why this matters.** Every later step assumes the data is sane. If the medical questions had garbled formatting, every chef would lose. We want to discover that *now*, not at hour 28 of the big run.

**What was learned.**
- 12,723 questions split into train/dev/test = 10,178 + 1,272 + 1,273
- 18 textbooks totalling ~12.85 million words
- One textbook (Harrison's Internal Medicine) is 25% of the corpus → this will bias retrieval slightly. We acknowledge it.

**Smoke test.** The notebook ran end-to-end and produced 4 output files. ✓

---

### Step 2 — Cut up the textbooks (chunking)

**What you'll do.** Cut each textbook into small pieces of ~400 words each, with 80 words of overlap between consecutive pieces. End up with ~36,000 "cards."

**Why ~400 words?** A typical textbook paragraph that *answers* a USMLE question is 150–300 words. Smaller chunks (e.g. 200) often split a relevant paragraph in half — the chef finds half the answer, misses the other half. Bigger chunks (e.g. 800) drag in noise.

**Why the 80-word overlap?** Imagine a paragraph that ends with "...first-line treatment for community-acquired pneumonia is amoxicillin." If a chunk boundary falls right before "amoxicillin," that exact answer might never be retrieved. The overlap means consecutive chunks share their first/last 80 words, so important sentences appear in both.

**Example.** Take this textbook paragraph:
> "Hypertension is treated with first-line agents including ACE inhibitors, angiotensin receptor blockers, calcium channel blockers, and thiazide diuretics. ACE inhibitors block the conversion of angiotensin I to angiotensin II, leading to vasodilation..."

- Bad chunking (200/0): chunk 1 ends at "thiazide diuretics. ACE inhibitors". Chunk 2 starts at "block the conversion." → both chunks now make less sense alone.
- Good chunking (400/80): chunk 1 covers the full paragraph + start of next. Chunk 2 starts with the last 80 words of chunk 1's paragraph + continues. → both chunks retain context.

**Smoke test.** Print 5 random chunks. Read them. Are they sensible standalone passages? If you see chunks that are 90% bullet points or table fragments, the chunker needs tuning.

**Cost / time.** ~1–2 minutes on M1 Pro CPU. Free.

---

### Step 3 — Make the chunks searchable (embeddings + indices)

**What you'll do.** Take each chunk and turn it into a "fingerprint" of 1024 numbers. Store all fingerprints in two databases.

**What's an embedding?** Imagine taking the meaning of a passage and squashing it into a list of 1024 numbers (called a *vector*). Passages about the same topic end up with similar lists. Passages about different topics end up with very different lists.

**Example.** The chunks:
- "Aspirin is a non-steroidal anti-inflammatory drug..." → fingerprint A: `[0.34, -0.12, 0.88, ..., 0.21]`
- "Ibuprofen is a non-steroidal anti-inflammatory drug..." → fingerprint B: similar to A
- "The mitochondrion is the powerhouse of the cell..." → fingerprint C: very different

When the user later asks "What's a NSAID?", the question itself gets a fingerprint, and the system finds chunks whose fingerprints are *closest* to it. Chunks A and B will pop up; C won't.

**Why this embedder?** `BGE-large-en-v1.5` is a strong general-purpose 1024-d embedder that scores ~75 nDCG@10 on TREC-COVID (a medical-IR benchmark). A medical-fine-tuned ablation (e.g. MedEmbed-large) was scoped out of this thesis for compute-budget reasons and is identified as future work in the writeup.

**Why two databases?** ChromaDB stores the fingerprints for fast similarity search. BM25 stores the raw words for keyword search. Naive uses ChromaDB; Sparse uses BM25; Hybrid uses both.

**Smoke test.** Type a query like *"What is the first-line treatment for community-acquired pneumonia?"* into each database. Look at the top 3 results. Do they actually mention pneumonia and antibiotics? If yes, the indices are wired up correctly.

**Cost / time.** ~50–70 minutes total on M1 Pro CPU (25–35 min per embedder × 2). Free. **One-time job** — once built, every later experiment loads them in 30 seconds.

---

### Step 4 — Build a "study guide" (the golden dataset)

**What you'll do.** Take **300** of the 12,723 questions (sized for budget-efficiency — see plan.md §0 #9). For each one, ask `openai/gpt-oss-120b` (running free on Groq, recalibrated 2026-05-04 from GPT-4o) to:
1. Find the *exact* textbook passage that supports the right answer.
2. Write a 3-sentence reference explanation grounded in that passage.
3. List the atomic claims that any correct answer must cover.

**Why this is needed.** RAGAS, the evaluator that scores faithfulness and context recall, needs *reference answers* to compare against. The raw MedQA dataset only gives you the right multiple-choice letter — that's not enough for RAGAS. The golden dataset fills the gap.

**Example.** Question: *"A 21-year-old male with fever, dysuria, and knee pain. Best treatment?"*
- MedQA gives: correct answer = "Ceftriaxone"
- Golden dataset adds:
  - Reference passage: from Pharmacology Katzung, the section on cephalosporins for gonococcal infection
  - Reference explanation: "The clinical picture suggests disseminated gonococcal infection. Ceftriaxone, a third-generation cephalosporin, is the recommended treatment because it inhibits bacterial cell-wall synthesis and is effective against Neisseria gonorrhoeae."
  - Atomic claims: ["The answer must identify Ceftriaxone", "The mechanism must mention cell-wall synthesis", "The organism must be Neisseria gonorrhoeae"]

**Smoke test (CRITICAL).** Build **50 questions first** before the rest of the 300. Costs **$0** with the open-weights `gpt-oss-120b` constructor (recalibrated 2026-05-04 — was ~$2 with GPT-4o). The 50-row pilot is now also the **quality gate**: if accept rate drops below 80 % or JSON malformation exceeds 5 %, fall back to GPT-4o-mini ($1 pilot / $3 full) before scaling to 250 more.

**Cost / time.** **$0** on the locked plan (Groq free tier, `gpt-oss-120b` for all 300 rows) + ~1.5–2 hours wall-clock. The M1 Pro just orchestrates. Fallback ladder if pilot fails: GPT-4o-mini (~$3 for 300) → full GPT-4o (~$12).

---

### Step 5 — Run the 5 chefs (Phase 4 — the main experiment)

**What you'll do.** Run each of the 5 architectures (No-RAG + 4 RAGs) over all 12,723 questions. For each architecture, save:
- What it retrieved
- What answer it generated
- Whether it was correct
- The RAGAS scores (Faithfulness etc.)
- How long it took

All 5 chefs use the same BGE-large embedder; you run each one once. (An earlier plan considered running the dense-retrieval chefs twice with a medical-fine-tuned embedder — that was dropped to keep compute manageable, and is identified as future work.)

**Why this is the main event.** This is where you generate the data that fills 5 of the 12 results tables.

**Example.** EXP_02 Naive RAG running on the warfarin question:
1. Embed the question → fingerprint.
2. ChromaDB returns top-5 chunks. Top hit: a Harrison's passage on GI bleeding management.
3. Build a prompt: *"Given these 5 textbook passages, answer the question..."*
4. Send to LLaMA 3.3 70B via Groq. Get back: *"The next step is to administer IV fluids and prepare for endoscopy. Reverse anticoagulation with vitamin K and FFP."*
5. Score against the gold standard:
   - Exact match: did the AI pick the same A/B/C/D as the gold answer? (yes/no)
   - Faithfulness: are the AI's claims actually supported by the retrieved passages? (0–1, higher = better)
   - Context Recall: did the retrieved passages actually contain the gold answer? (0–1)

**Smoke test.** Run on **5 questions first**. Eyeball the outputs. Does the AI's answer make sense given what was retrieved? If retrieval is grabbing nonsense passages, fix the chunking/embedding before you spend 30 hours running the full 12,723.

**Why the cache matters here.** If you start a 6-hour run and it dies at hour 4 due to a Groq rate limit, every cached response is a free re-run. You restart, the cache hits, you continue from where you stopped. Without the cache, every restart costs another 6 hours. **This is the single most important engineering pattern in the whole project.**

**Cost / time.** ~50–60 hours wall-clock on Groq (broken across multiple overnight runs). API cost: small (Groq is cheap), plus ~$50–80 for the Claude RAGAS judge.

---

### Step 6 — The three novelty ideas (Groups B, C, D)

These are what make your thesis interesting beyond just "we compared 4 RAGs." They turn the comparison into a *recommendation framework*.

#### Novelty 1 — Adaptive routing (Phase 5)

**Problem.** Multi-Hop is the most accurate but also 3× slower. If a question is simple ("What does the heart do?"), running Multi-Hop is overkill — Naive would have answered just as well, faster, cheaper.

**Solution.** Build a classifier that labels each question as Simple / Moderate / Complex (using question length, option count, presence of cue words like "most likely" or "best next step"). Then route:
- Simple → Naive (cheapest)
- Moderate → Hybrid (balanced)
- Complex → Multi-Hop (best for hard cases)

**Thesis claim.** "Adaptive routing achieves Y% of Multi-Hop's accuracy at 60% of the cost."

#### Novelty 2 — Confidence-aware rejection (Phase 7)

**Problem.** Even the best architecture sometimes gives confident wrong answers. In medicine, a confident wrong answer is worse than no answer.

**Solution.** Build an 8-dimensional "confidence score" combining: retrieval certainty + RAGAS Faithfulness + Answer Relevancy + LIME-SHAP agreement. If confidence < threshold (e.g. 0.7), the system says *"Evidence insufficient — defer to a human"* instead of guessing.

**Thesis claim.** "At threshold 0.7, the system rejects 18% of questions but the remaining 82% have hallucination rate dropped from 14% to 4%." Trade-off visualised in Table 11.

#### Novelty 3 — Hallucination taxonomy (Phase 8)

**Problem.** "Hallucination rate = 14%" tells you nothing about *what kind* of mistakes the model makes.

**Solution.** Define 6 categories of mistakes (unsupported diagnosis, unsupported treatment, wrong reasoning chain, partial evidence misuse, option mismatch, context omission). Manually label 150 hallucinated outputs. Cross-tab: which architecture makes which kind of mistake?

**Thesis claim.** "Naive RAG fails most often by *context omission* (relevant passage not retrieved). Multi-Hop fails most often by *wrong reasoning chain* (hops drift off-topic). The failure modes are diagnostic of architectural strengths and weaknesses."

---

### Step 7 — Final synthesis (Phase 9)

**What you'll do.** Combine every architecture's metrics into a single weighted score:

```
final_score = 0.25 · accuracy
            + 0.25 · faithfulness
            + 0.20 · retrieval quality
            + 0.15 · safety (rejection)
            + 0.10 · explainability
            + 0.05 · latency
```

Rank the architectures. Map ranks to deployment recommendations: *"For low-cost simple lookups, use Naive. For balanced production deployments, use Adaptive. For complex reasoning, use Multi-Hop."*

This is **Table 12** — the crown of the thesis.

---

### Step 8 — Thesis writing + (optional) demo UI

The thesis chapters lift directly from these documents:
- **Methodology chapter** ← `plan.md` + `architecture.md` + `dataset.md` + `tech_stack.md`
- **Results chapter** ← the 12 + 1 results tables you've populated
- **Discussion** ← what each architecture's failure pattern means + LIME/SHAP findings + the hallucination taxonomy
- **Conclusion** ← one paragraph per use case from Table 12

Optionally: build a Streamlit demo UI so the viva examiner can *see* the architectures comparing answers side-by-side, instead of reading numbers in tables. (See `plan.md` §12 for the optional Phase 10.)

---

## 5. The smoke-test discipline — the rule that saves you weeks

**The rule.** Before any task that costs >$1 in API calls or >30 minutes wall time, run a tiny version on 3–50 questions first. Verify the output is sane. Then scale up.

| Big task | Tiny test before it | Saves |
|---|---|---|
| Build 300-row golden dataset (~$12, ~2 h) | Build 50-row golden first (~$2, ~30 min) | $10 + 1.5 h if there's a bug |
| Run EXP_02 on 12,723 questions (~6 h Groq) | Run EXP_02 on 5 dev questions first (~1 min) | 6 hours of wasted Groq time + having to debug a long log |
| Run LIME/SHAP on 200 questions (~3 h) | Run LIME on 5 questions first (~10 min) | 3 hours of broken explanation outputs |
| Embed all 36k chunks twice (~50 min) | Embed 100 chunks first (~30 sec) | 50 minutes of wrong chunk count or wrong overlap |

**The mental script for every long-running step:**
1. *"What's the smallest version of this I can run that would catch the most likely bugs?"*
2. *"What does 'sensible output' look like? If I see X, that's probably right; if I see Y, something's wrong."*
3. *"What's the cost differential between the small and the large version?"*

If the answer to (3) is "less than 5%", run the small version. Always.

**Plus: cache everything.** Even when you do scale up, every API response gets cached on disk by `(model, temp, prompt_hash)`. If you find a bug at hour 5 of a 6-hour run and need to fix it, you re-run only the affected calls — the rest are free cache hits.

---

## 6. How this becomes a thesis — example results table

Here's a *fictional* version of what your final Table 12 might look like (real numbers TBD):

| Architecture | Accuracy | Faithfulness | Hallucination Rate | Latency | Final Weighted Score | Best Use Case |
|---|---:|---:|---:|---:|---:|---|
| No-RAG | 0.62 | 0.41 | 38% | 1.2 s | 0.48 | Baseline only |
| Naive RAG (BGE) | 0.74 | 0.71 | 18% | 1.8 s | 0.69 | Low-cost simple lookups |
| Sparse RAG | 0.69 | 0.78 | 12% | 1.6 s | 0.69 | Exact-term medical queries |
| Hybrid RAG (BGE) | 0.78 | 0.82 | 9% | 2.1 s | 0.76 | Balanced retrieval |
| Hybrid RAG | 0.78 | 0.82 | 9% | 2.1 s | 0.76 | Balanced retrieval |
| Multi-Hop RAG | 0.81 | 0.85 | 7% | 5.4 s | 0.78 | Complex multi-step reasoning |
| **Adaptive RAG** | **0.79** | **0.83** | **8%** | **2.6 s** | **0.78** | **Production deployment** |

From this one table, your thesis defends:

1. **RAG dramatically reduces hallucinations** — No-RAG hallucinates on 38% of answers; the best RAG drops it to 7%.
2. **Architecture choice matters** — there's a 19-point spread between worst and best RAG.
3. **Multi-Hop wins on accuracy but at 3× the latency** — the trade-off is real and quantified.
4. **Adaptive routing is the practical winner** — within 1 point of the best fixed architecture but at half the latency.
5. **Confidence rejection halves hallucinations** — separate Table 11 result.
6. **Each architecture has signature failure modes** — separate Table 7 result (the taxonomy).

That's 6 thesis-defensible claims from one experiment programme. Plus the methodological contribution: *"This is the first controlled, single-variable comparison of multiple RAG architectures on a medical benchmark covering accuracy + hallucination + explainability simultaneously."*

---

## 7. The full picture — visual summary

```
                  ┌─────────────────────────────────────┐
                  │   12,723 medical questions          │
                  │   18 medical textbooks              │
                  └──────────────┬──────────────────────┘
                                 │
                                 ▼
            ┌─────────────────────────────────────────┐
            │  Step 2: Chunking — cut into ~36k cards │
            │  Step 3: Embed with BGE-large           │
            │          + BM25 keyword index           │
            └──────────────┬──────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
   ┌──────────────────┐     ┌────────────────────────────┐
   │ Step 4: Build    │     │ Step 5: Run 5 chefs over   │
   │ golden 300       │     │ all 12,723 questions       │
   │ (gpt-oss-120b)   │     │ — both embedders for the 3 │
   └─────────┬────────┘     │ dense-retrieval ones       │
             │              └─────────────┬──────────────┘
             └─────┬────────────────────┘
                   │
                   ▼
         ┌─────────────────────────────┐
         │ Score with RAGAS (Claude    │
         │ as judge) + non-LLM metrics │
         └────────────┬────────────────┘
                      │
                      ▼
        ┌─────────────────────────────────────────┐
        │ Step 6: Add the three novelties         │
        │  - Adaptive routing                     │
        │  - Confidence-aware rejection           │
        │  - Hallucination taxonomy               │
        └────────────┬────────────────────────────┘
                     │
                     ▼
            ┌──────────────────────────┐
            │ Step 7: Final synthesis  │
            │ (weighted ranking)       │
            └────────────┬─────────────┘
                         │
                         ▼
        ┌─────────────────────────────────────┐
        │ Step 8: Thesis report (+ optional   │
        │         Streamlit demo)             │
        └─────────────────────────────────────┘
```

---

## 8. What "done" looks like

You're done with the project when:

1. ☐ All 12 + 1 results tables in the Excel workbook are filled
2. ☐ Every claim in the thesis traces back to a specific row in a results table or a specific notebook output
3. ☐ The thesis methodology section explains: why these 4 architectures, why this dataset, why this LLM, why this embedder (BGE-large + future-work caveat), why Claude as judge, why 300 golden, why 80-token overlap
4. ☐ The discussion chapter has at least 5 distinct findings (one per claim listed in §6 above)
5. ☐ You can answer in the viva: *"What would you change if you started over?"* with a thoughtful answer (not "nothing")
6. ☐ (Optional) Streamlit demo runs at a public URL and you can demo it in the viva

---

## 9. The sequence in plain English (what to actually do next)

This is the same as `plan.md` §16 but in beginner language:

1. **Right now:** make sure your `.env` has the **2 required** API keys: `GROQ_API_KEY` (generator + constructor — both via Groq) and `ANTHROPIC_API_KEY` (RAGAS judge). `OPENAI_API_KEY` is now optional. Confirm with a 3-call smoke test.
2. **Next session:** Notebook 01 — chunk the textbooks. Eyeball 5 chunks; do they look sensible?
3. **Following session:** Notebook 02 — embed everything (twice) + build indices. Smoke-query *"pneumonia treatment"*; do the top 3 results mention pneumonia?
4. **Then:** Build a few `src/` modules (loaders, the Groq client, the cache decorator).
5. **Then:** Notebook 03 — full smoke test on 3 dev questions end-to-end. If all 3 produce sensible answers, infrastructure is solid.
6. **Then:** Notebook 04 — golden dataset. **Pilot 50 first** ($2), then 250 more = 300 total ($10 more).
7. **(Optional) Then:** Phase 10 Stage A — Streamlit app scaffolding. Locks the `summary.json` shape before expensive runs.
8. **Then:** the big experiments. Run **EXP_01 No-RAG first** because it's the simplest — flushes any remaining bugs before the more complex experiments.
9. **Then:** EXP_02 → EXP_05 in sequence. Smoke each on 5 questions before scaling to 12,723.
10. **Then:** Phase 5 → 6 → 7 → 8 → 9. Each consumes the prior phase's outputs.
11. **Final:** thesis writing + optional UI polish.

At every step: **smoke test, verify, then scale.**

---

*This guide is for people new to ML/RAG. When you're ready for the technical depth, see `plan.md` (decisions), `architecture.md` (code), `dataset.md` (data), `tech_stack.md` (tools & rationale), `thesis_understanding.md` (proposal mapping). When you forget the big picture, come back here.*
