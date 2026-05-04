# Dataset Reference — MedQA Questions & Textbook Corpus

Field-level reference for everything under [medqa-data/](../../medqa-data/), with the headline EDA findings from [notebooks/00_data_processing_and_eda.ipynb](../../notebooks/00_data_processing_and_eda.ipynb). All numbers in this doc come from running that notebook against the current repo state — re-run it to refresh.

> Companion docs: [`thesis_understanding.md`](thesis_understanding.md) · [`tech_stack.md`](tech_stack.md) · [`../plan.md`](../plan.md) (golden-set construction is in `plan.md` §5)

---

## 1. What's in the repo

```
medqa-data/
├── questions/
│   ├── US/                         ← in scope (English)
│   │   ├── train.jsonl                10,178  questions  (5-option)
│   │   ├── dev.jsonl                   1,272  questions  (5-option)
│   │   ├── test.jsonl                  1,273  questions  (5-option)
│   │   ├── US_qbank.jsonl             14,369  questions  (full pool, superset of splits)
│   │   ├── 4_options/              ← reduced to 4 options + metamap_phrases attached
│   │   │   ├── phrases_no_exclude_train.jsonl   10,178
│   │   │   ├── phrases_no_exclude_dev.jsonl      1,272
│   │   │   └── phrases_no_exclude_test.jsonl     1,273
│   │   └── metamap_extracted_phrases/{train,dev,test}/
│   ├── Mainland/                  ← out of scope (Chinese)
│   └── Taiwan/                    ← out of scope (Chinese / translated)
└── textbooks/
    ├── en/   ← 18 English medical textbooks  (in scope, ~12.85M words)
    ├── zh_paragraph/   ← out of scope (Chinese)
    └── zh_sentence/    ← out of scope (Chinese)
```

This thesis uses **only** `questions/US/` (English) and `textbooks/en/`. The Chinese variants are present in the upstream MedQA distribution but excluded per proposal §6 (out-of-scope).

---

## 2. MedQA US questions

### 2.1 Two parallel variants — pick the right one for the right job

| Variant | File pattern | Options | Extra fields | When to use |
|---|---|---|---|---|
| **5-option (canonical MedQA)** | `questions/US/{train,dev,test}.jsonl` | A–E | — | Reproducing original MedQA accuracy numbers; matches the published benchmark. |
| **4-option** | `questions/US/4_options/phrases_no_exclude_{split}.jsonl` | A–D | `metamap_phrases` (pre-extracted clinical concepts) | All 16 thesis experiments — the 4-option variant matches the proposal (which describes A–D) and ships with metamap phrases that downstream code uses for retrieval-side query enrichment. |

**Important caveat — the answer letter is not stable across variants.** When the 5-opt → 4-opt reduction removes a distractor, remaining options are re-lettered. So a question whose correct answer was `D` in the 5-opt file may be `B` in the 4-opt file. The `answer` text is preserved; the `answer_idx` letter is not. **Never join the two variants by `answer_idx` — join by `question` text.**

### 2.2 Schema

Per JSONL row:

| Field | Type | 5-opt | 4-opt | Description | Example |
|---|---|:---:|:---:|---|---|
| `question` | str | ✓ | ✓ | Clinical scenario / question stem | `"A 67-year-old man with transitional cell carcinoma..."` |
| `answer` | str | ✓ | ✓ | Verbatim text of the correct option | `"Cross-linking of DNA"` |
| `options` | dict[str → str] | ✓ | ✓ | Letter → option text | `{"A": "...", "B": "...", "C": "...", "D": "...", "E": "..."}` |
| `answer_idx` | str | ✓ | ✓ | Letter of the correct option (`"A"`–`"E"` or `"A"`–`"D"`) | `"E"` |
| `meta_info` | str | ✓ | ✓ | USMLE step bucket | `"step1"` or `"step2&3"` |
| `metamap_phrases` | list[str] | ✗ | ✓ | NCBI MetaMap-extracted clinical concepts from the question text | `["67 year old man presents", "fever", ...]` |

There is **no** explicit `id` field in the upstream files. The fresh-build golden dataset (Notebook 04, see `plan.md` §5) will generate its own `golden_id` and `question_id` when constructing reference rows.

### 2.3 Quality checks (current state)

Direct from `data/processed/eda_summary.json`:

| Check | 5-opt | 4-opt |
|---|---|---|
| Total rows | 12,723 | 12,723 |
| Missing `question` / `answer` / `answer_idx` / `options` / `meta_info` | 0 | 0 |
| `options` count per row (min / max) | 5 / 5 | 4 / 4 |
| Rows where `options[answer_idx] != answer` | **0** | **0** |
| Duplicate `question` text | **2** | **2** |

The two duplicate-question rows are real near-identical questions in the upstream MedQA pool — they don't break anything but should be deduplicated before any per-question evaluation join.

### 2.4 Splits & step mix

| Split | Rows | Step 1 | Step 2&3 | Long-vignette (>200 words) |
|---|---:|---:|---:|---:|
| train | 10,178 | 5,629 | 4,549 | 400 |
| dev | 1,272 | 701 | 571 | 56 |
| test | 1,273 | 679 | 594 | 62 |
| **Total** | **12,723** | **7,009** (55.1%) | **5,714** (44.9%) | **518** (4.07%) |

`train + dev + test = 12,723` — matches the proposal's published count. `US_qbank.jsonl` (14,369) is the full upstream question pool that the splits were sampled from; the splits are the canonical evaluation surface and `US_qbank` is rarely used directly.

### 2.5 Question length

Word count per question (from `question` field, whitespace-tokenised):

| Stat | Value (words) |
|---|---:|
| Mean | 116.6 |
| Median | 112 |
| Std | 44.9 |
| 25th pct | 84 |
| 75th pct | 144 |
| 90th pct | 174 |
| 95th pct | 195 |
| 99th pct | 240 |
| Max | 530 |

**Long-vignette flag (>200 words):** 518 / 12,723 = **4.07%** of the corpus.

This is a critical number to remember:
- The **golden-set construction** (`plan.md` §5) deliberately oversamples long vignettes to ~20% of its 300-question target (60 rows forced long). Don't conflate the two rates — the golden set is *not* a representative sample of MedQA.
- The thesis's Multi-Hop RAG architecture is hypothesised to win on long vignettes specifically. Stratifying eval results by `is_long_vignette` is therefore meaningful, but the underlying base rate is small (~4%) — keep that in mind when interpreting per-stratum confidence intervals.

### 2.6 Correct-answer letter balance

5-option (per letter, % of total):

| A | B | C | D | E |
|---:|---:|---:|---:|---:|
| 20.12 | 21.54 | 20.30 | 20.30 | 17.74 |

4-option (per letter, % of total):

| A | B | C | D |
|---:|---:|---:|---:|
| 25.68 | 25.77 | 25.58 | 22.97 |

Roughly even in both variants. The slight `D` deficit in 4-opt and `E` deficit in 5-opt are mild but probably worth noting in the writeup as a sanity check that the LLM cannot "cheat" by always picking a single letter.

### 2.7 Metamap phrases (4-option variant only)

| Stat | Value |
|---|---:|
| Mean phrases per question | 35.94 |
| Median | 34 |
| Used by | Golden-set retrieval query construction (`question + answer + metamap_phrases` joined into one search string — see `plan.md` §5 Stage B). Also a candidate input feature for the EXP_06 complexity classifier. |

For the four RAG architectures' retrieval, you can choose to either:
1. **Use only the question text** (most faithful to clinical realism — a doctor wouldn't pre-extract phrases).
2. **Use `question + " " + " ".join(metamap_phrases)`** (better recall, used by the golden-set builder). The proposal doesn't pin this — pick during EXP_02 validation and fix for the rest.

---

## 3. Textbook corpus (English) — the retrieval knowledge base

18 textbooks, 12,851,737 words total (~88 MB raw text). Sorted by size:

| # | Book | File | Words | Share | MB |
|---:|---|---|---:|---:|---:|
| 1 | InternalMed_Harrison | `InternalMed_Harrison.txt` | 3,206,314 | **24.95%** | 21.34 |
| 2 | Surgery_Schwartz | `Surgery_Schwartz.txt` | 1,601,640 | 12.46% | 10.95 |
| 3 | Neurology_Adams | `Neurology_Adams.txt` | 1,266,188 | 9.85% | 8.00 |
| 4 | Obstentrics_Williams | `Obstentrics_Williams.txt` | 958,801 | 7.46% | 6.28 |
| 5 | Gynecology_Novak | `Gynecology_Novak.txt` | 816,205 | 6.35% | 5.39 |
| 6 | Cell_Biology_Alberts | `Cell_Biology_Alberts.txt` | 754,863 | 5.87% | 4.67 |
| 7 | Pharmacology_Katzung | `Pharmacology_Katzung.txt` | 730,734 | 5.69% | 4.90 |
| 8 | Immunology_Janeway | `Immunology_Janeway.txt` | 498,650 | 3.88% | 3.18 |
| 9 | Histology_Ross | `Histology_Ross.txt` | 462,808 | 3.60% | 2.91 |
| 10 | Pathology_Robbins | `Pathology_Robbins.txt` | 453,577 | 3.53% | 3.63 |
| 11 | Pediatrics_Nelson | `Pediatrics_Nelson.txt` | 425,380 | 3.31% | 2.87 |
| 12 | Psichiatry_DSM-5 | `Psichiatry_DSM-5.txt` | 418,604 | 3.26% | 2.77 |
| 13 | Physiology_Levy | `Physiology_Levy.txt` | 408,117 | 3.18% | 2.93 |
| 14 | Anatomy_Gray | `Anatomy_Gray.txt` | 352,200 | 2.74% | 2.18 |
| 15 | Biochemistry_Lippincott | `Biochemistry_Lippincott.txt` | 203,998 | 1.59% | 1.29 |
| 16 | First_Aid_Step2 | `First_Aid_Step2.txt` | 146,337 | 1.14% | 0.99 |
| 17 | First_Aid_Step1 | `First_Aid_Step1.txt` | 89,908 | 0.70% | 0.64 |
| 18 | Pathoma_Husain | `Pathoma_Husain.txt` | 57,413 | 0.45% | 0.38 |

> Filename typos preserved as-is on disk: `Obstentrics_Williams.txt` (should be `Obstetrics`) and `Psichiatry_DSM-5.txt` (should be `Psychiatry`). Don't "fix" these in code without renaming the files first or every chunk-id reference will break.

### 3.1 Corpus-frequency bias — the Harrison effect

**Top 3 books = 47.26% of all words.** `InternalMed_Harrison` alone is **~25%**. After chunking, the same book contributes ~25% of all chunks in the index, so its passages compete for top-k slots in **every** retrieval call regardless of what's actually being asked.

Consequence:
- Earlier exploratory retrieval runs found that ~34% of top-10 retrieved candidates came from Harrison's — *higher* than its 25% share — because Harrison's covers a broad range of clinical topics that match many question types.
- This bias is **shared by every architecture in the comparison** (same indices, same corpus), so it **cancels out for relative ranking**. It does *not* cancel out for absolute retrieval-quality claims about the system.
- For the thesis discussion chapter: acknowledge this bias as a limitation. Don't try to "fix" it (downsampling Harrison would distort the corpus's actual coverage of internal-medicine topics).

### 3.2 Format gotchas

The textbooks are **plain text** scraped from PDFs. Sample (first ~600 chars of `Pharmacology_Katzung.txt`):

```
(All nonresearch use illegal under federal law.)

Flunitrazepam (Rohypnol) Narcotics:

Hallucinogens:

LSD MDA, STP, DMT, DET, mescaline, peyote, bufotenine, ibogaine, psilocybin,
phencyclidine (PCP; veterinary drug only) (No telephone prescriptions, no refills.)2

Opioids: Opium: Opium alkaloids and derived phenanthrene alkaloids: codeine,
morphine (Avinza, Kadian, MSContin, Roxanol), hydrocodone and hydrocodone
combinations (Zohydro ER, Hycodan, Vicodin, Lortab), ...
```

Implications for chunking:
- **Sentence boundaries are unreliable** — many "sentences" are bullet fragments or table rows. A naive sentence-splitter will produce a long tail of 1–3-word "sentences." This is why the locked plan uses a **recursive 400-token chunker with 80-token overlap** (see `plan.md` §0 #5 and §4 Notebook 01) rather than sentence-based chunking.
- **Headings and section markers are lost** in the PDF→text conversion. Chunks are retrieved as flat strings; do not assume hierarchical structure.
- **Encoding issues** are minor but present (occasional ligature artefacts, broken accented characters). Use `errors="ignore"` when loading.

### 3.3 Subjects covered

The 18 books span the standard preclinical + clinical USMLE syllabus:

| Domain | Books |
|---|---|
| Foundational sciences | Cell_Biology_Alberts, Histology_Ross, Anatomy_Gray, Physiology_Levy, Biochemistry_Lippincott, Immunology_Janeway |
| Pathology | Pathology_Robbins, Pathoma_Husain |
| Pharmacology | Pharmacology_Katzung |
| Internal medicine | InternalMed_Harrison |
| Surgery | Surgery_Schwartz |
| Specialties | Neurology_Adams, Pediatrics_Nelson, Obstentrics_Williams, Gynecology_Novak, Psichiatry_DSM-5 |
| Board-prep summaries | First_Aid_Step1, First_Aid_Step2 |

This is roughly aligned with the USMLE Step 1 + Step 2&3 question pool — supports the assumption that retrieval can find answers for almost any MedQA question if the chunking + retrieval are good enough.

---

## 4. Two evaluation surfaces — raw 12,723 vs. golden subset

The thesis uses both, for different purposes:

| Property | Raw MedQA US (full) | Golden RAGAS subset (built in Notebook 04) |
|---|---|---|
| Source | `medqa-data/questions/US/` (4-option variant preferred) | `data/processed/golden_ragas_300.jsonl` (built in `plan.md` §5) |
| Size | **12,723 questions** (train + dev + test combined) | **300 stratified rows** — staged 50-pilot + 250-production |
| Built by | Original MedQA paper (Jin et al., 2020) | This thesis — Notebook 04, `gpt-4o`-via-OpenAI three-pass pipeline + automated audit (✅ produced 234 accepted of 300 attempted on 2026-05-04 at $6.61) |
| Has reference *explanations* | No (only `answer` and `answer_idx`) | Yes (`reference_explanation`, `gold_context`, `hallucination_check_points`) |
| Suitable for | **Exact-match accuracy**, **Retrieval Recall@K**, **latency** for all 16 experiments | **Full RAGAS suite** — Faithfulness, Context Precision, Context Recall, Answer Relevancy, Answer Correctness |
| Stratification fields | `meta_info`, length-based long-vignette flag | `question_type` ∈ {diagnosis, treatment, mechanism, management, other}, `requires_multihop`, plus all raw-MedQA fields |
| Long-vignette rate | 4.07% (representative) | ~20% (deliberately oversampled for Multi-Hop fairness, 60 forced rows) |

**Why every experiment evaluates on the full 12,723** — exact-match accuracy and retrieval recall don't need reference explanations; they only need `answer_idx` and the retrieved-chunk IDs. So they scale to the full corpus. The 300-row golden subset is *only* needed for the RAGAS-suite metrics that demand a reference answer + reference context.

---

## 5. Provenance & licensing

- **MedQA**: Jin et al. (2020), *"What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams."* arXiv:2009.13081. License: **CC-BY-4.0**. No PHI / patient data.
- **Textbook corpus**: shipped alongside MedQA in the same upstream distribution. Used here as a research artefact for retrieval grounding, **not** redistributed.
- **MetaMap phrases** (4-option variant): pre-extracted via NCBI's MetaMap tool by the MedQA authors.

This research is non-clinical: no real patients, no PHI, no clinical deployment. See proposal §7.11.1 for the full ethical compliance statement.

---

## 6. Processed artefacts (after running Notebook 00)

| File | Rows | Purpose |
|---|---:|---|
| `data/processed/medqa_5opt.parquet` | 12,723 | Cleaned 5-option dataset with derived columns: `question_word_count`, `question_char_count`, `is_long_vignette`, `avg_option_word_count`, `max_option_word_count`, `option_word_lengths_json`, `options_json`. Use this for accuracy reporting that matches published MedQA numbers. |
| `data/processed/medqa_4opt.parquet` | 12,723 | Cleaned 4-option dataset + `n_metamap_phrases`, `options_json`, `metamap_phrases_json`. **Use this for all 16 experiments.** |
| `data/processed/textbook_stats.parquet` | 18 | Per-book size / share. Cheap to load when you need to weight or stratify retrieval results by book of origin. |
| `data/processed/eda_summary.json` | — | Headline numbers for cross-checking against this doc; can be diffed in CI to detect dataset drift. |

These four files are the **canonical input** for every downstream notebook. Once they exist, no notebook needs to read the raw `medqa-data/` JSONL again until corpus indexing (Notebook 01 — to be built).

---

*Last refreshed: regenerate by running [notebooks/00_data_processing_and_eda.ipynb](../../notebooks/00_data_processing_and_eda.ipynb) end-to-end.*
