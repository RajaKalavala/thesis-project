"""Collect per-architecture raw metrics from disk for the EXP_16 ranking.

Three surfaces are read:
- `results/exp_*__test_1273/summary.json` → Acuuracy, mean_latency_s
- `results/exp_*__golden_234/summary.json` + `ragas_scores.csv` → RAGAS
  (Faithfulness, Hallucination_Rate, Context Precision, Context Recall)
- `results/exp_12_agreement/stage_b_retrievalchanged_mhop.jsonl` →
  LIME-SHAP explainability (Multi-Hop only; Variants inherit by routing share)

Adaptive variants do not have a directly-measured RAGAS — we re-derive them
via the per-question score-join (re-implementing notebook 05 §8 in a
self-contained, testable function).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# Canonical architecture order for the ranking table.
ARCHITECTURES: list[str] = [
    "NoRAG",
    "Naive",
    "Sparse",
    "Hybrid",
    "MultiHop",
    "Adaptive_A",
    "Adaptive_B",
]

# Per-architecture result-folder prefixes (test_1273 + golden_234 share the prefix).
_PREFIX: dict[str, str] = {
    "NoRAG": "exp_01_base_llm",
    "Naive": "exp_02_naive_rag",
    "Sparse": "exp_03_sparse_rag",
    "Hybrid": "exp_04_hybrid_rag",
    "MultiHop": "exp_05_multi_hop_rag",
    "Adaptive_A": "exp_07_adaptive_variant_a",
    "Adaptive_B": "exp_07_adaptive_variant_b",
}

# Routing tables for adaptive RAGAS score-join (matches notebook 05 §8 verbatim).
_ROUTING: dict[str, dict[str, str]] = {
    "Adaptive_A": {"Simple": "Naive", "Moderate": "Hybrid", "Complex": "MultiHop"},
    "Adaptive_B": {"Simple": "NoRAG", "Moderate": "MultiHop", "Complex": "MultiHop"},
}

# Per-architecture Groq calls per question. Constants from the architecture.
CALLS_PER_Q: dict[str, float] = {
    "NoRAG": 1.0,
    "Naive": 1.0,
    "Sparse": 1.0,
    "Hybrid": 1.0,
    "MultiHop": 3.0,
    # Adaptive computed dynamically from bucket counts × routed-arch's calls/Q.
}


@dataclass(frozen=True)
class ArchitectureMetrics:
    """Tidy container for per-architecture raw metrics.

    Attributes match the column names in the ranking CSV.
    """

    architecture: str
    accuracy_test_1273: float
    mean_latency_s_test_1273: float
    groq_calls_per_q: float
    faithfulness_golden_234: float
    hallucination_rate_golden_234: float
    context_precision_golden_234: float
    context_recall_golden_234: float
    answer_correctness_golden_234: float
    answer_relevance_golden_234: float
    lime_shap_spearman: float
    n_explainability_questions: int


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _test_1273_summary(repo_root: Path, arch: str) -> dict | None:
    return _load_json(repo_root / "results" / f"{_PREFIX[arch]}__test_1273" / "summary.json")


def _golden_234_summary(repo_root: Path, arch: str) -> dict | None:
    return _load_json(repo_root / "results" / f"{_PREFIX[arch]}__golden_234" / "summary.json")


def _golden_234_ragas_csv(repo_root: Path, arch: str) -> pd.DataFrame | None:
    path = repo_root / "results" / f"{_PREFIX[arch]}__golden_234" / "ragas_scores.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _load_complexity_bucket_lookup(repo_root: Path) -> dict[str, str]:
    """Build {question_text: 'Simple'|'Moderate'|'Complex'} from EXP_06 labels."""
    labels = pd.read_parquet(repo_root / "data/processed/complexity_labels.parquet")
    md = pd.read_parquet(repo_root / "data/processed/medqa_4opt.parquet")
    md = md.reset_index(drop=False).rename(columns={"index": "row_idx"})
    md["question_id"] = "medqa_" + md["row_idx"].astype(str)
    joined = labels.merge(md[["question_id", "question"]], on="question_id")
    return dict(zip(joined["question"], joined["complexity"].astype(str)))


def _load_golden_with_buckets(repo_root: Path, q_to_bucket: dict[str, str]) -> pd.DataFrame:
    path = repo_root / "data/processed/golden_ragas_300.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    df = pd.DataFrame(rows)
    df["question_id"] = df["question_id"].apply(lambda i: f"golden_{int(i):03d}")
    df["bucket"] = df["question"].map(q_to_bucket)
    return df


def _score_join_adaptive(
    repo_root: Path,
    routing: dict[str, str],
    golden_buckets: pd.DataFrame,
    ragas_per_arch: dict[str, pd.DataFrame],
) -> dict[str, float]:
    """Per-question score-join: pick the underlying arch's RAGAS value per Q,
    then mean. Returns means for every RAGAS metric in `RAGAS_METRICS`.
    """
    metrics = [
        "faithfulness",
        "context_precision",
        "context_recall",
        "answer_relevancy",
        "answer_correctness",
    ]
    acc: dict[str, list[float]] = {m: [] for m in metrics}
    for _, g in golden_buckets.iterrows():
        bucket = g["bucket"]
        if bucket not in routing:
            continue
        underlying = routing[bucket]
        if underlying not in ragas_per_arch:
            continue
        rag_table = ragas_per_arch[underlying]
        if g["question_id"] not in rag_table.index:
            continue
        row = rag_table.loc[g["question_id"]]
        for m in metrics:
            v = row.get(m)
            if v is not None and pd.notna(v):
                acc[m].append(float(v))
    return {m: (float(np.mean(vs)) if vs else float("nan")) for m, vs in acc.items()}


def _adaptive_calls_per_q(
    repo_root: Path,
    arch: str,
    q_to_bucket: dict[str, str],
) -> float:
    """Bucket-weighted average of CALLS_PER_Q across the test_1273 routing."""
    md = pd.read_parquet(repo_root / "data/processed/medqa_4opt.parquet")
    test_questions = md[md["split"] == "test"]["question"].tolist()
    buckets = pd.Series([q_to_bucket.get(q) for q in test_questions]).dropna()
    bucket_counts = buckets.value_counts().to_dict()
    routing = _ROUTING[arch]
    total = sum(bucket_counts.values())
    return sum(
        bucket_counts.get(b, 0) * CALLS_PER_Q[routing[b]] for b in routing
    ) / total


def _hallucination_rate_from_ragas(ragas_df: pd.DataFrame) -> float:
    """RAGAS Hallucination Rate = fraction of rows with faithfulness < 0.5.

    Mirrors the convention used in the per-architecture output notes
    (e.g. EXP_02: F=0.131 → HR=0.896). NaN faithfulness rows are excluded
    from the denominator.
    """
    if "faithfulness" not in ragas_df.columns:
        return float("nan")
    f = pd.to_numeric(ragas_df["faithfulness"], errors="coerce").dropna()
    if f.empty:
        return float("nan")
    return float((f < 0.5).mean())


def _explainability_per_arch(
    repo_root: Path,
    q_to_bucket: dict[str, str],
) -> dict[str, tuple[float, int]]:
    """Per-architecture mean LIME-SHAP Spearman ρ (correctness signal).

    Reads each architecture's cross-arch agreement JSONL directly where present
    (Phase 6 v2, 2026-05-12): mhop / naive / sparse / hybrid. NoRAG remains 0
    by construction (no chunks → undefined attribution). Adaptive Variants
    use route-weighted blending of the underlying architectures' measured
    Spearman ρ values, using the test_1273 bucket counts.

    Earlier version of this function used Multi-Hop's measured value × the
    Multi-Hop routing share and set Naive/Sparse/Hybrid to NaN. That was
    overturned by the 2026-05-12 cross-arch extension which produced clean
    Spearman ρ values for every single-shot architecture on its own
    retrieval-changed subset.
    """
    out: dict[str, tuple[float, int]] = {a: (float("nan"), 0) for a in ARCHITECTURES}

    # Map architecture → on-disk JSONL filename (single-shot + Multi-Hop only;
    # NoRAG and Adaptive are computed below).
    AGREEMENT_FILES = {
        "Naive":    "stage_b_retrievalchanged_naive.jsonl",
        "Sparse":   "stage_b_retrievalchanged_sparse.jsonl",
        "Hybrid":   "stage_b_retrievalchanged_hybrid.jsonl",
        "MultiHop": "stage_b_retrievalchanged_mhop.jsonl",
    }
    direct_spearman: dict[str, float] = {}
    for arch, fname in AGREEMENT_FILES.items():
        path = repo_root / "results/exp_12_agreement" / fname
        if not path.exists():
            continue
        rows = [json.loads(l.replace("NaN", "null")) for l in path.read_text().splitlines()]
        if not rows:
            continue
        df = pd.DataFrame(rows)
        sp = pd.to_numeric(df.get("correctness_spearman"), errors="coerce").dropna()
        if sp.empty:
            continue
        mean_sp = float(sp.mean())
        direct_spearman[arch] = mean_sp
        out[arch] = (mean_sp, int(len(sp)))

    # Adaptive Variants: route-weighted blend across the underlying architectures
    # actually routed to per bucket. NoRAG's contribution to Variant B is 0
    # (no chunks → no attribution).
    md = pd.read_parquet(repo_root / "data/processed/medqa_4opt.parquet")
    test_qs = md[md["split"] == "test"]["question"].tolist()
    buckets = pd.Series([q_to_bucket.get(q) for q in test_qs]).dropna()
    bucket_counts = buckets.value_counts().to_dict()
    total = sum(bucket_counts.values()) or 1
    SPEARMAN_BY_UNDERLYING = {
        "NoRAG":    0.0,                                # no chunks → 0
        "Naive":    direct_spearman.get("Naive", 0.0),
        "Hybrid":   direct_spearman.get("Hybrid", 0.0),
        "MultiHop": direct_spearman.get("MultiHop", 0.0),
    }
    for v in ("Adaptive_A", "Adaptive_B"):
        routing = _ROUTING[v]
        weighted = sum(
            bucket_counts.get(b, 0) * SPEARMAN_BY_UNDERLYING.get(arch, 0.0)
            for b, arch in routing.items()
        ) / total
        # Carry the total n that contributed (sum of underlying-arch n's
        # weighted by bucket share) for transparency in the raw table.
        n_total = sum(
            bucket_counts.get(b, 0)
            for b in routing  # all 3 buckets contribute
        )
        out[v] = (float(weighted), int(n_total))
    return out


def collect_architecture_metrics(repo_root: Path) -> pd.DataFrame:
    """Return one row per architecture with every metric needed for EXP_16.

    Columns:
        architecture, accuracy_test_1273, mean_latency_s_test_1273,
        groq_calls_per_q, faithfulness_golden_234,
        hallucination_rate_golden_234, context_precision_golden_234,
        context_recall_golden_234, answer_correctness_golden_234,
        answer_relevance_golden_234, lime_shap_spearman,
        n_explainability_questions.

    Missing values are returned as NaN (no silent imputation); the ranker
    and normaliser decide what to do with them.
    """
    repo_root = Path(repo_root)
    q_to_bucket = _load_complexity_bucket_lookup(repo_root)

    # RAGAS tables for the fixed archs (needed both for direct lookup and for
    # the adaptive score-join).
    ragas_per_arch: dict[str, pd.DataFrame] = {}
    for arch in ("NoRAG", "Naive", "Sparse", "Hybrid", "MultiHop"):
        df = _golden_234_ragas_csv(repo_root, arch)
        if df is not None:
            ragas_per_arch[arch] = df.set_index("question_id")

    golden_buckets = _load_golden_with_buckets(repo_root, q_to_bucket)
    explain = _explainability_per_arch(repo_root, q_to_bucket)

    rows: list[dict] = []
    for arch in ARCHITECTURES:
        test_sum = _test_1273_summary(repo_root, arch)
        gold_sum = _golden_234_summary(repo_root, arch)

        accuracy = (test_sum or {}).get("Acuuracy", float("nan"))
        latency = (test_sum or {}).get("mean_latency_s", float("nan"))

        if arch in _ROUTING:
            ragas_means = _score_join_adaptive(
                repo_root, _ROUTING[arch], golden_buckets, ragas_per_arch
            )
            # Hallucination rate via score-join: compute per-question
            # faithfulness via routing, then HR = (F<0.5).mean().
            faithful_per_q: list[float] = []
            for _, g in golden_buckets.iterrows():
                if g["bucket"] not in _ROUTING[arch]:
                    continue
                under = _ROUTING[arch][g["bucket"]]
                if under not in ragas_per_arch:
                    continue
                tbl = ragas_per_arch[under]
                if g["question_id"] not in tbl.index:
                    continue
                v = tbl.loc[g["question_id"]].get("faithfulness")
                if pd.notna(v):
                    faithful_per_q.append(float(v))
            hr = float(np.mean(np.array(faithful_per_q) < 0.5)) if faithful_per_q else float("nan")
            calls = _adaptive_calls_per_q(repo_root, arch, q_to_bucket)
        else:
            ragas_means = {
                "faithfulness": (gold_sum or {}).get("RAGAS_Faithfulness", float("nan")),
                "context_precision": (gold_sum or {}).get("RAGAS_Context_Precision", float("nan")),
                "context_recall": (gold_sum or {}).get("RAGAS_Context_Recall", float("nan")),
                "answer_relevancy": (gold_sum or {}).get("RAGAS_Answer_Relevance", float("nan")),
                "answer_correctness": (gold_sum or {}).get("Answer_Correctness", float("nan")),
            }
            hr_from_summary = (gold_sum or {}).get("RAGAS_Hallucination_Rate")
            if hr_from_summary is None and arch in ragas_per_arch:
                hr = _hallucination_rate_from_ragas(ragas_per_arch[arch].reset_index())
            else:
                hr = float(hr_from_summary) if hr_from_summary is not None else float("nan")
            calls = CALLS_PER_Q.get(arch, float("nan"))

        spearman, n_xai = explain.get(arch, (float("nan"), 0))

        rows.append(
            {
                "architecture": arch,
                "accuracy_test_1273": float(accuracy) if accuracy is not None else float("nan"),
                "mean_latency_s_test_1273": float(latency) if latency is not None else float("nan"),
                "groq_calls_per_q": float(calls),
                "faithfulness_golden_234": _safe_float(ragas_means["faithfulness"]),
                "hallucination_rate_golden_234": float(hr),
                "context_precision_golden_234": _safe_float(ragas_means["context_precision"]),
                "context_recall_golden_234": _safe_float(ragas_means["context_recall"]),
                "answer_correctness_golden_234": _safe_float(ragas_means["answer_correctness"]),
                "answer_relevance_golden_234": _safe_float(ragas_means["answer_relevancy"]),
                "lime_shap_spearman": float(spearman),
                "n_explainability_questions": int(n_xai),
            }
        )

    return pd.DataFrame(rows)


def _safe_float(x) -> float:
    if x is None:
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")
