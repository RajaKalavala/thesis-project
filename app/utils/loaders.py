"""Cached data loaders for the Streamlit demo.

Every function is wrapped with ``@st.cache_data`` so the artefact is parsed once
per Streamlit session and reused across reruns. Anchored to the repo root via
``REPO_ROOT`` so the app works from any CWD (Streamlit Cloud will set CWD to
the repo root, locally the user might run from anywhere).

Data surfaces
-------------
- **golden_234**  → 5 fixed architectures (NoRAG / Naive / Sparse / Hybrid /
  MultiHop) have ``predictions.jsonl`` + ``retrieval.jsonl`` + ``ragas_scores.csv``.
  Adaptive variants did NOT run on this surface.
- **test_1273**   → all 7 architectures have predictions + retrieval. No RAGAS.
- **stage_b_retrievalchanged**  → LIME / SHAP / agreement for Naive, Sparse,
  Hybrid, Multi-Hop. Keyed by ``medqa_NNNNN``, not ``golden_NNN``.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
DATA_DIR = REPO_ROOT / "data" / "processed"

# Display name → results-dir basename (golden_234 surface)
ARCH_TO_EXP_GOLDEN: dict[str, str] = {
    "NoRAG": "exp_01_base_llm",
    "Naive": "exp_02_naive_rag",
    "Sparse": "exp_03_sparse_rag",
    "Hybrid": "exp_04_hybrid_rag",
    "MultiHop": "exp_05_multi_hop_rag",
}

# Display name → results-dir basename (test_1273 surface; includes Adaptive)
ARCH_TO_EXP_TEST: dict[str, str] = {
    **ARCH_TO_EXP_GOLDEN,
    "Adaptive_A": "exp_07_adaptive_variant_a",
    "Adaptive_B": "exp_07_adaptive_variant_b",
}

# LIME/SHAP/agreement use short arch keys on the stage_b surface
ARCH_TO_XAI_KEY: dict[str, str] = {
    "Naive": "naive",
    "Sparse": "sparse",
    "Hybrid": "hybrid",
    "MultiHop": "mhop",
}


# ---------------------------------------------------------------------------
# JSONL helper
# ---------------------------------------------------------------------------
def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Golden dataset (question text, options, gold answer, metadata)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_golden() -> pd.DataFrame:
    """Load the 234-row golden RAGAS dataset.

    ``question_id`` is normalised to the string form ``golden_NNN`` (3-digit
    zero-padded) so it joins cleanly with predictions/retrieval files.
    """
    rows = _read_jsonl(DATA_DIR / "golden_ragas_300.jsonl")
    df = pd.DataFrame(rows)
    df["question_id"] = df["question_id"].apply(lambda i: f"golden_{int(i):03d}")
    return df


# ---------------------------------------------------------------------------
# Chunks (text + book) — used to render retrieved passages
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_chunks() -> pd.DataFrame:
    """Load all 67,599 corpus chunks indexed by chunk_id."""
    df = pd.read_parquet(DATA_DIR / "chunks.parquet")
    return df.set_index("chunk_id", drop=False)


@lru_cache(maxsize=1)
def _chunk_lookup() -> dict[str, dict[str, Any]]:
    """Fast O(1) chunk text lookup. Built lazily, kept across reruns via lru_cache."""
    df = load_chunks()
    return df.to_dict(orient="index")


def get_chunk(chunk_id: str) -> dict[str, Any] | None:
    return _chunk_lookup().get(chunk_id)


# ---------------------------------------------------------------------------
# Per-architecture predictions / retrieval / RAGAS
# ---------------------------------------------------------------------------
def _exp_dir(arch: str, surface: str) -> Path:
    """Resolve the results dir for (arch, surface) where surface ∈ {'golden_234', 'test_1273'}."""
    if surface == "golden_234":
        base = ARCH_TO_EXP_GOLDEN[arch]
    elif surface == "test_1273":
        base = ARCH_TO_EXP_TEST[arch]
    else:
        raise ValueError(f"unknown surface: {surface}")
    return RESULTS_DIR / f"{base}__{surface}"


@st.cache_data(show_spinner=False)
def load_predictions(arch: str, surface: str = "golden_234") -> pd.DataFrame:
    return pd.DataFrame(_read_jsonl(_exp_dir(arch, surface) / "predictions.jsonl"))


@st.cache_data(show_spinner=False)
def load_retrieval(arch: str, surface: str = "golden_234") -> pd.DataFrame:
    path = _exp_dir(arch, surface) / "retrieval.jsonl"
    if not path.exists():
        return pd.DataFrame(columns=["question_id", "retrieved_chunk_ids", "retrieved_chunk_scores"])
    return pd.DataFrame(_read_jsonl(path))


@st.cache_data(show_spinner=False)
def load_ragas(arch: str) -> pd.DataFrame:
    """RAGAS scores on golden_234 (the only surface RAGAS ran on)."""
    path = _exp_dir(arch, "golden_234") / "ragas_scores.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_summary(arch: str, surface: str = "golden_234") -> dict[str, Any]:
    path = _exp_dir(arch, surface) / "summary.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Confidence signals + threshold sweeps (Multi-Hop golden_234 only)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_confidence_signals() -> pd.DataFrame:
    """Per-question confidence-signal feature table (n=234)."""
    p = RESULTS_DIR / "exp_08_confidence_signals" / "exp_05_multi_hop_rag__golden_234__signals.parquet"
    return pd.read_parquet(p)


@st.cache_data(show_spinner=False)
def load_threshold_sweeps() -> pd.DataFrame:
    """Threshold sweeps for the four confidence configurations on Multi-Hop."""
    p = RESULTS_DIR / "exp_09_confidence_rejection" / "exp_05_multi_hop_rag__golden_234__threshold_sweeps.csv"
    return pd.read_csv(p)


# ---------------------------------------------------------------------------
# LIME / SHAP — stage_b retrieval-changed surface (medqa_NNNNN keyed)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_lime(arch: str) -> pd.DataFrame:
    key = ARCH_TO_XAI_KEY.get(arch)
    if key is None:
        return pd.DataFrame()
    p = RESULTS_DIR / "exp_10_lime_passage" / f"stage_b_retrievalchanged_{key}.jsonl"
    if not p.exists():
        return pd.DataFrame()
    return pd.DataFrame(_read_jsonl(p))


@st.cache_data(show_spinner=False)
def load_shap(arch: str) -> pd.DataFrame:
    key = ARCH_TO_XAI_KEY.get(arch)
    if key is None:
        return pd.DataFrame()
    p = RESULTS_DIR / "exp_11_shap_passage" / f"stage_b_retrievalchanged_{key}.jsonl"
    if not p.exists():
        return pd.DataFrame()
    return pd.DataFrame(_read_jsonl(p))


@st.cache_data(show_spinner=False)
def load_xai_aggregates() -> dict[str, Any]:
    p = RESULTS_DIR / "exp_12_agreement" / "cross_arch_aggregates.json"
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Taxonomy labels
# ---------------------------------------------------------------------------
TAXONOMY_FILES = {
    "NoRAG": "stage_c_norag_wrong.jsonl",
    "Naive": "stage_c_naive_wrong.jsonl",
    "Sparse": "stage_c_sparse_wrong.jsonl",
    "Hybrid": "stage_c_hybrid_wrong.jsonl",
    "MultiHop": "stage_c_multihop_wrong.jsonl",
}


@st.cache_data(show_spinner=False)
def load_taxonomy(arch: str) -> pd.DataFrame:
    fname = TAXONOMY_FILES.get(arch)
    if fname is None:
        return pd.DataFrame()
    p = RESULTS_DIR / "exp_14_taxonomy_labels" / fname
    if not p.exists():
        return pd.DataFrame()
    return pd.DataFrame(_read_jsonl(p))


@st.cache_data(show_spinner=False)
def load_taxonomy_proportions() -> pd.DataFrame:
    p = RESULTS_DIR / "exp_15_taxonomy_analysis" / "table7_proportions.csv"
    return pd.read_csv(p)


# ---------------------------------------------------------------------------
# Phase 9 synthesis artefacts
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_synthesis() -> dict[str, pd.DataFrame | dict[str, Any]]:
    """All Phase-9 synthesis artefacts (table 12 ranking, raw + normalised
    component scores, sensitivity ranks, Pareto status, summary.json)."""
    base = RESULTS_DIR / "exp_16_final_synthesis"
    out: dict[str, pd.DataFrame | dict[str, Any]] = {
        "ranking": pd.read_csv(base / "table12_final_ranking.csv"),
        "components_raw": pd.read_csv(base / "component_scores_raw.csv"),
        "components_normalised": pd.read_csv(base / "component_scores_normalised.csv"),
        "sensitivity": pd.read_csv(base / "sensitivity_ranks.csv"),
        "pareto": pd.read_csv(base / "pareto_status.csv"),
        "recommendations": pd.read_csv(base / "recommendations.csv"),
    }
    with open(base / "summary.json") as f:
        out["summary"] = json.load(f)
    return out


# ---------------------------------------------------------------------------
# Question Inspector helper — joins everything for ONE golden question
# ---------------------------------------------------------------------------
def get_inspector_view(question_id: str) -> dict[str, Any]:
    """Return everything Page 1 needs for a single golden question across all 5 archs."""
    golden = load_golden()
    qrow = golden[golden["question_id"] == question_id]
    if qrow.empty:
        return {}
    qrow = qrow.iloc[0].to_dict()

    archs_view: list[dict[str, Any]] = []
    for arch in ARCH_TO_EXP_GOLDEN:
        preds = load_predictions(arch, "golden_234")
        retr = load_retrieval(arch, "golden_234")
        ragas = load_ragas(arch)

        pred_row = preds[preds["question_id"] == question_id]
        if pred_row.empty:
            continue
        pred_row = pred_row.iloc[0].to_dict()

        retr_row = retr[retr["question_id"] == question_id]
        retr_row = retr_row.iloc[0].to_dict() if not retr_row.empty else None

        ragas_row = None
        if not ragas.empty:
            r = ragas[ragas["question_id"] == question_id]
            if not r.empty:
                ragas_row = r.iloc[0].to_dict()

        archs_view.append({
            "arch": arch,
            "pred": pred_row,
            "retrieval": retr_row,
            "ragas": ragas_row,
        })

    return {"question": qrow, "archs": archs_view}
