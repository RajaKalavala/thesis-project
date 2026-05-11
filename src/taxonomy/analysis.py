"""Aggregate analysis of taxonomy labels (EXP_15).

Two functions:
- `crosstab_category_by_arch` — produces the 6-row × N-arch table that
  populates Excel Table 7. Counts + within-architecture proportions.
- `cohens_kappa` — inter-rater agreement between two label series (e.g.
  manual rater vs gpt-4o-mini classifier on a calibration set). Returns a
  scalar κ ∈ [-1, 1].

Both functions are NaN-safe: labels of `None` (parse failures) are
excluded from the contingency table and from κ.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.taxonomy.categories import CATEGORY_ORDER


def load_labels_jsonl(path: Path) -> pd.DataFrame:
    """Read a `run_taxonomy_batch` output and return a tidy dataframe."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    return pd.DataFrame(rows)


def crosstab_category_by_arch(
    df: pd.DataFrame, *, normalize: str | None = None
) -> pd.DataFrame:
    """Pivot `df['category']` × `df['architecture']` into the 6-row table.

    Args:
        df: A taxonomy-labels dataframe (one row per question-arch label).
            Rows where `category` is None are dropped from the contingency.
        normalize: None (raw counts), 'columns' (within-architecture
            proportions — Table 7's canonical form), or 'index' (within-
            category proportions across architectures).

    Returns:
        A dataframe with `CATEGORY_ORDER` as rows and architectures (sorted)
        as columns. Missing (category, architecture) cells are 0.
    """
    valid = df.dropna(subset=["category"]).copy()
    valid["category"] = pd.Categorical(
        valid["category"], categories=list(CATEGORY_ORDER), ordered=True
    )
    archs = sorted(valid["architecture"].unique())
    tab = pd.crosstab(
        valid["category"], valid["architecture"], dropna=False
    ).reindex(index=list(CATEGORY_ORDER), columns=archs, fill_value=0)
    if normalize == "columns":
        col_sums = tab.sum(axis=0).replace(0, np.nan)
        return tab.divide(col_sums, axis=1).fillna(0.0).round(4)
    if normalize == "index":
        row_sums = tab.sum(axis=1).replace(0, np.nan)
        return tab.divide(row_sums, axis=0).fillna(0.0).round(4)
    return tab


def cohens_kappa(a: pd.Series, b: pd.Series) -> float:
    """Cohen's κ between two paired label series.

    Series are aligned by index; rows where either side is NaN/None are
    dropped. Categories are the union of the two series' labels.
    Returns κ ∈ [-1, 1]. NaN if either side has zero variance after drop.
    """
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    if df.empty:
        return float("nan")
    cats = sorted(set(df["a"].unique()) | set(df["b"].unique()))
    if len(cats) < 2:
        return float("nan")
    # Build contingency matrix
    cat_to_idx = {c: i for i, c in enumerate(cats)}
    n = len(cats)
    O = np.zeros((n, n), dtype=float)
    for av, bv in zip(df["a"], df["b"]):
        O[cat_to_idx[av], cat_to_idx[bv]] += 1
    total = O.sum()
    if total == 0:
        return float("nan")
    # Observed agreement
    p_o = np.trace(O) / total
    # Expected agreement (chance)
    row_marg = O.sum(axis=1) / total
    col_marg = O.sum(axis=0) / total
    p_e = float(np.dot(row_marg, col_marg))
    if abs(1 - p_e) < 1e-12:
        return float("nan")
    return float((p_o - p_e) / (1 - p_e))


def headline_table(df: pd.DataFrame, *, baseline_acc_per_arch: dict[str, float] | None = None) -> pd.DataFrame:
    """Per-architecture summary: n_wrong_labelled, n_parse_failures, top_category.

    Useful as a one-line-per-arch summary alongside the cross-tab.
    """
    out = []
    for arch in sorted(df["architecture"].unique()):
        sub = df[df.architecture == arch]
        valid = sub.dropna(subset=["category"])
        cnt = Counter(valid["category"])
        top = cnt.most_common(1)
        row = {
            "architecture": arch,
            "n_total_labelled": len(sub),
            "n_parse_failures": int((~sub["parse_ok"]).sum()) if "parse_ok" in sub.columns else 0,
            "top_category": top[0][0] if top else None,
            "top_category_n": top[0][1] if top else 0,
        }
        if baseline_acc_per_arch and arch in baseline_acc_per_arch:
            row["baseline_acc"] = baseline_acc_per_arch[arch]
        out.append(row)
    return pd.DataFrame(out)
