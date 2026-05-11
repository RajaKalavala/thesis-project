"""Unit tests for `src/taxonomy/`.

No OpenAI calls — exercises the categories schema, prompt builder, label
parser, cross-tab and Cohen's κ on synthetic data. The full
`classify_one` LLM call is exercised in the notebook smoke stage.

    .venv/bin/python tests/test_taxonomy.py
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from src.taxonomy.analysis import (
    cohens_kappa,
    crosstab_category_by_arch,
    headline_table,
    load_labels_jsonl,
)
from src.taxonomy.categories import (
    CATEGORIES,
    CATEGORY_ORDER,
    format_categories_for_prompt,
    is_valid_category,
)
from src.taxonomy.labeller import (
    build_user_message,
    _payload_to_label,
)


# ------------------- categories.py --------------------


def test_category_order_matches_categories_keys() -> None:
    assert set(CATEGORY_ORDER) == set(CATEGORIES.keys())
    assert len(CATEGORY_ORDER) == 6


def test_is_valid_category() -> None:
    assert is_valid_category("unsupported_diagnosis")
    assert is_valid_category("context_omission")
    assert not is_valid_category("Unsupported_Diagnosis")  # case-sensitive
    assert not is_valid_category("nonsense")
    assert not is_valid_category("")


def test_format_categories_for_prompt_lists_all_six() -> None:
    s = format_categories_for_prompt()
    for cat in CATEGORY_ORDER:
        assert cat in s
    # Numbered 1.–6.
    for i in range(1, 7):
        assert f"{i}." in s


# ------------------- labeller.py --------------------


def test_build_user_message_includes_required_fields() -> None:
    msg = build_user_message(
        question="Test question?",
        options={"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
        chunks=[{"text": "chunk one text"}, {"text": "chunk two text"}],
        pred_letter="A",
        pred_text="The answer is A because of reasons.",
        gold_letter="B",
    )
    assert "Test question?" in msg
    assert "(A) alpha" in msg
    assert "(D) delta" in msg
    assert "[1] chunk one text" in msg
    assert "[2] chunk two text" in msg
    assert "MODEL'S PREDICTED LETTER: A" in msg
    assert "GOLD CORRECT LETTER: B" in msg


def test_build_user_message_handles_no_chunks() -> None:
    msg = build_user_message(
        question="Q", options={"A": "x", "B": "y", "C": "z", "D": "w"},
        chunks=[], pred_letter="A", pred_text="response", gold_letter="B",
    )
    assert "(no chunks retrieved" in msg


def test_build_user_message_truncates_long_chunks() -> None:
    long_text = "alpha " * 500  # ~3000 chars
    msg = build_user_message(
        question="Q", options={"A": "x", "B": "y"},
        chunks=[{"text": long_text}],
        pred_letter="A", pred_text="r", gold_letter="B",
        chunk_text_char_limit=100,
    )
    # Truncated to ~100 chars + ellipsis
    assert "..." in msg
    # Chunk shouldn't appear at full length
    assert long_text not in msg


def test_payload_to_label_valid_json() -> None:
    payload = {"text": json.dumps({"category": "wrong_reasoning_chain", "rationale": "chunk [3] supports B"}), "latency_s": 0.5}
    label = _payload_to_label(
        payload, "q1", "arch_x", "B", "A", "gpt-4o-mini", was_cached=False,
    )
    assert label.category == "wrong_reasoning_chain"
    assert label.rationale == "chunk [3] supports B"
    assert label.parse_ok is True
    assert label.gold_letter == "B"
    assert label.pred_letter == "A"


def test_payload_to_label_invalid_category() -> None:
    payload = {"text": json.dumps({"category": "MADE_UP_CATEGORY", "rationale": "whatever"})}
    label = _payload_to_label(payload, "q1", "arch", "B", "A", "gpt-4o-mini", was_cached=False)
    assert label.category is None
    assert label.parse_ok is False
    assert "INVALID_CATEGORY" in label.rationale


def test_payload_to_label_malformed_json() -> None:
    payload = {"text": "not json {at all"}
    label = _payload_to_label(payload, "q1", "arch", "B", "A", "gpt-4o-mini", was_cached=False)
    assert label.category is None
    assert label.parse_ok is False
    assert "PARSE_FAILURE" in label.rationale


# ------------------- analysis.py --------------------


def _label_df(rows: list[tuple[str, str, str | None]]) -> pd.DataFrame:
    """Build a small label dataframe from (question_id, architecture, category) tuples."""
    return pd.DataFrame([
        {
            "question_id": qid, "architecture": arch, "category": cat,
            "rationale": "stub", "gold_letter": "B", "pred_letter": "A",
            "raw_response": "{}", "was_cached": False, "latency_s": 0.1,
            "model": "gpt-4o-mini", "parse_ok": cat is not None,
        }
        for qid, arch, cat in rows
    ])


def test_crosstab_basic_counts() -> None:
    df = _label_df([
        ("q1", "Naive", "unsupported_diagnosis"),
        ("q2", "Naive", "unsupported_diagnosis"),
        ("q3", "Naive", "context_omission"),
        ("q4", "Multi-Hop", "wrong_reasoning_chain"),
        ("q5", "Multi-Hop", "wrong_reasoning_chain"),
    ])
    tab = crosstab_category_by_arch(df)
    # 6 rows in canonical order, 2 cols (Multi-Hop, Naive sorted alphabetically)
    assert list(tab.index) == list(CATEGORY_ORDER)
    assert sorted(tab.columns) == ["Multi-Hop", "Naive"]
    assert tab.loc["unsupported_diagnosis", "Naive"] == 2
    assert tab.loc["context_omission", "Naive"] == 1
    assert tab.loc["wrong_reasoning_chain", "Multi-Hop"] == 2
    assert tab.loc["unsupported_treatment", "Multi-Hop"] == 0


def test_crosstab_normalize_columns_sums_to_one() -> None:
    df = _label_df([
        ("q1", "Naive", "unsupported_diagnosis"),
        ("q2", "Naive", "context_omission"),
        ("q3", "Naive", "context_omission"),
    ])
    tab = crosstab_category_by_arch(df, normalize="columns")
    assert math.isclose(tab["Naive"].sum(), 1.0, abs_tol=1e-6)
    assert math.isclose(tab.loc["context_omission", "Naive"], 2 / 3, abs_tol=1e-4)


def test_crosstab_drops_none_category() -> None:
    df = _label_df([
        ("q1", "Naive", None),  # parse failure
        ("q2", "Naive", "context_omission"),
    ])
    tab = crosstab_category_by_arch(df)
    assert tab["Naive"].sum() == 1  # only one valid


def test_cohens_kappa_perfect_agreement() -> None:
    a = pd.Series(["x", "y", "x", "y", "z"])
    b = pd.Series(["x", "y", "x", "y", "z"])
    assert math.isclose(cohens_kappa(a, b), 1.0, abs_tol=1e-9)


def test_cohens_kappa_no_agreement() -> None:
    a = pd.Series(["x", "x", "x", "y", "y"])
    b = pd.Series(["y", "y", "y", "x", "x"])
    k = cohens_kappa(a, b)
    assert k < 0  # systematic disagreement → negative kappa


def test_cohens_kappa_drops_nan() -> None:
    a = pd.Series(["x", "y", None, "z"])
    b = pd.Series(["x", "y", "x", "z"])
    k = cohens_kappa(a, b)
    # Compute on n=3 paired rows; all agree → κ=1
    assert math.isclose(k, 1.0, abs_tol=1e-9)


def test_headline_table_per_arch() -> None:
    df = _label_df([
        ("q1", "Naive", "unsupported_diagnosis"),
        ("q2", "Naive", "unsupported_diagnosis"),
        ("q3", "Naive", "context_omission"),
        ("q4", "Multi-Hop", "wrong_reasoning_chain"),
    ])
    out = headline_table(df)
    assert len(out) == 2
    naive_row = out[out.architecture == "Naive"].iloc[0]
    assert naive_row["n_total_labelled"] == 3
    assert naive_row["top_category"] == "unsupported_diagnosis"
    assert naive_row["top_category_n"] == 2


def test_load_labels_round_trip() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "labels.jsonl"
        with p.open("w") as f:
            f.write(json.dumps({
                "question_id": "q1", "architecture": "a", "category": "context_omission",
                "rationale": "r", "gold_letter": "B", "pred_letter": "A",
                "raw_response": "{}", "was_cached": False, "latency_s": 0.1,
                "model": "gpt-4o-mini", "parse_ok": True,
            }) + "\n")
        df = load_labels_jsonl(p)
        assert len(df) == 1
        assert df.iloc[0]["category"] == "context_omission"


if __name__ == "__main__":
    tests = [
        test_category_order_matches_categories_keys,
        test_is_valid_category,
        test_format_categories_for_prompt_lists_all_six,
        test_build_user_message_includes_required_fields,
        test_build_user_message_handles_no_chunks,
        test_build_user_message_truncates_long_chunks,
        test_payload_to_label_valid_json,
        test_payload_to_label_invalid_category,
        test_payload_to_label_malformed_json,
        test_crosstab_basic_counts,
        test_crosstab_normalize_columns_sums_to_one,
        test_crosstab_drops_none_category,
        test_cohens_kappa_perfect_agreement,
        test_cohens_kappa_no_agreement,
        test_cohens_kappa_drops_nan,
        test_headline_table_per_arch,
        test_load_labels_round_trip,
    ]
    for t in tests:
        t()
        print(f"  ✓ {t.__name__}")
    print(f"\nAll {len(tests)} taxonomy tests passed.")
