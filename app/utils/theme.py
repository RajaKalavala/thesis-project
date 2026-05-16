"""Visual primitives shared across pages: palette, badges, traffic lights."""
from __future__ import annotations

PALETTE = {
    "NoRAG": "#94a3b8",
    "Naive": "#60a5fa",
    "Sparse": "#a78bfa",
    "Hybrid": "#34d399",
    "MultiHop": "#f59e0b",
    "Adaptive_A": "#ec4899",
    "Adaptive_B": "#f43f5e",
}

WEIGHT_LABELS = {
    "Accuracy": "Accuracy",
    "Faithfulness": "Faithfulness",
    "Retrieval": "Retrieval",
    "Safety": "Safety",
    "Explainability": "Explainability",
    "Latency": "Latency",
}

# Plan §11 locked weights — reset target for Page 2
LOCKED_WEIGHTS = {
    "Accuracy": 0.25,
    "Faithfulness": 0.25,
    "Retrieval": 0.20,
    "Safety": 0.15,
    "Explainability": 0.10,
    "Latency": 0.05,
}


def faith_colour(score: float | None) -> str:
    """Traffic-light colour for a RAGAS Faithfulness score in [0, 1]."""
    if score is None or score != score:  # NaN check
        return "#cbd5e1"  # neutral grey
    if score < 0.30:
        return "#ef4444"  # red
    if score < 0.70:
        return "#f59e0b"  # amber
    return "#10b981"     # green


def faith_label(score: float | None) -> str:
    if score is None or score != score:
        return "N/A"
    if score < 0.30:
        return "Ungrounded"
    if score < 0.70:
        return "Partial"
    return "Grounded"


def correctness_badge(is_correct: bool) -> str:
    return "✓" if is_correct else "✗"


def correctness_colour(is_correct: bool) -> str:
    return "#10b981" if is_correct else "#ef4444"


def memorisation_flag(is_correct: bool, faithfulness: float | None) -> bool:
    """True when the prediction is correct but RAGAS Faithfulness < 0.5.

    This is the central thesis claim made concrete: a correct answer that the
    retrieved chunks do NOT support means the LLM answered from pre-training
    memory, not from the corpus.
    """
    if not is_correct:
        return False
    if faithfulness is None or faithfulness != faithfulness:
        return False
    return faithfulness < 0.5


def fmt_metric(value: float | None, decimals: int = 3) -> str:
    if value is None or value != value:
        return "—"
    return f"{value:.{decimals}f}"
