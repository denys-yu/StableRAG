"""Metric computation for repeatability experiments."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd


def _first_difference_position(a: str, b: str) -> int:
    """Return 0 when identical; otherwise return first differing char position (1-based)."""

    if a == b:
        return 0
    for idx, (char_a, char_b) in enumerate(zip(a, b), start=1):
        if char_a != char_b:
            return idx
    return min(len(a), len(b)) + 1


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) + 1e-12
    return float(np.dot(vec_a, vec_b) / denom)


def compute_per_question_metrics(
    *,
    records: pd.DataFrame,
    embed_texts_fn,
    embedding_model: str,
) -> pd.DataFrame:
    grouped_outputs: dict[tuple[str, str], list[str]] = {}
    for (variant, qid), frame in records.groupby(["variant", "qid"], sort=False):
        ordered = frame.sort_values("run_index")
        grouped_outputs[(variant, qid)] = [str(text) for text in ordered["output_text"].tolist()]

    unique_outputs = sorted({text for outputs in grouped_outputs.values() for text in outputs})
    vectors = embed_texts_fn(texts=unique_outputs, model=embedding_model) if unique_outputs else []
    text_to_vec = {text: np.array(vec, dtype=np.float64) for text, vec in zip(unique_outputs, vectors)}

    per_question_rows: list[dict[str, float | int | str]] = []
    for (variant, qid), outputs in grouped_outputs.items():
        repeats = len(outputs)
        baseline = outputs[0] if outputs else ""
        emr = int(all(candidate == baseline for candidate in outputs[1:]))

        semantic_scores: list[float] = []
        divergence_scores: list[int] = []
        if repeats <= 1:
            semantic_scores = [1.0]
            divergence_scores = [0]
        else:
            base_vec = text_to_vec[baseline]
            for candidate in outputs[1:]:
                semantic_scores.append(_cosine_similarity(base_vec, text_to_vec[candidate]))
                divergence_scores.append(_first_difference_position(baseline, candidate))

        per_question_rows.append(
            {
                "variant": variant,
                "qid": qid,
                "repeats": repeats,
                "emr": emr,
                "semantic_similarity": float(np.mean(semantic_scores)),
                "divergence": float(np.mean(divergence_scores)),
            }
        )

    return pd.DataFrame(per_question_rows)


def compute_summary_metrics(
    *,
    records: pd.DataFrame,
    per_question: pd.DataFrame,
) -> pd.DataFrame:
    summary_rows: list[dict[str, float | int | str]] = []
    question_counts = defaultdict(int)
    for variant, frame in per_question.groupby("variant"):
        question_counts[variant] = frame["qid"].nunique()

    for variant, frame in records.groupby("variant", sort=False):
        pq = per_question[per_question["variant"] == variant]
        summary_rows.append(
            {
                "variant": variant,
                "num_questions": question_counts[variant],
                "repeats": int(frame["run_index"].max() + 1 if not frame.empty else 0),
                "EMR": float(pq["emr"].mean() if not pq.empty else 0.0),
                "semantic_sim_mean": float(pq["semantic_similarity"].mean() if not pq.empty else 0.0),
                "divergence_mean": float(pq["divergence"].mean() if not pq.empty else 0.0),
                "latency_mean_ms": float(frame["latency_ms"].mean() if not frame.empty else 0.0),
                "latency_p95_ms": float(frame["latency_ms"].quantile(0.95) if not frame.empty else 0.0),
                "tokens_mean": float(frame["total_tokens"].mean() if not frame.empty else 0.0),
            }
        )

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        return summary
    return summary[
        [
            "variant",
            "num_questions",
            "repeats",
            "EMR",
            "semantic_sim_mean",
            "divergence_mean",
            "latency_mean_ms",
            "latency_p95_ms",
            "tokens_mean",
        ]
    ]


def format_numeric_columns(df: pd.DataFrame, *, columns: Iterable[str], digits: int = 4) -> pd.DataFrame:
    formatted = df.copy()
    for column in columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(lambda value: f"{float(value):.{digits}f}")
    return formatted
