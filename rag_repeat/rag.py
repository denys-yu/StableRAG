"""Prompting, chunking, and deterministic output rendering utilities."""

from __future__ import annotations

import re
from typing import Any, Iterable

from pydantic import BaseModel, Field


SYSTEM_PROMPT = (
    "You are a QA assistant. Answer using ONLY the provided CONTEXT. "
    "If the answer is not in CONTEXT, output NOT_FOUND."
)


class StructuredAnswer(BaseModel):
    """Structured schema used in ablation variants."""

    final_answer: str = Field(description="Concise final answer to the question.")
    evidence: list[int] = Field(default_factory=list, description="1-based context line numbers that support the answer.")
    not_found: bool = Field(description="True only when the answer is not present in context.")


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace for deterministic canonical rendering."""

    return re.sub(r"\s+", " ", text).strip()


def split_into_chunks(text: str, chunk_size_chars: int, overlap_chars: int) -> list[str]:
    """Deterministically split a text into overlapping character chunks."""

    if chunk_size_chars <= 0:
        raise ValueError("chunk_size_chars must be positive.")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be non-negative.")
    if overlap_chars >= chunk_size_chars:
        raise ValueError("overlap_chars must be smaller than chunk_size_chars.")

    normalized = normalize_whitespace(text)
    if not normalized:
        return []
    if len(normalized) <= chunk_size_chars:
        return [normalized]

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + chunk_size_chars)
        if end < len(normalized):
            lower_bound = start + max(1, int(chunk_size_chars * 0.6))
            split_pos = normalized.rfind(" ", lower_bound, end)
            if split_pos > start:
                end = split_pos
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        start = max(0, end - overlap_chars)
    return chunks


def build_user_prompt(question: str, chunks: Iterable[dict[str, Any]]) -> str:
    """Build the fixed user prompt from a question and retrieved chunks."""

    context_lines = []
    for idx, chunk in enumerate(chunks, start=1):
        context_lines.append(f"[{idx}] {normalize_whitespace(str(chunk['text']))}")
    context_text = "\n".join(context_lines)
    return f"QUESTION: {question}\n\nCONTEXT:\n{context_text}\n\nReturn the answer."


def render_structured_answer(answer: StructuredAnswer, canonical: bool) -> str:
    """Render structured outputs; canonical mode is used in `structured_render_t0`."""

    final_answer = answer.final_answer
    evidence = list(answer.evidence)
    not_found = answer.not_found

    if canonical:
        final_answer = normalize_whitespace(final_answer)
        evidence = sorted(int(item) for item in evidence)

    evidence_blob = "[" + ",".join(str(item) for item in evidence) + "]"
    return (
        f"Answer: {final_answer}\n"
        f"Evidence: {evidence_blob}\n"
        f"NotFound: {'true' if not_found else 'false'}"
    )
