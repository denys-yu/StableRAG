"""Load run logs and write paper-ready summary outputs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

import pandas as pd

from rag_repeat.config import Settings
from rag_repeat.metrics import compute_per_question_metrics, compute_summary_metrics, format_numeric_columns
from rag_repeat.openai_client import OpenAIClientWrapper


RUN_FILE_RE = re.compile(r"^(?P<tag>\d{8}_\d{6})__(?P<variant>.+)\.jsonl$")


def discover_run_files(settings: Settings, runs_dir_arg: str | None) -> list[Path]:
    if runs_dir_arg:
        path_arg = Path(runs_dir_arg)
        if path_arg.is_file():
            return [path_arg]
        if path_arg.is_dir():
            files = sorted(path_arg.glob("*.jsonl"))
            if files:
                return files

        base_dir = path_arg.parent if path_arg.parent != Path(".") else settings.runs_dir
        prefix = path_arg.name
        if not base_dir.exists():
            base_dir = settings.runs_dir
        files = sorted(base_dir.glob(f"{prefix}__*.jsonl"))
        if files:
            return files
        raise FileNotFoundError(f"No run files found for selector: {runs_dir_arg}")

    all_files = sorted(settings.runs_dir.glob("*.jsonl"))
    tagged = []
    for file in all_files:
        match = RUN_FILE_RE.match(file.name)
        if match:
            tagged.append((match.group("tag"), file))
    if not tagged:
        raise FileNotFoundError(f"No run files found in {settings.runs_dir}")
    latest_tag = max(tag for tag, _ in tagged)
    return [file for tag, file in tagged if tag == latest_tag]


def load_run_records(run_files: Sequence[Path]) -> pd.DataFrame:
    rows = []
    for file in run_files:
        with file.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    if not rows:
        raise ValueError("No rows loaded from run files.")
    return pd.DataFrame(rows)


def _df_to_markdown(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    lines = [header, separator]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in df.columns) + " |")
    return "\n".join(lines) + "\n"


def summarize_runs(
    *,
    settings: Settings,
    client: OpenAIClientWrapper,
    runs_dir_arg: str | None = None,
) -> dict[str, Path]:
    run_files = discover_run_files(settings, runs_dir_arg)
    records = load_run_records(run_files)
    per_question = compute_per_question_metrics(
        records=records,
        embed_texts_fn=client.embed_texts,
        embedding_model=settings.embedding_model,
    )
    summary = compute_summary_metrics(records=records, per_question=per_question)

    settings.results_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = settings.results_dir / "summary.csv"
    summary_md_path = settings.results_dir / "summary.md"
    per_question_path = settings.results_dir / "per_question.csv"

    summary.to_csv(summary_csv_path, index=False)
    per_question.to_csv(per_question_path, index=False)

    markdown_frame = format_numeric_columns(
        summary,
        columns=[
            "EMR",
            "semantic_sim_mean",
            "divergence_mean",
            "latency_mean_ms",
            "latency_p95_ms",
            "tokens_mean",
        ],
        digits=4,
    )
    summary_md_path.write_text(_df_to_markdown(markdown_frame), encoding="utf-8")

    return {
        "summary_csv": summary_csv_path,
        "summary_md": summary_md_path,
        "per_question_csv": per_question_path,
    }
