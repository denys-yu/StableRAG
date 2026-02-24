"""SQuALITY dataset download and conversion utilities."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterator
from urllib.request import urlretrieve


SQUALITY_BASE_URL = "https://raw.githubusercontent.com/nyu-mll/SQuALITY/main/data/{version}/txt/{split}.jsonl"
SPLITS = ("train", "dev", "test")


def download_squality(data_dir: Path, version: str = "v1-3") -> dict[str, Path]:
    """Download SQuALITY split files to data/squality/<version>/txt if missing."""

    target_dir = data_dir / "squality" / version / "txt"
    target_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}

    for split in SPLITS:
        target = target_dir / f"{split}.jsonl"
        out[split] = target
        if target.exists() and target.stat().st_size > 0:
            print(f"[squality] Using cached {split}: {target}")
            continue
        url = SQUALITY_BASE_URL.format(version=version, split=split)
        print(f"[squality] Downloading {split} from {url}")
        urlretrieve(url, target)
        print(f"[squality] Saved {split} -> {target}")

    return out


def iter_squality_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield SQuALITY records from jsonl or concatenated JSON objects."""

    line_rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                parsed = json.loads(line)
                if not isinstance(parsed, dict):
                    raise ValueError("Expected JSON object rows.")
                line_rows.append(parsed)
        for row in line_rows:
            yield row
        return
    except (json.JSONDecodeError, ValueError):
        pass

    decoder = json.JSONDecoder()
    blob = path.read_text(encoding="utf-8")
    idx = 0
    while idx < len(blob):
        while idx < len(blob) and blob[idx].isspace():
            idx += 1
        if idx >= len(blob):
            break
        obj, end_idx = decoder.raw_decode(blob, idx)
        if not isinstance(obj, dict):
            raise ValueError(f"Expected JSON object while parsing {path}.")
        yield obj
        idx = end_idx


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_squality(
    split: str = "dev",
    max_stories: int = 8,
    include_plot_question: bool = False,
    questions_per_story: int | None = None,
    seed: int = 42,
    out_corpus_path: Path = Path("data/corpus.jsonl"),
    out_questions_path: Path = Path("data/questions.jsonl"),
    out_meta_path: Path | None = Path("data/squality_meta.json"),
    version: str = "v1-3",
) -> tuple[int, int]:
    """Convert SQuALITY split into project corpus/questions files."""

    data_dir = out_corpus_path.parent
    split_path = data_dir / "squality" / version / "txt" / f"{split}.jsonl"
    if not split_path.exists():
        raise FileNotFoundError(f"SQuALITY split file not found: {split_path}")

    stories = list(iter_squality_records(split_path))
    rng = random.Random(seed)
    rng.shuffle(stories)
    selected = stories[:max_stories]

    corpus_rows: list[dict[str, Any]] = []
    question_rows: list[dict[str, Any]] = []

    for story in selected:
        metadata = story.get("metadata", {}) or {}
        doc_id = str(metadata.get("uid") or metadata.get("passage_id") or "")
        if not doc_id:
            continue
        document_text = str(story.get("document", "")).strip()
        if not document_text:
            continue

        corpus_rows.append(
            {
                "doc_id": doc_id,
                "text": document_text,
                "source": "squality",
                "split": split,
                "passage_id": metadata.get("passage_id"),
                "license": metadata.get("license"),
            }
        )

        raw_questions = list(story.get("questions", []))
        question_candidates: list[dict[str, Any]] = []
        for question in raw_questions:
            question_number = int(question.get("question_number", 0))
            if not include_plot_question and question_number == 1:
                continue
            question_candidates.append(question)
        question_candidates.sort(key=lambda item: int(item.get("question_number", 0)))

        if questions_per_story is not None:
            question_candidates = question_candidates[:questions_per_story]

        for question in question_candidates:
            question_number = int(question.get("question_number", 0))
            question_text = str(question.get("question_text", "")).strip()
            if not question_text:
                continue
            references = [
                str(response.get("response_text", "")).strip()
                for response in list(question.get("responses", []))
                if str(response.get("response_text", "")).strip()
            ]
            question_rows.append(
                {
                    "qid": f"{doc_id}_q{question_number}",
                    "doc_id": doc_id,
                    "question": question_text,
                    "references": references,
                    "dataset": "squality",
                    "split": split,
                }
            )

    _write_jsonl(out_corpus_path, corpus_rows)
    _write_jsonl(out_questions_path, question_rows)

    if out_meta_path is not None:
        meta_payload = {
            "dataset": "squality",
            "version": version,
            "split": split,
            "seed": seed,
            "max_stories": max_stories,
            "include_plot_question": include_plot_question,
            "questions_per_story": questions_per_story,
            "num_stories_selected": len(corpus_rows),
            "num_questions_selected": len(question_rows),
            "doc_ids": [row["doc_id"] for row in corpus_rows],
        }
        out_meta_path.parent.mkdir(parents=True, exist_ok=True)
        out_meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return len(corpus_rows), len(question_rows)
