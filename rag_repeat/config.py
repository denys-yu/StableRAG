"""Runtime configuration for the RAG repeatability harness."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _int_from_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _optional_int_from_env(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return int(value)


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    chroma_dir: Path
    runs_dir: Path
    results_dir: Path
    corpus_path: Path
    questions_path: Path
    retrieval_frozen_path: Path
    replay_cache_path: Path
    collection_name: str
    chunk_size_chars: int
    chunk_overlap_chars: int
    top_k: int
    default_repeats: int
    default_seed: int
    structured_seed: int | None
    chat_model: str
    embedding_model: str
    prompt_template_version: str
    retry_max_attempts: int
    retry_base_delay_seconds: float
    openai_api_key: str

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    load_dotenv()
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    return Settings(
        project_root=project_root,
        data_dir=data_dir,
        chroma_dir=project_root / "chroma",
        runs_dir=project_root / "runs",
        results_dir=project_root / "results",
        corpus_path=data_dir / "corpus.jsonl",
        questions_path=data_dir / "questions.jsonl",
        retrieval_frozen_path=data_dir / "retrieval_frozen.jsonl",
        replay_cache_path=project_root / "runs" / "replay_cache.json",
        collection_name=os.getenv("RAG_REPEAT_COLLECTION", "corpus_chunks"),
        chunk_size_chars=_int_from_env("RAG_REPEAT_CHUNK_SIZE", 600),
        chunk_overlap_chars=_int_from_env("RAG_REPEAT_CHUNK_OVERLAP", 50),
        top_k=_int_from_env("RAG_REPEAT_TOP_K", 4),
        default_repeats=_int_from_env("RAG_REPEAT_REPEATS", 10),
        default_seed=_int_from_env("RAG_REPEAT_SEED", 12345),
        structured_seed=_optional_int_from_env("RAG_REPEAT_STRUCTURED_SEED"),
        chat_model=os.getenv("RAG_REPEAT_CHAT_MODEL", "gpt-4o-mini"),
        embedding_model=os.getenv("RAG_REPEAT_EMBEDDING_MODEL", "text-embedding-3-small"),
        prompt_template_version=os.getenv("RAG_REPEAT_PROMPT_VERSION", "v1"),
        retry_max_attempts=_int_from_env("RAG_REPEAT_RETRY_MAX_ATTEMPTS", 5),
        retry_base_delay_seconds=float(os.getenv("RAG_REPEAT_RETRY_BASE_DELAY", "1.0")),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
    )
