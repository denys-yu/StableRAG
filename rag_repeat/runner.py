"""Experiment runner for repeated generation on frozen retrieval inputs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from tqdm import tqdm

from rag_repeat.chroma_store import read_jsonl
from rag_repeat.config import Settings
from rag_repeat.openai_client import OpenAIClientWrapper
from rag_repeat.rag import SYSTEM_PROMPT, StructuredAnswer, build_user_prompt, normalize_whitespace, render_structured_answer
from rag_repeat.variants import VariantSpec


@dataclass
class ReplayPayload:
    output_text: str
    system_fingerprint: str | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ReplayCache:
    """Disk-backed replay cache for byte-identical output replay."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._cache: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            self._cache = {}
            return
        self._cache = json.loads(self._path.read_text(encoding="utf-8"))

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(self._cache, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        temp_path.replace(self._path)

    def get(self, key: str) -> ReplayPayload | None:
        payload = self._cache.get(key)
        if payload is None:
            return None
        return ReplayPayload(
            output_text=str(payload["output_text"]),
            system_fingerprint=payload.get("system_fingerprint"),
            prompt_tokens=int(payload.get("prompt_tokens", 0)),
            completion_tokens=int(payload.get("completion_tokens", 0)),
            total_tokens=int(payload.get("total_tokens", 0)),
        )

    def set(self, key: str, payload: ReplayPayload) -> None:
        self._cache[key] = asdict(payload)
        self._persist()


def _json_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_key(
    *,
    variant: VariantSpec,
    settings: Settings,
    qid: str,
    chunk_ids: Sequence[str],
    chunk_texts: Sequence[str],
) -> str:
    chunk_text_blob = "\n".join(chunk_texts)
    payload = {
        "model": settings.chat_model,
        "temperature": variant.temperature,
        "top_p": variant.top_p,
        "seed": variant.seed,
        "prompt_template_version": settings.prompt_template_version,
        "qid": qid,
        "chunk_ids": list(chunk_ids),
        "chunk_text_hash": _text_hash(chunk_text_blob),
    }
    return _json_hash(payload)


def run_variants(
    *,
    settings: Settings,
    client: OpenAIClientWrapper,
    variants: Sequence[VariantSpec],
    repeats: int,
    retrieval_frozen_path: Path,
) -> list[Path]:
    rows = read_jsonl(retrieval_frozen_path)
    if not rows:
        raise ValueError(f"No frozen retrieval rows found in {retrieval_frozen_path}.")

    run_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    cache = ReplayCache(settings.replay_cache_path)
    output_paths: list[Path] = []

    for variant in variants:
        output_path = settings.runs_dir / f"{run_tag}__{variant.name}.jsonl"
        output_paths.append(output_path)

        with output_path.open("w", encoding="utf-8") as handle:
            progress = tqdm(rows, desc=f"Running {variant.name}", total=len(rows))
            for row in progress:
                qid = str(row["qid"])
                question = str(row["question"])
                chunks = list(row.get("chunks", []))
                chunk_ids = [str(chunk["chunk_id"]) for chunk in chunks]
                chunk_texts = [str(chunk["text"]) for chunk in chunks]
                user_prompt = build_user_prompt(question, chunks)

                for run_index in range(repeats):
                    start = perf_counter()
                    request_params = {
                        "model": settings.chat_model,
                        "temperature": variant.temperature,
                        "top_p": variant.top_p,
                        "seed": variant.seed,
                        "prompt_template_version": settings.prompt_template_version,
                        "structured_output": variant.structured_output,
                        "canonical_render": variant.canonical_render,
                        "replay_cache": variant.replay_cache,
                    }

                    replay_hit = False
                    replay_payload: ReplayPayload | None = None
                    cache_key: str | None = None
                    if variant.replay_cache:
                        cache_key = _cache_key(
                            variant=variant,
                            settings=settings,
                            qid=qid,
                            chunk_ids=chunk_ids,
                            chunk_texts=chunk_texts,
                        )
                        replay_payload = cache.get(cache_key)
                        replay_hit = replay_payload is not None

                    if replay_payload is not None:
                        output_text = replay_payload.output_text
                        system_fingerprint = replay_payload.system_fingerprint
                        prompt_tokens = replay_payload.prompt_tokens
                        completion_tokens = replay_payload.completion_tokens
                        total_tokens = replay_payload.total_tokens
                    else:
                        if variant.structured_output:
                            completion = client.chat_completion_parse(
                                model=settings.chat_model,
                                system_prompt=SYSTEM_PROMPT,
                                user_prompt=user_prompt,
                                temperature=variant.temperature,
                                top_p=variant.top_p,
                                seed=variant.seed,
                                response_model=StructuredAnswer,
                            )
                            parsed = StructuredAnswer.model_validate(completion.parsed)
                            output_text = render_structured_answer(parsed, canonical=variant.canonical_render)
                        else:
                            completion = client.chat_completion(
                                model=settings.chat_model,
                                system_prompt=SYSTEM_PROMPT,
                                user_prompt=user_prompt,
                                temperature=variant.temperature,
                                top_p=variant.top_p,
                                seed=variant.seed,
                            )
                            output_text = completion.text.strip()
                            if variant.canonical_render:
                                output_text = normalize_whitespace(output_text)

                        system_fingerprint = completion.system_fingerprint
                        prompt_tokens = completion.usage.prompt_tokens
                        completion_tokens = completion.usage.completion_tokens
                        total_tokens = completion.usage.total_tokens

                        if variant.replay_cache and cache_key is not None:
                            cache.set(
                                cache_key,
                                ReplayPayload(
                                    output_text=output_text,
                                    system_fingerprint=system_fingerprint,
                                    prompt_tokens=prompt_tokens,
                                    completion_tokens=completion_tokens,
                                    total_tokens=total_tokens,
                                ),
                            )

                    latency_ms = (perf_counter() - start) * 1000.0
                    record = {
                        "qid": qid,
                        "variant": variant.name,
                        "run_index": run_index,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "request_params": request_params,
                        "system_fingerprint": system_fingerprint,
                        "latency_ms": latency_ms,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                        "output_text": output_text,
                        "output_hash": _text_hash(output_text),
                        "cache_hit": replay_hit,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_paths
