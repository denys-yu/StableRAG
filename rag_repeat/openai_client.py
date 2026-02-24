"""Thin OpenAI SDK wrapper with retries and normalized response objects."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar

from openai import APIConnectionError, APIError, APITimeoutError, InternalServerError, OpenAI, RateLimitError
from pydantic import BaseModel


T = TypeVar("T")
StructuredModel = TypeVar("StructuredModel", bound=BaseModel)


@dataclass(frozen=True)
class CompletionUsageStats:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class TextCompletionResult:
    text: str
    system_fingerprint: str | None
    usage: CompletionUsageStats


@dataclass(frozen=True)
class StructuredCompletionResult:
    parsed: BaseModel
    raw_text: str
    system_fingerprint: str | None
    usage: CompletionUsageStats


class OpenAIClientWrapper:
    """High-level OpenAI calls used by indexing, generation, and metric scoring."""

    def __init__(
        self,
        *,
        api_key: str,
        retry_max_attempts: int = 5,
        retry_base_delay_seconds: float = 1.0,
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required. Set it in .env or environment variables.")
        self._client = OpenAI(api_key=api_key)
        self._retry_max_attempts = retry_max_attempts
        self._retry_base_delay_seconds = retry_base_delay_seconds

    def _call_with_retries(self, fn: Callable[[], T]) -> T:
        for attempt in range(self._retry_max_attempts):
            try:
                return fn()
            except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError, APIError):
                if attempt >= self._retry_max_attempts - 1:
                    raise
                sleep_seconds = self._retry_base_delay_seconds * (2**attempt)
                time.sleep(sleep_seconds)
        raise RuntimeError("Retry loop exhausted unexpectedly.")

    @staticmethod
    def _usage_to_stats(usage_obj: object) -> CompletionUsageStats:
        if usage_obj is None:
            return CompletionUsageStats(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        prompt_tokens = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage_obj, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage_obj, "total_tokens", prompt_tokens + completion_tokens) or 0)
        return CompletionUsageStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    def chat_completion(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        top_p: float,
        seed: int | None = None,
    ) -> TextCompletionResult:
        request = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "top_p": top_p,
        }
        if seed is not None:
            request["seed"] = seed

        completion = self._call_with_retries(lambda: self._client.chat.completions.create(**request))
        message = completion.choices[0].message
        text = (message.content or "").strip()
        usage = self._usage_to_stats(completion.usage)
        return TextCompletionResult(
            text=text,
            system_fingerprint=getattr(completion, "system_fingerprint", None),
            usage=usage,
        )

    def chat_completion_parse(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        top_p: float,
        response_model: type[StructuredModel],
        seed: int | None = None,
    ) -> StructuredCompletionResult:
        request = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "response_format": response_model,
        }
        if seed is not None:
            request["seed"] = seed

        completion = self._call_with_retries(lambda: self._client.chat.completions.parse(**request))
        message = completion.choices[0].message
        parsed = message.parsed
        if parsed is None:
            raise ValueError("Structured parse returned no parsed object.")
        usage = self._usage_to_stats(completion.usage)
        return StructuredCompletionResult(
            parsed=parsed,
            raw_text=(message.content or "").strip(),
            system_fingerprint=getattr(completion, "system_fingerprint", None),
            usage=usage,
        )

    def embed_texts(self, *, texts: Sequence[str], model: str, batch_size: int = 64) -> list[list[float]]:
        vectors: list[list[float]] = []
        for idx in range(0, len(texts), batch_size):
            batch = list(texts[idx : idx + batch_size])
            if not batch:
                continue
            response = self._call_with_retries(
                lambda: self._client.embeddings.create(model=model, input=batch)
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend([row.embedding for row in ordered])
        return vectors
