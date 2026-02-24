"""Variant definitions for repeatability and ablation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class VariantSpec:
    name: str
    temperature: float
    top_p: float
    seed: int | None
    structured_output: bool
    canonical_render: bool
    replay_cache: bool


def default_variant_specs(*, default_seed: int, structured_seed: int | None) -> dict[str, VariantSpec]:
    return {
        "baseline_t0": VariantSpec(
            name="baseline_t0",
            temperature=0.0,
            top_p=1.0,
            seed=None,
            structured_output=False,
            canonical_render=False,
            replay_cache=False,
        ),
        "seed_t0": VariantSpec(
            name="seed_t0",
            temperature=0.0,
            top_p=1.0,
            seed=default_seed,
            structured_output=False,
            canonical_render=False,
            replay_cache=False,
        ),
        "structured_t0": VariantSpec(
            name="structured_t0",
            temperature=0.0,
            top_p=1.0,
            seed=structured_seed,
            structured_output=True,
            canonical_render=False,
            replay_cache=False,
        ),
        "structured_render_t0": VariantSpec(
            name="structured_render_t0",
            temperature=0.0,
            top_p=1.0,
            seed=structured_seed,
            structured_output=True,
            canonical_render=True,
            replay_cache=False,
        ),
        "replay_cache": VariantSpec(
            name="replay_cache",
            temperature=0.0,
            top_p=1.0,
            seed=default_seed,
            structured_output=False,
            canonical_render=False,
            replay_cache=True,
        ),
    }


def resolve_variants(csv_variants: str | None, all_variants: Mapping[str, VariantSpec]) -> list[VariantSpec]:
    if csv_variants is None or not csv_variants.strip():
        names = ["baseline_t0", "seed_t0", "structured_t0", "structured_render_t0"]
    else:
        names = [token.strip() for token in csv_variants.split(",") if token.strip()]
    missing = [name for name in names if name not in all_variants]
    if missing:
        supported = ", ".join(sorted(all_variants))
        raise ValueError(f"Unsupported variants: {missing}. Supported variants: {supported}")
    return [all_variants[name] for name in names]
