"""Command-line entrypoint for the RAG repeatability harness."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from rag_repeat.chroma_store import ChromaStore, build_index, freeze_retrieval
from rag_repeat.config import get_settings
from rag_repeat.openai_client import OpenAIClientWrapper
from rag_repeat.reporting import summarize_runs
from rag_repeat.runner import run_variants
from rag_repeat.variants import default_variant_specs, resolve_variants


def _make_client(settings):
    return OpenAIClientWrapper(
        api_key=settings.openai_api_key,
        retry_max_attempts=settings.retry_max_attempts,
        retry_base_delay_seconds=settings.retry_base_delay_seconds,
    )


def cmd_build_index(_: argparse.Namespace) -> None:
    settings = get_settings()
    settings.ensure_directories()
    client = _make_client(settings)
    store = ChromaStore(
        persist_directory=settings.chroma_dir,
        collection_name=settings.collection_name,
    )
    chunk_count = build_index(
        corpus_path=settings.corpus_path,
        store=store,
        embed_texts_fn=client.embed_texts,
        embedding_model=settings.embedding_model,
        chunk_size_chars=settings.chunk_size_chars,
        overlap_chars=settings.chunk_overlap_chars,
    )
    print(f"Indexed {chunk_count} chunks into collection '{settings.collection_name}'.")


def cmd_freeze_retrieval(args: argparse.Namespace) -> None:
    settings = get_settings()
    settings.ensure_directories()
    client = _make_client(settings)
    store = ChromaStore(
        persist_directory=settings.chroma_dir,
        collection_name=settings.collection_name,
    )
    frozen_count = freeze_retrieval(
        questions_path=settings.questions_path,
        output_path=settings.retrieval_frozen_path,
        store=store,
        embed_texts_fn=client.embed_texts,
        embedding_model=settings.embedding_model,
        top_k=args.top_k or settings.top_k,
    )
    print(f"Frozen retrieval for {frozen_count} questions into {settings.retrieval_frozen_path}.")


def cmd_run(args: argparse.Namespace) -> list[Path]:
    settings = get_settings()
    settings.ensure_directories()
    client = _make_client(settings)
    variants_map = default_variant_specs(
        default_seed=settings.default_seed,
        structured_seed=settings.structured_seed,
    )
    selected_variants = resolve_variants(args.variants, variants_map)
    run_files = run_variants(
        settings=settings,
        client=client,
        variants=selected_variants,
        repeats=args.repeats or settings.default_repeats,
        retrieval_frozen_path=settings.retrieval_frozen_path,
    )
    print("Run files:")
    for file in run_files:
        print(file)
    return run_files


def cmd_summarize(args: argparse.Namespace) -> None:
    settings = get_settings()
    settings.ensure_directories()
    client = _make_client(settings)
    output_paths = summarize_runs(
        settings=settings,
        client=client,
        runs_dir_arg=args.runs_dir,
    )
    print(f"Wrote summary CSV: {output_paths['summary_csv']}")
    print(f"Wrote summary Markdown: {output_paths['summary_md']}")
    print(f"Wrote per-question CSV: {output_paths['per_question_csv']}")


def cmd_all(args: argparse.Namespace) -> None:
    settings = get_settings()
    settings.ensure_directories()
    client = _make_client(settings)
    store = ChromaStore(
        persist_directory=settings.chroma_dir,
        collection_name=settings.collection_name,
    )

    chunk_count = build_index(
        corpus_path=settings.corpus_path,
        store=store,
        embed_texts_fn=client.embed_texts,
        embedding_model=settings.embedding_model,
        chunk_size_chars=settings.chunk_size_chars,
        overlap_chars=settings.chunk_overlap_chars,
    )
    print(f"Indexed {chunk_count} chunks.")

    frozen_count = freeze_retrieval(
        questions_path=settings.questions_path,
        output_path=settings.retrieval_frozen_path,
        store=store,
        embed_texts_fn=client.embed_texts,
        embedding_model=settings.embedding_model,
        top_k=args.top_k or settings.top_k,
    )
    print(f"Frozen retrieval rows: {frozen_count}.")

    variants_map = default_variant_specs(
        default_seed=settings.default_seed,
        structured_seed=settings.structured_seed,
    )
    selected_variants = resolve_variants(args.variants, variants_map)
    run_files = run_variants(
        settings=settings,
        client=client,
        variants=selected_variants,
        repeats=args.repeats or settings.default_repeats,
        retrieval_frozen_path=settings.retrieval_frozen_path,
    )
    print(f"Generated run files: {len(run_files)}")

    output_paths = summarize_runs(
        settings=settings,
        client=client,
        runs_dir_arg=None,
    )
    print(f"Wrote summary CSV: {output_paths['summary_csv']}")
    print(f"Wrote summary Markdown: {output_paths['summary_md']}")
    print(f"Wrote per-question CSV: {output_paths['per_question_csv']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG repeatability experiment harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_build = subparsers.add_parser("build_index", help="Build local Chroma index from corpus.jsonl")
    parser_build.set_defaults(func=cmd_build_index)

    parser_freeze = subparsers.add_parser("freeze_retrieval", help="Retrieve top-k chunks once and freeze to JSONL")
    parser_freeze.add_argument("--top_k", type=int, default=None, help="Top-k retrieval count")
    parser_freeze.set_defaults(func=cmd_freeze_retrieval)

    parser_run = subparsers.add_parser("run", help="Run repeated generation experiments on frozen retrieval")
    parser_run.add_argument("--repeats", type=int, default=None, help="Number of repeated runs per question")
    parser_run.add_argument(
        "--variants",
        type=str,
        default=None,
        help="Comma-separated variants (default: baseline_t0,seed_t0,structured_t0,structured_render_t0)",
    )
    parser_run.set_defaults(func=cmd_run)

    parser_summarize = subparsers.add_parser("summarize", help="Create summary CSV/Markdown tables")
    parser_summarize.add_argument(
        "--runs_dir",
        type=str,
        default=None,
        help="Run selector: directory, single file, or timestamp prefix (e.g., runs/20260224_120000)",
    )
    parser_summarize.set_defaults(func=cmd_summarize)

    parser_all = subparsers.add_parser("all", help="Execute build_index -> freeze_retrieval -> run -> summarize")
    parser_all.add_argument("--repeats", type=int, default=None, help="Number of repeats for run command")
    parser_all.add_argument(
        "--variants",
        type=str,
        default=None,
        help="Comma-separated variants list for run command",
    )
    parser_all.add_argument("--top_k", type=int, default=None, help="Top-k retrieval for freeze step")
    parser_all.set_defaults(func=cmd_all)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
