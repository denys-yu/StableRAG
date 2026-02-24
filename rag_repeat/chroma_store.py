"""Local ChromaDB persistence and deterministic retrieval helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import chromadb
from tqdm import tqdm

from rag_repeat.rag import split_into_chunks


@dataclass(frozen=True)
class ChunkRow:
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_chunk_rows(
    corpus_rows: Sequence[dict[str, Any]],
    *,
    chunk_size_chars: int,
    overlap_chars: int,
) -> list[ChunkRow]:
    chunk_rows: list[ChunkRow] = []
    for row in corpus_rows:
        doc_id = str(row["doc_id"])
        doc_text = str(row["text"])
        chunks = split_into_chunks(
            doc_text,
            chunk_size_chars=chunk_size_chars,
            overlap_chars=overlap_chars,
        )
        for chunk_index, chunk_text in enumerate(chunks):
            chunk_rows.append(
                ChunkRow(
                    chunk_id=f"{doc_id}__chunk_{chunk_index:03d}",
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    text=chunk_text,
                )
            )
    return chunk_rows


class ChromaStore:
    """Wrapper for local persistent Chroma collection operations."""

    def __init__(self, *, persist_directory: Path, collection_name: str) -> None:
        self._persist_directory = persist_directory
        self._collection_name = collection_name
        self._client = chromadb.PersistentClient(path=str(persist_directory))

    def reset_collection(self) -> None:
        try:
            self._client.delete_collection(name=self._collection_name)
        except Exception:
            pass
        self._client.create_collection(name=self._collection_name, metadata={"hnsw:space": "cosine"})

    def _get_collection(self):
        return self._client.get_collection(name=self._collection_name)

    def count(self) -> int:
        return int(self._get_collection().count())

    def add_chunks(
        self,
        chunk_rows: Sequence[ChunkRow],
        embeddings: Sequence[Sequence[float]],
        *,
        batch_size: int = 64,
    ) -> None:
        if len(chunk_rows) != len(embeddings):
            raise ValueError("Chunk and embedding counts must match.")
        collection = self._get_collection()
        for start in range(0, len(chunk_rows), batch_size):
            rows_batch = chunk_rows[start : start + batch_size]
            emb_batch = embeddings[start : start + batch_size]
            collection.add(
                ids=[row.chunk_id for row in rows_batch],
                embeddings=[list(vec) for vec in emb_batch],
                documents=[row.text for row in rows_batch],
                metadatas=[
                    {
                        "doc_id": row.doc_id,
                        "chunk_index": row.chunk_index,
                    }
                    for row in rows_batch
                ],
            )

    def query(self, query_embedding: Sequence[float], *, top_k: int) -> list[dict[str, Any]]:
        collection = self._get_collection()
        n_results = min(max(top_k * 3, top_k), max(collection.count(), 1))
        response = collection.query(query_embeddings=[list(query_embedding)], n_results=n_results)
        ids = response.get("ids", [[]])[0]
        distances = response.get("distances", [[]])[0]
        docs = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]

        rows: list[dict[str, Any]] = []
        for chunk_id, distance, text, metadata in zip(ids, distances, docs, metadatas):
            rows.append(
                {
                    "chunk_id": str(chunk_id),
                    "distance": float(distance),
                    "doc_id": str(metadata.get("doc_id", "")),
                    "chunk_index": int(metadata.get("chunk_index", 0)),
                    "text": str(text),
                }
            )

        rows.sort(key=lambda item: (item["distance"], item["chunk_id"]))
        return rows[:top_k]


def build_index(
    *,
    corpus_path: Path,
    store: ChromaStore,
    embed_texts_fn,
    embedding_model: str,
    chunk_size_chars: int,
    overlap_chars: int,
) -> int:
    corpus_rows = read_jsonl(corpus_path)
    if not corpus_rows:
        raise ValueError(f"No corpus rows found in {corpus_path}.")

    chunk_rows = build_chunk_rows(
        corpus_rows,
        chunk_size_chars=chunk_size_chars,
        overlap_chars=overlap_chars,
    )
    if not chunk_rows:
        raise ValueError("Chunking produced no rows. Check corpus content.")

    store.reset_collection()
    embeddings = embed_texts_fn(
        texts=[row.text for row in chunk_rows],
        model=embedding_model,
    )
    store.add_chunks(chunk_rows, embeddings)
    return len(chunk_rows)


def freeze_retrieval(
    *,
    questions_path: Path,
    output_path: Path,
    store: ChromaStore,
    embed_texts_fn,
    embedding_model: str,
    top_k: int,
) -> int:
    question_rows = read_jsonl(questions_path)
    if not question_rows:
        raise ValueError(f"No questions found in {questions_path}.")

    question_texts = [str(row["question"]) for row in question_rows]
    query_embeddings = embed_texts_fn(texts=question_texts, model=embedding_model)
    frozen_rows: list[dict[str, Any]] = []
    for question_row, query_embedding in tqdm(
        list(zip(question_rows, query_embeddings)),
        desc="Freezing retrieval",
        total=len(question_rows),
    ):
        chunks = store.query(query_embedding, top_k=top_k)
        frozen_rows.append(
            {
                "qid": str(question_row["qid"]),
                "question": str(question_row["question"]),
                "chunks": [
                    {
                        "chunk_id": chunk["chunk_id"],
                        "distance": chunk["distance"],
                        "text": chunk["text"],
                    }
                    for chunk in chunks
                ],
            }
        )

    write_jsonl(output_path, frozen_rows)
    return len(frozen_rows)
