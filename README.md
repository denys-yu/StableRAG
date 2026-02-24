# RAG Repeatability Harness

Minimal experimental harness to evaluate repeatability of a simple RAG pipeline:

- OpenAI API for generation and embeddings
- Local persistent ChromaDB for retrieval
- Repeatability + ablation + latency/token overhead outputs for paper tables

## Requirements

- Python 3.10+
- OpenAI API key with access to:
  - Chat model (default: `gpt-4o-mini`)
  - Embedding model (default: `text-embedding-3-small`)

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# then edit .env and set OPENAI_API_KEY
```

## Data

Local controlled datasets are provided:

- `data/corpus.jsonl` (48 short documents)
- `data/questions.jsonl` (40 questions)

No external dataset download is required.

## Commands

Run commands from project root:

1) Build index

```powershell
python -m rag_repeat.cli build_index
```

2) Freeze retrieval

```powershell
python -m rag_repeat.cli freeze_retrieval
```

3) Run experiments

```powershell
python -m rag_repeat.cli run --repeats 10 --variants baseline_t0,seed_t0,structured_t0,structured_render_t0
```

Optional replay variant:

```powershell
python -m rag_repeat.cli run --repeats 10 --variants baseline_t0,seed_t0,structured_t0,structured_render_t0,replay_cache
```

4) Summarize results

```powershell
python -m rag_repeat.cli summarize
```

Or summarize a specific run tag:

```powershell
python -m rag_repeat.cli summarize --runs_dir runs/20260224_130000
```

Run everything in one command:

```powershell
python -m rag_repeat.cli all
```

PyCharm run helper:

```powershell
python scripts/run_all.py
```

## Outputs

- Frozen retrieval file: `data/retrieval_frozen.jsonl`
- Raw run logs: `runs/<timestamp>__<variant>.jsonl`
- Summary table CSV: `results/summary.csv`
- Summary table Markdown: `results/summary.md`
- Per-question metrics: `results/per_question.csv`

## Notes on Determinism

- Retrieval is frozen once (`freeze_retrieval`) to remove retrieval variance.
- Retrieval sorting is stable: `(distance ASC, chunk_id ASC)`.
- `structured_render_t0` canonicalizes structured output by:
  - normalizing whitespace
  - sorting evidence indices
  - rendering exact format:
    - `Answer: <final_answer>`
    - `Evidence: [..]`
    - `NotFound: <true/false>`

## Config Overrides

Environment variable overrides:

- `RAG_REPEAT_CHAT_MODEL` (default `gpt-4o-mini`)
- `RAG_REPEAT_EMBEDDING_MODEL` (default `text-embedding-3-small`)
- `RAG_REPEAT_TOP_K` (default `4`)
- `RAG_REPEAT_REPEATS` (default `10`)
- `RAG_REPEAT_SEED` (default `12345`)
- `RAG_REPEAT_STRUCTURED_SEED` (optional; unset by default)
