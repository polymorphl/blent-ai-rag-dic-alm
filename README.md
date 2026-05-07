# RAG Conversational Agent for Financial KID Documents

A retrieval-augmented generation (RAG) chatbot that allows ALM (Asset-Liability Management) teams at a life insurance company to query a corpus of Key Information Documents (KIDs).

## Context

The ALM department needs to analyse a large volume of KIDs to guide investment decisions. This project provides a conversational assistant that makes it easy to retrieve information scattered across all available documents.

## Development steps

### 1. PDF chunking and embedding

- PDF parsing with **pymupdf**
- Chunking with `RecursiveCharacterTextSplitter` (`chunk_size=1000`, `chunk_overlap=200`)
- Embedding with **intfloat/multilingual-e5-large** (French, local via `sentence-transformers`)
- Local vector storage with **ChromaDB** (persisted to `data/chroma_db/`)

To inspect chunk quality and calibrate `chunk_size` / `chunk_overlap` before indexing:

```bash
# Single PDF
uv run python -m src.ingestion.explore_chunks seed/DIC/Allianz.pdf

# Entire corpus
uv run python -m src.ingestion.explore_chunks seed/DIC
```

### 2. RAG pipeline

- Open-weight model **Mistral** (configurable via `LLM_MODEL` in `src/config.py`) served locally via **Ollama**
- Retrieval of the top `RAG_K` chunks from ChromaDB, injected into the prompt with source filename and page number
- Conversational memory: in-session exchange history passed to the model at each turn (capped at `HISTORY_MAX_TURNS`)
- Two interfaces: CLI REPL (`src/cli.py`) and web chat UI (`src/app.py` — Streamlit)
- Embedding model is pre-loaded at startup in both interfaces, before the first query

### 3. Evaluation

Two metrics computed against a structured evaluation dataset (619 queries):

- **BertScore F1** — semantic similarity between generated and expected answers (`bert-base-multilingual-cased`)
- **Recall@k** — fraction of relevant corpus chunks retrieved per query

| File | Contents |
|---|---|
| `corpus.json` | All chunks making up the documents |
| `queries.json` | Queries formulated against each corpus |
| `relevant_docs.json` | Relevant documents for each query |
| `answers.json` | Expected answer for each query |
| `errors.json` | Any inference errors encountered |

Each file is a `{"uuid": "..."}` dictionary — the UUID is the join key across files.

**Minimum required threshold: F1 BertScore ≥ 60%**

Pre-computed results are included in the repository:

| File | Contents |
|---|---|
| `data/eval_cache.json` | Cached RAG answers for all 619 queries |
| `data/eval_results.json` | Latest evaluation results (mean F1: **0.689** ✓, mean Recall@k: 0.044) |

## Prerequisites

- [uv](https://docs.astral.sh/uv/) — Python package manager

## Getting started

### 1. Ingest the PDF corpus

```bash
uv sync
```

Unzip the PDF corpus into `seed/DIC/`:

```bash
unzip seed/DIC.zip -d seed/
```

Then run the ingestion pipeline:

```bash
uv run python -m src.main
```

### 1b. Unzip the evaluation dataset

```bash
unzip seed/dataset_eval.zip -d seed/dataset_eval
```

### 2. Start Ollama

**Option A — Native install (recommended)**

macOS:
```bash
brew install ollama
ollama serve          # start the server (keep this terminal open)
ollama pull mistral   # in another terminal
```

Linux:
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve # start the server (keep this terminal open)
ollama pull mistral
```

**Option B — Docker**

```bash
docker run -d --name ollama -p 11434:11434 ollama/ollama
docker exec ollama ollama pull mistral
```

To stop and restart the container:
```bash
docker stop ollama
docker start ollama
```

### 3. Run the RAG assistant

**CLI (interactive REPL):**

```bash
uv run python -m src.cli
```

**Web interface (Streamlit):**

```bash
PYTHONPATH=. uv run streamlit run src/app.py --server.fileWatcherType none
```

Open `http://localhost:8501` in your browser.

> `src/static` is a symlink to `seed/DIC/` — used by Streamlit's static file serving to expose the PDF files as clickable links in the chat UI.

### 4. Run the evaluation

Evaluate the RAG pipeline against the structured dataset (requires Ollama running):

```bash
uv run python -m src.eval
```

Results are saved to `data/eval_results.json`. The minimum required threshold is **BertScore F1 ≥ 60%**.

A pre-computed cache (`data/eval_cache.json`) is included in the repository — the first run will skip the 619 Ollama calls and go straight to metric computation.

To force a full re-run (ignore cache):

```bash
uv run python -m src.eval --no-cache
```

## Tests

```bash
uv run pytest
```
