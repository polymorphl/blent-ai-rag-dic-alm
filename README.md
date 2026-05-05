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

- Open-weight model (Mistral / Llama) running locally
- Relevant chunk retrieval and sourced answer generation
- Conversational memory to preserve exchange history

### 3. Evaluation

**F1 BertScore** computed against a structured evaluation dataset:

| File | Contents |
|---|---|
| `corpus.json` | All chunks making up the documents |
| `queries.json` | Queries formulated against each corpus |
| `relevant_docs.json` | Relevant documents for each query |
| `answers.json` | Expected answer for each query |
| `errors.json` | Any inference errors encountered |

Each file is a `{"uuid": "..."}` dictionary — the UUID is the join key across files.

**Minimum required threshold: F1 BertScore ≥ 60%**

## Getting started

```bash
uv sync
```

Unzip the PDF corpus into `seed/DIC/`:

```bash
unzip DIC.zip -d seed/DIC
```

Then run the ingestion pipeline:

```bash
uv run python -m src.main
```

## Tests

```bash
uv run pytest
```
