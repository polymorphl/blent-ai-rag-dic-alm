# RAG Conversational Agent for Financial KID Documents

A retrieval-augmented generation (RAG) chatbot that allows ALM (Asset-Liability Management) teams at a life insurance company to query a corpus of Key Information Documents (KIDs).

## Context

The ALM department needs to analyse a large volume of KIDs to guide investment decisions. This project provides a conversational assistant that makes it easy to retrieve information scattered across all available documents.

## Development steps

### 1. PDF chunking and embedding

- Intelligent parsing and chunking of PDF documents (`chunk_size`, `chunk_overlap`)
- Embedding suited to the language and financial domain
- Local vector storage (ChromaDB or FAISS)

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
uv run python -m src.main
```

## Tests

```bash
uv run pytest
```
