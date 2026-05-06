import hashlib
from pathlib import Path
import chromadb
from langchain_core.documents import Document
from src import config

_client: chromadb.PersistentClient | None = None


def _get_collection() -> chromadb.Collection:
    global _client
    if _client is None:
        Path(config.CHROMA_DIR).mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    return _client.get_or_create_collection(config.CHROMA_COLLECTION)


def add(chunks: list[Document], embeddings: list[list[float]]) -> None:
    """Upsert chunks and their embeddings into ChromaDB (idempotent via deterministic MD5 IDs)."""
    collection = _get_collection()
    ids, texts, metadatas, vecs = [], [], [], []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        raw = f"{chunk.metadata['source']}_{i}_{chunk.page_content[:50]}"
        ids.append(hashlib.md5(raw.encode()).hexdigest())
        texts.append(chunk.page_content)
        metadatas.append(chunk.metadata)
        vecs.append(emb)
    collection.upsert(ids=ids, documents=texts, embeddings=vecs, metadatas=metadatas)


def query(embedding: list[float], k: int = 5) -> list[dict]:
    """Return the k nearest chunks with their text, source filename, similarity score, and metadata."""
    collection = _get_collection()
    results = collection.query(query_embeddings=[embedding], n_results=k)
    docs, metas, dists = results["documents"][0], results["metadatas"][0], results["distances"][0]
    return [
        {"text": doc, "source": meta.get("source"), "score": 1 - dist, "metadata": meta}
        for doc, meta, dist in zip(docs, metas, dists)
    ]
