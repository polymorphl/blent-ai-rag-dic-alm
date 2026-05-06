from langchain_ollama import ChatOllama
from src import config
from src.ingestion import embedder, vector_store
from src.rag import prompt


class RagEngine:
    def __init__(self) -> None:
        self._llm = ChatOllama(model=config.LLM_MODEL)

    def ask(self, question: str, history: list[dict]) -> dict:
        clean_history = [{"role": m["role"], "content": m["content"]} for m in history]
        embedding = embedder.embed_query(question)
        chunks = vector_store.query(embedding, k=config.RAG_K)
        prompt_text = prompt.build(question, clean_history, chunks)
        try:
            response = self._llm.invoke(prompt_text)
            answer = response.content
        except Exception:
            answer = config.LLM_UNAVAILABLE_MSG
        return {"answer": answer, "sources": _deduplicate_sources(chunks)}


def _deduplicate_sources(chunks: list[dict]) -> list[str]:
    pages: dict[str, list[str]] = {}
    for chunk in chunks:
        source = chunk.get("source") or "inconnu"
        raw = chunk.get("metadata", {}).get("page")
        page = str(raw + 1) if isinstance(raw, int) else "?"
        if source not in pages:
            pages[source] = []
        if page not in pages[source]:
            pages[source].append(page)
    return sorted(f"{src} ({', '.join(f'p.{p}' for p in pgs)})" for src, pgs in pages.items())
