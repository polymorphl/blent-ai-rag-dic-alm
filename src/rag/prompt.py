from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from src import config

_SYSTEM = (
    "Tu es un assistant spécialisé pour les équipes ALM d'une compagnie d'assurance vie.\n"
    "Réponds uniquement en te basant sur les extraits de DIC fournis.\n"
    "Si la réponse ne figure pas dans les documents, dis-le explicitement."
)


def build(question: str, history: list[dict], chunks: list[dict]) -> list[BaseMessage]:
    system_content = _SYSTEM
    if chunks:
        doc_parts = []
        for chunk in chunks:
            source = chunk.get("source") or "inconnu"
            raw = chunk.get("metadata", {}).get("page")
            page = str(raw + 1) if isinstance(raw, int) else "?"
            doc_parts.append(f"[Source: {source}, page {page}]\n{chunk.get('text', '')}")
        system_content += "\n\nDocuments pertinents :\n\n" + "\n\n".join(doc_parts)

    messages: list[BaseMessage] = [SystemMessage(content=system_content)]

    recent = history[-(config.HISTORY_MAX_TURNS * 2):]
    for msg in recent:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=question))
    return messages
