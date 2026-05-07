import torch
from sentence_transformers import SentenceTransformer
from src import config

_model: SentenceTransformer | None = None


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL, device=get_device())
    return _model


def warm_up() -> None:
    """Pre-load the embedding model to avoid lazy-loading during the first query."""
    _get_model()


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed a list of document passages (adds required 'passage:' prefix for E5 models)."""
    model = _get_model()
    prefixed = [f"passage: {t}" for t in texts]
    return model.encode(prefixed, convert_to_numpy=True, show_progress_bar=True).tolist()


def embed_query(text: str) -> list[float]:
    """Embed a user query (adds required 'query:' prefix for E5 models)."""
    model = _get_model()
    return model.encode(f"query: {text}", convert_to_numpy=True).tolist()
