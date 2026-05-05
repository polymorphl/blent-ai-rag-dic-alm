import pytest
from langchain_core.documents import Document
from src import config
from src.ingestion import vector_store as vs


@pytest.fixture(autouse=True)
def isolated_chroma(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(config, "CHROMA_COLLECTION", "test_collection")


def test_add_and_query_returns_result():
    chunks = [Document(
        page_content="Le risque de ce produit est élevé.",
        metadata={"source": "allianz.pdf", "page": 0},
    )]
    embeddings = [[0.1] * config.EMBEDDING_DIM]
    vs.add(chunks, embeddings)
    results = vs.query([0.1] * config.EMBEDDING_DIM, k=1)
    assert len(results) == 1


def test_query_result_has_source():
    chunks = [Document(
        page_content="Les coûts annuels sont de 1,5%.",
        metadata={"source": "pictet.pdf", "page": 1},
    )]
    embeddings = [[0.2] * config.EMBEDDING_DIM]
    vs.add(chunks, embeddings)
    results = vs.query([0.2] * config.EMBEDDING_DIM, k=1)
    assert results[0]["source"] == "pictet.pdf"


def test_add_is_idempotent():
    chunks = [Document(
        page_content="Même chunk deux fois.",
        metadata={"source": "lazard.pdf", "page": 0},
    )]
    embeddings = [[0.3] * config.EMBEDDING_DIM]
    vs.add(chunks, embeddings)
    vs.add(chunks, embeddings)
    results = vs.query([0.3] * config.EMBEDDING_DIM, k=5)
    assert len(results) == 1
