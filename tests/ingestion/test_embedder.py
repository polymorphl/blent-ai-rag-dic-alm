from src import config
from src.ingestion.embedder import embed_query, embed_documents

EXPECTED_DIM = config.EMBEDDING_DIM


def test_embed_query_returns_correct_dimension():
    vec = embed_query("Quel est le risque de ce produit ?")
    assert len(vec) == EXPECTED_DIM


def test_embed_documents_returns_correct_dimension():
    vecs = embed_documents(["Produit financier à risque modéré."])
    assert len(vecs) == 1
    assert len(vecs[0]) == EXPECTED_DIM


def test_embed_documents_batch():
    texts = ["Premier document.", "Deuxième document.", "Troisième document."]
    vecs = embed_documents(texts)
    assert len(vecs) == 3
    assert all(len(v) == EXPECTED_DIM for v in vecs)
