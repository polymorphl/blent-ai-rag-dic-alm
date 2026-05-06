from langchain_core.documents import Document
from src.ingestion.chunker import chunk_documents


def _make_doc(text: str, source: str = "test.pdf") -> Document:
    return Document(page_content=text, metadata={"source": source, "page": 0})


def test_chunks_respect_size_limit():
    long_text = "Ceci est une phrase test. " * 200  # ~5200 chars
    chunks = chunk_documents([_make_doc(long_text)])
    assert all(len(c.page_content) <= 1000 for c in chunks)


def test_produces_multiple_chunks():
    long_text = "Ceci est une phrase test. " * 200
    chunks = chunk_documents([_make_doc(long_text)])
    assert len(chunks) > 1


def test_metadata_source_propagated():
    long_text = "Ceci est une phrase test. " * 200
    chunks = chunk_documents([_make_doc(long_text, source="allianz.pdf")])
    assert all(c.metadata["source"] == "allianz.pdf" for c in chunks)


def test_short_document_is_single_chunk():
    text = "Document très court."
    chunks = chunk_documents([_make_doc(text)])
    assert len(chunks) == 1
    assert chunks[0].page_content == text
