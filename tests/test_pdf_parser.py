import fitz
import pytest
from langchain_core.documents import Document
from src.ingestion.pdf_parser import parse_pdfs


@pytest.fixture
def sample_pdf_dir(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 100), "Ceci est un document de test.\nIl contient plusieurs lignes.")
    doc.save(pdf_path)
    doc.close()
    return tmp_path


def test_parse_returns_documents(sample_pdf_dir):
    docs = parse_pdfs(sample_pdf_dir)
    assert len(docs) > 0
    assert all(isinstance(d, Document) for d in docs)


def test_parse_metadata(sample_pdf_dir):
    docs = parse_pdfs(sample_pdf_dir)
    for doc in docs:
        assert "source" in doc.metadata
        assert "page" in doc.metadata


def test_parse_text_non_empty(sample_pdf_dir):
    docs = parse_pdfs(sample_pdf_dir)
    assert all(doc.page_content.strip() for doc in docs)


def test_parse_source_is_filename(sample_pdf_dir):
    docs = parse_pdfs(sample_pdf_dir)
    assert docs[0].metadata["source"] == "sample.pdf"
