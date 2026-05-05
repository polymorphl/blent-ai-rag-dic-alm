from pathlib import Path
from src import config
from src.ingestion.pdf_parser import parse_pdfs
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import embed_documents
from src.ingestion import vector_store


def run_ingestion(pdf_dir: Path = Path(config.PDF_DIR)) -> None:
    """Parse, chunk, embed and index all PDFs from pdf_dir into ChromaDB."""
    if not pdf_dir.is_dir():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir} — unzip DIC.zip into seed/DIC first")
    if not any(pdf_dir.glob("*.pdf")):
        raise FileNotFoundError(f"No PDF files found in {pdf_dir}")
    print(f"📁 Parsing PDFs from {pdf_dir} ...")
    docs = parse_pdfs(pdf_dir)
    print(f"📄 {len(docs)} pages parsed")

    chunks = chunk_documents(docs)
    print(f"📦 {len(chunks)} chunks created")

    texts = [c.page_content for c in chunks]
    embeddings = embed_documents(texts)
    print(f"🤖 {len(embeddings)} embeddings computed")

    vector_store.add(chunks, embeddings)
    print("✅ Done -- Indexed in ChromaDB")
