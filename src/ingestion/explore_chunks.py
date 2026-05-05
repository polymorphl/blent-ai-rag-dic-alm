import sys
from pathlib import Path
from src import config
from src.ingestion.pdf_parser import parse_pdfs
from src.ingestion.chunker import chunk_documents


def explore(target: Path) -> None:
    """Print chunk stats and 3 sample chunks for a PDF file or a directory of PDFs."""
    if target.is_file():
        docs = parse_pdfs(target.parent)
        docs = [d for d in docs if d.metadata["source"] == target.name]
    else:
        docs = parse_pdfs(target)

    chunks = chunk_documents(docs)
    sizes = [len(c.page_content) for c in chunks]

    print(f"📁 Target       : {target}")
    print(f"📄 Pages parsed : {len(docs)}")
    print(f"📦 Chunks       : {len(chunks)}")
    print(f"📏 Size min/max/avg : {min(sizes)} / {max(sizes)} / {int(sum(sizes)/len(sizes))}")
    print()

    for i, chunk in enumerate(chunks[:3]):
        print(f"--- Chunk {i + 1} (source={chunk.metadata['source']}, page={chunk.metadata['page']}) ---")
        print(chunk.page_content[:400])
        print()


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(config.PDF_DIR)
    explore(path)
