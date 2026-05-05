import re
from pathlib import Path
import fitz
from langchain_core.documents import Document


def parse_pdfs(pdf_dir: Path) -> list[Document]:
    """Extract text from all PDFs in pdf_dir, one Document per page with source and page metadata."""
    docs = []
    for pdf_path in sorted(Path(pdf_dir).glob("*.pdf")):
        with fitz.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf):
                text = _clean_text(page.get_text())
                if text:
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": pdf_path.name, "page": page_num},
                    ))
    return docs


def _clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()
