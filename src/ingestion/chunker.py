from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src import config


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks using recursive character splitting."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    return splitter.split_documents(docs)
