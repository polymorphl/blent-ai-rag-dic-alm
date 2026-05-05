# Chunking — increase CHUNK_SIZE for more context per chunk, increase CHUNK_OVERLAP to reduce boundary cuts
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200

# Embedding — multilingual E5 model, runs locally via sentence-transformers (~1.2 GB download on first use)
EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"
EMBEDDING_DIM: int = 1024  # output dimension of EMBEDDING_MODEL

# Vector store — ChromaDB persisted to disk; delete CHROMA_DIR to force a full re-index
CHROMA_DIR: str = "data/chroma_db"
CHROMA_COLLECTION: str = "dic_alm"

# Source documents
PDF_DIR: str = "seed/DIC"
