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

# LLM — Ollama model tag; change to "llama3" etc. after pulling the alternative model
LLM_MODEL: str = "mistral"

# Retrieval — number of chunks fetched from ChromaDB per query
RAG_K: int = 5

# Conversation memory — max turns kept in the prompt (1 turn = 1 user + 1 assistant message)
HISTORY_MAX_TURNS: int = 10

# CLI messages
CLI_WELCOME: str = "Assistant ALM — tapez 'quit' pour quitter ou Ctrl+C pour interrompre."
CLI_GOODBYE: str = "Au revoir."
CLI_USER_PREFIX: str = "Vous"
CLI_ASSISTANT_PREFIX: str = "Assistant"
CLI_SOURCES_PREFIX: str = "Sources"
CLI_EXIT_COMMANDS: tuple[str, ...] = ("quit", "exit", "quitter")
CLI_SPINNER_TEXT: str = "Recherche en cours..."
CLI_LOADING_TEXT: str = "Chargement du modèle d'embedding..."

# Streamlit UI
APP_TITLE: str = "Assistant ALM — DIC"
APP_INPUT_PLACEHOLDER: str = "Posez votre question sur les DIC..."
APP_SPINNER_TEXT: str = "Recherche en cours..."
APP_SOURCES_PREFIX: str = "Sources : "
APP_LOADING_TEXT: str = "Chargement du modèle d'embedding..."

# Error messages
LLM_UNAVAILABLE_MSG: str = "Le modèle est actuellement indisponible. Vérifiez qu'Ollama est démarré."
