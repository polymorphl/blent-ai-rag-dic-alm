from unittest.mock import MagicMock, patch
from src import config
from src.rag.engine import RagEngine, _format_chunk_meta

_CHUNKS = [
    {"text": "Le SRI est 3.", "source": "Allianz.pdf", "score": 0.9, "metadata": {"page": 2}},
    {"text": "Frais de 1%.", "source": "BNP.pdf", "score": 0.8, "metadata": {"page": 5}},
    {"text": "Autre info.", "source": "Allianz.pdf", "score": 0.7, "metadata": {"page": 4}},
]


def test_format_chunk_meta_returns_source_and_1based_page():
    chunk = {"source": "Allianz.pdf", "metadata": {"page": 2}}
    source, page = _format_chunk_meta(chunk)
    assert source == "Allianz.pdf"
    assert page == "3"


def test_format_chunk_meta_missing_source_falls_back_to_inconnu():
    chunk = {"metadata": {"page": 0}}
    source, _ = _format_chunk_meta(chunk)
    assert source == "inconnu"


def test_format_chunk_meta_missing_page_falls_back_to_question_mark():
    chunk = {"source": "doc.pdf", "metadata": {}}
    _, page = _format_chunk_meta(chunk)
    assert page == "?"


def test_format_chunk_meta_non_int_page_falls_back_to_question_mark():
    chunk = {"source": "doc.pdf", "metadata": {"page": "3"}}
    _, page = _format_chunk_meta(chunk)
    assert page == "?"


@patch("src.rag.engine.vector_store.query", return_value=_CHUNKS)
@patch("src.rag.engine.embedder.embed_query", return_value=[0.1] * 1024)
@patch("src.rag.engine.ChatOllama")
def test_ask_returns_answer_and_sources(mock_cls, mock_embed, mock_query):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Le SRI d'Allianz est 3.")
    mock_cls.return_value = mock_llm

    result = RagEngine().ask("Quel est le SRI ?", [])

    assert result["answer"] == "Le SRI d'Allianz est 3."
    assert any("Allianz.pdf" in s for s in result["sources"])
    assert any("BNP.pdf" in s for s in result["sources"])


@patch("src.rag.engine.vector_store.query", return_value=_CHUNKS)
@patch("src.rag.engine.embedder.embed_query", return_value=[0.1] * 1024)
@patch("src.rag.engine.ChatOllama")
def test_ask_deduplicates_sources(mock_cls, mock_embed, mock_query):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="réponse")
    mock_cls.return_value = mock_llm

    result = RagEngine().ask("question", [])

    allianz = [s for s in result["sources"] if "Allianz.pdf" in s]
    assert len(allianz) == 1


@patch("src.rag.engine.vector_store.query", return_value=[])
@patch("src.rag.engine.embedder.embed_query", return_value=[0.1] * 1024)
@patch("src.rag.engine.ChatOllama")
def test_ask_empty_chunks_returns_no_sources(mock_cls, mock_embed, mock_query):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Je ne sais pas.")
    mock_cls.return_value = mock_llm

    result = RagEngine().ask("question hors-sujet", [])

    assert result["sources"] == []
    assert result["answer"] == "Je ne sais pas."


@patch("src.rag.engine.vector_store.query", return_value=_CHUNKS)
@patch("src.rag.engine.embedder.embed_query", return_value=[0.1] * 1024)
@patch("src.rag.engine.ChatOllama")
def test_ask_ollama_unavailable_returns_error_message(mock_cls, mock_embed, mock_query):
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("connection refused")
    mock_cls.return_value = mock_llm

    result = RagEngine().ask("question", [])

    assert result["answer"] == config.LLM_UNAVAILABLE_MSG
    assert isinstance(result["sources"], list)


@patch("src.rag.engine.vector_store.query", return_value=_CHUNKS)
@patch("src.rag.engine.embedder.embed_query", return_value=[0.1] * 1024)
@patch("src.rag.engine.ChatOllama")
def test_ask_forwards_history_to_prompt(mock_cls, mock_embed, mock_query):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="réponse")
    mock_cls.return_value = mock_llm

    history = [
        {"role": "user", "content": "question précédente"},
        {"role": "assistant", "content": "réponse précédente"},
    ]

    with patch("src.rag.engine.prompt.build", wraps=lambda q, h, c: "prompt") as mock_build:
        RagEngine().ask("nouvelle question", history)
        _, call_history, _ = mock_build.call_args[0]
        assert call_history == history
