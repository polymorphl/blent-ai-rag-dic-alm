from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from src.rag import prompt
from src import config


def test_system_instruction_always_present():
    result = prompt.build("question", [], [])
    assert isinstance(result[0], SystemMessage)
    assert "ALM" in result[0].content


def test_question_is_last_human_message():
    result = prompt.build("Quel est le SRI ?", [], [])
    assert isinstance(result[-1], HumanMessage)
    assert "Quel est le SRI ?" in result[-1].content


def test_chunk_text_and_source_are_in_system_message():
    chunks = [{"text": "Le SRI est 3.", "source": "Allianz.pdf", "metadata": {"page": 3}}]
    result = prompt.build("q", [], chunks)
    assert "Allianz.pdf" in result[0].content
    assert "Le SRI est 3." in result[0].content


def test_history_turns_produce_message_pairs():
    history = [
        {"role": "user", "content": "Question précédente"},
        {"role": "assistant", "content": "Réponse précédente"},
    ]
    result = prompt.build("nouvelle question", history, [])
    assert len(result) == 4  # system + user + assistant + current question
    assert isinstance(result[1], HumanMessage)
    assert "Question précédente" in result[1].content
    assert isinstance(result[2], AIMessage)
    assert "Réponse précédente" in result[2].content


def test_history_truncated_to_max_turns():
    history = []
    for i in range(config.HISTORY_MAX_TURNS + 3):
        history.append({"role": "user", "content": f"TURN_{i}_question"})
        history.append({"role": "assistant", "content": f"TURN_{i}_réponse"})
    result = prompt.build("nouvelle", history, [])
    content = " ".join(m.content for m in result)
    assert "TURN_0_" not in content
    assert "TURN_1_" not in content
    assert "TURN_2_" not in content
    assert "TURN_3_" in content
    assert f"TURN_{config.HISTORY_MAX_TURNS + 2}_" in content


def test_role_mapping():
    history = [
        {"role": "user", "content": "bonjour"},
        {"role": "assistant", "content": "salut"},
    ]
    result = prompt.build("question", history, [])
    assert isinstance(result[1], HumanMessage)
    assert isinstance(result[2], AIMessage)


def test_no_history_messages_when_history_empty():
    result = prompt.build("question", [], [])
    assert len(result) == 2  # system + current question only
