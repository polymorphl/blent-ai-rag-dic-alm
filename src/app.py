import streamlit as st
from src import config
from src.rag.engine import RagEngine


@st.cache_resource
def _get_engine() -> RagEngine:
    from src.ingestion import embedder
    embedder.warm_up()
    return RagEngine()


def _render_sources(sources: list[str]) -> None:
    lines = [config.APP_SOURCES_PREFIX]
    for source in sources:
        filename = source.split(" (p.")[0]
        url = f"/app/static/{filename}"
        lines.append(f"- [{source}]({url})")
    st.caption("\n".join(lines))


def main() -> None:
    st.title(config.APP_TITLE)

    with st.spinner(config.APP_LOADING_TEXT):
        _get_engine()

    if "history" not in st.session_state:
        st.session_state.history = []

    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                _render_sources(msg["sources"])

    question = st.chat_input(config.APP_INPUT_PLACEHOLDER)
    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner(config.APP_SPINNER_TEXT):
                result = _get_engine().ask(question, st.session_state.history)
            st.write(result["answer"])
            if result["sources"]:
                _render_sources(result["sources"])

        st.session_state.history.append({"role": "user", "content": question})
        st.session_state.history.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        })


main()
