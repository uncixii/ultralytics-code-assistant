"""Streamlit UI for the source-grounded Ultralytics Code Assistant."""

from __future__ import annotations

import streamlit as st

from code_assistant.answer import (
    OpenRouterClient,
    answer_question,
    split_answer_status,
)
from code_assistant.config import load_settings
from code_assistant.embedder import CodeEmbedder
from code_assistant.errors import error_message
from code_assistant.retrieval import Evidence, select_evidence
from code_assistant.source import SOURCE_REVISION
from code_assistant.vector_store import AtlasVectorStore

st.set_page_config(page_title="Ultralytics Code Assistant", page_icon="🔎")
st.title("Ultralytics Code Assistant")
st.caption(
    "Answers grounded in the pinned Ultralytics models, engine and data Python source."
)


@st.cache_resource(show_spinner="Loading the local embedding model…")
def resources() -> tuple:
    settings = load_settings()
    settings.require_mongodb_uri()
    embedder = CodeEmbedder(settings)
    store = AtlasVectorStore(
        settings, dimensions=embedder.dimensions, code_filters=True
    )
    store.ping()
    count = store.count_code_chunks(
        source_revision=SOURCE_REVISION, model_key=embedder.model_key
    )
    return settings, embedder, store, count


def show_sources(sources: list[Evidence]) -> None:
    for source in sources:
        with st.expander(f"[{source.number}] `{source.citation}` · `{source.symbol}`"):
            st.code(source.code, language="python")
            st.link_button("View pinned source", source.source_url)


def show_answer(text: str, status: str) -> None:
    if status == "insufficient_evidence":
        st.info(text)
    else:
        st.markdown(text)


try:
    settings, embedder, store, indexed_chunks = resources()
    st.sidebar.success(f"Indexed chunks: {indexed_chunks}")
    st.sidebar.caption(f"Source revision: `{SOURCE_REVISION[:12]}`")
    st.sidebar.caption(f"Embedding: `{settings.embedding_model}`")
    if not indexed_chunks:
        st.warning(
            "No chunks found for this source and embedding model. Run the source indexer first."
        )
        st.stop()
except Exception as exc:
    st.error(f"Initialization failed: {error_message(exc)}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            show_answer(message["content"], message.get("status", "answered"))
        else:
            st.markdown(message["content"])
        if message.get("sources"):
            show_sources(message["sources"])

question = st.chat_input("Ask about the indexed Ultralytics source")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        try:
            evidence = []
            with st.spinner("Finding source evidence…"):
                evidence = select_evidence(
                    question,
                    store=store,
                    embedder=embedder,
                    source_revision=SOURCE_REVISION,
                )
            if evidence:
                response_parts = []

                def show_token(token: str) -> None:
                    response_parts.append(token)
                    response = "".join(response_parts)
                    # Wait for the full status line so protocol fragments stay hidden.
                    if "\n" in response:
                        _, text = split_answer_status(response)
                        if text:
                            answer_placeholder.markdown(text + " ▌")

                with st.spinner("Generating a grounded answer…"):
                    client = OpenRouterClient(
                        settings.openrouter_api_key or "",
                        settings.openrouter_model or "",
                        stream=True,
                        on_token=show_token,
                    )
                    result = answer_question(question, evidence, client)
            else:
                result = answer_question(question, [], None)
            with answer_placeholder.container():
                show_answer(result.text, result.status)
            show_sources(evidence)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result.text,
                    "status": result.status,
                    "sources": evidence,
                }
            )
        except Exception as exc:
            answer_placeholder.empty()
            st.error(f"Request failed: {error_message(exc)}")
            if "evidence" in locals() and evidence:
                st.info("Source evidence remains available below.")
                show_sources(evidence)
