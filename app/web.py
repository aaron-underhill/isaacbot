import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from config import (
    APP_VERSION,
    CHAT_MODEL,
    MAX_CHAT_HISTORY_MESSAGES,
    MAX_OUTPUT_TOKENS,
    PERSONA_NAME,
    SHOW_RETRIEVED_MEMORIES,
    TOP_K_MEMORIES,
    USER_NAME,
)
from persona import build_instructions
from rag import LocalRAGIndex


load_dotenv()

st.set_page_config(
    page_title=f"{PERSONA_NAME}Bot v{APP_VERSION}",
    page_icon="💬",
    layout="centered",
)

st.title(f"{PERSONA_NAME}Bot v{APP_VERSION}")
st.caption(
    f"A simulation based on historical {USER_NAME}–{PERSONA_NAME} conversations."
)

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not set. Copy .env.example to .env and add it.")
    st.stop()


@st.cache_resource
def get_backend():
    client = OpenAI()
    rag = LocalRAGIndex(client)
    return client, rag


def build_retrieval_query(user_text, history):
    recent = history[-6:]
    lines = [f"{item['role'].upper()}: {item['content']}" for item in recent]
    lines.append(f"USER: {user_text}")
    return "\n".join(lines)


client, rag = get_backend()

if "history" not in st.session_state:
    st.session_state.history = []

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []


with st.sidebar:
    st.header(f"{PERSONA_NAME}Bot")
    st.write(f"Model: `{CHAT_MODEL}`")

    if st.button("Clear conversation"):
        st.session_state.history = []
        st.session_state.last_sources = []
        st.rerun()

    # Retrieved windows are raw corpus text, so they are hidden by default.
    if SHOW_RETRIEVED_MEMORIES:
        st.divider()
        st.subheader("Retrieved memories")

        if not st.session_state.last_sources:
            st.caption("No memories retrieved yet.")
        else:
            for number, result in enumerate(st.session_state.last_sources, start=1):
                meta = result["metadata"]
                with st.expander(
                    f"{number}. score {result['score']:.3f} • session {meta['session_id']}"
                ):
                    st.text(meta["text"])


for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.write(message["content"])


user_text = st.chat_input(f"Message {PERSONA_NAME}Bot...")

if user_text:
    with st.chat_message("user"):
        st.write(user_text)

    retrieval_query = build_retrieval_query(user_text, st.session_state.history)
    sources = rag.search(retrieval_query, top_k=TOP_K_MEMORIES)
    st.session_state.last_sources = sources

    instructions = build_instructions(sources)

    model_input = (
        st.session_state.history[-MAX_CHAT_HISTORY_MESSAGES:]
        + [{"role": "user", "content": user_text}]
    )

    with st.chat_message("assistant"):
        with st.spinner(f"{PERSONA_NAME} is typing..."):
            response = client.responses.create(
                model=CHAT_MODEL,
                instructions=instructions,
                input=model_input,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )
            reply = (response.output_text or "").strip() or "(no text response)"
        st.write(reply)

    st.session_state.history.append({"role": "user", "content": user_text})
    st.session_state.history.append({"role": "assistant", "content": reply})
