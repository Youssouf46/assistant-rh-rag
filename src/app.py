"""Interface web de l'assistant RH (chat).
"""
from itertools import chain

import streamlit as st

import config
from agent import stream_agent

st.set_page_config(page_title="Assistant RH souverain", page_icon="⚖️")
st.title("⚖️ Assistant RH souverain")
st.caption(f"Droit du travail, secteur privé · Sources : service-public.fr · Modèle local : {config.LLM_MODEL}")


def show_sources(chunks):
    """Affiche les sources dans un bloc dépliable, avec un extrait de chaque morceau."""
    with st.expander("📚 Sources"):
        for i, chunk in enumerate(chunks, start=1):
            st.markdown(f"**[{i}]** [{chunk.metadata['title']}]({chunk.metadata['source']}) · pertinence {chunk.metadata['score']:.2f}")
            st.caption(chunk.page_content[:300] + "…")


# La mémoire de la conversation (conservée entre deux relances du script)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 1. Réafficher l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("search_question"):
            st.caption(f"🔎 Question comprise : {message['search_question']}")
        st.markdown(message["content"])
        if message.get("chunks"):
            show_sources(message["chunks"])


# 2. Traiter une nouvelle question
if question := st.chat_input("Posez votre question sur le droit du travail…"):
    # Les 2 derniers échanges (4 messages), pour comprendre les questions de suite
    history = [(m["role"], m["content"]) for m in st.session_state.messages[-4:]]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        state = {}
        tokens = stream_agent(question, history, state)
        with st.spinner("Recherche dans les sources…"):
            first_token = next(tokens)   # l'agent reformule et cherche, jusqu'au premier mot de la réponse
        search_question = state["search_question"]
        if search_question != question:
            st.caption(f"🔎 Question comprise : {search_question}")
        answer = st.write_stream(chain([first_token], tokens))
        if state["chunks"]:
            show_sources(state["chunks"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "chunks": state["chunks"],
        "search_question": search_question if search_question != question else None,
    })