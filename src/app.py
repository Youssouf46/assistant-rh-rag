"""Interface web de l'assistant RH (chat).
"""
import streamlit as st

import config
from ask import build_chain, format_context, retrieve

st.set_page_config(page_title="Assistant RH souverain", page_icon="⚖️")
st.title("⚖️ Assistant RH souverain")
st.caption(f"Droit du travail, secteur privé · Sources : service-public.fr · Modèle local : {config.LLM_MODEL}")


def show_sources(chunks):
    """Affiche les sources dans un bloc dépliable, avec un extrait de chaque morceau."""
    with st.expander("📚 Sources"):
        for i, chunk in enumerate(chunks, start=1):
            st.markdown(f"**[{i}]** [{chunk.metadata['title']}]({chunk.metadata['source']})")
            st.caption(chunk.page_content[:300] + "…")


# La mémoire de la conversation (conservée entre deux relances du script)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 1. Réafficher l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "chunks" in message:
            show_sources(message["chunks"])

# 2. Traiter une nouvelle question
if question := st.chat_input("Posez votre question sur le droit du travail…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche dans les sources…"):
            chunks = retrieve(question)
        stream = build_chain().stream({"context": format_context(chunks), "question": question})
        answer = st.write_stream(stream)
        show_sources(chunks)

    st.session_state.messages.append({"role": "assistant", "content": answer, "chunks": chunks})
