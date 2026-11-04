"""Interface web de l'assistant RH (chat).
"""
from itertools import chain, takewhile

import streamlit as st

import config
from agent import stream_agent

st.set_page_config(page_title="Assistant RH souverain", page_icon="⚖️")
st.title("⚖️ Assistant RH souverain")
st.caption(f"Droit du travail, secteur privé · Sources : service-public.fr · Modèle local : {config.LLM_MODEL}")
verify_mode = st.sidebar.toggle("🔍 Mode vérifié", help="Vérifie chaque phrase citée contre sa source (1 à 2 min de plus sur CPU).")


def show_sources(chunks):
    """Affiche les sources dans un bloc dépliable, avec un extrait de chaque morceau."""
    with st.expander("📚 Sources"):
        for i, chunk in enumerate(chunks, start=1):
            st.markdown(f"**[{i}]** [{chunk.metadata['title']}]({chunk.metadata['source']}) · pertinence {chunk.metadata['score']:.2f}")
            st.caption(chunk.page_content[:300] + "…")


def show_checks(checks):
    """Affiche le résultat de la vérification : les phrases non justifiées par leur source."""
    doubtful = [check for check in checks if not check["justifiee"]]
    if not doubtful:
        st.success(f"✅ Vérifiée : les {len(checks)} phrases citées sont justifiées par leurs sources.")
        return
    st.warning(f"⚠️ {len(doubtful)} phrase(s) sur {len(checks)} à vérifier dans les sources :")
    for check in doubtful:
        sources = ", ".join(f"[{n}]" for n in check["sources"])
        reason = check["raison"].replace("*", "")
        if len(reason) > 200:
            reason = reason[:200].rsplit(" ", 1)[0] + "…"
        st.markdown(f"- « {check['phrase']} » {sources}  \n  *Piste du vérificateur : {reason}*")



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
        if message.get("checks"):
            show_checks(message["checks"])


# 2. Traiter une nouvelle question
if question := st.chat_input("Posez votre question sur le droit du travail…"):
    # Les 2 derniers échanges (4 messages), pour comprendre les questions de suite
    history = [(m["role"], m["content"]) for m in st.session_state.messages[-4:]]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        state = {}
        events = stream_agent(question, history, state, verify_mode)
        with st.spinner("Recherche dans les sources…"):
            first_token = next(events)   # l'agent reformule et cherche, jusqu'au premier mot de la réponse
        search_question = state["search_question"]
        if search_question != question:
            st.caption(f"🔎 Question comprise : {search_question}")
        # On affiche les mots jusqu'au signal de fin (None) ; l'agent reste en pause juste après
        answer = st.write_stream(takewhile(lambda token: token is not None, chain([first_token], events)))
        if state["chunks"]:
            show_sources(state["chunks"])
        if verify_mode and state["chunks"]:
            with st.spinner("Vérification de chaque phrase…"):
                for _ in events:         # on laisse l'agent terminer : le nœud verify s'exécute
                    pass
            show_checks(state.get("checks", []))

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "chunks": state["chunks"],
        "search_question": search_question if search_question != question else None,
        "checks": state.get("checks"),
    })