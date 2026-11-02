"""Assistant RH : répondre à une question à partir des fiches officielles, en citant les sources.
"""
import sys
import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

import config
from search import get_store

PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Tu es un assistant RH spécialisé en droit du travail français (secteur privé).
Tu réponds uniquement à partir des passages numérotés du contexte ci-dessous.

Règles :
1. Commence par une réponse directe à la question (oui, non, un chiffre…), telle qu'elle découle des passages.
2. Donne ensuite la règle générale, puis les exceptions éventuelles.
3. Après chaque phrase, indique entre crochets le numéro du passage qui la justifie, par exemple [3].
4. N'ajoute aucune information, définition ou exemple qui ne figure pas dans les passages.
5. Si les passages ne permettent pas de répondre, écris uniquement : "Je ne trouve pas cette information dans mes sources." et rien d'autre.

Contexte :
{context}"""),
    ("human", "{question}"),
])

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Tu es un outil de reformulation de questions, pas un assistant : tu ne réponds jamais aux questions.
On te donne un extrait de conversation et une nouvelle question.
Réécris la nouvelle question pour qu'elle soit compréhensible seule :
- si elle fait référence à la conversation ("et si…", "et pour…", "dans ce cas"), ajoute le sujet manquant ;
- sinon, recopie-la sans la modifier.
N'ajoute aucun détail absent de la conversation (année, montant…).
Renvoie uniquement la question réécrite, en texte brut."""),
    ("human", "Conversation :\n{history}\n\nNouvelle question : {question}\n\nQuestion réécrite :"),
])


FOLLOW_UP_STARTS = ("et ", "et,", "dans ce cas", "alors", "mais ", "si ", "pour un", "pour une", "même ")


def is_follow_up(question: str) -> bool:
    """Une question de suite commence par « Et… », « Dans ce cas… »… ou est très courte."""
    q = question.lower().strip()
    return q.startswith(FOLLOW_UP_STARTS) or len(q.split()) <= 3


def condense_question(question: str, history: list[tuple[str, str]], model: str = config.LLM_MODEL) -> str:
    """Transforme une question de suite en question autonome, grâce à l'historique."""
    if not history or not is_follow_up(question):
        return question
    history_text = "\n".join(f"{role} : {content}" for role, content in history)
    llm = ChatOllama(model=model, temperature=0)
    chain = CONDENSE_PROMPT | llm | StrOutputParser()
    rewritten = chain.invoke({"history": history_text, "question": question}).replace("*", "").strip()
    # Garde-fou : une « reformulation » trop longue est sûrement une réponse : on garde la question d'origine
    return rewritten if len(rewritten) <= 200 else question




def format_context(chunks) -> str:
    """Numérote les morceaux pour que le modèle puisse les citer : [1], [2]..."""
    return "\n\n".join(f"[{i}] {chunk.page_content}" for i, chunk in enumerate(chunks, start=1))


def retrieve(question: str):
    """R = Retrieval : les morceaux les plus proches de la question."""
    return get_store().similarity_search(question, k=config.TOP_K)


def build_chain(model: str = config.LLM_MODEL):
    """A + G : le prompt enrichi, puis le modèle, puis l'extraction du texte."""
    llm = ChatOllama(model=model, temperature=0)
    return PROMPT | llm | StrOutputParser()


def ask(question: str, model: str = config.LLM_MODEL):
    chunks = retrieve(question)
    answer = build_chain(model).invoke({"context": format_context(chunks), "question": question})
    return answer, chunks



if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or input("Votre question : ")
    start = time.time()
    answer, chunks = ask(question)

    print(f"\n{answer}\n")
    print("Sources :")
    for i, chunk in enumerate(chunks, start=1):
        print(f"  [{i}] {chunk.metadata['title']} - {chunk.metadata['source']}")
    print(f"\n({time.time() - start:.0f} s)")
