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



def format_context(chunks) -> str:
    """Numérote les morceaux pour que le modèle puisse les citer : [1], [2]..."""
    return "\n\n".join(f"[{i}] {chunk.page_content}" for i, chunk in enumerate(chunks, start=1))


def ask(question: str, model: str = config.LLM_MODEL):
    # R = Retrieval : les morceaux les plus proches de la question
    chunks = get_store().similarity_search(question, k=config.TOP_K)

    # A + G = Augmented Generation : le prompt enrichi, puis la réponse du modèle
    llm = ChatOllama(model=model, temperature=0)
    chain = PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": format_context(chunks), "question": question})
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
