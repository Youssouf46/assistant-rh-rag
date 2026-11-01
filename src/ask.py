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
Règles :
- Réponds en français, de façon claire et concise.
- Utilise UNIQUEMENT les informations du contexte ci-dessous, jamais tes connaissances personnelles.
- Si le contexte ne contient pas la réponse, réponds exactement : "Je ne trouve pas cette information dans mes sources."
- Après chaque information, cite le numéro du passage utilisé entre crochets, par exemple [2].

Contexte :
{context}"""),
    ("human", "{question}"),
])


def format_context(chunks) -> str:
    """Numérote les morceaux pour que le modèle puisse les citer : [1], [2]..."""
    return "\n\n".join(f"[{i}] {chunk.page_content}" for i, chunk in enumerate(chunks, start=1))


def ask(question: str):
    # R = Retrieval : les morceaux les plus proches de la question
    chunks = get_store().similarity_search(question, k=config.TOP_K)

    # A + G = Augmented Generation : le prompt enrichi, puis la réponse de Mistral
    llm = ChatOllama(model=config.LLM_MODEL, temperature=0)
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
