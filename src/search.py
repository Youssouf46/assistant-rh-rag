"""Recherche hybride : sens (vecteurs) + mots exacts (BM25), puis fusion des deux classements (sans LLM).
"""
import re
import sys
from functools import cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from rank_bm25 import BM25Okapi

import config

# Mots trop fréquents pour aider à trouver le bon morceau
STOPWORDS = set("""le la les l un une des de du d à au aux et ou en est sont pour par sur dans
que qui quel quelle quels quelles ce cette ces se sa son ses il elle ils je j tu vous nous on
ne pas plus a ai y s si mon ma mes m peut être avoir comment combien""".split())


def get_store() -> Chroma:
    """Ouvre la base Chroma"""
    return Chroma(
        collection_name=config.COLLECTION,
        embedding_function=OllamaEmbeddings(model=config.EMBED_MODEL),
        persist_directory=str(config.CHROMA_DIR),
    )


def tokenize(text: str) -> list[str]:
    """Découpe un texte en mots utiles : minuscules, sans ponctuation ni mots vides."""
    return [word for word in re.findall(r"\w+", text.lower()) if word not in STOPWORDS]


@cache
def get_bm25():
    """Construit l'index BM25 une seule fois, à partir des morceaux déjà stockés dans Chroma."""
    data = get_store().get()
    chunks = [Document(page_content=text, metadata=meta)
              for text, meta in zip(data["documents"], data["metadatas"])]
    return BM25Okapi([tokenize(chunk.page_content) for chunk in chunks]), chunks


def keyword_search(question: str, k: int) -> list[Document]:
    """Recherche par mots-clés (BM25) : les morceaux qui contiennent les mots de la question."""
    bm25, chunks = get_bm25()
    scores = bm25.get_scores(tokenize(question))
    best = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:k]
    return [chunks[i] for i in best]


def reciprocal_rank_fusion(rankings: list[list[Document]], c: int = 60) -> list[Document]:
    """Fusionne plusieurs classements : chaque morceau gagne 1 / (c + rang) dans chaque liste."""
    scores, by_text = {}, {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.page_content] = scores.get(chunk.page_content, 0) + 1 / (c + rank)
            by_text[chunk.page_content] = chunk
    best = sorted(scores, key=scores.get, reverse=True)
    return [by_text[text] for text in best]


def hybrid_search(question: str, k: int = config.TOP_K) -> list[Document]:
    vector_results = get_store().similarity_search(question, k=config.CANDIDATES)
    keyword_results = keyword_search(question, k=config.CANDIDATES)
    return reciprocal_rank_fusion([vector_results, keyword_results])[:k]


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or input("Votre question : ")
    results = {
        "Vecteurs": get_store().similarity_search(question, k=config.TOP_K),
        "Mots-clés (BM25)": keyword_search(question, k=config.TOP_K),
        "Hybride": hybrid_search(question),
    }
    for name, chunks in results.items():
        print(f"\n=== {name} ===")
        for rank, chunk in enumerate(chunks, start=1):
            print(f"#{rank}  {chunk.page_content[:150]!r}")
