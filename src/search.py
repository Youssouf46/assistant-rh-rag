"""Recherche sémantique : retrouver les morceaux les plus proches d'une question (sans LLM).
"""
import sys

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

import config


def get_store() -> Chroma:
    """Ouvre la base Chroma"""
    return Chroma(
        collection_name=config.COLLECTION,
        embedding_function=OllamaEmbeddings(model=config.EMBED_MODEL),
        persist_directory=str(config.CHROMA_DIR),
    )


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or input("Votre question : ")
    results = get_store().similarity_search_with_score(question, k=config.TOP_K)

    for rank, (chunk, distance) in enumerate(results, start=1):
        print(f"\n#{rank}  distance {distance:.3f}  -  {chunk.metadata['title']}")
        print(chunk.page_content[:300])
