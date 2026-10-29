"""Indexation : charger les sources, les découper, les stocker dans Chroma.
"""
import re
import shutil
import time

import requests
from bs4 import BeautifulSoup
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

# Blocs de la page qui ne sont pas du contenu (partage, formulaires, infobulles…)
NOISE = ".rs_skip, .fr-tooltip, .fr-tabs__list, .sp-panel-opened"
BLOCKS = ["h1", "h2", "h3", "h4", "p", "li", "td", "th"]


def clean_text(article) -> str:
    """Retire le bruit et renvoie le texte avec un paragraphe par ligne."""
    for noise in article.select(NOISE):
        noise.decompose()
    for block in article.find_all(BLOCKS):
        block.append("\n")
    text = article.get_text()
    text = re.sub(r"[ \t\xa0]+", " ", text)   # espaces multiples -> un espace
    text = re.sub(r" *\n[\n ]*", "\n", text)  # lignes vides -> un seul saut de ligne
    return text.strip()


def load_web_pages():
    lines = config.SOURCES_FILE.read_text(encoding="utf-8").splitlines()
    urls = [line.strip() for line in lines if line.strip() and not line.startswith("#")]

    pages = []
    for url in urls:
        html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
        soup = BeautifulSoup(html, "html.parser")

        title = soup.title.get_text(strip=True).split("|")[0].strip()
        text = clean_text(soup.find("article", class_="article"))

        pages.append({"url": url, "title": title, "text": text})
        print(f"{title} - {len(text)} caractères")

    return pages


def split_pages(pages) -> list[Document]:
    """Transforme les pages en Documents LangChain, puis les découpe en morceaux."""
    docs = [
        Document(page_content=page["text"], metadata={"source": page["url"], "title": page["title"]})
        for page in pages
    ]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        chunk.page_content = f"{chunk.metadata['title']}\n{chunk.page_content}"
    return chunks


def index_chunks(chunks) -> Chroma:
    """Calcule l'embedding de chaque morceau et les stocke dans Chroma (sur le disque)."""
    if config.CHROMA_DIR.exists():
        shutil.rmtree(config.CHROMA_DIR)  # on repart de zéro pour éviter les doublons
    return Chroma.from_documents(
        chunks,
        OllamaEmbeddings(model=config.EMBED_MODEL),
        collection_name=config.COLLECTION,
        persist_directory=str(config.CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


if __name__ == "__main__":
    print("1/3 Chargement des pages...")
    pages = load_web_pages()

    print("\n2/3 Découpage...")
    chunks = split_pages(pages)
    print(f"{len(pages)} pages -> {len(chunks)} morceaux")

    print("\n3/3 Calcul des embeddings et indexation...")
    start = time.time()
    store = index_chunks(chunks)
    print(f"{len(store.get()['ids'])} morceaux indexés en {time.time() - start:.0f} s dans {config.CHROMA_DIR}")
