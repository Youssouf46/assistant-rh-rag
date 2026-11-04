"""Paramètres centraux du projet : on les modifie ici, pas dans le code."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
SOURCES_FILE = ROOT / "data" / "sources.txt"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION = "droit_travail"

# Modèles Ollama (100 % local)
LLM_MODEL = "ministral-3:8b"     # génère les réponses
EMBED_MODEL = "bge-m3"    # transforme le texte en vecteurs (très bon en français)

# Découpage des documents
CHUNK_SIZE = 1000         # taille d'un morceau (en caractères)
CHUNK_OVERLAP = 150       # chevauchement pour ne pas couper une idée en deux

# Recherche
TOP_K = 4                 # nombre de morceaux envoyés au modèle
CANDIDATES = 10           # morceaux pris dans chaque recherche avant la fusion
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"   # cross-encoder multilingue (plus rapide : "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
RERANK_CANDIDATES = 10    # morceaux issus de la fusion, relus par le reranker
RELEVANCE_THRESHOLD = 0.05   # note minimale du meilleur morceau (reranker) pour répondre ; sinon refus sans appeler le LLM