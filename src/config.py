"""Paramètres centraux du projet : on les modifie ici, pas dans le code."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
SOURCES_FILE = ROOT / "data" / "sources.txt"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION = "droit_travail"

# Modèles Ollama (100 % local)
LLM_MODEL = "mistral"     # génère les réponses
EMBED_MODEL = "bge-m3"    # transforme le texte en vecteurs (très bon en français)

# Découpage des documents
CHUNK_SIZE = 1000         # taille d'un morceau (en caractères)
CHUNK_OVERLAP = 150       # chevauchement pour ne pas couper une idée en deux

# Recherche
TOP_K = 4                 # nombre de morceaux envoyés au modèle
