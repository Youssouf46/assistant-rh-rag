# Assistant RH souverain : RAG 100 % local sur le droit du travail français

Assistant qui répond aux questions des salariés et des services RH (congés payés, heures supplémentaires, télétravail, rupture conventionnelle…) à partir de **sources officielles**, en **citant ses sources**.

Tout tourne **en local** avec des modèles open source (Ollama) : aucune donnée ne quitte la machine, ce qui répond aux contraintes du **RGPD** et de la souveraineté des données.

> 🚧 Projet en cours de construction. Ce README est complété à chaque étape.

---

## Fonctionnement (RAG : *Retrieval-Augmented Generation*)

```
 Sources officielles            INDEXATION (une fois)                    QUESTION (à chaque fois)
 (service-public.fr,      ┌──────────────────────────────┐        ┌───────────────────────────────┐
  Code du travail, PDF)──▶│ charger → découper en        │        │ question                      │
                          │ morceaux → embeddings (bge-m3)│        │   → embedding (bge-m3)        │
                          │ → base vectorielle Chroma ────┼──────▶ │   → recherche des morceaux    │
                          └──────────────────────────────┘        │     les plus proches          │
                                                                  │   → prompt + morceaux         │
                                                                  │   → Mistral (Ollama)          │
                                                                  │   → réponse + sources [1][2]  │
                                                                  └───────────────────────────────┘
```

## Stack technique

| Brique | Outil |
|---|---|
| LLM local | [Ollama](https://ollama.com) + **Mistral 7B** |
| Embeddings | **bge-m3** (multilingue, via Ollama) |
| Orchestration RAG | **LangChain** |
| Base vectorielle | **ChromaDB** |
| Sources | Fiches officielles [service-public.fr](https://www.service-public.fr) (droit du travail, secteur privé) |

## Installation

Prérequis : Python 3.12, [Ollama](https://ollama.com/download), environ 8 Go de RAM libre.

```bash
# 1. Modèles locaux
ollama pull mistral
ollama pull bge-m3

# 2. Environnement Python
python -m venv .venv
.venv\Scripts\activate          # Windows (Linux/macOS : source .venv/bin/activate)
pip install -r requirements.txt
```

## Structure du projet

```
├── data/
│   ├── sources.txt     # URL des pages officielles à indexer
│   └── raw/            # PDF à indexer (optionnel)
├── src/
│   ├── config.py       # paramètres : modèles, taille des morceaux, nombre de résultats
│   └── ingest.py       # chargement et nettoyage des sources (puis découpage et indexation)
├── requirements.txt
└── README.md
```

## Feuille de route

- [x] Installation : Ollama, modèles, environnement Python
- [x] Prise en main d'Ollama (LLM et embeddings)
- [ ] Chargement et découpage des documents
- [ ] Embeddings et base vectorielle Chroma
- [ ] Chaîne RAG avec citations des sources
- [ ] Interface Streamlit
- [ ] Recherche hybride (BM25 + vecteurs) et reranking
- [ ] Agent LangGraph (reformulation, vérification, refus hors sujet)
- [ ] Évaluation avec RAGAS
- [ ] API FastAPI, Docker, CI GitHub Actions

## Avertissement

Projet de démonstration : les réponses ne remplacent pas un conseil juridique. Toujours vérifier la source citée.
