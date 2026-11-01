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
| LLM local | [Ollama](https://ollama.com) + **Ministral 3 8B** (Mistral AI) |
| Embeddings | **bge-m3** (multilingue, via Ollama) |
| Orchestration RAG | **LangChain** |
| Base vectorielle | **ChromaDB** |
| Sources | Fiches officielles [service-public.fr](https://www.service-public.fr) (droit du travail, secteur privé) |

## Installation

Prérequis : Python 3.12, [Ollama](https://ollama.com/download), environ 8 Go de RAM libre.

```bash
# 1. Modèles locaux
ollama pull ministral-3:8b
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
│   ├── ingest.py       # chargement, nettoyage, découpage et indexation dans Chroma
│   ├── search.py       # recherche sémantique (sans LLM)
│   ├── ask.py          # assistant : recherche + LLM + réponse avec sources
│   └── compare.py      # comparaison de modèles sur les mêmes questions
├── requirements.txt
└── README.md
```

## Choix du modèle

Comparaison sur les mêmes questions, avec la même recherche, le même prompt et `temperature=0` (`python src/compare.py`) :

| Question | Mistral 7B | Ministral 3 8B |
|---|---|---|
| Combien de jours de congés payés ? | Réponse exacte mais noyée dans des informations hors sujet | ✅ 30 jours ouvrables, 2,5 jours par mois, exceptions sourcées |
| Mon employeur peut-il m'imposer le télétravail ? | ❌ « Oui », règle inventée | ✅ « Non, sauf circonstances exceptionnelles ou force majeure » |
| Montant du SMIC ? (absent des sources) | ❌ Répond à d'autres questions | ✅ « Je ne trouve pas cette information dans mes sources. » |
| Temps de réponse (CPU) | 49 à 72 s | 22 à 46 s |

→ **Ministral 3 8B** est retenu.

## Feuille de route

- [x] Installation : Ollama, modèles, environnement Python
- [x] Prise en main d'Ollama (LLM et embeddings)
- [x] Chargement, nettoyage et découpage des documents (136 morceaux, titre en tête de chaque morceau)
- [x] Embeddings et base vectorielle Chroma (recherche sémantique)
- [x] Chaîne RAG avec citations des sources
- [x] Amélioration du prompt et choix du modèle par comparaison
- [ ] Interface Streamlit
- [ ] Recherche hybride (BM25 + vecteurs) et reranking
- [ ] Agent LangGraph (reformulation, vérification, refus hors sujet)
- [ ] Évaluation avec RAGAS
- [ ] API FastAPI, Docker, CI GitHub Actions

## Avertissement

Projet de démonstration : les réponses ne remplacent pas un conseil juridique. Toujours vérifier la source citée.
