"""Évaluation : mesurer la recherche et les réponses sur un jeu de questions de référence.
"""
import json
import re
import sys
import time
from datetime import date

import config
from agent import NOT_FOUND, agent
from search import get_store, hybrid_search, keyword_search, search_and_rerank

EVAL_DIR = config.ROOT / "data" / "eval"

# Les 4 façons de chercher que l'on compare
RETRIEVERS = {
    "Vecteurs": lambda question: get_store().similarity_search(question, k=config.TOP_K),
    "BM25": lambda question: keyword_search(question, k=config.TOP_K),
    "Hybride": lambda question: hybrid_search(question),
    "Hybride + reranking": lambda question: search_and_rerank(question),
}


def normalize(text: str) -> str:
    """Minuscules, espaces simples, « 25 % » = « 25% » : pour comparer des textes sans piège de mise en forme."""
    return re.sub(r"\s+", " ", text.lower()).replace(" %", "%")


def rank_of_passage(chunks, passage: str):
    """Rang (1, 2, 3…) du premier morceau qui contient le passage attendu, ou None s'il est absent."""
    return next((rank for rank, chunk in enumerate(chunks, start=1) if passage in chunk.page_content), None)


def evaluate_retrieval(items: list[dict]) -> dict:
    """Pour chaque méthode : le passage attendu est-il dans les 4 morceaux envoyés au LLM, et à quel rang ?"""
    print(f"\n=== Recherche ({len(items)} questions) ===")
    print(f"{'Méthode':22} {'Rappel@4':>9} {'MRR':>6}   Questions ratées")
    scores = {}
    for name, retrieve in RETRIEVERS.items():
        ranks = [rank_of_passage(retrieve(item["question"]), item["passage"]) for item in items]
        recall = sum(rank is not None for rank in ranks) / len(ranks)
        mrr = sum(1 / rank for rank in ranks if rank) / len(ranks)
        missed = [i + 1 for i, rank in enumerate(ranks) if rank is None]
        scores[name] = {"rappel@4": round(recall, 3), "mrr": round(mrr, 3), "ratees": missed}
        print(f"{name:22} {recall:>8.0%} {mrr:>6.2f}   {missed or '-'}")
    return scores


def evaluate_refusal(items: list[dict]) -> dict:
    """Le seuil de pertinence sépare-t-il bien les questions hors sujet des autres ?"""
    print(f"\n=== Refus hors sujet (seuil {config.RELEVANCE_THRESHOLD}) ===")
    errors = []
    for item in items:
        best_score = search_and_rerank(item["question"])[0].metadata["score"]
        refused = best_score < config.RELEVANCE_THRESHOLD
        if refused != item.get("hors_sujet", False):
            errors.append(f"{item['question']} (note {best_score:.3f})")
    accuracy = 1 - len(errors) / len(items)
    print(f"Exactitude : {accuracy:.0%} ({len(items) - len(errors)}/{len(items)})")
    for error in errors:
        print(f"  ❌ {error}")
    return {"exactitude": round(accuracy, 3), "erreurs": errors}


def evaluate_answers(items: list[dict]) -> dict:
    """Réponses complètes de l'agent, avec vérification : faits attendus présents ? phrases fidèles aux sources ?"""
    print(f"\n=== Réponses ({len(items)} questions, mode vérifié) ===")
    details = []
    for i, item in enumerate(items, start=1):
        start = time.time()
        result = agent.invoke({"question": item["question"], "history": [], "verify": True})
        answer = normalize(result["answer"])
        if item.get("hors_sujet"):
            correctness = 1.0 if result["answer"] == NOT_FOUND else 0.0
            faithfulness = None
        else:
            found = [fact for fact in item["faits"] if normalize(fact) in answer]
            correctness = len(found) / len(item["faits"])
            checks = result.get("checks", [])
            faithfulness = sum(check["justifiee"] for check in checks) / len(checks) if checks else None
        details.append({"question": item["question"], "exactitude": correctness, "fidelite": faithfulness,
                        "secondes": round(time.time() - start), "reponse": result["answer"],
                        "phrases_douteuses": [c["phrase"] for c in result.get("checks", []) if not c["justifiee"]]})
        fid = "-" if faithfulness is None else f"{faithfulness:.0%}"
        print(f"{i:>2}. exactitude {correctness:>4.0%}  fidélité {fid:>4}  ({time.time() - start:.0f} s)  {item['question']}")

    in_scope = [d for d in details if d["fidelite"] is not None]
    summary = {
        "exactitude": round(sum(d["exactitude"] for d in details) / len(details), 3),
        "fidelite": round(sum(d["fidelite"] for d in in_scope) / len(in_scope), 3) if in_scope else None,
        "secondes_par_question": round(sum(d["secondes"] for d in details) / len(details)),
    }
    print(f"\nMoyennes : exactitude {summary['exactitude']:.0%}, fidélité {summary['fidelite']:.0%}, "
          f"{summary['secondes_par_question']} s par question")
    return {"moyennes": summary, "details": details}


if __name__ == "__main__":
    items = json.loads((EVAL_DIR / "questions.json").read_text(encoding="utf-8"))
    in_scope = [item for item in items if not item.get("hors_sujet")]

    report = {"date": str(date.today()), "modele": config.LLM_MODEL, "reranker": config.RERANK_MODEL}
    report["recherche"] = evaluate_retrieval(in_scope)
    report["refus"] = evaluate_refusal(items)
    if "--reponses" in sys.argv:
        report["reponses"] = evaluate_answers(items)

    output = EVAL_DIR / f"resultats_{date.today()}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRésultats enregistrés dans {output}")