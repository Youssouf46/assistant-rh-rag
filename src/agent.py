"""Agent LangGraph : reformuler → chercher → répondre ou refuser → (si demandé) vérifier chaque phrase.
"""
import sys
import time
from typing import TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

import config
from ask import build_chain, condense_question, format_context, retrieve
from verify import check_claims, split_claims

NOT_FOUND = "Je ne trouve pas cette information dans mes sources."


class State(TypedDict):
    """Le « tableau blanc » partagé : chaque nœud lit ce qu'il lui faut et écrit son résultat."""
    question: str                      # question posée par l'utilisateur
    history: list[tuple[str, str]]     # derniers messages (rôle, texte)
    search_question: str               # question autonome, utilisée pour chercher et répondre
    chunks: list[Document]             # morceaux retenus après reranking
    answer: str                        # réponse finale
    verify: bool                       # l'utilisateur a-t-il demandé la vérification ?
    checks: list[dict]                 # résultat de la vérification, phrase par phrase


# --- Les nœuds : une fonction = une étape ---

def condense(state: State) -> dict:
    return {"search_question": condense_question(state["question"], state["history"])}


def search(state: State) -> dict:
    return {"chunks": retrieve(state["search_question"])}


def generate(state: State) -> dict:
    context = format_context(state["chunks"])
    return {"answer": build_chain().invoke({"context": context, "question": state["search_question"]})}


def refuse(state: State) -> dict:
    return {"answer": NOT_FOUND, "chunks": []}


def verify(state: State) -> dict:
    return {"checks": check_claims(split_claims(state["answer"]), state["chunks"])}


# --- L'aiguillage : le code choisit le nœud suivant ---

def route_after_search(state: State) -> str:
    """Le meilleur morceau est-il assez pertinent ? Sinon on refuse, sans appeler le LLM."""
    best_score = state["chunks"][0].metadata["score"]
    return "generate" if best_score >= config.RELEVANCE_THRESHOLD else "refuse"


def route_after_generate(state: State) -> str:
    """Vérification seulement si l'utilisateur l'a demandée (elle coûte 1 à 2 min sur CPU)."""
    return "verify" if state.get("verify") else END


def build_agent():
    graph = StateGraph(State)
    graph.add_node("condense", condense)
    graph.add_node("search", search)
    graph.add_node("generate", generate)
    graph.add_node("refuse", refuse)
    graph.add_node("verify", verify)

    graph.add_edge(START, "condense")
    graph.add_edge("condense", "search")
    graph.add_conditional_edges("search", route_after_search, ["generate", "refuse"])
    graph.add_conditional_edges("generate", route_after_generate, ["verify", END])
    graph.add_edge("refuse", END)
    graph.add_edge("verify", END)
    return graph.compile()


agent = build_agent()


def stream_agent(question: str, history: list[tuple[str, str]], state: dict, verify_mode: bool = False):
    """Lance l'agent et renvoie la réponse mot par mot, puis None quand elle est finie (la vérification
    éventuelle continue ensuite) ; `state` se remplit au passage (question comprise, sources, vérification)."""
    inputs = {"question": question, "history": history, "verify": verify_mode}
    for mode, data in agent.stream(inputs, stream_mode=["updates", "messages"]):
        if mode == "updates":                   # un nœud vient de finir : on garde ce qu'il a écrit
            for node_output in data.values():
                state.update(node_output)
            if "refuse" in data:
                yield data["refuse"]["answer"]
            if "generate" in data or "refuse" in data:
                yield None                      # signal : la réponse est complète
        else:                                   # un mot produit par un LLM
            token, metadata = data
            if metadata["langgraph_node"] == "generate":
                yield token.content


if __name__ == "__main__":
    if sys.argv[1:] == ["--graph"]:
        print(agent.get_graph().draw_mermaid())
        sys.exit()

    args = sys.argv[1:]
    verify_mode = "--verifier" in args
    question = " ".join(a for a in args if a != "--verifier") or input("Votre question : ")
    start = time.time()
    result = agent.invoke({"question": question, "history": [], "verify": verify_mode})

    print(f"\n{result['answer']}\n")
    print("Sources :")
    for i, chunk in enumerate(result["chunks"], start=1):
        print(f"  [{i}] {chunk.metadata['title']} (pertinence {chunk.metadata['score']:.2f})")
    for check in result.get("checks", []):
        mark = "✅" if check["justifiee"] else "⚠️"
        print(f"\n{mark} {check['phrase']} {check['sources']}\n   {check['raison']}")
    print(f"\n({time.time() - start:.0f} s)")