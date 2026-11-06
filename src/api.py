"""API web de l'assistant RH (FastAPI) : poser une question par HTTP, recevoir la réponse sourcée en JSON.
"""
from fastapi import FastAPI
from pydantic import BaseModel, Field

import config
from agent import agent

app = FastAPI(
    title="Assistant RH souverain",
    description="Questions sur le droit du travail français (secteur privé), réponses sourcées par les fiches "
                "service-public.fr. 100 % local : Ollama, Chroma, LangGraph.",
    version="1.0.0",
)


# --- Les formats d'échange (contrats de l'API) ---

class Message(BaseModel):
    role: str = Field(description='"user" ou "assistant"')
    content: str


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500, examples=["Combien de jours de congés payés a un salarié ?"])
    history: list[Message] = Field(default=[], description="Derniers échanges, pour comprendre les questions de suite")
    verify: bool = Field(default=False, description="Vérifier chaque phrase citée (1 à 2 min de plus sur CPU)")


class Source(BaseModel):
    title: str
    url: str
    score: float = Field(description="Pertinence donnée par le reranker, de 0 à 1")
    extract: str


class SentenceCheck(BaseModel):
    phrase: str
    sources: list[int]
    justifiee: bool
    raison: str


class AskResponse(BaseModel):
    answer: str
    search_question: str = Field(description="La question comprise (reformulée si c'était une question de suite)")
    sources: list[Source]
    checks: list[SentenceCheck] | None = Field(default=None, description="Présent seulement si verify = true")


# --- Les routes ---

@app.get("/health")
def health() -> dict:
    """Vérifie que l'API répond et indique les modèles utilisés."""
    return {"status": "ok", "llm": config.LLM_MODEL, "embeddings": config.EMBED_MODEL, "reranker": config.RERANK_MODEL}


@app.post("/ask")
def ask(request: AskRequest) -> AskResponse:
    """Pose une question à l'agent LangGraph et renvoie la réponse, la question comprise et les sources."""
    history = [(message.role, message.content) for message in request.history[-4:]]
    result = agent.invoke({"question": request.question, "history": history, "verify": request.verify})
    sources = [
        Source(title=chunk.metadata["title"], url=chunk.metadata["source"],
               score=round(chunk.metadata["score"], 3), extract=chunk.page_content[:300])
        for chunk in result["chunks"]
    ]
    return AskResponse(answer=result["answer"], search_question=result["search_question"],
                       sources=sources, checks=result.get("checks"))