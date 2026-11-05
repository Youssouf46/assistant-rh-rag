"""Vérification de la fidélité : chaque phrase citée est-elle justifiée par son passage ?
"""
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

import config

CHECK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Tu vérifies si une phrase est justifiée par un passage d'une fiche officielle.
La phrase est extraite d'une réponse à la question indiquée : elle peut être courte (un chiffre, un élément de liste), juge-la dans le contexte de cette question.
La phrase est justifiée seulement si le passage dit la même chose, avec les mêmes conditions.
Elle n'est pas justifiée si elle ajoute une information absente du passage, le contredit, ou présente comme une règle générale ce que le passage limite à un cas particulier."""),
    ("human", "Question : {question}\n\nPassage :\n{passage}\n\nPhrase à vérifier :\n{claim}"),
])

# Un morceau de texte suivi d'un ou plusieurs numéros de source : « … 2,5 jours [1]. » ou « … [1, 2] »
CLAIM = re.compile(r"(.+?)((?:\s*\[\d+(?:\s*,\s*\d+)*\])+)")
# Les étiquettes de mise en forme en tête de ligne : « Règle générale : », « Exceptions : »…
LABEL = re.compile(r"^(?:réponse directe|réponse|règle générale|exceptions?|cas général)\s*:\s*", re.IGNORECASE)


class Check(BaseModel):
    """Le format imposé à la réponse du LLM."""
    justifiee: bool = Field(description="true si le passage dit cela, avec les mêmes conditions")
    raison: str = Field(description="une phrase courte d'explication (20 mots maximum)")


def clean(text: str) -> str:
    """Retire le gras, les puces, la ponctuation de bord et l'étiquette de tête."""
    text = text.replace("*", "").strip(" -•:.;,")
    return LABEL.sub("", text).strip()


def split_claims(answer: str) -> list[tuple[str, list[int]]]:
    """Découpe la réponse en phrases citées : [("Le salarié a droit à…", [1]), ...].
    Une puce reçoit le titre qui l'annonce (« Exceptions : … », « 12 heures par jour si : … ») pour garder son sens."""
    claims = []
    section, parent = "", ""   # dernier titre hors liste, dernière puce qui annonce des sous-puces
    for line in answer.splitlines():
        stripped = line.strip()
        is_item = stripped.startswith(("-", "•"))
        is_sub_item = is_item and line.startswith((" ", "\t"))
        title = stripped.replace("*", "").strip(" -•:") if stripped.endswith(":") else ""
        if not is_item:
            section, parent = title, ""
        elif not is_sub_item:
            parent = title
        context = parent if is_sub_item else section if is_item else ""
        for text, citations in CLAIM.findall(line):
            numbers = sorted({int(n) for n in re.findall(r"\d+", citations)})
            text = clean(text)
            if context:
                text = f"{context} : {text}"
            if len(text) > 15:   # ignore les fragments trop courts pour être vérifiés seuls (« Non », « Oui »)
                claims.append((text, numbers))
    return claims


def check_claims(claims: list[tuple[str, list[int]]], chunks, question: str) -> list[dict]:
    """Le LLM vérifie chaque phrase contre le ou les passages qu'elle cite, une par une."""
    checker = CHECK_PROMPT | ChatOllama(model=config.LLM_MODEL, temperature=0).with_structured_output(Check)
    results = []
    for text, numbers in claims:
        passages = [chunks[n - 1].page_content for n in numbers if 1 <= n <= len(chunks)]
        if not passages:
            results.append({"phrase": text, "sources": numbers, "justifiee": False, "raison": "Ce numéro de source n'existe pas."})
            continue
        verdict = checker.invoke({"question": question, "passage": "\n\n".join(passages), "claim": text})
        results.append({"phrase": text, "sources": numbers, "justifiee": verdict.justifiee, "raison": verdict.raison})
    return results