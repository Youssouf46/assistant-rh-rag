"""Tests de l'évaluation : outils de comparaison et format du jeu de test."""
import json

import pytest
from langchain_core.documents import Document

import config
from evaluate import normalize, rank_of_passage

QUESTIONS = json.loads((config.ROOT / "data" / "eval" / "questions.json").read_text(encoding="utf-8"))


def test_normalize_ignore_majuscules_espaces_et_espace_avant_pourcent():
    assert normalize("Majorées de  25 %\net 50 %") == "majorées de 25% et 50%"


def test_rank_of_passage():
    chunks = [Document(page_content="rien"), Document(page_content="… 15 jours calendaires …")]
    assert rank_of_passage(chunks, "15 jours calendaires") == 2
    assert rank_of_passage(chunks, "absent") is None


def test_jeu_de_test_assez_grand_et_sans_doublon():
    questions = [item["question"] for item in QUESTIONS]
    assert len(questions) >= 20
    assert len(questions) == len(set(questions))


@pytest.mark.parametrize("item", [q for q in QUESTIONS if not q.get("hors_sujet")], ids=lambda q: q["question"][:40])
def test_chaque_question_dans_les_fiches_a_un_passage_et_des_faits(item):
    assert item["passage"].strip()
    assert item["faits"] and all(fact.strip() for fact in item["faits"])


def test_il_y_a_des_questions_hors_sujet():
    assert any(item.get("hors_sujet") for item in QUESTIONS)