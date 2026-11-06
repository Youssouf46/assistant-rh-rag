"""Tests de la détection des questions de suite."""
import pytest

from ask import is_follow_up


@pytest.mark.parametrize("question", [
    "Et si je suis à temps partiel ?",
    "Et pour un apprenti ?",
    "Dans ce cas, ai-je droit à une indemnité ?",
    "Et le préavis ?",
])
def test_questions_de_suite_detectees(question):
    assert is_follow_up(question)


@pytest.mark.parametrize("question", [
    "Mon patron peut-il m'obliger à télétravailler ?",
    "Combien de jours de congés payés a un salarié ?",
    "Quel est le montant du SMIC ?",
])
def test_questions_claires_non_reformulees(question):
    assert not is_follow_up(question)