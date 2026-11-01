"""Comparer plusieurs modèles sur les mêmes questions (même recherche, même prompt).
"""
import time

from ask import ask

MODELS = ["mistral", "ministral-3:8b"]
QUESTIONS = [
    "Combien de jours de congés payés a un salarié en France ?",
    "Mon patron peut-il m'obliger à télétravailler ?",
    "Quel est le montant du SMIC ?",
]

for question in QUESTIONS:
    print(f"\n{'=' * 80}\nQUESTION : {question}")
    for model in MODELS:
        start = time.time()
        answer, _ = ask(question, model=model)
        print(f"\n--- {model} ({time.time() - start:.0f} s)\n{answer}")
