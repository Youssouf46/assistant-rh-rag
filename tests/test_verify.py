"""Tests du découpage des réponses en phrases citées (vérificateur)."""
from langchain_core.documents import Document

from verify import check_claims, split_claims


def test_une_phrase_citee():
    assert split_claims("Le salarié a droit à 2,5 jours ouvrables par mois [1].") == [
        ("Le salarié a droit à 2,5 jours ouvrables par mois", [1])]


def test_plusieurs_numeros_de_source():
    claims = split_claims("En cas de circonstances exceptionnelles, le télétravail peut être imposé [1, 2].")
    assert claims[0][1] == [1, 2]


def test_retire_le_gras_et_l_etiquette_regle_generale():
    # Défaut corrigé à l'étape 8 : l'étiquette déclenchait le critère strict « règle générale »
    text, _ = split_claims("**Règle générale** : Le salarié qui travaille le 1er mai est payé double [1].")[0]
    assert text == "Le salarié qui travaille le 1er mai est payé double"


def test_une_puce_recoit_le_titre_de_sa_section():
    answer = "**Exceptions** :\n- Salarié travaillant à domicile [1]."
    assert split_claims(answer)[0][0] == "Exceptions : Salarié travaillant à domicile"


def test_une_sous_puce_recoit_sa_puce_parente():
    answer = "- **12 heures** par jour si :\n  - Une convention collective le prévoit [1]."
    assert split_claims(answer)[0][0] == "12 heures par jour si : Une convention collective le prévoit"


def test_ignore_les_fragments_trop_courts_et_les_phrases_sans_citation():
    assert split_claims("**Non** [2].\nLe télétravail est un mode d'organisation du travail.") == []


def test_numero_de_source_inexistant_signale_sans_appeler_le_llm():
    chunks = [Document(page_content=f"passage {i}") for i in range(4)]
    result = check_claims([("Une phrase qui cite une source absente", [7])], chunks, "question")
    assert result[0]["justifiee"] is False
    assert "n'existe pas" in result[0]["raison"]