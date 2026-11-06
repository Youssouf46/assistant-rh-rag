"""Tests de la recherche : découpage en mots (BM25) et fusion des classements (RRF)."""
from langchain_core.documents import Document

from search import reciprocal_rank_fusion, tokenize


def doc(text: str) -> Document:
    return Document(page_content=text)


def test_tokenize_met_en_minuscules_et_retire_ponctuation_et_mots_vides():
    assert tokenize("Comment les congés sont-ils calculés ?") == ["congés", "calculés"]


def test_tokenize_garde_les_mots_utiles_de_la_question_temps_partiel():
    words = tokenize("Et si je suis à temps partiel ?")
    assert "temps" in words and "partiel" in words
    assert "et" not in words and "je" not in words


def test_rrf_un_morceau_present_dans_les_deux_listes_passe_devant():
    # Le cas réel de l'étape 6a : « temps partiel » 7e en vecteurs et 3e en BM25 → 1/67 + 1/63 ≈ 0,0308,
    # devant un morceau 1er dans une seule liste (1/61 ≈ 0,0164).
    target = doc("temps partiel")
    vectors = [doc("vecteur 1")] + [doc(f"vecteur {i}") for i in range(2, 7)] + [target]
    keywords = [doc("bm25 1"), doc("bm25 2"), target]
    fused = reciprocal_rank_fusion([vectors, keywords])
    assert fused[0].page_content == "temps partiel"


def test_rrf_supprime_les_doublons():
    a, b = doc("a"), doc("b")
    fused = reciprocal_rank_fusion([[a, b], [b, a]])
    assert sorted(d.page_content for d in fused) == ["a", "b"]


def test_rrf_garde_l_ordre_si_une_seule_liste():
    fused = reciprocal_rank_fusion([[doc("1er"), doc("2e"), doc("3e")]])
    assert [d.page_content for d in fused] == ["1er", "2e", "3e"]