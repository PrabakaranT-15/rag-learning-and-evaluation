"""Pure-function tests for the RRF fusion math in rag/hybrid_retriever.py.

No network, no embeddings, no Chroma - _rrf_scores and tokenize are plain
functions over plain data, so these run instantly and without an API key.
"""

from rag.hybrid_retriever import _rrf_scores, tokenize, RRF_K


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("What is the Windowpane-Test?") == ["what", "is", "the", "windowpane", "test"]


def test_tokenize_keeps_numbers():
    assert tokenize("250C for 20 minutes") == ["250c", "for", "20", "minutes"]


def test_rrf_agreement_beats_single_ranker_top_hit():
    """An id ranked 2nd by BOTH rankers should outscore an id ranked 1st by
    only ONE of them - that's the whole point of fusing two signals."""

    semantic = ["y", "x"]
    bm25 = ["z", "x"]

    scores = _rrf_scores([semantic, bm25])

    assert scores["x"] > scores["y"]
    assert scores["x"] > scores["z"]
    assert scores["y"] == scores["z"]


def test_rrf_id_absent_from_one_ranker_still_scores():
    semantic = ["a", "b"]
    bm25 = ["c", "d"]

    scores = _rrf_scores([semantic, bm25])

    assert scores["a"] == 1.0 / (RRF_K + 1)
    assert scores["c"] == 1.0 / (RRF_K + 1)
    assert set(scores) == {"a", "b", "c", "d"}


def test_rrf_empty_lists_produce_no_scores():
    assert _rrf_scores([[], []]) == {}
