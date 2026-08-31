"""Pure-function tests for rag/evaluation.py's failure classification.

Uses synthetic retrieved rows and a synthetic generated answer - no network,
no embeddings, no Chroma.
"""

from rag.evaluation import answer_present, hit_at_k, classify


QUESTION_RECORD = {
    "expected_recipe_id": "ferment_002",
    "answer_tokens": [["float", "sink"]],
}


def rows(*recipe_ids):
    return [
        {"rank": i + 1, "chunk_id": f"chunk_{i}", "recipe_id": rid, "text": ""}
        for i, rid in enumerate(recipe_ids)
    ]


def test_answer_present_matches_any_full_group():
    assert answer_present("it should FLOAT rather than sink", [["float", "sink"]])
    assert not answer_present("it should float", [["float", "sink"]])


def test_hit_at_k_respects_k():
    r = rows("ferment_001", "ferment_002", "ferment_003")
    assert hit_at_k(r, "ferment_002", k=3)
    assert not hit_at_k(r, "ferment_002", k=1)


def test_classify_pass_when_doc_hit_and_answer_correct():
    r = rows("ferment_002", "ferment_001")
    outcome = classify(QUESTION_RECORD, r, "it should float rather than sink", k=3)
    assert outcome["label"] == "pass"
    assert outcome["expected_rank"] == 1


def test_classify_wrong_document_when_expected_recipe_missing():
    r = rows("ferment_005", "ferment_006", "ferment_001")
    outcome = classify(QUESTION_RECORD, r, "no idea", k=3)
    assert outcome["label"] == "wrong_document"
    assert outcome["expected_rank"] is None


def test_classify_right_document_wrong_answer_when_doc_present_but_answer_wrong():
    r = rows("ferment_005", "ferment_002", "ferment_006")
    outcome = classify(QUESTION_RECORD, r, "I could not find this information.", k=3)
    assert outcome["label"] == "right_document_wrong_answer"
    assert outcome["expected_rank"] == 2
