"""Week 4: separating retrieval failures from generation failures, with evidence.

The two kinds of "wrong" (from the Week 4 brief):

  wrong_document              the expected recipe never appears in the top-k
                               retrieved chunks. A smarter model cannot fix
                               this - it never saw the right source.

  right_document_wrong_answer the expected recipe IS in the top-k retrieved
                               chunks, but the generated answer still doesn't
                               contain the expected answer token(s) (wrong,
                               incomplete, or a false refusal). This is a
                               generation-side problem.

classify() only ever trusts what it can check against the retrieved rows and
the literal generated text - never the model's own claim - the same
evidence-based spirit as generate_answers.py's citation verification.
"""

import re


def normalise(text):
    return re.sub(r"\s+", " ", text).lower()


def answer_present(text, token_groups):
    """True if any one full token group is entirely present in `text`.

    token_groups is a list of groups, e.g. [["shiitake", "kombu"], ["tamari"]] -
    the answer counts as present if EITHER the first group's tokens ALL
    appear, OR the second group's tokens all appear. Same shape as
    evaluate_chunkers.py's ANSWER_TOKENS, generalised into a reusable helper.
    """

    haystack = normalise(text)

    for group in token_groups:
        if all(normalise(token) in haystack for token in group):
            return True

    return False


def rows_from_results(results):
    """Flatten a Chroma-shaped retrieve()/retrieve_hybrid() result into rows.

    Returns [{"rank": 1, "chunk_id": ..., "recipe_id": ..., "text": ...}, ...],
    1-indexed by retrieval order - the same row shape evaluate_chunkers.py
    already builds by hand, factored out here for reuse.
    """

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    rows = []

    for rank, (chunk_id, text, meta) in enumerate(
        zip(ids, documents, metadatas), start=1
    ):
        rows.append({
            "rank": rank,
            "chunk_id": chunk_id,
            "recipe_id": meta.get("recipe_id"),
            "text": text,
        })

    return rows


def hit_at_k(rows, expected_recipe_id, k):
    """True if the expected recipe appears anywhere in the top-k rows."""

    return any(
        row["recipe_id"] == expected_recipe_id and row["rank"] <= k
        for row in rows
    )


def classify(question_record, rows, generated_answer, k=3):
    """Classify one question's outcome as pass / wrong_document /
    right_document_wrong_answer, with the evidence used to decide.

    `rows` must already be truncated/ranked consistently with `k` (callers
    pass whatever top-k they actually retrieved); hit_at_k additionally
    guards on rank <= k so passing a larger row list is harmless.
    """

    expected_recipe_id = question_record["expected_recipe_id"]
    token_groups = question_record["answer_tokens"]

    doc_hit = hit_at_k(rows, expected_recipe_id, k)
    expected_rank = next(
        (row["rank"] for row in rows if row["recipe_id"] == expected_recipe_id),
        None,
    )

    answer_correct = answer_present(generated_answer, token_groups)

    if doc_hit and answer_correct:
        label = "pass"
    elif not doc_hit:
        label = "wrong_document"
    else:
        label = "right_document_wrong_answer"

    if label == "wrong_document":
        evidence = (
            f"expected recipe '{expected_recipe_id}' not in top-{k} "
            f"(first seen at rank {expected_rank})"
            if expected_rank is not None
            else f"expected recipe '{expected_recipe_id}' not retrieved at all"
        )
    elif label == "right_document_wrong_answer":
        evidence = (
            f"expected recipe '{expected_recipe_id}' at rank {expected_rank} "
            f"(within top-{k}), but none of the expected answer tokens "
            f"{token_groups} appear in the generated answer"
        )
    else:
        evidence = f"expected recipe at rank {expected_rank}, answer tokens matched"

    return {
        "label": label,
        "expected_recipe_id": expected_recipe_id,
        "expected_rank": expected_rank,
        "doc_hit_at_k": doc_hit,
        "answer_correct": answer_correct,
        "k": k,
        "evidence": evidence,
    }
