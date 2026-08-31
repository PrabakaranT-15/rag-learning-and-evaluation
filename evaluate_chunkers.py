"""Search-only Hit@5 evaluation over the SAME 8 questions for each strategy.

No generation happens here. This is pure retrieval measurement, so the number
reported is a property of the chunker and not of the LLM.

Two metrics are recorded per question:

  hit_recipe@5    the expected recipe_id appears somewhere in the top 5.
                  Comparable across strategies, because the baseline chunker
                  has no concept of sections and would auto-fail a
                  section-level metric for reasons that have nothing to do
                  with retrieval quality.

  answer_in@5     at least one retrieved chunk literally contains the answer
                  token (for example "40 g" or "85%"). This is the metric that
                  actually predicts whether generation can succeed: retrieving
                  the right recipe is worthless if the chunk that came back
                  does not contain the number.

The answer tokens are derived from the expected_answer field that was written
into eval_questions.json BEFORE any retrieval was run.
"""

import json
import re

from rag.vector_store import get_collection, retrieve


import sys

CONDITIONS = {
    # Requirement-6-compliant condition: ONLY the 6 new cards are in the index.
    "cards6": [
        ("fermentation_baseline_500",
         "Strategy A - baseline chunker @ 500/100 (app default)"),
        ("fermentation_baseline_120",
         "Strategy A - baseline chunker @ 120/20 (granularity matched)"),
        ("fermentation_structure_aware",
         "Strategy B - structure-aware chunker"),
    ],
    # Realistic condition: the 6 new cards sit among fermentation/bread-adjacent
    # cards from the existing corpus, as Task Set B assumes ("already indexes
    # the old cards"). Same 8 questions, same embedding model.
    "realistic": [
        ("realistic_baseline_500",
         "Strategy A - baseline @ 500/100 (+ distractors)"),
        ("realistic_baseline_120",
         "Strategy A - baseline @ 120/20 (+ distractors)"),
        ("realistic_structure_aware",
         "Strategy B - structure-aware (+ distractors)"),
    ],
}

CONDITION = sys.argv[1] if len(sys.argv) > 1 else "cards6"
COLLECTIONS = CONDITIONS[CONDITION]

TOP_K = 5

# Answer tokens derived from expected_answer in eval_questions.json.
# A question passes answer_in@5 if ANY of its token groups is fully present
# in ANY single retrieved chunk.
ANSWER_TOKENS = {
    "Q1": [["40 g"], ["40g"]],
    "Q2": [["85%"]],
    "Q3": [["40%"]],
    "Q4": [["100 g", "5%"]],
    "Q5": [["250C", "20 minutes"]],
    "Q6": [["8 to 12 hours"]],
    "Q7": [["shiitake", "kombu"], ["tamari"]],
    "Q8": [["gluten"], ["dairy"], ["egg"]],
}


def normalise(text):
    return re.sub(r"\s+", " ", text).lower()


def answer_present(chunk_text, token_groups):
    """True if any complete token group appears in this single chunk."""

    haystack = normalise(chunk_text)

    for group in token_groups:
        if all(normalise(token) in haystack for token in group):
            return True

    return False


def run_question(collection, question_record):

    results = retrieve(collection, question_record["question"], TOP_K)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]

    rows = []
    hit_recipe = False
    answer_in = False

    for rank, (cid, doc, meta, dist) in enumerate(
        zip(ids, documents, metadatas, distances), start=1
    ):

        is_expected = meta.get("recipe_id") == question_record["expected_recipe_id"]

        if is_expected:
            hit_recipe = True

        if answer_present(doc, ANSWER_TOKENS[question_record["id"]]):
            answer_in = True

        rows.append({
            "rank": rank,
            "chunk_id": cid,
            "distance": round(float(dist), 4),
            "recipe_id": meta.get("recipe_id"),
            "source_file": meta.get("source_file"),
            "section": meta.get("section") or "(none - flat window)",
            "block_type": meta.get("block_type"),
            "is_expected_recipe": is_expected,
            "text": doc,
        })

    return {
        "rows": rows,
        "hit_recipe": hit_recipe,
        "answer_in": answer_in,
    }


def main():

    with open("eval_questions.json", encoding="utf-8") as handle:
        spec = json.load(handle)

    questions = spec["questions"]

    all_results = {}
    dump_lines = []

    dump_lines.append("SEARCH-ONLY DUMP - all 8 questions, all strategies")
    dump_lines.append("Retrieval only. No LLM involved. top_k = 5.")
    dump_lines.append("Embedding model held constant: gemini-embedding-001")
    dump_lines.append("=" * 78)

    for collection_name, label in COLLECTIONS:

        collection = get_collection(collection_name)
        all_results[collection_name] = {}

        for question_record in questions:

            outcome = run_question(collection, question_record)
            all_results[collection_name][question_record["id"]] = outcome

            dump_lines.append("")
            dump_lines.append("=" * 78)
            dump_lines.append(f"{question_record['id']}  |  {label}")
            dump_lines.append(f"collection: {collection_name}")
            dump_lines.append("=" * 78)
            dump_lines.append(f"Question        : {question_record['question']}")
            dump_lines.append(f"Expected answer : {question_record['expected_answer']}")
            dump_lines.append(f"Expected recipe : {question_record['expected_recipe_id']}")
            dump_lines.append(f"Expected section: {question_record['expected_section']}")
            dump_lines.append(f"Table-row question: {question_record['depends_on_table_row']}")
            dump_lines.append("-" * 78)

            for row in outcome["rows"]:
                marker = "  <== EXPECTED RECIPE" if row["is_expected_recipe"] else ""
                dump_lines.append(
                    f"  {row['rank']}. chunk_id={row['chunk_id']}{marker}"
                )
                dump_lines.append(
                    f"     distance={row['distance']}  recipe_id={row['recipe_id']}  "
                    f"source_file={row['source_file']}"
                )
                dump_lines.append(
                    f"     section={row['section']}  block_type={row['block_type']}"
                )
                snippet = re.sub(r"\s+", " ", row["text"])[:320]
                dump_lines.append(f"     text: {snippet}...")
                dump_lines.append("")

            dump_lines.append(
                f"  hit_recipe@5 : {'PASS' if outcome['hit_recipe'] else 'FAIL'}"
            )
            dump_lines.append(
                f"  answer_in@5  : {'PASS' if outcome['answer_in'] else 'FAIL'}"
            )

    # ------------------------------------------------------------- summary
    summary = {}

    for collection_name, _ in COLLECTIONS:
        hit = sum(1 for q in questions if all_results[collection_name][q["id"]]["hit_recipe"])
        ans = sum(1 for q in questions if all_results[collection_name][q["id"]]["answer_in"])
        summary[collection_name] = {"hit_recipe": hit, "answer_in": ans}

    print("\n" + "=" * 78)
    print(f"PER-QUESTION RESULTS - condition: {CONDITION}  (R = hit_recipe@5, A = answer_in@5)")
    print("=" * 78)

    header = f"{'Q':<4}{'table?':<8}"
    for name, _ in COLLECTIONS:
        header += f"{name.replace('fermentation_', ''):<26}"
    print(header)

    for question_record in questions:
        line = f"{question_record['id']:<4}{str(question_record['depends_on_table_row']):<8}"
        for name, _ in COLLECTIONS:
            outcome = all_results[name][question_record["id"]]
            cell = (
                f"R:{'PASS' if outcome['hit_recipe'] else 'FAIL'} "
                f"A:{'PASS' if outcome['answer_in'] else 'FAIL'}"
            )
            line += f"{cell:<26}"
        print(line)

    print("-" * 78)
    total = f"{'TOT':<4}{'':<8}"
    for name, _ in COLLECTIONS:
        s = summary[name]
        cell = "R:{}/8 A:{}/8".format(s["hit_recipe"], s["answer_in"])
        total += f"{cell:<26}"
    print(total)

    with open(f"search_dump_{CONDITION}.txt", "w", encoding="utf-8") as handle:
        handle.write("\n".join(dump_lines) + "\n")

    with open(f"eval_results_{CONDITION}.json", "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "results": all_results}, handle, indent=2)

    print("\nWrote search_dump.txt and eval_results.json")


if __name__ == "__main__":
    main()
