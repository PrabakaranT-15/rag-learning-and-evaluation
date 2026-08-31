"""Week 5: collect ~20 real traces from the app's actual retrieve+generate path.

A "trace" is a complete, replayable record of one request: the question, every
retrieved chunk (id, recipe_id, section, text, distance), and the generated
answer. This does NOT curate for failure - week5_trace_questions.json is a
fair, ordinary sample (see its _note), unlike week4_failing_questions.json.

Model note: the app's actual default is generator.DEFAULT_MODEL
(gemini-3.6-flash), but that model is capped at 20 generate_content calls per
day on the free tier (see generator.py's WEEK4_MODEL docstring) - a single
20-question batch would exhaust the ENTIRE day's quota by itself, with zero
margin for retries. WEEK4_MODEL is used here instead, for the same reason it
was introduced in Week 4. This is a real, disclosed substitution, not a
silent one - it is called out again in the Week 5 report.

Collection note: the distractor-enriched realistic_structure_aware collection
from Week 3/4 needs a recipes/ folder (195 cards) that is no longer present
on this machine and could not be located anywhere under the user profile.
Rebuilding it produced collections with ZERO actual distractors under a
misleading name - those were caught and deleted rather than used. This run
therefore uses fermentation_structure_aware (the 6 real cards, no
distractors) instead. Disclosed here and in the Week 5 report - this still
surfaces real generation-layer failures (wrong citations, bad refusals,
near-duplicate kimchi confusion), just not the distractor-collision failures
Week 3/4 specifically found.

Collection: fermentation_structure_aware (6 cards, no distractors).
Retrieval: semantic-only, top_k=5 - the app's own defaults.
"""

import json

from rag.vector_store import get_collection, retrieve
from rag.evaluation import rows_from_results
from rag.generator import generate_recipe_answer, WEEK4_MODEL


COLLECTION = "fermentation_structure_aware"
TOP_K = 5


def main():

    with open("week5_trace_questions.json", encoding="utf-8") as handle:
        spec = json.load(handle)

    questions = spec["questions"]
    collection = get_collection(COLLECTION)

    traces = []

    print("=" * 78)
    print(f"WEEK 5 TRACE COLLECTION  -  collection: {COLLECTION}  top_k: {TOP_K}")
    print(f"model: {WEEK4_MODEL}  (see script docstring for why, not DEFAULT_MODEL)")
    print("=" * 78)

    for i, record in enumerate(questions, start=1):

        question = record["question"]
        print(f"\n[{i}/{len(questions)}] {record['id']}: {question}")

        results = retrieve(collection, question, TOP_K)
        rows = rows_from_results(results)
        answer = generate_recipe_answer(question, results, model=WEEK4_MODEL)

        print(f"  -> {answer.strip()[:150]}")

        traces.append({
            "id": record["id"],
            "question": question,
            "retrieved": rows,
            "answer": answer.strip(),
        })

    with open("week5_traces.json", "w", encoding="utf-8") as handle:
        json.dump(
            {"collection": COLLECTION, "top_k": TOP_K, "model": WEEK4_MODEL, "traces": traces},
            handle, indent=2,
        )

    print("\nWrote week5_traces.json")


if __name__ == "__main__":
    main()
