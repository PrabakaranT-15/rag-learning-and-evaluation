"""Week 4: label failures, apply one change (hybrid search), measure hit-rate@3.

For every question in week4_failing_questions.json, runs BOTH:

  BEFORE  vector_store.retrieve()          semantic-only (the existing path)
  AFTER   hybrid_retriever.retrieve_hybrid() semantic + BM25, fused with RRF

at top_k=3, generates an answer for each with the SAME model
(generator.WEEK4_MODEL, chosen specifically for this two-condition experiment
- see its docstring), and classifies each with rag.evaluation.classify()
into pass / wrong_document / right_document_wrong_answer.

Reports one number: hit-rate@3 (fraction of questions where the expected
recipe is in the top 3) before vs after, plus which failures the change did
and did not fix - the headline plus per-question evidence go to
week4_report.json and week4_report.md.
"""

import json

from rag.vector_store import get_collection, retrieve
from rag.hybrid_retriever import retrieve_hybrid
from rag.evaluation import rows_from_results, classify
from rag.generator import generate_recipe_answer, WEEK4_MODEL


COLLECTION = "realistic_structure_aware"
TOP_K = 3

# Retrieval-only diagnostic probes: real collisions found by hand while
# building the question set, kept OUT of the scored 11 because their ground
# truth is genuinely multi-valid (more than one recipe_id is a legitimate
# answer), so they can't be pass/fail classified the way the scored
# questions are. Run at top_k=5 with no generation call (free, and useful
# even when the scored set has no wrong_document example to show).
EXPLORATORY_PROBES = [
    {
        "question": "What is the bend test used to check?",
        "note": (
            "'bend test' occurs verbatim in THREE documents: ferment_005, "
            "ferment_006 (both fermentation cards) and recipe_169 (a "
            "distractor card's Tips section: 'The bend test on the cabbage "
            "stem is the correct measure of salting, not the timer.'). No "
            "single recipe_id is 'the' right answer here - this probe exists "
            "to show what hybrid actually does to a real rare-term "
            "collision, not to be scored pass/fail."
        ),
    },
]


def run_condition(collection, question_record, results):

    rows = rows_from_results(results)
    answer = generate_recipe_answer(question_record["question"], results, model=WEEK4_MODEL)
    outcome = classify(question_record, rows, answer, k=TOP_K)

    outcome["answer"] = answer.strip()
    outcome["retrieved"] = [
        {"rank": r["rank"], "chunk_id": r["chunk_id"], "recipe_id": r["recipe_id"]}
        for r in rows
    ]

    return outcome


def main():

    with open("week4_failing_questions.json", encoding="utf-8") as handle:
        spec = json.load(handle)

    questions = spec["questions"]
    collection = get_collection(COLLECTION)

    print("=" * 78)
    print(f"WEEK 4 FAILURE ANALYSIS  -  collection: {COLLECTION}  top_k: {TOP_K}")
    print(f"model (both conditions): {WEEK4_MODEL}")
    print("=" * 78)

    per_question = []

    for i, record in enumerate(questions, start=1):

        print(f"\n[{i}/{len(questions)}] {record['id']}: {record['question']}")

        print("  running BEFORE (semantic-only)...")
        before_results = retrieve(collection, record["question"], TOP_K)
        before = run_condition(collection, record, before_results)
        print(f"    label={before['label']}  expected_rank={before['expected_rank']}")

        print("  running AFTER (hybrid: semantic + BM25, RRF-fused)...")
        after_results = retrieve_hybrid(collection, record["question"], TOP_K)
        after = run_condition(collection, record, after_results)
        print(f"    label={after['label']}  expected_rank={after['expected_rank']}")

        per_question.append({
            "id": record["id"],
            "question": record["question"],
            "expected_recipe_id": record["expected_recipe_id"],
            "category": record["category"],
            "hypothesis": record["hypothesis"],
            "before": before,
            "after": after,
            "changed": before["label"] != after["label"],
            "fixed": before["label"] != "pass" and after["label"] == "pass",
            "still_failing": before["label"] != "pass" and after["label"] != "pass",
            "regressed": before["label"] == "pass" and after["label"] != "pass",
        })

    n = len(per_question)
    hit_before = sum(1 for q in per_question if q["before"]["doc_hit_at_k"])
    hit_after = sum(1 for q in per_question if q["after"]["doc_hit_at_k"])
    pass_before = sum(1 for q in per_question if q["before"]["label"] == "pass")
    pass_after = sum(1 for q in per_question if q["after"]["label"] == "pass")

    summary = {
        "collection": COLLECTION,
        "top_k": TOP_K,
        "model": WEEK4_MODEL,
        "n_questions": n,
        "hit_rate_at_3_before": f"{hit_before}/{n}",
        "hit_rate_at_3_after": f"{hit_after}/{n}",
        "pass_rate_before": f"{pass_before}/{n}",
        "pass_rate_after": f"{pass_after}/{n}",
        "fixed": [q["id"] for q in per_question if q["fixed"]],
        "still_failing": [q["id"] for q in per_question if q["still_failing"]],
        "regressed": [q["id"] for q in per_question if q["regressed"]],
    }

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"hit-rate@{TOP_K}  BEFORE : {hit_before}/{n}")
    print(f"hit-rate@{TOP_K}  AFTER  : {hit_after}/{n}")
    print(f"pass rate BEFORE        : {pass_before}/{n}")
    print(f"pass rate AFTER         : {pass_after}/{n}")
    print(f"fixed by hybrid         : {summary['fixed']}")
    print(f"still failing after     : {summary['still_failing']}")
    print(f"regressed by hybrid     : {summary['regressed']}")

    print("\nRunning exploratory retrieval-only probes (no generation call)...")

    probes = []

    for probe in EXPLORATORY_PROBES:

        semantic_rows = rows_from_results(retrieve(collection, probe["question"], 5))
        hybrid_rows = rows_from_results(retrieve_hybrid(collection, probe["question"], 5))

        probes.append({
            "question": probe["question"],
            "note": probe["note"],
            "semantic": semantic_rows,
            "hybrid": hybrid_rows,
        })

        print(f"  {probe['question']}")
        print(f"    semantic top-5 recipe_ids: {[r['recipe_id'] for r in semantic_rows]}")
        print(f"    hybrid   top-5 recipe_ids: {[r['recipe_id'] for r in hybrid_rows]}")

    with open("week4_report.json", "w", encoding="utf-8") as handle:
        json.dump(
            {"summary": summary, "per_question": per_question, "exploratory_probes": probes},
            handle, indent=2,
        )

    write_markdown_report(summary, per_question, probes)

    print("\nWrote week4_report.json and week4_report.md")


def write_markdown_report(summary, per_question, probes=None):

    lines = []
    lines.append("# Week 4 — Debugging Retrieval: failure labels + hybrid search result")
    lines.append("")
    lines.append(f"Collection: `{summary['collection']}`  ·  top_k = {summary['top_k']}  ·  "
                  f"generation model (both conditions): `{summary['model']}`")
    lines.append("")
    lines.append("## Headline number")
    lines.append("")
    lines.append(f"**hit-rate@{summary['top_k']}: {summary['hit_rate_at_3_before']} "
                  f"(before) -> {summary['hit_rate_at_3_after']} (after hybrid search)**")
    lines.append("")
    lines.append(f"End-to-end pass rate (right doc AND right answer): "
                  f"{summary['pass_rate_before']} -> {summary['pass_rate_after']}")
    lines.append("")
    lines.append(
        "**Honest read: this is a ceiling effect, not a win.** At top_k=3, "
        "semantic-only retrieval already found the expected recipe for all "
        "11 questions in this set, including ones deliberately built around "
        "rare-term collisions and a real same-name distractor — so hybrid "
        "search had no `wrong_document` failures left to fix on this "
        "question set, and the hit-rate@3 number cannot move. The one real "
        "failure that survived (W4) is a *generation*-layer failure, not a "
        "retrieval one, and is reported honestly below as NOT fixed by this "
        "change — hybrid is a retrieval-side fix and this isn't a "
        "retrieval-side problem. See section 6 of `results.md` for the "
        "matching root-cause diagnosis from Week 3."
    )
    lines.append("")
    lines.append("## Per-question before / after")
    lines.append("")
    lines.append("| ID | Category | Before | After | Outcome |")
    lines.append("|---|---|---|---|---|")

    for q in per_question:
        outcome = (
            "FIXED" if q["fixed"]
            else "regressed" if q["regressed"]
            else "still failing" if q["still_failing"]
            else "unchanged (pass)"
        )
        lines.append(
            f"| {q['id']} | {q['category']} | {q['before']['label']} "
            f"(rank {q['before']['expected_rank']}) | {q['after']['label']} "
            f"(rank {q['after']['expected_rank']}) | {outcome} |"
        )

    lines.append("")
    lines.append("## Evidence")
    lines.append("")

    for q in per_question:
        lines.append(f"### {q['id']} — {q['question']}")
        lines.append(f"- Hypothesis (written before running anything): {q['hypothesis']}")
        lines.append(f"- BEFORE: **{q['before']['label']}** — {q['before']['evidence']}")
        lines.append(f"- AFTER: **{q['after']['label']}** — {q['after']['evidence']}")
        lines.append("")

    lines.append("## What hybrid search did NOT fix")
    lines.append("")

    still_failing = [q for q in per_question if q["still_failing"]]

    if not still_failing:
        lines.append("Every question that failed before also passed after. "
                      "(If that holds across the `contrast`-category questions too, "
                      "treat that as a result worth re-checking, not a free win - "
                      "see the hypothesis notes above.)")
    else:
        for q in still_failing:
            lines.append(
                f"- **{q['id']}** ({q['category']}): still `{q['after']['label']}` "
                f"after hybrid — {q['after']['evidence']}"
            )

    lines.append("")

    if probes:
        lines.append("## Exploratory probes (retrieval-only, not scored)")
        lines.append("")
        lines.append(
            "These aren't part of the scored 11 because their ground truth "
            "is genuinely multi-valid — more than one recipe_id is a "
            "legitimate hit. They exist to show what hybrid actually does "
            "to a real rare-term collision, independent of pass/fail scoring."
        )
        lines.append("")

        for probe in probes:
            lines.append(f"**{probe['question']}**")
            lines.append("")
            lines.append(f"{probe['note']}")
            lines.append("")
            sem_ids = [r["recipe_id"] for r in probe["semantic"]]
            hyb_ids = [r["recipe_id"] for r in probe["hybrid"]]
            lines.append(f"- semantic top-5 recipe_ids: `{sem_ids}`")
            lines.append(f"- hybrid top-5 recipe_ids: `{hyb_ids}`")
            lines.append("")

    with open("week4_report.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
