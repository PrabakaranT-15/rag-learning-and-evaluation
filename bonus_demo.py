"""Bonus: structure-aware WINS retrieval but LOSES the answer.

The bonus asks for a question where the tight ingredient-row chunk retrieves
precisely and then gives the model no method prose to explain the answer.

The question below deliberately needs BOTH halves of the card:
  - the quantity, which lives only in the ingredient table
  - the stage at which it is added, which lives only in the method prose

At top_k=1 the tension is forced into the open. At top_k=5 both chunkers
retrieve enough to answer, which is itself part of the finding and is reported.
"""

import json

from rag.vector_store import get_collection, retrieve
from rag.generator import generate_recipe_answer


QUESTION = (
    "How much fine sea salt does the 2kg country sourdough use, "
    "and at what stage of mixing is it added?"
)

# Both halves of a complete answer.
NEEDS = {"quantity": "40 g", "stage": "autolyse"}

RUNS = [
    ("fermentation_baseline_500", "Strategy A - baseline 500/100"),
    ("fermentation_structure_aware", "Strategy B - structure-aware"),
]


def run(collection_name, label, top_k):

    collection = get_collection(collection_name)
    results = retrieve(collection, QUESTION, top_k)
    answer = generate_recipe_answer(QUESTION, results)

    retrieved = []
    for cid, doc, meta in zip(
        results["ids"][0], results["documents"][0], results["metadatas"][0]
    ):
        retrieved.append({
            "chunk_id": cid,
            "recipe_id": meta.get("recipe_id"),
            "section": meta.get("section") or "(flat window)",
            "has_quantity": NEEDS["quantity"].lower() in doc.lower(),
            "has_stage": NEEDS["stage"].lower() in doc.lower(),
        })

    return {
        "collection": collection_name,
        "label": label,
        "top_k": top_k,
        "retrieved": retrieved,
        "context_has_quantity": any(r["has_quantity"] for r in retrieved),
        "context_has_stage": any(r["has_stage"] for r in retrieved),
        "answer": answer.strip(),
        "answer_has_quantity": NEEDS["quantity"].lower() in answer.lower(),
        "answer_has_stage": NEEDS["stage"].lower() in answer.lower(),
    }


def main():

    lines = []
    report = []

    lines.append("=" * 78)
    lines.append("BONUS: precision vs completeness")
    lines.append("=" * 78)
    lines.append(f"QUESTION: {QUESTION}")
    lines.append("A complete answer needs BOTH:")
    lines.append("  quantity '40 g'   -> lives ONLY in the ingredient table")
    lines.append("  stage 'autolyse'  -> lives ONLY in the method prose")

    for top_k in (1, 5):

        lines.append("")
        lines.append("#" * 78)
        lines.append(f"# top_k = {top_k}")
        lines.append("#" * 78)

        for collection_name, label in RUNS:

            outcome = run(collection_name, label, top_k)
            report.append(outcome)

            lines.append("")
            lines.append(f"--- {label}  (top_k={top_k}) ---")
            for i, r in enumerate(outcome["retrieved"], 1):
                lines.append(
                    f"  {i}. {r['chunk_id']}  section={r['section']}  "
                    f"has'40 g'={r['has_quantity']}  has'autolyse'={r['has_stage']}"
                )
            lines.append(
                f"  CONTEXT COMPLETE: quantity={outcome['context_has_quantity']} "
                f"stage={outcome['context_has_stage']}"
            )
            lines.append("  ANSWER:")
            for line in outcome["answer"].split("\n"):
                lines.append(f"    {line}")
            lines.append(
                f"  ANSWER COMPLETE: quantity={outcome['answer_has_quantity']} "
                f"stage={outcome['answer_has_stage']}"
            )

    text = "\n".join(lines)
    print(text)

    with open("bonus_demo.txt", "w", encoding="utf-8") as handle:
        handle.write(text + "\n")

    with open("bonus_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
