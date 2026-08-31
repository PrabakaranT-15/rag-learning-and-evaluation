"""Grounded generation: 3 answerable questions + 3 out-of-corpus questions.

Every citation emitted by the model is verified programmatically against
ChromaDB: the chunk_id must actually exist, and the chunk it resolves to must
actually contain the claimed value. A citation that resolves to nothing, or to
a chunk that does not contain the answer, is reported as FAILED rather than
quietly accepted.
"""

import json
import re

from rag.vector_store import get_collection, retrieve, chroma_client
from rag.generator import generate_recipe_answer, REFUSAL_TEXT


COLLECTION = "fermentation_structure_aware"
TOP_K = 5

CITATION_RE = re.compile(
    r"\[recipe_id=([^\s|\]]+)\s*\|\s*chunk_id=([^\s|\]]+)\s*\|\s*source_file=([^\]]+)\]"
)

# Token that must physically appear in the cited chunk for the claim to check out.
VERIFY_TOKEN = {
    "Q1": "40 g",
    "Q5": "250C",
    "Q7": "kombu",
}


def verify_citations(collection, answer, verify_token):
    """Resolve every citation against Chroma and check it contains the claim."""

    findings = []

    for recipe_id, chunk_id, source_file in CITATION_RE.findall(answer):

        chunk_id = chunk_id.strip()

        record = collection.get(ids=[chunk_id], include=["documents", "metadatas"])

        if not record["ids"]:
            findings.append({
                "chunk_id": chunk_id,
                "resolves": False,
                "contains_claim": False,
                "note": "chunk_id does not exist in the collection",
            })
            continue

        document = record["documents"][0]
        metadata = record["metadatas"][0]

        contains = verify_token.lower() in document.lower() if verify_token else None

        findings.append({
            "chunk_id": chunk_id,
            "resolves": True,
            "recipe_id_claimed": recipe_id,
            "recipe_id_actual": metadata.get("recipe_id"),
            "recipe_id_matches": recipe_id == metadata.get("recipe_id"),
            "source_file_actual": metadata.get("source_file"),
            "contains_claim": contains,
            "verify_token": verify_token,
            "chunk_text": document,
        })

    return findings


def main():

    with open("eval_questions.json", encoding="utf-8") as handle:
        spec = json.load(handle)

    questions = {q["id"]: q for q in spec["questions"]}
    collection = get_collection(COLLECTION)

    transcript = []
    report = {"answerable": [], "refusals": []}

    transcript.append("=" * 78)
    transcript.append("GROUNDED GENERATION TRANSCRIPTS")
    transcript.append(f"collection: {COLLECTION}   top_k: {TOP_K}")
    transcript.append("model: gemini-3.6-flash")
    transcript.append("=" * 78)

    # ------------------------------------------------ 3 answerable questions
    for qid in spec["answerable_for_generation"]:

        record = questions[qid]
        results = retrieve(collection, record["question"], TOP_K)
        answer = generate_recipe_answer(record["question"], results)

        findings = verify_citations(collection, answer, VERIFY_TOKEN.get(qid, ""))

        transcript.append("")
        transcript.append("=" * 78)
        transcript.append(f"ANSWERABLE {qid}")
        transcript.append("=" * 78)
        transcript.append(f"QUESTION: {record['question']}")
        transcript.append(f"KNOWN ANSWER: {record['expected_answer']}")
        transcript.append("")
        transcript.append("--- MODEL ANSWER ---")
        transcript.append(answer.strip())
        transcript.append("")
        transcript.append("--- CITATION VERIFICATION ---")

        if not findings:
            transcript.append("  NO CITATIONS EMITTED  <-- FAIL")

        for finding in findings:
            transcript.append(f"  chunk_id: {finding['chunk_id']}")
            transcript.append(f"    resolves in ChromaDB : {finding['resolves']}")
            if finding["resolves"]:
                transcript.append(
                    f"    recipe_id matches    : {finding['recipe_id_matches']} "
                    f"(claimed {finding['recipe_id_claimed']}, "
                    f"actual {finding['recipe_id_actual']})"
                )
                transcript.append(
                    f"    contains '{finding['verify_token']}' : {finding['contains_claim']}"
                )
            else:
                transcript.append(f"    note: {finding['note']}")

        report["answerable"].append({
            "id": qid,
            "question": record["question"],
            "answer": answer,
            "citations": [
                {k: v for k, v in f.items() if k != "chunk_text"} for f in findings
            ],
        })

    # ---------------------------------------------- 3 unanswerable questions
    for item in spec["unanswerable_questions"]:

        results = retrieve(collection, item["question"], TOP_K)
        answer = generate_recipe_answer(item["question"], results)

        refused = REFUSAL_TEXT.lower() in answer.lower()

        transcript.append("")
        transcript.append("=" * 78)
        transcript.append(f"OUT-OF-CORPUS {item['id']}")
        transcript.append("=" * 78)
        transcript.append(f"QUESTION: {item['question']}")
        transcript.append(f"WHY UNANSWERABLE: {item['why_unanswerable']}")
        transcript.append("")
        transcript.append("--- RETRIEVED (retrieval still returns chunks) ---")
        for rank, (cid, meta) in enumerate(
            zip(results["ids"][0], results["metadatas"][0]), 1
        ):
            transcript.append(f"  {rank}. {cid}  (recipe_id={meta.get('recipe_id')})")
        transcript.append("")
        transcript.append("--- MODEL ANSWER ---")
        transcript.append(answer.strip())
        transcript.append("")
        transcript.append(f"REFUSED CORRECTLY: {refused}")

        report["refusals"].append({
            "id": item["id"],
            "question": item["question"],
            "answer": answer,
            "refused": refused,
        })

    text = "\n".join(transcript)
    print(text)

    with open("generation_transcripts.txt", "w", encoding="utf-8") as handle:
        handle.write(text + "\n")

    with open("generation_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("\n\nWrote generation_transcripts.txt and generation_report.json")


if __name__ == "__main__":
    main()
