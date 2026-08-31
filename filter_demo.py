"""Demonstrate a dietary_tags metadata filter that changes the top-1 result.

The corpus deliberately contains two near-identical kimchi cards:

  ferment_005  Traditional Napa Cabbage Kimchi  (non-vegetarian: saeujeot + fish sauce)
  ferment_006  Vegan Napa Cabbage Kimchi        (vegan: shiitake + kombu + tamari)

They are semantically almost the same document, so an unfiltered vector search
for a kimchi question has no way to prefer one over the other. Filtering on
dietary tags is the only thing that can, which is exactly the point of the
requirement.

Note on the filter field: ChromaDB's where-clause has no substring operator, so
the comma-joined `dietary_tags` string cannot be matched partially. At ingest
time vector_store.diet_flags() derives a boolean per tag from that exact string
(vegan -> diet_vegan=True). The filter below therefore still filters on the
dietary tags, via a Chroma-compatible representation of them.
"""

import json

from rag.vector_store import get_collection, retrieve


COLLECTION = "fermentation_structure_aware"
QUERY = "How do I make fermented napa cabbage with chilli flakes and radish?"
TOP_K = 5
FILTER = {"diet_vegan": True}


def show(title, results):

    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    lines = [title, "-" * len(title)]

    for rank, (cid, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists), 1):
        lines.append(
            f"  {rank}. chunk_id={cid}\n"
            f"     score(distance)={float(dist):.4f}   recipe_id={meta['recipe_id']}\n"
            f"     recipe_name={meta.get('recipe_name')}\n"
            f"     dietary_tags={meta['dietary_tags']}\n"
            f"     source_file={meta['source_file']}"
        )

    text = "\n".join(lines)
    print(text)
    return text, (ids[0] if ids else None), (metas[0] if metas else None)


def main():

    collection = get_collection(COLLECTION)

    print(f"Collection : {COLLECTION}")
    print(f"Query      : {QUERY}")
    print(f"Filter     : {FILTER}\n")

    unfiltered = retrieve(collection, QUERY, TOP_K)
    text_u, top_u, meta_u = show("UNFILTERED (no where-clause)", unfiltered)

    print()

    filtered = retrieve(collection, QUERY, TOP_K, where=FILTER)
    text_f, top_f, meta_f = show("FILTERED (where dietary tags include vegan)", filtered)

    changed = top_u != top_f

    print("\n" + "=" * 70)
    print(f"top-1 unfiltered : {top_u}  ({meta_u['recipe_id']}, {meta_u['dietary_tags']})")
    print(f"top-1 filtered   : {top_f}  ({meta_f['recipe_id']}, {meta_f['dietary_tags']})")
    print(f"TOP-1 CHANGED    : {changed}")

    with open("filter_demo.txt", "w", encoding="utf-8") as handle:
        handle.write(
            f"Collection: {COLLECTION}\nQuery: {QUERY}\nFilter: {FILTER}\n\n"
            f"{text_u}\n\n{text_f}\n\n"
            f"top-1 unfiltered: {top_u} ({meta_u['recipe_id']}, {meta_u['dietary_tags']})\n"
            f"top-1 filtered:   {top_f} ({meta_f['recipe_id']}, {meta_f['dietary_tags']})\n"
            f"TOP-1 CHANGED: {changed}\n"
        )

    print("\nWrote filter_demo.txt")


if __name__ == "__main__":
    main()
