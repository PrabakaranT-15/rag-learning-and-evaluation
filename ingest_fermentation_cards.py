"""Ingest ONLY the 6 new fermentation recipe cards.

Task Set B requirement 6: do not re-index the whole recipe corpus. This script
touches nothing except the 6 cards in fermentation_cards/, and it appends via
vector_store.add_chunks() rather than build_index(), so no existing collection
is cleared.

Three collections are built, with the embedding model held constant across all
three (gemini-embedding-001), so the only variable that moves between them is
the chunker:

  fermentation_baseline_500      Strategy A as the app is actually configured
                                 (rag/chunker.py, chunk_size=500, overlap=100)
  fermentation_baseline_120      Strategy A, same code, granularity matched to
                                 Strategy B so the comparison isolates structure
  fermentation_structure_aware   Strategy B (rag/structure_aware_chunker.py)
"""

import json

from rag.recipe_loader import load_cards, to_pages, metadata_for_source
from rag.chunker import create_chunks
from rag.structure_aware_chunker import create_structure_aware_chunks
from rag.vector_store import (
    add_chunks,
    attach_metadata,
    list_collections,
    chroma_client,
)


CARD_DIR = "fermentation_cards"

BASELINE_500 = "fermentation_baseline_500"
BASELINE_120 = "fermentation_baseline_120"
STRUCTURE_AWARE = "fermentation_structure_aware"


def snapshot_untouched():
    """Record the size of every pre-existing collection, to prove we didn't wipe it."""

    snapshot = {}

    for name in list_collections():
        snapshot[name] = chroma_client.get_collection(name).count()

    return snapshot


def main():

    print("=" * 70)
    print("INGEST: 6 new fermentation cards only")
    print("=" * 70)

    before = snapshot_untouched()
    print("\nCollections BEFORE ingest:")
    for name, count in before.items():
        print(f"  {name}: {count} chunks")

    cards = load_cards(CARD_DIR)
    print(f"\nLoaded {len(cards)} cards from {CARD_DIR}/")
    for card in cards:
        print(f"  {card['recipe_id']}  {card['recipe_name']}  [{card['dietary_tags']}]")

    if len(cards) != 6:
        raise SystemExit(f"Expected exactly 6 cards, found {len(cards)}")

    meta_by_source = metadata_for_source(cards)
    pages = to_pages(cards)

    # ---- Strategy A: the EXISTING chunker, unmodified ---------------------
    baseline_500 = create_chunks(pages, chunk_size=500, overlap=100)
    baseline_120 = create_chunks(pages, chunk_size=120, overlap=20)

    # ---- Strategy B: structure-aware -------------------------------------
    structure = create_structure_aware_chunks(cards, target_words=140)

    runs = [
        (BASELINE_500, attach_metadata(baseline_500, meta_by_source, "baseline_500")),
        (BASELINE_120, attach_metadata(baseline_120, meta_by_source, "baseline_120")),
        (STRUCTURE_AWARE, attach_metadata(structure, meta_by_source, "structure_aware")),
    ]

    report = {}

    for name, chunks in runs:

        print(f"\n--- {name} ---")
        print(f"  chunks produced: {len(chunks)}")

        result = add_chunks(chunks, name)

        print(f"  metadata validation: PASSED (all chunks have "
              f"source_file, recipe_id, cuisine, dietary_tags)")
        print(f"  collection count: {result['count_before']} -> {result['count_after']}")

        report[name] = {
            "chunks": len(chunks),
            "count_after": result["count_after"],
        }

    after = snapshot_untouched()

    print("\n" + "=" * 70)
    print("Collections AFTER ingest:")
    for name, count in after.items():
        delta = count - before.get(name, 0)
        flag = "  <-- pre-existing, UNTOUCHED" if name in before and delta == 0 else ""
        print(f"  {name}: {count} chunks (+{delta}){flag}")

    with open("ingest_report.json", "w", encoding="utf-8") as handle:
        json.dump(
            {"before": before, "after": after, "runs": report},
            handle,
            indent=2,
        )

    print("\nWrote ingest_report.json")


if __name__ == "__main__":
    main()
