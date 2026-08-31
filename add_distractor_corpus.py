"""Build the REALISTIC evaluation condition.

Why this exists
---------------
Task Set B assumes "Your RAG app already indexes the old cards". With only the
6 new fermentation cards in the index, the corpus is 10-33 chunks and top-5
retrieval returns a third of everything, so Hit@5 came back 8/8 for every
strategy and measured nothing. That is a degenerate experiment, not a result.

This script adds a distractor set drawn from the existing 195-card recipe
corpus so the 6 new cards have to compete. The 6 new cards are still the only
thing being *measured*; the distractors are background, exactly as the old
cards would already have been in a real deployment.

Distractor selection is deterministic and documented, not cherry-picked:

  1. FORCED: every card in the existing corpus that is fermentation-adjacent
     or bread-adjacent, because those are the genuine semantic competitors
     for these 8 questions (fermented batters, yeasted doughs, kimchi).
  2. SAMPLED: disabled (SAMPLE_EVERY=0). The forced set above IS the hard
     distractor set; bulk random cards add volume but little competition.

Nothing here re-indexes the fermentation collections; this appends to NEW
collection names so the requirement-6-compliant 6-card-only collections stay
exactly as they were.
"""

import os
import glob

from rag.recipe_loader import load_cards, load_recipe_card, to_pages, metadata_for_source
from rag.chunker import create_chunks
from rag.structure_aware_chunker import create_structure_aware_chunks
from rag.vector_store import add_chunks, attach_metadata


FORCED = [
    "recipe_001_masala_dosa.md",
    "recipe_002_plain_dosa.md",
    "recipe_003_idli.md",
    "recipe_016_appam.md",
    "recipe_018_adai.md",
    "recipe_025_onion_uttapam.md",
    "recipe_020_kerala_parotta.md",
    "recipe_038_butter_naan.md",
    "recipe_039_roti.md",
    "recipe_040_aloo_paratha.md",
    "recipe_056_margherita_pizza.md",
    "recipe_066_focaccia.md",
    "recipe_067_ciabatta.md",
    "recipe_078_pizza_marinara.md",
    "recipe_081_calzone.md",
    "recipe_169_napa_cabbage_kimchi.md",
    "recipe_162_kimchi_fried_rice.md",
    "recipe_166_kimchi_jjigae.md",
    "recipe_192_cornbread.md",
    "recipe_195_blueberry_muffins.md",
]

SAMPLE_EVERY = 0   # 0 = forced set only (see note below)


def select_distractors():

    all_paths = sorted(glob.glob(os.path.join("recipes", "*", "*.md")))
    by_name = {os.path.basename(p): p for p in all_paths}

    chosen = []
    seen = set()

    for name in FORCED:
        if name in by_name:
            chosen.append(by_name[name])
            seen.add(name)
        else:
            print(f"  WARNING forced card not found: {name}")

    if SAMPLE_EVERY:
        remaining = [p for p in all_paths if os.path.basename(p) not in seen]
        for i, path in enumerate(remaining):
            if i % SAMPLE_EVERY == 0:
                chosen.append(path)

    return sorted(set(chosen))


def main():

    paths = select_distractors()
    print(f"Distractor cards selected: {len(paths)}")
    print(f"  forced fermentation/bread-adjacent: {len(FORCED)}")
    print(f"  deterministic sample: {len(paths) - len(FORCED)} "
          f"(SAMPLE_EVERY={SAMPLE_EVERY})")

    distractors = [load_recipe_card(p) for p in paths]
    new_cards = load_cards("fermentation_cards")

    combined = new_cards + distractors
    print(f"Total cards in realistic condition: {len(combined)} "
          f"(6 new + {len(distractors)} distractors)")

    meta_by_source = metadata_for_source(combined)
    pages = to_pages(combined)

    runs = [
        ("realistic_baseline_500",
         attach_metadata(create_chunks(pages, 500, 100), meta_by_source, "baseline_500")),
        ("realistic_baseline_120",
         attach_metadata(create_chunks(pages, 120, 20), meta_by_source, "baseline_120")),
        ("realistic_structure_aware",
         attach_metadata(create_structure_aware_chunks(combined, 140), meta_by_source,
                         "structure_aware")),
    ]

    for name, chunks in runs:
        print(f"\n--- {name}: {len(chunks)} chunks ---")
        result = add_chunks(chunks, name)
        print(f"  {result['count_before']} -> {result['count_after']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
