"""Week 8 Module 4: collect a batch of REAL agent trajectories to analyze for
the outcome-vs-trajectory gap - patterned on week5_collect_traces.py's "run
the real path, save everything, analyze later" shape.

Uses BATCH_MODEL (DEFAULT_MODEL / gpt-oss-120b), NOT WEEK4_MODEL - a
disclosed reversal of every prior week's convention, forced by real
circumstance: WEEK4_MODEL (gpt-oss-20b)'s free-tier 200k-tokens/day budget
was fully exhausted collecting this same batch earlier today (see
week8_collect_before.log / week8_collect.log), and Groq bills each model's
daily quota separately - DEFAULT_MODEL's pool was untouched. Both before-
and after-fix batches in this Week 8 deliverable use DEFAULT_MODEL
consistently, so the comparison stays apples-to-apples; only the choice of
WHICH model differs from earlier weeks, not the requirement to hold the
model constant across a before/after pair.

SCENARIOS extends race_agent_vs_workflow.py's 4-scenario set (kept for
continuity with the Week 7 data) with new cases chosen to probe specifically
for the outcome-vs-trajectory gap: straightforward passes where skipping
verification would still "look" correct, multiple dietary axes (not just
vegan), a short/ambiguous query, restriction-less questions, and one case
where two different recipes are BOTH legitimately compliant (not a trap -
noted explicitly, since the point there is watching what the agent does
with real ambiguity, not catching a wrong answer).

Ground truth (dietary_tags) is copied verbatim from the fermentation_cards/
frontmatter, not guessed - see each card's YAML header.

Run with: python week8_collect_trajectories.py
"""

import json
import time

from rag import generator
from rag.generator import DEFAULT_MODEL as BATCH_MODEL
from rag.vector_store import get_collection
from rag.agent import run_agent
from rag.trajectory_eval import finished_recipe_id


COLLECTION = "fermentation_structure_aware"

SCENARIOS = [
    # --- the 4 original Week 7 race scenarios, kept for continuity ---
    {
        "name": "kimchi_vegan_trap",
        "query": "napa cabbage kimchi with chilli flakes and radish",
        "restriction": "vegan",
        "compliant_recipe_ids": {"ferment_006"},
        "note": "unfiltered top-1 is the non-vegan ferment_005 - only correct if actually verified.",
    },
    {
        "name": "brioche_dairy_free_no_match",
        "query": "sourdough brioche",
        "restriction": "dairy-free",
        "compliant_recipe_ids": set(),
        "note": "the only brioche card contains dairy - correct behaviour is a refusal.",
    },
    {
        "name": "sourdough_vegan_straightforward",
        "query": "country sourdough bread",
        "restriction": "vegan",
        "compliant_recipe_ids": {"ferment_001"},
        "note": "already compliant on the obvious top candidate - tempts skipping verification.",
    },
    {
        "name": "rye_levain_nut_free_straightforward",
        "query": "rye levain starter",
        "restriction": "nut-free",
        "compliant_recipe_ids": {"ferment_002"},
        "note": "straightforward pass.",
    },
    # --- new Week 8 probing cases ---
    {
        "name": "focaccia_vegan_straightforward",
        "query": "sourdough focaccia",
        "restriction": "vegan",
        "compliant_recipe_ids": {"ferment_003"},
        "note": "another easy pass - tempts skipping verification.",
    },
    {
        "name": "brioche_vegan_no_match",
        "query": "sourdough brioche",
        "restriction": "vegan",
        "compliant_recipe_ids": set(),
        "note": "brioche contains dairy+egg - not vegan; refusal expected.",
    },
    {
        "name": "kimchi_vegetarian_trap",
        "query": "napa cabbage kimchi",
        "restriction": "vegetarian",
        "compliant_recipe_ids": {"ferment_006"},
        "note": "same trap shape as kimchi_vegan_trap but a different restriction axis - "
                "ferment_005 is tagged non-vegetarian.",
    },
    {
        "name": "rye_levain_egg_free_straightforward",
        "query": "rye levain starter",
        "restriction": "egg-free",
        "compliant_recipe_ids": {"ferment_002"},
        "note": "straightforward pass, different axis.",
    },
    {
        "name": "rye_levain_no_restriction",
        "query": "what's a good starter recipe for rye levain?",
        "restriction": None,
        "compliant_recipe_ids": {"ferment_002"},
        "note": "no restriction to verify - tests the agent's search-and-judge behaviour alone.",
    },
    {
        "name": "focaccia_no_restriction",
        "query": "how do I make sourdough focaccia?",
        "restriction": None,
        "compliant_recipe_ids": {"ferment_003"},
        "note": "no restriction to verify.",
    },
    {
        "name": "kimchi_short_ambiguous_vegan_trap",
        "query": "kimchi",
        "restriction": "vegan",
        "compliant_recipe_ids": {"ferment_006"},
        "note": "same trap, shorter/more ambiguous query - stresses retrieval, not just verification.",
    },
    {
        "name": "brioche_egg_free_no_match",
        "query": "sourdough brioche",
        "restriction": "egg-free",
        "compliant_recipe_ids": set(),
        "note": "brioche contains egg - refusal expected.",
    },
    {
        "name": "sourdough_gluten_free_no_match",
        "query": "country sourdough bread",
        "restriction": "gluten-free",
        "compliant_recipe_ids": set(),
        "note": "the corpus has no gluten-free bread - refusal expected.",
    },
    {
        "name": "kimchi_dairy_free_either_ok",
        "query": "napa cabbage kimchi",
        "restriction": "dairy-free",
        "compliant_recipe_ids": {"ferment_005", "ferment_006"},
        "note": "NOT a trap - both kimchi cards are tagged dairy-free, so either is a "
                "legitimately correct answer. Included to watch what the agent does with "
                "genuine ambiguity, not to catch a wrong pick.",
    },
]


def run_one(scenario, collection):

    calls_before = len(generator.call_log)
    start = time.monotonic()

    result = run_agent(collection, scenario["query"], scenario["restriction"], model=BATCH_MODEL)

    recipe_id = finished_recipe_id(result["steps"])
    compliant = scenario["compliant_recipe_ids"]
    correct = (recipe_id in compliant) if compliant else (recipe_id is None)

    return {
        "name": scenario["name"],
        "query": scenario["query"],
        "restriction": scenario["restriction"],
        "compliant_recipe_ids": sorted(compliant),
        "note": scenario["note"],
        "answer": result["answer"],
        "steps": result["steps"],
        "stopped_reason": result["stopped_reason"],
        "llm_calls": result["llm_calls"],
        "elapsed_seconds": round(result["elapsed_seconds"], 2),
        "finished_recipe_id": recipe_id,
        "correct": correct,
    }


def _write(runs, complete):
    """Save whatever has been collected so far - called after every single
    scenario, not just at the end. A live-API batch like this can be killed
    by a rate limit or quota exhaustion partway through; without incremental
    saves, a crash on scenario 9 would silently discard 8 scenarios' worth
    of real, already-paid-for transcripts. `complete` records whether this
    is the full intended batch or a partial one, so a reader of the file
    never mistakes a partial run for the whole thing."""

    with open("week8_trajectories.json", "w", encoding="utf-8") as handle:
        json.dump(
            {"collection": COLLECTION, "model": BATCH_MODEL, "complete": complete, "runs": runs},
            handle, indent=2,
        )


def main():

    collection = get_collection(COLLECTION)
    runs = []

    try:
        for i, scenario in enumerate(SCENARIOS, start=1):
            print(f"[{i}/{len(SCENARIOS)}] {scenario['name']}: {scenario['query']!r} "
                  f"(restriction={scenario['restriction']!r})")

            run = run_one(scenario, collection)
            runs.append(run)

            print(f"    finished_recipe_id={run['finished_recipe_id']!r}  "
                  f"correct={run['correct']}  steps={len(run['steps'])}  "
                  f"calls={run['llm_calls']}  {run['elapsed_seconds']}s  "
                  f"stopped={run['stopped_reason']}")

            _write(runs, complete=False)

    except Exception:
        _write(runs, complete=False)
        print(f"\nCRASHED after {len(runs)}/{len(SCENARIOS)} scenarios - "
              f"wrote what was collected so far to week8_trajectories.json (complete=false)")
        raise

    _write(runs, complete=True)

    correct_count = sum(1 for r in runs if r["correct"])
    print(f"\n{correct_count}/{len(runs)} scenarios reached the expected outcome.")
    print("Wrote week8_trajectories.json")


if __name__ == "__main__":
    main()
