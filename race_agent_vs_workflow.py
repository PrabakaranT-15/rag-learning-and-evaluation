"""Week 7 Module 4 deliverable: race the recipe agent (rag/agent.py) against
the fixed workflow (rag/fixed_workflow.py) on the same scenarios, and report
real numbers - wall-clock time, LLM call count (cost proxy), and whether the
final answer is actually correct - so "which would you ship, and why" has
evidence behind it instead of a guess.

Uses WEEK4_MODEL rather than DEFAULT_MODEL for the same reason
week6_run_evals.py does: DEFAULT_MODEL's free-tier quota is a hard 20
generate_content calls/day shared with every other script, and one agent
run alone can burn several of those on planning steps.

Run with: python race_agent_vs_workflow.py
"""

import json
import time

from rag import generator
from rag.generator import WEEK4_MODEL
from rag.vector_store import get_collection
from rag.agent import run_agent
from rag.fixed_workflow import run_fixed_workflow


COLLECTION = "fermentation_structure_aware"

# compliant_recipe_ids is the ground truth (from fermentation_cards/*.md
# frontmatter, not shown to either method) used only to grade the answers
# after the fact - never fed into search_recipes/check_restriction.
SCENARIOS = [
    {
        "name": "kimchi_vegan_trap",
        "query": "napa cabbage kimchi with chilli flakes and radish",
        "restriction": "vegan",
        "compliant_recipe_ids": {"ferment_006"},
        "note": "filter_demo.py already showed unfiltered top-1 is the non-vegan ferment_005 - "
                "the fixed workflow only avoids this because it filters up front.",
    },
    {
        "name": "brioche_dairy_free_no_match",
        "query": "sourdough brioche",
        "restriction": "dairy-free",
        "compliant_recipe_ids": set(),
        "note": "the only brioche card in the corpus contains dairy - correct behaviour is a refusal.",
    },
    {
        "name": "sourdough_vegan_straightforward",
        "query": "country sourdough bread",
        "restriction": "vegan",
        "compliant_recipe_ids": {"ferment_001"},
        "note": "already compliant on the obvious top candidate - tests agent overhead when no retry is needed.",
    },
    {
        "name": "rye_levain_nut_free_straightforward",
        "query": "rye levain starter",
        "restriction": "nut-free",
        "compliant_recipe_ids": {"ferment_002"},
        "note": "straightforward pass.",
    },
]


def _graded(answer, compliant_ids):
    """Evidence-based grading, same spirit as rag/evaluation.py: check what
    is actually IN the answer text, never trust a method's own claim."""

    lowered = (answer or "").lower()

    if not compliant_ids:
        return any(p in lowered for p in ("could not find", "no recipe", "cannot", "not satisf"))

    return any(rid in lowered for rid in compliant_ids)


def run_one(method_name, run_fn, scenario):

    calls_before = len(generator.call_log)
    start = time.monotonic()

    result = run_fn()

    elapsed = time.monotonic() - start
    llm_calls = len(generator.call_log) - calls_before
    correct = _graded(result["answer"], scenario["compliant_recipe_ids"])

    return {
        "method": method_name,
        "scenario": scenario["name"],
        "elapsed_seconds": round(elapsed, 2),
        "llm_calls": llm_calls,
        "correct": correct,
        "answer": result["answer"],
        "steps": result.get("steps"),
        "stopped_reason": result.get("stopped_reason"),
    }


def main():

    collection = get_collection(COLLECTION)
    rows = []

    for scenario in SCENARIOS:

        print(f"\n=== {scenario['name']} ===")
        print(f"query: {scenario['query']!r}  restriction: {scenario['restriction']!r}")
        print(f"note: {scenario['note']}")

        print("  running fixed workflow...")
        fixed_row = run_one(
            "fixed_workflow",
            lambda s=scenario: run_fixed_workflow(collection, s["query"], s["restriction"], model=WEEK4_MODEL),
            scenario,
        )
        rows.append(fixed_row)
        print(f"    {fixed_row['elapsed_seconds']}s  {fixed_row['llm_calls']} LLM call(s)  "
              f"correct={fixed_row['correct']}")

        print("  running agent...")
        agent_row = run_one(
            "agent",
            lambda s=scenario: run_agent(collection, s["query"], s["restriction"], model=WEEK4_MODEL),
            scenario,
        )
        rows.append(agent_row)
        print(f"    {agent_row['elapsed_seconds']}s  {agent_row['llm_calls']} LLM call(s)  "
              f"correct={agent_row['correct']}  stopped={agent_row['stopped_reason']}")

    with open("week7_race_report.json", "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    print("\n" + "=" * 78)
    print(f"{'scenario':<34}{'method':<16}{'time(s)':>9}{'calls':>8}{'correct':>9}")
    for row in rows:
        print(f"{row['scenario']:<34}{row['method']:<16}{row['elapsed_seconds']:>9}"
              f"{row['llm_calls']:>8}{str(row['correct']):>9}")

    fixed_rows = [r for r in rows if r["method"] == "fixed_workflow"]
    agent_rows = [r for r in rows if r["method"] == "agent"]

    print("\nTOTALS")
    print(f"  fixed_workflow: {sum(r['elapsed_seconds'] for r in fixed_rows):.1f}s total, "
          f"{sum(r['llm_calls'] for r in fixed_rows)} LLM calls, "
          f"{sum(r['correct'] for r in fixed_rows)}/{len(fixed_rows)} correct")
    print(f"  agent:          {sum(r['elapsed_seconds'] for r in agent_rows):.1f}s total, "
          f"{sum(r['llm_calls'] for r in agent_rows)} LLM calls, "
          f"{sum(r['correct'] for r in agent_rows)}/{len(agent_rows)} correct")

    print("\nWrote week7_race_report.json")


if __name__ == "__main__":
    main()
