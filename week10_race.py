"""Week 10 Module 5, Task Set B: race the kitchen squad (rag/orchestrator.py)
against the single agent (rag/agent.py) on the SAME 10 Week-6 eval cases.

Reuses 10 of the 20 questions from week5_trace_questions.json /
week6_report.json verbatim - not a new eval set (the brief's own "common
mistake" #2 is building a fresh set because the old one "doesn't suit the
orchestrator"; reusing the frozen T-ids is the whole point of the
comparison). The 10 below were chosen for a genuine mix, not cherry-picked
for either arm to look good:

  T01, T02, T06, T18   - plain retrieval, no substitution/allergen angle
  T04, T13             - substitution-relevant (kimchi vegan swap, fish sauce)
  T07, T12, T14        - allergen/nutrition-relevant (calories, gluten, dairy)
  T20                  - a reasoning case (smell -> technical term)

Writes: race_table.md, handoffs.log, failure_case.md, verdict.md, plus the
bonus agent_card.json / a2a_mapping.md - all five to the repo root, same
convention as every other week's *_report.* artifact.

Partial results are written after every case (same pattern as
week8_injection_attack.py's _write()) so a rate limit partway through does
not discard real evidence already collected.
"""

import json
import statistics
import time

from rag import generator
from rag.agent import run_agent
from rag.generator import WEEK4_MODEL, cost_of
from rag.judge import judge_answer
from rag.orchestrator import run_orchestrator
from rag.trajectory_eval import _percentile
from rag.vector_store import get_collection


COLLECTION = "fermentation_structure_aware"
JUDGE_MODEL = WEEK4_MODEL

CASE_IDS = ["T01", "T02", "T04", "T06", "T07", "T12", "T13", "T14", "T18", "T20"]

# Week 10 rubric requirement #4: force the allergen worker to fail on ONE
# case. T07 ("how many calories...") is chosen deliberately - no recipe
# card carries nutrition data (established fact from results.md/week5), so
# this is the one case in the set where the allergen/nutrition specialist
# is the ONLY possible source of an answer, making the injected failure a
# real test rather than one the recipe text alone could paper over.
FAILURE_CASE_ID = "T07"

# app.py's own restriction-detection keywords, duplicated here rather than
# imported (app.py is a Streamlit entrypoint script, not a module meant to
# be imported) - kept applied identically to BOTH arms so a restriction-
# bearing question isn't fed to one arm and not the other.
RESTRICTION_KEYWORDS = [
    "vegan", "vegetarian", "dairy-free", "gluten-free",
    "egg-free", "nut-free", "non-vegetarian",
]


def detect_restriction(question):
    lowered = question.lower()
    return next((r for r in RESTRICTION_KEYWORDS if r in lowered), None)


def _load_questions():
    data = json.load(open("week5_trace_questions.json", encoding="utf-8"))
    by_id = {q["id"]: q["question"] for q in data["questions"]}
    return {cid: by_id[cid] for cid in CASE_IDS}


def _slice_usage(before):
    """Real tokens + cost for every generator._generate() call made since
    index `before` in generator.call_log - the same before/after slicing
    idiom run_agent() already uses for llm_calls, extended to the token/
    cost fields Week 10 actually needs (see rag/generator.py's Week 10
    addition to call_log)."""

    calls = generator.call_log[before:]
    tokens = sum((c.get("total_tokens") or 0) for c in calls)
    cost = sum((c.get("cost_usd") or 0.0) for c in calls)
    return tokens, cost


def run_single_agent_case(question, restriction):
    calls_before = len(generator.call_log)
    collection = get_collection(COLLECTION)

    result = run_agent(collection, question, restriction)
    tokens, cost = _slice_usage(calls_before)

    # Context for the judge must include everything the ANSWER was actually
    # grounded in - not just get_recipe's text. run_agent()'s own
    # _with_extra_context() (rag/agent.py) folds any non-core tool's real
    # output (e.g. get_substitutes, check_allergens - anything outside
    # CORE_RECIPE_TOOLS) into the final generation as legitimate grounding
    # material and the model correctly cites it
    # ([source_file=mcp-tool:get_substitutes]); a judge context missing
    # those blocks cannot see what a well-grounded answer actually relied
    # on and will wrongly fail it as unsupported. Mirrors CORE_RECIPE_TOOLS
    # exactly so "extra" here means the same thing it means in agent.py.
    CORE_RECIPE_TOOLS = {"search_recipes", "check_restriction", "get_recipe", "finish", "_abort"}

    recipe_id = None
    context_parts = []
    for step in result["steps"]:
        if step["tool"] == "get_recipe" and isinstance(step["observation"], dict):
            recipe_id = step["observation"].get("recipe_id")
            context_parts.append(f"RECIPE TEXT:\n{step['observation'].get('text', '')}")
        elif step["tool"] not in CORE_RECIPE_TOOLS:
            observation = step["observation"]
            if isinstance(observation, dict) and "error" not in observation:
                context_parts.append(f"TOOL {step['tool']} RESULT:\n{json.dumps(observation)}")

    context = "\n\n".join(context_parts) if context_parts else "(no recipe found)"

    verdict = judge_answer(question, context, result["answer"], model=JUDGE_MODEL)

    return {
        "arm": "single_agent",
        "answer": result["answer"],
        "recipe_id": recipe_id,
        "elapsed_seconds": result["elapsed_seconds"],
        "tokens": tokens,
        "cost_usd": cost,
        "verdict": verdict["verdict"],
        "judge_reason": verdict["reason"],
        "handoffs": [],
    }


def run_orchestrator_case(question, restriction, force_allergen_failure=False):
    calls_before = len(generator.call_log)
    collection = get_collection(COLLECTION)

    result = run_orchestrator(collection, question, restriction, force_allergen_failure=force_allergen_failure)
    tokens, cost = _slice_usage(calls_before)

    substitution_note = next((h["note"] for h in result["handoffs"] if h["handoff"] == "manager->substitution_worker"), "")
    allergen_note = next((h["note"] for h in result["handoffs"] if h["handoff"] == "manager->allergen_worker"), "")
    context = (
        f"RECIPE TEXT (recipe_id={result.get('recipe_id')}):\n{result.get('recipe_text', '')}\n\n"
        f"SUBSTITUTION WORKER NOTE:\n{substitution_note}\n\n"
        f"ALLERGEN WORKER NOTE:\n{allergen_note}"
    )

    verdict = judge_answer(question, context, result["answer"], model=JUDGE_MODEL)

    handoff_log = [
        {"handoff": h["handoff"], "tokens": h["tokens"]}
        for h in result["handoffs"]
    ]

    return {
        "arm": "orchestrator",
        "answer": result["answer"],
        "recipe_id": result.get("recipe_id"),
        "elapsed_seconds": result["elapsed_seconds"],
        "tokens": tokens,
        "cost_usd": cost,
        "verdict": verdict["verdict"],
        "judge_reason": verdict["reason"],
        "handoffs": handoff_log,
        "specialist_notes": {"substitution": substitution_note, "allergen": allergen_note},
    }


def _write_partial(results, complete):
    with open("week10_race_raw.json", "w", encoding="utf-8") as fh:
        json.dump({"complete": complete, "results": results}, fh, indent=2)


def main():
    questions = _load_questions()
    results = {}

    try:
        for case_id, question in questions.items():
            restriction = detect_restriction(question)
            print(f"[{case_id}] {question!r} (restriction={restriction!r})", flush=True)

            single = run_single_agent_case(question, restriction)
            print(f"  single_agent: {single['verdict']} - {single['tokens']} tokens, "
                  f"{single['elapsed_seconds']:.1f}s", flush=True)

            multi = run_orchestrator_case(question, restriction)
            print(f"  orchestrator: {multi['verdict']} - {multi['tokens']} tokens, "
                  f"{multi['elapsed_seconds']:.1f}s", flush=True)

            results[case_id] = {"question": question, "restriction": restriction, "single_agent": single, "orchestrator": multi}
            _write_partial(results, complete=False)

        # Rubric requirement #4: the injected failure, as a SEPARATE
        # one-off experiment on top of the frozen 10-case race above (not
        # baked into the race_table numbers - it is its own deliverable,
        # failure_case.md).
        print(f"\n[FAILURE INJECTION] {FAILURE_CASE_ID} with allergen worker forced to 500", flush=True)
        failure_question = questions[FAILURE_CASE_ID]
        failure_restriction = detect_restriction(failure_question)
        failure_result = run_orchestrator(
            get_collection(COLLECTION), failure_question, failure_restriction, force_allergen_failure=True
        )
        results["_failure_injection"] = {
            "case_id": FAILURE_CASE_ID, "question": failure_question, "result": failure_result,
        }
        _write_partial(results, complete=True)

    except Exception:
        _write_partial(results, complete=False)
        print(f"\nCRASHED after {len(results)} case(s) - wrote partial results to week10_race_raw.json")
        raise

    print("\nAll cases complete. Wrote week10_race_raw.json")
    return results


if __name__ == "__main__":
    main()
