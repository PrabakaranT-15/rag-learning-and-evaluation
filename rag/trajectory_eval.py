"""Week 8 Module 4: trajectory evaluation for rag/agent.py's run_agent().

Judges the WHOLE path an agent took, not just its final answer - the
"outcome vs. trajectory gap" the brief asks for. Every check here is a plain
function over a transcript's recorded (tool, args, observation) steps, in
the same "never trust the model's own claim, only what's actually in the
data" spirit as rag/evaluation.py's classify(): no LLM judge, because these
are all mechanically checkable from the transcript alone.

THE CORE FINDING THIS MODULE EXISTS TO CATCH
----------------------------------------------
run_agent()'s `finish` handling (before the Week 8 fix) trusted the model's
own decision to call finish(recipe_id) - the planner's system prompt SAYS
"never finish without a passing check_restriction", but nothing in the
Python code verified that actually happened. That gap is exactly the
outcome-vs-trajectory failure this module is built to detect:
finished_without_verification() flags a run that reached a (maybe correct)
final answer via a path that skipped real verification - right today on an
easy case, silently wrong the day a similar-looking recipe genuinely isn't
compliant.
"""


import math
import statistics

from rag.tools import known_dietary_tags_by_recipe, restriction_verified


def finished_recipe_id(transcript):
    """The recipe_id the agent actually finished on, or None if it never
    reached a finish step, or refused (finish with recipe_id=null)."""

    for step in transcript:
        if step.get("tool") == "finish":
            return (step.get("args") or {}).get("recipe_id")
    return None


def finished_without_verification(transcript, restriction):
    """The core outcome-vs-trajectory check: did the agent finish on a real
    recipe_id, with a restriction in play, WITHOUT that finish being backed
    by a real, passing check_restriction call (rag.tools.restriction_verified
    - the SAME definition rag/agent.py's live finish-gate enforces)?

    Never flags a restriction-less run (nothing to verify) or an honest
    refusal (finish with recipe_id=None) - a violation requires an actual
    finished recipe_id plus a restriction that was never really checked
    against it. Runs collected before the Week 8 fix was added to
    rag/agent.py can still show this as True - the gate didn't exist yet to
    stop them; that's exactly the gap this function exists to surface."""

    if not restriction:
        return False

    recipe_id = finished_recipe_id(transcript)
    if not recipe_id:
        return False

    return not restriction_verified(transcript, recipe_id, restriction)


def hallucinated_check_input(transcript):
    """check_restriction calls whose dietary_tags argument doesn't match ANY
    dietary_tags string the agent actually observed via search_recipes -
    i.e. it wasn't quoting real retrieved data, it made the tags string up
    (or copied it from the wrong place). A diagnostic signal, not a formal
    proof - it flags "unattributable", not necessarily "wrong"."""

    seen_tags = set(known_dietary_tags_by_recipe(transcript).values())

    violations = []
    for i, step in enumerate(transcript, start=1):
        if step.get("tool") != "check_restriction":
            continue
        used_tags = (step.get("args") or {}).get("dietary_tags")
        if used_tags and used_tags not in seen_tags:
            violations.append({"step": i, "dietary_tags": used_tags})
    return violations


def tool_choice_violations(transcript):
    """Steps whose tool choice was structurally unreasonable given the
    state built up so far - not a subjective judgement, just: don't check a
    restriction before anything has been searched, don't fetch a recipe
    nobody has seen yet."""

    violations = []
    known_recipe_ids = set()

    for i, step in enumerate(transcript, start=1):
        tool = step.get("tool")

        if tool == "check_restriction" and not known_recipe_ids:
            violations.append({
                "step": i,
                "reason": "checked a restriction before any search_recipes call",
            })

        if tool == "get_recipe":
            recipe_id = (step.get("args") or {}).get("recipe_id")
            if recipe_id and recipe_id not in known_recipe_ids:
                violations.append({
                    "step": i,
                    "reason": f"fetched recipe_id={recipe_id!r} never seen in a prior search",
                })

        if tool == "search_recipes":
            observation = step.get("observation")
            if isinstance(observation, list):
                for candidate in observation:
                    recipe_id = candidate.get("recipe_id") if isinstance(candidate, dict) else None
                    if recipe_id:
                        known_recipe_ids.add(recipe_id)

    return violations


def tool_choice_accuracy(transcript):
    """Fraction of real (non-_abort) steps that were NOT a tool_choice_violations
    hit. None for an empty transcript, never a divide-by-zero guess."""

    real_steps = [s for s in transcript if s.get("tool") != "_abort"]
    if not real_steps:
        return None

    return 1 - (len(tool_choice_violations(transcript)) / len(real_steps))


def _percentile(values, pct):
    """Linear-interpolation percentile - no numpy dependency needed for one
    metric used in one weekly report."""

    if not values:
        return None

    ordered = sorted(values)
    rank = (len(ordered) - 1) * (pct / 100)
    lower, upper = math.floor(rank), math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def cost_stats(runs):
    """Mean and p99 of llm_calls and elapsed_seconds across a batch of
    run_agent() results (or any dicts carrying those two keys) - the "cost
    per task (mean & p99)" metric named in the Week 8 topics list."""

    calls = [r["llm_calls"] for r in runs]
    seconds = [r["elapsed_seconds"] for r in runs]

    return {
        "n": len(runs),
        "llm_calls_mean": statistics.fmean(calls) if calls else None,
        "llm_calls_p99": _percentile(calls, 99),
        "elapsed_seconds_mean": statistics.fmean(seconds) if seconds else None,
        "elapsed_seconds_p99": _percentile(seconds, 99),
    }
