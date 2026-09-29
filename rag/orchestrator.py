"""Week 10 Module 5: a manager + two narrow specialists, raced against
rag/agent.py's single agent on the SAME test set (see week10_race.py).

THE PATTERN
-----------
manager (this module's run_orchestrator) -> always delegates to BOTH
specialists, regardless of whether this particular question needs either
one -> each specialist re-sends the recipe's own text as part of its own
prompt (a real, separate LLM call - a "hand-off") -> manager synthesises a
final answer from the recipe text plus both specialists' notes.

That "always delegates to both" is deliberate, not a missed optimisation:
the brief's own point is measuring the REAL cost of a fixed decomposition
pattern, the same way rag/fixed_workflow.py measures a fixed sequence
against rag/agent.py's agentic one. A manager that cleverly skipped a
specialist when it judged one unnecessary would no longer be testing "does
this fixed multi-agent shape pay for itself" - it would be reinventing the
single agent's own job (deciding, per question, whether a step is needed)
one level up.

NARROW SPECIALISTS, ON PURPOSE
------------------------------
Each specialist can only call ITS OWN one or two tools (SUBSTITUTION_TOOLS /
ALLERGEN_TOOLS below), enforced by this module, not by the shared
MCPToolRegistry (rag/mcp_client.py) - the registry stays generic and serves
every caller identically; narrowing which of its tools any one caller may
use is a per-caller policy, not the registry's job. This existing to be
enforced at all is the direct answer to Week 10's own explicit warning:
"giving the substitution worker every tool the single agent had ... deletes
the narrow-prompt-fewer-tools constraint that was the only plausible source
of a win."
"""

import json
import time
from pathlib import Path

from rag import generator
from rag.generator import DEFAULT_MODEL, REFUSAL_TEXT, generate_recipe_answer
from rag.mcp_client import MCPToolRegistry
from rag.tools import get_recipe, restriction_where, search_recipes


_INGREDIENTS_PATH = Path(__file__).resolve().parent.parent / "data" / "ingredients.json"

# Same reasoning as rag/agent.py's _get_registry(): one MCPToolRegistry per
# process, reused across every run_orchestrator() call.
_registry = None


def _get_registry():
    global _registry
    if _registry is None:
        _registry = MCPToolRegistry()
    return _registry


def _known_ingredient_names():
    catalog = json.loads(_INGREDIENTS_PATH.read_text(encoding="utf-8"))
    names = []
    for entry in catalog:
        names.append(entry["name"])
        names.extend(entry.get("aliases", []))
    return names


def _extract_known_ingredients(text, known_names=None, limit=None):
    """Ingredient names from the ingredient-database catalog that literally
    appear in `text` (case-insensitive), in catalog order - how a narrow
    specialist decides which real database entries are even relevant to a
    given recipe/question, without parsing the recipe's ingredient table
    itself. Returns at most `limit` names if given."""

    known_names = known_names if known_names is not None else _known_ingredient_names()
    lowered = text.lower()

    found = []
    seen = set()
    for name in known_names:
        needle = name.lower()
        if needle in lowered and needle not in seen:
            seen.add(needle)
            found.append(name)
            if limit and len(found) >= limit:
                break

    return found


def _relevant_ingredients_for_question(question, recipe_text, known_names=None, limit=3):
    """Ingredients named in the QUESTION itself take priority (e.g. T13:
    "substitute for the fish sauce" -> ["Anchovy fish sauce"]); only if the
    question names none does this fall back to the recipe's own text - the
    substitution specialist's job is question-driven, not "check
    everything" (that is the allergen specialist's job, deliberately
    different - see run_allergen_specialist)."""

    known_names = known_names if known_names is not None else _known_ingredient_names()

    in_question = _extract_known_ingredients(question, known_names, limit=limit)
    if in_question:
        return in_question

    return _extract_known_ingredients(recipe_text, known_names, limit=limit)


SUBSTITUTION_TOOLS = {"get_substitutes"}
ALLERGEN_TOOLS = {"check_allergens"}


def _restricted_call(registry, allowed_tools, tool, args):
    """Refuse to call anything outside a specialist's own declared tool set
    - the enforcement point for "narrow specialists", not just a naming
    convention. See the module docstring."""

    if tool not in allowed_tools:
        return {"error": f"specialist is not permitted to call '{tool}' - only {sorted(allowed_tools)}"}

    return registry.call_tool(tool, args)


def _timed_generate(prompt, model):
    """generator._generate(), returning (text, tokens_used_for_this_one_call)
    rather than requiring every caller to slice generator.call_log itself."""

    before = len(generator.call_log)
    text = generator._generate(prompt, model)
    tokens = sum((c.get("total_tokens") or 0) for c in generator.call_log[before:])
    return text, tokens


# --------------------------------------------------------- substitution worker

SUBSTITUTION_PROMPT = """You are a narrow kitchen specialist with exactly one \
job: report on ingredient substitutes, using ONLY the tool results below. You \
do not know anything about this recipe beyond the text given to you here.

HARD RULES:
1. Only state a substitute if it appears in the TOOL RESULTS below - never \
invent a swap ratio or a substitute from general cooking knowledge.
2. If the tool results contain no substitute relevant to this question, say \
plainly that no substitute data was found - do not guess.
3. Keep your note to 2-3 sentences.

RECIPE TEXT (for context only - do not answer from this directly, only the \
tool results below are authoritative for substitutes):
{recipe_text}

TOOL RESULTS (get_substitutes, real output, one call per ingredient):
{tool_results}

USER QUESTION:
{question}

Write the substitution note now.
"""


def run_substitution_specialist(question, recipe_text, model=DEFAULT_MODEL):
    """One hand-off: manager -> substitution worker. Deterministically picks
    which ingredient(s) to look up (question-driven - see
    _relevant_ingredients_for_question), calls get_substitutes on each
    (its only permitted tool), then ONE LLM call phrases a grounded note
    from those real results. Returns {"note", "tool_calls", "tokens",
    "handoff"}."""

    registry = _get_registry()
    ingredients = _relevant_ingredients_for_question(question, recipe_text)

    tool_calls = []
    for name in ingredients:
        result = _restricted_call(registry, SUBSTITUTION_TOOLS, "get_substitutes", {"name": name})
        tool_calls.append({"tool": "get_substitutes", "args": {"name": name}, "result": result})

    if not tool_calls:
        tool_calls.append({
            "tool": "get_substitutes", "args": {},
            "result": {"note": "no ingredient in this recipe or question matched the ingredient database"},
        })

    prompt = SUBSTITUTION_PROMPT.format(
        recipe_text=recipe_text[:2000],
        tool_results=json.dumps(tool_calls, indent=2),
        question=question,
    )

    note, tokens = _timed_generate(prompt, model)

    return {
        "handoff": "manager->substitution_worker",
        "note": note,
        "tool_calls": tool_calls,
        "tokens": tokens,
    }


# -------------------------------------------------------------- allergen worker

ALLERGEN_PROMPT = """You are a narrow kitchen specialist with exactly one job: \
report on allergens and nutrition, using ONLY the tool result below. You do \
not know anything about this recipe beyond the text given to you here.

HARD RULES:
1. Only state an allergen or a nutrition figure if it appears in the TOOL \
RESULT below - never invent one from general food knowledge.
2. If the tool result is an error (the allergen worker was unavailable), say \
so plainly and explicitly refuse to state or imply ANY allergen or nutrition \
fact for this recipe - do not guess, do not fall back on general knowledge \
about what a dish like this "usually" contains.
3. Keep your note to 2-3 sentences.

RECIPE TEXT (for context only - do not answer from this directly, only the \
tool result below is authoritative for allergens/nutrition):
{recipe_text}

TOOL RESULT (check_allergens, real output for every ingredient this recipe's \
text was recognised to contain):
{tool_result}

USER QUESTION:
{question}

Write the allergen/nutrition note now.
"""


def run_allergen_specialist(question, recipe_text, model=DEFAULT_MODEL, force_failure=False, max_retries=1):
    """One hand-off: manager -> allergen/nutrition worker. Deterministically
    recognises every catalog ingredient anywhere in the recipe text (broad,
    NOT question-driven - allergen-checking has to look at everything, not
    just what the question happens to name), calls check_allergens ONCE
    with all of them (its only permitted tool), then ONE LLM call phrases a
    grounded note.

    `force_failure`: Week 10's required failure injection. When True, the
    tool call is replaced with a simulated HTTP 500 (not a real network
    fault - a deterministic, reproducible stand-in for "the allergen
    worker's service is down"), retried up to `max_retries` times (a
    realistic, not artificially broken, retry policy), and if still failing
    the LLM call still happens - but is handed the FAILURE as its only
    "tool result", so what it does next (degrade honestly / lie anyway) is
    a genuine, unscripted test of this project's existing grounded-
    generation discipline under a missing-data condition, not a canned
    demo. See week10_race.py's failure_case.md for the actual recorded
    outcome."""

    registry = _get_registry()
    ingredients = _extract_known_ingredients(recipe_text, limit=10)

    if force_failure:
        # A forced failure is deterministic - a real retry against a
        # genuinely down dependency would behave the same way. This is the
        # honest naive policy: retry a fixed number of times, then give up
        # and degrade, not an infinite or backing-off retry.
        attempts = 1 + max_retries
        result = {"error": "HTTP 500: allergen-worker service unavailable"}
    elif ingredients:
        attempts = 1
        result = _restricted_call(registry, ALLERGEN_TOOLS, "check_allergens", {"ingredients": ingredients})
    else:
        attempts = 1
        result = {"note": "no ingredient in this recipe matched the ingredient database"}

    tool_call = {"tool": "check_allergens", "args": {"ingredients": ingredients}, "result": result, "attempts": attempts}

    prompt = ALLERGEN_PROMPT.format(
        recipe_text=recipe_text[:2000],
        tool_result=json.dumps(result, indent=2),
        question=question,
    )

    note, tokens = _timed_generate(prompt, model)

    return {
        "handoff": "manager->allergen_worker",
        "note": note,
        "tool_calls": [tool_call],
        "tokens": tokens,
        "failed": bool(force_failure),
    }


# --------------------------------------------------------------------- manager

SYNTHESIS_PROMPT = """You are a recipe assistant synthesising a final answer \
from a recipe and two specialist reports. Answer ONLY using the material \
below - the same grounding discipline as this project's single-agent path.

HARD RULES:
1. Every factual claim must be supported by the RECIPE TEXT or a specialist \
NOTE below - never use outside knowledge.
2. Do not invent ingredient quantities, percentages, temperatures, timings, \
allergens or nutrition values beyond what is stated below.
3. If a specialist note says it could not confirm something (including an \
explicit worker failure), your answer must also not confirm that thing - \
carry the uncertainty through, do not paper over it for a tidier answer.
4. You MAY, however, recognise when a lay description in the question (a \
smell, a taste, a texture, a common name) refers to the same thing as a \
technical term that IS stated in the recipe text or a specialist note, and \
answer using that stated fact - this is using the given material, not \
outside knowledge (e.g. "smells like nail polish" connecting to a stated \
"acetone note" is a permitted connection, not an invented one).
5. If nothing below answers the question, reply with EXACTLY:
{refusal}
6. Cite claims from the recipe as [recipe_id=<id> | chunk_id=<id> | \
source_file=<file>] using only real values from the RECIPE TEXT block. Cite \
claims from a specialist as [specialist=substitution_worker] or \
[specialist=allergen_worker].

RECIPE TEXT:
{recipe_text}

SUBSTITUTION WORKER NOTE:
{substitution_note}

ALLERGEN/NUTRITION WORKER NOTE:
{allergen_note}

USER QUESTION:
{question}
"""


def run_orchestrator(collection, query, restriction=None, model=DEFAULT_MODEL, force_allergen_failure=False):
    """The manager: deterministic retrieval (same tools rag/fixed_workflow.py
    uses - not a hand-off, this is the manager doing its own job, not
    delegating) -> ALWAYS hand off to both specialists -> synthesise.

    Returns {"answer", "handoffs" (list of per-handoff dicts with real
    token counts), "llm_calls", "elapsed_seconds"} - the same shape
    race_agent_vs_workflow.py already expects from run_agent(), so
    week10_race.py can treat both arms uniformly."""

    start = time.monotonic()
    calls_before = len(generator.call_log)

    # Same restriction-as-upfront-filter fairness as rag/fixed_workflow.py -
    # otherwise a restriction-bearing question could retrieve a DIFFERENT
    # (wrong) recipe for this arm than for the single agent, biasing the
    # race for a reason that has nothing to do with the multi-agent
    # pattern being tested (exactly the brief's "you tested your context
    # strategy, not the pattern" trap, one level up).
    where = restriction_where(restriction)
    candidates = search_recipes(collection, query, where=where, top_k=5)

    if not candidates:
        return {
            "answer": f"I could not find a recipe for '{query}'.",
            "handoffs": [],
            "llm_calls": len(generator.call_log) - calls_before,
            "elapsed_seconds": time.monotonic() - start,
        }

    top = candidates[0]
    recipe_id = top["recipe_id"]
    full = get_recipe(collection, recipe_id)
    recipe_text = "\n\n".join(full["documents"][0])

    substitution = run_substitution_specialist(query, recipe_text, model=model)
    allergen = run_allergen_specialist(query, recipe_text, model=model, force_failure=force_allergen_failure)

    synthesis_prompt = SYNTHESIS_PROMPT.format(
        refusal=REFUSAL_TEXT,
        recipe_text=recipe_text,
        substitution_note=substitution["note"],
        allergen_note=allergen["note"],
        question=query,
    )
    answer, synthesis_tokens = _timed_generate(synthesis_prompt, model)

    handoffs = [
        substitution,
        allergen,
        {"handoff": "manager->synthesis", "note": None, "tool_calls": [], "tokens": synthesis_tokens},
    ]

    return {
        "answer": answer,
        "recipe_id": recipe_id,
        "recipe_text": recipe_text,
        "handoffs": handoffs,
        "llm_calls": len(generator.call_log) - calls_before,
        "elapsed_seconds": time.monotonic() - start,
    }
