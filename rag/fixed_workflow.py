"""Week 7 Module 4: the plain fixed-sequence baseline for the SAME task
rag/agent.py's run_agent() solves.

This is deliberately not "worse on purpose" - it is what you would write if
you already knew, up front, that the request needs a dietary filter: apply
the filter in the one search call you make, then generate. No branching, no
retries, no model call deciding what to do next. That is exactly the
brief's point about when NOT to use an agent: if you already know the
steps, a fixed sequence is faster, cheaper, and just as reliable for
anything the filter alone can settle.

It shares rag/tools.py with the agent so retrieval/verification code is not
duplicated between the two paths - only the CONTROL FLOW differs, which is
the thing race_agent_vs_workflow.py is actually trying to measure.
"""

from rag.generator import generate_recipe_answer, DEFAULT_MODEL
from rag.tools import search_recipes, restriction_where, get_recipe


def run_fixed_workflow(collection, query, restriction, model=DEFAULT_MODEL, top_k=5):
    """One search (filtered up front, by a human who already knows the
    schema) -> one generation call. No loop, no retry."""

    where = restriction_where(restriction)
    candidates = search_recipes(collection, query, where=where, top_k=top_k)

    if not candidates:
        return {
            "answer": f"I could not find a recipe for '{query}' that satisfies '{restriction}'.",
            "candidates": [],
        }

    top = candidates[0]
    full = get_recipe(collection, top["recipe_id"])

    question = f'Find a recipe for "{query}" that satisfies the dietary restriction "{restriction}".'
    answer = generate_recipe_answer(question, full, model=model)

    return {"answer": answer, "candidates": candidates}
