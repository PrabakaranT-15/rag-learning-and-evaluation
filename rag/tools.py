"""Week 7 Module 4: the tools a recipe agent can call, shared with the fixed
workflow baseline so both methods run on IDENTICAL retrieval/verification
code (rag/agent.py, rag/fixed_workflow.py). The only thing that should ever
differ between "agent" and "fixed workflow" is the CONTROL FLOW around these
functions - an LLM deciding step-by-step vs. a hardcoded sequence - not the
underlying retrieval or checking logic. Otherwise a speed/cost/reliability
comparison between them wouldn't be measuring what it claims to measure.
"""

from rag.vector_store import diet_flags
from rag.hybrid_retriever import retrieve_hybrid


def search_recipes(collection, query, where=None, top_k=5):
    """Hybrid search, flattened into a list of small candidate dicts instead
    of Chroma's ids[0]/documents[0]/metadatas[0]/distances[0] shape - easier
    for the agent's JSON action loop (and its logged transcript) to carry
    around than the raw parallel-arrays result."""

    results = retrieve_hybrid(collection, query, top_k, where=where)

    candidates = []
    for chunk_id, document, metadata, distance in zip(
        results["ids"][0], results["documents"][0],
        results["metadatas"][0], results["distances"][0],
    ):
        candidates.append({
            "chunk_id": chunk_id,
            "recipe_id": metadata.get("recipe_id"),
            "recipe_name": metadata.get("recipe_name"),
            "dietary_tags": metadata.get("dietary_tags"),
            "section": metadata.get("section") or "(flat window)",
            "distance": round(float(distance), 4),
            "text": document,
        })

    return candidates


def check_restriction(dietary_tags, restriction):
    """Deterministic pass/fail: is `restriction` (e.g. "vegan") one of the
    comma-separated dietary_tags on a candidate? No model call - this is
    exactly the kind of step that does not need an agent to decide, which is
    why it is a plain function rather than something routed through the LLM
    planner."""

    tags = {t.strip().lower() for t in (dietary_tags or "").split(",") if t.strip()}
    satisfied = restriction.strip().lower() in tags

    return {
        "restriction": restriction,
        "dietary_tags": dietary_tags,
        "satisfied": satisfied,
    }


def get_recipe(collection, recipe_id):
    """Fetch every chunk belonging to one recipe_id, in a Chroma-query-shaped
    result (ids/documents/metadatas/distances, each in a one-element outer
    list) so it can be passed straight into
    rag.generator.generate_recipe_answer() exactly like retrieve()'s output."""

    rows = collection.get(
        where={"recipe_id": recipe_id},
        include=["documents", "metadatas"],
    )

    ordered = sorted(zip(rows["ids"], rows["documents"], rows["metadatas"]), key=lambda r: r[0])

    ids = [r[0] for r in ordered]
    documents = [r[1] for r in ordered]
    metadatas = [r[2] for r in ordered]

    return {
        "ids": [ids],
        "documents": [documents],
        "metadatas": [metadatas],
        "distances": [[0.0] * len(ids)],
    }


def restriction_where(restriction):
    """Translate a plain restriction word into the Chroma metadata filter the
    FIXED workflow applies up front (rag/fixed_workflow.py) - built from the
    same diet_x boolean convention vector_store.diet_flags() already uses at
    ingest time, so "vegan" -> {"diet_vegan": True}.

    restriction may be None (not every question names a dietary restriction) -
    in that case there is nothing to filter on, so this returns None rather
    than calling diet_flags() on a non-string."""

    if not restriction:
        return None

    return diet_flags(restriction) or None


# ------------------------------------------------------- Week 8 addition
# Shared between rag/agent.py's live finish-gate (enforcing "was this
# actually verified?" DURING the loop, before a finish is ever accepted)
# and rag/trajectory_eval.py's after-the-fact grading of a finished
# transcript - both need exactly the same definition of "verified", or a
# transcript could pass one check and fail the other for no real reason.


def known_dietary_tags_by_recipe(transcript):
    """recipe_id -> the dietary_tags string actually returned for it by a
    search_recipes step in this transcript - the real, retrieved data, as
    opposed to anything the model might claim or misremember."""

    known = {}
    for step in transcript:
        if step.get("tool") != "search_recipes":
            continue
        observation = step.get("observation")
        if not isinstance(observation, list):
            continue
        for candidate in observation:
            recipe_id = candidate.get("recipe_id") if isinstance(candidate, dict) else None
            if recipe_id and recipe_id not in known:
                known[recipe_id] = candidate.get("dietary_tags")
    return known


def _check_restriction_observations(transcript):
    """Every check_restriction step's (dietary_tags argument, satisfied
    result) pair, in order."""

    pairs = []
    for step in transcript:
        if step.get("tool") != "check_restriction":
            continue
        observation = step.get("observation")
        if isinstance(observation, dict):
            pairs.append(((step.get("args") or {}).get("dietary_tags"), observation.get("satisfied")))
    return pairs


def restriction_verified(transcript, recipe_id, restriction):
    """True only if this transcript already contains a check_restriction
    call that reported satisfied=True against recipe_id's REAL (retrieved,
    not claimed) dietary_tags for this restriction.

    Matches by the dietary_tags STRING, not by recipe_id, because
    check_restriction()'s own contract is purely a function of that string -
    it never sees a recipe_id. If two recipes in a corpus happen to share an
    identical dietary_tags string, a check_restriction call verified against
    that string legitimately covers either one; that's the same resolution
    check_restriction() itself has, not a gap introduced here."""

    if not restriction or not recipe_id:
        return False

    real_tags = known_dietary_tags_by_recipe(transcript).get(recipe_id)

    return any(
        used_tags == real_tags and satisfied
        for used_tags, satisfied in _check_restriction_observations(transcript)
    )
