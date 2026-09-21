"""Week 9 Module 5 Track B: a standalone MCP server exposing an ingredient
database - nutrition, allergens, and dietary-aware substitutes.

This is the "bolt-on" deliverable. It knows nothing about rag/agent.py, the
recipe corpus, ChromaDB, or this project's tools - any MCP host, this
project's own agent or a classmate's, gets these three tools for free purely
from tools/list, by adding one entry to mcp_servers.json (or pointing at the
HTTP URL from run_http.py). rag/agent.py is never edited to pick this server
up.

Data lives in data/ingredients.json rather than a real database - the point
of this deliverable is the MCP surface (discoverable tools, dietary-aware
filtering, deterministic answers grounded in real records), not the storage
engine. Swapping the JSON file for a real DB later would not change a single
tool signature.
"""

import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "ingredients.json"

mcp = FastMCP("ingredient-db", host="127.0.0.1", port=8931)


def _load_catalog():
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def _find(name):
    """Case-insensitive lookup by name or alias. Returns None, not an
    exception, on a miss - a wrong or misspelled ingredient name from a
    caller is an expected input, not a bug."""

    needle = name.strip().lower()

    for entry in _load_catalog():
        if entry["name"].lower() == needle:
            return entry
        if needle in (alias.lower() for alias in entry.get("aliases", [])):
            return entry

    return None


@mcp.tool()
def get_ingredient_info(name: str) -> dict:
    """Look up one ingredient's category, dietary tags, allergens and
    nutrition (per 100g) by name - case-insensitive, common aliases included
    (e.g. "bread flour" for "Strong white bread flour"). Returns an 'error'
    key, not an exception, if the ingredient isn't in the database."""

    entry = _find(name)

    if entry is None:
        return {"error": f"'{name}' not found in ingredient database"}

    return {
        "name": entry["name"],
        "category": entry["category"],
        "dietary_tags": entry["dietary_tags"],
        "allergens": entry["allergens"],
        "nutrition_per_100g": entry["nutrition_per_100g"],
    }


@mcp.tool()
def get_substitutes(name: str, dietary_restriction: str | None = None) -> dict:
    """Common substitutes for one ingredient, each with a swap ratio and a
    note on what changes. Pass `dietary_restriction` (e.g. "vegan",
    "dairy-free") to keep only substitutes that themselves satisfy it - e.g.
    get_substitutes("Anchovy fish sauce", "vegan") returns only the
    fish-free swap, not every substitute on file."""

    entry = _find(name)

    if entry is None:
        return {"error": f"'{name}' not found in ingredient database"}

    substitutes = entry.get("substitutes", [])

    if dietary_restriction:
        needle = dietary_restriction.strip().lower()
        substitutes = [
            s for s in substitutes
            if needle in (t.lower() for t in s.get("dietary_tags", []))
        ]

    return {
        "name": entry["name"],
        "dietary_restriction": dietary_restriction,
        "substitutes": substitutes,
    }


@mcp.tool()
def check_allergens(ingredients: list) -> dict:
    """Given a list of ingredient names (e.g. a recipe's ingredient list),
    return each one's known allergens plus the deduplicated set across all of
    them - for scanning a recipe before serving it to someone with an
    allergy. An ingredient not found in the database maps to null, not an
    empty list, so a caller can tell "no known allergens" apart from
    "unknown ingredient, can't say"."""

    by_ingredient = {}
    all_allergens = set()

    for name in ingredients:
        entry = _find(name)
        if entry is None:
            by_ingredient[name] = None
            continue
        by_ingredient[name] = entry["allergens"]
        all_allergens.update(entry["allergens"])

    return {"by_ingredient": by_ingredient, "all_allergens": sorted(all_allergens)}


if __name__ == "__main__":
    import sys

    # Local/agent use: stdio (the agent launches this as a subprocess).
    # Shareable use ("someone else's agent calls it"): streamable-HTTP, so it
    # has a URL a different host process can connect to over the network.
    transport = "streamable-http" if "--http" in sys.argv else "stdio"
    mcp.run(transport=transport)
