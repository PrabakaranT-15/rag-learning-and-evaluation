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
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "ingredients.json"

mcp = FastMCP("ingredient-db", host="127.0.0.1", port=8931)

# Week 9 "Remote MCP & auth" / access-control: the shared secret an HTTP
# caller must present. Only enforced in --http mode (see __main__) - stdio
# is this repo's own agent launching a local subprocess, never exposed to a
# network, so there is nothing for a shared secret to protect there.
API_KEY_ENV = "INGREDIENT_SERVER_API_KEY"


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


# --------------------------------------------------------- resources & prompts
# MCP has three primitives - tools, resources, prompts. Everything above is a
# tool (an action: look something up, filter, aggregate). These two add the
# other two primitives, on the same server, so this deliverable demonstrates
# the full protocol surface rather than only the one primitive used so far.

@mcp.resource("ingredients://catalog")
def ingredient_catalog() -> str:
    """The full ingredient database, as a READABLE resource rather than a
    callable action - a host can list_resources()/read_resource() this to
    browse what's available before deciding which tool to call, the same way
    it might read a file or a DB row. Returns the raw JSON verbatim; no
    lookup, filtering or aggregation - that's what the tools above are for."""

    return DATA_PATH.read_text(encoding="utf-8")


@mcp.prompt()
def allergen_check_prompt(ingredients: list[str]) -> str:
    """A reusable prompt template for allergen-checking a recipe's ingredient
    list - MCP's third primitive. A host asks for this prompt BY NAME with a
    list of ingredients and gets back ready-to-send instruction text that
    points the caller at check_allergens rather than guessing from general
    food knowledge - the same grounding discipline this project's recipe
    generator enforces (rag/generator.py), applied to prompt templates too."""

    listed = ", ".join(ingredients)
    return (
        f"Using the check_allergens tool, verify whether any of these "
        f"ingredients carry a known allergen: {listed}. Report only the "
        f"allergens the tool actually returns for them - never guess an "
        f"allergen from general food knowledge, and say so explicitly for "
        f"any ingredient the tool reports as unknown."
    )


# ---------------------------------------------------------------- access control
# Minimal shared-secret bearer-token check for the HTTP transport - the
# concrete answer to "checking a tool before you trust someone else's" and
# "Remote MCP & auth" for THIS server's own exposed surface. A full OAuth
# authorization-server flow (FastMCP's built-in AuthSettings) is real but
# disproportionate machinery for a single static-secret course deliverable;
# a bearer check in front of the same streamable-HTTP app it would otherwise
# guard is the honest minimum version of the same control, and it is
# genuinely enforced (see tests/test_ingredient_server.py's HTTP auth test),
# not just described.

class _BearerTokenMiddleware:
    """Rejects any HTTP request that doesn't present
    `Authorization: Bearer <API_KEY_ENV>` - before it ever reaches the MCP
    session manager. Only installed when API_KEY_ENV is actually set (see
    __main__), so local dev/testing without a key still works exactly as
    before - the same "safe by default once configured, not broken by
    default until configured" tradeoff diet_flags() and validate_chunk_metadata()
    already make elsewhere in this project."""

    def __init__(self, app, expected_token):
        self._app = app
        self._expected = f"Bearer {expected_token}"

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        provided = headers.get(b"authorization", b"").decode("latin-1")

        if provided != self._expected:
            from starlette.responses import JSONResponse
            response = JSONResponse(
                {"error": "unauthorized: missing or invalid bearer token"},
                status_code=401,
            )
            await response(scope, receive, send)
            return

        await self._app(scope, receive, send)


if __name__ == "__main__":
    import sys

    # Local/agent use: stdio (the agent launches this as a subprocess, never
    # over a network - see rag/mcp_client.py). Shareable use ("someone
    # else's agent calls it"): streamable-HTTP, with the bearer check above
    # in front of it, so it has a URL AND a credential a different host
    # process needs to connect.
    if "--http" in sys.argv:
        import uvicorn

        api_key = os.getenv(API_KEY_ENV)
        app = mcp.streamable_http_app()

        if api_key:
            app = _BearerTokenMiddleware(app, api_key)
            print(f"[ingredient-db] HTTP auth ENABLED - callers need "
                  f"'Authorization: Bearer <{API_KEY_ENV}>'", file=sys.stderr)
        else:
            print(f"[ingredient-db] WARNING: {API_KEY_ENV} not set - serving "
                  f"UNAUTHENTICATED. Set it before exposing this beyond "
                  f"localhost.", file=sys.stderr)

        uvicorn.run(app, host=mcp.settings.host, port=mcp.settings.port)
    else:
        mcp.run(transport="stdio")
