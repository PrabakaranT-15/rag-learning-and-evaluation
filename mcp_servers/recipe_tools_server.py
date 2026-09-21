"""Week 9 Module 5: rag/tools.py's search_recipes/check_restriction/get_recipe,
exposed as an MCP server instead of plain Python imports.

rag/agent.py used to import these three functions directly and dispatch to
them by a hardcoded tool-name string. Now it discovers them at runtime via
MCP's tools/list (see rag/mcp_client.py) and never imports this module - the
planner's available-tool list, and its dispatch, both come from whatever this
server (and any other configured server) advertises.

Tool functions take a `collection_name` string rather than a live Chroma
Collection object: a Python object can't cross the stdio/HTTP process
boundary MCP calls happen over, so each tool looks its collection up fresh
from the same on-disk ./data/chroma store rag/vector_store.py already points
at. rag/fixed_workflow.py is untouched and keeps calling rag/tools.py
in-process - only the agent's path to these functions changed.
"""

from mcp.server.fastmcp import FastMCP

from rag import tools as recipe_tools
from rag.vector_store import get_collection

mcp = FastMCP("recipe-tools")


@mcp.tool()
def search_recipes(collection_name: str, query: str, where: dict | None = None, top_k: int = 5) -> list:
    """Hybrid search over a recipe collection. Returns up to top_k candidates,
    ranked best-first, each with chunk_id, recipe_id, recipe_name,
    dietary_tags, section and a text snippet. `where` is an optional Chroma
    metadata filter, e.g. {"diet_vegan": true}, for narrowing to recipes that
    already satisfy a dietary restriction."""

    collection = get_collection(collection_name)
    return recipe_tools.search_recipes(collection, query, where=where, top_k=top_k)


@mcp.tool()
def check_restriction(dietary_tags: str, restriction: str) -> dict:
    """Deterministic check: is `restriction` (e.g. "vegan", "dairy-free") one
    of the comma-separated tags in `dietary_tags`? Copy dietary_tags straight
    from a prior search_recipes candidate - never invent it. Returns
    {restriction, dietary_tags, satisfied}."""

    return recipe_tools.check_restriction(dietary_tags, restriction)


@mcp.tool()
def get_recipe(collection_name: str, recipe_id: str) -> dict:
    """Fetch every chunk of one recipe, joined into a single text blob
    (truncated to 3000 characters for the planner's context window), plus a
    chunk count. Use only once a final candidate has been picked - the actual
    answer is generated from the recipe's full, untruncated text separately."""

    collection = get_collection(collection_name)
    result = recipe_tools.get_recipe(collection, recipe_id)
    text = "\n\n".join(result["documents"][0])
    return {"recipe_id": recipe_id, "chunks": len(result["ids"][0]), "text": text[:3000]}


if __name__ == "__main__":
    mcp.run(transport="stdio")
