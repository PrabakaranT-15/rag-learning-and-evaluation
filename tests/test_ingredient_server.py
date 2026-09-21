"""Week 9 Module 5: pure-function tests for mcp_servers/ingredient_server.py.

@mcp.tool() leaves the underlying function directly callable, so these call
it exactly as the module does internally - no subprocess, no MCP transport,
no network. That keeps this suite fast and offline, consistent with the
other tests/ files, which test the logic behind a tool rather than the
transport it's served over.
"""

from mcp_servers.ingredient_server import check_allergens, get_ingredient_info, get_substitutes


def test_get_ingredient_info_known_ingredient():
    result = get_ingredient_info("bread flour")

    assert result["name"] == "Strong white bread flour"
    assert "gluten" in result["allergens"]
    assert "vegan" in result["dietary_tags"]


def test_get_ingredient_info_matches_by_alias_case_insensitively():
    assert get_ingredient_info("BREAD FLOUR")["name"] == get_ingredient_info("Strong white bread flour")["name"]


def test_get_ingredient_info_unknown_ingredient_returns_error_not_exception():
    result = get_ingredient_info("unobtainium")

    assert "error" in result


def test_get_substitutes_filters_by_dietary_restriction():
    unfiltered = get_substitutes("Anchovy fish sauce")
    vegan_only = get_substitutes("Anchovy fish sauce", dietary_restriction="vegan")

    assert len(vegan_only["substitutes"]) == len(unfiltered["substitutes"])
    assert all("vegan" in s["dietary_tags"] for s in vegan_only["substitutes"])


def test_get_substitutes_restriction_with_no_match_returns_empty_list():
    result = get_substitutes("Water", dietary_restriction="vegan")

    assert result["substitutes"] == []


def test_check_allergens_aggregates_across_ingredients():
    result = check_allergens(["Eggs", "Whole milk", "Napa cabbage"])

    assert result["by_ingredient"]["Eggs"] == ["egg"]
    assert result["by_ingredient"]["Napa cabbage"] == []
    assert set(result["all_allergens"]) == {"egg", "dairy"}


def test_check_allergens_unknown_ingredient_is_null_not_empty():
    result = check_allergens(["not a real ingredient"])

    assert result["by_ingredient"]["not a real ingredient"] is None
