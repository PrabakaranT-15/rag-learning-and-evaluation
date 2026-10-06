"""Week 11 failure -> test loop: replay every recorded production failure in
tests/failures/cases.json. No network, no API keys."""

import json
from pathlib import Path

import pytest

from rag.guards import dairy_free_swap_violations
from rag.restrictions import detect_restriction
from mcp_servers.ingredient_server import get_substitutes

CASES = json.loads((Path(__file__).parent / "failures" / "cases.json").read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_recorded_failure_stays_fixed(case):
    check = case["check"]

    if check == "detect_restriction":
        assert detect_restriction(case["question"]) == case["expected"]

    elif check == "dairy_free_answer":
        assert dairy_free_swap_violations(case["answer"]) == case["expected_flags"]

    elif check == "tool_swaps":
        result = get_substitutes(case["ingredient"], case["restriction"])
        assert result["substitutes"], "no dairy-free swap offered at all"
        for swap in result["substitutes"]:
            assert case["restriction"] in swap["dietary_tags"]
            assert swap["name"].lower() not in case["forbidden_names"]


def test_cases_file_is_well_formed():
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids)), "duplicate failure-case ids"
    assert all(c.get("found_via") and c.get("complaint") for c in CASES)
