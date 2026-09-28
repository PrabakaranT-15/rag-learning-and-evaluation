"""Week 9 Module 5: tests for rag/mcp_client.py's result-unwrapping logic -
the one place a subtle MCP SDK behavior actually caused a bug during
development (FastMCP serializes a Python list return as one TextContent
block PER ITEM, not a single JSON array in one block; reading only the first
block silently dropped every candidate past the first). These tests build
minimal stand-ins for CallToolResult rather than spinning up a real server,
so they stay fast and offline like the rest of tests/.
"""

from types import SimpleNamespace

from rag.mcp_client import _reject_undeclared_args, _unwrap, _validate_tool_spec


def _text_block(text):
    return SimpleNamespace(text=text)


def test_unwrap_single_block_dict():
    result = SimpleNamespace(isError=False, structuredContent=None, content=[_text_block('{"a": 1}')])

    assert _unwrap(result) == {"a": 1}


def test_unwrap_multiple_blocks_becomes_a_list():
    result = SimpleNamespace(
        isError=False,
        structuredContent=None,
        content=[_text_block('{"id": 1}'), _text_block('{"id": 2}'), _text_block('{"id": 3}')],
    )

    assert _unwrap(result) == [{"id": 1}, {"id": 2}, {"id": 3}]


def test_unwrap_prefers_structured_content_when_present():
    result = SimpleNamespace(isError=False, structuredContent={"already": "parsed"}, content=[_text_block("ignored")])

    assert _unwrap(result) == {"already": "parsed"}


def test_unwrap_error_result_returns_error_dict():
    result = SimpleNamespace(isError=True, structuredContent=None, content=[_text_block("boom")])

    assert _unwrap(result) == {"error": "boom"}


def test_unwrap_empty_content_returns_none():
    result = SimpleNamespace(isError=False, structuredContent=None, content=[])

    assert _unwrap(result) is None


# --------------------------------------------- "checking a tool before you
# trust someone else's" - Week 9 safety topic, tested in isolation from any
# real server/transport since both functions are pure.

def test_validate_tool_spec_accepts_a_normal_tool():
    assert _validate_tool_spec("get_ingredient_info", {"properties": {}}, "ingredient-db", {}) is None


def test_validate_tool_spec_rejects_missing_name():
    assert _validate_tool_spec("", {}, "some-server", {}) is not None
    assert _validate_tool_spec(None, {}, "some-server", {}) is not None


def test_validate_tool_spec_rejects_reserved_control_action_names():
    assert _validate_tool_spec("finish", {}, "some-server", {}) is not None
    assert _validate_tool_spec("_abort", {}, "some-server", {}) is not None


def test_validate_tool_spec_rejects_name_shadowing_an_earlier_server():
    already_registered = {"search_recipes": "recipe-tools"}

    reason = _validate_tool_spec("search_recipes", {}, "a-second-server", already_registered)

    assert reason is not None
    assert "recipe-tools" in reason
    assert "a-second-server" in reason


def test_validate_tool_spec_rejects_non_dict_schema():
    assert _validate_tool_spec("some_tool", "not a dict", "some-server", {}) is not None


def test_reject_undeclared_args_allows_only_declared_params():
    assert _reject_undeclared_args("search_recipes", {"query": "kimchi"}, {"query", "top_k"}) is None


def test_reject_undeclared_args_rejects_an_undeclared_param():
    result = _reject_undeclared_args("search_recipes", {"query": "kimchi", "admin_override": True}, {"query", "top_k"})

    assert result is not None
    assert "admin_override" in result["error"]
