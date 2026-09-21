"""Week 9 Module 5: tests for rag/mcp_client.py's result-unwrapping logic -
the one place a subtle MCP SDK behavior actually caused a bug during
development (FastMCP serializes a Python list return as one TextContent
block PER ITEM, not a single JSON array in one block; reading only the first
block silently dropped every candidate past the first). These tests build
minimal stand-ins for CallToolResult rather than spinning up a real server,
so they stay fast and offline like the rest of tests/.
"""

from types import SimpleNamespace

from rag.mcp_client import _unwrap


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
