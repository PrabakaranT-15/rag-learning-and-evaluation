"""Week 9 Module 5: a generic MCP tool registry.

rag/agent.py used to import search_recipes/check_restriction/get_recipe
directly from rag/tools.py and dispatch to them by a hardcoded tool-name
string. Now it asks THIS module for "whatever tools are available", and this
module answers by connecting to every server listed in mcp_servers.json and
calling tools/list on each - so bolting on a new server (one more entry in
that file, e.g. mcp_servers/ingredient_server.py) never requires touching
rag/agent.py. That is the actual thing Week 9's mentor review checks for.

rag/agent.py and app.py are synchronous throughout (Streamlit's execution
model is synchronous), but the MCP SDK's ClientSession and transports are
asyncio-only. Rather than infect every caller with async/await, this module
owns one background event loop thread and exposes a plain synchronous facade
(list_tools/call_tool) that blocks on that loop via
asyncio.run_coroutine_threadsafe - the standard way to bridge a sync app into
one long-lived asyncio resource.
"""

import asyncio
import json
import threading
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "mcp_servers.json"

# rag/agent.py's own loop-control actions - never something a discovered
# server should be able to supply. See _validate_tool_spec below.
_RESERVED_TOOL_NAMES = {"finish", "_abort"}


def _validate_tool_spec(tool_name, input_schema, server_name, server_by_name):
    """"Checking a tool before you trust someone else's" (Week 9 safety
    topic) - sanity-check ONE discovered tool before it is ever registered,
    rather than assuming every tools/list entry from every configured
    server is automatically safe to hand to the planner.

    Returns None if the tool is fine to register, or a human-readable
    rejection reason otherwise. Two concrete things this catches that
    "just trust tools/list" would not:

    - a malformed entry (no name, non-dict schema) from a buggy server;
    - NAME SHADOWING: a later-connected server declaring a tool name that
      collides with an earlier, already-trusted one (or with `finish`/
      `_abort`, this loop's own control actions) - first-registered wins,
      the later one is rejected rather than silently overwriting it. A
      compromised or misconfigured second server could otherwise redefine
      what "search_recipes" means to the planner without anyone noticing.

    Never raises - a hostile or buggy server is expected input for a
    registry that connects to whatever mcp_servers.json lists, not a bug
    in this code.
    """

    if not tool_name or not isinstance(tool_name, str):
        return "tool has no valid name"

    if tool_name in _RESERVED_TOOL_NAMES:
        return (
            f"'{tool_name}' collides with a reserved control action "
            f"(finish/_abort) - refusing to let a discovered server shadow "
            f"the agent's own loop control"
        )

    if tool_name in server_by_name:
        return (
            f"'{tool_name}' was already registered by server "
            f"'{server_by_name[tool_name]}' - refusing to let server "
            f"'{server_name}' silently shadow it"
        )

    if not isinstance(input_schema, dict):
        return f"'{tool_name}' has a non-dict input_schema"

    return None


def _reject_undeclared_args(tool_name, args, declared_params):
    """The call-time half of "checking a tool before you trust someone
    else's": refuse to forward any argument the tool's OWN input_schema
    never declared, rather than silently passing through whatever a
    planner (possibly swayed by injected document text - see Week 8) made
    up. Returns an {"error": ...} dict to hand back to the planner as an
    actionable observation, or None if every arg is declared and the call
    should proceed."""

    unexpected = sorted(set(args) - declared_params)
    if not unexpected:
        return None

    return {
        "error": (
            f"refusing to call '{tool_name}' with undeclared argument(s) "
            f"{unexpected} - not present in its own input_schema"
        )
    }


class MCPToolRegistry:
    """Connects to every MCP server listed in `config_path` and aggregates
    their tools into one flat catalog. `call_tool` dispatches purely by tool
    name - it does not know or care which server, or how many servers,
    actually implement it."""

    def __init__(self, config_path=DEFAULT_CONFIG_PATH):
        self._config_path = Path(config_path)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._session_by_tool = {}
        self._server_by_tool = {}
        self._tools = []
        self._rejected_tools = []
        self._shutdown_event = None
        self._serve_task = None
        self._run(self._start_serving())

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    async def _start_serving(self):
        """Connect and then hand control to a single long-lived task
        (_serve) that owns the AsyncExitStack for its entire life.

        anyio's cancel scopes (used by stdio_client/streamablehttp_client)
        must be entered AND exited from the same asyncio Task - not just the
        same event loop/thread. Opening the stack in one
        run_coroutine_threadsafe call (one Task) and closing it in a later,
        separate run_coroutine_threadsafe call (a different Task) is exactly
        what used to raise "Attempted to exit cancel scope in a different
        task than it was entered in" from close(). Connecting, waiting for
        a shutdown signal, and tearing down all happen inside _serve's one
        task now, so enter and exit always share a Task.
        """

        self._shutdown_event = asyncio.Event()
        ready = self._loop.create_future()
        self._serve_task = asyncio.ensure_future(self._serve(ready))
        await ready

    async def _serve(self, ready):
        async with AsyncExitStack() as stack:
            config = json.loads(self._config_path.read_text(encoding="utf-8"))

            for server in config.get("servers", []):
                session = await self._open_session(server, stack)
                listed = (await session.list_tools()).tools

                for tool in listed:
                    input_schema = tool.inputSchema or {}
                    reason = _validate_tool_spec(
                        tool.name, input_schema, server["name"], self._server_by_tool
                    )
                    if reason:
                        self._rejected_tools.append({
                            "tool": tool.name, "server": server["name"], "reason": reason,
                        })
                        print(f"[mcp_client] REJECTED tool {tool.name!r} "
                              f"from {server['name']!r}: {reason}")
                        continue

                    self._session_by_tool[tool.name] = session
                    self._server_by_tool[tool.name] = server["name"]
                    self._tools.append({
                        "name": tool.name,
                        "description": tool.description or "",
                        "input_schema": input_schema,
                        "server": server["name"],
                    })

            ready.set_result(None)
            await self._shutdown_event.wait()

    async def _open_session(self, server, stack):
        transport = server["transport"]

        if transport == "stdio":
            command = server["command"]
            if command == "python":
                import sys
                command = sys.executable
            params = StdioServerParameters(
                command=command,
                args=server.get("args", []),
                cwd=str(Path(self._config_path).resolve().parent),
            )
            read, write = await stack.enter_async_context(stdio_client(params))

        elif transport == "http":
            read, write, _ = await stack.enter_async_context(
                streamablehttp_client(server["url"])
            )

        else:
            raise ValueError(f"server '{server['name']}': unknown transport '{transport}'")

        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session

    def list_tools(self):
        """[{name, description, input_schema, server}, ...] - everything
        discovered across every configured server, unfiltered."""

        return list(self._tools)

    def tool_param_names(self, tool_name):
        """The parameter names a discovered tool's schema declares, or an
        empty set for an unknown tool. Lets a caller (rag/agent.py) decide
        whether to inject something like `collection_name` generically,
        without hardcoding which tools need it."""

        for tool in self._tools:
            if tool["name"] == tool_name:
                return set(tool["input_schema"].get("properties", {}))
        return set()

    def call_tool(self, name, args):
        """Call a discovered tool by name. Returns its parsed JSON result, or
        {"error": ...} if the tool is unknown, the call itself failed, or
        the args include something the tool's own schema never declared
        (see _reject_undeclared_args) - never raises, since a bad
        planner-chosen tool name/args is expected input for rag/agent.py's
        loop, not a bug."""

        session = self._session_by_tool.get(name)
        if session is None:
            return {"error": f"unknown tool '{name}'"}

        args = args or {}
        rejection = _reject_undeclared_args(name, args, self.tool_param_names(name))
        if rejection:
            return rejection

        result = self._run(session.call_tool(name, args))
        return _unwrap(result)

    def rejected_tools(self):
        """Every discovered tool that _validate_tool_spec refused to
        register, and why - for a caller (or this project's own report) to
        show that "trust but verify" is a real, observable behavior and not
        just a docstring claim."""

        return list(self._rejected_tools)

    def close(self):
        if self._shutdown_event is not None:
            self._loop.call_soon_threadsafe(self._shutdown_event.set)

            async def _wait_for_teardown():
                await self._serve_task

            self._run(_wait_for_teardown())
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)


def _unwrap(result):
    """A CallToolResult's structuredContent is already the parsed return
    value when the server provides it; otherwise fall back to parsing the
    text content block(s), which is how FastMCP serializes a plain
    dict/list return value.

    A tool returning a Python list comes back as ONE TextContent block PER
    ITEM (not a single JSON array in one block) - FastMCP's content-block
    convention, not a JSON-encoding choice this client makes. Reading only
    content[0] would silently drop every candidate past the first, so every
    block is parsed and, when there's more than one, collected into a list."""

    if getattr(result, "isError", False):
        text = result.content[0].text if result.content else "tool call failed"
        return {"error": text}

    if getattr(result, "structuredContent", None) is not None:
        return result.structuredContent

    if not result.content:
        return None

    def _parse(block):
        if not hasattr(block, "text"):
            return block
        try:
            return json.loads(block.text)
        except json.JSONDecodeError:
            return block.text

    if len(result.content) == 1:
        return _parse(result.content[0])

    return [_parse(block) for block in result.content]
