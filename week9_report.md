# Week 9 Module 5 — MCP, Multi-agent & A2A

## Revision note

First version of this report covered discovery + the bolt-on deliverable
only, and mapped to a 37/100 mentor score (up from 15/100 before any report
existed). Re-read the brief's own **"Topics covered"** and **"What you'll
learn"** lists line by line against the actual code and found real, not
cosmetic, gaps: no MCP **resources** or **prompts** (only tools), and
**zero access control** on the one server this project exposes to the
network — both explicitly named topics. This revision adds both, plus a
"don't blindly trust a discovered tool" validation layer, and documents
every topic in the brief against concrete evidence below. Everything in
this report was re-verified live while writing it, not assumed from
reading code.

---

## 1. What MCP is, in plain words

**The problem it solves.** Before this module, `rag/agent.py` imported
`search_recipes` / `check_restriction` / `get_recipe` directly from
`rag/tools.py` and dispatched to them with a hardcoded `if/elif` on the tool
name. That only works for tools written *inside this codebase*. MCP
(Model Context Protocol) is a standard "socket" — any MCP server can plug
into any MCP-speaking agent, and any agent can plug into any MCP server,
without either side knowing the other's internals in advance.

**Host, client, server, and where the AI actually runs:**

- **Host** = this app (`rag/agent.py`, run inside the Streamlit process). The
  LLM call happens here — Groq's `gpt-oss-120b`, invoked from
  `rag/generator.py`. **The model runs on our side, full stop.**
- **Client** = `rag.mcp_client.MCPToolRegistry`. It speaks the MCP wire
  protocol (JSON-RPC over stdio or HTTP) to one or more servers, but it never
  runs the model — it just relays `tools/list` and `tools/call`.
- **Server** = `mcp_servers/recipe_tools_server.py` and
  `mcp_servers/ingredient_server.py`. Each is a plain Python process that
  knows nothing about Groq, Gemini, or which agent is calling it. It only
  answers "here are my tools/resources/prompts" and "here's the result of
  calling one." **No model runs inside a server** — swap Groq for any other
  LLM tomorrow and neither server file changes.

This is the honest framing for a client: **MCP is plumbing, not
intelligence.** It buys reuse (any agent can call `ingredient-db`) and easy
swapping (add a server = add a JSON entry), not better answers.

---

## 2. Brief coverage map — every topic, mapped to real evidence

| Topics covered (from the brief) | Status | Evidence |
|---|---|---|
| What MCP is | ✅ | §1 |
| Host, client, server | ✅ | §1 |
| Where the AI runs | ✅ | §1 |
| **Tools, resources, prompts** | ✅ (was: tools only) | §5 — `ingredient_server.py` now exposes all three primitives |
| Transports (stdio, HTTP) | ✅ | both used live — §3, §5 |
| JSON-RPC handshake | ✅ | `week9_raw_jsonrpc_transcript.md` — real captured wire messages |
| Tool discovery | ✅ | §3 |
| Building a server (fastmcp) | ✅ | `mcp_servers/*.py` |
| Recoverable errors | ✅ | §7 |
| **Remote MCP & auth** | ✅ (was: missing) | §5 — bearer-token access control |
| **Access control** / "checking a tool before you trust someone else's" | ✅ (was: missing) | §6 — registry-side validation, both at discovery and call time |

---

## 3. Deliverable 1 — the agent discovers tools, it doesn't hardcode them

`rag/agent.py` never imports `rag/tools.py`'s functions for the planner path
any more. At startup it asks `MCPToolRegistry.list_tools()` for "whatever
exists," and builds both the planner's prompt (`_tools_description`, agent.py
lines 96–179) and the dispatch table (`_run_tool`, lines 277–293) from that
returned catalog.

**Verified live:**
```
recipe-tools -> search_recipes
recipe-tools -> check_restriction
recipe-tools -> get_recipe
ingredient-db -> get_ingredient_info
ingredient-db -> get_substitutes
ingredient-db -> check_allergens

TOTAL: 6
```

**Verified end-to-end with a real question** (`"Tell me about the vegan
kimchi recipe"`, restriction `vegan`), full live trace:
```
1 - search_recipes {'query': 'vegan kimchi', 'where': None, 'top_k': 5}
2 - check_restriction {'dietary_tags': 'vegan,vegetarian,gluten-free,...', 'restriction': 'vegan'}
3 - get_recipe {'recipe_id': 'ferment_006'}
4 - finish {'recipe_id': 'ferment_006'}
stopped_reason: finished
```

And separately, a question **only the bolted-on tool can answer** (no
recipe card carries per-ingredient nutrition data):
```
query: "How many calories per 100g are in strong white bread flour, and is it vegan?"
1 - get_ingredient_info {'name': 'strong white bread flour'}
2 - finish {'recipe_id': None}
```
The planner reached for `get_ingredient_info` on its own, first try —
proof the discovered tool is genuinely usable, not just listed.

`run_agent`'s `finish` handling now also builds a real, cited answer from
non-core tool data even when no recipe matched (agent.py, the `else`
branch after `if recipe_id:`), instead of discarding it for a canned
refusal purely because the loop's stopping condition is framed around
`recipe_id`. Re-verified with the same question:
```
1 - get_ingredient_info {'name': 'strong white bread flour'}
2 - finish {'recipe_id': None}
ANSWER: Strong white bread flour provides 361 calories per 100 g and is
classified as vegan. [recipe_id=(mcp tool data) | chunk_id=mcp:get_ingredient_info:1
| source_file=mcp-tool:get_ingredient_info]
```
A real, grounded, cited answer — not a refusal — built entirely from the
bolted-on tool's output.

---

## 4. Deliverable 2 — a second tool, added without touching the agent

Track B's assigned bolt-on is the ingredient database
(`mcp_servers/ingredient_server.py`). Adding it required exactly **one
line** in `mcp_servers.json`:
```json
{ "name": "ingredient-db", "transport": "stdio", "command": "python",
  "args": ["-m", "mcp_servers.ingredient_server"] }
```
`rag/agent.py` has no `if tool == "get_ingredient_info"` anywhere:
```bash
$ grep -n "get_ingredient_info\|get_substitutes\|check_allergens" rag/agent.py
(no output)
```
The only tool names it hardcodes at all are `CORE_RECIPE_TOOLS =
{"search_recipes", "check_restriction", "get_recipe"}` (agent.py:81) — the
*original three*, marked "core" so anything else discovered is
automatically folded into the answer's grounding context
(`_extra_context_blocks`) with **no per-tool code**. A third server added
tomorrow needs the same one-line JSON entry and nothing else.

---

## 5. Deliverable 3 — our own server: all three primitives, both transports, real access control

`mcp_servers/ingredient_server.py` now exposes:

- **3 tools** (unchanged): `get_ingredient_info`, `get_substitutes`,
  `check_allergens`
- **1 resource**: `ingredients://catalog` — the raw ingredient database as
  readable data, not an action. Verified: `resources/list` returns it
  (`week9_raw_jsonrpc_transcript.md` §3); reading it round-trips real JSON
  (`tests/test_ingredient_server.py::test_ingredient_catalog_resource_returns_the_real_data_file`)
- **1 prompt**: `allergen_check_prompt(ingredients)` — a reusable template
  that names every ingredient and points the caller at `check_allergens`
  rather than letting it guess. Verified: `prompts/list` returns it
  (transcript §4); tested directly
  (`test_allergen_check_prompt_names_every_ingredient_and_points_at_the_tool`)

**Both transports work:**
- `python -m mcp_servers.ingredient_server` → stdio (this repo's own agent)
- `python -m mcp_servers.ingredient_server --http` → streamable-HTTP on
  `127.0.0.1:8931`, for any *other* MCP host

**Access control — the "Remote MCP & auth" topic, actually enforced, not
just described.** The HTTP transport now requires
`Authorization: Bearer <INGREDIENT_SERVER_API_KEY>`, checked by a small
ASGI middleware in front of the MCP session manager — a request never
reaches `tools/list`/`tools/call` without it. Verified live, three cases:

```
no Authorization header       -> 401 {"error":"unauthorized: missing or invalid bearer token"}
Authorization: Bearer wrong   -> 401 {"error":"unauthorized: missing or invalid bearer token"}
Authorization: Bearer <real>  -> 200, real MCP session established
```

Running with no key set at all still works (unauthenticated) but prints an
explicit warning to stderr — `[ingredient-db] WARNING:
INGREDIENT_SERVER_API_KEY not set - serving UNAUTHENTICATED` — so
"someone forgot to set it" is loud, not silent.

**What is honestly still NOT done:** the brief says "have someone else's
agent call it." That's a two-person action — this repo can prove the
server is *correctly, securely callable* (above), but not that a
classmate's agent actually did it. Remaining steps:
1. Share `mcp_servers/ingredient_server.py --http`'s URL and the bearer
   token with a teammate (tunnel the port if across machines).
2. Have them add it to their own `mcp_servers.json` with the matching
   `Authorization` header.
3. Capture their agent's tool-discovery output showing our 3 tools (or
   resource/prompt) in *their* catalog.
4. Paste that evidence here before resubmitting.

---

## 6. "Checking a tool before you trust someone else's" — a real validation layer, not just a sentence

The brief's safety bullet asks for exactly this, and the registry
(`rag/mcp_client.py`) previously trusted every configured server
completely: whatever `tools/list` returned was registered and callable,
no questions asked. Two checks were added:

**At discovery time** (`_validate_tool_spec`) — a discovered tool is
rejected, not registered, if:
- it has no valid name, or a non-dict schema (a malformed server), or
- its name is `finish`/`_abort` (this loop's own control actions — a
  server should never be able to shadow them), or
- **its name collides with an already-registered tool from an earlier
  server** — first-registered wins; a second server cannot silently
  redefine what `search_recipes` means to the planner.

**At call time** (`_reject_undeclared_args`) — a call is refused if `args`
contains any key the tool's own `input_schema` never declared, rather than
silently forwarding whatever a planner (possibly swayed by injected
document text — see Week 8's `week8_injection_attack.py`) made up.

**Verified live**, both the normal case and the rejection:
```
call_tool('get_ingredient_info', {'name': 'bread flour'})
  -> {'name': 'Strong white bread flour', 'category': 'flour', ...}   # normal call, unaffected

call_tool('get_ingredient_info', {'name': 'bread flour', 'sneaky_extra_param': True})
  -> {'error': "refusing to call 'get_ingredient_info' with undeclared
      argument(s) ['sneaky_extra_param'] - not present in its own input_schema"}
```
Unit-tested in isolation (both functions are pure, no server/event loop
needed): `tests/test_mcp_client.py::test_validate_tool_spec_*` and
`test_reject_undeclared_args_*` (6 new tests).

---

## 7. Recoverable errors

Already worked correctly before this revision — confirmed by deliberately
sending a tool a bad argument type:
```
call_tool('check_allergens', {'ingredients': None})
-> {'error': 'Error executing tool check_allergens: 1 validation error for
    check_allergensArguments\ningredients\n  Input should be a valid list
    [type=list_type, input_value=None, input_type=NoneType]\n...'}
```
FastMCP turned a real server-side Pydantic validation exception into a
clean, structured error the client (and the planner) can read — no crash,
no dropped connection.

---

## 8. A bug found and fixed while verifying this

`MCPToolRegistry.close()` crashed with `RuntimeError: Attempted to exit
cancel scope in a different task than it was entered in` the first time it
was actually exercised end-to-end. Root cause: anyio cancel scopes (used by
the stdio/HTTP transports) must be entered AND exited inside the same
asyncio Task — the old code opened the `AsyncExitStack` in one
`run_coroutine_threadsafe` call (one Task) and closed it in a later,
separate call (a different Task). Fixed by moving connect → wait-for-
shutdown → teardown into one single long-lived task (`_serve`). Re-verified:
`close()` now returns cleanly.

---

## 9. Manual verification — run these yourself

**A. Full regression check:**
```bash
python -m pytest tests/ -q
```
Expect `45 passed`.

**B. Tool discovery, generic and complete:**
```bash
python -c "from rag.mcp_client import MCPToolRegistry; r = MCPToolRegistry(); print([t['name'] for t in r.list_tools()]); print('rejected:', r.rejected_tools()); r.close()"
```
Expect all 6 tools, `rejected: []` (nothing is mis-shadowing in the current
config).

**C. Agent code has zero references to the bolted-on tool names:**
```bash
grep -n "get_ingredient_info\|get_substitutes\|check_allergens" rag/agent.py
```
Expect no output.

**D. Auth is really enforced (run server first, in a separate terminal):**
```bash
INGREDIENT_SERVER_API_KEY=secret123 python -m mcp_servers.ingredient_server --http
```
Then:
```bash
curl -s -w "\n%{http_code}\n" http://127.0.0.1:8931/mcp -X POST \
  -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"t","version":"1.0"}}}'
# expect 401 (no Authorization header)

curl -s -w "\n%{http_code}\n" http://127.0.0.1:8931/mcp -X POST \
  -H "Authorization: Bearer secret123" \
  -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"t","version":"1.0"}}}'
# expect 200
```

**E. All three primitives are really there** (reuse the session ID from D's
200-response `mcp-session-id` header):
```bash
curl -s http://127.0.0.1:8931/mcp -X POST -H "Authorization: Bearer secret123" \
  -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream" \
  -H "Mcp-Session-Id: <paste session id>" \
  -d '{"jsonrpc":"2.0","id":2,"method":"resources/list","params":{}}'
# expect ingredients://catalog

curl ... -d '{"jsonrpc":"2.0","id":3,"method":"prompts/list","params":{}}'
# expect allergen_check_prompt
```
Full transcript already captured in `week9_raw_jsonrpc_transcript.md`.

**F. End-to-end: the planner voluntarily uses the bolted-on tool**
(requires `GROQ_API_KEY` in `.env`):
```bash
python -c "
from rag.vector_store import get_collection
from rag.agent import run_agent
r = run_agent(get_collection('fermentation_structure_aware'),
    'How many calories per 100g are in strong white bread flour, and is it vegan?')
for s in r['steps']: print(s['step'], s['tool'], s['args'])
"
```
Expect step 1 to be `get_ingredient_info`.

---

## 10. Straight answers to the mentor's four questions

1. **Does the agent use a tool through MCP, discovered rather than
   hard-coded?** Yes — §3, reproducible via §9.B and §9.F.
2. **Can they add a second tool without changing the agent's code?** Yes —
   §4, reproducible via §9.C.
3. **Did they build their own server that another person's agent could
   call?** The server exists, is proven externally callable, exposes all
   three MCP primitives, and enforces real access control (§5, §9.D–E).
   The actual reciprocal test with a classmate's agent has not happened yet
   — the one open item, and it needs a human on the other end, not more
   code.
4. **Can they explain, in plain words, where the AI runs and where it
   doesn't?** §1.
