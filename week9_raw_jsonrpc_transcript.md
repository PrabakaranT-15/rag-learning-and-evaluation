# Week 9 — raw MCP JSON-RPC transcript

Captured live against `mcp_servers/ingredient_server.py` running in
`--http` mode with `INGREDIENT_SERVER_API_KEY` set, using `curl` directly —
no MCP SDK, no Python, on the calling side. This is "looking at the raw
messages once, so MCP stops being a mystery" (Topics covered): the literal
bytes an MCP host sends and receives, before any client library hides them.

## 0. Auth is enforced before anything else

```
$ curl -s -w "\nstatus: %{http_code}\n" http://127.0.0.1:8931/mcp -X POST \
    -H "Content-Type: application/json" \
    -H "Accept: application/json, text/event-stream" \
    -d '{"jsonrpc":"2.0","id":1,"method":"initialize", ...}'
    # (no Authorization header)

{"error":"unauthorized: missing or invalid bearer token"}
status: 401
```

```
$ curl ... -H "Authorization: Bearer wrongtoken" ...
{"error":"unauthorized: missing or invalid bearer token"}
status: 401
```

## 1. `initialize` — the handshake, with the correct bearer token

Request:
```json
{"jsonrpc":"2.0","id":1,"method":"initialize",
 "params":{"protocolVersion":"2024-11-05","capabilities":{},
           "clientInfo":{"name":"manual-verify","version":"1.0"}}}
```

Response (`200 OK`, note the `mcp-session-id` response header — every
following call in this transcript replays it):
```json
{"jsonrpc":"2.0","id":1,"result":{
  "protocolVersion":"2024-11-05",
  "capabilities":{"experimental":{},
                   "prompts":{"listChanged":false},
                   "resources":{"subscribe":false,"listChanged":false},
                   "tools":{"listChanged":false}},
  "serverInfo":{"name":"ingredient-db","version":"1.30.0"}}}
```

Note the server already declares all three primitive capabilities —
`tools`, `resources`, `prompts` — in this one handshake message, before a
single `list` call is made.

## 2. `tools/list`

```json
{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}
```
→ 3 tools: `get_ingredient_info`, `get_substitutes`, `check_allergens`, each
with a real JSON-Schema `inputSchema` (visible in the full response in the
session log this was captured from).

## 3. `resources/list`

```json
{"jsonrpc":"2.0","id":3,"method":"resources/list","params":{}}
```
→
```json
{"jsonrpc":"2.0","id":3,"result":{"resources":[
  {"name":"ingredient_catalog","uri":"ingredients://catalog",
   "description":"The full ingredient database, as a READABLE resource...",
   "mimeType":"text/plain"}
]}}
```

## 4. `prompts/list`

```json
{"jsonrpc":"2.0","id":4,"method":"prompts/list","params":{}}
```
→
```json
{"jsonrpc":"2.0","id":4,"result":{"prompts":[
  {"name":"allergen_check_prompt",
   "description":"A reusable prompt template for allergen-checking...",
   "arguments":[{"name":"ingredients","required":true}]}
]}}
```

## What this proves, end to end

- The wire protocol really is JSON-RPC 2.0 (`jsonrpc`, `id`, `method`,
  `params` / `result`) over HTTP — not a proprietary format the SDK hides.
- All three MCP primitives (tools, resources, prompts) are live on this one
  server, each independently listable.
- The bearer-token access control is enforced at the transport layer,
  before the JSON-RPC method is even dispatched — a request with no or the
  wrong token never reaches `tools/list` etc. at all.

Reproduce this yourself:
```bash
INGREDIENT_SERVER_API_KEY=secret123 python -m mcp_servers.ingredient_server --http
# then curl as shown above, from a second terminal
```
