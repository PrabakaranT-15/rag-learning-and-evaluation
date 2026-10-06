# Week 11 — Production: Observability, Cost & the Failure → Test Loop

Track B (Recipes & food) · drill: *"find the dairy-free swap that wasn't"*

## Status — what is done, and what is not

| Deliverable | State |
|---|---|
| Per-request logging (find any past answer) | **Done and tested** — `rag/observability.py`, 11 tests |
| Time + cost per step, not one lump | **Done and tested** — nested spans, cost rolls up to the parent step |
| Support drill (planted bad answer, vague complaint) | **Done** — run below on a *synthetic* log |
| Cost per request: baseline | **Done** — from Week 10's real live run |
| Cost per request: one improvement, **measured** | **Built, not yet measured live** — needs `GROQ_API_KEY` + `GOOGLE_API_KEY` (both empty in `.env`). Run `python week11_measure.py`, then `python week11_cost_report.py` |
| 10× plan | **Done** (below) |
| One real failure → permanent test | **Done** — 6 cases replayed by `tests/test_failures.py` |

Full suite: `python -m pytest tests -q` → **63 passed**, no network or API key needed.

Nothing in this report is an invented measurement. The drill uses 301 records
that are labelled `"synthetic": true` and live in their own file; the cost
baseline is Week 10's recorded live data; the *after* number is deliberately
left blank until it can be measured for real.

---

## 1. What is logged per request

Every request — chatbot, agent, fixed workflow, orchestrator — writes one JSON
line to `logs/requests.jsonl` when it ends (`trace_request()` in
[rag/observability.py](rag/observability.py)):

| Field | Why it is there |
|---|---|
| `trace_id`, `timestamp`, `kind` | find it, order it, know which path answered |
| `question`, `answer` | what a customer actually complains about |
| `restriction`, `collection`, `model`, `retrieval_mode`, `top_k` | replay it exactly |
| `retrieved_chunk_ids`, `recipe_id`, `steps` | was it the wrong *document* or the wrong *answer*? |
| `total_ms`, `llm_calls`, `cache_hits`, `prompt/completion/total_tokens`, `cost_usd` | request-level totals |
| `guard_flags` | automatic checks that already fired on the answer (see §5) |
| `error`, `stopped_reason` | crashes and budget stops |
| **`spans[]`** | **each step's own `duration_ms`, tokens and `cost_usd`**, with `parent_id` so a `plan` step contains its `llm` call |

Not logged, on purpose: API keys, and full prompts (only `prompt_hash` and
`prompt_chars`) — so a log can be shared with a mentor or attached to a ticket.
Logging failures are swallowed: observability can never take the app down.

Spans are OpenTelemetry-shaped (`span_id`, `parent_id`, `start_ms`,
`duration_ms`, attributes), so forwarding to LangSmith / Phoenix / an OTel
collector later means adding an exporter, not touching any call site.

Every model call goes through one function (`generator._generate`), which
charges the call's *real* Groq token counts and price to the current span.

## 2. The support drill

**Complaint (vague, no trace id):** *"Yesterday somebody was told the brioche
had a dairy-free option, and it wasn't."*

```bash
python week11_support_drill.py     # builds logs/drill_requests.jsonl: 301 records, 1 planted
```

**Step 1 — narrow by what the complaint gives you (topic + day).**
```bash
python week11_logs.py --log logs/drill_requests.jsonl --text "brioche dairy-free" --since 2026-10-05 --until 2026-10-06
```
→ **12 matches out of 301.** Eleven are legitimate oat-milk answers; one is not.
Reading 12 by hand is feasible, but not at thousands of requests — hence step 2.

**Step 2 — let the guard point at it.**
```bash
python week11_logs.py --log logs/drill_requests.jsonl --flagged
```
→ **exactly 1 match:** trace `d41ry7ee0001`, kind `agent`, 2026-10-05 14:37 UTC:

```
question : Is there a dairy-free way to make the brioche?
answer   : Yes - a dairy-free swap for the butter in the brioche is ghee at a 1:1 ratio ...
GUARD    : {'dairy_free_swap': ['ghee']}
steps    : embedding 1483.8 ms · llm 2179.6 ms $0.000633 · llm 1893.8 ms · llm 2276.6 ms
```

**Step 3 — root cause.** Ghee is clarified butter: it is dairy. It is also not
in `data/ingredients.json` at all — the ingredient tool's only dairy-free butter
swaps are *vegan block butter* and *coconut oil* — so the model produced the swap
from general knowledge, which the grounded prompt forbids, and attached a
citation to it. The tool is not at fault (verified by test F006); the answer
was ungrounded.

**Step 4 — it can't come back silently.** See §5: the guard now flags this
class of answer on every request, and the case is a permanent test.

## 3. Cost per request

**Baseline** (real: Groq token counts × the rate card in `rag/generator.py`,
from Week 10's live race, 10 questions — `python week11_cost_report.py`):

| Arm | $/request | tokens/request | s/request |
|---|---|---|---|
| Single agent | **$0.00177** | 7,944 | 36.6 |
| Orchestrator | $0.00114 | 4,283 | 28.1 |

Where the agent's money goes: it makes ~4–6 model calls per question (every
`plan` step re-sends the growing transcript) plus one answer call. The
per-step `cost_usd` in the new logs now shows this per request instead of as a
Week 10 aggregate.

**Improvement made: exact-match caching** ([rag/cache.py](rag/cache.py), SQLite
so the Streamlit process and the MCP subprocesses share it).

| Cache | Default | Saves |
|---|---|---|
| Embeddings, keyed on (model, text) | **on** | a Gemini call *and* its ~1.1 s pacing sleep, for every repeated query — the chatbot path, the agent's search, and every retry all embed the same query |
| LLM responses, keyed on (model, prompt) | off; **on in `app.py`** | the whole call: 0 tokens, $0 on an identical prompt |

The LLM cache is off by default so the Week 4–10 experiments, which deliberately
re-sample a model, stay reproducible exactly as they were measured.

**Measuring it** — `python week11_measure.py` runs the same 10 questions twice
through a throw-away cache (cold, then warm) and writes two ordinary request
logs; `python week11_cost_report.py` compares them. Expected from the design:
the warm pass makes **0 billed LLM calls** and skips every embedding sleep. The
real number — and the honest caveat that a repeated question is the best case,
so the true saving is *that × your repeat rate* — will be printed by the report.
**This measurement has not been run yet** (no API keys on this machine).

**Deliberately not done:**
- *Semantic caching.* A near-duplicate question can have a different safe
  answer ("is it dairy-free?" vs "is it gluten-free?" embed very close). On an
  allergen/diet app, returning a wrong cached answer is worse than paying for a
  fresh one. Exact-match only.
- *Model routing (LiteLLM).* `gpt-oss-20b` is half the price of `-120b`
  ($0.075/$0.30 vs $0.15/$0.60 per M tokens, in `generator.py`), so routing
  easy questions to it is the next-largest lever — but it needs the Week 6
  judge to prove quality holds, so it is a plan, not a change made blind.
- *Prompt caching.* Groq applies it server-side; nothing to implement.
- *Fine-tuning.* Last resort, and nothing here points to it: the failures found
  are grounding/filtering bugs, not model-capability gaps.

## 4. The 10× plan — what breaks first

Today: one Streamlit process, one Chroma `PersistentClient`, two stdio MCP
subprocesses, a global 1 s sleep between Groq calls (`MIN_INTERVAL_SECONDS`).
In order of when each one fails:

| # | Breaks | Why (from the code) | Plan |
|---|---|---|---|
| 1 | **LLM throughput** | The 1 s pacing sleep caps a process at ≤60 calls/min. An agent request is ~5 calls, so ≈12 agent requests/min/process — 10× the traffic queues, it doesn't just slow down | Per-process pacing → a shared rate limiter sized to the real Groq tier; fall back to `gpt-oss-20b` on `RateLimitError` instead of sleeping up to 60 s; route simple questions to the chatbot path (1 call) |
| 2 | **Latency** | Streamlit runs everything synchronously and the page runs chatbot **and** agent per question (~37 s agent p50) | Serve the answer path as an API (FastAPI) with the agent optional/async; stream the chatbot answer first |
| 3 | **Cost** | $0.00177/agent request × 10× traffic | Caching (done) → routing to the 20B model → cap `max_steps` |
| 4 | **Embedding quota** | Gemini free tier, 1.1 s pacing, one call per chunk | Embedding cache (done); batch embeddings on ingest; paid tier |
| 5 | **Vector store** | Local single-process Chroma, one writer | Hosted/server Chroma or pgvector; stateless app processes |
| 6 | **MCP subprocesses** | Each app process spawns its own two servers | Run them once as HTTP servers (the ingredient server already supports `--http` + bearer auth) |
| 7 | **The log itself** | One JSONL file, one lock | Rotate daily; ship to a log store / OTel collector; sample full spans if volume demands |

## 5. The failure → test loop

`tests/failures/cases.json` holds every confirmed failure with where it was
found; `tests/test_failures.py` replays **all** of them on every `pytest` run.

| Case | Found via | Guards against |
|---|---|---|
| F001–F003 | code review of `app.py` | "non-vegetarian" / "non-vegan" being read as "vegetarian" / "vegan" and filtering to the **opposite diet** (fixed in `rag/restrictions.py`); F003 stops over-fixing |
| **F004** | **the support drill, trace `d41ry7ee0001`** | a "dairy-free" answer that recommends ghee |
| F005 | false-positive check on F004 | the guard must not flag the *correct* swaps (oat milk, vegan butter) |
| F006 | root-cause check on F004 | the ingredient tool never offers a non-dairy-free "dairy-free" swap |

The loop in practice: **log → find → add one JSON entry → `pytest`.** F004 goes
further than a test: `rag/guards.py` runs on every answer and writes
`guard_flags` into the log, so the next ghee-style answer is found with
`--flagged` instead of waiting for a customer.

Writing these tests also caught two bugs in the new code itself, both fixed:
the trace timer read 0 ms for short steps on Windows (`time.monotonic()` ticks
~15 ms; now `perf_counter`), and the first guard treated the word "free" in
"dairy-**free** swap is ghee" as a negation, so it missed the exact failure it
was written for.

## Files

| New | Purpose |
|---|---|
| `rag/observability.py` | traces, spans, JSONL log, search/format |
| `rag/cache.py` | embedding + LLM exact-match caches |
| `rag/guards.py` | `guard_flags` (dairy-free-swap check) |
| `rag/restrictions.py` | restriction detection (moved from `app.py`, fixed) |
| `week11_logs.py` | search CLI: `--text --since --flagged --trace-id --slow --stats` |
| `week11_support_drill.py` | builds the synthetic drill log |
| `week11_measure.py`, `week11_cost_report.py` | live cold/warm measurement and the cost report |
| `tests/test_observability.py`, `tests/test_failures.py`, `tests/failures/cases.json` | 17 new tests |

Changed: `generator.py` (cache + per-call span), `embeddings.py` (cache + span),
`agent.py`, `fixed_workflow.py`, `orchestrator.py`, `app.py` (one trace per
request), `.gitignore`.
