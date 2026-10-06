# RAG Recipe Assistant

A week-by-week RAG training project: structure-aware chunking, hybrid retrieval
(semantic + keyword, RRF), grounded generation, and three ways of answering a
recipe question that are raced against each other:

| Path | Code | Idea |
|---|---|---|
| Chatbot (fixed workflow) | `rag/fixed_workflow.py`, `app.py:run_fixed_path` | filter → search → generate |
| Agent | `rag/agent.py` | plan → act → observe loop, tools discovered over MCP |
| Orchestrator | `rag/orchestrator.py` | manager + substitution/allergen specialists |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # add GROQ_API_KEY and GOOGLE_API_KEY
python ingest_fermentation_cards.py   # builds the 3 Chroma collections in data/chroma
streamlit run app.py
pytest tests/
```

`data/chroma/` and `documents/*.pdf` are gitignored, so a fresh clone must run
the ingest step (it embeds one chunk per call; expect a few minutes).

## Layout

- `rag/` – library code (chunkers, vector store, hybrid retriever, generator, agent, MCP client, evaluation)
- `mcp_servers/` + `mcp_servers.json` – MCP servers the agent discovers tools from
- `fermentation_cards/`, `data/ingredients.json` – corpus and ingredient database
- `week*_*.py`, `*_report.*`, `race_table.md`, `verdict.md` – per-week experiments and their results

## Observability (Week 11)

Every request is logged to `logs/requests.jsonl` with per-step time, tokens and cost.

```bash
python week11_logs.py --text "brioche dairy-free" --since 2026-10-05   # find a past answer
python week11_logs.py --flagged                                         # answers a guard flagged
python week11_logs.py --stats                                           # cost / latency summary
python week11_cost_report.py                                            # cost baseline + cache improvement
```

See [week11_report.md](week11_report.md). Caches: `RAG_EMBED_CACHE=0` / `RAG_LLM_CACHE=1` toggle them.
