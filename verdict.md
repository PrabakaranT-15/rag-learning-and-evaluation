# Week 10 — verdict.md

**Verdict: KEEP, PROVISIONALLY the orchestrator.**

Pass rate: single agent 90% vs orchestrator 90% — the team does not answer more questions correctly (see race_table.md's T20 note - one case in each arm's verdict is judge noise, not a clean signal).
Tokens: the orchestrator uses **0.5x** the single agent's tokens for these 10 cases (cheaper), dominated by `manager->substitution_worker` (37% of its tokens).

**Sunk-cost bias, named out loud:** a team was already built for this task, and "since it's built, might as well use it" is exactly the reasoning this verdict has to resist - the orchestrator's build cost is sunk and irrelevant to whether it should ship; only the numbers above are.

This result runs counter to the brief's own expectation ("often the single agent wins") and the reason is architectural, not because delegation is inherently free: this orchestrator's manager retrieves the recipe DETERMINISTICALLY (one search_recipes + one get_recipe call, no LLM planning step), while the single agent's plan-act-observe loop spends multiple real LLM calls just deciding what to search/verify/fetch before it ever generates an answer - MCP tool discovery and agentic flexibility have a real per-question token cost of their own, and this race measured that cost as larger than three narrow, fixed hand-offs plus one synthesis call. That is a genuine finding about THIS fixed decomposition, not a general law that multi-agent is cheaper - a manager with its own agentic retrieval loop would likely erase this gap.
