# Week 10 — race_table.md

> **⚠ Known open issue, not yet resolved:** the single-agent judge context
> originally only included `get_recipe`'s text, missing any extra MCP tool
> data (e.g. `get_substitutes`) a well-grounded answer actually relied on -
> confirmed to wrongly fail one case (T13) before the fix. The fix has been
> re-verified for ['T01', 'T02', 'T04', 'T06', 'T07']. Cases **['T12', 'T13', 'T14', 'T18', 'T20']**
> (marked ⚠ below) still carry the ORIGINAL, unfixed judge context - their
> single-agent pass/fail is provisional, blocked on Groq's daily token cap
> for `openai/gpt-oss-120b`, not yet re-measured. Do not cite this table's
> aggregate pass rate as final until this line is removed.

10 cases, same questions (week5_trace_questions.json T-ids), same judge
(`rag/judge.py`, model `openai/gpt-oss-20b`), same collection
(`fermentation_structure_aware`), both arms run live.

## The four numbers, both arms

| Metric | Single agent (rag/agent.py) | Orchestrator (rag/orchestrator.py) |
|---|---|---|
| Pass rate | 9/10 (90%) | 9/10 (90%) |
| p50 latency | 36.2s | 30.2s |
| p99 latency | 63.7s | 37.7s |
| Total tokens (10 cases) | 79,438 | 42,827 |
| Cost per question | $0.00177 | $0.00114 |

## Context re-send multiplier

**0.5x** (42,827 multi-agent tokens / 79,438 single-agent tokens, 10 cases)

Dominant hand-off: **manager->substitution_worker**, 37% of all orchestrator tokens
(15,719 of 42,827).

Full per-handoff breakdown across all 10 cases:
- manager->substitution_worker: 15,719 tokens (37%)
- manager->synthesis: 14,984 tokens (35%)
- manager->allergen_worker: 12,124 tokens (28%)

## Per-case detail

| Case | Question | Verdict (single/multi) | Tokens (single/multi) |
|---|---|---|---|
| T01 | How much salt goes into the country sourdough reci | pass / pass | 6454 / 4277 |
| T02 | How long should I proof the brioche dough? | pass / pass | 12878 / 4473 |
| T04 | Can I make the kimchi vegan? | pass / pass | 9228 / 4661 |
| T06 | What's the hydration percentage for the sourdough  | pass / pass | 6357 / 4748 |
| T07 | How many calories are in a slice of the brioche? | pass / pass | 6415 / 4244 |
| T12 ⚠ | Is the sourdough bread gluten free? | pass / pass | 6740 / 4446 |
| T13 ⚠ | What's a good substitute for the fish sauce in the | fail / pass | 12162 / 4139 |
| T14 ⚠ | how much butter in the brioche | pass / pass | 6232 / 4300 |
| T18 ⚠ | What's the ratio of flour to water in the rye leva | pass / pass | 6178 / 3775 |
| T20 ⚠ | Why does my sourdough starter smell like nail poli | pass / fail | 6794 / 3764 |

## A named caveat, not swept under the rug: T20 is judge noise, not signal

Both T20 verdicts above were inspected by hand and neither reflects what
actually happened:

- **Single agent "pass"** — the single agent's planner failed to produce
  parseable JSON on this question and `run_agent()` returned its safe-stop
  message verbatim: *"I ran into trouble deciding my next step and cannot
  complete this safely."* The judge scored this a PASS with the reason "the
  assistant correctly refuses to answer due to lack of relevant context" -
  that is a judge misread of a planner crash as an intentional refusal, not
  a real pass.
- **Orchestrator "fail"** — the orchestrator gave a substantive, correctly
  grounded answer connecting the user's "nail polish" smell to the recipe's
  own stated "faint acetone note" - exactly the lay-term-to-technical-term
  reasoning `rag/judge.py`'s own `JUDGE_PROMPT` explicitly instructs a judge
  to accept. The judge failed it anyway ("includes an inference... not
  explicitly stated"), even though the acetone note IS explicitly stated in
  the context quoted in the answer itself.

This is exactly the kind of judge unreliability Week 6
(`judge_human_agreement_rate` in `week6_report.json`) already established
this project's judge is subject to - reported here rather than quietly
re-run until it looked cleaner. The pass-rate row above is the raw, as-
measured number; this note is what a reader should weigh it against.

Raw per-call data: `week10_race_raw.json`.
