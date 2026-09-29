# Week 10 — failure_case.md

## What was injected

Case **T07**: 'How many calories are in a slice of the brioche?'

`rag/orchestrator.py`'s `run_allergen_specialist(..., force_failure=True)` was
used to simulate the allergen/nutrition worker returning an HTTP 500 on this
one case — a deterministic stand-in for "the worker's service is down", not
a real network fault. The manager retried once (see `attempts` below), then
proceeded to synthesis with the failure explicitly stated as the allergen
worker's only "result".

This case was chosen deliberately: no recipe card in this corpus carries
nutrition data (an established fact from `results.md`/Week 5), so the
allergen/nutrition specialist is the ONLY possible source of an answer here
— the recipe text alone cannot paper over the failure, making this a real
test rather than one the primary retrieval could quietly rescue.

## The actual tool-call result the specialist received

```json
{
  "tool": "check_allergens",
  "args": {
    "ingredients": [
      "Strong white bread flour",
      "bread flour",
      "white bread flour",
      "Fine sea salt",
      "sea salt",
      "Rye levain",
      "levain",
      "Unsalted butter",
      "butter",
      "Whole milk"
    ]
  },
  "result": {
    "error": "HTTP 500: allergen-worker service unavailable"
  },
  "attempts": 2
}
```

## What the allergen worker's own note said

> The allergen‑worker service returned an error, so I cannot provide any calorie or nutritional information for this recipe. I must therefore refuse to give a figure.

## What the orchestrator's FINAL answer actually said

> I cannot answer this from the provided recipe documents.

## Verdict: DEGRADED — the orchestrator correctly refused rather than answering without allergen/nutrition data

Per the rubric ("retried, degraded to a partial answer, or lied by
synthesising an allergen claim the worker never made — and say which in one
line"):

**One line: the orchestrator degraded to a refusal, matching the single agent's own behavior on this same question (both correctly refuse — no card in this corpus has nutrition data, worker-down or not).**
