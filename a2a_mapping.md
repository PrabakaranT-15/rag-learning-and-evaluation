# Week 10 bonus — A2A task lifecycle mapping

## Should the injected failure (T07) have ended `failed` or `input-required`?

**`failed`, not `input-required`.**

A2A's `input-required` state exists for when the AGENT needs more information
FROM THE USER to proceed — e.g. "which of your allergies should I check
against?" T07's failure is not that kind of gap: the calories question
already has everything it needs from the user (the question itself), and the
allergen/nutrition worker is unavailable for an infrastructure reason (a
simulated 500), not because the user withheld anything. Pausing at
`input-required` and asking the user a question that wouldn't fix the actual
problem (a downed service) would be a false invitation to hand over more
information for no benefit. `failed`, with the specific cause attached (worker
unavailable), is the honest state — the correct next step is retry-later or
escalate, not "ask the user something".

**When WOULD `input-required` be the right call for this orchestrator?** If a
question depended on the user's own allergy list and the user never stated it
(e.g. "is this recipe safe for me?" with no allergy named anywhere in the
conversation) — that is missing information only the user can supply, which
is exactly what `input-required` is for.

## What A2A buys over a plain REST call to the worker

A2A gives the orchestrator a standard way to discover what a worker can do
(the AgentCard's `skills`) and a standard task-state machine
(`submitted → working → input-required/failed → completed`) that a
plain REST call has no shared vocabulary for — a REST 500 is just an HTTP
status code the caller has to interpret itself, while an A2A task explicitly
distinguishes "the task needs more from the user" from "the task failed", so
every A2A-speaking orchestrator handles both cases the same documented way
instead of reinventing its own ad hoc error convention per worker.
