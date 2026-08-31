# Week 5 — Error Analysis: reading 20 real traces

Track B (Recipes & food). Collection: `fermentation_structure_aware` (6 cards,
no distractors — see the corpus note below). Retrieval: semantic-only,
top_k=5. Generation model: `gemma-4-31b-it` (see `week5_collect_traces.py`
docstring for why not the app's literal default — same free-tier daily-quota
reason as Week 4).

**Sampling note:** these 20 questions (`week5_trace_questions.json`) were
written as an ordinary user would ask them — easy, casual, vague, comparison,
and out-of-scope questions mixed in on purpose, *not* curated against known
weaknesses the way `week4_failing_questions.json` was. That's what makes this
a fair sample rather than a rigged one.

**Corpus note:** the distractor-enriched `realistic_structure_aware`
collection from Weeks 3–4 needed a `recipes/` folder (195 cards) that no
longer exists on this machine. Rebuilding it silently produced a collection
with zero real distractors under a misleading name — that was caught and
discarded rather than used. This run is therefore the 6-card corpus only.
That changes what kinds of failures are possible (no distractor-collision
failures here) but does not fabricate anything — every trace below is a real
question through the real pipeline.

Full replayable traces (question + all 5 retrieved chunks + answer) are in
`week5_traces.json`.

---

## Open coding — one honest note per trace, written before grouping

| ID | Question | Note |
|---|---|---|
| T01 | How much salt goes into the country sourdough recipe? | Correct, accurate, well-cited. |
| T02 | How long should I proof the brioche dough? | Correct, accurate, well-cited. |
| T03 | What temperature do I bake the focaccia at? | Correct; answered only temperature, which is all that was asked. |
| T04 | Can I make the kimchi vegan? | Correct, accurate, well-cited. |
| T05 | How do I know my starter is ready to use? | Correct and thorough, well-cited. |
| T06 | What's the hydration percentage for the sourdough loaf? | Correct, accurate, well-cited. |
| T07 | How many calories are in a slice of the brioche? | Correct refusal — no nutrition data exists in the corpus. |
| T08 | whats the best flour for the focaccia | Correct fact, but silently reads "best" as "which" — the card makes no comparative claim, and the answer doesn't flag that it reinterpreted the question. |
| T09 | How spicy is the kimchi? | Refuses entirely, even though the retrieved context has a directly relevant fact (gochugaru = 5% of cabbage weight) it could have surfaced with a hedge instead of a flat refusal. |
| T10 | What's the difference between the two kimchi recipes? | Cites three correct facts (dietary status, ferment time, yield) but never mentions the fish-sauce/shrimp vs. shiitake/kombu/tamari swap — the actual headline difference — because that specific chunk didn't make the top-5 for this query. A retrieval miss on the most salient content, not a generation error. |
| T11 | how long does the levain take to ripen | Correct, accurate, well-cited. |
| T12 | Is the sourdough bread gluten free? | Technically correct for every item cited, but answered for all four gluten-containing recipes when the question likely meant just the one literally called "Sourdough Loaf." |
| T13 | What's a good substitute for the fish sauce in the kimchi? | Correct, accurate, well-cited. |
| T14 | how much butter in the brioche | Correct, accurate, well-cited. |
| T15 | Can I freeze the ciabatta dough? | Correct refusal — ciabatta isn't in this corpus, and it did NOT hallucinate an answer from a similar bread card. |
| T16 | whats the egg wash for | Correct refusal — the card never states the purpose of the egg wash, only that one is applied. |
| T17 | How do I store leftover kimchi? | Answers a different, nearby question ("when does it go in the fridge during fermentation") instead of what was actually asked ("how do I store already-finished leftovers"), presented as if it fully answers the question. |
| T18 | What's the ratio of flour to water in the rye levain? | Correct, accurate, well-cited. |
| T19 | Tell me how to make naan bread | Correct refusal — naan isn't in this corpus, no hallucination. |
| T20 | Why does my sourdough starter smell like nail polish? | **Refuses despite the rank-1 retrieved chunk directly explaining this** ("smells sharply sour with a faint acetone note" — acetone IS the nail-polish smell). The model failed to connect the lay term to the technical one and declined to answer a question it had already been handed the answer to. |

---

## Named problem groups (formed only after reading all 20)

### 1. Over-refusal despite relevant grounding
The answer already contains — or almost contains — what was asked, but the
model declines anyway rather than connecting a lay term to a technical one,
or hedging on a partial fact.
- **T20** (severe): rank-1 chunk states "acetone note" outright; the question
  asks about "nail polish" smell — the same thing. Refused anyway.
- **T09** (mild): gochugaru quantity is retrieved and relevant to "how
  spicy," but there's no explicit heat-level claim to hedge from, so refusal
  here is more defensible than T20's.

### 2. Comparison/summary answers omit the single most relevant fact
When asked to compare two things, the model returns several correct but
secondary facts while missing the one fact that actually answers the
question, because that specific chunk wasn't retrieved.
- **T10**: the ingredient-substitution sentence (the actual difference
  between the two kimchi recipes) never appears in top-5 for a comparison
  query, even though standalone questions about it (T04, T13) retrieve it
  fine.

### 3. Answers a nearby question instead of the one asked
A real, grounded, adjacent question gets answered and presented as if it
fully addresses the literal one.
- **T17**: "storing leftovers" (an already-finished dish) answered as
  "when fermentation ends and it goes in the fridge" (an in-progress dish).

### 4. Silent scope reinterpretation
The assistant answers a broader or subtly different version of the question
without saying it did so. Facts are correct, just possibly more/different
than intended.
- **T08**: "best" flour → "which" flour.
- **T12**: "the sourdough bread" → all four gluten-containing recipes.

---

## Ranked by frequency × severity

| Rank | Problem | Frequency | Severity | Why it ranks here |
|---|---|---|---|---|
| 1 | Over-refusal despite relevant grounding | 2/20 | High | Contains the single worst failure in the batch (T20) — actively withholds a directly-answerable, reassuring fact from someone who might be worried enough to discard a perfectly good starter. |
| 2 | Comparison answers omit the key fact | 1/20 | Medium-High | Only one instance here, but structural — any comparison-style question is at risk, since the substitution fact is provably retrievable (T04, T13 find it fine) but doesn't surface for comparison phrasing. |
| 3 | Answers a nearby question instead of the one asked | 1/20 | Medium | Misleading but not fabricated; a user with genuine leftovers gets fermentation-stage advice instead. |
| 4 | Silent scope reinterpretation | 2/20 | Low | No factual errors, just possibly answers more than intended. |

---

## Chosen target and prediction

**Target: Problem 1 — over-refusal despite relevant grounding (T20 as the
lead case).**

**Prediction, written before any fix is made:** the refusal instruction in
`rag/generator.py`'s `RECIPE_PROMPT` is currently strict about not
"estimating, approximating, or reasoning from general cooking knowledge" —
which is right for preventing fabrication, but T20 shows it's also
suppressing legitimate reasoning *within* the retrieved context (recognizing
that "nail polish smell" and "acetone note" name the same thing). I predict
that adding an explicit instruction to reason about synonyms and lay
descriptions of what's *literally in the retrieved text* — while keeping the
"no outside knowledge" rule for anything not grounded — will flip T20 from
refusal to a correct, cited answer, **without** causing T07, T15, T16, or T19
(genuine out-of-corpus refusals) to start hallucinating, since those four
have no relevant content in their retrieved context at all to reason from.
That's the number Week 6 should check first: do the 4 correct refusals stay
refusals, and does T20 (and ideally T09) flip to a grounded answer.
