# Week 4 — Debugging Retrieval: failure labels + hybrid search result

Collection: `realistic_structure_aware`  ·  top_k = 3  ·  generation model (both conditions): `gemma-4-31b-it`

## Headline number

**hit-rate@3: 11/11 (before) -> 11/11 (after hybrid search)**

End-to-end pass rate (right doc AND right answer): 10/11 -> 10/11

**Honest read: this is a ceiling effect, not a win.** At top_k=3, semantic-only retrieval already found the expected recipe for all 11 questions in this set, including ones deliberately built around rare-term collisions and a real same-name distractor — so hybrid search had no `wrong_document` failures left to fix on this question set, and the hit-rate@3 number cannot move. The one real failure that survived (W4) is a *generation*-layer failure, not a retrieval one, and is reported honestly below as NOT fixed by this change — hybrid is a retrieval-side fix and this isn't a retrieval-side problem. See section 6 of `results.md` for the matching root-cause diagnosis from Week 3.

## Per-question before / after

| ID | Category | Before | After | Outcome |
|---|---|---|---|---|
| W1 | keyword_exact | pass (rank 1) | pass (rank 1) | unchanged (pass) |
| W2 | keyword_exact | pass (rank 1) | pass (rank 1) | unchanged (pass) |
| W3 | keyword_exact | pass (rank 1) | pass (rank 1) | unchanged (pass) |
| W4 | contrast | right_document_wrong_answer (rank 1) | right_document_wrong_answer (rank 3) | still failing |
| W5 | contrast | pass (rank 2) | pass (rank 2) | unchanged (pass) |
| W6 | cross_distractor | pass (rank 1) | pass (rank 1) | unchanged (pass) |
| W7 | cross_distractor | pass (rank 1) | pass (rank 1) | unchanged (pass) |
| W8 | cross_distractor | pass (rank 1) | pass (rank 1) | unchanged (pass) |
| W9 | keyword_exact | pass (rank 1) | pass (rank 1) | unchanged (pass) |
| W10 | keyword_exact | pass (rank 1) | pass (rank 1) | unchanged (pass) |
| W11 | keyword_exact | pass (rank 1) | pass (rank 1) | unchanged (pass) |

## Evidence

### W1 — According to the float test, how do you know the rye levain starter is ripe?
- Hypothesis (written before running anything): 'float test' is a rare, distinctive phrase used nowhere else in the corpus - expected to be helped by BM25/hybrid if semantic search dilutes it among the other sourdough/levain cards.
- BEFORE: **pass** — expected recipe at rank 1, answer tokens matched
- AFTER: **pass** — expected recipe at rank 1, answer tokens matched

### W2 — What test proves the sourdough brioche dough has been kneaded enough before the butter stage finishes?
- Hypothesis (written before running anything): 'windowpane test' is a rare exact phrase - expected to be helped by hybrid.
- BEFORE: **pass** — expected recipe at rank 1, answer tokens matched
- AFTER: **pass** — expected recipe at rank 1, answer tokens matched

### W3 — How much saeujeot does the traditional napa kimchi use, and what percentage of the cabbage weight is that?
- Hypothesis (written before running anything): 'saeujeot' is a rare term appearing only on ferment_005, but ferment_005 and ferment_006 are near-duplicate documents (same cabbage/salt/gochugaru/radish rows) - risk that semantic search pulls the vegan sibling instead. Expected to be helped by hybrid since the exact term only occurs in ferment_005.
- BEFORE: **pass** — expected recipe at rank 1, answer tokens matched
- AFTER: **pass** — expected recipe at rank 1, answer tokens matched

### W4 — In the vegan napa kimchi, what replaces the salted shrimp and anchovy fish sauce from the traditional recipe, and how much longer does it take to ferment?
- Hypothesis (written before running anything): Same shape as the already-documented Q7 failure (results.md section 6): the query names the OTHER card's distinctive ingredients ('salted shrimp', 'anchovy fish sauce'), which occur densely in ferment_005's own text. Hybrid/BM25 is NOT expected to fix this - term-overlap scoring can favor the wrong, lexically dense chunk just as much as embedding distance did.
- BEFORE: **right_document_wrong_answer** — expected recipe 'ferment_006' at rank 1 (within top-3), but none of the expected answer tokens [['shiitake', 'kombu'], ['tamari']] appear in the generated answer
- AFTER: **right_document_wrong_answer** — expected recipe 'ferment_006' at rank 3 (within top-3), but none of the expected answer tokens [['shiitake', 'kombu'], ['tamari']] appear in the generated answer

### W5 — How many days does the traditional napa kimchi ferment at 20C before it goes into the refrigerator?
- Hypothesis (written before running anything): ferment_006's method explicitly discusses 'the traditional card' and its fermentation time for comparison ('around a day longer than the traditional card requires'), so a query about the traditional recipe's own timing risks pulling the vegan card's comparison sentence instead. Not confidently expected to be fixed by hybrid, since both cards contain the phrase '2 days'/'2 to 3 days' near kimchi vocabulary.
- BEFORE: **pass** — expected recipe at rank 2, answer tokens matched
- AFTER: **pass** — expected recipe at rank 2, answer tokens matched

### W6 — At what oven temperature and for how many minutes is the sourdough focaccia baked?
- Hypothesis (written before running anything): The realistic index's forced distractor set includes recipe_066_focaccia.md - a real, differently-timed focaccia recipe with the same dish name. Expected to stress whether the retriever pulls the distractor's bake time instead of ferment_003's. Uncertain whether hybrid helps or hurts, since both documents share the word 'focaccia' densely.
- BEFORE: **pass** — expected recipe at rank 1, answer tokens matched
- AFTER: **pass** — expected recipe at rank 1, answer tokens matched

### W7 — What internal temperature should the 2kg country sourdough loaf reach when it's fully baked?
- Hypothesis (written before running anything): Baking-temperature questions are generic phrasing shared with many bread distractors (naan, roti, ciabatta, pizza dough) in the realistic index - expected to mostly already pass, included as a control question.
- BEFORE: **pass** — expected recipe at rank 1, answer tokens matched
- AFTER: **pass** — expected recipe at rank 1, answer tokens matched

### W8 — How long does the enriched sourdough brioche dough cold retard for before shaping?
- Hypothesis (written before running anything): Control question - specific numeric answer, no strong lexical overlap with distractors, expected to already pass.
- BEFORE: **pass** — expected recipe at rank 1, answer tokens matched
- AFTER: **pass** — expected recipe at rank 1, answer tokens matched

### W9 — What percentage of the cabbage weight does the gochugaru make up in the vegan napa kimchi?
- Hypothesis (written before running anything): The gochugaru row ('Gochugaru (Korean chilli flakes) | 100 g | 5%') is BYTE-IDENTICAL text in both ferment_005 and ferment_006 - nothing in that single row distinguishes which card it belongs to. Only 'vegan' framing elsewhere in the chunk can disambiguate. High risk of wrong_document; uncertain whether hybrid helps, since BM25 on this row alone can't disambiguate either.
- BEFORE: **pass** — expected recipe at rank 1, answer tokens matched
- AFTER: **pass** — expected recipe at rank 1, answer tokens matched

### W10 — What smell and appearance indicate the rye levain has gone past ripe and over-fermented?
- Hypothesis (written before running anything): Paraphrased question (no exact phrase reused from the card) testing whether semantic search alone is enough without needing BM25 - included as a semantic-should-win control.
- BEFORE: **pass** — expected recipe at rank 1, answer tokens matched
- AFTER: **pass** — expected recipe at rank 1, answer tokens matched

### W11 — Which recipe describes a ripe starter as smelling sharply sour with a faint acetone note?
- Hypothesis (written before running anything): 'acetone' is a single rare, highly distinctive word used nowhere else in the corpus - the clearest exact-term case, expected to be helped by hybrid if it isn't already at rank 1 semantically.
- BEFORE: **pass** — expected recipe at rank 1, answer tokens matched
- AFTER: **pass** — expected recipe at rank 1, answer tokens matched

## What hybrid search did NOT fix

- **W4** (contrast): still `right_document_wrong_answer` after hybrid — expected recipe 'ferment_006' at rank 3 (within top-3), but none of the expected answer tokens [['shiitake', 'kombu'], ['tamari']] appear in the generated answer

## Exploratory probes (retrieval-only, not scored)

These aren't part of the scored 11 because their ground truth is genuinely multi-valid — more than one recipe_id is a legitimate hit. They exist to show what hybrid actually does to a real rare-term collision, independent of pass/fail scoring.

**What is the bend test used to check?**

'bend test' occurs verbatim in THREE documents: ferment_005, ferment_006 (both fermentation cards) and recipe_169 (a distractor card's Tips section: 'The bend test on the cabbage stem is the correct measure of salting, not the timer.'). No single recipe_id is 'the' right answer here - this probe exists to show what hybrid actually does to a real rare-term collision, not to be scored pass/fail.

- semantic top-5 recipe_ids: `['ferment_005', 'recipe_169', 'ferment_006', 'ferment_004', 'recipe_040']`
- hybrid top-5 recipe_ids: `['recipe_169', 'ferment_005', 'ferment_006', 'recipe_002', 'ferment_002']`

