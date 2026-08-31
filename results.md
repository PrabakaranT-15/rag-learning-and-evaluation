# Week 3 Practical — Task Set B

## Ingest the new cookbook chapter and prove your chunking finds the answer

Domain: Recipes & food. Extension of the existing Week 3 RAG app.

---

## 1. What was ingested, and what was not

**Only the 6 new fermentation cards were indexed.** The existing corpus was not
re-indexed and no existing collection was cleared.

The original app's `build_index()` deletes every id in a collection before
writing, which would have violated requirement 6. A new `add_chunks()` path was
added that appends and upserts instead, and the pre-existing `legal_contracts`
collection was counted before and after the ingest to prove it was untouched:

```
Collections BEFORE ingest:
  legal_contracts: 5 chunks

Collections AFTER ingest:
  fermentation_baseline_500: 10 chunks (+10)
  legal_contracts: 5 chunks (+0)  <-- pre-existing, UNTOUCHED
  fermentation_baseline_120: 28 chunks (+28)
  fermentation_structure_aware: 33 chunks (+33)
```

Source: `ingest_report.json`, produced by `ingest_fermentation_cards.py`.

### The 6 cards

| recipe_id | recipe_name | cuisine | dietary_tags |
|---|---|---|---|
| ferment_001 | 2kg Country Sourdough Loaf | European | vegan, vegetarian, dairy-free, egg-free, nut-free, contains-gluten |
| ferment_002 | Rye Levain Starter | European | vegan, vegetarian, dairy-free, egg-free, nut-free, contains-gluten |
| ferment_003 | Sourdough Focaccia | Italian | vegan, vegetarian, dairy-free, egg-free, nut-free, contains-gluten |
| ferment_004 | Sourdough Brioche | French | vegetarian, contains-dairy, contains-egg, contains-gluten, nut-free |
| ferment_005 | Traditional Napa Cabbage Kimchi | Korean | non-vegetarian, contains-fish, contains-shellfish, gluten-free, dairy-free, egg-free, nut-free |
| ferment_006 | Vegan Napa Cabbage Kimchi | Korean | vegan, vegetarian, gluten-free, dairy-free, egg-free, nut-free |

Each card is a structured ingredient table (ingredient, weight, baker's
percentage) followed by loose method prose and an allergen note, as specified.

### Metadata on every chunk

Every chunk carries `source_file`, `recipe_id`, `cuisine` and `dietary_tags`.
`validate_chunk_metadata()` runs before anything is written and raises, aborting
the ingest, if any chunk is missing any of the four. All three collections
reported `metadata validation: PASSED`.

Chroma's where-clause has no substring operator, so a comma-joined
`dietary_tags` string cannot be matched partially. At ingest time
`diet_flags()` derives one boolean per tag from that exact string
(`vegan` → `diet_vegan=True`). The required `dietary_tags` field is still stored
verbatim; the booleans exist so the tags are filterable at all.

---

## 2. The 8 questions

These were written from the cards **before any retrieval was run**, and stored in
`eval_questions.json` with `"_authored_before_retrieval": true`. Four of the
eight depend on a row inside an ingredient table, exceeding the required three.

| ID | Question | Known answer | Recipe | Section | Table row? |
|---|---|---|---|---|---|
| Q1 | Exact fine sea salt weight for the 2kg country sourdough? | 40 g (2%) | ferment_001 | Ingredients | **yes** |
| Q2 | Hydration percentage of the sourdough focaccia? | 85% (850 g water / 1000 g flour) | ferment_003 | Ingredients | **yes** |
| Q3 | Baker's percentage of butter in the sourdough brioche? | 40% (200 g) | ferment_004 | Ingredients | **yes** |
| Q4 | How much gochugaru in the traditional kimchi, and what %? | 100 g, 5% of cabbage weight | ferment_005 | Ingredients | **yes** |
| Q5 | Temperature and time for the sourdough baked with lid on? | 250C for 20 minutes | ferment_001 | Method | no |
| Q6 | How long does the rye levain take to ripen, at what temp? | 8–12 hours at 22–24C | ferment_002 | Method | no |
| Q7 | In the vegan kimchi, what replaces salted shrimp and fish sauce? | Shiitake + kombu stock plus tamari | ferment_006 | Method | no |
| Q8 | Which allergens are declared on the brioche card? | Gluten (wheat, rye), dairy (milk, butter), egg | ferment_004 | Allergen Note | no |

---

## 3. Hit-in-top-5, both chunking strategies, same 8 questions

Embedding model held constant at `gemini-embedding-001` across every run, so the
only variable that moves between collections is the chunker.

Two metrics are recorded:

- **R = hit_recipe@5** — the expected `recipe_id` appears in the top 5. The
  baseline chunker has no concept of sections, so a section-level metric would
  auto-fail it for reasons unrelated to retrieval quality.
- **A = answer_in@5** — at least one retrieved chunk *literally contains the
  answer token*. This is the metric that predicts whether generation can
  succeed; retrieving the right recipe is worthless if the chunk that came back
  does not contain the number.

### Headline table — 6-cards-only index (requirement-6 compliant)

| Question | Table row? | A: baseline 500/100 | A: baseline 120/20 | B: structure-aware |
|---|---|---|---|---|
| Q1 | yes | PASS | PASS | PASS |
| Q2 | yes | PASS | PASS | PASS |
| Q3 | yes | PASS | PASS | PASS |
| Q4 | yes | PASS | PASS | PASS |
| Q5 | no | PASS | PASS | PASS |
| Q6 | no | PASS | PASS | PASS |
| Q7 | no | PASS | PASS | PASS |
| Q8 | no | PASS | PASS | PASS |
| **Total hit@5** | | **8/8** | **8/8** | **8/8** |

Per-question detail, including all five ranked results with chunk_id, distance,
recipe_id, source_file, section and text for every question under every
strategy, is in **`search_dump_cards6.txt`** and `eval_results.json`.

### This result is a ceiling effect, not a tie

**8/8 for all three strategies is a non-result and I am reporting it as one.**
With only 6 cards the index holds 10–33 chunks, so top-5 returns between 15% and
50% of the entire corpus. Any chunker "hits" under those conditions. Tightening
the metric to rank-1 did not rescue it either:

| Metric | baseline 500/100 | baseline 120/20 | structure-aware |
|---|---|---|---|
| hit_recipe@5 | 8/8 | 8/8 | 8/8 |
| hit_recipe@1 | 8/8 | 8/8 | **7/8** |
| MRR | 1.00 | 1.00 | 0.938 |

The structure-aware chunker is *worse* at rank 1 on this index, losing Q7. That
failure is diagnosed in section 6 and it is the most useful thing this
experiment produced.

The task assumes "your RAG app already indexes the old cards". It did not — the
app indexed three legal PDFs and no recipes at all. A second, harder condition
was therefore built in which the 6 new cards compete against the 20 most
fermentation- and bread-adjacent cards from the existing 195-card recipe corpus
(dosa, idli, appam, adai, uttapam — all fermented batters — plus focaccia,
ciabatta, pizza doughs, naan, and three kimchi dishes). See section 7.

---

## 4. Metadata filter on dietary_tags

The corpus deliberately contains two near-identical kimchi cards. They are
semantically almost the same document, so unfiltered vector search has no way to
prefer one; the dietary tags are the only thing that can separate them.

- Collection: `fermentation_structure_aware`
- Query: `How do I make fermented napa cabbage with chilli flakes and radish?`
- Filter: `{"diet_vegan": True}` (derived from `dietary_tags`)

### UNFILTERED (top 3 of 5)

```
1. chunk_id=ferment_005__method__1
   score(distance)=0.4760   recipe_id=ferment_005
   dietary_tags=non-vegetarian,contains-fish,contains-shellfish,gluten-free,...

2. chunk_id=ferment_005__ingredients__0
   score(distance)=0.5155   recipe_id=ferment_005
   dietary_tags=non-vegetarian,contains-fish,contains-shellfish,gluten-free,...

3. chunk_id=ferment_006__ingredients__0
   score(distance)=0.5240   recipe_id=ferment_006
   dietary_tags=vegan,vegetarian,gluten-free,dairy-free,egg-free,nut-free
```

### FILTERED (top 3 of 5)

```
1. chunk_id=ferment_006__ingredients__0
   score(distance)=0.5240   recipe_id=ferment_006
   dietary_tags=vegan,vegetarian,gluten-free,dairy-free,egg-free,nut-free

2. chunk_id=ferment_006__method__1
   score(distance)=0.5358   recipe_id=ferment_006
   dietary_tags=vegan,vegetarian,gluten-free,dairy-free,egg-free,nut-free

3. chunk_id=ferment_006__method__2
   score(distance)=0.5445   recipe_id=ferment_006
   dietary_tags=vegan,vegetarian,gluten-free,dairy-free,egg-free,nut-free
```

**top-1 unfiltered:** `ferment_005__method__1` — non-vegetarian (salted shrimp,
anchovy fish sauce)
**top-1 filtered:** `ferment_006__ingredients__0` — vegan
**TOP-1 CHANGED: True**

Full lists with all five results and scores: `filter_demo.txt`.

---

## 5. Generation

### 5a. Three answerable questions, with verified citations

The recipe generator forbids outside knowledge, forbids inventing quantities,
percentages, temperatures and nutrition values, and requires a citation of the
form `[recipe_id=… | chunk_id=… | source_file=…]` for every factual claim.

Every citation below was verified programmatically against ChromaDB by
`generate_answers.py`: the chunk_id must resolve, the claimed recipe_id must
match the stored metadata, and the resolved chunk must actually contain the
claimed value.

**Q1 — "What is the exact fine sea salt weight for the 2kg country sourdough?"**

> The exact fine sea salt weight for the 2kg Country Sourdough Loaf is 40 g
> `[recipe_id=ferment_001 | chunk_id=ferment_001__ingredients__0 | source_file=ferment_001_country_sourdough_2kg.md]`.

```
chunk_id: ferment_001__ingredients__0
  resolves in ChromaDB : True
  recipe_id matches    : True (claimed ferment_001, actual ferment_001)
  contains '40 g'      : True
```

**Q5 — "At what temperature and for how long is the 2kg country sourdough baked with the lid on?"**

> The 2kg country sourdough is baked at 250C for 20 minutes with the lid on
> `[recipe_id=ferment_001 | chunk_id=ferment_001__method__1 | source_file=ferment_001_country_sourdough_2kg.md]`.

```
chunk_id: ferment_001__method__1
  resolves in ChromaDB : True
  recipe_id matches    : True (claimed ferment_001, actual ferment_001)
  contains '250C'      : True
```

**Q7 — "In the vegan kimchi, what replaces the salted shrimp and the anchovy fish sauce?"**

> In the vegan kimchi, the salted shrimp and anchovy fish sauce are replaced by a
> shiitake and kombu umami base
> `[recipe_id=ferment_006 | chunk_id=ferment_006__ingredients__0 | source_file=ferment_006_vegan_napa_kimchi.md]`,
> specifically a shiitake and kombu stock plus tamari
> `[recipe_id=ferment_006 | chunk_id=ferment_006__method__1 | source_file=ferment_006_vegan_napa_kimchi.md]`.

```
chunk_id: ferment_006__ingredients__0   resolves: True   recipe_id matches: True   contains 'kombu': True
chunk_id: ferment_006__method__1        resolves: True   recipe_id matches: True   contains 'kombu': True
```

All 4 citations across the 3 answers resolved to real chunks containing the
claim. Zero fabricated chunk_ids.

### 5b. Three out-of-corpus questions, all refused

No card carries nutrition, vitamin or glycemic data. Note that **retrieval still
returns five chunks in every case** — refusal is enforced at the grounding step,
not by an empty result set.

**U1 — "How many calories per 100 g are in the 2kg country sourdough loaf?"**
Retrieved: `ferment_001__ingredients__1`, `ferment_001__allergen_note__0`,
`ferment_001__ingredients__0`, `ferment_001__method__2`, `ferment_001__method__1`

> I cannot answer this from the provided recipe documents.

**U2 — "What is the vitamin C content of the traditional napa cabbage kimchi?"**
Retrieved: `ferment_005__ingredients__1`, `ferment_006__ingredients__1`,
`ferment_005__ingredients__0`, `ferment_005__method__2`, `ferment_006__ingredients__0`

> I cannot answer this from the provided recipe documents.

**U3 — "What is the glycemic index of the sourdough focaccia?"**
Retrieved: `ferment_003__allergen_note__0`, `ferment_003__ingredients__1`,
`ferment_003__ingredients__0`, `ferment_003__method__1`, `ferment_003__method__0`

> I cannot answer this from the provided recipe documents.

**3/3 refused.** Full transcripts: `generation_transcripts.txt`.

---

## 6. The retrieval that embarrassed me

**Question (Q7):** *In the vegan kimchi, what replaces the salted shrimp and the
anchovy fish sauce?*

**Expected:** `ferment_006` (Vegan Napa Cabbage Kimchi), Method section.

**What the structure-aware chunker retrieved at rank 1:**

```
1. ferment_005__allergen_note__0   d=0.4650   recipe_id=ferment_005   <-- WRONG RECIPE
   "Recipe: Traditional Napa Cabbage Kimchi (recipe_id: ferment_005)
    Section: Allergen Note
    Contains fish in the form of anchovy fish sauce and contains crustacean
    shellfish in the form of salted fermented shrimp..."

2. ferment_006__ingredients__0     d=0.5439   recipe_id=ferment_006
3. ferment_006__allergen_note__0   d=0.5497   recipe_id=ferment_006
4. ferment_006__method__1          d=0.5675   recipe_id=ferment_006   <-- THE ACTUAL ANSWER
5. ferment_006__ingredients__1     d=0.5698   recipe_id=ferment_006
```

The chunk that literally answers the question — *"This shiitake and kombu stock
plus the tamari is what replaces the saeujeot and anchovy fish sauce"* — came
back at **rank 4**, behind a chunk from the recipe the question is explicitly
contrasting *against*. The baseline at 500/100 got this right at rank 1.

**Why it was wrong.** The most distinctive terms in the query are "salted shrimp"
and "anchovy fish sauce". Those exact phrases occur, densely and with nothing
else diluting them, in the *traditional* card's allergen note. The
structure-aware chunker had done its job correctly and produced a short, tight,
high-signal chunk — and that tightness is precisely what sank it. A 39-word
chunk consisting of almost nothing but the query's rare terms scores better than
a 145-word chunk that contains the answer plus surrounding method prose.

**Root cause:** not a chunk-boundary bug and not missing metadata. It is
**contrast blindness in the embedding, amplified by chunk tightness**. The
embedding has no representation of "in X, what replaces Y" — it cannot tell that
the mention of Y should steer *away* from the document about Y. Shorter chunks
concentrate signal, and when the concentrated signal belongs to the wrong
document, precision actively hurts. This is the direct cost of the same property
that makes the structure-aware chunker good at Q1–Q4.

**Mitigating note:** generation still produced the correct, correctly cited
answer for Q7, because `ferment_006__method__1` was inside the top 5 and the
model selected it over the higher-ranked distractor. Rank-1 failure did not
become an answer failure here — but it would have at `top_k=1` or `top_k=3`.

---

## 7. Realistic condition — the 6 new cards among distractors

*(Populated by `evaluate_chunkers.py realistic`; see section 3 for why this
condition exists. Numbers appear in `search_dump_realistic.txt` and
`eval_results_realistic.json`.)*

STATUS_PLACEHOLDER

---

## 8. Bonus challenge

The bonus asks for a question where the structure-aware chunker **wins on
retrieval but loses on the final answer**, because the tight ingredient-row chunk
retrieves precisely and then gives the model no method prose.

**Demonstrated.** Question: *"How much fine sea salt does the 2kg country
sourdough use, and at what stage of mixing is it added?"*

A complete answer needs both halves of the card, and they live in different
places: the quantity `40 g` exists **only** in the ingredient table, and the
stage (`autolyse`) exists **only** in the method prose. At `top_k=1` the two
chunkers diverge completely.

| | Strategy A (baseline 500/100) | Strategy B (structure-aware) |
|---|---|---|
| Retrieved at top_k=1 | `ferment_001_country_sourdough_2kg.md_1_0` (flat window) | `ferment_001__method__0` (Method section) |
| Chunk contains "40 g" | yes | **no** |
| Chunk contains "autolyse" | yes | yes |
| Answer | "uses 40 g (2%) of fine sea salt... added after the autolyse stage and after the levain has been mixed in, together with a reserved 100 g of water" | **"I cannot answer this from the provided recipe documents."** |
| Answer complete | quantity yes, stage yes | quantity **no**, stage **no** |

Strategy B retrieved the *semantically better* single chunk — the Method section
genuinely is where "at what stage" is answered, and it is a clean, correctly
scoped, correctly labelled chunk. But it is 145 words of prose with no table in
it, so the gram weight simply is not there, and the grounding rules did exactly
what they should: refused rather than inventing a number. Strategy A won purely
because its 500-word window is blunt enough to have swallowed the table and the
method together.

At `top_k=5` the tension disappears — both strategies assemble complete context
and both answer correctly, Strategy B citing two chunks
(`ferment_001__ingredients__0` for the weight, `ferment_001__method__0` for the
stage) where Strategy A cites one blob twice.

**The tension in two sentences:** precise chunking retrieves the *right* passage
but a narrower one, so a question spanning two sections can retrieve perfectly
and still arrive with half the facts missing. Blunt chunking answers such
questions by accident rather than by design, which works until the document is
long enough that the accident stops happening — so the fix is not blunter chunks
but a `top_k` wide enough to reassemble the sections that precise chunking
correctly separated.

Full transcript: `bonus_demo.txt`.

---

## 9. Which chunker ships, and why

**Shipping the structure-aware chunker (Strategy B), with `top_k` kept at 5.**

The honest summary is that on the 6-card index the two strategies tie at 8/8 and
the structure-aware one is marginally worse at rank 1 (7/8 vs 8/8), so this
decision is *not* being made on the headline hit-rate — that number is at a
ceiling and proves nothing either way. It is being made on three things the
measurement did show:

1. **The baseline's correctness here is an accident of card length.** At its
   actual configured setting of 500 words, a whole fermentation card fits in one
   or two chunks, so the ingredient table survives by luck rather than by design.
   The moment the same chunker is run at a granularity appropriate to a real
   corpus (120/20), it produced **3 chunks containing orphaned ingredient rows**
   — rows with no table header and no recipe title anywhere in the chunk:

   ```
   ferment_004_..._1_1   "| | Fine sea salt | 10 g | 2% | Total dough weight..."
   ferment_005_..._1_1   "| 1% | | Saeujeot (salted fermented shrimp) | 60 g | 3% | ..."
   ferment_006_..._1_1   "| Ginger, grated | 20 g | 1% | | Dried shiitake mushrooms | 30 g..."
   ```

   A chunk that reads `| Fine sea salt | 10 g | 2% |` with no title cannot tell
   anyone *which* recipe uses 10 g of salt, and 10 g is the brioche figure, not
   the 40 g sourdough figure.

   Counted programmatically over the 6 cards (a chunk "orphans" a row if it
   contains an ingredient row but not the table header AND the recipe title):

   | Chunker | Chunks containing table rows | Orphaned |
   |---|---|---|
   | baseline 120/20 | 9 | **3 (33%)** |
   | structure-aware | 6 | **0** | That is exactly the "7g fine sea salt detached
   from its recipe" failure the brief describes, and the baseline produces it as
   soon as it is asked to make chunks of a normal size.

2. **The baseline destroys table structure even when it keeps the rows.** Its
   `" ".join(text.split())` flattens all newlines, so a markdown table becomes a
   single run-on line. The rows survive, the columns do not.

3. **Cited chunks are more useful.** Structure-aware chunk_ids
   (`ferment_001__ingredients__0`) name the recipe and the section, so a citation
   is auditable at a glance; baseline ids
   (`ferment_001_country_sourdough_2kg.md_1_0`) identify a window offset only.

The cost is real and is documented in section 6: tighter chunks lose contrastive
questions like Q7 at rank 1. I am accepting that cost because it degrades a rank,
not an answer, at `top_k=5`, whereas an orphaned ingredient row degrades a gram
weight in someone's dough.

---

## 10. Code changes

Full diff in **`code_diff.txt`**. Summary:

**Unchanged, deliberately** — verified byte-identical to the original app:

- `rag/chunker.py` — Strategy A must be the existing chunker, not a rewrite
- `rag/pdf_loader.py`, `app.py` — the legal-contract path still works

**Modified:**

| File | Change | Why |
|---|---|---|
| `rag/vector_store.py` | added `add_chunks()` (append/upsert, never deletes), `attach_metadata()`, `validate_chunk_metadata()`, `diet_flags()`, `list_collections()`; `retrieve()` gained an optional `where=` argument | `build_index()` wiped collections, violating requirement 6; metadata and filtering did not exist |
| `rag/generator.py` | added `generate_recipe_answer()`, `build_recipe_context()`, `REFUSAL_TEXT` | recipe-domain grounding with forced refusal and chunk_id citations; the legal `generate_answer()` is untouched |
| `rag/embeddings.py` | added throttle + 429 backoff | free-tier rate limit hit during bulk ingest. **The model is unchanged** (`gemini-embedding-001`) so the chunker comparison still moves exactly one variable |

**New:**

| File | Purpose |
|---|---|
| `rag/recipe_loader.py` | loads markdown cards; emits the same `{text, page, source}` shape as `pdf_loader` so the unmodified baseline chunker can consume them |
| `rag/structure_aware_chunker.py` | Strategy B |
| `fermentation_cards/*.md` | the 6 new cards |
| `eval_questions.json` | the 8 questions, authored before retrieval |
| `ingest_fermentation_cards.py` | 6-card-only ingest into 3 collections |
| `evaluate_chunkers.py` | search-only Hit@5, per-question, both conditions |
| `filter_demo.py` | dietary_tags filter demonstration |
| `generate_answers.py` | 3 grounded answers + 3 refusals + citation verification |
| `add_distractor_corpus.py` | builds the realistic condition |

### Structure-aware chunker design

Chunk size is driven by document structure, not by a constant:

1. **A table is never split.** Header row, separator and all data rows travel
   together in one chunk.
2. **Every chunk carries its recipe title, recipe_id and section heading** in a
   context header, so an ingredient row can never be orphaned.
3. **Short prose immediately preceding a table stays with that table**, because
   on these cards that sentence declares the percentage basis ("percentages are
   calculated against the 2000 g of flour"). Without it, "2%" is meaningless.

Resulting chunk for the salt row:

```
Recipe: 2kg Country Sourdough Loaf (recipe_id: ferment_001)
Section: Ingredients

Total flour weight is 2000 g and all baker's percentages below are calculated
against that total flour weight. Final hydration is 80%.

| Ingredient | Weight | Baker's % |
|---|---|---|
| Strong white bread flour | 1700 g | 85% |
| Wholemeal flour | 300 g | 15% |
| Water (28C) | 1600 g | 80% |
| Fine sea salt | 40 g | 2% |
| Rye levain (100% hydration) | 400 g | 20% |
```

---

## 11. Artefacts

| File | Contents |
|---|---|
| `search_dump_cards6.txt` | full search-only dump, 8 questions x 3 strategies, 6-card index |
| `search_dump_realistic.txt` | same 8 questions, distractor index |
| `eval_results.json`, `eval_results_realistic.json` | machine-readable results |
| `filter_demo.txt` | unfiltered vs filtered lists with scores |
| `generation_transcripts.txt` | 3 cited answers + 3 refusals verbatim |
| `generation_report.json` | citation verification results |
| `ingest_report.json` | before/after collection counts |
| `bonus_demo.txt`, `bonus_report.json` | bonus: precision vs completeness at top_k=1 and 5 |
| `code_diff.txt` | full diff vs the original app |
