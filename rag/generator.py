from google import genai
from dotenv import load_dotenv
import os
import time

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=API_KEY)


# ---------------------------------------------------------------- call pacing
# Week 4 addition. Call pacing ONLY - the model, the prompts and the returned
# text are untouched. The free tier allows 5 generate_content requests per
# minute, and a 20-question evaluation run hits that immediately, so requests
# are spaced and 429s are retried instead of aborting the run.

MIN_INTERVAL_SECONDS = 2.0
MAX_RETRIES = 8

# Week 3 default. Unchanged, so every Week 3 script behaves exactly as before.
DEFAULT_MODEL = "gemini-3.6-flash"

# Week 4 experiment model.
#
# Every gemini-*-flash model on this free tier is capped at 20 generate_content
# requests PER DAY (quotaId GenerateRequestsPerDayPerProjectPerModel-FreeTier,
# quotaValue 20), which a 20-question before/after experiment exhausts before it
# can finish even once - it needs 20 baseline answers plus one per question
# whose retrieved context changed. This model was measured to have a daily cap
# well above that.
#
# It is applied to BOTH the baseline run and the improved run, so the generator
# is still held constant across the comparison, which is what the experiment
# requires. The prompt is unchanged, and retrieval - the thing actually being
# measured - does not involve this model at all.
WEEK4_MODEL = "gemma-4-31b-it"

_last_call = [0.0]


def _generate(prompt, model=DEFAULT_MODEL):
    """Throttled + retried wrapper around client.models.generate_content."""

    for attempt in range(MAX_RETRIES):

        wait = MIN_INTERVAL_SECONDS - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)

        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )
            _last_call[0] = time.time()
            return response.text

        except Exception as error:

            _last_call[0] = time.time()

            if "RESOURCE_EXHAUSTED" not in str(error) and "429" not in str(error):
                raise

            if attempt == MAX_RETRIES - 1:
                raise

            backoff = min(60, 20 + 10 * attempt)
            print(f"  [rate limited, backing off {backoff}s]", flush=True)
            time.sleep(backoff)


def generate_answer(question, results):
    """Generate an answer using retrieved context and the Gemini model."""

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context_parts = []

    for document, metadata in zip(documents, metadatas):

        context_parts.append(
            f"""
SOURCE: {metadata['source']}
PAGE: {metadata['page']}

{document}
"""
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a legal contract document assistant.

Answer the user's question ONLY using the
provided amendment documents.

Rules:

1. Do not use outside knowledge.
2. Do not invent contract terms.
3. Do not make assumptions.
4. If the answer cannot be found in the
   provided context, say:

"I could not find this information in the
provided amendment documents."

5. Give a concise answer.
6. Mention the relevant source and page.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}
"""

    return _generate(prompt)


# ------------------------------------------------------- Task Set B additions

REFUSAL_TEXT = "I cannot answer this from the provided recipe documents."


RECIPE_PROMPT = """You are a recipe assistant working from fermentation recipe cards.

Answer ONLY using the retrieved context below.

HARD RULES:

1. Every factual claim must be supported by the context.
2. Do not use outside knowledge of any kind.
3. Do not invent ingredient quantities.
4. Do not invent baker's percentages or hydration figures.
5. Do not invent nutrition values, calories, vitamins or glycemic index.
6. Do not invent temperatures or timings.
7. Do not estimate, approximate or reason from general cooking knowledge.
8. If the context does not contain the answer, reply with EXACTLY this sentence
   and nothing else:

{refusal}

9. If the answer IS in the context, cite it. After each factual claim add a
   citation in this exact form:

   [recipe_id=<recipe_id> | chunk_id=<chunk_id> | source_file=<source_file>]

   Use only the recipe_id, chunk_id and source_file values given in the context
   blocks. Never invent a chunk_id.

RETRIEVED CONTEXT:

{context}

USER QUESTION:

{question}
"""


def build_recipe_context(results):
    """Render retrieved chunks into labelled context blocks with their real ids."""

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    blocks = []

    for chunk_id, document, metadata in zip(ids, documents, metadatas):
        blocks.append(
            "-------- CONTEXT BLOCK --------\n"
            f"chunk_id: {chunk_id}\n"
            f"recipe_id: {metadata.get('recipe_id')}\n"
            f"recipe_name: {metadata.get('recipe_name')}\n"
            f"source_file: {metadata.get('source_file')}\n"
            f"section: {metadata.get('section') or '(flat window)'}\n"
            f"dietary_tags: {metadata.get('dietary_tags')}\n\n"
            f"{document}"
        )

    return "\n\n".join(blocks)


def generate_recipe_answer(question, results, model=DEFAULT_MODEL):
    """Strictly grounded answer over recipe chunks, with forced refusal.

    `model` defaults to the Week 3 model so existing callers are unaffected.
    The Week 4 harness passes WEEK4_MODEL, identically for both runs.
    """

    context = build_recipe_context(results)

    prompt = RECIPE_PROMPT.format(
        refusal=REFUSAL_TEXT,
        context=context,
        question=question,
    )

    return _generate(prompt, model)
