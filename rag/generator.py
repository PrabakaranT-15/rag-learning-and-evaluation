from groq import Groq, RateLimitError, InternalServerError, BadRequestError
from dotenv import load_dotenv
import os
import time

load_dotenv()

# Week 7: generation backend switched from Gemini (google.genai) to Groq.
# Embeddings (rag/embeddings.py) stay on Gemini - every chunk already in
# Chroma was embedded with gemini-embedding-001, and mixing embedding spaces
# would break retrieval - only generation (this module) moved. Groq's own
# free tier is generous enough that this also sidesteps the gemini-3.6-flash
# 20-calls/day cap and the 503 "high demand" errors that made the app.py
# agent demo hit its own time budget.
API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=API_KEY)


# ---------------------------------------------------------------- call pacing
# Pacing + retry, same shape as the old Gemini wrapper - just against Groq's
# rate limits (RateLimitError) and transient server errors
# (InternalServerError) instead of Gemini's RESOURCE_EXHAUSTED/503 strings.

MIN_INTERVAL_SECONDS = 1.0
MAX_RETRIES = 8

# Week 8: a much larger, dedicated budget for the tool-call-misfire retry
# in _generate() below - see that except BadRequestError block for why a
# larger number than MAX_RETRIES is needed specifically for it.
MAX_TOOL_MISFIRE_RETRIES = 25

# Primary model for live traffic (app.py's chat and the agent planner).
DEFAULT_MODEL = "openai/gpt-oss-120b"

# Lighter/faster model for batch-heavy scripts (week4/5/6 evaluation runs,
# race_agent_vs_workflow.py) that fire many more calls than a single chat
# turn - kept as a separate constant so those scripts can ask for it
# explicitly, the same pattern used when this was gemma-4-31b-it under
# Gemini's quota.
WEEK4_MODEL = "openai/gpt-oss-20b"

_last_call = [0.0]

# Week 7 addition: every real call to the model is logged here (model name +
# prompt length only - never the API key or full prompt text). This is the
# cost/call-count signal race_agent_vs_workflow.py reads to compare the
# agent against the fixed workflow - counting real billable requests, not
# wall-clock alone, since wall-clock is dominated by MIN_INTERVAL_SECONDS
# pacing rather than actual work.
call_log = []


def _generate(prompt, model=DEFAULT_MODEL):
    """Throttled + retried wrapper around Groq's chat.completions.create.

    reasoning_format="hidden" asks Groq's gpt-oss models to strip their
    internal reasoning trace out of the response and return only the final
    answer in `content`. That's necessary but not sufficient: live testing
    (see rag/agent.py's planner) showed these models sometimes finish
    generation (finish_reason="stop") having spent their whole turn on
    hidden reasoning and emitted no visible content at all - a real,
    reproducible sampling quirk, not a truncation or rate-limit error. An
    empty `content` is therefore treated as retryable here, same as a rate
    limit: resample rather than hand a caller (e.g. the agent's JSON action
    parser) an empty string it can't do anything with.
    """

    # Independent per-error-type counters, not a single shared attempt
    # index - each failure mode gets its own give-up threshold, so a run of
    # tool-call misfires (cheap, no backoff needed) doesn't eat into the
    # budget a real rate limit needs, or vice versa.
    rate_limit_attempts = 0
    empty_attempts = 0
    misfire_attempts = 0

    while True:

        wait = MIN_INTERVAL_SECONDS - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                reasoning_format="hidden",
            )
            _last_call[0] = time.time()
            call_log.append({"model": model, "prompt_chars": len(prompt)})

            content = response.choices[0].message.content
            if content:
                return content

            empty_attempts += 1
            if empty_attempts >= MAX_RETRIES:
                return content

            print("  [empty response, resampling]", flush=True)
            continue

        except (RateLimitError, InternalServerError) as error:

            _last_call[0] = time.time()
            rate_limit_attempts += 1

            if rate_limit_attempts >= MAX_RETRIES:
                raise

            backoff = min(60, 10 + 10 * rate_limit_attempts)
            print(f"  [rate limited/unavailable, backing off {backoff}s]", flush=True)
            time.sleep(backoff)

        except BadRequestError as error:

            # Week 8: the gpt-oss models occasionally misfire into Groq's
            # native tool-calling path ("Tool choice is none, but model
            # called a tool") even though this app never registers any
            # tools on the API call - it's asking for JSON in the prompt
            # text only. Observed live while running week8's batch agent
            # trajectories: a transient sampling quirk, same spirit as the
            # empty-content retry above, not a real malformed request from
            # this app - so it's resampled, not raised, but ONLY for this
            # specific error code, and with a MUCH larger budget than
            # MAX_RETRIES: each retry is immediate (no backoff), and this
            # particular quirk was observed misfiring several times in a
            # row within a single planning step - a small budget shared
            # with rate-limit/empty-content handling exhausted for real and
            # killed an otherwise-healthy run. Any other 400 (a genuine bad
            # request) still raises immediately rather than being silently
            # retried.
            _last_call[0] = time.time()

            if "tool_use_failed" not in str(error):
                raise

            misfire_attempts += 1
            if misfire_attempts >= MAX_TOOL_MISFIRE_RETRIES:
                raise

            print("  [model misfired into native tool-calling, resampling]", flush=True)


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


# Week 5 found (week5_error_analysis.md, trace T20) that rule 7 below was
# suppressing legitimate reasoning, not just fabrication: the model refused
# "why does my starter smell like nail polish?" even though the rank-1 chunk
# stated "a faint acetone note" - the same thing in different words - because
# rule 7 read as a blanket ban on any reasoning, not just reasoning that pulls
# in facts from outside the context. RECIPE_PROMPT_BASELINE is kept verbatim
# (unused by any current caller) purely so Week 6's before/after evaluation
# can regenerate genuinely "before the fix" answers against the SAME model
# and retrieval, isolating the prompt as the only variable - see
# week6_run_evals.py.
RECIPE_PROMPT_BASELINE = """You are a recipe assistant working from fermentation recipe cards.

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

RECIPE_PROMPT_FIXED = """You are a recipe assistant working from fermentation recipe cards.

Answer ONLY using the retrieved context below.

HARD RULES:

1. Every factual claim must be supported by the context.
2. Do not use outside knowledge of any kind.
3. Do not invent ingredient quantities.
4. Do not invent baker's percentages or hydration figures.
5. Do not invent nutrition values, calories, vitamins or glycemic index.
6. Do not invent temperatures or timings.
7. Do not estimate, approximate, or reason from general cooking knowledge that
   is not stated in the context. You MAY, however, recognize when a lay
   description in the question (a smell, a taste, a texture, a common name)
   refers to the same thing as a technical term that IS stated in the
   context, and answer using that stated fact - this is using the context,
   not outside knowledge.
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

# The prompt every current caller (app.py, generate_answers.py, bonus_demo.py,
# week4/5 scripts) gets when they don't ask for a specific prompt_version.
RECIPE_PROMPT = RECIPE_PROMPT_FIXED


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


def generate_recipe_answer(question, results, model=DEFAULT_MODEL, prompt_version="fixed"):
    """Strictly grounded answer over recipe chunks, with forced refusal.

    `model` defaults to the Week 3 model so existing callers are unaffected.
    The Week 4 harness passes WEEK4_MODEL, identically for both runs.

    `prompt_version` selects RECIPE_PROMPT_FIXED (default - what every real
    caller gets) or RECIPE_PROMPT_BASELINE (Week 6's before/after evaluation
    only, to regenerate pre-fix answers under the current model for a fair
    comparison).
    """

    context = build_recipe_context(results)

    template = RECIPE_PROMPT_BASELINE if prompt_version == "baseline" else RECIPE_PROMPT_FIXED

    prompt = template.format(
        refusal=REFUSAL_TEXT,
        context=context,
        question=question,
    )

    return _generate(prompt, model)
