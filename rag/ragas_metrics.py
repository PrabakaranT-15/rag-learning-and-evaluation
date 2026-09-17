"""Week 6: RAGAS-style metrics (faithfulness, answer relevancy, context
precision, context recall) - reimplemented directly on this project's own
Gemini client, NOT the `ragas` pip package.

WHY NOT THE `ragas` PACKAGE
---------------------------
`ragas` was installed and tried first. It pulls in `langchain-google-genai`
to talk to Gemini, which pulls in the full `langchain` 1.x ecosystem. But
`ragas` 0.4.3's own code has two hardcoded legacy imports
(`langchain_community.chat_models.vertexai`, then `langchain_openai`'s chat
models) that are incompatible with each other's required `langchain-core`
version - no combination of langchain-core/langchain-community pins made
both imports succeed at once in this environment, and every pin attempt
cascaded into new, unrelated breakage (at one point pulling in the entire
Google Cloud Vertex AI / BigQuery / Cloud Storage SDKs just to satisfy one
import statement). That dependency chain was fully uninstalled and the
project's original package versions restored - `pytest tests/` and the app's
own imports were re-verified clean afterwards.

This module implements the same four metrics, using the same methodology
RAGAS itself uses (LLM-based claim decomposition + verification for
faithfulness/context recall, LLM-generated reverse questions + embedding
similarity for answer relevancy, LLM relevance judgements + rank-weighted
precision for context precision) - just calling this project's own
`_generate()` (rag/generator.py) and `create_embedding()` (rag/embeddings.py)
instead of going through langchain.

ONE DELIBERATE DEVIATION from RAGAS's default: RAGAS scores a "noncommittal"
(evasive/refusing) answer as 0 relevancy by default, on the assumption a
refusal is always a dodge. In this app, REFUSAL_TEXT is frequently the
objectively CORRECT answer for a genuinely out-of-corpus question (see
week5_error_analysis.md - T07/T15/T16/T19). Scoring those refusals as
"irrelevant" would make correct behaviour look like a defect, so
answer_relevancy returns None (not 0) for an exact refusal, with a note
explaining why, instead of silently following RAGAS's default.

Every metric returns None (never a guessed number) when the model's output
can't be parsed or when the input makes the metric undefined (e.g. an answer
with zero factual claims), so a caller can report "unscoreable" instead of
quietly averaging in a wrong number.
"""

import json
import math
import re

from rag.generator import _generate, WEEK4_MODEL, REFUSAL_TEXT
from rag.embeddings import create_embedding


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _ask_json(prompt, model):
    """Call the LLM and parse its response as JSON, tolerating markdown code
    fences. Returns None (never raises) on unparseable output."""

    raw = _generate(prompt, model)
    cleaned = _FENCE_RE.sub("", raw).strip()

    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return None


def _cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def extract_claims(answer, model=WEEK4_MODEL):
    """Decompose an answer into individual, checkable factual claims - the
    same claim-decomposition step RAGAS's faithfulness/context-recall
    metrics are built on."""

    prompt = f"""Break the following answer into a list of individual, standalone \
factual claims. Each claim should be a single fact that could be checked \
independently. If the answer makes no factual claims at all (for example, a \
refusal to answer), return an empty list.

ANSWER:
{answer}

Respond with ONLY a JSON array of strings, nothing else. Example:
["Claim one.", "Claim two."]
"""

    result = _ask_json(prompt, model)
    return result if isinstance(result, list) else None


def faithfulness(question, contexts, answer, model=WEEK4_MODEL):
    """Fraction of the answer's claims that are actually supported by the
    retrieved context. None if the answer has no factual claims to check
    (e.g. a refusal) - that isn't a faithfulness score of 0 or 1, it's
    undefined, same as RAGAS reports NaN in this case."""

    claims = extract_claims(answer, model)
    if not claims:
        return {"score": None, "claims": [], "note": "no factual claims to check"}

    context_text = "\n\n".join(contexts)
    claims_json = json.dumps(claims)

    prompt = f"""Given the retrieved context below, decide for EACH claim in the list \
whether it is directly supported by the context (true) or not (false).

RETRIEVED CONTEXT:
{context_text}

CLAIMS (JSON array):
{claims_json}

Respond with ONLY a JSON array of booleans, the same length and order as the \
claims array. Example: [true, false, true]
"""

    verdicts = _ask_json(prompt, model)
    if not isinstance(verdicts, list) or len(verdicts) != len(claims):
        return {"score": None, "claims": claims, "note": "judge output did not parse"}

    supported = sum(1 for v in verdicts if v is True)
    return {
        "score": supported / len(claims),
        "claims": list(zip(claims, verdicts)),
        "note": None,
    }


def answer_relevancy(question, answer, model=WEEK4_MODEL, n_questions=3):
    """How well the answer addresses the original question: generate
    `n_questions` questions that the answer would be a good answer to, embed
    each alongside the real question, and average their cosine similarity.
    A relevant answer's reverse-engineered questions should closely resemble
    the question actually asked."""

    if answer.strip() == REFUSAL_TEXT:
        return {"score": None, "note": "refusal - relevancy not meaningful (see module docstring)"}

    prompt = f"""Given the answer below, write {n_questions} different questions that this \
answer would be a good, direct answer to. Do not use any outside knowledge -
base the questions only on what the answer actually says.

ANSWER:
{answer}

Respond with ONLY a JSON array of {n_questions} question strings.
"""

    generated_questions = _ask_json(prompt, model)
    if not isinstance(generated_questions, list) or not generated_questions:
        return {"score": None, "note": "judge output did not parse"}

    question_embedding = create_embedding(question)
    similarities = [
        _cosine_similarity(question_embedding, create_embedding(gq))
        for gq in generated_questions
    ]

    return {
        "score": sum(similarities) / len(similarities),
        "generated_questions": generated_questions,
        "similarities": similarities,
        "note": None,
    }


def context_precision(question, contexts, answer, model=WEEK4_MODEL):
    """Reference-free context precision (RAGAS's "without reference"
    variant): are the USEFUL chunks concentrated near the top of the
    ranking, or buried under irrelevant ones? Judged against the answer
    actually produced, since no ground-truth reference is available for
    most of this project's question sets.

    Rank-weighted, matching RAGAS's definition:
        score = sum(precision@k * relevant_k) / (count of relevant chunks)
    where precision@k = (# relevant chunks in the first k) / k. A chunk
    marked relevant at rank 1 contributes more than one marked relevant at
    rank 5, rewarding good chunks appearing EARLY, not just present.
    """

    if not contexts:
        return {"score": None, "note": "no retrieved context to judge"}

    numbered_context = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, start=1))

    prompt = f"""A question was asked and answered using the numbered context chunks \
below, in rank order (chunk [1] was retrieved first). For EACH chunk, decide \
whether it was actually useful for producing the given answer.

QUESTION:
{question}

RETRIEVED CONTEXT (rank order):
{numbered_context}

ANSWER PRODUCED:
{answer}

Respond with ONLY a JSON array of booleans, one per chunk, in the same \
[1]..[{len(contexts)}] order. Example: [true, false, false]
"""

    verdicts = _ask_json(prompt, model)
    if not isinstance(verdicts, list) or len(verdicts) != len(contexts):
        return {"score": None, "note": "judge output did not parse"}

    relevant_count = sum(1 for v in verdicts if v is True)
    if relevant_count == 0:
        return {"score": 0.0, "verdicts": verdicts, "note": None}

    weighted_sum = 0.0
    hits_so_far = 0
    for k, is_relevant in enumerate(verdicts, start=1):
        if is_relevant:
            hits_so_far += 1
            weighted_sum += hits_so_far / k

    return {"score": weighted_sum / relevant_count, "verdicts": verdicts, "note": None}


def context_recall(question, contexts, reference_answer, model=WEEK4_MODEL):
    """Fraction of the REFERENCE answer's claims that can be attributed to
    the retrieved context - i.e. did retrieval bring back everything needed
    to reconstruct the known-correct answer, not just something plausible.

    Requires a reference/expected answer (eval_questions.json's
    `expected_answer` field); returns None where none exists, rather than
    substituting the model's own answer as a fake reference, which would
    make retrieval look perfect by definition.
    """

    if not reference_answer:
        return {"score": None, "note": "no reference answer available for this question"}

    claims = extract_claims(reference_answer, model)
    if not claims:
        return {"score": None, "claims": [], "note": "reference answer has no factual claims"}

    context_text = "\n\n".join(contexts)
    claims_json = json.dumps(claims)

    prompt = f"""Given the retrieved context below, decide for EACH claim in the list \
whether it can be attributed to (found in) the context.

RETRIEVED CONTEXT:
{context_text}

CLAIMS (JSON array, from a known-correct reference answer):
{claims_json}

Respond with ONLY a JSON array of booleans, the same length and order as the \
claims array.
"""

    verdicts = _ask_json(prompt, model)
    if not isinstance(verdicts, list) or len(verdicts) != len(claims):
        return {"score": None, "claims": claims, "note": "judge output did not parse"}

    attributable = sum(1 for v in verdicts if v is True)
    return {
        "score": attributable / len(claims),
        "claims": list(zip(claims, verdicts)),
        "note": None,
    }
