"""Week 6: LLM-as-judge for answers a simple token/rule check can't grade.

rag/evaluation.py's classify() (Week 4) can only tell you whether an expected
recipe_id was retrieved and whether an expected literal token string appears
in the answer. It cannot tell you whether an answer that says something
DIFFERENT from those tokens is still correct - week5_error_analysis.md's four
named problem groups (over-refusal, comparison omission, answering a nearby
question, silent scope reinterpretation) all required a human to actually
read the answer. This module automates that reading so it can be applied to
answers nobody has manually graded yet.

The judge is binary (PASS/FAIL), not 1-10: every failure mode Week 5 found is
a yes/no question - "did this answer correctly and completely address what
was asked, using only the given context" - not a matter of degree. A 1-10
scale would add judge-to-judge noise at the boundary without measuring
anything a binary call doesn't already capture.

Per the Week 6 brief, this judge must not be trusted until it is checked
against human grading - see week6_run_evals.py, which runs it over the 20
already hand-graded week5 traces and reports agreement before using it to
score anything new.
"""

import re

from rag.generator import _generate, WEEK4_MODEL


JUDGE_PROMPT = """You are grading whether an AI assistant's answer about a fermentation \
recipe correctly and completely addresses the user's question, using ONLY the \
retrieved context it was given.

Grade PASS only if ALL of the following hold:
- Every claim in the answer is actually supported by the retrieved context.
- The answer addresses what was literally asked, not a different but nearby question.
- If the question can be answered by connecting a lay description (a smell, a taste, a \
common name) to a technical term that IS stated in the retrieved context, the answer \
makes that connection instead of refusing.
- The answer does NOT refuse when the retrieved context actually contains the answer.
- The answer DOES refuse (or clearly hedges) when the retrieved context does not \
contain the answer - do not reward guessing.
- If the question asks for a comparison or summary, the answer includes the single most \
relevant distinguishing fact, not just secondary correct facts.

Grade FAIL otherwise.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

ANSWER TO GRADE:
{answer}

Respond in EXACTLY this format, nothing else:
VERDICT: PASS or FAIL
REASON: one sentence
"""


_VERDICT_RE = re.compile(r"VERDICT:\s*(PASS|FAIL)", re.IGNORECASE)
_REASON_RE = re.compile(r"REASON:\s*(.+)", re.IGNORECASE)


def judge_answer(question, context, answer, model=WEEK4_MODEL):
    """Ask the LLM judge to grade one (question, context, answer) triple.

    Returns {"verdict": "pass"|"fail"|None, "reason": str, "raw": str}.
    verdict is None (never guessed) when the model doesn't follow the output
    format, so a caller scoring a batch can report unparsed judge output
    instead of silently miscounting it as a pass or a fail.
    """

    prompt = JUDGE_PROMPT.format(question=question, context=context, answer=answer)
    raw = _generate(prompt, model)

    verdict_match = _VERDICT_RE.search(raw)
    reason_match = _REASON_RE.search(raw)

    return {
        "verdict": verdict_match.group(1).lower() if verdict_match else None,
        "reason": reason_match.group(1).strip() if reason_match else raw.strip(),
        "raw": raw,
    }
