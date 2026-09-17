"""Week 6: one-command eval harness — regression tests + judge validation +
before/after score per problem type, for the RECIPE_PROMPT fix predicted at
the end of week5_error_analysis.md.

WHAT THIS RUNS
--------------
1. Regenerates all 20 week5 traces under RECIPE_PROMPT_FIXED, holding the
   model (WEEK4_MODEL) AND the retrieved context IDENTICAL to the week5
   run (reusing week5_traces.json's cached retrieval instead of re-querying
   Chroma), so the prompt text is the only variable that changed. The week5
   "answer" field already IS the real, disclosed baseline (RECIPE_PROMPT
   before the fix) — it is not resimulated.
2. Hard regression checks (free, rule-based) on the 6 traces
   week5_error_analysis.md specifically named: T20 must flip from refusal to
   a grounded answer (the target failure), T07/T15/T16/T19 (genuine
   out-of-corpus refusals) must NOT start hallucinating because of the fix.
   T09 is checked and reported but treated as a soft/"ideally" case per the
   prediction, not a hard pass/fail gate.
3. LLM-judge validation: the judge (rag/judge.py) grades all 20 BASELINE
   answers and its verdicts are compared against HUMAN_VERDICT below, which
   is a direct transcription of week5_error_analysis.md's open-coding table
   — not a re-grade. Only if this agreement is reasonable should the judge's
   verdicts on the FIXED answers be trusted.
4. Before/after score per problem type (from week5_error_analysis.md's
   "Named problem groups" section), two ways:
     (a) human-graded before vs. judge-graded after — the real-world number.
     (b) judge-graded before vs. judge-graded after — the same scorer used
         both times, which isolates the prompt change from any human/judge
         disagreement.
5. A citation-verification safety net on the 6 Week 3/4 questions
   (Q1/Q5/Q7 answerable, U1/U2/U3 unanswerable) that already had verified
   citations before this change — confirms the fix didn't break them.
6. RAGAS-style metrics (rag/ragas_metrics.py — a from-scratch reimplementation
   on this project's own Gemini client, NOT the `ragas` pip package; see that
   module's docstring for why) computed on the 20 fixed answers: faithfulness,
   answer relevancy, and context precision. context_recall additionally runs
   on Q1/Q5/Q7 from the Week 3/4 set, the only questions with a known
   `expected_answer` reference to check retrieval completeness against.

Run with: python week6_run_evals.py
"""

import json

from rag.vector_store import get_collection, retrieve
from rag.generator import generate_recipe_answer, REFUSAL_TEXT, WEEK4_MODEL
from rag.judge import judge_answer
from rag.ragas_metrics import faithfulness, answer_relevancy, context_precision, context_recall
from generate_answers import verify_citations, VERIFY_TOKEN


TRACES_COLLECTION = "fermentation_structure_aware"

# Transcribed directly from week5_error_analysis.md's open-coding table and
# "Named problem groups" section — this is the human's own grading, not a
# new judgement. "none" = no named problem (a correct answer, including
# correct refusals).
HUMAN_VERDICT = {
    "T01": "pass", "T02": "pass", "T03": "pass", "T04": "pass", "T05": "pass",
    "T06": "pass", "T07": "pass", "T08": "fail", "T09": "fail", "T10": "fail",
    "T11": "pass", "T12": "fail", "T13": "pass", "T14": "pass", "T15": "pass",
    "T16": "pass", "T17": "fail", "T18": "pass", "T19": "pass", "T20": "fail",
}

PROBLEM_GROUP = {
    "T20": "1_over_refusal", "T09": "1_over_refusal",
    "T10": "2_comparison_omission",
    "T17": "3_answers_nearby_question",
    "T08": "4_silent_scope_reinterpretation", "T12": "4_silent_scope_reinterpretation",
}

# The hard regression gate: does the fix flip the target failure without
# breaking the genuine refusals. T09 is intentionally absent — the week5
# prediction only calls it an "ideally", not a requirement.
HARD_REGRESSION = {
    "T20": {"expect": "flips", "must_contain_any": ["acetone"]},
    "T07": {"expect": "stays_refused"},
    "T15": {"expect": "stays_refused"},
    "T16": {"expect": "stays_refused"},
    "T19": {"expect": "stays_refused"},
}
SOFT_REGRESSION = {"T09": {"must_contain_any": ["gochugaru", "5%"]}}


def context_from_trace(trace):
    """Reconstruct the context text the judge should read — the same
    retrieved chunk text the generator actually saw, cached in the trace."""

    return "\n\n".join(row["text"] for row in trace["retrieved"])


def results_from_trace(collection, trace):
    """Rebuild a Chroma-query-shaped results dict from a cached trace's
    retrieved chunk_ids, by fetching current metadata from Chroma.

    collection.get(ids=...) does NOT preserve input order, so results are
    re-sorted back into the trace's original rank order before use — this is
    what keeps retrieval byte-for-byte identical to the week5 run while still
    getting the recipe_name/source_file/section/dietary_tags metadata
    build_recipe_context() needs (the cached trace only stored rank/chunk_id/
    recipe_id/text).
    """

    chunk_ids = [row["chunk_id"] for row in trace["retrieved"]]
    fetched = collection.get(ids=chunk_ids, include=["documents", "metadatas"])
    by_id = dict(zip(fetched["ids"], zip(fetched["documents"], fetched["metadatas"])))

    documents = [by_id[cid][0] for cid in chunk_ids]
    metadatas = [by_id[cid][1] for cid in chunk_ids]

    return {"ids": [chunk_ids], "documents": [documents], "metadatas": [metadatas]}


def run_fixed_generation(collection, traces):
    """Regenerate every trace's answer under RECIPE_PROMPT_FIXED, same model,
    same retrieved context as week5 — only the prompt text changes."""

    fixed = {}

    for i, trace in enumerate(traces, start=1):
        results = results_from_trace(collection, trace)
        answer = generate_recipe_answer(
            trace["question"], results, model=WEEK4_MODEL, prompt_version="fixed"
        ).strip()
        fixed[trace["id"]] = answer
        print(f"  [{i}/{len(traces)}] {trace['id']}: {answer[:100]}")

    return fixed


def check_hard_regressions(fixed_answers):
    results = {}

    for tid, rule in HARD_REGRESSION.items():
        answer = fixed_answers[tid]
        is_refusal = answer == REFUSAL_TEXT

        if rule["expect"] == "stays_refused":
            ok = is_refusal
            detail = "still refuses" if ok else f"REGRESSION: started answering: {answer[:120]!r}"
        else:  # "flips"
            contains_fact = any(tok.lower() in answer.lower() for tok in rule["must_contain_any"])
            ok = (not is_refusal) and contains_fact
            if is_refusal:
                detail = "STILL REFUSES (fix did not work)"
            elif not contains_fact:
                detail = f"answered but missing expected fact {rule['must_contain_any']}: {answer[:120]!r}"
            else:
                detail = "flipped to a correct, grounded answer"

        results[tid] = {"ok": ok, "detail": detail}

    return results


def check_soft_regressions(fixed_answers):
    results = {}
    for tid, rule in SOFT_REGRESSION.items():
        answer = fixed_answers[tid]
        is_refusal = answer == REFUSAL_TEXT
        contains_fact = any(tok.lower() in answer.lower() for tok in rule["must_contain_any"])
        flipped = (not is_refusal) and contains_fact
        results[tid] = {
            "flipped": flipped,
            "detail": "flipped (bonus)" if flipped else "still refuses (acceptable per prediction)",
        }
    return results


def run_judge_pass(traces, answers_by_id, label):
    verdicts = {}
    print(f"\n  judging {label} answers...")
    for i, trace in enumerate(traces, start=1):
        context = context_from_trace(trace)
        result = judge_answer(trace["question"], context, answers_by_id[trace["id"]])
        verdicts[trace["id"]] = result
        print(f"  [{i}/{len(traces)}] {trace['id']}: {result['verdict']} — {result['reason'][:90]}")
    return verdicts


def validate_judge(judge_baseline_verdicts):
    agreements = 0
    disagreements = []

    for tid, human in HUMAN_VERDICT.items():
        judge_verdict = judge_baseline_verdicts[tid]["verdict"]
        if judge_verdict == human:
            agreements += 1
        else:
            disagreements.append((tid, human, judge_verdict))

    rate = agreements / len(HUMAN_VERDICT)
    return rate, disagreements


def fail_rate(verdict_map, ids, verdict_key=None):
    """Fraction of `ids` graded 'fail'. `verdict_map` is either
    {id: "pass"/"fail"} (HUMAN_VERDICT) or {id: {"verdict": ...}} (judge)."""

    values = []
    for tid in ids:
        entry = verdict_map[tid]
        values.append(entry if verdict_key is None else entry[verdict_key])
    fails = sum(1 for v in values if v == "fail")
    return fails, len(values)


def problem_group_report(judge_baseline_verdicts, judge_fixed_verdicts):
    groups = sorted(set(PROBLEM_GROUP.values())) + ["overall"]
    all_ids = list(HUMAN_VERDICT.keys())

    print(f"\n{'group':<34}{'human-before':<14}{'judge-before':<14}{'judge-after':<12}")
    for group in groups:
        ids = all_ids if group == "overall" else [t for t, g in PROBLEM_GROUP.items() if g == group]
        h_fail, h_n = fail_rate(HUMAN_VERDICT, ids)
        jb_fail, jb_n = fail_rate(judge_baseline_verdicts, ids, verdict_key="verdict")
        jf_fail, jf_n = fail_rate(judge_fixed_verdicts, ids, verdict_key="verdict")
        print(f"{group:<34}{h_fail}/{h_n:<12}{jb_fail}/{jb_n:<12}{jf_fail}/{jf_n}")


def run_citation_safety_net():
    """Re-run Q1/Q5/Q7 (+ U1/U2/U3) under the fixed prompt and confirm the
    already-verified citations / refusals from generation_report.json still
    hold. Retrieval is re-run here (unlike the week5 traces) since no cached
    retrieval was saved for this smaller Week 3/4 set.

    Uses WEEK4_MODEL rather than DEFAULT_MODEL (which generate_answers.py
    used originally) for the same reason Week 4/5 switched: DEFAULT_MODEL's
    free-tier quota is a hard 20 calls/day shared with every other script,
    and this whole evaluation run needs far more than that budget allows.

    Returns (problems, answerable_details) — answerable_details carries the
    question/answer/retrieved-context/expected_answer for Q1/Q5/Q7 so the
    RAGAS-style context_recall step can reuse this generation instead of
    re-querying the model."""

    with open("eval_questions.json", encoding="utf-8") as handle:
        spec = json.load(handle)

    by_id = {q["id"]: q for q in spec["questions"]}
    answerable_ids = spec["answerable_for_generation"]
    unanswerable = spec["unanswerable_questions"]

    collection = get_collection(TRACES_COLLECTION)
    problems = []
    answerable_details = []

    print("\n  answerable (citations must still verify):")
    for qid in answerable_ids:
        question = by_id[qid]["question"]
        results = retrieve(collection, question, 5)
        answer = generate_recipe_answer(
            question, results, model=WEEK4_MODEL, prompt_version="fixed"
        ).strip()
        findings = verify_citations(collection, answer, VERIFY_TOKEN.get(qid))
        ok = bool(findings) and all(f["resolves"] and f.get("contains_claim") for f in findings)
        print(f"    {qid}: {'OK' if ok else 'REGRESSION'} — {answer[:100]}")
        if not ok:
            problems.append(qid)
        answerable_details.append({
            "id": qid,
            "question": question,
            "answer": answer,
            "contexts": results["documents"][0],
            "expected_answer": by_id[qid].get("expected_answer"),
        })

    print("\n  unanswerable (must still refuse exactly):")
    for item in unanswerable:
        question = item["question"]
        results = retrieve(collection, question, 5)
        answer = generate_recipe_answer(
            question, results, model=WEEK4_MODEL, prompt_version="fixed"
        ).strip()
        ok = answer == REFUSAL_TEXT
        print(f"    {item['id']}: {'OK' if ok else 'REGRESSION: ' + answer[:100]}")
        if not ok:
            problems.append(item["id"])

    return problems, answerable_details


def run_ragas_style_metrics(traces, fixed_answers, answerable_details):
    """faithfulness / answer_relevancy / context_precision on all 20 fixed
    trace answers, plus context_recall on Q1/Q5/Q7 (the only questions with
    a known expected_answer to check retrieval completeness against)."""

    per_trace = {}

    print("\n  faithfulness / answer_relevancy / context_precision (20 traces):")
    for i, trace in enumerate(traces, start=1):
        contexts = [row["text"] for row in trace["retrieved"]]
        answer = fixed_answers[trace["id"]]

        faith = faithfulness(trace["question"], contexts, answer)
        relevancy = answer_relevancy(trace["question"], answer)
        precision = context_precision(trace["question"], contexts, answer)

        per_trace[trace["id"]] = {
            "faithfulness": faith["score"],
            "answer_relevancy": relevancy["score"],
            "context_precision": precision["score"],
        }

        def fmt(v):
            return "—" if v is None else f"{v:.2f}"

        print(f"  [{i}/{len(traces)}] {trace['id']}: "
              f"faithfulness={fmt(faith['score'])} "
              f"relevancy={fmt(relevancy['score'])} "
              f"precision={fmt(precision['score'])}")

    print("\n  context_recall (Q1/Q5/Q7 — the only questions with an expected_answer reference):")
    recall_by_id = {}
    for item in answerable_details:
        recall = context_recall(item["question"], item["contexts"], item["expected_answer"])
        recall_by_id[item["id"]] = recall["score"]
        score_str = "—" if recall["score"] is None else f"{recall['score']:.2f}"
        print(f"    {item['id']}: context_recall={score_str}")

    return per_trace, recall_by_id


def average(values):
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present) / len(present)


def main():
    with open("week5_traces.json", encoding="utf-8") as handle:
        data = json.load(handle)

    traces = data["traces"]
    baseline_answers = {t["id"]: t["answer"] for t in traces}

    collection = get_collection(TRACES_COLLECTION)

    print("=" * 78)
    print("STEP 1/6 — regenerating 20 week5 traces under RECIPE_PROMPT_FIXED")
    print("=" * 78)
    fixed_answers = run_fixed_generation(collection, traces)

    print("\n" + "=" * 78)
    print("STEP 2/6 — hard regression checks (rule-based, free)")
    print("=" * 78)
    hard_results = check_hard_regressions(fixed_answers)
    for tid, result in hard_results.items():
        print(f"  {tid}: {'PASS' if result['ok'] else 'FAIL'} — {result['detail']}")
    soft_results = check_soft_regressions(fixed_answers)
    for tid, result in soft_results.items():
        print(f"  {tid} (soft): {result['detail']}")

    print("\n" + "=" * 78)
    print("STEP 3/6 — LLM-judge validation against human grading (week5_error_analysis.md)")
    print("=" * 78)
    judge_baseline_verdicts = run_judge_pass(traces, baseline_answers, "BASELINE")
    agreement_rate, disagreements = validate_judge(judge_baseline_verdicts)
    print(f"\n  judge/human agreement on 20 baseline traces: {agreement_rate:.0%}")
    for tid, human, judge_verdict in disagreements:
        print(f"    disagreement on {tid}: human={human} judge={judge_verdict} "
              f"({judge_baseline_verdicts[tid]['reason']})")

    print("\n" + "=" * 78)
    print("STEP 4/6 — judge-scoring the fixed answers + before/after by problem type")
    print("=" * 78)
    judge_fixed_verdicts = run_judge_pass(traces, fixed_answers, "FIXED")
    problem_group_report(judge_baseline_verdicts, judge_fixed_verdicts)

    print("\n" + "=" * 78)
    print("STEP 5/6 — citation-verification safety net on the Week 3/4 question set")
    print("=" * 78)
    citation_problems, answerable_details = run_citation_safety_net()

    print("\n" + "=" * 78)
    print("STEP 6/6 — RAGAS-style metrics (rag/ragas_metrics.py, custom — see its docstring)")
    print("=" * 78)
    ragas_per_trace, ragas_recall = run_ragas_style_metrics(traces, fixed_answers, answerable_details)
    ragas_averages = {
        "faithfulness": average([v["faithfulness"] for v in ragas_per_trace.values()]),
        "answer_relevancy": average([v["answer_relevancy"] for v in ragas_per_trace.values()]),
        "context_precision": average([v["context_precision"] for v in ragas_per_trace.values()]),
        "context_recall_Q1_Q5_Q7": average(list(ragas_recall.values())),
    }
    print("\n  averages across the 20 traces (None-scored items excluded):")
    for name, value in ragas_averages.items():
        print(f"    {name}: {'—' if value is None else f'{value:.3f}'}")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    hard_ok = all(r["ok"] for r in hard_results.values())
    print(f"  hard regression gate: {'PASS' if hard_ok else 'FAIL'}")
    print(f"  judge/human agreement: {agreement_rate:.0%} "
          f"({'trust the judge' if agreement_rate >= 0.8 else 'DO NOT trust the judge yet'})")
    print(f"  citation safety net: {'PASS' if not citation_problems else 'FAIL: ' + str(citation_problems)}")
    print(f"  RAGAS-style averages: {ragas_averages}")

    report = {
        "fixed_answers": fixed_answers,
        "hard_regressions": hard_results,
        "soft_regressions": soft_results,
        "judge_baseline_verdicts": judge_baseline_verdicts,
        "judge_fixed_verdicts": judge_fixed_verdicts,
        "judge_human_agreement_rate": agreement_rate,
        "judge_human_disagreements": disagreements,
        "citation_safety_net_problems": citation_problems,
        "ragas_style_per_trace": ragas_per_trace,
        "ragas_style_context_recall_Q1_Q5_Q7": ragas_recall,
        "ragas_style_averages": ragas_averages,
    }
    with open("week6_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("\nWrote week6_report.json")


if __name__ == "__main__":
    main()
