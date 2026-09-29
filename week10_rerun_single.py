"""One-off: re-run only the single-agent arm (bug fix in judge-context
construction - see week10_race.py's run_single_agent_case), keep the
already-recorded orchestrator results as-is."""

import json
import sys

from week10_race import _load_questions, detect_restriction, run_single_agent_case


def main():
    data = json.load(open("week10_race_raw.json", encoding="utf-8"))
    questions = _load_questions()

    resume_from = sys.argv[1:] or list(questions.keys())

    for case_id, question in questions.items():
        if case_id not in resume_from:
            continue
        restriction = detect_restriction(question)
        print(f"[{case_id}] re-running single agent...", flush=True)
        single = run_single_agent_case(question, restriction)
        print(f"  {single['verdict']} - {single['tokens']} tokens, {single['elapsed_seconds']:.1f}s", flush=True)
        data["results"][case_id]["single_agent"] = single

        with open("week10_race_raw.json", "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    print("done")


if __name__ == "__main__":
    main()
