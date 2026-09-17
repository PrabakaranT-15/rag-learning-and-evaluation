"""Week 8 Module 4: analyze week8_trajectories.json for the outcome-vs-
trajectory gap, tool-choice accuracy, and cost - the "one-command harness"
over a batch, same shape as week6_run_evals.py.

Run with: python week8_trajectory_report.py
"""

import json

from rag.trajectory_eval import (
    finished_without_verification,
    hallucinated_check_input,
    tool_choice_accuracy,
    tool_choice_violations,
    cost_stats,
)


def main():

    with open("week8_trajectories.json", encoding="utf-8") as handle:
        data = json.load(handle)

    runs = data["runs"]

    print("=" * 78)
    print(f"WEEK 8 TRAJECTORY REPORT  -  collection: {data['collection']}  model: {data['model']}")
    print(f"{len(runs)} scenarios")
    print("=" * 78)

    gap_examples = []
    hallucination_examples = []
    tool_violation_examples = []

    for run in runs:
        steps = run["steps"]
        restriction = run["restriction"]

        skipped_verification = finished_without_verification(steps, restriction)
        hallucinations = hallucinated_check_input(steps)
        tool_violations = tool_choice_violations(steps)

        run["skipped_verification"] = skipped_verification
        run["hallucinated_check_inputs"] = hallucinations
        run["tool_choice_violations"] = tool_violations

        if run["correct"] and skipped_verification:
            gap_examples.append(run)
        if hallucinations:
            hallucination_examples.append(run)
        if tool_violations:
            tool_violation_examples.append(run)

    correct_count = sum(1 for r in runs if r["correct"])
    gap_count = len(gap_examples)

    print(f"\nOUTCOME:            {correct_count}/{len(runs)} scenarios reached the expected recipe/refusal")
    print(f"OUTCOME-VS-TRAJECTORY GAP:  {gap_count}/{correct_count} of the CORRECT answers were reached "
          f"WITHOUT a passing check_restriction for the finished recipe")
    print(f"HALLUCINATED CHECK INPUTS:  {len(hallucination_examples)}/{len(runs)} runs")
    print(f"TOOL-CHOICE VIOLATIONS:     {len(tool_violation_examples)}/{len(runs)} runs")

    accuracies = [a for a in (tool_choice_accuracy(r["steps"]) for r in runs) if a is not None]
    mean_tool_choice_accuracy = sum(accuracies) / len(accuracies) if accuracies else None
    print(f"MEAN TOOL-CHOICE ACCURACY:  {mean_tool_choice_accuracy:.1%}" if mean_tool_choice_accuracy is not None
          else "MEAN TOOL-CHOICE ACCURACY:  n/a")

    costs = cost_stats(runs)
    print(f"\nCOST PER TASK  (n={costs['n']})")
    print(f"  llm_calls:        mean={costs['llm_calls_mean']:.2f}  p99={costs['llm_calls_p99']:.2f}")
    print(f"  elapsed_seconds:  mean={costs['elapsed_seconds_mean']:.2f}  p99={costs['elapsed_seconds_p99']:.2f}")

    if gap_examples:
        worked = gap_examples[0]
        print("\n" + "=" * 78)
        print(f"WORKED EXAMPLE - right answer, wrong path: {worked['name']}")
        print("=" * 78)
        print(f"query: {worked['query']!r}  restriction: {worked['restriction']!r}")
        print(f"finished_recipe_id: {worked['finished_recipe_id']}  (compliant, so graded 'correct')")
        print("but no check_restriction call verified it against its real dietary_tags. steps:")
        for step in worked["steps"]:
            print(f"  step {step['step']} · {step['tool']}  args={step['args']}")
        print(f"answer: {worked['answer'][:300]}")
    else:
        print("\nNo outcome-vs-trajectory gap found in this batch.")

    report = {
        "collection": data["collection"],
        "model": data["model"],
        "n_scenarios": len(runs),
        "correct_count": correct_count,
        "outcome_vs_trajectory_gap_count": gap_count,
        "hallucinated_check_input_count": len(hallucination_examples),
        "tool_choice_violation_count": len(tool_violation_examples),
        "mean_tool_choice_accuracy": mean_tool_choice_accuracy,
        "cost_stats": costs,
        "gap_example_names": [r["name"] for r in gap_examples],
        "runs": runs,
    }

    with open("week8_trajectory_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("\nWrote week8_trajectory_report.json")


if __name__ == "__main__":
    main()
