"""Pure-function tests for rag/trajectory_eval.py - synthetic transcripts,
no network, no LLM, no Chroma. Shapes match what rag/agent.py's run_agent()
actually appends to its transcript (see rag/agent.py's _run_tool / the
finish branch)."""

from rag.trajectory_eval import (
    finished_recipe_id,
    finished_without_verification,
    hallucinated_check_input,
    tool_choice_violations,
    tool_choice_accuracy,
    cost_stats,
)


def _search_step(step_no, recipe_id, dietary_tags):
    return {
        "step": step_no, "tool": "search_recipes", "args": {"query": "q"},
        "observation": [{"recipe_id": recipe_id, "dietary_tags": dietary_tags, "chunk_id": "c1"}],
    }


def _check_step(step_no, dietary_tags, restriction, satisfied):
    return {
        "step": step_no, "tool": "check_restriction",
        "args": {"dietary_tags": dietary_tags, "restriction": restriction},
        "observation": {"restriction": restriction, "dietary_tags": dietary_tags, "satisfied": satisfied},
    }


def _finish_step(step_no, recipe_id):
    return {"step": step_no, "tool": "finish", "args": {"recipe_id": recipe_id}, "observation": "loop ended"}


def test_finished_recipe_id_finds_the_finish_step():
    transcript = [_search_step(1, "r1", "vegan"), _finish_step(2, "r1")]
    assert finished_recipe_id(transcript) == "r1"


def test_finished_recipe_id_none_when_no_finish():
    assert finished_recipe_id([_search_step(1, "r1", "vegan")]) is None


def test_verified_finish_is_not_a_violation():
    transcript = [
        _search_step(1, "r1", "vegan,nut-free"),
        _check_step(2, "vegan,nut-free", "vegan", True),
        _finish_step(3, "r1"),
    ]
    assert finished_without_verification(transcript, "vegan") is False


def test_finish_without_any_check_is_a_violation():
    transcript = [_search_step(1, "r1", "vegan,nut-free"), _finish_step(2, "r1")]
    assert finished_without_verification(transcript, "vegan") is True


def test_finish_with_a_failed_check_is_still_a_violation():
    """A check_restriction call happened, but it reported satisfied=False -
    finishing anyway is exactly the gap this module exists to catch."""
    transcript = [
        _search_step(1, "r1", "non-vegan"),
        _check_step(2, "non-vegan", "vegan", False),
        _finish_step(3, "r1"),
    ]
    assert finished_without_verification(transcript, "vegan") is True


def test_no_restriction_is_never_a_violation():
    transcript = [_search_step(1, "r1", "vegan"), _finish_step(2, "r1")]
    assert finished_without_verification(transcript, None) is False


def test_honest_refusal_is_never_a_violation():
    transcript = [_search_step(1, "r1", "non-vegan"), _finish_step(2, None)]
    assert finished_without_verification(transcript, "vegan") is False


def test_hallucinated_check_input_flags_unseen_tags():
    transcript = [
        _search_step(1, "r1", "vegan,nut-free"),
        _check_step(2, "completely-made-up-tags", "vegan", True),
    ]
    violations = hallucinated_check_input(transcript)
    assert len(violations) == 1
    assert violations[0]["dietary_tags"] == "completely-made-up-tags"


def test_hallucinated_check_input_clean_when_tags_match_a_real_search():
    transcript = [
        _search_step(1, "r1", "vegan,nut-free"),
        _check_step(2, "vegan,nut-free", "vegan", True),
    ]
    assert hallucinated_check_input(transcript) == []


def test_tool_choice_violation_check_before_search():
    transcript = [_check_step(1, "vegan", "vegan", True)]
    violations = tool_choice_violations(transcript)
    assert any("before any search_recipes" in v["reason"] for v in violations)


def test_tool_choice_violation_get_recipe_on_unseen_id():
    transcript = [
        _search_step(1, "r1", "vegan"),
        {"step": 2, "tool": "get_recipe", "args": {"recipe_id": "r99"}, "observation": {}},
    ]
    violations = tool_choice_violations(transcript)
    assert any("r99" in v["reason"] for v in violations)


def test_tool_choice_accuracy_perfect_run():
    transcript = [
        _search_step(1, "r1", "vegan"),
        _check_step(2, "vegan", "vegan", True),
        _finish_step(3, "r1"),
    ]
    assert tool_choice_accuracy(transcript) == 1.0


def test_tool_choice_accuracy_none_on_empty_transcript():
    assert tool_choice_accuracy([]) is None


def test_cost_stats_mean_and_p99():
    runs = [
        {"llm_calls": 2, "elapsed_seconds": 1.0},
        {"llm_calls": 4, "elapsed_seconds": 3.0},
        {"llm_calls": 6, "elapsed_seconds": 5.0},
    ]
    stats = cost_stats(runs)
    assert stats["n"] == 3
    assert stats["llm_calls_mean"] == 4.0
    # p99 on 3 samples is linear-interpolated close to, not exactly, the max.
    assert stats["llm_calls_p99"] == 5.96
    assert stats["elapsed_seconds_mean"] == 3.0
