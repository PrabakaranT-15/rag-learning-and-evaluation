"""Week 11: the request log must record per-step time/cost and be searchable.
No network, no API keys."""

import json
import time

import pytest

from rag import observability as obs
from rag import cache


@pytest.fixture
def log(tmp_path, monkeypatch):
    path = tmp_path / "requests.jsonl"
    monkeypatch.setenv("RAG_LOG_PATH", str(path))
    return path


def _records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_request_writes_one_record_with_per_step_cost(log):
    with obs.trace_request("agent", "how much salt?", restriction=None) as trace:
        with obs.span("plan", step=1):
            with obs.span("llm"):
                obs.record_llm_usage(100, 20, 0.001)
        with obs.span("tool", step=1, tool="search_recipes"):
            time.sleep(0.01)
        trace.set(answer="40 g", retrieved_chunk_ids=["a__0"])

    (record,) = _records(log)
    assert record["question"] == "how much salt?"
    assert record["answer"] == "40 g"
    assert record["trace_id"] and record["timestamp"]
    assert record["llm_calls"] == 1 and record["total_tokens"] == 120
    assert record["cost_usd"] == pytest.approx(0.001)

    by_name = {s["name"]: s for s in record["spans"]}
    # cost rolls up from the llm span to its parent plan span...
    assert by_name["plan"]["cost_usd"] == pytest.approx(0.001)
    assert by_name["llm"]["parent_id"] == by_name["plan"]["span_id"]
    # ...and each step has its own time, not just one lump total.
    assert by_name["tool"]["duration_ms"] >= 10
    assert by_name["tool"]["tool"] == "search_recipes"


def test_nested_request_is_logged_once(log):
    with obs.trace_request("chatbot", "q"):
        with obs.trace_request("fixed_workflow", "q"):
            pass
    assert len(_records(log)) == 1


def test_error_is_recorded_and_reraised(log):
    with pytest.raises(ValueError):
        with obs.trace_request("agent", "q"):
            raise ValueError("boom")
    assert "ValueError: boom" in _records(log)[0]["error"]


def test_span_outside_a_request_is_a_noop():
    with obs.span("x") as node:
        assert node is None
    obs.record_llm_usage(1, 1, 1.0)  # must not raise


def test_logging_failure_never_breaks_the_app(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_LOG_PATH", str(tmp_path))  # a directory: open() fails
    with obs.trace_request("agent", "q") as trace:
        trace.set(answer="still returned")


def test_find_requests_by_vague_text_and_date(log):
    for q, a in [("brioche dairy-free?", "use ghee"), ("salt?", "40 g")]:
        with obs.trace_request("agent", q) as t:
            t.set(answer=a)

    assert len(obs.find_requests(log, text="DAIRY-free brioche")) == 1
    assert len(obs.find_requests(log, text="ghee")) == 1  # matches the answer too
    assert obs.find_requests(log, since="2999-01-01") == []
    assert len(obs.find_requests(log, kind="agent")) == 2


def test_find_requests_skips_corrupt_lines(log):
    with obs.trace_request("agent", "q"):
        pass
    with open(log, "a", encoding="utf-8") as handle:
        handle.write("{not json\n")
    assert len(list(obs.iter_records(log))) == 1


def test_llm_cache_hit_costs_nothing(tmp_path, monkeypatch, log):
    from rag import generator

    monkeypatch.setenv("RAG_CACHE_PATH", str(tmp_path / "cache.sqlite"))
    monkeypatch.setenv("RAG_LLM_CACHE", "1")
    calls = []

    def fake_uncached(prompt, model):
        calls.append(prompt)
        generator.call_log.append({"prompt_tokens": 500, "completion_tokens": 50, "cost_usd": 0.002})
        return "the answer"

    monkeypatch.setattr(generator, "_generate_uncached", fake_uncached)

    with obs.trace_request("chatbot", "q1"):
        assert generator._generate("same prompt", "m") == "the answer"
    with obs.trace_request("chatbot", "q2"):
        assert generator._generate("same prompt", "m") == "the answer"

    first, second = _records(log)
    assert len(calls) == 1                                  # model called once
    assert first["cost_usd"] == pytest.approx(0.002) and first["cache_hits"] == 0
    assert second["cost_usd"] == 0 and second["cache_hits"] == 1


def test_llm_cache_is_off_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_CACHE_PATH", str(tmp_path / "cache.sqlite"))
    monkeypatch.delenv("RAG_LLM_CACHE", raising=False)
    cache.put_llm("m", "p", "text")
    assert cache.get_llm("m", "p") is None


def test_embedding_cache_roundtrip_and_kill_switch(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_CACHE_PATH", str(tmp_path / "cache.sqlite"))
    monkeypatch.delenv("RAG_EMBED_CACHE", raising=False)
    cache.put_embedding("m", "hello", [0.1, 0.2])
    assert cache.get_embedding("m", "hello") == [0.1, 0.2]
    assert cache.get_embedding("m", "other") is None
    monkeypatch.setenv("RAG_EMBED_CACHE", "0")
    assert cache.get_embedding("m", "hello") is None


def test_fixed_workflow_is_traced_end_to_end(log, monkeypatch):
    """The real instrumented pipeline, with retrieval and the model stubbed:
    one record, retrieved chunk ids, spans per step, and a guard flag raised
    on a bad dairy-free answer."""

    from rag import fixed_workflow

    class FakeCollection:
        name = "fake"

    monkeypatch.setattr(fixed_workflow, "search_recipes", lambda c, q, where=None, top_k=5: [
        {"chunk_id": "ferment_004__ingredients__0", "recipe_id": "ferment_004"}])
    monkeypatch.setattr(fixed_workflow, "get_recipe", lambda c, rid: {"ids": [[]], "documents": [[]], "metadatas": [[]]})
    monkeypatch.setattr(fixed_workflow, "generate_recipe_answer",
                        lambda q, full, model=None: "A dairy-free swap is ghee.")

    fixed_workflow.run_fixed_workflow(FakeCollection(), "dairy-free brioche?", "dairy-free")

    (record,) = _records(log)
    assert record["kind"] == "fixed_workflow"
    assert record["retrieved_chunk_ids"] == ["ferment_004__ingredients__0"]
    assert {s["name"] for s in record["spans"]} >= {"retrieve", "get_recipe", "answer"}
    assert record["guard_flags"] == {"dairy_free_swap": ["ghee"]}
    assert obs.find_requests(log, flagged=True)[0]["trace_id"] == record["trace_id"]
