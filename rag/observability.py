"""Week 11 Module 6: per-request observability.

If it isn't logged, it didn't happen. Every user-facing request (a chatbot
answer, an agent run, an orchestrator run) is wrapped in `trace_request()`,
which writes ONE JSON line to logs/requests.jsonl when the request ends:

    trace_id, timestamp, kind, question, restriction, collection, model,
    retrieved chunk ids, the final answer, total time / tokens / cost, any
    error - and a `spans` list with the time, tokens and cost of EACH step
    (embedding, retrieval, every LLM call, every agent step), not just one
    lump total.

Spans nest (an agent `plan` span contains its `llm` span), so a slow or
expensive request can be read top-down: which step, and what inside it.

Design rules
------------
* Logging must never break the app: every write failure is swallowed.
* No API keys and no full prompts are logged - only a prompt hash and sizes -
  so the log is safe to share with a mentor or attach to a ticket.
* Standard library only. The record shape is deliberately flat/OpenTelemetry-
  like (trace_id, span_id, parent_id, start_ms, duration_ms, attributes) so it
  can be forwarded to LangSmith / Phoenix / an OTel collector later without
  changing any call site.
"""

import contextvars
import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "requests.jsonl"

_write_lock = threading.Lock()

_current_trace = contextvars.ContextVar("rag_current_trace", default=None)
_current_span = contextvars.ContextVar("rag_current_span", default=None)


def log_path():
    """Where records go. RAG_LOG_PATH overrides (tests, drills)."""

    return Path(os.getenv("RAG_LOG_PATH") or DEFAULT_LOG_PATH)


class Trace:
    """One request's accumulating record. Written to disk by trace_request()."""

    def __init__(self, kind, question, fields):
        self.trace_id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.question = question
        self.fields = dict(fields)
        self.spans = []
        self.started_wall = datetime.now(timezone.utc)
        self.started = time.perf_counter()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost_usd = 0.0
        self.llm_calls = 0
        self.cache_hits = 0
        self._next_span = 0

    def set(self, **fields):
        """Attach request-level facts (answer, retrieved_chunk_ids, ...)."""

        self.fields.update(fields)

    def new_span_id(self):
        self._next_span += 1
        return self._next_span

    def to_record(self, error=None):
        record = {
            "trace_id": self.trace_id,
            "timestamp": self.started_wall.isoformat(timespec="milliseconds"),
            "kind": self.kind,
            "question": self.question,
            **self.fields,
            "total_ms": round((time.perf_counter() - self.started) * 1000, 1),
            "llm_calls": self.llm_calls,
            "cache_hits": self.cache_hits,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "cost_usd": round(self.cost_usd, 8),
            "error": error,
            "spans": self.spans,
        }
        return record


def current_trace():
    """The active Trace, or None outside a request."""

    return _current_trace.get()


def _write(record):
    try:
        path = log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _write_lock, open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except Exception:
        # Observability must never take the app down.
        pass


@contextmanager
def trace_request(kind, question, **fields):
    """Wrap one whole request. Yields the Trace (call .set(answer=...) on it).

    Nested calls reuse the outer trace instead of writing a second record, so
    a helper that is both called directly and from inside a bigger request is
    logged exactly once, under the outermost request.
    """

    existing = _current_trace.get()
    if existing is not None:
        yield existing
        return

    trace = Trace(kind, question, fields)
    token = _current_trace.set(trace)
    span_token = _current_span.set(None)
    error = None
    try:
        yield trace
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _current_span.reset(span_token)
        _current_trace.reset(token)
        _write(trace.to_record(error=error))


class _Span:
    def __init__(self, trace, name, parent, attrs):
        self.trace = trace
        self.span_id = trace.new_span_id()
        self.name = name
        self.parent_id = parent.span_id if parent else None
        self.attrs = dict(attrs)
        self.started = time.perf_counter()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost_usd = 0.0

    def set(self, **attrs):
        self.attrs.update(attrs)


@contextmanager
def span(name, **attrs):
    """Time one step of the current request. A no-op (yields None) outside a
    request, so library code can be instrumented without caring whether
    anything is listening. Tokens/cost recorded inside roll up to the parent."""

    trace = _current_trace.get()
    if trace is None:
        yield None
        return

    parent = _current_span.get()
    node = _Span(trace, name, parent, attrs)
    token = _current_span.set(node)
    try:
        yield node
    except BaseException as exc:
        node.attrs["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _current_span.reset(token)
        if parent is not None:
            parent.prompt_tokens += node.prompt_tokens
            parent.completion_tokens += node.completion_tokens
            parent.cost_usd += node.cost_usd
        entry = {
            "span_id": node.span_id,
            "parent_id": node.parent_id,
            "name": node.name,
            "start_ms": round((node.started - trace.started) * 1000, 1),
            "duration_ms": round((time.perf_counter() - node.started) * 1000, 1),
            "prompt_tokens": node.prompt_tokens,
            "completion_tokens": node.completion_tokens,
            "cost_usd": round(node.cost_usd, 8),
            **node.attrs,
        }
        trace.spans.append(entry)


def record_llm_usage(prompt_tokens, completion_tokens, cost_usd, cache_hit=False):
    """Called by rag/generator.py for every model call: charges the tokens and
    dollars to the current span (and request). Cache hits count as a call but
    cost nothing."""

    trace = _current_trace.get()
    if trace is None:
        return

    prompt_tokens = prompt_tokens or 0
    completion_tokens = completion_tokens or 0
    cost_usd = cost_usd or 0.0

    trace.llm_calls += 1
    trace.cache_hits += 1 if cache_hit else 0
    trace.prompt_tokens += prompt_tokens
    trace.completion_tokens += completion_tokens
    trace.cost_usd += cost_usd

    node = _current_span.get()
    if node is not None:
        node.prompt_tokens += prompt_tokens
        node.completion_tokens += completion_tokens
        node.cost_usd += cost_usd


# ------------------------------------------------------------------ reading

def iter_records(path=None):
    """Yield every well-formed record in a log file; skip corrupt lines (a
    crash mid-write must not make the rest of the log unreadable)."""

    path = Path(path) if path else log_path()
    if not path.exists():
        return

    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def find_requests(path=None, text=None, trace_id=None, kind=None,
                  since=None, until=None, min_cost=None, min_ms=None, has_error=None,
                  flagged=None):
    """Filter the log. `text` matches (case-insensitive, ALL words) against the
    question OR the answer - the way a vague customer complaint actually has to
    be searched. `since`/`until` are ISO timestamps ("2026-10-05" works)."""

    words = [w for w in (text or "").lower().split() if w]
    matches = []

    for record in iter_records(path):
        if trace_id and record.get("trace_id") != trace_id:
            continue
        if kind and record.get("kind") != kind:
            continue
        if since and record.get("timestamp", "") < since:
            continue
        if until and record.get("timestamp", "") > until:
            continue
        if min_cost is not None and (record.get("cost_usd") or 0) < min_cost:
            continue
        if min_ms is not None and (record.get("total_ms") or 0) < min_ms:
            continue
        if has_error is not None and bool(record.get("error")) != has_error:
            continue
        if flagged is not None and bool(record.get("guard_flags")) != flagged:
            continue
        if words:
            haystack = f"{record.get('question', '')} {record.get('answer', '')}".lower()
            if not all(w in haystack for w in words):
                continue
        matches.append(record)

    return matches


def format_record(record, show_spans=True):
    """Human-readable one-request view: the answer plus the per-step
    time/cost table the Week 11 brief asks for."""

    lines = [
        f"trace_id : {record.get('trace_id')}   [{record.get('kind')}]   {record.get('timestamp')}",
        f"question : {record.get('question')}",
        f"answer   : {(record.get('answer') or '')[:400]}",
        f"total    : {record.get('total_ms')} ms, {record.get('llm_calls')} LLM call(s), "
        f"{record.get('total_tokens')} tokens, ${record.get('cost_usd')}",
    ]
    if record.get("retrieved_chunk_ids"):
        lines.append(f"retrieved: {', '.join(record['retrieved_chunk_ids'])}")
    if record.get("guard_flags"):
        lines.append(f"GUARD    : {record['guard_flags']}")
    if record.get("error"):
        lines.append(f"ERROR    : {record['error']}")

    if show_spans and record.get("spans"):
        lines.append("steps    :")
        for s in sorted(record["spans"], key=lambda s: s.get("start_ms", 0)):
            indent = "  " * (1 + (1 if s.get("parent_id") else 0))
            extra = f" tool={s['tool']}" if s.get("tool") else ""
            lines.append(
                f"{indent}{s['name']:<16} {s['duration_ms']:>9} ms  "
                f"{(s.get('prompt_tokens', 0) + s.get('completion_tokens', 0)):>6} tok  "
                f"${s.get('cost_usd', 0):.6f}{extra}"
            )
    return "\n".join(lines)
