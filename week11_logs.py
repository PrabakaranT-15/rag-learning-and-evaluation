"""Week 11: find any past answer in the request log.

    python week11_logs.py --text "brioche dairy-free" --since 2026-10-05
    python week11_logs.py --flagged                 # answers a guard flagged
    python week11_logs.py --trace-id 3fa9c21b04d7   # one request, full spans
    python week11_logs.py --slow 20000              # requests slower than 20 s
    python week11_logs.py --stats                   # cost / latency summary
    python week11_logs.py --log logs/drill_requests.jsonl ...   # another log
"""

import argparse
import json
import statistics

from rag.observability import find_requests, format_record, iter_records, log_path


def _stats(records):
    if not records:
        print("no records")
        return
    costs = [r.get("cost_usd") or 0 for r in records]
    ms = sorted(r.get("total_ms") or 0 for r in records)
    by_kind = {}
    for r in records:
        by_kind.setdefault(r.get("kind"), []).append(r)
    print(f"requests          : {len(records)}")
    print(f"total cost        : ${sum(costs):.6f}")
    print(f"cost per request  : ${statistics.fmean(costs):.6f} (mean)")
    print(f"latency p50 / p95 : {ms[len(ms) // 2]:.0f} ms / {ms[min(len(ms) - 1, int(len(ms) * 0.95))]:.0f} ms")
    print(f"LLM calls / req   : {statistics.fmean(r.get('llm_calls') or 0 for r in records):.2f}")
    print(f"cache hits        : {sum(r.get('cache_hits') or 0 for r in records)}")
    print(f"guard-flagged     : {sum(1 for r in records if r.get('guard_flags'))}")
    print(f"errors            : {sum(1 for r in records if r.get('error'))}")
    for kind, rows in sorted(by_kind.items(), key=lambda kv: str(kv[0])):
        mean = statistics.fmean(r.get("cost_usd") or 0 for r in rows)
        print(f"  {str(kind):<15} n={len(rows):<4} ${mean:.6f}/req")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log")
    ap.add_argument("--text", help="all words must appear in the question or answer")
    ap.add_argument("--trace-id")
    ap.add_argument("--kind", help="chatbot | agent | orchestrator | fixed_workflow")
    ap.add_argument("--since", help="ISO date/time, e.g. 2026-10-05")
    ap.add_argument("--until")
    ap.add_argument("--flagged", action="store_true", help="only answers a guard flagged")
    ap.add_argument("--errors", action="store_true")
    ap.add_argument("--slow", type=float, metavar="MS")
    ap.add_argument("--expensive", type=float, metavar="USD")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-spans", action="store_true")
    ap.add_argument("--limit", type=int, default=20)
    a = ap.parse_args()

    filtered = any([a.text, a.trace_id, a.kind, a.since, a.until, a.flagged, a.errors, a.slow, a.expensive])
    if a.stats and not filtered:
        return _stats(list(iter_records(a.log)))

    found = find_requests(
        a.log, text=a.text, trace_id=a.trace_id, kind=a.kind, since=a.since, until=a.until,
        min_cost=a.expensive, min_ms=a.slow,
        has_error=True if a.errors else None, flagged=True if a.flagged else None,
    )

    if a.stats:
        return _stats(found)

    print(f"{len(found)} match(es) in {a.log or log_path()}\n")
    for record in found[: a.limit]:
        if a.json:
            print(json.dumps(record, indent=2, ensure_ascii=False))
        else:
            print(format_record(record, show_spans=not a.no_spans))
        print("-" * 78)
    if len(found) > a.limit:
        print(f"... {len(found) - a.limit} more (raise --limit)")


if __name__ == "__main__":
    main()
