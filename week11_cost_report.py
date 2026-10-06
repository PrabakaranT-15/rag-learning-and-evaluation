"""Week 11: cost per request - the baseline, and the measured improvement.

    python week11_cost_report.py

Prints two things:

1. BASELINE - real per-request numbers already recorded by Week 10's live race
   (week10_race_raw.json: real Groq token counts, real prices). This is the
   "before" for every arm, measured, not estimated.

2. MEASURED IMPROVEMENT - if logs/week11_cold.jsonl and logs/week11_warm.jsonl
   exist (written by week11_measure.py, which replays the same questions twice
   through the exact-match cache), compares them from the request logs.

Nothing here is estimated: a number appears only if a log or a recorded run
contains it. If the live measurement has not been run yet, section 2 says so.
"""

import json
import statistics
from pathlib import Path

from rag.observability import iter_records

ROOT = Path(__file__).resolve().parent
RACE_RAW = ROOT / "week10_race_raw.json"
COLD = ROOT / "logs" / "week11_cold.jsonl"
WARM = ROOT / "logs" / "week11_warm.jsonl"


def summarize(records):
    records = list(records)
    if not records:
        return None
    ms = sorted(r.get("total_ms") or 0 for r in records)
    return {
        "n": len(records),
        "cost": statistics.fmean(r.get("cost_usd") or 0 for r in records),
        "tokens": statistics.fmean(r.get("total_tokens") or 0 for r in records),
        "llm_calls": statistics.fmean(r.get("llm_calls") or 0 for r in records),
        "cache_hits": sum(r.get("cache_hits") or 0 for r in records),
        "p50_ms": ms[len(ms) // 2],
        "total_cost": sum(r.get("cost_usd") or 0 for r in records),
    }


def baseline_from_week10():
    if not RACE_RAW.exists():
        return {}
    results = json.loads(RACE_RAW.read_text(encoding="utf-8"))["results"]
    out = {}
    for arm in ("single_agent", "orchestrator"):
        rows = [v[arm] for v in results.values() if v.get(arm)]
        if rows:
            out[arm] = {
                "n": len(rows),
                "cost": statistics.fmean(r["cost_usd"] for r in rows),
                "tokens": statistics.fmean(r["tokens"] for r in rows),
                "seconds": statistics.fmean(r["elapsed_seconds"] for r in rows),
            }
    return out


def main():
    print("=" * 74)
    print("1. BASELINE - Week 10 live race (real Groq tokens x real price)")
    print("=" * 74)
    base = baseline_from_week10()
    if not base:
        print("week10_race_raw.json not found")
    for arm, b in base.items():
        print(f"  {arm:<13} n={b['n']}  ${b['cost']:.5f}/request  {b['tokens']:.0f} tokens/request  {b['seconds']:.1f} s/request")

    print()
    print("=" * 74)
    print("2. MEASURED IMPROVEMENT - same questions, cold run vs warm (cached) run")
    print("=" * 74)
    cold = summarize(iter_records(COLD))
    warm = summarize(iter_records(WARM))

    if not cold or not warm:
        print("  NOT MEASURED YET - needs live API keys:")
        print("      python week11_measure.py")
        print("  then re-run this report.")
        return

    print(f"  {'':<10}{'n':>4}{'$/request':>12}{'tokens':>9}{'LLM calls':>11}{'cache hits':>12}{'p50 ms':>10}")
    for name, s in (("cold", cold), ("warm", warm)):
        print(f"  {name:<10}{s['n']:>4}{s['cost']:>12.6f}{s['tokens']:>9.0f}{s['llm_calls']:>11.2f}{s['cache_hits']:>12}{s['p50_ms']:>10.0f}")

    saved = 1 - warm["cost"] / cold["cost"] if cold["cost"] else 0
    faster = 1 - warm["p50_ms"] / cold["p50_ms"] if cold["p50_ms"] else 0
    print(f"\n  cost per request: ${cold['cost']:.6f} -> ${warm['cost']:.6f}  ({saved:.0%} lower on a repeated question)")
    print(f"  p50 latency     : {cold['p50_ms']:.0f} ms -> {warm['p50_ms']:.0f} ms  ({faster:.0%} lower)")
    print("\n  A repeated question is the best case. Real saving = this x your repeat rate:")
    for hit_rate in (0.1, 0.3, 0.5):
        blended = cold["cost"] * (1 - hit_rate) + warm["cost"] * hit_rate
        print(f"    {hit_rate:.0%} repeat traffic -> ${blended:.6f}/request")


if __name__ == "__main__":
    main()
