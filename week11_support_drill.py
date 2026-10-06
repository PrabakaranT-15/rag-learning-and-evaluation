"""Week 11 support drill (Track B - Recipes & food): "find the dairy-free swap
that wasn't".

Builds logs/drill_requests.jsonl: 300 synthetic request records over three
days with ONE planted bad answer - a "dairy-free" brioche swap that recommends
ghee (which is dairy) - buried among them. Every record, planted or not, has
"synthetic": true, and this file is separate from the real logs/requests.jsonl,
so it can never be mistaken for production data.

The drill (week11_report.md has the worked run):

  Vague complaint: "Yesterday somebody told me the brioche had a dairy-free
  option and it wasn't. I don't have a trace id."

    python week11_support_drill.py
    python week11_logs.py --log logs/drill_requests.jsonl --text "brioche dairy-free" --since 2026-10-05 --until 2026-10-06
    python week11_logs.py --log logs/drill_requests.jsonl --flagged
"""

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

from rag.guards import check_answer

OUT = Path(__file__).resolve().parent / "logs" / "drill_requests.jsonl"
SEED = 11

PLANTED_QUESTION = "Is there a dairy-free way to make the brioche?"
PLANTED_ANSWER = (
    "Yes - a dairy-free swap for the butter in the brioche is ghee at a 1:1 ratio, "
    "which keeps the dough rich [recipe_id=ferment_004 | chunk_id=ferment_004__ingredients__0 | "
    "source_file=ferment_004_sourdough_brioche.md]."
)

NOISE = [
    ("How much salt goes into the country sourdough recipe?", "The country sourdough uses 40 g of fine sea salt, 2% of the flour [recipe_id=ferment_001]."),
    ("How long should I proof the brioche dough?", "Proof the brioche dough until doubled, per the card [recipe_id=ferment_004]."),
    ("Can I make the kimchi vegan?", "Use the vegan napa cabbage kimchi card, which replaces the fish sauce [recipe_id=ferment_006]."),
    ("What's the hydration percentage for the sourdough?", "The hydration is stated in the ingredient table [recipe_id=ferment_001]."),
    ("How many calories are in a slice of the brioche?", "I cannot answer this from the provided recipe documents."),
    ("Is the sourdough bread gluten free?", "No - it contains wheat and rye, therefore gluten [recipe_id=ferment_001]."),
    ("What's a good substitute for the fish sauce in the kimchi?", "From the ingredient database: a vegan fish-free swap is listed [mcp-tool:get_substitutes]."),
    ("how much butter in the brioche", "The brioche card lists the butter weight in its ingredient table [recipe_id=ferment_004]."),
    ("What's the ratio of flour to water in the rye levain?", "See the levain ingredient table [recipe_id=ferment_002]."),
    ("Why does my sourdough starter smell like nail polish?", "The card notes a faint acetone note in a ripe starter [recipe_id=ferment_002]."),
    ("Is the focaccia dairy-free?", "Yes - the focaccia is tagged dairy-free and uses olive oil [recipe_id=ferment_003]."),
    ("Can I replace the milk in the brioche with oat milk to make it dairy-free?", "From the ingredient database, oat milk is a listed 1:1 dairy-free swap for whole milk [mcp-tool:get_substitutes]."),
    ("Which recipes are vegan?", "The sourdough loaf, rye levain, focaccia and vegan kimchi are tagged vegan."),
    ("What temperature for focaccia?", "I cannot answer this from the provided recipe documents."),
]
KINDS = [("chatbot", 0.45), ("agent", 0.45), ("orchestrator", 0.10)]
KIND_COST = {"chatbot": 0.00035, "agent": 0.0018, "orchestrator": 0.0011}


def _spans(kind, rng, cost):
    spans = [{"span_id": 1, "parent_id": None, "name": "embedding", "start_ms": 0,
              "duration_ms": round(rng.uniform(250, 1500), 1), "prompt_tokens": 0,
              "completion_tokens": 0, "cost_usd": 0.0, "cache_hit": False}]
    t = spans[0]["duration_ms"]
    steps = {"chatbot": 1, "agent": rng.randint(3, 5), "orchestrator": 3}[kind]
    for _ in range(steps):
        llm_ms = rng.uniform(900, 2600)
        spans.append({"span_id": len(spans) + 1, "parent_id": None, "name": "llm", "start_ms": round(t, 1),
                      "duration_ms": round(llm_ms, 1), "prompt_tokens": 0, "completion_tokens": 0,
                      "cost_usd": round(cost / steps, 8), "cache_hit": False})
        t += llm_ms
    return spans, t, steps


def build():
    rng = random.Random(SEED)
    start = datetime(2026, 10, 4, 8, 0, tzinfo=timezone.utc)
    records = []

    for i in range(300):
        question, answer = rng.choice(NOISE)
        kind = rng.choices([k for k, _ in KINDS], [w for _, w in KINDS])[0]
        cost = KIND_COST[kind] * rng.uniform(0.7, 1.4)
        spans, total_ms, llm_calls = _spans(kind, rng, cost)
        ts = start + timedelta(minutes=i * 14.3 + rng.uniform(0, 9))
        tokens = int(cost / 0.0000003)
        records.append({
            "trace_id": f"{rng.getrandbits(48):012x}", "timestamp": ts.isoformat(timespec="milliseconds"),
            "kind": kind, "question": question, "answer": answer,
            "collection": "fermentation_structure_aware", "synthetic": True,
            "total_ms": round(total_ms, 1), "llm_calls": llm_calls, "cache_hits": 0,
            "prompt_tokens": int(tokens * 0.8), "completion_tokens": int(tokens * 0.2),
            "total_tokens": tokens, "cost_usd": round(cost, 8),
            "guard_flags": check_answer(answer), "error": None, "spans": spans,
        })

    spans, total_ms, llm_calls = _spans("agent", rng, 0.0019)
    records.append({
        "trace_id": "d41ry7ee0001", "timestamp": "2026-10-05T14:37:21.408+00:00", "kind": "agent",
        "question": PLANTED_QUESTION, "answer": PLANTED_ANSWER,
        "collection": "fermentation_structure_aware", "restriction": "dairy-free",
        "recipe_id": "ferment_004", "synthetic": True, "planted": True,
        "total_ms": round(total_ms, 1), "llm_calls": llm_calls, "cache_hits": 0,
        "prompt_tokens": 5100, "completion_tokens": 640, "total_tokens": 5740, "cost_usd": 0.0019,
        "guard_flags": check_answer(PLANTED_ANSWER), "error": None, "spans": spans,
    })

    records.sort(key=lambda r: r["timestamp"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        for r in records:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    return records


if __name__ == "__main__":
    recs = build()
    print(f"wrote {len(recs)} synthetic records to {OUT} (1 planted)")
