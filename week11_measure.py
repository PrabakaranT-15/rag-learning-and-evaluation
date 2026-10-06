"""Week 11: measure the cache improvement LIVE (needs GROQ_API_KEY and
GOOGLE_API_KEY, and the fermentation_structure_aware collection ingested).

Runs the same questions through the agent twice with the exact-match cache on
(rag/cache.py), using a throw-away cache file so the first run is genuinely
cold:

    python week11_measure.py                 # 10 questions, agent
    python week11_measure.py --n 5 --kind chatbot

Writes logs/week11_cold.jsonl and logs/week11_warm.jsonl - ordinary request
logs, so `python week11_logs.py --log logs/week11_warm.jsonl --stats` and
`python week11_cost_report.py` read them like any other log.
"""

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--kind", choices=["agent", "chatbot"], default="agent")
    ap.add_argument("--collection", default="fermentation_structure_aware")
    args = ap.parse_args()

    LOGS.mkdir(exist_ok=True)
    cache_file = LOGS / "measure_cache.sqlite"
    for stale in (cache_file, LOGS / "week11_cold.jsonl", LOGS / "week11_warm.jsonl"):
        if stale.exists():
            stale.unlink()

    # Set BEFORE importing rag.*: MCP server subprocesses inherit these too,
    # so their query embeddings use the same throw-away cache.
    os.environ["RAG_CACHE_PATH"] = str(cache_file)
    os.environ["RAG_LLM_CACHE"] = "1"
    os.environ["RAG_EMBED_CACHE"] = "1"

    from rag.vector_store import get_collection
    from rag.agent import run_agent
    from rag.fixed_workflow import run_fixed_workflow

    questions = json.loads((ROOT / "week5_trace_questions.json").read_text(encoding="utf-8"))
    questions = (questions["questions"] if isinstance(questions, dict) else questions)[: args.n]
    collection = get_collection(args.collection)

    for label in ("cold", "warm"):
        os.environ["RAG_LOG_PATH"] = str(LOGS / f"week11_{label}.jsonl")
        print(f"--- {label} pass ({len(questions)} questions, {args.kind}) ---")
        for q in questions:
            text = q["question"]
            restriction = q.get("restriction")
            if args.kind == "agent":
                result = run_agent(collection, text, restriction)
            else:
                result = run_fixed_workflow(collection, text, restriction)
            print(f"  {text[:60]:<60} -> {result['answer'][:50]!r}")

    print("\nDone. Now run: python week11_cost_report.py")


if __name__ == "__main__":
    main()
