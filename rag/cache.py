"""Week 11 Module 6: exact-match caches, the cheapest cost lever there is.

Two caches, one SQLite file (data/cache.sqlite - SQLite so the Streamlit
process AND the MCP server subprocesses, which each embed queries, can share
it safely):

* embeddings - text -> vector. An embedding is a pure function of (model,
  text), so caching it is always correct. ON by default; disable with
  RAG_EMBED_CACHE=0. Saves a Gemini call AND its ~1.1 s pacing sleep, which
  matters because every recipe question embeds the same query in the chatbot
  path, again in the agent's search, and again on every retry.
* llm - (model, prompt) -> answer text. Identical prompt, identical answer,
  zero tokens. OFF by default (RAG_LLM_CACHE=1 to enable; app.py enables it)
  so the Week 4-10 experiments, which deliberately re-sample a model, stay
  reproducible exactly as they were measured.

This is EXACT-match caching, not semantic caching: "How much salt?" and "how
much salt is in it" are different keys. Semantic caching (nearest-neighbour on
the question embedding) would raise the hit rate but can return a wrong
answer for a question that merely sounds similar - on a recipe/allergen app
that risk is not worth it, so it is documented in week11_report.md as a
deliberate non-choice.

Every function here swallows storage errors and behaves like a cache miss:
a broken cache must make the app slower, never wrong or down.
"""

import hashlib
import json
import os
import sqlite3
from pathlib import Path


DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "cache.sqlite"


def _db_path():
    return Path(os.getenv("RAG_CACHE_PATH") or DEFAULT_DB_PATH)


def embed_cache_enabled():
    return os.getenv("RAG_EMBED_CACHE", "1") != "0"


def llm_cache_enabled():
    return os.getenv("RAG_LLM_CACHE", "0") == "1"


def _key(*parts):
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def prompt_hash(text):
    """Short stable id for a prompt, for logs (the prompt itself is not logged)."""

    return _key(text)[:16]


def _connect():
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.execute("CREATE TABLE IF NOT EXISTS embeddings (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
    conn.execute("CREATE TABLE IF NOT EXISTS llm (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
    return conn


def _get(table, key):
    try:
        conn = _connect()
        try:
            row = conn.execute(f"SELECT v FROM {table} WHERE k = ?", (key,)).fetchone()
        finally:
            conn.close()
        return row[0] if row else None
    except Exception:
        return None


def _put(table, key, value):
    try:
        conn = _connect()
        try:
            with conn:
                conn.execute(f"INSERT OR REPLACE INTO {table} (k, v) VALUES (?, ?)", (key, value))
        finally:
            conn.close()
    except Exception:
        pass


# ---------------------------------------------------------------- embeddings

def get_embedding(model, text):
    if not embed_cache_enabled():
        return None
    raw = _get("embeddings", _key(model, text))
    return json.loads(raw) if raw is not None else None


def put_embedding(model, text, vector):
    if embed_cache_enabled():
        _put("embeddings", _key(model, text), json.dumps(list(vector)))


# ----------------------------------------------------------------------- llm

def get_llm(model, prompt):
    if not llm_cache_enabled():
        return None
    return _get("llm", _key(model, prompt))


def put_llm(model, prompt, text):
    if llm_cache_enabled() and text:
        _put("llm", _key(model, prompt), text)
