import html
import json
import os
import time

import streamlit as st

from rag import generator
from rag.pdf_loader import load_pdf
from rag.chunker import create_chunks
from rag.vector_store import build_index, retrieve, get_collection, list_collections
from rag.hybrid_retriever import retrieve_hybrid
from rag.generator import generate_answer, generate_recipe_answer
from rag.evaluation import classify, rows_from_results, normalise
from rag.agent import run_agent
from rag.tools import restriction_where


st.set_page_config(
    page_title="RAG Recipe Assistant",
    page_icon="🧑‍🍳",
    layout="wide",
)

# ----------------------------------------------------------------- icons
# Small stroke-based inline SVGs (20px grid, 1.8 stroke) instead of emoji,
# for the landmark UI elements (hero, stat tiles, mode cards, sources) -
# emoji stays only in secondary/collapsed spots (trace expander labels)
# that already worked fine and weren't part of the redesign ask.

def _icon(paths, size=18, stroke="currentColor", width=1.8):
    body = "".join(f'<path d="{d}"/>' for d in paths)
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="{stroke}" stroke-width="{width}" stroke-linecap="round" '
        f'stroke-linejoin="round">{body}</svg>'
    )

ICON_CHEF = _icon([
    "M6 21h12M8 21v-6M16 21v-6",
    "M7.5 15h9a1 1 0 0 0 1-1.1c-.3-2.8-1-4-1-6.4a4.5 4.5 0 0 0-9 0c0 2.4-.7 3.6-1 6.4A1 1 0 0 0 7.5 15Z",
], size=26, width=1.7)
ICON_DOC = _icon(["M6 3h9l4 4v14H6z", "M15 3v4h4", "M9 12h6M9 16h6"])
ICON_LAYERS = _icon(["m12 3 9 5-9 5-9-5 9-5Z", "m3 13 9 5 9-5", "m3 8 9 5 9-5"])
ICON_SEARCH = _icon(["M11 19a8 8 0 1 0 0-16 8 8 0 0 0 0 16Z", "m21 21-4.3-4.3"])
ICON_COMPARE = _icon(["M8 3 4 7l4 4", "M4 7h11a4 4 0 0 1 4 4v1", "m16 21 4-4-4-4", "M20 17H9a4 4 0 0 1-4-4v-1"])
ICON_BOLT = _icon(["M13 2 4 14h7l-1 8 9-12h-7l1-8Z"])
ICON_LOOP = _icon(["M12 3a9 9 0 1 0 9 9", "M17 3v5h-5"])
ICON_FOLDER = _icon(["M4 6a1 1 0 0 1 1-1h4l2 2h8a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V6Z"])
ICON_GEAR = _icon([
    "M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z",
    "M19.4 13a1.7 1.7 0 0 0 .3 1.9l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.5V19a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1-1.6 1.7 1.7 0 0 0-1.9.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.9 1.7 1.7 0 0 0-1.5-1H5a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.6-1 1.7 1.7 0 0 0-.3-1.9l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.9.3H11a1.7 1.7 0 0 0 1-1.5V5a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.9-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.9V11a1.7 1.7 0 0 0 1.5 1H19a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1Z",
], width=1.5)
ICON_CHECK = _icon(["m5 13 4 4L19 7"], width=2.4)

st.markdown(
    """
    <style>
    :root {
        --bg-card: #131A2E;
        --border: rgba(255,255,255,0.08);
        --text-muted: #8B93AD;
        --blue: #3B82F6;
        --green: #22C55E;
        --amber: #F59E0B;
        --violet: #A855F7;
        --gradient: linear-gradient(135deg, #3B82F6 0%, #A855F7 100%);
    }

    .block-container { padding-top: 2rem; padding-bottom: 3rem; }

    /* ---------- hero ---------- */
    .hero {
        display: flex; align-items: center; gap: 1.1rem;
        background: linear-gradient(135deg, rgba(59,130,246,0.10) 0%, rgba(168,85,247,0.07) 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.6rem 1.9rem;
        margin-bottom: 1.2rem;
    }
    .hero-icon {
        width: 54px; height: 54px; border-radius: 14px; flex-shrink: 0;
        background: var(--gradient);
        display: flex; align-items: center; justify-content: center;
        color: #ffffff;
        box-shadow: 0 8px 22px rgba(139,92,246,0.35);
    }
    .hero-title { margin: 0 0 0.25rem 0; font-size: 1.85rem; font-weight: 700; line-height: 1.25; }
    .hero-title .accent {
        background: var(--gradient); -webkit-background-clip: text;
        background-clip: text; color: transparent;
    }
    .hero p { margin: 0; color: var(--text-muted); font-size: 0.96rem; }

    /* ---------- stat tiles ---------- */
    .stat-grid {
        display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.9rem;
        margin-bottom: 1.3rem;
    }
    .stat-tile {
        background: var(--bg-card); border: 1px solid var(--border);
        border-radius: 14px; padding: 1rem 1.1rem;
    }
    .stat-tile-icon {
        width: 32px; height: 32px; border-radius: 9px;
        display: flex; align-items: center; justify-content: center;
        margin-bottom: 0.6rem;
    }
    .stat-tile-icon.blue { background: rgba(59,130,246,0.16); color: var(--blue); }
    .stat-tile-icon.green { background: rgba(34,197,94,0.16); color: var(--green); }
    .stat-tile-icon.amber { background: rgba(245,158,11,0.16); color: var(--amber); }
    .stat-tile-icon.violet { background: rgba(168,85,247,0.16); color: var(--violet); }
    .stat-tile-label {
        font-size: 0.7rem; letter-spacing: 0.05em; text-transform: uppercase;
        color: var(--text-muted); margin-bottom: 0.2rem;
    }
    .stat-tile-value { font-size: 1.35rem; font-weight: 700; line-height: 1.2; }
    .stat-tile-sub { font-size: 0.76rem; color: var(--text-muted); margin-top: 0.15rem; }

    /* ---------- sidebar section headings ---------- */
    .sidebar-heading {
        display: flex; align-items: center; gap: 0.55rem; margin: 0 0 0.2rem 0;
    }
    .sidebar-heading .icon-badge {
        width: 26px; height: 26px; border-radius: 7px; flex-shrink: 0;
        background: rgba(139,92,246,0.16); color: var(--violet);
        display: flex; align-items: center; justify-content: center;
    }
    .sidebar-heading .titles { display: flex; flex-direction: column; gap: 0; }
    .sidebar-heading .titles .t { font-size: 0.92rem; font-weight: 600; }
    .sidebar-heading .titles .s { font-size: 0.72rem; color: var(--text-muted); }

    .index-badge {
        display: flex; align-items: center; gap: 0.5rem;
        background: var(--bg-card); border: 1px solid var(--border);
        border-radius: 10px; padding: 0.6rem 0.75rem; font-size: 0.8rem;
    }
    .index-badge .icon-badge {
        width: 22px; height: 22px; border-radius: 6px; flex-shrink: 0;
        background: rgba(34,197,94,0.16); color: var(--green);
        display: flex; align-items: center; justify-content: center;
    }

    /* ---------- agent vs chatbot mode cards ---------- */
    .mode-card {
        display: flex; gap: 0.7rem; align-items: flex-start;
        background: var(--bg-card); border: 1px solid var(--border);
        border-radius: 12px; padding: 0.7rem 0.85rem; margin-bottom: 0.55rem;
    }
    .mode-card .mode-icon {
        width: 30px; height: 30px; border-radius: 8px; flex-shrink: 0;
        display: flex; align-items: center; justify-content: center;
    }
    .mode-card.chatbot .mode-icon { background: rgba(59,130,246,0.16); color: var(--blue); }
    .mode-card.agent .mode-icon { background: rgba(168,85,247,0.16); color: var(--violet); }
    .mode-card .mode-title-row { display: flex; align-items: center; gap: 0.4rem; }
    .mode-card .mode-title { font-size: 0.85rem; font-weight: 600; }
    .mode-card.chatbot .mode-title { color: #7CADFF; }
    .mode-card.agent .mode-title { color: #C79BFF; }
    .mode-card .mode-tag {
        font-size: 0.6rem; font-weight: 700; letter-spacing: 0.03em;
        padding: 0.08rem 0.4rem; border-radius: 999px;
        background: rgba(34,197,94,0.18); color: #4ADE80;
    }
    .mode-card .mode-desc { font-size: 0.76rem; color: var(--text-muted); line-height: 1.45; margin-top: 0.15rem; }

    /* ---------- answer-card headers (chatbot / agent columns) ---------- */
    .answer-card-header { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.7rem; }
    .answer-card-header .icon-badge {
        width: 32px; height: 32px; border-radius: 9px; flex-shrink: 0;
        display: flex; align-items: center; justify-content: center;
    }
    .answer-card-header.chatbot .icon-badge { background: rgba(59,130,246,0.16); color: var(--blue); }
    .answer-card-header.agent .icon-badge { background: rgba(168,85,247,0.16); color: var(--violet); }
    .answer-card-header .t { font-size: 0.95rem; font-weight: 600; }
    .answer-card-header.chatbot .t { color: #7CADFF; }
    .answer-card-header.agent .t { color: #C79BFF; }
    .answer-card-header .s { font-size: 0.74rem; color: var(--text-muted); }

    /* ---------- source cards ---------- */
    .source-card {
        background: var(--bg-card); border: 1px solid var(--border);
        border-radius: 12px; padding: 0.8rem 1rem; margin-bottom: 0.65rem;
    }
    .source-card-head { display: flex; align-items: center; justify-content: space-between; gap: 0.6rem; margin-bottom: 0.35rem; }
    .source-card-title { display: flex; align-items: center; gap: 0.5rem; font-size: 0.84rem; font-weight: 600; min-width: 0; }
    .source-card-title span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .source-score {
        flex-shrink: 0; font-size: 0.7rem; font-weight: 700;
        padding: 0.14rem 0.55rem; border-radius: 999px;
        background: rgba(34,197,94,0.16); color: #4ADE80;
    }
    .source-card-body { font-size: 0.8rem; color: var(--text-muted); line-height: 1.5; margin-bottom: 0.35rem; }
    .source-card-meta { font-size: 0.7rem; color: #5B6480; }

    /* ---------- chunk explorer (sidebar) ---------- */
    .chunk-text {
        font-size: 0.78rem; color: var(--text); line-height: 1.5;
        white-space: pre-wrap; margin-bottom: 0.5rem;
    }
    .chunk-meta {
        display: flex; flex-direction: column; gap: 0.3rem;
        padding-top: 0.5rem; border-top: 1px solid var(--border);
    }
    .chunk-meta-row {
        display: flex; justify-content: space-between; gap: 0.6rem; font-size: 0.7rem;
    }
    .chunk-meta-row .k { color: var(--text-muted); flex-shrink: 0; }
    .chunk-meta-row .v { color: var(--text); text-align: right; word-break: break-word; }

    /* ---------- verdict ---------- */
    .verdict-card {
        background: var(--bg-card); border: 1px solid var(--border);
        border-radius: 10px; padding: 0.7rem 0.9rem; margin: 0.5rem 0;
    }
    .verdict-card.pass { border-left: 3px solid var(--green); }
    .verdict-card.fail { border-left: 3px solid #F87171; }
    .verdict-card.warn { border-left: 3px solid var(--amber); }

    /* ---------- native widget polish ---------- */
    div[data-testid="stExpander"] {
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
    }
    div[data-testid="stChatMessage"] {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.7rem;
    }
    button[kind="primary"] {
        background: var(--gradient) !important;
        border: none !important;
        box-shadow: 0 6px 18px rgba(139,92,246,0.35);
    }

    /* Wide markdown tables (e.g. an ingredient table in a recipe answer)
       must scroll within their own card, not overflow into the column
       next to it - the two-column chatbot/agent layout has no room for a
       table to render at its natural content width. */
    div[data-testid="stMarkdownContainer"] table {
        display: block;
        max-width: 100%;
        overflow-x: auto;
    }
    div[data-testid="stMarkdownContainer"] th,
    div[data-testid="stMarkdownContainer"] td {
        white-space: nowrap;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        overflow-x: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-icon">{ICON_CHEF}</div>
        <div>
            <div class="hero-title">RAG <span class="accent">Recipe Assistant</span></div>
            <p>Ask questions, retrieve the right recipes, and get grounded answers —
            compared side by side, chatbot vs. agent.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------- helpers

def is_recipe_collection(collection):
    """A collection is recipe-shaped if its chunks carry recipe_id metadata."""

    sample = collection.get(limit=1, include=["metadatas"])
    metadatas = sample.get("metadatas") or []
    return bool(metadatas) and metadatas[0].get("recipe_id") is not None


@st.cache_data
def load_eval_questions():
    path = "week4_failing_questions.json"

    if not os.path.exists(path):
        return []

    with open(path, encoding="utf-8") as handle:
        return json.load(handle)["questions"]


def find_matching_eval_question(question_text, questions):
    target = normalise(question_text)
    return next((q for q in questions if normalise(q["question"]) == target), None)


VERDICT_STYLE = {
    "pass": ("✅", "PASS", "pass"),
    "wrong_document": ("🟥", "WRONG DOCUMENT", "fail"),
    "right_document_wrong_answer": ("🟨", "RIGHT DOCUMENT, WRONG ANSWER", "warn"),
}

# Week 7: dietary restriction words both methods can act on. Both the
# chatbot (as an up-front metadata filter) and the agent (as a
# check-and-retry step) run on every recipe question regardless of whether
# one of these is present - a question with no restriction here is itself a
# useful comparison case (does the agent still do anything useful when
# there is nothing to verify?), not a reason to skip either method.
RESTRICTION_KEYWORDS = [
    "vegan", "vegetarian", "dairy-free", "gluten-free",
    "egg-free", "nut-free", "non-vegetarian",
]


def detect_restriction(question):
    lowered = question.lower()
    return next((r for r in RESTRICTION_KEYWORDS if r in lowered), None)


def render_verdict(label, evidence):
    icon, name, css_class = VERDICT_STYLE.get(label, ("❔", label.upper(), "warn"))
    st.markdown(
        f'<div class="verdict-card {css_class}"><strong>{icon} {name}</strong>'
        f'<br><span style="color:var(--text-muted);">{evidence}</span></div>',
        unsafe_allow_html=True,
    )


def _relevance_pct(distance):
    """Distance -> a clamped 0-100 "match" percentage for the source-card
    badge. Hybrid distances are already 1 - normalised_rrf_score (bounded
    0..1); plain semantic distances aren't guaranteed to be, so this is a
    clamped, decorative signal - good enough for at-a-glance ranking, not a
    precise score."""

    return max(0, min(100, round((1 - distance) * 100)))


def render_sources(results):
    """Retrieved chunks as a collapsible 'Sources' block, styled as compact
    ranked cards (title + match-% badge + snippet + meta row)."""

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    with st.expander(f"📚 Sources ({len(ids)})"):
        for rank, (chunk_id, document, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances), start=1
        ):
            title = f"#{rank} · {chunk_id}"
            if metadata.get("recipe_id"):
                title += f" · {metadata['recipe_id']}"

            meta_bits = [f"distance {distance:.4f}"]
            if metadata.get("semantic_rank") is not None or metadata.get("keyword_rank") is not None:
                meta_bits.append(f"semantic rank {metadata.get('semantic_rank')}")
                meta_bits.append(f"keyword rank {metadata.get('keyword_rank')}")
                meta_bits.append(f"RRF score {metadata.get('rrf_score')}")

            snippet = document if len(document) <= 320 else document[:320].rsplit(" ", 1)[0] + "…"

            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-card-head">
                        <div class="source-card-title">{ICON_DOC}<span>{html.escape(title)}</span></div>
                        <div class="source-score">{_relevance_pct(distance)}% match</div>
                    </div>
                    <div class="source-card-body">{html.escape(snippet)}</div>
                    <div class="source-card-meta">{" · ".join(meta_bits)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_chunk_metadata(metadata):
    """Every stored metadata field for one chunk, as plain key/value rows -
    the raw ingest-time output (rag/vector_store.py's attach_metadata /
    diet_flags), not a curated subset, since the point of this view is
    debugging what actually got stored."""

    rows = "".join(
        f'<div class="chunk-meta-row"><span class="k">{html.escape(str(key))}</span>'
        f'<span class="v">{html.escape(str(value))}</span></div>'
        for key, value in sorted(metadata.items())
        if value not in (None, "")
    )
    st.markdown(f'<div class="chunk-meta">{rows}</div>', unsafe_allow_html=True)


def render_chunk_explorer(collection, recipe_shaped):
    """Sidebar tool: pick one indexed document, see every chunk that came
    from it - its full text and its full stored metadata - without leaving
    the app. Read-only, purely for inspecting how chunking/ingest actually
    shaped the data behind a given source file."""

    doc_field = "source_file" if recipe_shaped else "source"

    sample = collection.get(include=["metadatas"])
    docs = sorted({m.get(doc_field) for m in sample["metadatas"] if m.get(doc_field)})

    if not docs:
        st.caption("No documents indexed yet.")
        return

    selected_doc = st.selectbox("Document", docs, key="chunk_explorer_doc")

    rows = collection.get(where={doc_field: selected_doc}, include=["documents", "metadatas"])
    ordered = sorted(zip(rows["ids"], rows["documents"], rows["metadatas"]), key=lambda r: r[0])

    st.caption(f"{len(ordered)} chunk(s) from this document")

    for rank, (chunk_id, document, metadata) in enumerate(ordered, start=1):
        with st.expander(f"{rank} · {chunk_id}"):
            st.markdown(f'<div class="chunk-text">{html.escape(document)}</div>', unsafe_allow_html=True)
            render_chunk_metadata(metadata)


def _step_count(n):
    return f"{n} step" if n == 1 else f"{n} steps"


# Human-readable labels for rag.agent.run_agent()'s internal stopped_reason
# codes - a client-facing trace should never show raw snake_case state.
STOPPED_REASON_LABEL = {
    "finished": "completed",
    "max_steps_exceeded": "stopped at its step limit",
    "time_budget_exceeded": "stopped at its time limit",
    "planner_parse_error": "stopped after a planning error",
}


def render_agent_trace(steps, stopped_reason):
    """The agent's plan -> act -> observe steps as a collapsible block -
    "every step visible" as a first-class part of the chat UI, not just the
    console log the race script writes."""

    status = STOPPED_REASON_LABEL.get(stopped_reason, stopped_reason)

    with st.expander(f"🤖 How the agent got here — {_step_count(len(steps))}, {status}"):
        for step in steps:
            st.markdown(f"**Step {step['step']} · `{step['tool']}`**")
            if step.get("thought"):
                st.caption(step["thought"])
            st.json({"args": step["args"], "observation": step["observation"]})
            st.divider()


def build_workflow_trace(prompt, retrieval_mode, top_k, results, generation_tool, where=None):
    """The fixed workflow's own steps, in the SAME {step, tool, args,
    observation} shape render_agent_trace expects - so a viewer can compare
    the two traces directly instead of one path having visible steps and
    the other only a chunk dump. Built after the fact from the `results`
    the fixed path already computed - no behaviour changes, just logging
    what already happened.

    Unlike the agent's steps, there is no "thought" here: nothing decided
    to search or to generate, the sequence is hardcoded - that absence is
    itself the point of comparison.
    """

    ids = results["ids"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    search_tool = "retrieve_hybrid" if retrieval_mode.startswith("Hybrid") else "retrieve"

    candidates = []
    for chunk_id, metadata, distance in zip(ids, metadatas, distances):
        entry = {"chunk_id": chunk_id, "distance": round(float(distance), 4)}
        if metadata.get("recipe_id"):
            entry["recipe_id"] = metadata["recipe_id"]
            entry["recipe_name"] = metadata.get("recipe_name")
        else:
            entry["source"] = metadata.get("source")
            entry["page"] = metadata.get("page")
        candidates.append(entry)

    search_args = {"query": prompt, "top_k": top_k}
    if where:
        search_args["where"] = where

    return [
        {
            "step": 1,
            "tool": search_tool,
            "args": search_args,
            "observation": candidates,
        },
        {
            "step": 2,
            "tool": generation_tool,
            "args": {},
            "observation": "answer generated from the retrieved chunks above",
        },
    ]


def render_workflow_trace(steps):
    """The fixed workflow's hardcoded steps, rendered like render_agent_trace
    so the two are easy to compare side by side in the chat."""

    with st.expander(f"⚡ How the chatbot got here — {_step_count(len(steps))}, fixed sequence"):
        for step in steps:
            st.markdown(f"**Step {step['step']} · `{step['tool']}`**")
            st.json({"args": step["args"], "observation": step["observation"]})
            st.divider()


def run_fixed_path(collection, prompt, retrieval_mode, top_k, recipe_shaped, restriction=None):
    """The one-shot retrieve-then-generate path - the CHATBOT baseline this
    app compares the agent loop (rag/agent.py) against. Timed the same way
    race_agent_vs_workflow.py times both methods (wall-clock +
    generator.call_log delta), so its numbers are comparable to whatever
    run_agent() just reported - not just its steps, but its cost.

    When a dietary restriction is detected, it is applied as a metadata
    filter UP FRONT, before the one search call - mirroring
    rag/fixed_workflow.py's fixed sequence (filter -> search -> generate,
    decided in advance by the code, never by the model) rather than the
    agent's search-then-verify-then-maybe-retry loop. That's the one
    difference the side-by-side comparison in the UI is meant to show."""

    calls_before = len(generator.call_log)
    start = time.monotonic()

    where = restriction_where(restriction) if recipe_shaped else None

    if retrieval_mode.startswith("Hybrid"):
        results = retrieve_hybrid(collection, prompt, top_k, where=where)
    else:
        results = retrieve(collection, prompt, top_k, where=where)

    generation_tool = "generate_recipe_answer" if recipe_shaped else "generate_answer"
    answer = (
        generate_recipe_answer(prompt, results)
        if recipe_shaped else generate_answer(prompt, results)
    )

    workflow_steps = build_workflow_trace(prompt, retrieval_mode, top_k, results, generation_tool, where)

    return {
        "answer": answer,
        "results": results,
        "workflow_steps": workflow_steps,
        "elapsed_seconds": time.monotonic() - start,
        "llm_calls": len(generator.call_log) - calls_before,
    }


def _call_count(n):
    return f"{n} call" if n == 1 else f"{n} calls"


def render_answer_card_header(kind, icon, title, subtitle):
    st.markdown(
        f"""
        <div class="answer-card-header {kind}">
            <div class="icon-badge">{icon}</div>
            <div><div class="t">{title}</div><div class="s">{subtitle}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_race_summary(agent_result, fixed_result):
    """One-line cost/speed comparison, live in the chat - the same signal
    race_agent_vs_workflow.py writes to a report file, surfaced immediately
    instead of requiring a separate script run."""

    st.caption(
        f"🏁 **🤖 Agent**: {agent_result['elapsed_seconds']:.1f}s · "
        f"{_call_count(agent_result['llm_calls'])}   "
        f"vs.   **⚡ Chatbot**: {fixed_result['elapsed_seconds']:.1f}s · "
        f"{_call_count(fixed_result['llm_calls'])}"
    )


def render_stat_tiles(collection, retrieval_mode, recipe_shaped):
    """A glanceable summary row above the chat - live counts from the
    selected collection plus the current retrieval/comparison settings, so
    the state of the workspace is visible before asking anything."""

    total_chunks = collection.count()

    doc_field = "recipe_id" if recipe_shaped else "source"
    sample = collection.get(include=["metadatas"])
    distinct_docs = len({m.get(doc_field) for m in sample["metadatas"] if m.get(doc_field)})

    mode_label = "Hybrid" if retrieval_mode.startswith("Hybrid") else "Semantic"
    mode_sub = "Semantic + keyword" if mode_label == "Hybrid" else "Embedding similarity"

    compare_label = "Agent + Chatbot" if recipe_shaped else "Chatbot only"
    compare_sub = "Both run automatically" if recipe_shaped else "Agent needs recipe data"

    st.markdown(
        f"""
        <div class="stat-grid">
          <div class="stat-tile">
            <div class="stat-tile-icon blue">{ICON_DOC}</div>
            <div class="stat-tile-label">Documents</div>
            <div class="stat-tile-value">{distinct_docs}</div>
            <div class="stat-tile-sub">Indexed source files</div>
          </div>
          <div class="stat-tile">
            <div class="stat-tile-icon green">{ICON_LAYERS}</div>
            <div class="stat-tile-label">Chunks</div>
            <div class="stat-tile-value">{total_chunks}</div>
            <div class="stat-tile-sub">Searchable text chunks</div>
          </div>
          <div class="stat-tile">
            <div class="stat-tile-icon amber">{ICON_SEARCH}</div>
            <div class="stat-tile-label">Retrieval Mode</div>
            <div class="stat-tile-value">{mode_label}</div>
            <div class="stat-tile-sub">{mode_sub}</div>
          </div>
          <div class="stat-tile">
            <div class="stat-tile-icon violet">{ICON_COMPARE}</div>
            <div class="stat-tile-label">Comparison</div>
            <div class="stat-tile-value">{compare_label}</div>
            <div class="stat-tile-sub">{compare_sub}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_heading(icon, title, subtitle):
    st.markdown(
        f"""
        <div class="sidebar-heading">
            <div class="icon-badge">{icon}</div>
            <div class="titles"><div class="t">{title}</div><div class="s">{subtitle}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------- sidebar

with st.sidebar:

    sidebar_heading(ICON_FOLDER, "Knowledge Base", "Manage and configure your documents")
    st.write("")

    uploaded_files = st.file_uploader(
        "Upload amendment PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    chunk_size = st.slider("Chunk size", min_value=100, max_value=1000, value=500, step=100)
    overlap = st.slider("Chunk overlap", min_value=0, max_value=300, value=100, step=50)

    build_button = st.button("Build Index", type="primary", width="stretch")

    if st.session_state.get("indexed"):
        st.markdown(
            f"""
            <div class="index-badge">
                <div class="icon-badge">{ICON_CHECK}</div>
                <span>Index ready — {st.session_state.get('chunk_count', 0)} chunks</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    sidebar_heading(ICON_SEARCH, "Retrieval", "How to search your knowledge base")
    st.write("")

    collections = list_collections()

    if collections:
        default_collection = (
            "legal_contracts" if "legal_contracts" in collections else collections[0]
        )
        collection_name = st.selectbox(
            "Collection", collections, index=collections.index(default_collection)
        )
    else:
        collection_name = None
        st.caption("No collections yet — build an index above.")

    retrieval_mode = st.radio(
        "Retrieval mode",
        ["Semantic only", "Hybrid (semantic + keyword)"],
    )

    top_k = st.slider("Top-K", min_value=1, max_value=10, value=5)

    st.divider()

    sidebar_heading(ICON_COMPARE, "Agent vs. Chatbot", "Two ways to answer, every time")
    st.write("")

    st.markdown(
        f"""
        <div class="mode-card chatbot">
            <div class="mode-icon">{ICON_BOLT}</div>
            <div>
                <div class="mode-title-row"><span class="mode-title">Chatbot</span><span class="mode-tag">ALWAYS RUNS</span></div>
                <div class="mode-desc">One fixed search, then answer. Same steps, every time.</div>
            </div>
        </div>
        <div class="mode-card agent">
            <div class="mode-icon">{ICON_LOOP}</div>
            <div>
                <div class="mode-title-row"><span class="mode-title">Agent</span><span class="mode-tag">ALWAYS RUNS</span></div>
                <div class="mode-desc">Decides its own next step — searches again, verifies, retries.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Recipe questions run both, side by side. Open each answer's trace to see exactly how they got there.")

    if collection_name:
        st.divider()

        sidebar_heading(ICON_LAYERS, "Chunk Explorer", "Inspect chunks & metadata by document")
        st.write("")

        _explorer_collection = get_collection(collection_name)
        render_chunk_explorer(_explorer_collection, is_recipe_collection(_explorer_collection))


if collection_name:
    _stats_collection = get_collection(collection_name)
    render_stat_tiles(_stats_collection, retrieval_mode, is_recipe_collection(_stats_collection))


if build_button:

    if not uploaded_files:
        st.warning("Please upload at least one PDF.")

    elif overlap >= chunk_size:
        st.error("Overlap must be smaller than chunk size.")

    else:
        with st.spinner("Chunking documents and building the index..."):

            all_pages = []

            for uploaded_file in uploaded_files:

                os.makedirs("documents", exist_ok=True)

                path = os.path.join("documents", uploaded_file.name)

                with open(path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                pages = load_pdf(path)
                all_pages.extend(pages)

            chunks = create_chunks(all_pages, chunk_size, overlap)

            build_index(chunks, "legal_contracts")

            st.session_state.indexed = True
            st.session_state.chunk_count = len(chunks)

        st.success(f"✅ Indexed {len(chunks)} chunks from {len(uploaded_files)} document(s).")
        st.rerun()


# --------------------------------------------------------------- chat

if "messages" not in st.session_state:
    st.session_state.messages = []

def render_assistant_message(message):
    """Shared rendering for both a fresh answer and a replayed one, so the
    two code paths can never drift apart on what gets shown."""

    agent_result = message.get("agent_result")
    fixed_result = message.get("fixed_result")

    if agent_result and fixed_result:
        # Recipe question: both methods ran on the SAME input, side by side -
        # so "this is the chatbot, this is the agent loop" is visible
        # directly under the chat box, not buried in a collapsed expander.
        render_race_summary(agent_result, fixed_result)

        col_chatbot, col_agent = st.columns(2)

        with col_chatbot:
            with st.container(border=True):
                render_answer_card_header("chatbot", ICON_BOLT, "Chatbot", "Fixed workflow")
                st.markdown(fixed_result["answer"])
                if message.get("verdict"):
                    render_verdict(message["verdict"]["label"], message["verdict"]["evidence"])
                render_workflow_trace(fixed_result["workflow_steps"])
                render_sources(fixed_result["results"])

        with col_agent:
            with st.container(border=True):
                render_answer_card_header("agent", ICON_LOOP, "Agent", "Decided its own steps")
                st.markdown(agent_result["answer"])
                render_agent_trace(agent_result["steps"], agent_result["stopped_reason"])

    else:
        # Non-recipe collection (e.g. legal_contracts): the agent's tools
        # (rag/tools.py) are recipe-specific, so only the chatbot path runs.
        st.markdown(message["content"])

        if message.get("verdict"):
            render_verdict(message["verdict"]["label"], message["verdict"]["evidence"])

        if fixed_result:
            render_workflow_trace(fixed_result["workflow_steps"])
            render_sources(fixed_result["results"])


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_assistant_message(message)
        else:
            st.markdown(message["content"])

prompt = st.chat_input("Ask a question about your documents...")

if prompt:

    if not collection_name:
        st.warning("Build an index first (see the sidebar).")

    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):

            collection = get_collection(collection_name)
            recipe_shaped = is_recipe_collection(collection)
            restriction = detect_restriction(prompt) if recipe_shaped else None

            verdict = None
            agent_result = None
            fixed_result = None

            if recipe_shaped:

                spinner_msg = (
                    f"Running both methods - chatbot vs. agent verifying '{restriction}'..."
                    if restriction else
                    "Running both methods - chatbot vs. agent loop..."
                )
                with st.spinner(spinner_msg):
                    fixed_result = run_fixed_path(
                        collection, prompt, retrieval_mode, top_k, recipe_shaped, restriction
                    )
                    agent_result = run_agent(collection, prompt, restriction)

                answer = agent_result["answer"]

                eval_match = find_matching_eval_question(prompt, load_eval_questions())
                if eval_match:
                    rows = rows_from_results(fixed_result["results"])
                    verdict = classify(eval_match, rows, fixed_result["answer"], k=top_k)

            else:

                with st.spinner("Retrieving and generating an answer..."):
                    fixed_result = run_fixed_path(
                        collection, prompt, retrieval_mode, top_k, recipe_shaped
                    )

                answer = fixed_result["answer"]

            message = {
                "role": "assistant",
                "content": answer,
                "verdict": verdict,
                "agent_result": agent_result,
                "fixed_result": fixed_result,
            }
            render_assistant_message(message)

        st.session_state.messages.append(message)
