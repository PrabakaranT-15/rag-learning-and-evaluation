import json
import os

import streamlit as st

from rag.pdf_loader import load_pdf
from rag.chunker import create_chunks
from rag.vector_store import build_index, retrieve, get_collection, list_collections
from rag.hybrid_retriever import retrieve_hybrid
from rag.generator import generate_answer, generate_recipe_answer
from rag.evaluation import classify, rows_from_results, normalise


st.set_page_config(
    page_title="Ask Anything About This Document",
    page_icon="📄",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }

    .hero {
        background: linear-gradient(135deg, #FBEADD 0%, #F7F1E8 100%);
        border: 1px solid #F0DCC4;
        border-radius: 14px;
        padding: 1.6rem 1.9rem;
        margin-bottom: 1.4rem;
    }
    .hero h1 { margin: 0 0 0.3rem 0; font-size: 1.9rem; }
    .hero p { margin: 0; color: #6B5B4D; font-size: 0.98rem; }

    .verdict-card {
        background: #FFFFFF;
        border: 1px solid #EDE3D6;
        border-radius: 10px;
        padding: 0.7rem 0.9rem;
        margin: 0.5rem 0;
    }
    .verdict-card.pass { border-left: 5px solid #2E9E5B; }
    .verdict-card.fail { border-left: 5px solid #D64545; }
    .verdict-card.warn { border-left: 5px solid #DB6B2C; }

    div[data-testid="stExpander"] {
        border: 1px solid #EDE3D6 !important;
        border-radius: 10px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }

    div[data-testid="stChatMessage"] {
        background: #FFFFFF;
        border: 1px solid #EDE3D6;
        border-radius: 14px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.7rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>📄 Ask Anything About This Document</h1>
        <p>Upload documents, build a searchable index, and chat with them — answers
        are grounded only in your own documents.</p>
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


def render_verdict(label, evidence):
    icon, name, css_class = VERDICT_STYLE.get(label, ("❔", label.upper(), "warn"))
    st.markdown(
        f'<div class="verdict-card {css_class}"><strong>{icon} {name}</strong>'
        f'<br><span style="color:#6B5B4D;">{evidence}</span></div>',
        unsafe_allow_html=True,
    )


def render_sources(results):
    """Retrieved chunks as a collapsible 'Sources' block, ChatGPT/Claude-style."""

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    with st.expander(f"📚 Sources ({len(ids)})"):
        for rank, (chunk_id, document, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances), start=1
        ):
            label = f"#{rank}  {chunk_id}"
            if metadata.get("recipe_id"):
                label += f"   ·   recipe_id={metadata['recipe_id']}"

            st.markdown(f"**{label}**")

            if metadata.get("semantic_rank") is not None or metadata.get("keyword_rank") is not None:
                st.markdown(
                    f":blue-badge[semantic rank {metadata.get('semantic_rank')}]  "
                    f":violet-badge[keyword rank {metadata.get('keyword_rank')}]  "
                    f":gray-badge[RRF score {metadata.get('rrf_score')}]"
                )

            st.write(document)
            st.markdown(f":gray-badge[distance / score {distance:.4f}]")
            st.divider()


# --------------------------------------------------------------- sidebar

with st.sidebar:

    st.markdown("### 📁 Document Settings")

    uploaded_files = st.file_uploader(
        "Upload amendment PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    chunk_size = st.slider("Chunk size", min_value=100, max_value=1000, value=500, step=100)
    overlap = st.slider("Chunk overlap", min_value=0, max_value=300, value=100, step=50)

    build_button = st.button("🔨 Build Index", type="primary", width="stretch")

    if st.session_state.get("indexed"):
        st.caption(f"✅ Index ready — {st.session_state.get('chunk_count', 0)} chunks")

    st.divider()

    st.markdown("### 🔎 Retrieval Settings")

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

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            if message.get("verdict"):
                render_verdict(message["verdict"]["label"], message["verdict"]["evidence"])
            if message.get("results"):
                render_sources(message["results"])

prompt = st.chat_input("Ask a question about your documents...")

if prompt:

    if not collection_name:
        st.warning("Build an index first (see the sidebar).")

    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating an answer..."):

                collection = get_collection(collection_name)
                recipe_shaped = is_recipe_collection(collection)

                if retrieval_mode.startswith("Hybrid"):
                    results = retrieve_hybrid(collection, prompt, top_k)
                else:
                    results = retrieve(collection, prompt, top_k)

                answer = (
                    generate_recipe_answer(prompt, results)
                    if recipe_shaped else generate_answer(prompt, results)
                )

                verdict = None
                if recipe_shaped:
                    eval_match = find_matching_eval_question(prompt, load_eval_questions())
                    if eval_match:
                        rows = rows_from_results(results)
                        verdict = classify(eval_match, rows, answer, k=top_k)

            st.markdown(answer)
            if verdict:
                render_verdict(verdict["label"], verdict["evidence"])
            render_sources(results)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "results": results,
            "verdict": verdict,
        })
