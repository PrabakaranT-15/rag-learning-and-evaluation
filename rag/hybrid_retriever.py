"""Week 4: hybrid search (Chroma keyword search + existing semantic search, fused with RRF).

WHY THIS EXISTS
----------------
results.md section 6 ("the retrieval that embarrassed me") diagnosed a real
failure mode of the shipped structure-aware chunker: a short, keyword-dense
chunk from the WRONG recipe can out-rank the correct chunk on pure embedding
distance, when the query names terms ("salted shrimp", "anchovy fish sauce")
that occur densely in a *different* card. That is a semantic-search failure,
not a chunk-boundary or metadata bug.

Hybrid search adds a second, independent ranking signal - exact keyword
overlap - and fuses it with the existing semantic ranking using Reciprocal
Rank Fusion (RRF). It is the single change measured by
week4_failure_analysis.py. It is NOT expected to fix every failure: a short
chunk that is lexically dense in the query's rare terms can still win on
keyword overlap too. That is reported honestly rather than hidden - see
week4_report.md's "not fixed by this change" section.

KEYWORD SIDE: CHROMA'S BUILT-IN FULL-TEXT FILTER
-------------------------------------------------
This used to run its own BM25 index (rank_bm25.BM25Okapi) over a full,
locally-cached copy of every document in the collection. It now uses
Chroma's own document filter (`where_document={"$contains": ...}`) to ask
Chroma itself which documents contain the query's terms, instead of
maintaining a second, duplicate copy of the corpus.

Chroma's newer `collection.search()` hybrid-ranking API would fuse semantic
+ keyword scores server-side, but it is explicitly experimental and raises
NotImplementedError for the local/PersistentClient backend this app uses
(SDK 1.5.9) - only the distributed/hosted backend implements it. `$contains`
is a plain substring filter with no relevance score, and it is case-sensitive
(it matches raw stored text, not lowercased text), so it is used here only to
ask Chroma for the *candidate* documents containing any case-variant of a
query token; the candidates are then ranked in Python by case-insensitive
term-occurrence count. That count-based ranking - not the RRF fusion - is
the part standing in for BM25's job. RRF fusion against the semantic ranking
below is unchanged.

rag/vector_store.py and retrieve() are UNCHANGED. This module only adds a
new, optional retrieval path; every existing caller is unaffected.
"""

import re

from rag.vector_store import retrieve as semantic_retrieve


RRF_K = 60  # standard RRF constant - de-emphasises the exact top rank a bit
             # so a single lucky top-1 hit in one ranker can't dominate fusion.

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Cap how many distinct query tokens feed the keyword filter, so a long
# question doesn't blow up the $or clause sent to Chroma.
MAX_KEYWORD_TOKENS = 20


def tokenize(text):
    """Lowercase alnum-run tokenizer, used both for the keyword filter's
    query tokens and for scoring candidate documents case-insensitively."""

    return _TOKEN_RE.findall(text.lower())


def _keyword_candidates(collection, tokens, where=None):
    """Ask Chroma's own document filter for candidates containing any query
    token, in any of a few common casings.

    `$contains` is a case-sensitive substring test on the raw stored text,
    so each token is offered to Chroma as lower/upper/capitalised variants
    or-ed together. This only narrows the candidate set - final ranking is
    done in Python from the (small) set of documents this returns.
    """

    if not tokens:
        return {"ids": [], "documents": [], "metadatas": []}

    or_clauses = [
        {"$contains": variant}
        for token in tokens[:MAX_KEYWORD_TOKENS]
        for variant in {token, token.capitalize(), token.upper()}
    ]

    where_document = {"$or": or_clauses} if len(or_clauses) > 1 else or_clauses[0]

    kwargs = {"where_document": where_document, "include": ["documents", "metadatas"]}
    if where:
        kwargs["where"] = where

    return collection.get(**kwargs)


def _rank_by_term_overlap(tokens, ids, documents):
    """Rank candidate documents by case-insensitive query-token occurrence
    count - the simple frequency signal standing in for BM25 here."""

    scored = []

    for doc_id, doc in zip(ids, documents):
        doc_lower = doc.lower()
        score = sum(doc_lower.count(token) for token in tokens)
        if score > 0:
            scored.append((doc_id, score))

    scored.sort(key=lambda pair: pair[1], reverse=True)

    return [doc_id for doc_id, _ in scored]


def _rrf_scores(ranked_id_lists, k=RRF_K):
    """Reciprocal Rank Fusion over one or more ranked id lists.

    score(id) = sum over lists containing id of 1 / (k + rank_in_that_list)
    rank is 1-based. An id absent from a list simply contributes 0 from it.
    """

    scores = {}

    for ranked_ids in ranked_id_lists:
        for rank, doc_id in enumerate(ranked_ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    return scores


def retrieve_hybrid(collection, question, top_k, candidate_pool=15, where=None):
    """Hybrid retrieval: semantic search + Chroma keyword search, fused by RRF.

    Returns a result shaped exactly like vector_store.retrieve()'s output
    (ids/documents/metadatas/distances, each wrapped in a one-element outer
    list to match Chroma's query() shape) so every existing caller
    (generator.py, the debug page) can consume it without changes. The
    "distance" field here is 1 - normalised_rrf_score (so smaller is still
    "closer", consistent with Chroma's convention) and each row's metadata
    additionally carries semantic_rank / keyword_rank / rrf_score for
    inspection.
    """

    pool = max(candidate_pool, top_k)

    # ---- semantic candidate pool (existing, unmodified retrieval path) ----
    semantic_results = semantic_retrieve(collection, question, pool, where=where)

    semantic_ids = semantic_results["ids"][0]
    semantic_rank_of = {doc_id: rank for rank, doc_id in enumerate(semantic_ids, start=1)}

    # ---- keyword candidate pool, via Chroma's own document filter ----
    # `where` is passed straight into collection.get() alongside
    # where_document, so Chroma applies the same metadata filter natively -
    # no need to separately re-derive an allowed id set.
    tokens = tokenize(question)
    keyword_result = _keyword_candidates(collection, tokens, where=where)
    keyword_ids = _rank_by_term_overlap(
        tokens, keyword_result["ids"], keyword_result["documents"]
    )[:pool]
    keyword_rank_of = {doc_id: rank for rank, doc_id in enumerate(keyword_ids, start=1)}

    # ---- fuse ----
    fused_scores = _rrf_scores([semantic_ids, keyword_ids])

    fused_ids = sorted(fused_scores, key=lambda doc_id: fused_scores[doc_id], reverse=True)[:top_k]

    if not fused_ids:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    max_score = fused_scores[fused_ids[0]] or 1.0

    by_id = {}
    for doc_id, doc, meta in zip(
        keyword_result["ids"], keyword_result["documents"], keyword_result["metadatas"]
    ):
        by_id[doc_id] = (doc, meta)
    for doc_id, doc, meta in zip(
        semantic_ids, semantic_results["documents"][0], semantic_results["metadatas"][0]
    ):
        by_id.setdefault(doc_id, (doc, meta))

    documents, metadatas, distances = [], [], []

    for doc_id in fused_ids:
        doc, meta = by_id[doc_id]

        rrf_score = fused_scores[doc_id]

        enriched_meta = dict(meta)
        enriched_meta["semantic_rank"] = semantic_rank_of.get(doc_id)
        enriched_meta["keyword_rank"] = keyword_rank_of.get(doc_id)
        enriched_meta["rrf_score"] = round(rrf_score, 6)

        documents.append(doc)
        metadatas.append(enriched_meta)
        distances.append(round(1.0 - (rrf_score / max_score), 6))

    return {
        "ids": [fused_ids],
        "documents": [documents],
        "metadatas": [metadatas],
        "distances": [distances],
    }
