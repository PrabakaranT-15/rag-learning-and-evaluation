import chromadb
from rag.embeddings import create_embedding


chroma_client = chromadb.PersistentClient(path="./data/chroma")


# ---------------------------------------------------------------- existing API
# build_index() is UNCHANGED and still used by the legal-contract Streamlit app.
# It clears the collection before writing, which is why the Task Set B ingest
# path below uses add_chunks() instead: requirement 6 forbids re-indexing the
# existing corpus.

def build_index(chunks, collection_name):
    """Build (or rebuild) a ChromaDB collection from the given chunks."""

    collection = chroma_client.get_or_create_collection(
        name=collection_name
    )

    # Clear existing collection
    existing = collection.get()

    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    for chunk in chunks:

        embedding = create_embedding(chunk["text"])

        collection.add(
            ids=[chunk["id"]],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[{
                "source": chunk["source"],
                "page": chunk["page"]
            }]
        )

    return collection


def get_collection(collection_name):
    """Get an existing ChromaDB collection by name."""

    return chroma_client.get_collection(collection_name)


# ------------------------------------------------------- Task Set B additions

REQUIRED_METADATA = ["source_file", "recipe_id", "cuisine", "dietary_tags"]


def diet_flags(dietary_tags):
    """Derive boolean flags from the dietary_tags string.

    ChromaDB's where-filter has no substring operator, so a comma-joined
    dietary_tags string cannot be filtered directly. These booleans are derived
    from that exact string at ingest time so dietary filtering is still possible
    without losing the required dietary_tags field itself.
    """

    tags = [t.strip() for t in dietary_tags.split(",") if t.strip()]

    return {f"diet_{t.replace('-', '_')}": True for t in tags}


def validate_chunk_metadata(chunks):
    """Fail the ingest loudly if any chunk is missing required metadata.

    Task Set B: 'A chunk with no source_file is a failed ingest.'
    """

    problems = []

    for chunk in chunks:

        metadata = chunk.get("metadata", {})

        for field in REQUIRED_METADATA:
            if not metadata.get(field):
                problems.append(
                    f"chunk {chunk.get('id', '<no id>')} missing '{field}'"
                )

    if problems:
        raise ValueError(
            "INGEST ABORTED - {} chunk(s) failed metadata validation:\n  {}".format(
                len(problems), "\n  ".join(problems[:20])
            )
        )

    return True


def attach_metadata(chunks, metadata_by_source, strategy):
    """Attach recipe metadata + diet flags to chunks, keyed by source file.

    The baseline chunker only carries {source, page}, so recipe_id / cuisine /
    dietary_tags are re-attached here by looking the source file back up. This
    is what lets Strategy A stay byte-for-byte unchanged while still satisfying
    the metadata requirement.
    """

    enriched = []

    for chunk in chunks:

        base = dict(metadata_by_source[chunk["source"]])

        base["strategy"] = strategy
        base["page"] = chunk.get("page", 1)
        base["section"] = chunk.get("section", "")
        base["block_type"] = chunk.get("block_type", "window")
        base["source"] = chunk["source"]          # kept for app.py compatibility
        base.update(diet_flags(base["dietary_tags"]))

        enriched.append({
            "id": chunk["id"],
            "text": chunk["text"],
            "metadata": base,
        })

    return enriched


def add_chunks(chunks, collection_name):
    """APPEND chunks to a collection without deleting anything already in it.

    Unlike build_index(), this never clears the collection. Existing vectors
    (for example the legal_contracts corpus, or previously ingested recipe
    cards) are left untouched.
    """

    validate_chunk_metadata(chunks)

    collection = chroma_client.get_or_create_collection(name=collection_name)

    before = collection.count()

    # Skip ids already present so a rate-limited run can be resumed without
    # paying for the same embeddings twice.
    existing_ids = set(collection.get(include=[])["ids"])

    todo = [c for c in chunks if c["id"] not in existing_ids]

    if len(todo) < len(chunks):
        print(f"  resuming: {len(chunks) - len(todo)} chunks already embedded, "
              f"{len(todo)} remaining")

    for chunk in todo:

        embedding = create_embedding(chunk["text"])

        collection.upsert(
            ids=[chunk["id"]],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[chunk["metadata"]],
        )

    after = collection.count()

    return {
        "collection": collection,
        "name": collection_name,
        "count_before": before,
        "count_after": after,
        "added": after - before,
    }


def retrieve(collection, question, top_k, where=None):
    """Query the collection, optionally with a metadata filter.

    The `where` argument is new; existing callers that omit it behave exactly
    as before.
    """

    query_embedding = create_embedding(question)

    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }

    if where:
        kwargs["where"] = where

    return collection.query(**kwargs)


def list_collections():
    """Names of every collection currently in the store."""

    return [
        c.name if hasattr(c, "name") else c
        for c in chroma_client.list_collections()
    ]
