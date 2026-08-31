def chunk_text(text, chunk_size, overlap):
    """  overlap ahagurathukku use pandrathu """

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def create_chunks(pages, chunk_size, overlap):
    """chunk create pannum """

    all_chunks = []

    for page in pages:

        chunks = chunk_text(
            page["text"],
            chunk_size,
            overlap
        )

        for i, chunk in enumerate(chunks):

            all_chunks.append({
                "id": f"{page['source']}_{page['page']}_{i}",
                "text": chunk,
                "page": page["page"],
                "source": page["source"]
            })

    return all_chunks
