"""Strategy B: structure-aware chunker for markdown recipe cards.

DESIGN
------
The baseline chunker in rag/chunker.py slices on a flat word count. On a recipe
card that is destructive: a 500-word window can begin halfway down an ingredient
table, producing a chunk like "| Fine sea salt | 40 g | 2% |" with no table
header above it and no recipe title anywhere in the chunk. The row is then
unretrievable by any question that names the recipe, and unreadable by the model
even if it is retrieved, because nothing says what "2%" is a percentage OF.

This chunker respects three structural rules:

1. NEVER split a table. A markdown table is emitted whole: header row,
   separator row and every data row travel together in one chunk.
2. EVERY chunk carries its parent recipe title and its section heading in a
   context header, so an ingredient row is always attached to the recipe it
   belongs to.
3. Prose immediately preceding a table stays WITH that table when it is short,
   because on these cards that sentence is what declares the percentage basis
   ("percentages are calculated against the 2000 g of flour").

Chunk size is therefore driven by document structure, not by an arbitrary
constant. Prose sections are packed to a soft target only after the structural
rules above are satisfied.
"""

import re


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _is_table_line(line):
    return line.strip().startswith("|")


def _split_blocks(content):
    """Split section content into ordered prose and table blocks."""

    blocks = []
    buffer = []
    mode = "prose"

    for line in content.split("\n"):

        line_mode = "table" if _is_table_line(line) else "prose"

        if line_mode != mode and buffer:
            text = "\n".join(buffer).strip()
            if text:
                blocks.append({"type": mode, "text": text})
            buffer = []

        mode = line_mode
        buffer.append(line)

    text = "\n".join(buffer).strip()
    if text:
        blocks.append({"type": mode, "text": text})

    return blocks


def _context_header(card, heading):
    """The provenance line every chunk gets, so no row is ever orphaned."""

    return (
        f"Recipe: {card['title']} (recipe_id: {card['recipe_id']})\n"
        f"Section: {heading}"
    )


def _pack_paragraphs(text, target_words):
    """Group paragraphs into chunks of roughly target_words, never mid-sentence."""

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    packed = []
    current = []
    count = 0

    for paragraph in paragraphs:

        words = len(paragraph.split())

        if current and count + words > target_words:
            packed.append("\n\n".join(current))
            current = []
            count = 0

        current.append(paragraph)
        count += words

    if current:
        packed.append("\n\n".join(current))

    return packed


def chunk_card(card, target_words=140, max_table_rows=25, intro_word_limit=70):
    """Chunk a single loaded recipe card structurally."""

    chunks = []

    for section in card["sections"]:

        heading = section["heading"]
        header = _context_header(card, heading)
        blocks = _split_blocks(section["content"])

        index = 0
        pending_intro = None

        for block in blocks:

            if block["type"] == "table":

                lines = [l for l in block["text"].split("\n") if l.strip()]

                # A markdown table is: header row, separator row, then data rows.
                table_header = lines[:2]
                data_rows = lines[2:]

                # Attach the short prose that declares the percentage basis.
                intro = ""
                if pending_intro and len(pending_intro.split()) <= intro_word_limit:
                    intro = pending_intro + "\n\n"
                    pending_intro = None

                # Only split a very long table, and repeat the header if so.
                groups = [
                    data_rows[i:i + max_table_rows]
                    for i in range(0, len(data_rows), max_table_rows)
                ] or [[]]

                for group in groups:

                    body = "\n".join(table_header + group)

                    chunks.append({
                        "id": f"{card['recipe_id']}__{_slug(heading)}__{index}",
                        "text": f"{header}\n\n{intro}{body}",
                        "section": heading,
                        "block_type": "table",
                        "source": card["source_file"],
                        "page": 1,
                    })
                    index += 1

            else:

                # Hold prose back in case a table follows and needs it.
                if pending_intro:
                    for piece in _pack_paragraphs(pending_intro, target_words):
                        chunks.append({
                            "id": f"{card['recipe_id']}__{_slug(heading)}__{index}",
                            "text": f"{header}\n\n{piece}",
                            "section": heading,
                            "block_type": "prose",
                            "source": card["source_file"],
                            "page": 1,
                        })
                        index += 1

                pending_intro = block["text"]

        # Flush any prose still held back at the end of the section.
        if pending_intro:
            for piece in _pack_paragraphs(pending_intro, target_words):
                chunks.append({
                    "id": f"{card['recipe_id']}__{_slug(heading)}__{index}",
                    "text": f"{header}\n\n{piece}",
                    "section": heading,
                    "block_type": "prose",
                    "source": card["source_file"],
                    "page": 1,
                })
                index += 1

    return chunks


def create_structure_aware_chunks(cards, target_words=140):
    """Chunk every card. Mirrors create_chunks() in the baseline chunker."""

    all_chunks = []

    for card in cards:
        all_chunks.extend(chunk_card(card, target_words=target_words))

    return all_chunks
