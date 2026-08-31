"""Load markdown recipe cards into the shape the rest of the pipeline expects.

This is the recipe-domain counterpart to pdf_loader.py. pdf_loader returns
page dicts from a PDF; this returns the same {text, page, source} shape so the
EXISTING baseline chunker in rag/chunker.py works unchanged, plus the recipe
metadata (recipe_id, cuisine, dietary_tags) that Task Set B requires on every
chunk.
"""

import os
import re


REQUIRED_FRONTMATTER = ["recipe_id", "cuisine", "dietary_tags"]


def _parse_frontmatter(text):
    """Pull the YAML-ish frontmatter block off the top of a card."""

    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.S)

    if not match:
        return {}, text

    meta = {}

    for line in match.group(1).split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()

    body = text[match.end():]

    return meta, body


def _split_sections(body):
    """Split a card body into its '## ' sections, keeping the H1 title separate."""

    title = ""

    title_match = re.search(r"^#\s+(.+)$", body, re.M)

    if title_match:
        title = title_match.group(1).strip()

    sections = []

    parts = re.split(r"^##\s+(.+)$", body, flags=re.M)

    # parts[0] is whatever sits before the first '## ' heading
    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        sections.append({"heading": heading, "content": content})

    return title, sections


def load_recipe_card(path):
    """Load one markdown recipe card into a dict with metadata and sections."""

    with open(path, encoding="utf-8") as handle:
        raw = handle.read()

    meta, body = _parse_frontmatter(raw)

    source_file = os.path.basename(path)

    missing = [k for k in REQUIRED_FRONTMATTER if not meta.get(k)]

    if missing:
        raise ValueError(
            f"{source_file}: card is missing required frontmatter {missing}. "
            "Every chunk must carry recipe_id, cuisine and dietary_tags."
        )

    title, sections = _split_sections(body)

    return {
        "source_file": source_file,
        "path": path,
        "recipe_id": meta["recipe_id"],
        "recipe_name": meta.get("recipe_name", title),
        "cuisine": meta["cuisine"],
        "dietary_tags": meta["dietary_tags"],
        "title": title or meta.get("recipe_name", ""),
        "body": body.strip(),
        "raw": raw,
        "sections": sections,
    }


def load_cards(directory):
    """Load every .md card in a directory, sorted by filename."""

    cards = []

    for name in sorted(os.listdir(directory)):
        if name.endswith(".md"):
            cards.append(load_recipe_card(os.path.join(directory, name)))

    return cards


def to_pages(cards):
    """Adapt cards to the {text, page, source} shape pdf_loader produces.

    This lets the UNCHANGED baseline chunker (rag/chunker.py) consume recipe
    cards exactly as it consumes PDF pages, so Strategy A really is the
    existing chunker and not a rewrite of it.
    """

    pages = []

    for card in cards:
        pages.append({
            "text": card["body"],
            "page": 1,
            "source": card["source_file"],
        })

    return pages


def metadata_for_source(cards):
    """Map source_file -> recipe metadata, for re-attaching after baseline chunking."""

    return {
        card["source_file"]: {
            "source_file": card["source_file"],
            "recipe_id": card["recipe_id"],
            "recipe_name": card["recipe_name"],
            "cuisine": card["cuisine"],
            "dietary_tags": card["dietary_tags"],
        }
        for card in cards
    }
