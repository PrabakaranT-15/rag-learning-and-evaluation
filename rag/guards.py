"""Week 11 Module 6: cheap, deterministic answer checks that run on every
request and write their findings into the request log (`guard_flags`).

They do NOT block or rewrite an answer - they make a bad one findable. A
flagged answer is exactly what the support drill searches for, and the
failure -> test loop (tests/failures/) turns each confirmed one into a
permanent regression case.

dairy_free_swap_violations() exists because of the Track B drill failure: an
answer called a swap "dairy-free" while the swap itself (ghee) is dairy.
"""

import re


# Always dairy, whatever else is said around them. Deliberately NOT included:
# "milk", "butter", "cream" - "oat milk" / "vegan butter" / "coconut cream" are
# the correct dairy-free swaps, so those words alone prove nothing.
ALWAYS_DAIRY = ("ghee", "clarified butter", "buttermilk", "whey", "casein",
                "paneer", "cheese", "yogurt", "yoghurt")

_DAIRY_FREE_RE = re.compile(r"dairy[- ]free|non[- ]dairy|free (?:of|from) dairy", re.IGNORECASE)

# A dairy term right after one of these phrases is being ruled OUT, not
# recommended: "instead of ghee", "no whey", "free of cheese", "avoid yogurt".
# Phrases, not single words: a bare "free" would also swallow the very claim we
# are checking ("a dairy-FREE swap is ghee").
_RULED_OUT_RE = re.compile(
    r"(?:instead of|rather than|free (?:of|from)|\b(?:no|not|without|avoid|avoids|never|"
    r"replace|replaces|replacing))\s+(?:\w+\s+)?$"
)


def _sentences(text):
    return [s for s in re.split(r"(?<=[.!?\n])\s+", text) if s.strip()]


def dairy_free_swap_violations(answer):
    """Dairy terms recommended in a sentence that also claims 'dairy-free'.

    Returns a sorted list of offending terms (empty = nothing found). Only
    looks at sentences that make the dairy-free claim, and skips a term that
    is being ruled out ("instead of ghee")."""

    found = set()

    for sentence in _sentences(answer or ""):
        if not _DAIRY_FREE_RE.search(sentence):
            continue
        lowered = sentence.lower()

        for term in ALWAYS_DAIRY:
            for match in re.finditer(rf"\b{re.escape(term)}\b", lowered):
                if _RULED_OUT_RE.search(lowered[:match.start()]):
                    continue
                found.add(term)

    return sorted(found)


def check_answer(answer):
    """All guards over one answer -> {guard_name: [details]} for the ones that
    fired (empty dict = clean). Stored as `guard_flags` on the request log."""

    flags = {}

    dairy = dairy_free_swap_violations(answer)
    if dairy:
        flags["dairy_free_swap"] = dairy

    return flags
