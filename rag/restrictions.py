"""Dietary-restriction detection for a free-text question.

Lives here (not in app.py) so it can be unit-tested without importing
Streamlit - and because it is the module behind the first production failure
this project turned into a permanent test (tests/failures/, Week 11):
"non-vegetarian" used to be read as "vegetarian" and the search was filtered
to the OPPOSITE diet.
"""

import re


# Order matters: the negated form is checked first so it can never be shadowed
# by the plain word it contains.
RESTRICTION_KEYWORDS = [
    "non-vegetarian", "vegan", "vegetarian", "dairy-free", "gluten-free",
    "egg-free", "nut-free",
]

# Whole-word match, and "vegan"/"vegetarian" must not be the tail of a
# "non-"/"non " negation.
_RESTRICTION_PATTERNS = [
    (r, re.compile(rf"(?<![\w-])(?<!non-)(?<!non ){re.escape(r)}(?![\w-])"))
    if r != "non-vegetarian" else (r, re.compile(r"(?<![\w-])non[- ]vegetarian(?![\w-])"))
    for r in RESTRICTION_KEYWORDS
]


def detect_restriction(question):
    """The first dietary restriction named in `question`, or None."""

    lowered = question.lower()
    return next((r for r, pattern in _RESTRICTION_PATTERNS if pattern.search(lowered)), None)
