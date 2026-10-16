"""Surface-form normalisation and acronym detection for entity mentions.

Normalisation is conservative: it removes variation that does not change meaning
(case, leading determiners, spacing around hyphens, simple English plurals), and
nothing else. It produces a *key* used for deduplication; the original mention text is
always kept for display and provenance.
"""

from __future__ import annotations

import re
import unicodedata

LEADING_DETERMINERS = {
    "a", "an", "the", "this", "these", "that", "those", "our", "their", "its", "such",
    "each", "every", "some", "any", "several", "various", "both",
}  # fmt: skip
# Words whose final "s" is not a plural marker.
NON_PLURAL_S = {
    "analysis", "bias", "corpus", "class", "loss", "process", "access", "basis", "status",
    "focus", "consensus", "thesis", "hypothesis", "synthesis", "parsing", "news", "physics",
    "linguistics", "statistics", "mathematics", "semantics", "pragmatics", "syntax",
    "gas", "lens", "series", "species", "ross", "less", "progress", "success",
}  # fmt: skip
ACRONYM_DEF = re.compile(
    r"(?P<long>[A-Za-z][\w\- ]{2,80}?)\s*\(\s*(?P<short>[A-Z][A-Za-z0-9\-]{1,10})\s*\)"
)


PTB_BRACKETS = {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]", "-LCB-": "{", "-RCB-": "}"}


def unescape_ptb(tokens: list[str]) -> list[str]:
    """Replace Penn Treebank bracket escapes used in SciERC (e.g. -LRB-) by brackets."""
    return [PTB_BRACKETS.get(t, t) for t in tokens]


NO_SPACE_BEFORE = {",", ".", ";", ":", ")", "]", "}", "?", "!", "'s", "n't", "%"}
NO_SPACE_AFTER = {"(", "[", "{"}


def detokenize(tokens: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Join PTB-style tokens into readable text and return each token's char offsets."""
    tokens = unescape_ptb(tokens)
    parts: list[str] = []
    offsets: list[tuple[int, int]] = []
    pos = 0
    for i, tok in enumerate(tokens):
        if i > 0 and tok not in NO_SPACE_BEFORE and tokens[i - 1] not in NO_SPACE_AFTER:
            parts.append(" ")
            pos += 1
        offsets.append((pos, pos + len(tok)))
        parts.append(tok)
        pos += len(tok)
    return "".join(parts), offsets


def singularize(word: str) -> str:
    w = word
    if len(w) <= 3 or w in NON_PLURAL_S or not w.isalpha():
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("sses", "shes", "ches", "xes")):
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


TRAILING_ACRONYM = re.compile(r"^(?P<long>.+?)\s*\(\s*(?P<short>[A-Za-z][A-Za-z0-9\-]{1,10})\s*\)$")


def normalize_mention(text: str) -> str:
    """Canonical key for a mention, e.g. 'The Neural Networks' -> 'neural network'.

    A trailing acronym definition is dropped when it matches the long form, so that
    'conditional random field (CRF)' and 'conditional random field' share a key.
    """
    m = TRAILING_ACRONYM.match(text.strip())
    if m and acronym_matches(m.group("short"), m.group("long")):
        text = m.group("long")
    t = unicodedata.normalize("NFKC", text).lower()
    t = re.sub(r"\s*-\s*", "-", t)
    t = re.sub(r"[\"'`‘’“”]", "", t)
    t = re.sub(r"\s+", " ", t).strip(" .,;:()[]")
    words = t.split()
    while words and words[0] in LEADING_DETERMINERS:
        words = words[1:]
    if not words:
        return t
    words[-1] = singularize(words[-1])
    return " ".join(words)


def is_acronym(text: str) -> bool:
    s = text.strip()
    return 2 <= len(s) <= 10 and sum(c.isupper() for c in s) >= 2 and " " not in s


def acronym_matches(short: str, long: str) -> bool:
    """True if the capital letters of `short` are the initials of `long` (in order)."""
    letters = [c.lower() for c in short if c.isalpha()]
    words = [w for w in re.split(r"[\s\-]+", long.lower()) if w]
    initials = "".join(w[0] for w in words)
    if "".join(letters) == initials:
        return True
    # Allow skipped function words ("hidden Markov model" -> HMM, "part of speech" -> POS).
    it = iter(initials)
    return all(any(ch == c for c in it) for ch in letters) and initials[0] == letters[0]


def find_acronyms(text: str) -> dict[str, str]:
    """Map acronym -> long form for definitions such as 'conditional random field (CRF)'."""
    found: dict[str, str] = {}
    for m in ACRONYM_DEF.finditer(text):
        short = m.group("short")
        words = m.group("long").split()
        # Take the shortest suffix of the preceding words that matches the acronym.
        n_letters = sum(c.isalpha() for c in short)
        for k in range(1, min(len(words), n_letters + 3) + 1):
            cand = " ".join(words[-k:])
            if acronym_matches(short, cand):
                found[short] = cand
                break
    return found
