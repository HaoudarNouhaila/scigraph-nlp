"""Sentence splitting and tokenisation of new text, with character offsets kept for provenance."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from scigraph.baseline.rules import load_spacy

# SciERC tokens are PTB-like: punctuation split off, hyphenated words kept together.
_WS = re.compile(r"\s+")


@dataclass
class TokenizedSentence:
    index: int
    text: str
    tokens: list[str]
    char_start: int
    char_end: int
    # Character offset (in the source document) of every token.
    token_offsets: list[tuple[int, int]] = field(default_factory=list)


def clean_text(text: str) -> str:
    """Repair common PDF artefacts: hyphenated line breaks and hard line wraps."""
    text = text.replace("­", "")  # soft hyphen
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def split_sentences(text: str, max_tokens: int = 120) -> list[TokenizedSentence]:
    nlp = load_spacy()
    doc = nlp(text)
    out: list[TokenizedSentence] = []
    for sent in doc.sents:
        toks = [t for t in sent if not t.is_space]
        if not toks:
            continue
        # Very long "sentences" (tables, reference lists) are cut into chunks.
        for k in range(0, len(toks), max_tokens):
            chunk = toks[k : k + max_tokens]
            out.append(
                TokenizedSentence(
                    index=len(out),
                    text=text[chunk[0].idx : chunk[-1].idx + len(chunk[-1].text)],
                    tokens=[t.text for t in chunk],
                    char_start=chunk[0].idx,
                    char_end=chunk[-1].idx + len(chunk[-1].text),
                    token_offsets=[(t.idx, t.idx + len(t.text)) for t in chunk],
                )
            )
    return out
