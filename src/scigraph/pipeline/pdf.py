"""PDF text extraction with PyMuPDF.

Only the body text is kept. Pages are read block by block in reading order; blocks that
look like running headers/footers (repeated on most pages) or page numbers are dropped,
and extraction stops at the references section, which otherwise produces many spurious
entities.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

REFERENCES_HEADING = re.compile(r"^\s*(\d+\.?\s*)?(references|bibliography)\s*$", re.I)
PAGE_NUMBER = re.compile(r"^\s*\d{1,4}\s*$")


@dataclass
class PdfDocument:
    title: str
    text: str
    n_pages: int
    metadata: dict
    # (character offset in `text`, 1-based page number) for the first block of each page.
    page_starts: list[tuple[int, int]] = field(default_factory=list)


def _blocks(page) -> list[str]:
    blocks = page.get_text("blocks", sort=True)
    return [b[4].strip() for b in blocks if b[6] == 0 and b[4].strip()]  # type 0 = text


def extract_pdf(source: str | Path | bytes, max_pages: int = 50) -> PdfDocument:
    import fitz  # PyMuPDF

    doc = (
        fitz.open(stream=source, filetype="pdf") if isinstance(source, bytes) else fitz.open(source)
    )
    try:
        pages = [_blocks(doc[i]) for i in range(min(len(doc), max_pages))]
        meta = dict(doc.metadata or {})
        n_pages = len(doc)
    finally:
        doc.close()

    # Blocks repeated on at least half of the pages are headers or footers.
    counts = Counter(b for page in pages for b in set(page))
    repeated = {b for b, c in counts.items() if len(pages) > 2 and c >= len(pages) / 2}

    kept: list[str] = []
    stop = False
    for page in pages:
        for block in page:
            first_line = block.splitlines()[0] if block else ""
            if REFERENCES_HEADING.match(first_line):
                stop = True
                break
            if block in repeated or PAGE_NUMBER.match(block):
                continue
            kept.append(block)
        if stop:
            break

    title = (meta.get("title") or "").strip()
    if not title and kept:
        title = kept[0].splitlines()[0][:200]
    return PdfDocument(title=title, text="\n\n".join(kept), n_pages=n_pages, metadata=meta)
