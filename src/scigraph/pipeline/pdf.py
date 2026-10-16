"""PDF text extraction with PyMuPDF.

Only the body text is kept. Pages are read block by block in reading order; blocks that
look like running headers/footers (repeated on most pages) or page numbers are dropped,
and extraction stops at the references section, which otherwise produces many spurious
entities.
"""

from __future__ import annotations

import bisect
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from scigraph.pipeline.text import clean_text

REFERENCES_HEADING = re.compile(r"^\s*(\d+\.?\s*)?(references|bibliography)\s*$", re.I)
BLOCK_SEPARATOR = "\n\n"
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
    page_starts: list[tuple[int, int]] = []
    offset = 0
    stop = False
    for page_no, page in enumerate(pages, start=1):
        for block in page:
            first_line = block.splitlines()[0] if block else ""
            if REFERENCES_HEADING.match(first_line):
                stop = True
                break
            if block in repeated or PAGE_NUMBER.match(block):
                continue
            # Blocks are cleaned here so that later cleaning does not shift offsets.
            cleaned = clean_text(block)
            if not cleaned:
                continue
            if not page_starts or page_starts[-1][1] != page_no:
                page_starts.append((offset, page_no))
            kept.append(cleaned)
            offset += len(cleaned) + len(BLOCK_SEPARATOR)
        if stop:
            break

    title = (meta.get("title") or "").strip()
    if not title and kept:
        title = kept[0][:200]
    return PdfDocument(
        title=title,
        text=BLOCK_SEPARATOR.join(kept),
        n_pages=n_pages,
        metadata=meta,
        page_starts=page_starts,
    )


def page_of(offset: int, page_starts: list[tuple[int, int]]) -> int | None:
    """Page number containing a character offset of :attr:`PdfDocument.text`."""
    if not page_starts:
        return None
    i = bisect.bisect_right([o for o, _ in page_starts], offset) - 1
    return page_starts[max(i, 0)][1]
