"""Loading, validation and statistics for the SciERC processed JSON release.

The official release (``sciERC_processed.tar.gz``) stores one JSON document per line::

    {"doc_key": str, "sentences": [[tok, ...], ...],
     "ner": [[[start, end, label], ...], ...],          # one list per sentence
     "relations": [[[s1, e1, s2, e2, label], ...], ...],
     "clusters": [[[start, end], ...], ...]}

All offsets are document-level token indices with inclusive ends. This module converts
them into sentence-local offsets (:class:`scigraph.schema.Sentence`).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from scigraph.schema import ENTITY_TYPES, RELATION_TYPES, Relation, Sentence, Span

SPLITS = ("train", "dev", "test")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows: Iterable[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class SciERCFormatError(ValueError):
    pass


def document_to_sentences(doc: dict) -> list[Sentence]:
    """Convert one SciERC document into sentences with local offsets.

    Raises :class:`SciERCFormatError` when an annotation crosses a sentence boundary or
    uses an unknown label, so that silent data corruption cannot happen.
    """
    sentences: list[Sentence] = []
    offset = 0
    doc_key = doc["doc_key"]
    n = len(doc["sentences"])
    ner = doc.get("ner", [[] for _ in range(n)])
    rels = doc.get("relations", [[] for _ in range(n)])
    for i, tokens in enumerate(doc["sentences"]):
        lo, hi = offset, offset + len(tokens) - 1

        def local(start: int, end: int, label: str) -> Span:
            if not (lo <= start <= end <= hi):  # noqa: B023 - bound per iteration
                raise SciERCFormatError(
                    f"{doc_key} sentence {i}: span ({start}, {end}) outside [{lo}, {hi}]"  # noqa: B023
                )
            return Span(start - lo, end - lo, label)  # noqa: B023

        ents = []
        for start, end, label in ner[i]:
            if label not in ENTITY_TYPES:
                raise SciERCFormatError(f"{doc_key}: unknown entity label {label}")
            ents.append(local(start, end, label))
        ent_lookup = {(e.start, e.end): e for e in ents}
        relations = []
        for s1, e1, s2, e2, label in rels[i]:
            if label not in RELATION_TYPES:
                raise SciERCFormatError(f"{doc_key}: unknown relation label {label}")
            h = local(s1, e1, "?")
            t = local(s2, e2, "?")
            head = ent_lookup.get((h.start, h.end))
            tail = ent_lookup.get((t.start, t.end))
            if head is None or tail is None:
                raise SciERCFormatError(f"{doc_key}: relation argument is not an entity")
            relations.append(Relation(head, tail, label))
        sentences.append(Sentence(doc_key, i, list(tokens), ents, relations))
        offset += len(tokens)
    return sentences


def load_split(processed_dir: str | Path, split: str) -> list[Sentence]:
    """Load a prepared split (sentence-level JSONL written by ``scripts/prepare_data.py``)."""
    rows = read_jsonl(Path(processed_dir) / f"{split}.jsonl")
    return [Sentence.from_dict(r) for r in rows]


def overlap_kind(a: Span, b: Span) -> str | None:
    """Return 'nested', 'crossing', 'identical' or None for two spans."""
    if a.end < b.start or b.end < a.start:
        return None
    if a.start == b.start and a.end == b.end:
        return "identical"
    if (a.start <= b.start and b.end <= a.end) or (b.start <= a.start and a.end <= b.end):
        return "nested"
    return "crossing"


def compute_statistics(docs: list[dict]) -> dict:
    """Corpus statistics computed directly from the raw documents of one split."""
    sents = [s for d in docs for s in document_to_sentences(d)]
    ent_counts: Counter[str] = Counter()
    rel_counts: Counter[str] = Counter()
    span_lengths: Counter[int] = Counter()
    overlap_counts: Counter[str] = Counter()
    sentences_with_overlap = 0
    reversed_pairs = 0
    self_relations = 0
    n_tokens = 0
    candidate_pairs = 0
    n_coref_clusters = sum(len(d.get("clusters", [])) for d in docs)
    for s in sents:
        n_tokens += len(s.tokens)
        candidate_pairs += len(s.entities) * (len(s.entities) - 1)
        for e in s.entities:
            ent_counts[e.label] += 1
            span_lengths[e.length] += 1
        has_overlap = False
        for i in range(len(s.entities)):
            for j in range(i + 1, len(s.entities)):
                kind = overlap_kind(s.entities[i], s.entities[j])
                if kind:
                    overlap_counts[kind] += 1
                    has_overlap = True
        sentences_with_overlap += int(has_overlap)
        for r in s.relations:
            rel_counts[r.label] += 1
            if r.head == r.tail:
                self_relations += 1
            if r.head.start > r.tail.start:
                reversed_pairs += 1
    total_ents = sum(ent_counts.values())
    max_len = max(span_lengths) if span_lengths else 0
    covered = sum(c for k, c in span_lengths.items() if k <= 8)
    return {
        "documents": len(docs),
        "sentences": len(sents),
        "tokens": n_tokens,
        "entities": total_ents,
        "relations": sum(rel_counts.values()),
        "coreference_clusters": n_coref_clusters,
        "mean_sentence_length": round(n_tokens / len(sents), 2) if sents else 0,
        "max_sentence_length": max((len(s.tokens) for s in sents), default=0),
        "sentences_per_document": round(len(sents) / len(docs), 2) if docs else 0,
        # Ordered pairs of distinct gold entities in the same sentence: the candidate set
        # of the relation classifier in Experiment A.
        "candidate_entity_pairs": candidate_pairs,
        "positive_pair_ratio_pct": round(100 * sum(rel_counts.values()) / candidate_pairs, 2)
        if candidate_pairs
        else 0,
        "entity_types": dict(sorted(ent_counts.items())),
        "relation_types": dict(sorted(rel_counts.items())),
        "max_entity_length": max_len,
        "entity_length_histogram": {str(k): span_lengths[k] for k in sorted(span_lengths)},
        "entities_within_8_tokens_pct": round(100 * covered / total_ents, 2) if total_ents else 0,
        "overlapping_entity_pairs": dict(overlap_counts),
        "sentences_with_overlapping_entities": sentences_with_overlap,
        "relations_with_head_after_tail": reversed_pairs,
        "self_relations": self_relations,
    }
