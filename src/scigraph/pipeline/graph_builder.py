"""Turn extracted mentions and relations into deduplicated entities and triples.

Deduplication rules (deliberately stricter than "same string, same node"):

1. Two mentions refer to the same entity only if they have the same entity type *and*
   the same normalised key (:func:`normalize_mention`). "parsing" as a Task and
   "parsing" as a Method stay separate nodes.
2. An acronym defined in a document ("conditional random field (CRF)") is resolved to
   its long form inside that document only; the acronym is stored as an alias.
3. ``Generic`` mentions ("this approach", "it") and bare head nouns ("method",
   "model", "system") are only meaningful in context. They are scoped to their document
   and never merged across documents.
4. Near-duplicates (high token overlap, same type) are *not* merged; they are reported
   as ``candidate_duplicates`` for human review.

Every entity keeps its mentions and every triple keeps its evidence sentences, so each
fact in the graph can be traced back to the text it came from.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field

from scigraph.pipeline.normalize import find_acronyms, is_acronym, normalize_mention

CONTEXT_DEPENDENT_KEYS = {
    "method", "approach", "model", "system", "algorithm", "technique", "framework",
    "task", "problem", "result", "results", "data", "performance", "it", "they", "this",
    "paper", "work", "one", "feature", "information", "process", "application",
}  # fmt: skip


def _hash(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]


@dataclass
class Mention:
    id: str
    doc_id: str
    sent_index: int
    start: int  # token offsets inside the sentence, inclusive end
    end: int
    char_start: int
    char_end: int
    text: str
    type: str
    score: float
    entity_id: str = ""


@dataclass
class Entity:
    id: str
    name: str
    type: str
    key: str
    scope: str  # "global" or the document id for document-scoped entities
    aliases: list[str] = field(default_factory=list)
    mention_ids: list[str] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)
    max_score: float = 0.0


@dataclass
class Evidence:
    doc_id: str
    sent_index: int
    sentence: str
    head_text: str
    tail_text: str
    score: float


@dataclass
class Triple:
    id: str
    head_id: str
    relation: str
    tail_id: str
    score: float
    evidence: list[Evidence] = field(default_factory=list)

    @property
    def support(self) -> int:
        return len(self.evidence)


@dataclass
class KnowledgeGraph:
    entities: dict[str, Entity] = field(default_factory=dict)
    mentions: dict[str, Mention] = field(default_factory=dict)
    triples: dict[str, Triple] = field(default_factory=dict)
    documents: dict[str, dict] = field(default_factory=dict)
    candidate_duplicates: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "documents": list(self.documents.values()),
            "entities": [asdict(e) for e in self.entities.values()],
            "mentions": [asdict(m) for m in self.mentions.values()],
            "triples": [{**asdict(t), "support": t.support} for t in self.triples.values()],
            "candidate_duplicates": self.candidate_duplicates,
        }


def entity_scope(doc_id: str, etype: str, key: str) -> str:
    if etype == "Generic" or key in CONTEXT_DEPENDENT_KEYS or len(key) < 2:
        return doc_id
    return "global"


class GraphBuilder:
    """Accumulates extraction results document by document."""

    def __init__(self) -> None:
        self.kg = KnowledgeGraph()

    def add_document(self, doc_id: str, meta: dict, text: str, sentences: list[dict]) -> None:
        """Add one processed document.

        ``sentences`` items have keys ``index``, ``text``, ``tokens``, ``token_offsets``,
        ``entities`` (list of dicts with start, end, label, score) and ``relations``
        (list of dicts with head, tail as entity indices, label, score).
        """
        self.kg.documents[doc_id] = {"id": doc_id, **meta}
        acronyms = find_acronyms(text)
        acr_keys = {a: normalize_mention(long) for a, long in acronyms.items()}

        for sent in sentences:
            ment_ids = []
            for ent in sent["entities"]:
                ment = self._add_mention(doc_id, sent, ent, acr_keys)
                ment_ids.append(ment.id)
            for rel in sent["relations"]:
                self._add_triple(
                    doc_id,
                    sent,
                    ment_ids[rel["head"]],
                    ment_ids[rel["tail"]],
                    rel["label"],
                    rel["score"],
                )

    def _add_mention(self, doc_id: str, sent: dict, ent: dict, acr_keys: dict) -> Mention:
        tokens = sent["tokens"]
        offsets = sent.get("token_offsets") or []
        cs = offsets[ent["start"]][0] if offsets else -1
        ce = offsets[ent["end"]][1] if offsets else -1
        if offsets and "char_start" in sent:
            base = sent["char_start"]
            text = sent["text"][cs - base : ce - base]
        else:
            text = ent.get("text") or " ".join(tokens[ent["start"] : ent["end"] + 1])
        mid = f"m:{_hash(doc_id, str(sent['index']), str(ent['start']), str(ent['end']))}"
        mention = Mention(
            mid,
            doc_id,
            sent["index"],
            ent["start"],
            ent["end"],
            cs,
            ce,
            text,
            ent["label"],
            float(ent.get("score", 1.0)),
        )
        key = normalize_mention(text)
        alias = None
        surface = text.strip()
        if is_acronym(surface) and surface in acr_keys:
            alias, key = surface, acr_keys[surface]
        scope = entity_scope(doc_id, ent["label"], key)
        eid = f"e:{_hash(scope, ent['label'], key)}"
        entity = self.kg.entities.get(eid)
        if entity is None:
            entity = Entity(eid, text, ent["label"], key, scope)
            self.kg.entities[eid] = entity
        if alias and alias not in entity.aliases:
            entity.aliases.append(alias)
        elif text != entity.name and text not in entity.aliases and not alias:
            entity.aliases.append(text)
        entity.mention_ids.append(mid)
        if doc_id not in entity.doc_ids:
            entity.doc_ids.append(doc_id)
        entity.max_score = max(entity.max_score, mention.score)
        mention.entity_id = eid
        self.kg.mentions[mid] = mention
        return mention

    def _add_triple(self, doc_id, sent, head_mid, tail_mid, label, score) -> None:
        h = self.kg.mentions[head_mid]
        t = self.kg.mentions[tail_mid]
        if h.entity_id == t.entity_id:
            return  # a relation between two mentions of the same entity carries no fact
        tid = f"t:{_hash(h.entity_id, label, t.entity_id)}"
        triple = self.kg.triples.get(tid)
        if triple is None:
            triple = Triple(tid, h.entity_id, label, t.entity_id, float(score))
            self.kg.triples[tid] = triple
        triple.score = max(triple.score, float(score))
        triple.evidence.append(
            Evidence(doc_id, sent["index"], sent["text"], h.text, t.text, float(score))
        )

    def finalize(self, near_duplicate_threshold: float = 0.75) -> KnowledgeGraph:
        """Pick display names and list near-duplicate entity pairs for review."""
        for e in self.kg.entities.values():
            surfaces = Counter(self.kg.mentions[m].text for m in e.mention_ids)
            e.name = surfaces.most_common(1)[0][0]
            e.aliases = sorted({a for a in e.aliases if a != e.name})
        by_type: dict[str, list] = defaultdict(list)
        for e in self.kg.entities.values():
            if e.scope == "global":
                by_type[e.type].append(e)
        dups = []
        for etype, ents in by_type.items():
            toks = [(e, set(e.key.split())) for e in ents]
            for i in range(len(toks)):
                for j in range(i + 1, len(toks)):
                    a, sa = toks[i]
                    b, sb = toks[j]
                    if not sa or not sb:
                        continue
                    jac = len(sa & sb) / len(sa | sb)
                    if jac >= near_duplicate_threshold:
                        dups.append(
                            {
                                "a": a.id,
                                "b": b.id,
                                "a_name": a.name,
                                "b_name": b.name,
                                "type": etype,
                                "jaccard": round(jac, 3),
                            }
                        )
        self.kg.candidate_duplicates = dups
        return self.kg
