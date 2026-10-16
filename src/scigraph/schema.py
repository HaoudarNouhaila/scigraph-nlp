"""Label inventories and shared data structures for the SciERC annotation scheme.

Entity and relation types follow Luan et al. (2018), "Multi-Task Identification of
Entities, Relations, and Coreference for Scientific Knowledge Graph Construction".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

ENTITY_TYPES: tuple[str, ...] = (
    "Task",
    "Method",
    "Metric",
    "Material",
    "OtherScientificTerm",
    "Generic",
)

RELATION_TYPES: tuple[str, ...] = (
    "USED-FOR",
    "FEATURE-OF",
    "HYPONYM-OF",
    "PART-OF",
    "COMPARE",
    "CONJUNCTION",
    "EVALUATE-FOR",
)

# COMPARE and CONJUNCTION are symmetric in the annotation guideline. All other
# relations are directional (head -> tail).
SYMMETRIC_RELATIONS: frozenset[str] = frozenset({"COMPARE", "CONJUNCTION"})

NO_RELATION = "NO-RELATION"
NO_ENTITY = "O"


@dataclass(frozen=True)
class Span:
    """A token span inside one sentence. `end` is inclusive, as in SciERC."""

    start: int
    end: int
    label: str

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class Relation:
    head: Span
    tail: Span
    label: str


@dataclass
class Sentence:
    """A sentence with sentence-local token offsets."""

    doc_key: str
    sent_id: int
    tokens: list[str]
    entities: list[Span] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)

    def text_of(self, span: Span) -> str:
        return " ".join(self.tokens[span.start : span.end + 1])

    def to_dict(self) -> dict:
        return {
            "doc_key": self.doc_key,
            "sent_id": self.sent_id,
            "tokens": self.tokens,
            "entities": [asdict(e) for e in self.entities],
            "relations": [
                {"head": asdict(r.head), "tail": asdict(r.tail), "label": r.label}
                for r in self.relations
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Sentence:
        ents = [Span(**e) for e in d.get("entities", [])]
        rels = [
            Relation(Span(**r["head"]), Span(**r["tail"]), r["label"])
            for r in d.get("relations", [])
        ]
        return cls(d["doc_key"], d["sent_id"], d["tokens"], ents, rels)
