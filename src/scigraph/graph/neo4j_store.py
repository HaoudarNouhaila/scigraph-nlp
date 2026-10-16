"""Neo4j persistence for the knowledge graph.

Graph model::

    (:Document {id, title, source})
    (:Mention {id, text, type, sent_index, start, end, char_start, char_end, score})
    (:Entity {id, name, type, key, scope, aliases, status, max_score})
    (:Document)-[:HAS_MENTION]->(:Mention)-[:REFERS_TO]->(:Entity)
    (:Entity)-[:USED_FOR|FEATURE_OF|HYPONYM_OF|PART_OF|COMPARE|CONJUNCTION|EVALUATE_FOR
              {id, score, support, evidence_docs, evidence_sentences, status, source,
               verified_by, verified_at}]->(:Entity)

``status`` separates machine predictions (``predicted``) from facts confirmed by a human
or taken from gold annotations (``verified``). Predictions never overwrite the status of
a verified edge. All writes use ``UNWIND`` over batches inside managed transactions.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from scigraph.pipeline.graph_builder import KnowledgeGraph
from scigraph.schema import RELATION_TYPES


def rel_type(label: str) -> str:
    """SciERC label to Neo4j relationship type, e.g. USED-FOR -> USED_FOR."""
    if label not in RELATION_TYPES:
        raise ValueError(f"Unknown relation label: {label}")
    return label.replace("-", "_")


REL_TYPES_NEO4J = tuple(rel_type(r) for r in RELATION_TYPES)

SCHEMA_STATEMENTS = [
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
    "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT mention_id IF NOT EXISTS FOR (m:Mention) REQUIRE m.id IS UNIQUE",
    "CREATE INDEX entity_key IF NOT EXISTS FOR (e:Entity) ON (e.key)",
    "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
    "CREATE INDEX entity_status IF NOT EXISTS FOR (e:Entity) ON (e.status)",
    "CREATE FULLTEXT INDEX entity_name_fulltext IF NOT EXISTS FOR (e:Entity) "
    "ON EACH [e.name, e.key]",
]


@dataclass
class Neo4jSettings:
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = ""
    database: str = "neo4j"

    @classmethod
    def from_env(cls) -> Neo4jSettings:
        _load_dotenv()
        return cls(
            uri=os.getenv("NEO4J_URI", cls.uri),
            user=os.getenv("NEO4J_USER", cls.user),
            password=os.getenv("NEO4J_PASSWORD", ""),
            database=os.getenv("NEO4J_DATABASE", cls.database),
        )


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env reader (no extra dependency); existing variables win."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def batched(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


class Neo4jStore:
    def __init__(self, settings: Neo4jSettings | None = None, driver=None):
        from neo4j import GraphDatabase

        self.settings = settings or Neo4jSettings.from_env()
        self.driver = driver or GraphDatabase.driver(
            self.settings.uri, auth=(self.settings.user, self.settings.password)
        )

    def close(self) -> None:
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- low level -------------------------------------------------------------------
    def run_read(self, query: str, **params) -> list[dict]:
        records, _, _ = self.driver.execute_query(
            query, params, database_=self.settings.database, routing_="r"
        )
        return [r.data() for r in records]

    def _write_batches(self, query: str, rows: list[dict], batch_size: int, **params) -> int:
        n = 0
        with self.driver.session(database=self.settings.database) as session:
            for chunk in batched(rows, batch_size):
                session.execute_write(lambda tx, c=chunk: tx.run(query, rows=c, **params).consume())
                n += len(chunk)
        return n

    def verify_connectivity(self) -> None:
        self.driver.verify_connectivity()

    def create_schema(self) -> None:
        with self.driver.session(database=self.settings.database) as session:
            for stmt in SCHEMA_STATEMENTS:
                session.run(stmt).consume()

    def clear(self) -> None:
        """Delete all nodes, in batches (for tests and demos only)."""
        with self.driver.session(database=self.settings.database) as session:
            session.run(
                "MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 5000 ROWS"
            ).consume()

    # -- ingestion -------------------------------------------------------------------
    def ingest(
        self,
        kg: KnowledgeGraph,
        status: str = "predicted",
        source: str = "scigraph-pipeline",
        batch_size: int = 500,
        model_version: str = "",
    ) -> dict[str, int]:
        if status not in ("predicted", "verified"):
            raise ValueError("status must be 'predicted' or 'verified'")
        counts = {}
        counts["documents"] = self._write_batches(
            """
            UNWIND $rows AS row
            MERGE (d:Document {id: row.id})
            SET d.title = row.title, d.source = row.source, d.extractor = row.extractor,
                d.model_version = row.model_version
            """,
            [
                {
                    "id": d["id"],
                    "title": d.get("title", ""),
                    "source": d.get("source", ""),
                    "extractor": d.get("extractor", ""),
                    "model_version": d.get("model_version", ""),
                }
                for d in kg.documents.values()
            ],
            batch_size,
        )
        counts["entities"] = self._write_batches(
            """
            UNWIND $rows AS row
            MERGE (e:Entity {id: row.id})
            ON CREATE SET e.status = $status, e.created_from = $source
            SET e.name = row.name, e.type = row.type, e.key = row.key, e.scope = row.scope,
                e.aliases = row.aliases, e.max_score = row.max_score
            WITH e
            WHERE $status = 'verified'
            SET e.status = 'verified'
            """,
            [
                {
                    "id": e.id,
                    "name": e.name,
                    "type": e.type,
                    "key": e.key,
                    "scope": e.scope,
                    "aliases": e.aliases,
                    "max_score": e.max_score,
                }
                for e in kg.entities.values()
            ],
            batch_size,
            status=status,
            source=source,
        )
        counts["mentions"] = self._write_batches(
            """
            UNWIND $rows AS row
            MATCH (d:Document {id: row.doc_id})
            MATCH (e:Entity {id: row.entity_id})
            MERGE (m:Mention {id: row.id})
            SET m.text = row.text, m.type = row.type, m.sent_index = row.sent_index,
                m.start = row.start, m.end = row.end, m.char_start = row.char_start,
                m.char_end = row.char_end, m.score = row.score, m.page = row.page
            MERGE (d)-[:HAS_MENTION]->(m)
            MERGE (m)-[:REFERS_TO]->(e)
            """,
            [vars(m) for m in kg.mentions.values()],
            batch_size,
        )
        by_type: dict[str, list[dict]] = {}
        for t in kg.triples.values():
            by_type.setdefault(rel_type(t.relation), []).append(
                {
                    "id": t.id,
                    "head": t.head_id,
                    "tail": t.tail_id,
                    "score": t.score,
                    "docs": sorted({ev.doc_id for ev in t.evidence}),
                    "sentences": [
                        f"{ev.doc_id}#{ev.sent_index}"
                        + (f" (page {ev.page})" if ev.page else "")
                        + f": {ev.sentence}"
                        for ev in t.evidence
                    ][:20],
                }
            )
        n_rel = 0
        for rtype, rows in by_type.items():
            # Relationship types cannot be parameterised in Cypher; rtype comes from the
            # fixed whitelist REL_TYPES_NEO4J, never from user input.
            assert rtype in REL_TYPES_NEO4J
            n_rel += self._write_batches(
                f"""
                UNWIND $rows AS row
                MATCH (h:Entity {{id: row.head}})
                MATCH (t:Entity {{id: row.tail}})
                MERGE (h)-[r:{rtype} {{id: row.id}}]->(t)
                ON CREATE SET r.status = $status, r.source = $source, r.support = 0,
                              r.evidence_docs = [], r.evidence_sentences = [],
                              r.model_version = $model_version
                SET r.score = CASE WHEN r.score IS NULL OR row.score > r.score
                                   THEN row.score ELSE r.score END,
                    r.evidence_docs = r.evidence_docs +
                        [d IN row.docs WHERE NOT d IN r.evidence_docs],
                    r.evidence_sentences = (r.evidence_sentences +
                        [s IN row.sentences WHERE NOT s IN r.evidence_sentences])[0..50]
                SET r.support = size(r.evidence_docs)
                FOREACH (_ IN CASE WHEN $status = 'verified' THEN [1] ELSE [] END |
                    SET r.status = 'verified')
                """,
                rows,
                batch_size,
                status=status,
                source=source,
                model_version=model_version,
            )
        counts["relations"] = n_rel
        return counts

    def set_relation_status(
        self, relation_id: str, status: str, reviewer: str, note: str = ""
    ) -> dict | None:
        """Human review of one predicted edge: mark it verified or rejected."""
        if status not in ("verified", "rejected", "predicted"):
            raise ValueError("invalid status")
        with self.driver.session(database=self.settings.database) as session:
            rec = session.execute_write(
                lambda tx: tx.run(
                    """
                    MATCH (h:Entity)-[r {id: $id}]->(t:Entity)
                    SET r.status = $status, r.verified_by = $reviewer,
                        r.verified_at = toString(datetime()), r.review_note = $note
                    RETURN r.id AS id, type(r) AS relation, h.name AS head, t.name AS tail,
                           r.status AS status
                    """,
                    id=relation_id,
                    status=status,
                    reviewer=reviewer,
                    note=note,
                ).single()
            )
        return rec.data() if rec else None

    def counts(self) -> dict[str, int]:
        rows = self.run_read(
            """
            CALL () { MATCH (e:Entity) RETURN count(e) AS entities }
            CALL () { MATCH (d:Document) RETURN count(d) AS documents }
            CALL () { MATCH (m:Mention) RETURN count(m) AS mentions }
            CALL () { MATCH (:Entity)-[r]->(:Entity) WHERE r.status <> 'inferred'
                     RETURN count(r) AS relations }
            CALL () { MATCH (:Entity)-[r:INFERRED]->(:Entity) RETURN count(r) AS inferred }
            RETURN entities, documents, mentions, relations, inferred
            """
        )
        return rows[0] if rows else {}

    def export_triples(self, status: Iterable[str] = ("predicted", "verified")) -> list[dict]:
        return self.run_read(
            """
            MATCH (h:Entity)-[r]->(t:Entity)
            WHERE r.status IN $status AND type(r) <> 'INFERRED'
            RETURN h.id AS head_id, h.name AS head, h.type AS head_type, type(r) AS relation,
                   t.id AS tail_id, t.name AS tail, t.type AS tail_type, r.score AS score,
                   r.support AS support, r.status AS status, r.id AS id, r.evidence_docs AS docs
            """,
            status=list(status),
        )
