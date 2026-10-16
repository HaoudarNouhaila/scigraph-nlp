"""Whitelisted, parameterised graph queries.

Clients (API, UI) select a query by name and pass parameters; they never send Cypher.
Every parameter is validated against a declared type and range before execution. Values
that Cypher cannot parameterise (relationship types, path length bounds) are drawn from
fixed whitelists or validated integers and only then formatted into the statement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from scigraph.graph.neo4j_store import REL_TYPES_NEO4J
from scigraph.schema import ENTITY_TYPES

STATUSES = ("predicted", "verified")
# Relationship-type pattern restricting traversals to extracted/verified relation types,
# so that :INFERRED edges and provenance edges (HAS_MENTION, REFERS_TO) are never used.
REL_PATTERN = "|".join(REL_TYPES_NEO4J)


class QueryValidationError(ValueError):
    pass


@dataclass(frozen=True)
class Param:
    kind: type
    default: object = None
    required: bool = False
    min: int | None = None
    max: int | None = None
    choices: tuple | None = None
    max_length: int = 200


@dataclass(frozen=True)
class NamedQuery:
    name: str
    description: str
    cypher: str
    params: dict[str, Param] = field(default_factory=dict)


def _status_filter(var: str) -> str:
    return f"{var}.status IN $statuses"


QUERIES: dict[str, NamedQuery] = {}


def register(q: NamedQuery) -> None:
    # "%s" in a query stands for the whitelisted relationship-type pattern.
    if "%s" in q.cypher:
        q = NamedQuery(q.name, q.description, q.cypher.replace("%s", REL_PATTERN), q.params)
    QUERIES[q.name] = q


register(
    NamedQuery(
        "search_entities",
        "Full-text search over entity names, optionally restricted to one type.",
        """
    CALL db.index.fulltext.queryNodes('entity_name_fulltext', $text) YIELD node, score
    WHERE ($type IS NULL OR node.type = $type) AND node.status IN $statuses
    RETURN node.id AS id, node.name AS name, node.type AS type, node.status AS status,
           size(node.aliases) AS n_aliases, score
    ORDER BY score DESC LIMIT $limit
    """,
        {
            "text": Param(str, required=True),
            "type": Param(str, choices=ENTITY_TYPES),
            "limit": Param(int, 20, min=1, max=100),
        },
    )
)

register(
    NamedQuery(
        "entity_neighbors",
        "Outgoing and incoming relations of one entity.",
        """
    MATCH (e:Entity {id: $entity_id})-[r:%s]-(n:Entity)
    WHERE r.status IN $statuses AND ($relation IS NULL OR type(r) = $relation)
    RETURN type(r) AS relation, startNode(r) = e AS outgoing, n.id AS id, n.name AS name,
           n.type AS type, r.score AS score, r.support AS support, r.status AS status,
           r.id AS relation_id
    ORDER BY r.support DESC, r.score DESC LIMIT $limit
    """,
        {
            "entity_id": Param(str, required=True),
            "relation": Param(str, choices=REL_TYPES_NEO4J),
            "limit": Param(int, 50, min=1, max=500),
        },
    )
)

register(
    NamedQuery(
        "methods_for_task",
        "Methods linked to a task by USED_FOR (method USED_FOR task).",
        """
    MATCH (m:Entity {type: 'Method'})-[r:USED_FOR]->(t:Entity {type: 'Task'})
    WHERE toLower(t.name) CONTAINS toLower($task) AND r.status IN $statuses
    RETURN m.name AS method, t.name AS task, r.score AS score, r.support AS support,
           r.status AS status, r.id AS relation_id
    ORDER BY r.support DESC, r.score DESC LIMIT $limit
    """,
        {"task": Param(str, required=True), "limit": Param(int, 25, min=1, max=200)},
    )
)

register(
    NamedQuery(
        "metrics_for_task",
        "Metrics used to evaluate a task (metric EVALUATE_FOR task).",
        """
    MATCH (m:Entity {type: 'Metric'})-[r:EVALUATE_FOR]->(t:Entity)
    WHERE toLower(t.name) CONTAINS toLower($task) AND r.status IN $statuses
    RETURN m.name AS metric, t.name AS target, t.type AS target_type, r.support AS support,
           r.status AS status, r.id AS relation_id
    ORDER BY r.support DESC LIMIT $limit
    """,
        {"task": Param(str, required=True), "limit": Param(int, 25, min=1, max=200)},
    )
)

register(
    NamedQuery(
        "datasets_for_task",
        "Materials (datasets, corpora) used for or evaluating a task.",
        """
    MATCH (d:Entity {type: 'Material'})-[r:USED_FOR|EVALUATE_FOR]->(t:Entity)
    WHERE toLower(t.name) CONTAINS toLower($task) AND r.status IN $statuses
    RETURN d.name AS material, type(r) AS relation, t.name AS target, r.support AS support,
           r.status AS status, r.id AS relation_id
    ORDER BY r.support DESC LIMIT $limit
    """,
        {"task": Param(str, required=True), "limit": Param(int, 25, min=1, max=200)},
    )
)

register(
    NamedQuery(
        "top_entities",
        "Entities with the highest number of relations, optionally by type.",
        """
    MATCH (e:Entity)-[r:%s]-(:Entity)
    WHERE ($type IS NULL OR e.type = $type) AND r.status IN $statuses
    RETURN e.id AS id, e.name AS name, e.type AS type, count(r) AS degree
    ORDER BY degree DESC LIMIT $limit
    """,
        {"type": Param(str, choices=ENTITY_TYPES), "limit": Param(int, 20, min=1, max=200)},
    )
)

register(
    NamedQuery(
        "entity_provenance",
        "Documents and sentences in which an entity is mentioned.",
        """
    MATCH (d:Document)-[:HAS_MENTION]->(m:Mention)-[:REFERS_TO]->(e:Entity {id: $entity_id})
    RETURN d.id AS doc_id, d.title AS title, m.text AS mention, m.sent_index AS sentence,
           m.score AS score
    ORDER BY d.id, m.sent_index LIMIT $limit
    """,
        {"entity_id": Param(str, required=True), "limit": Param(int, 50, min=1, max=500)},
    )
)

register(
    NamedQuery(
        "relation_statistics",
        "Number of relations per type and status.",
        """
    MATCH (:Entity)-[r:%s]->(:Entity)
    RETURN type(r) AS relation, r.status AS status, count(*) AS count
    ORDER BY count DESC
    """,
        {},
    )
)

register(
    NamedQuery(
        "review_queue",
        "Predicted relations ordered by ascending confidence, for human review.",
        """
    MATCH (h:Entity)-[r:%s]->(t:Entity)
    WHERE r.status = 'predicted' AND r.score <= $max_score
    RETURN r.id AS relation_id, h.name AS head, type(r) AS relation, t.name AS tail,
           r.score AS score, r.evidence_sentences[0] AS evidence
    ORDER BY r.score ASC LIMIT $limit
    """,
        {"max_score": Param(float, 1.0, min=0, max=1), "limit": Param(int, 50, min=1, max=500)},
    )
)


register(
    NamedQuery(
        "relations_between",
        "Direct relations between two entities, in either direction.",
        """
    MATCH (a:Entity {id: $source_id})-[r:%s]-(b:Entity {id: $target_id})
    WHERE r.status IN $statuses
    RETURN startNode(r).name AS head, type(r) AS relation, endNode(r).name AS tail,
           r.score AS score, r.support AS support, r.status AS status, r.id AS relation_id
    """,
        {"source_id": Param(str, required=True), "target_id": Param(str, required=True)},
    )
)

register(
    NamedQuery(
        "relation_evidence",
        "Documents and sentences supporting one relation.",
        """
    MATCH (h:Entity)-[r {id: $relation_id}]->(t:Entity)
    RETURN h.name AS head, type(r) AS relation, t.name AS tail, r.status AS status,
           r.source AS source, r.model_version AS model_version, r.evidence_docs AS documents,
           r.evidence_sentences AS sentences
    """,
        {"relation_id": Param(str, required=True)},
    )
)

register(
    NamedQuery(
        "concepts_for_method",
        "Entities related to a method, grouped by relation type.",
        """
    MATCH (m:Entity {type: 'Method'})-[r:%s]-(c:Entity)
    WHERE toLower(m.name) CONTAINS toLower($method) AND r.status IN $statuses
    RETURN m.name AS method, type(r) AS relation, startNode(r) = m AS outgoing,
           c.name AS concept, c.type AS concept_type, r.support AS support,
           r.status AS status, r.id AS relation_id
    ORDER BY r.support DESC LIMIT $limit
    """,
        {"method": Param(str, required=True), "limit": Param(int, 50, min=1, max=500)},
    )
)

register(
    NamedQuery(
        "frequent_pairs",
        "Concept pairs whose relation is supported by the most documents.",
        """
    MATCH (h:Entity)-[r:%s]->(t:Entity)
    WHERE r.status IN $statuses AND r.support >= $min_support
    RETURN h.name AS head, h.type AS head_type, type(r) AS relation, t.name AS tail,
           t.type AS tail_type, r.support AS support, r.status AS status,
           r.id AS relation_id
    ORDER BY r.support DESC, h.name LIMIT $limit
    """,
        {"min_support": Param(int, 1, min=1, max=1000), "limit": Param(int, 25, min=1, max=500)},
    )
)

register(
    NamedQuery(
        "inferred_relations",
        "Relations derived by explicit rules (kept separate from extracted relations).",
        """
    MATCH (h:Entity)-[r:INFERRED]->(t:Entity)
    RETURN h.name AS head, r.relation AS relation, t.name AS tail, r.rule AS rule,
           r.basis AS basis, r.score AS score, r.premises AS premises
    ORDER BY r.score DESC LIMIT $limit
    """,
        {"limit": Param(int, 50, min=1, max=500)},
    )
)


def validate_params(query: NamedQuery, raw: dict) -> dict:
    unknown = set(raw) - set(query.params) - {"statuses"}
    if unknown:
        raise QueryValidationError(f"Unknown parameters: {sorted(unknown)}")
    out = {}
    for name, spec in query.params.items():
        value = raw.get(name, spec.default)
        if value is None:
            if spec.required:
                raise QueryValidationError(f"Missing required parameter: {name}")
            out[name] = None
            continue
        try:
            value = spec.kind(value)
        except (TypeError, ValueError) as exc:
            raise QueryValidationError(f"{name} must be {spec.kind.__name__}") from exc
        if isinstance(value, str):
            value = value.strip()
            if not value and spec.required:
                raise QueryValidationError(f"{name} must not be empty")
            if len(value) > spec.max_length:
                raise QueryValidationError(f"{name} longer than {spec.max_length} characters")
        if spec.choices is not None and value not in spec.choices:
            raise QueryValidationError(f"{name} must be one of {list(spec.choices)}")
        if spec.min is not None and value < spec.min:
            raise QueryValidationError(f"{name} must be >= {spec.min}")
        if spec.max is not None and value > spec.max:
            raise QueryValidationError(f"{name} must be <= {spec.max}")
        out[name] = value
    statuses = raw.get("statuses") or list(STATUSES)
    if isinstance(statuses, str):
        statuses = [statuses]
    if not set(statuses) <= set(STATUSES):
        raise QueryValidationError(f"statuses must be a subset of {list(STATUSES)}")
    out["statuses"] = list(statuses)
    if "text" in out and out["text"] is not None:
        out["text"] = lucene_escape(out["text"])
    return out


def lucene_escape(text: str) -> str:
    """Escape Lucene query syntax so that user text is searched literally."""
    special = '+-&|!(){}[]^"~*?:\\/'
    escaped = "".join("\\" + c if c in special else c for c in text)
    return escaped


def run_named_query(store, name: str, params: dict) -> list[dict]:
    if name not in QUERIES:
        raise QueryValidationError(f"Unknown query: {name}")
    q = QUERIES[name]
    return store.run_read(q.cypher, **validate_params(q, params))


def path_query(max_hops: int) -> str:
    """Shortest paths between two entities, bounded by a validated integer hop count."""
    if not isinstance(max_hops, int) or not 1 <= max_hops <= 5:
        raise QueryValidationError("max_hops must be an integer between 1 and 5")
    return f"""
    MATCH (a:Entity {{id: $source_id}}), (b:Entity {{id: $target_id}})
    MATCH p = allShortestPaths((a)-[:{REL_PATTERN}*..{max_hops}]-(b))
    WHERE all(r IN relationships(p) WHERE r.status IN $statuses)
    RETURN [n IN nodes(p) | {{id: n.id, name: n.name, type: n.type}}] AS nodes,
           [r IN relationships(p) | {{type: type(r), from: startNode(r).id, to: endNode(r).id,
                                      score: r.score, status: r.status}}] AS relations
    LIMIT $limit
    """


def find_paths(
    store, source_id: str, target_id: str, max_hops: int = 3, limit: int = 10, statuses=STATUSES
) -> list[dict]:
    if not (1 <= int(limit) <= 50):
        raise QueryValidationError("limit must be between 1 and 50")
    if not set(statuses) <= set(STATUSES):
        raise QueryValidationError("invalid statuses")
    return store.run_read(
        path_query(int(max_hops)),
        source_id=source_id,
        target_id=target_id,
        limit=int(limit),
        statuses=list(statuses),
    )


def khop_neighbourhood(
    store, entity_id: str, hops: int = 2, limit: int = 200, statuses=STATUSES
) -> list[dict]:
    """Edges within `hops` of an entity (for graph visualisation)."""
    if not 1 <= int(hops) <= 3:
        raise QueryValidationError("hops must be between 1 and 3")
    if not 1 <= int(limit) <= 1000:
        raise QueryValidationError("limit must be between 1 and 1000")
    cypher = f"""
    MATCH p = (e:Entity {{id: $entity_id}})-[:{REL_PATTERN}*1..{int(hops)}]-(:Entity)
    WHERE all(r IN relationships(p) WHERE r.status IN $statuses)
    UNWIND relationships(p) AS r
    WITH DISTINCT r
    RETURN startNode(r).id AS source, startNode(r).name AS source_name,
           startNode(r).type AS source_type, type(r) AS relation, endNode(r).id AS target,
           endNode(r).name AS target_name, endNode(r).type AS target_type, r.score AS score
    LIMIT $limit
    """
    return store.run_read(cypher, entity_id=entity_id, statuses=list(statuses), limit=int(limit))
