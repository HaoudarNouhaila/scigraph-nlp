"""Rule-based inference over the stored graph, kept apart from extracted knowledge.

The graph distinguishes four kinds of statements:

* extracted relations (``status = 'predicted'``): model output with evidence sentences;
* verified relations (``status = 'verified'``): gold annotations or human-reviewed edges;
* inferred relations: ``:INFERRED`` edges derived by the explicit rules below, carrying
  the rule name and the ids of their premises; they have no evidence sentence of their own;
* hypotheses: candidate links ranked by graph structure
  (:func:`scigraph.graph.analytics.candidate_links`), computed on demand and never stored.

Queries over extracted or verified knowledge never return ``:INFERRED`` edges, because
they filter on ``status IN ['predicted', 'verified']``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    name: str
    description: str
    cypher: str


# Each rule matches premises restricted to $statuses and MERGEs an :INFERRED edge whose
# id is derived from the rule and the endpoints, so re-running a rule is idempotent.
RULES: tuple[Rule, ...] = (
    Rule(
        "hyponym_transitivity",
        "a HYPONYM-OF b and b HYPONYM-OF c imply a HYPONYM-OF c",
        """
        MATCH (a:Entity)-[r1:HYPONYM_OF]->(b:Entity)-[r2:HYPONYM_OF]->(c:Entity)
        WHERE a <> c AND r1.status IN $statuses AND r2.status IN $statuses
          AND NOT (a)-[:HYPONYM_OF]->(c)
        RETURN a.id AS head, c.id AS tail, 'HYPONYM_OF' AS relation,
               [r1.id, r2.id] AS premises, r1.score * r2.score AS score,
               CASE WHEN r1.status = 'verified' AND r2.status = 'verified'
                    THEN 'verified' ELSE 'predicted' END AS basis
        """,
    ),
    Rule(
        "part_of_transitivity",
        "a PART-OF b and b PART-OF c imply a PART-OF c",
        """
        MATCH (a:Entity)-[r1:PART_OF]->(b:Entity)-[r2:PART_OF]->(c:Entity)
        WHERE a <> c AND r1.status IN $statuses AND r2.status IN $statuses
          AND NOT (a)-[:PART_OF]->(c)
        RETURN a.id AS head, c.id AS tail, 'PART_OF' AS relation,
               [r1.id, r2.id] AS premises, r1.score * r2.score AS score,
               CASE WHEN r1.status = 'verified' AND r2.status = 'verified'
                    THEN 'verified' ELSE 'predicted' END AS basis
        """,
    ),
    Rule(
        "conjunction_symmetry",
        "CONJUNCTION is symmetric: a CONJUNCTION b implies b CONJUNCTION a",
        """
        MATCH (a:Entity)-[r:CONJUNCTION]->(b:Entity)
        WHERE r.status IN $statuses AND NOT (b)-[:CONJUNCTION]->(a)
        RETURN b.id AS head, a.id AS tail, 'CONJUNCTION' AS relation, [r.id] AS premises,
               r.score AS score, r.status AS basis
        """,
    ),
)

WRITE_INFERRED = """
UNWIND $rows AS row
MATCH (h:Entity {id: row.head})
MATCH (t:Entity {id: row.tail})
MERGE (h)-[r:INFERRED {id: row.id}]->(t)
SET r.relation = row.relation, r.rule = row.rule, r.premises = row.premises,
    r.score = row.score, r.basis = row.basis, r.status = 'inferred'
"""


def materialize(store, statuses=("verified",)) -> dict[str, int]:
    """Apply every rule and write the conclusions as :INFERRED edges.

    By default only verified premises are used, so that inferred edges do not amplify
    extraction errors; pass ``statuses=("predicted", "verified")`` to include predictions.
    """
    import hashlib

    counts = {}
    for rule in RULES:
        rows = store.run_read(rule.cypher, statuses=list(statuses))
        for row in rows:
            key = f"{rule.name}|{row['head']}|{row['relation']}|{row['tail']}"
            row["id"] = "i:" + hashlib.sha1(key.encode()).hexdigest()[:16]
            row["rule"] = rule.name
            row["score"] = float(row["score"]) if row["score"] is not None else None
        if rows:
            store._write_batches(WRITE_INFERRED, rows, 500)
        counts[rule.name] = len(rows)
    return counts


def list_inferred(store, limit: int = 100) -> list[dict]:
    return store.run_read(
        """
        MATCH (h:Entity)-[r:INFERRED]->(t:Entity)
        RETURN h.name AS head, r.relation AS relation, t.name AS tail, r.rule AS rule,
               r.basis AS basis, r.score AS score, r.premises AS premises
        ORDER BY r.score DESC LIMIT $limit
        """,
        limit=int(limit),
    )


def clear_inferred(store) -> None:
    with store.driver.session(database=store.settings.database) as session:
        session.run("MATCH ()-[r:INFERRED]->() DELETE r").consume()
