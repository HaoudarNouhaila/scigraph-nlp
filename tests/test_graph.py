import pytest

from scigraph.graph import analytics
from scigraph.graph.queries import (
    QUERIES,
    QueryValidationError,
    lucene_escape,
    path_query,
    validate_params,
)
from scigraph.graph.rdf_export import run_sparql, triples_to_rdf, validate_shacl
from scigraph.pipeline.extractor import RuleExtractor, process_text
from scigraph.pipeline.graph_builder import GraphBuilder

from .conftest import neo4j_available

ROWS = [
    {
        "id": "t1",
        "head_id": "a",
        "head": "CRF",
        "head_type": "Method",
        "relation": "USED_FOR",
        "tail_id": "b",
        "tail": "NER",
        "tail_type": "Task",
        "score": 0.95,
        "support": 2,
        "docs": ["d1"],
    },
    {
        "id": "t2",
        "head_id": "c",
        "head": "F1",
        "head_type": "Metric",
        "relation": "EVALUATE_FOR",
        "tail_id": "b",
        "tail": "NER",
        "tail_type": "Task",
        "score": 0.8,
        "support": 1,
        "docs": ["d1"],
    },
    {
        "id": "t3",
        "head_id": "d",
        "head": "CoNLL",
        "head_type": "Material",
        "relation": "USED_FOR",
        "tail_id": "c",
        "tail": "F1",
        "tail_type": "Metric",
        "score": 0.6,
        "support": 1,
        "docs": ["d2"],
    },
]


def test_validate_params_types_ranges_and_choices():
    q = QUERIES["search_entities"]
    out = validate_params(q, {"text": " crf ", "limit": "5"})
    assert out["text"] == "crf" and out["limit"] == 5 and out["type"] is None
    for bad in (
        {"text": "x", "limit": 0},
        {"text": "x", "type": "Person"},
        {"limit": 3},
        {"text": "x", "cypher": "MATCH (n) DELETE n"},
        {"text": "x", "statuses": ["deleted"]},
    ):
        with pytest.raises(QueryValidationError):
            validate_params(q, bad)


def test_lucene_escape_and_path_bounds():
    assert lucene_escape("a:b (c)") == "a\\:b \\(c\\)"
    assert "*..3" in path_query(3)
    for bad in (0, 6, "3; MATCH (n) DETACH DELETE n"):
        with pytest.raises(QueryValidationError):
            path_query(bad)


def test_networkx_analytics():
    g = analytics.build_graph(ROWS)
    s = analytics.summary(g)
    assert s["nodes"] == 4 and s["edges"] == 3 and s["connected_components"] == 1
    top = analytics.centrality(g, 2)["degree"]
    assert top[0]["name"] in {"NER", "F1"}
    paths = analytics.shortest_paths(g, "a", "d", max_hops=3)
    assert [n["name"] for n in paths[0]["nodes"]] == ["CRF", "NER", "F1", "CoNLL"]
    assert paths[0]["relations"][0] == {"type": "USED_FOR", "from": "a", "to": "b"}
    assert analytics.shortest_paths(g, "a", "d", max_hops=2) == []


def test_candidate_links_are_hypotheses_between_unlinked_pairs():
    rows = ROWS + [
        {
            "id": "t4",
            "head_id": "e",
            "head": "HMM",
            "head_type": "Method",
            "relation": "EVALUATE_FOR",
            "tail_id": "c",
            "tail": "F1",
            "tail_type": "Metric",
            "score": 0.7,
            "support": 1,
            "docs": ["d3"],
        },
    ]
    g = analytics.build_graph(rows)
    hyps = analytics.candidate_links(g, 10)
    pairs = {(h["a"], h["b"]) for h in hyps} | {(h["b"], h["a"]) for h in hyps}
    assert ("HMM", "NER") in pairs  # HMM and NER share the neighbour F1
    assert ("CRF", "NER") not in pairs  # already linked
    assert all(h["status"] == "hypothesis" for h in hyps)


def test_traversals_exclude_inferred_and_provenance_edges():
    from scigraph.graph.queries import QUERIES

    assert "INFERRED" not in path_query(3)
    assert "USED_FOR|FEATURE_OF" in QUERIES["entity_neighbors"].cypher


def test_rdf_export_sparql_and_shacl():
    g = triples_to_rdf(ROWS)
    rows = run_sparql(g, "methods_used_for_tasks")
    assert rows == [{"method": "CRF", "task": "NER"}]
    metrics = run_sparql(g, "metrics_per_task")
    assert metrics[0]["metrics"] == "F1"
    conforms, _ = validate_shacl(g)
    assert conforms


@pytest.mark.neo4j
@pytest.mark.skipif(
    not neo4j_available(), reason="set SCIGRAPH_NEO4J_TESTS=1 with a disposable Neo4j instance"
)
def test_neo4j_ingest_query_review(sample_texts):
    from scigraph.graph.neo4j_store import Neo4jStore
    from scigraph.graph.queries import find_paths, run_named_query

    b = GraphBuilder()
    for t in sample_texts:
        result = process_text(t, RuleExtractor(), builder=b)
    kg = result["graph"]
    with Neo4jStore() as store:
        store.clear()
        store.create_schema()
        first = store.ingest(kg)
        second = store.ingest(kg)  # idempotent
        assert first == second
        counts = store.counts()
        assert counts["entities"] == len(kg.entities)
        assert counts["relations"] == len(kg.triples)
        stats = run_named_query(store, "relation_statistics", {})
        assert sum(r["count"] for r in stats) == len(kg.triples)
        triple = next(iter(kg.triples.values()))
        rec = store.set_relation_status(triple.id, "verified", "pytest")
        assert rec["status"] == "verified"
        store.ingest(kg)  # a new prediction must not downgrade a verified edge
        verified = [r for r in store.export_triples(("verified",)) if r["id"] == triple.id]
        assert verified
        paths = find_paths(store, triple.head_id, triple.tail_id, 2)
        assert paths and len(paths[0]["nodes"]) == 2

        from scigraph.graph import reasoning

        before = store.counts()["relations"]
        reasoning.materialize(store, ("predicted", "verified"))
        after = store.counts()
        assert after["relations"] == before  # inferred edges are counted separately
        assert all(
            r["status"] != "inferred"
            for r in store.export_triples(("predicted", "verified", "inferred"))
        )
        store.clear()
