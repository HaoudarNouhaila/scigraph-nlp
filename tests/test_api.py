import fitz
import pytest
from fastapi.testclient import TestClient

from scigraph.api import main
from scigraph.pipeline.extractor import RuleExtractor

from .conftest import neo4j_available


class FakeStore:
    """Records calls; returns canned rows. Stands in for Neo4j in API unit tests."""

    def __init__(self):
        self.calls = []
        self.ingested = []

    def run_read(self, query, **params):
        self.calls.append((query, params))
        return [{"id": "e:1", "name": "CRF", "type": "Method"}]

    def ingest(self, kg, status="predicted", source=""):
        self.ingested.append((kg, status, source))
        return {"entities": len(kg.entities), "relations": len(kg.triples)}

    def set_relation_status(self, rid, status, reviewer, note=""):
        return None if rid == "missing" else {"id": rid, "status": status}

    def export_triples(self, statuses=("predicted", "verified")):
        return [
            {
                "id": "t1",
                "head_id": "a",
                "head": "CRF",
                "head_type": "Method",
                "relation": "USED_FOR",
                "tail_id": "b",
                "tail": "NER",
                "tail_type": "Task",
                "score": 0.9,
                "support": 1,
                "status": "predicted",
                "docs": ["d1"],
            }
        ]

    def counts(self):
        return {"entities": 2}


@pytest.fixture
def client():
    store = FakeStore()
    main.app.dependency_overrides[main.get_extractor] = RuleExtractor
    main.app.dependency_overrides[main.get_store] = lambda: store
    with TestClient(main.app) as c:
        c.store = store
        yield c
    main.app.dependency_overrides.clear()


def test_extract_returns_entities_relations_and_triples(client, sample_texts):
    r = client.post("/extract", json={"text": sample_texts[0], "title": "t"})
    assert r.status_code == 200
    body = r.json()
    assert body["extractor"] == "rule-baseline"
    assert body["n_sentences"] >= 3 and body["triples"]
    ent = body["sentences"][0]["entities"][0]
    assert {"start", "end", "label", "score", "text"} <= set(ent)


@pytest.mark.parametrize(
    "payload", [{"text": ""}, {"text": "   "}, {"text": "x" * 20001}, {}, {"text": 5}]
)
def test_extract_rejects_invalid_input(client, payload):
    assert client.post("/extract", json=payload).status_code == 422


def test_extract_pdf(client):
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "We use a neural network for machine translation.")
    data = doc.tobytes()
    doc.close()
    r = client.post("/extract/pdf", files={"file": ("p.pdf", data, "application/pdf")})
    assert r.status_code == 200 and r.json()["n_sentences"] == 1
    bad = client.post("/extract/pdf", files={"file": ("p.pdf", b"hello", "application/pdf")})
    assert bad.status_code == 415


def test_ingest_marks_data_as_predicted(client, sample_texts):
    r = client.post("/graph/ingest", json={"text": sample_texts[1], "source": "unit-test"})
    assert r.status_code == 200
    _, status, source = client.store.ingested[0]
    assert status == "predicted" and source == "rule-baseline"
    assert client.post("/graph/ingest", json={"text": "x", "source": "a b"}).status_code == 422


def test_named_query_validation(client):
    ok = client.post("/graph/query", json={"name": "search_entities", "params": {"text": "crf"}})
    assert ok.status_code == 200 and ok.json()["rows"]
    query, params = client.store.calls[-1]
    assert "$text" in query and params["text"] == "crf"
    for body in (
        {"name": "drop_all", "params": {}},
        {"name": "search_entities", "params": {"text": "x", "limit": 10_000}},
        {"name": "MATCH (n) DETACH DELETE n", "params": {}},
    ):
        assert client.post("/graph/query", json=body).status_code == 422


def test_paths_bounds(client):
    assert (
        client.get("/graph/paths", params={"source_id": "a", "target_id": "b"}).status_code == 200
    )
    r = client.get("/graph/paths", params={"source_id": "a", "target_id": "b", "max_hops": 9})
    assert r.status_code == 422


def test_review_and_analytics_and_export(client):
    r = client.post("/graph/relations/t1/review", json={"status": "verified", "reviewer": "me"})
    assert r.json()["status"] == "verified"
    r = client.post(
        "/graph/relations/missing/review", json={"status": "verified", "reviewer": "me"}
    )
    assert r.status_code == 404
    r = client.post("/graph/relations/t1/review", json={"status": "deleted", "reviewer": "me"})
    assert r.status_code == 422
    assert client.get("/graph/analytics").json()["summary"]["nodes"] == 2
    ttl = client.get("/graph/export.ttl")
    assert ttl.status_code == 200 and "sg:usedFor" in ttl.text


def test_queries_catalogue(client):
    names = {q["name"] for q in client.get("/graph/queries").json()}
    assert {"search_entities", "entity_neighbors", "methods_for_task"} <= names


@pytest.mark.neo4j
@pytest.mark.skipif(
    not neo4j_available(), reason="set SCIGRAPH_NEO4J_TESTS=1 with a disposable Neo4j instance"
)
def test_api_against_real_neo4j(sample_texts):
    from scigraph.graph.neo4j_store import Neo4jStore

    with Neo4jStore() as s:
        s.clear()
    main._store = None
    main.app.dependency_overrides[main.get_extractor] = RuleExtractor
    try:
        with TestClient(main.app) as c:
            for t in sample_texts:
                assert c.post("/graph/ingest", json={"text": t}).status_code == 200
            stats = c.get("/graph/stats").json()
            assert stats["relations"] > 0
            rows = c.post("/graph/query", json={"name": "top_entities", "params": {"limit": 3}})
            top = rows.json()["rows"]
            assert top and top[0]["degree"] >= 1
            nb = c.get(f"/graph/neighbourhood/{top[0]['id']}", params={"hops": 2})
            assert nb.json()["edges"]
            assert c.get("/health").json()["neo4j"] is True
    finally:
        main.app.dependency_overrides.clear()
        with Neo4jStore() as s:
            s.clear()
