"""RDF export (Turtle) of extracted triples, SPARQL examples and SHACL validation.

Each extracted relation is written twice:

* as a direct triple ``<head> sg:usedFor <tail>`` for simple SPARQL queries;
* as a reified ``sg:Assertion`` carrying confidence, support, status and provenance.
"""

from __future__ import annotations

import re
from decimal import Decimal
from pathlib import Path

from rdflib import RDF, RDFS, XSD, Graph, Literal, Namespace, URIRef
from rdflib.namespace import PROV

SG = Namespace("https://w3id.org/scigraph-nlp/ontology#")
DATA = Namespace("https://w3id.org/scigraph-nlp/resource/")
ONTOLOGY_DIR = Path(__file__).resolve().parents[1] / "resources"

PROPERTY = {
    "USED_FOR": SG.usedFor, "FEATURE_OF": SG.featureOf, "HYPONYM_OF": SG.hyponymOf,
    "PART_OF": SG.partOf, "COMPARE": SG.compare, "CONJUNCTION": SG.conjunction,
    "EVALUATE_FOR": SG.evaluateFor,
}  # fmt: skip


def _local(identifier: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]", "_", identifier)


def triples_to_rdf(triples: list[dict], include_ontology: bool = False) -> Graph:
    """Build an RDF graph from triple rows (see :func:`analytics.triples_from_kg`)."""
    g = Graph()
    g.bind("sg", SG)
    g.bind("sgr", DATA)
    g.bind("prov", PROV)
    if include_ontology:
        g.parse(ONTOLOGY_DIR / "scigraph.ttl", format="turtle")
    for t in triples:
        rel = t["relation"].replace("-", "_")
        if rel not in PROPERTY:
            continue
        h = DATA[f"entity/{_local(t['head_id'])}"]
        o = DATA[f"entity/{_local(t['tail_id'])}"]
        for node, name, etype in ((h, t["head"], t["head_type"]), (o, t["tail"], t["tail_type"])):
            g.add((node, RDF.type, SG[etype]))
            g.add((node, RDFS.label, Literal(name, datatype=XSD.string)))
        g.add((h, PROPERTY[rel], o))
        a = DATA[f"assertion/{_local(t.get('id') or t['head_id'] + rel + t['tail_id'])}"]
        g.add((a, RDF.type, SG.Assertion))
        g.add((a, SG.subject, h))
        g.add((a, SG.predicate, PROPERTY[rel]))
        g.add((a, SG.object, o))
        if t.get("score") is not None:
            g.add(
                (
                    a,
                    SG.confidence,
                    Literal(Decimal(str(round(float(t["score"]), 4))), datatype=XSD.decimal),
                )
            )
        g.add((a, SG.support, Literal(int(t.get("support") or 1), datatype=XSD.integer)))
        g.add((a, SG.status, Literal(t.get("status", "predicted"))))
        for doc in t.get("docs") or []:
            d = DATA[f"document/{_local(doc)}"]
            g.add((d, RDF.type, SG.Document))
            g.add((a, PROV.wasDerivedFrom, d))
    return g


def export_turtle(triples: list[dict], path: str | Path) -> int:
    g = triples_to_rdf(triples)
    g.serialize(destination=str(path), format="turtle")
    return len(g)


SPARQL_EXAMPLES = {
    "methods_used_for_tasks": """
        PREFIX sg: <https://w3id.org/scigraph-nlp/ontology#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?method ?task WHERE {
            ?m a sg:Method ; rdfs:label ?method ; sg:usedFor ?t .
            ?t a sg:Task ; rdfs:label ?task .
        } ORDER BY ?task LIMIT 50
    """,
    "metrics_per_task": """
        PREFIX sg: <https://w3id.org/scigraph-nlp/ontology#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?task (GROUP_CONCAT(DISTINCT ?metric; separator=", ") AS ?metrics) WHERE {
            ?m a sg:Metric ; rdfs:label ?metric ; sg:evaluateFor ?t .
            ?t rdfs:label ?task .
        } GROUP BY ?task
    """,
    "high_confidence_assertions": """
        PREFIX sg: <https://w3id.org/scigraph-nlp/ontology#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?head ?predicate ?tail ?confidence WHERE {
            ?a a sg:Assertion ; sg:subject ?s ; sg:predicate ?predicate ; sg:object ?o ;
               sg:confidence ?confidence .
            ?s rdfs:label ?head . ?o rdfs:label ?tail .
            FILTER (?confidence >= 0.9)
        } ORDER BY DESC(?confidence) LIMIT 50
    """,
    "two_hop_method_chains": """
        PREFIX sg: <https://w3id.org/scigraph-nlp/ontology#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?a ?b ?c WHERE {
            ?x sg:usedFor ?y . ?y sg:usedFor ?z .
            ?x rdfs:label ?a . ?y rdfs:label ?b . ?z rdfs:label ?c .
        } LIMIT 50
    """,
}


def run_sparql(g: Graph, name: str) -> list[dict]:
    if name not in SPARQL_EXAMPLES:
        raise KeyError(name)
    res = g.query(SPARQL_EXAMPLES[name])
    return [{str(k): str(v) for k, v in row.asdict().items()} for row in res]


def validate_shacl(g: Graph) -> tuple[bool, str]:
    """Validate with pySHACL (optional dependency). Returns (conforms, report text)."""
    from pyshacl import validate

    shapes = Graph().parse(ONTOLOGY_DIR / "shapes.ttl", format="turtle")
    ont = Graph().parse(ONTOLOGY_DIR / "scigraph.ttl", format="turtle")
    conforms, _, text = validate(
        g, shacl_graph=shapes, ont_graph=ont, inference="rdfs", allow_warnings=True
    )
    return bool(conforms), str(text)


def ontology_graph() -> Graph:
    return Graph().parse(ONTOLOGY_DIR / "scigraph.ttl", format="turtle")


__all__ = [
    "SG",
    "DATA",
    "URIRef",
    "export_turtle",
    "run_sparql",
    "triples_to_rdf",
    "validate_shacl",
    "ontology_graph",
    "SPARQL_EXAMPLES",
]
