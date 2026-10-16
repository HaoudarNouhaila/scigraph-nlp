"""Experiment 5: correctness and latency of graph queries against reference answers.

The SciERC gold graph of the test split is loaded into Neo4j and the same graph is
built in NetworkX. Reference answers are derived from the gold annotations and from
NetworkX, then compared with the answers of the parameterised Cypher queries:

1. methods_for_task returns every Method annotated as USED-FOR the queried Task;
2. entity_neighbors returns exactly the NetworkX neighbours of an entity;
3. shortest paths in Neo4j have the NetworkX length for connected pairs (up to 4 hops)
   and no path exists for pairs in different components;
4. relation_evidence returns the documents recorded in the gold graph.

The script WIPES the target database; run it only against a disposable instance.

Usage::

    python scripts/evaluate_queries.py --reset --samples 50
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import networkx as nx

from scigraph.cli import _scierc_graph
from scigraph.graph.analytics import build_graph, triples_from_kg
from scigraph.graph.neo4j_store import Neo4jStore
from scigraph.graph.queries import find_paths, run_named_query


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, 1000 * (time.perf_counter() - t0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reset", action="store_true", help="required: wipes the database")
    p.add_argument("--samples", type=int, default=50)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--out", type=Path, default=Path("reports/metrics/query_evaluation.json"))
    args = p.parse_args()
    if not args.reset:
        raise SystemExit("Pass --reset to confirm that the target database may be wiped.")

    rng = random.Random(args.seed)
    kg = _scierc_graph("test", "data/processed", None).finalize()
    g = build_graph(triples_from_kg(kg, status="verified"))
    und = nx.Graph(g)
    latency: dict[str, list[float]] = {}
    results: dict[str, dict] = {}

    with Neo4jStore() as store:
        store.clear()
        store.create_schema()
        t0 = time.perf_counter()
        store.ingest(kg, status="verified", source="scierc-gold-test")
        ingest_s = time.perf_counter() - t0

        # 1. methods_for_task
        cases = [
            (kg.entities[t.head_id], kg.entities[t.tail_id])
            for t in kg.triples.values()
            if t.relation == "USED-FOR"
            and kg.entities[t.head_id].type == "Method"
            and kg.entities[t.tail_id].type == "Task"
        ]
        cases = rng.sample(cases, min(args.samples, len(cases)))
        ok = 0
        for method, task in cases:
            rows, ms = timed(
                run_named_query, store, "methods_for_task", {"task": task.name, "limit": 200}
            )
            latency.setdefault("methods_for_task", []).append(ms)
            ok += any(r["method"] == method.name and r["task"] == task.name for r in rows)
        results["methods_for_task"] = {"cases": len(cases), "passed": ok}

        # 2. entity_neighbors vs NetworkX neighbourhood
        nodes = rng.sample(sorted(und.nodes), min(args.samples, und.number_of_nodes()))
        ok = 0
        for n in nodes:
            rows, ms = timed(
                run_named_query, store, "entity_neighbors", {"entity_id": n, "limit": 500}
            )
            latency.setdefault("entity_neighbors", []).append(ms)
            ok += {r["id"] for r in rows} == set(und.neighbors(n))
        results["entity_neighbors"] = {"cases": len(nodes), "passed": ok}

        # 3. shortest paths, connected and disconnected pairs
        comps = [c for c in nx.connected_components(und) if len(c) >= 3]
        connected = []
        for _ in range(100 * args.samples):  # bounded rejection sampling
            if len(connected) >= args.samples or not comps:
                break
            a, b = rng.sample(sorted(rng.choice(comps)), 2)
            d = nx.shortest_path_length(und, a, b)
            if 2 <= d <= 4:
                connected.append((a, b, d))
        ok = 0
        for a, b, d in connected:
            paths, ms = timed(find_paths, store, a, b, 4, 5)
            latency.setdefault("shortest_paths", []).append(ms)
            ok += bool(paths) and len(paths[0]["relations"]) == d
        all_comps = list(nx.connected_components(und))
        disconnected = []
        for _ in range(args.samples):
            c1, c2 = rng.sample(all_comps, 2)
            disconnected.append((rng.choice(sorted(c1)), rng.choice(sorted(c2))))
        ok_none = sum(not find_paths(store, a, b, 4, 5) for a, b in disconnected)
        results["shortest_paths"] = {
            "connected_cases": len(connected),
            "connected_passed": ok,
            "disconnected_cases": len(disconnected),
            "disconnected_passed": ok_none,
        }

        # 4. relation provenance
        triples = rng.sample(
            sorted(kg.triples.values(), key=lambda t: t.id), min(args.samples, len(kg.triples))
        )
        ok = 0
        for t in triples:
            rows, ms = timed(run_named_query, store, "relation_evidence", {"relation_id": t.id})
            latency.setdefault("relation_evidence", []).append(ms)
            ok += bool(rows) and set(rows[0]["documents"]) == {e.doc_id for e in t.evidence}
        results["relation_evidence"] = {"cases": len(triples), "passed": ok}
        store.clear()

    report = {
        "protocol": (
            "Gold graph of the SciERC test split (100 documents) ingested into Neo4j 5.26; "
            "reference answers from gold annotations and NetworkX; random samples with seed "
            f"{args.seed}. Latency measured client-side on a local instance (Python driver)."
        ),
        "ingestion_seconds": round(ingest_s, 2),
        "graph": {"entities": len(kg.entities), "relations": len(kg.triples)},
        "results": results,
        "latency_ms": {
            k: {
                "median": round(statistics.median(v), 2),
                "p95": round(sorted(v)[int(0.95 * (len(v) - 1))], 2),
                "n": len(v),
            }
            for k, v in latency.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
