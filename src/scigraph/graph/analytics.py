"""Graph analytics with NetworkX on exported triples.

Works on any list of triple dictionaries with ``head_id``, ``head``, ``head_type``,
``relation``, ``tail_id``, ``tail``, ``tail_type`` (as returned by
:meth:`Neo4jStore.export_triples` or produced by :func:`triples_from_kg`).
"""

from __future__ import annotations

from collections import Counter

import networkx as nx

from scigraph.pipeline.graph_builder import KnowledgeGraph


def triples_from_kg(kg: KnowledgeGraph, status: str = "predicted") -> list[dict]:
    rows = []
    for t in kg.triples.values():
        h, tl = kg.entities[t.head_id], kg.entities[t.tail_id]
        rows.append(
            {
                "id": t.id,
                "head_id": h.id,
                "head": h.name,
                "head_type": h.type,
                "relation": t.relation.replace("-", "_"),
                "tail_id": tl.id,
                "tail": tl.name,
                "tail_type": tl.type,
                "score": t.score,
                "support": t.support,
                "status": status,
                "docs": sorted({ev.doc_id for ev in t.evidence}),
            }
        )
    return rows


def build_graph(triples: list[dict]) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for t in triples:
        g.add_node(t["head_id"], name=t["head"], type=t["head_type"])
        g.add_node(t["tail_id"], name=t["tail"], type=t["tail_type"])
        g.add_edge(
            t["head_id"],
            t["tail_id"],
            key=t.get("id"),
            relation=t["relation"],
            score=t.get("score"),
            support=t.get("support", 1),
            status=t.get("status", "predicted"),
        )
    return g


def summary(g: nx.MultiDiGraph) -> dict:
    und = nx.Graph(g)
    comps = list(nx.connected_components(und)) if und.number_of_nodes() else []
    return {
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "density": round(nx.density(und), 6) if und.number_of_nodes() > 1 else 0.0,
        "connected_components": len(comps),
        "largest_component": max((len(c) for c in comps), default=0),
        "node_types": dict(Counter(d.get("type") for _, d in g.nodes(data=True))),
        "relation_types": dict(Counter(d["relation"] for *_, d in g.edges(data=True))),
    }


def centrality(g: nx.MultiDiGraph, top_k: int = 20) -> dict[str, list[dict]]:
    if g.number_of_nodes() == 0:
        return {"degree": [], "pagerank": [], "betweenness": []}
    simple = nx.DiGraph(g)
    und = simple.to_undirected()
    measures = {
        "degree": nx.degree_centrality(und),
        "pagerank": nx.pagerank(simple, alpha=0.85),
        # Approximate betweenness on large graphs to keep it interactive.
        "betweenness": nx.betweenness_centrality(
            und,
            k=min(200, und.number_of_nodes()) if und.number_of_nodes() > 500 else None,
            seed=0,
        ),
    }
    out = {}
    for name, scores in measures.items():
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
        out[name] = [
            {
                "id": n,
                "name": g.nodes[n].get("name"),
                "type": g.nodes[n].get("type"),
                "score": round(s, 6),
            }
            for n, s in ranked
        ]
    return out


def communities(g: nx.MultiDiGraph, min_size: int = 3, top_k: int = 10) -> list[dict]:
    und = nx.Graph(g)
    if und.number_of_edges() == 0:
        return []
    comms = nx.community.greedy_modularity_communities(und)
    out = []
    for i, c in enumerate(sorted(comms, key=len, reverse=True)):
        if len(c) < min_size:
            continue
        sub = und.subgraph(c)
        hubs = sorted(sub.degree, key=lambda x: -x[1])[:5]
        out.append(
            {
                "community": i,
                "size": len(c),
                "types": dict(Counter(g.nodes[n].get("type") for n in c)),
                "hubs": [g.nodes[n].get("name") for n, _ in hubs],
            }
        )
        if len(out) >= top_k:
            break
    return out


def shortest_paths(
    g: nx.MultiDiGraph, source: str, target: str, max_hops: int = 4, limit: int = 10
) -> list[dict]:
    """Undirected shortest paths, reported with the original edge directions."""
    und = nx.MultiGraph(g)
    if source not in und or target not in und:
        return []
    out = []
    try:
        for path in nx.all_shortest_paths(und, source, target):
            if len(path) - 1 > max_hops:
                break
            rels = []
            for a, b in zip(path, path[1:], strict=False):
                if g.has_edge(a, b):
                    d = next(iter(g.get_edge_data(a, b).values()))
                    rels.append({"type": d["relation"], "from": a, "to": b})
                else:
                    d = next(iter(g.get_edge_data(b, a).values()))
                    rels.append({"type": d["relation"], "from": b, "to": a})
            out.append(
                {
                    "nodes": [
                        {"id": n, "name": g.nodes[n].get("name"), "type": g.nodes[n].get("type")}
                        for n in path
                    ],
                    "relations": rels,
                }
            )
            if len(out) >= limit:
                break
    except nx.NetworkXNoPath:
        return []
    return out


# Type pairs for which a missing link is a meaningful research hypothesis.
HYPOTHESIS_TYPES = {("Method", "Task"), ("Method", "Material"), ("Metric", "Task")}


def candidate_links(g: nx.MultiDiGraph, top_k: int = 20) -> list[dict]:
    """Rank unlinked entity pairs by the Adamic-Adar index of their shared neighbours.

    The output is a list of *hypotheses to validate*: two entities that share many
    neighbours but have no extracted relation. It is not evidence that a relation
    exists, and it is never written to the graph.
    """
    und = nx.Graph(g)
    cands = []
    for u in und:
        for v in nx.non_neighbors(und, u):
            tu, tv = g.nodes[u].get("type"), g.nodes[v].get("type")
            typed = (tu, tv) in HYPOTHESIS_TYPES or (tv, tu) in HYPOTHESIS_TYPES
            if u < v and typed and any(True for _ in nx.common_neighbors(und, u, v)):
                cands.append((u, v))
    scored = sorted(nx.adamic_adar_index(und, cands), key=lambda x: -x[2])[:top_k]
    return [
        {
            "a": g.nodes[u].get("name"),
            "a_type": g.nodes[u].get("type"),
            "b": g.nodes[v].get("name"),
            "b_type": g.nodes[v].get("type"),
            "shared_neighbours": sorted(
                g.nodes[n].get("name") for n in nx.common_neighbors(und, u, v)
            ),
            "adamic_adar": round(score, 4),
            "status": "hypothesis",
        }
        for u, v, score in scored
    ]
