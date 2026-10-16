"""Streamlit demonstration interface for SciGraph-NLP.

Run from the repository root::

    streamlit run app/streamlit_app.py

Pages follow the processing chain: document analysis (entities), relation extraction
(triples), the stored knowledge graph, graph exploration, predefined scientific queries,
and model evaluation. The app calls the ``scigraph`` package directly. Pages that need
Neo4j display a notice when the database is not reachable.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from scigraph.schema import ENTITY_TYPES

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "reports" / "metrics"
FIGURES = ROOT / "reports" / "figures"
TYPE_COLORS = {
    "Task": "#d9e7f5", "Method": "#e3efd9", "Metric": "#f6e3cf", "Material": "#ece0f1",
    "OtherScientificTerm": "#eeeeee", "Generic": "#f7f1c9",
}  # fmt: skip
STATUS_HELP = (
    "predicted: model output with evidence sentences. verified: gold annotation or "
    "human review. Inferred relations and hypotheses are shown separately."
)

st.set_page_config(page_title="SciGraph-NLP", layout="wide")


# -- shared resources -------------------------------------------------------------------
@st.cache_resource
def extractor(mode: str):
    from scigraph.pipeline.extractor import load_extractor

    return load_extractor(prefer=mode)


@st.cache_resource
def connect():
    from scigraph.graph.neo4j_store import Neo4jStore

    try:
        s = Neo4jStore()
        s.verify_connectivity()
        s.create_schema()
        return s
    except Exception as exc:  # noqa: BLE001 - shown to the user instead of a traceback
        return exc


def get_store():
    s = connect()
    if isinstance(s, Exception):
        st.info(
            "Neo4j is not reachable. Start it (README, section Installation) and set "
            f"NEO4J_URI and NEO4J_PASSWORD. Error: {type(s).__name__}"
        )
        st.stop()
    return s


def load_json(name: str):
    p = METRICS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def status_filter(key: str) -> list[str]:
    choice = st.radio(
        "Knowledge layer",
        ["predicted and verified", "verified only", "predicted only"],
        horizontal=True,
        key=key,
        help=STATUS_HELP,
    )
    return {
        "predicted and verified": ["predicted", "verified"],
        "verified only": ["verified"],
        "predicted only": ["predicted"],
    }[choice]


def highlight(sentence: dict) -> str:
    """Inline highlighting of the outermost entities; nested ones are listed in the table."""
    toks = sentence["tokens"]
    ents = sorted(sentence["entities"], key=lambda e: (e["start"], -e["end"]))
    out, i, k = [], 0, 0
    while i < len(toks):
        while k < len(ents) and ents[k]["start"] < i:
            k += 1
        if k < len(ents) and ents[k]["start"] == i:
            e = ents[k]
            text = html.escape(e.get("text") or " ".join(toks[e["start"] : e["end"] + 1]))
            out.append(
                f'<span style="background:{TYPE_COLORS[e["label"]]};padding:1px 4px;'
                f'border-radius:3px">{text} <small style="color:#555">{e["label"]}</small></span>'
            )
            i = e["end"] + 1
        else:
            out.append(html.escape(toks[i]))
            i += 1
    return " ".join(out)


def entity_picker(label: str, key: str) -> str | None:
    from scigraph.graph.queries import run_named_query

    text = st.text_input(label, key=f"{key}_text")
    if not text.strip():
        return None
    rows = run_named_query(get_store(), "search_entities", {"text": text, "limit": 20})
    if not rows:
        st.caption("No matching entity.")
        return None
    options = {f"{r['name']}  [{r['type']}, {r['status']}]": r["id"] for r in rows}
    return options[st.selectbox("Matches", list(options), key=f"{key}_sel")]


def dot_graph(edges: list[dict], focus: str | None = None) -> str:
    lines = [
        "digraph G {",
        'rankdir=LR; node [shape=box, style="rounded,filled", fontsize=10, fontname=Helvetica];',
        "edge [fontsize=8, fontname=Helvetica];",
    ]
    nodes = {}
    for e in edges:
        nodes[e["source"]] = (e["source_name"], e["source_type"])
        nodes[e["target"]] = (e["target_name"], e["target_type"])
    for nid, (name, typ) in nodes.items():
        pen = ", penwidth=2" if nid == focus else ""
        label = json.dumps(f"{name[:60]}\n({typ})")
        lines.append(f'"{nid}" [label={label}, fillcolor="{TYPE_COLORS.get(typ, "#fff")}"{pen}];')
    for e in edges:
        style = ', style="dashed"' if e.get("status") == "predicted" else ""
        lines.append(f'"{e["source"]}" -> "{e["target"]}" [label="{e["relation"]}"{style}];')
    lines.append("}")
    return "\n".join(lines)


def current_result():
    result = st.session_state.get("result")
    if result is None:
        st.info("Analyse a document on the Document Analysis page first.")
        st.stop()
    return result


# -- pages ------------------------------------------------------------------------------
def page_document_analysis():
    from scigraph.pipeline.extractor import process_pdf, process_text

    st.title("Document Analysis")
    st.write(
        "Input a scientific abstract or a PDF with extractable text. Sentences are split, "
        "and every span of up to eight tokens is classified into one of the SciERC entity "
        "types. Nested entities are kept."
    )
    mode = st.radio(
        "Extractor",
        ["auto", "rules"],
        horizontal=True,
        help="'auto' uses the trained SciBERT models when present, otherwise the rule baseline.",
    )
    ex = extractor(mode)
    st.caption(f"Active extractor: {ex.name}")
    source = st.radio("Input", ["Text", "PDF"], horizontal=True)
    if source == "Text":
        sample = (ROOT / "data" / "samples" / "abstract_1.txt").read_text(encoding="utf-8")
        text = st.text_area("Scientific text", value=sample, height=180, max_chars=20000)
        if st.button("Analyse", type="primary") and text.strip():
            st.session_state["result"] = process_text(text, ex, title="ui-input", source="ui")
    else:
        up = st.file_uploader(
            "PDF file (text layer required; scanned PDFs are not supported)", type=["pdf"]
        )
        if up is not None and st.button("Analyse", type="primary"):
            data = up.getvalue()
            if len(data) > 15 * 1024 * 1024:
                st.error("PDF larger than 15 MB.")
            else:
                st.session_state["result"] = process_pdf(data, ex, source=f"pdf:{up.name}")
    result = st.session_state.get("result")
    if not result:
        return
    kg = result["graph"]
    st.write(
        f"{result['n_sentences']} sentences, {len(kg.mentions)} entity mentions, "
        f"{len(kg.entities)} distinct entities, processed in {result['seconds']} s "
        f"by {result['extractor']}."
    )
    for s in result["sentences"]:
        st.markdown(highlight(s), unsafe_allow_html=True)
    rows = [
        {
            "sentence": s["index"],
            "page": s.get("page"),
            "entity": e["text"],
            "type": e["label"],
            "score": e["score"],
            "tokens": f"{e['start']}-{e['end']}",
        }
        for s in result["sentences"]
        for e in s["entities"]
    ]
    st.subheader("Entities")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def page_relation_extraction():
    from scigraph.pipeline.extractor import triples_table

    st.title("Relation Extraction")
    result = current_result()
    kg = result["graph"]
    st.write(
        "Each ordered pair of entities in a sentence is classified into one of seven relation "
        "types or no relation. Triples are then aggregated over mentions of the same entity."
    )
    rows = []
    for s in result["sentences"]:
        for r in s["relations"]:
            h, t = s["entities"][r["head"]], s["entities"][r["tail"]]
            rows.append(
                {
                    "sentence": s["index"],
                    "head": h["text"],
                    "relation": r["label"],
                    "tail": t["text"],
                    "score": r["score"],
                }
            )
    st.subheader("Predicted relations (per sentence)")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.subheader("Triples (subject, predicate, object)")
    st.dataframe(pd.DataFrame(triples_table(kg)), hide_index=True, use_container_width=True)
    if kg.candidate_duplicates:
        st.subheader("Possible duplicates (reported, not merged)")
        st.dataframe(pd.DataFrame(kg.candidate_duplicates), hide_index=True)
    st.download_button(
        "Download graph as JSON",
        json.dumps(kg.to_dict(), indent=2),
        file_name="extraction.json",
        mime="application/json",
    )
    if st.button("Store in Neo4j as predicted knowledge"):
        version = result.get("model_version") or {}
        counts = get_store().ingest(
            kg,
            status="predicted",
            source=result["extractor"],
            model_version=",".join(f"{k}:{v}" for k, v in version.items()),
        )
        st.success(f"Written: {counts}")


def page_knowledge_graph():
    from scigraph.graph import analytics

    st.title("Knowledge Graph")
    s = get_store()
    counts = s.counts()
    cols = st.columns(5)
    for col, key in zip(
        cols, ("documents", "entities", "mentions", "relations", "inferred"), strict=True
    ):
        col.metric(key.capitalize(), counts.get(key, 0))
    statuses = status_filter("kg_status")
    types = st.multiselect(
        "Entity types", ENTITY_TYPES, default=[t for t in ENTITY_TYPES if t != "Generic"]
    )
    max_nodes = st.slider("Most connected entities to display", 10, 120, 40)
    triples = [
        t for t in s.export_triples(statuses) if t["head_type"] in types and t["tail_type"] in types
    ]
    g = analytics.build_graph(triples)
    if g.number_of_nodes() == 0:
        st.write("No relation matches the filters.")
        return
    keep = {n for n, _ in sorted(g.degree, key=lambda x: -x[1])[:max_nodes]}
    edges = [
        {
            "source": t["head_id"],
            "source_name": t["head"],
            "source_type": t["head_type"],
            "target": t["tail_id"],
            "target_name": t["tail"],
            "target_type": t["tail_type"],
            "relation": t["relation"],
            "status": t["status"],
        }
        for t in triples
        if t["head_id"] in keep and t["tail_id"] in keep
    ]
    st.caption("Dashed edges are predicted, solid edges are verified.")
    st.graphviz_chart(dot_graph(edges), use_container_width=True)
    st.subheader("Structure")
    st.json(analytics.summary(g))
    cent = analytics.centrality(g, 15)
    c1, c2 = st.columns(2)
    c1.write("Degree centrality")
    c1.dataframe(pd.DataFrame(cent["degree"])[["name", "type", "score"]], hide_index=True)
    c2.write("PageRank")
    c2.dataframe(pd.DataFrame(cent["pagerank"])[["name", "type", "score"]], hide_index=True)
    st.write("Communities (greedy modularity)")
    st.dataframe(pd.DataFrame(analytics.communities(g, top_k=10)), hide_index=True)


def page_graph_exploration():
    from scigraph.graph.queries import find_paths, khop_neighbourhood, run_named_query

    st.title("Graph Exploration")
    s = get_store()
    statuses = status_filter("explore_status")
    eid = entity_picker("Search an entity", "explore")
    if eid:
        hops = st.slider("Neighbourhood depth (hops)", 1, 3, 1)
        edges = khop_neighbourhood(s, eid, hops, 150, statuses)
        if edges:
            st.graphviz_chart(dot_graph(edges, eid), use_container_width=True)
        st.subheader("Direct relations")
        st.dataframe(
            pd.DataFrame(
                run_named_query(s, "entity_neighbors", {"entity_id": eid, "statuses": statuses})
            ),
            hide_index=True,
        )
        st.subheader("Mentions in source documents")
        st.dataframe(
            pd.DataFrame(run_named_query(s, "entity_provenance", {"entity_id": eid})),
            hide_index=True,
        )

    st.subheader("Multi-hop paths between two entities")
    st.caption(
        "A path is a chain of extracted or verified relations. It is an indirect association, "
        "not evidence of a direct scientific relation between its endpoints."
    )
    c1, c2 = st.columns(2)
    with c1:
        a = entity_picker("From", "path_a")
    with c2:
        b = entity_picker("To", "path_b")
    max_hops = st.slider("Maximum hops", 1, 5, 3)
    if a and b and st.button("Find paths"):
        paths = find_paths(s, a, b, max_hops, 10, statuses)
        if not paths:
            st.write("No path within the hop limit.")
        edges = []
        for p in paths:
            names = {n["id"]: n for n in p["nodes"]}
            chain = []
            for r in p["relations"]:
                src, dst = names[r["from"]], names[r["to"]]
                chain.append(f"{src['name']} -[{r['type']}]-> {dst['name']}")
                edges.append(
                    {
                        "source": r["from"],
                        "source_name": src["name"],
                        "source_type": src["type"],
                        "relation": r["type"],
                        "target": r["to"],
                        "target_name": dst["name"],
                        "target_type": dst["type"],
                        "status": r["status"],
                    }
                )
            st.write("; ".join(chain))
        if edges:
            st.graphviz_chart(dot_graph(edges), use_container_width=True)

    st.subheader("Review predicted relations")
    queue = run_named_query(s, "review_queue", {"max_score": 1.0, "limit": 20})
    if not queue:
        st.caption("No predicted relation awaiting review.")
        return
    st.dataframe(pd.DataFrame(queue), hide_index=True)
    rid = st.selectbox("Relation", [r["relation_id"] for r in queue])
    reviewer = st.text_input("Reviewer name")
    c1, c2 = st.columns(2)
    for col, status in ((c1, "verified"), (c2, "rejected")):
        if col.button(f"Mark as {status}") and reviewer.strip():
            st.write(s.set_relation_status(rid, status, reviewer.strip()))


def page_scientific_queries():
    from scigraph.graph import analytics, reasoning
    from scigraph.graph.queries import QUERIES, run_named_query

    st.title("Scientific Queries")
    s = get_store()
    st.write("Predefined, parameterised Cypher queries. No free-form Cypher is accepted.")
    hidden = {"search_entities", "entity_neighbors", "entity_provenance", "review_queue"}
    name = st.selectbox("Query", [q for q in QUERIES if q not in hidden])
    q = QUERIES[name]
    st.caption(q.description)
    params: dict = {}
    for pname, spec in q.params.items():
        key = f"q_{name}_{pname}"
        if spec.choices:
            v = st.selectbox(pname, ["(any)", *spec.choices], key=key)
            params[pname] = None if v == "(any)" else v
        elif spec.kind is int:
            params[pname] = st.number_input(
                pname, spec.min or 1, spec.max or 1000, int(spec.default or 10), key=key
            )
        elif spec.kind is float:
            params[pname] = st.number_input(pname, 0.0, 1.0, float(spec.default or 1.0), key=key)
        else:
            params[pname] = st.text_input(pname, key=key)
    params["statuses"] = status_filter(f"{name}_status")
    if st.button("Run query", type="primary"):
        try:
            rows = run_named_query(s, name, params)
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        except Exception as exc:  # noqa: BLE001 - validation errors are shown to the user
            st.error(str(exc))
    with st.expander("Cypher statement"):
        st.code(q.cypher.strip(), language="cypher")

    st.subheader("Supporting sources of a relation")
    rid = st.text_input("Relation id (column relation_id in the results above)")
    if rid.strip():
        rows = run_named_query(s, "relation_evidence", {"relation_id": rid.strip()})
        if rows:
            st.json(rows[0])
        else:
            st.caption("Relation not found.")

    st.subheader("Inferred relations")
    st.caption("Derived by explicit rules from verified relations; stored as :INFERRED edges.")
    if st.button("Apply inference rules to verified relations"):
        st.write(reasoning.materialize(s, ("verified",)))
    st.dataframe(pd.DataFrame(reasoning.list_inferred(s, 50)), hide_index=True)

    st.subheader("Hypotheses to validate")
    st.caption(
        "Unlinked Method-Task, Method-Material and Metric-Task pairs ranked by the Adamic-Adar "
        "index of shared neighbours. These are candidates for review, not extracted facts."
    )
    if st.button("Compute hypotheses"):
        g = analytics.build_graph(s.export_triples(("verified", "predicted")))
        st.dataframe(pd.DataFrame(analytics.candidate_links(g, 20)), hide_index=True)


def page_model_evaluation():
    st.title("Model Evaluation")
    summary = load_json("summary.json")
    if summary:
        st.subheader("Test-set results")
        st.dataframe(pd.DataFrame(summary["rows"]), hide_index=True, use_container_width=True)
        for n in summary.get("notes", []):
            st.caption(n)
    else:
        st.caption("Run scripts/make_report.py after training to populate this page.")
    ner = load_json("ner_metrics.json")
    rel = load_json("re_metrics.json")
    if ner:
        st.subheader("NER per entity type (test)")
        st.dataframe(pd.DataFrame(ner["test"]["per_type"]).T)
    if rel:
        st.subheader("Relation extraction per type (test, gold entities)")
        a = rel["experiment_a_gold_entities"]["test"]["boundaries"]["per_type"]
        st.dataframe(pd.DataFrame(a).T)
    for f in sorted(FIGURES.glob("*.png")):
        st.image(str(f), caption=f.stem.replace("_", " "))
    stats = load_json("dataset_statistics.json")
    if stats:
        st.subheader("SciERC splits")
        keys = (
            "documents",
            "sentences",
            "tokens",
            "entities",
            "relations",
            "candidate_entity_pairs",
        )
        st.dataframe(
            pd.DataFrame({k: {f: stats[k][f] for f in keys} for k in ("train", "dev", "test")})
        )
    for name in ("graph_quality.json", "query_evaluation.json", "performance.json"):
        data = load_json(name)
        if data:
            st.subheader(name.removesuffix(".json").replace("_", " ").capitalize())
            st.json(data)


PAGES = [
    st.Page(page_document_analysis, title="Document Analysis", url_path="analysis", default=True),
    st.Page(page_relation_extraction, title="Relation Extraction", url_path="relations"),
    st.Page(page_knowledge_graph, title="Knowledge Graph", url_path="graph"),
    st.Page(page_graph_exploration, title="Graph Exploration", url_path="exploration"),
    st.Page(page_scientific_queries, title="Scientific Queries", url_path="queries"),
    st.Page(page_model_evaluation, title="Model Evaluation", url_path="evaluation"),
]

st.navigation(PAGES).run()
