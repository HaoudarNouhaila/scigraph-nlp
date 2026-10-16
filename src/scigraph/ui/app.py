"""Streamlit interface for SciGraph-NLP.

Run with::

    streamlit run src/scigraph/ui/app.py

Pages: Overview, Extract, Graph Explorer, Paths, Analytics, Evaluation. The UI calls the
Python package directly; Neo4j-backed pages show a notice when the database is not
reachable instead of failing.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from scigraph.schema import ENTITY_TYPES

ROOT = Path(__file__).resolve().parents[3]
METRICS = ROOT / "reports" / "metrics"
TYPE_COLORS = {
    "Task": "#d9e7f5", "Method": "#e3efd9", "Metric": "#f6e3cf", "Material": "#ece0f1",
    "OtherScientificTerm": "#eeeeee", "Generic": "#f7f1c9",
}  # fmt: skip

st.set_page_config(page_title="SciGraph-NLP", layout="wide")


# -- shared resources -------------------------------------------------------------------
@st.cache_resource
def extractor(mode: str):
    from scigraph.pipeline.extractor import load_extractor

    return load_extractor(prefer=mode)


@st.cache_resource
def store():
    from scigraph.graph.neo4j_store import Neo4jStore

    try:
        s = Neo4jStore()
        s.verify_connectivity()
        s.create_schema()
        return s
    except Exception as exc:  # noqa: BLE001
        return exc


def get_store():
    s = store()
    if isinstance(s, Exception):
        st.info(
            "Neo4j is not reachable. Start it (see README, section 'Neo4j') and set "
            f"NEO4J_URI / NEO4J_PASSWORD. Error: {type(s).__name__}"
        )
        st.stop()
    return s


def load_json(name: str):
    p = METRICS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def highlight(sentence: dict) -> str:
    """Inline highlighting of non-overlapping outermost entities in a sentence."""
    toks = sentence["tokens"]
    ents = sorted(sentence["entities"], key=lambda e: (e["start"], -e["end"]))
    out, i, k = [], 0, 0
    while i < len(toks):
        while k < len(ents) and ents[k]["start"] < i:
            k += 1
        if k < len(ents) and ents[k]["start"] == i:
            e = ents[k]
            text = html.escape(" ".join(toks[e["start"] : e["end"] + 1]))
            out.append(
                f'<span style="background:{TYPE_COLORS[e["label"]]};padding:1px 4px;'
                f'border-radius:3px">{text} <small style="color:#555">{e["label"]}</small></span>'
            )
            i = e["end"] + 1
        else:
            out.append(html.escape(toks[i]))
            i += 1
    return " ".join(out)


def entity_picker(label: str, key: str):
    from scigraph.graph.queries import run_named_query

    text = st.text_input(label, key=f"{key}_text")
    if not text.strip():
        return None
    rows = run_named_query(get_store(), "search_entities", {"text": text, "limit": 20})
    if not rows:
        st.caption("No matching entity.")
        return None
    options = {f"{r['name']}  [{r['type']}, {r['status']}]": r["id"] for r in rows}
    choice = st.selectbox("Matches", list(options), key=f"{key}_sel")
    return options[choice]


def dot_graph(edges: list[dict], focus: str | None = None) -> str:
    lines = [
        "digraph G {",
        'rankdir=LR; node [shape=box, style="rounded,filled", fontsize=10];',
        "edge [fontsize=8];",
    ]
    nodes = {}
    for e in edges:
        nodes[e["source"]] = (e["source_name"], e["source_type"])
        nodes[e["target"]] = (e["target_name"], e["target_type"])
    for nid, (name, typ) in nodes.items():
        pen = ", penwidth=2" if nid == focus else ""
        label = json.dumps(f"{name}\n({typ})")
        lines.append(f'"{nid}" [label={label}, fillcolor="{TYPE_COLORS.get(typ, "#fff")}"{pen}];')
    for e in edges:
        lines.append(f'"{e["source"]}" -> "{e["target"]}" [label="{e["relation"]}"];')
    lines.append("}")
    return "\n".join(lines)


# -- pages ------------------------------------------------------------------------------
def page_overview():
    st.title("SciGraph-NLP")
    st.write(
        "Automated construction of a scientific knowledge graph from abstracts and papers. "
        "Entities (Task, Method, Metric, Material, OtherScientificTerm, Generic) and relations "
        "(USED-FOR, FEATURE-OF, HYPONYM-OF, PART-OF, COMPARE, CONJUNCTION, EVALUATE-FOR) are "
        "extracted with SciBERT models trained on SciERC, normalised, deduplicated and stored "
        "in Neo4j with provenance."
    )
    stats = load_json("dataset_statistics.json")
    if stats:
        st.subheader("SciERC splits")
        rows = [
            {
                "split": k,
                **{f: v[f] for f in ("documents", "sentences", "tokens", "entities", "relations")},
            }
            for k, v in stats.items()
            if isinstance(v, dict) and "documents" in v
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        c1, c2 = st.columns(2)
        c1.bar_chart(pd.Series(stats["train"]["entity_types"], name="train entities"))
        c2.bar_chart(pd.Series(stats["train"]["relation_types"], name="train relations"))
    summary = load_json("summary.json")
    if summary:
        st.subheader("Measured results (test split)")
        st.dataframe(pd.DataFrame(summary["rows"]), hide_index=True)
    else:
        st.caption("Model results appear here once reports/metrics/summary.json exists.")


def page_extract():
    st.title("Extract")
    mode = st.radio(
        "Extractor",
        ["auto", "rules"],
        horizontal=True,
        help="'auto' uses the trained models when available, else the rule baseline.",
    )
    ex = extractor(mode)
    st.caption(f"Active extractor: {ex.name}")
    src = st.radio("Input", ["Text", "PDF"], horizontal=True)
    result = None
    from scigraph.pipeline.extractor import process_pdf, process_text, triples_table

    if src == "Text":
        text = st.text_area("Scientific text", height=200, max_chars=20000)
        if st.button("Extract", type="primary") and text.strip():
            result = process_text(text, ex, title="ui-input", source="ui")
    else:
        up = st.file_uploader("PDF file", type=["pdf"])
        if up is not None and st.button("Extract", type="primary"):
            data = up.getvalue()
            if len(data) > 15 * 1024 * 1024:
                st.error("PDF larger than 15 MB.")
            else:
                result = process_pdf(data, ex, source=f"pdf:{up.name}")
    if result:
        st.session_state["last_result"] = result
    result = st.session_state.get("last_result")
    if not result:
        return
    kg = result["graph"]
    st.write(
        f"{result['n_sentences']} sentences, {len(kg.entities)} entities, "
        f"{len(kg.triples)} triples, {result['seconds']} s"
    )
    with st.expander("Annotated sentences", expanded=True):
        for s in result["sentences"]:
            st.markdown(highlight(s), unsafe_allow_html=True)
    st.subheader("Triples")
    st.dataframe(pd.DataFrame(triples_table(kg)), hide_index=True)
    if kg.candidate_duplicates:
        st.subheader("Possible duplicates (not merged)")
        st.dataframe(pd.DataFrame(kg.candidate_duplicates), hide_index=True)
    if st.button("Store in Neo4j as predicted knowledge"):
        counts = get_store().ingest(kg, status="predicted", source=result["extractor"])
        st.success(f"Written: {counts}")


def page_explorer():
    from scigraph.graph.queries import QUERIES, khop_neighbourhood, run_named_query

    st.title("Graph Explorer")
    s = get_store()
    st.write(s.counts())
    eid = entity_picker("Search an entity", "explore")
    if eid:
        hops = st.slider("Neighbourhood depth", 1, 3, 1)
        edges = khop_neighbourhood(s, eid, hops, 150)
        if edges:
            st.graphviz_chart(dot_graph(edges, eid))
        st.subheader("Relations")
        st.dataframe(
            pd.DataFrame(run_named_query(s, "entity_neighbors", {"entity_id": eid})),
            hide_index=True,
        )
        st.subheader("Provenance")
        st.dataframe(
            pd.DataFrame(run_named_query(s, "entity_provenance", {"entity_id": eid})),
            hide_index=True,
        )

    st.subheader("Predefined queries")
    name = st.selectbox("Query", [q for q in QUERIES if q not in ("search_entities",)])
    q = QUERIES[name]
    st.caption(q.description)
    params = {}
    for pname, spec in q.params.items():
        if spec.choices:
            v = st.selectbox(pname, ["(any)", *spec.choices], key=f"q_{name}_{pname}")
            params[pname] = None if v == "(any)" else v
        elif spec.kind is int:
            params[pname] = st.number_input(
                pname,
                spec.min or 1,
                spec.max or 1000,
                int(spec.default or 10),
                key=f"q_{name}_{pname}",
            )
        elif spec.kind is float:
            params[pname] = st.number_input(
                pname, 0.0, 1.0, float(spec.default or 1.0), key=f"q_{name}_{pname}"
            )
        else:
            params[pname] = st.text_input(pname, key=f"q_{name}_{pname}")
    verified_only = st.checkbox("Verified knowledge only")
    params["statuses"] = ["verified"] if verified_only else ["predicted", "verified"]
    if st.button("Run query"):
        try:
            st.dataframe(pd.DataFrame(run_named_query(s, name, params)), hide_index=True)
        except Exception as exc:  # noqa: BLE001 - validation errors shown to the user
            st.error(str(exc))

    st.subheader("Review predicted relations")
    queue = run_named_query(s, "review_queue", {"max_score": 1.0, "limit": 20})
    if queue:
        st.dataframe(pd.DataFrame(queue), hide_index=True)
        rid = st.selectbox("Relation", [r["relation_id"] for r in queue])
        reviewer = st.text_input("Reviewer name")
        c1, c2 = st.columns(2)
        for col, status in ((c1, "verified"), (c2, "rejected")):
            if col.button(f"Mark {status}") and reviewer.strip():
                st.write(s.set_relation_status(rid, status, reviewer.strip()))


def page_paths():
    from scigraph.graph.queries import find_paths

    st.title("Multi-hop Paths")
    s = get_store()
    c1, c2 = st.columns(2)
    with c1:
        a = entity_picker("From", "path_a")
    with c2:
        b = entity_picker("To", "path_b")
    hops = st.slider("Maximum hops", 1, 5, 3)
    verified_only = st.checkbox("Verified knowledge only", key="paths_verified")
    if a and b and st.button("Find paths", type="primary"):
        statuses = ("verified",) if verified_only else ("predicted", "verified")
        paths = find_paths(s, a, b, hops, 10, statuses)
        if not paths:
            st.write("No path within the hop limit.")
        edges = []
        for p in paths:
            names = {n["id"]: n for n in p["nodes"]}
            chain = []
            for r in p["relations"]:
                chain.append(
                    f"{names[r['from']]['name']} -[{r['type']}]-> {names[r['to']]['name']}"
                )
                edges.append(
                    {
                        "source": r["from"],
                        "source_name": names[r["from"]]["name"],
                        "source_type": names[r["from"]]["type"],
                        "relation": r["type"],
                        "target": r["to"],
                        "target_name": names[r["to"]]["name"],
                        "target_type": names[r["to"]]["type"],
                    }
                )
            st.write("; ".join(chain))
        if edges:
            st.graphviz_chart(dot_graph(edges))


def page_analytics():
    from scigraph.graph import analytics

    st.title("Graph Analytics")
    s = get_store()
    verified_only = st.checkbox("Verified knowledge only", key="an_verified")
    triples = s.export_triples(("verified",) if verified_only else ("predicted", "verified"))
    g = analytics.build_graph(triples)
    summ = analytics.summary(g)
    st.json(summ)
    if summ["nodes"] == 0:
        return
    top_k = st.slider("Top entities", 5, 50, 15)
    cent = analytics.centrality(g, top_k)
    cols = st.columns(3)
    for col, name in zip(cols, ("degree", "pagerank", "betweenness"), strict=True):
        col.subheader(name.capitalize())
        col.dataframe(pd.DataFrame(cent[name])[["name", "type", "score"]], hide_index=True)
    st.subheader("Communities (greedy modularity)")
    st.dataframe(pd.DataFrame(analytics.communities(g, top_k=top_k)), hide_index=True)


def page_evaluation():
    st.title("Evaluation")
    summary = load_json("summary.json")
    if summary:
        st.subheader("Summary")
        st.dataframe(pd.DataFrame(summary["rows"]), hide_index=True)
        if summary.get("notes"):
            for n in summary["notes"]:
                st.caption(n)
    base = load_json("baseline.json")
    ner = load_json("ner_metrics.json")
    rel = load_json("re_metrics.json")
    if ner:
        st.subheader("NER per entity type (test)")
        st.dataframe(pd.DataFrame(ner["test"]["per_type"]).T)
    if rel:
        st.subheader("Relation extraction per type (test, boundaries)")
        a = rel["experiment_a_gold_entities"]["test"]["boundaries"]["per_type"]
        st.dataframe(pd.DataFrame(a).T)
        st.subheader("Confusion on gold-entity pairs (test, Experiment A)")
        conf = rel["experiment_a_gold_entities"]["test"]["confusion"]
        st.dataframe(pd.DataFrame(conf).T.fillna(0).astype(int))
    if base:
        st.subheader("Rule baseline (test)")
        st.json(
            {
                "ner_f1": base["test"]["ner"]["micro"]["f1"],
                "re_gold_entities_f1": base["test"]["relations_gold_entities"]["boundaries"][
                    "micro"
                ]["f1"],
                "re_predicted_entities_f1": base["test"]["relations_predicted_entities"][
                    "boundaries"
                ]["micro"]["f1"],
            }
        )
    figs = sorted((ROOT / "reports" / "figures").glob("*.png"))
    for f in figs:
        st.image(str(f), caption=f.stem)
    if not (ner or rel):
        st.caption("Neural model metrics appear once training has produced them.")


PAGES = [
    st.Page(page_overview, title="Overview", url_path="overview", default=True),
    st.Page(page_extract, title="Extract", url_path="extract"),
    st.Page(page_explorer, title="Graph Explorer", url_path="explorer"),
    st.Page(page_paths, title="Paths", url_path="paths"),
    st.Page(page_analytics, title="Analytics", url_path="analytics"),
    st.Page(page_evaluation, title="Evaluation", url_path="evaluation"),
]

assert set(TYPE_COLORS) == set(ENTITY_TYPES)
st.navigation(PAGES).run()
