"""Command-line interface.

Examples::

    scigraph extract data/samples/abstract_1.txt --out result.json
    scigraph ingest data/samples --status predicted
    scigraph ingest-gold --split test          # SciERC annotations as verified knowledge
    scigraph ingest-predictions --split test   # model predictions on the same documents
    scigraph export-ttl graph.ttl
    scigraph analytics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scigraph.pipeline.graph_builder import GraphBuilder


def _extractor(mode: str):
    from scigraph.pipeline.extractor import load_extractor

    return load_extractor(prefer=mode)


def cmd_extract(args) -> None:
    from scigraph.pipeline.extractor import process_pdf, process_text, triples_table

    ex = _extractor(args.extractor)
    path = Path(args.input)
    if path.suffix.lower() == ".pdf":
        result = process_pdf(str(path), ex, source=f"pdf:{path.name}")
    else:
        result = process_text(
            path.read_text(encoding="utf-8"), ex, title=path.stem, source=f"file:{path.name}"
        )
    out = {
        "doc_id": result["doc_id"],
        "extractor": result["extractor"],
        "triples": triples_table(result["graph"]),
        "graph": result["graph"].to_dict(),
    }
    text = json.dumps(out, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"{len(out['triples'])} triples written to {args.out}")
    else:
        print(text)


def cmd_ingest(args) -> None:
    from scigraph.graph.neo4j_store import Neo4jStore
    from scigraph.pipeline.extractor import model_version_string, process_pdf, process_text

    ex = _extractor(args.extractor)
    files = sorted(p for p in Path(args.path).rglob("*") if p.suffix.lower() in (".txt", ".pdf"))
    builder = GraphBuilder()
    for f in files:
        if f.suffix.lower() == ".pdf":
            process_pdf(str(f), ex, source=f"pdf:{f.name}", builder=builder)
        else:
            process_text(
                f.read_text(encoding="utf-8"),
                ex,
                title=f.stem,
                source=f"file:{f.name}",
                builder=builder,
            )
        print("processed", f.name)
    kg = builder.finalize()
    with Neo4jStore() as store:
        store.create_schema()
        print(
            store.ingest(
                kg, status="predicted", source=ex.name, model_version=model_version_string(ex)
            )
        )


def _scierc_graph(split: str, data_dir: str, predictions: str | None) -> GraphBuilder:
    """Graph built from SciERC sentences, from gold annotations or from prediction files."""
    from scigraph.data.scierc import load_split, read_jsonl
    from scigraph.pipeline.normalize import detokenize, unescape_ptb
    from scigraph.schema import NO_RELATION

    sents = load_split(data_dir, split)
    pred = None
    if predictions:
        pred = {(r["doc_key"], r["sent_id"]): r for r in read_jsonl(predictions)}
    by_doc: dict[str, list] = {}
    for s in sents:
        by_doc.setdefault(s.doc_key, []).append(s)
    builder = GraphBuilder()
    for doc_key, doc_sents in by_doc.items():
        packed = []
        sent_texts: list[str] = []
        for s in doc_sents:
            if pred is None:
                ents = [(e.start, e.end, e.label) for e in s.entities]
                rels = [
                    (r.head.start, r.head.end, r.tail.start, r.tail.end, r.label)
                    for r in s.relations
                ]
                ents = [(*e, 1.0) for e in ents]
                rels = [(*r, 1.0) for r in rels]
            else:
                # Output of scripts/predict_scierc.py: [start, end, label, score] and
                # [s1, e1, s2, e2, label, score].
                row = pred[(s.doc_key, s.sent_id)]
                ents = [tuple(e) for e in row["entities"]]
                rels = [tuple(r) for r in row["relations"]]
            index = {(e[0], e[1]): k for k, e in enumerate(ents)}
            tokens = unescape_ptb(s.tokens)
            sent_text, rel_offsets = detokenize(s.tokens)
            base = sum(len(t) + 1 for t in sent_texts)  # offset in " ".join(sent_texts)
            sent_texts.append(sent_text)
            packed.append(
                {
                    "index": s.sent_id,
                    "text": sent_text,
                    "char_start": base,
                    "tokens": tokens,
                    "token_offsets": [(base + a, base + b) for a, b in rel_offsets],
                    "entities": [
                        {"start": a, "end": b, "label": lab, "score": sc} for a, b, lab, sc in ents
                    ],
                    "relations": [
                        {"head": index[(a, b)], "tail": index[(c, d)], "label": lab, "score": sc}
                        for a, b, c, d, lab, sc in rels
                        if lab != NO_RELATION and (a, b) in index and (c, d) in index
                    ],
                }
            )
        text = " ".join(sent_texts)
        builder.add_document(
            f"scierc:{doc_key}", {"title": doc_key, "source": f"scierc-{split}"}, text, packed
        )
    return builder


def cmd_ingest_gold(args) -> None:
    from scigraph.graph.neo4j_store import Neo4jStore

    kg = _scierc_graph(args.split, args.data, None).finalize()
    with Neo4jStore() as store:
        store.create_schema()
        print(store.ingest(kg, status="verified", source=f"scierc-gold-{args.split}"))


def cmd_ingest_predictions(args) -> None:
    from scigraph.graph.neo4j_store import Neo4jStore
    from scigraph.pipeline.extractor import dir_fingerprint

    kg = _scierc_graph(args.split, args.data, args.predictions).finalize()
    version = ",".join(f"{m}:{dir_fingerprint(Path(f'outputs/{m}/best'))}" for m in ("ner", "re"))
    with Neo4jStore() as store:
        store.create_schema()
        print(
            store.ingest(
                kg, status="predicted", source="scibert-span-ner+marker-re", model_version=version
            )
        )


def cmd_export_ttl(args) -> None:
    from scigraph.graph.neo4j_store import Neo4jStore
    from scigraph.graph.rdf_export import export_turtle

    statuses = ("verified",) if args.verified_only else ("predicted", "verified")
    with Neo4jStore() as store:
        n = export_turtle(store.export_triples(statuses), args.out)
    print(f"{n} RDF triples written to {args.out}")


def cmd_analytics(args) -> None:
    from scigraph.graph import analytics
    from scigraph.graph.neo4j_store import Neo4jStore

    with Neo4jStore() as store:
        g = analytics.build_graph(store.export_triples())
    print(
        json.dumps(
            {
                "summary": analytics.summary(g),
                "centrality": analytics.centrality(g, args.top_k),
                "communities": analytics.communities(g, top_k=args.top_k),
            },
            indent=2,
        )
    )


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="scigraph")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="extract triples from a .txt or .pdf file")
    e.add_argument("input")
    e.add_argument("--out")
    e.add_argument("--extractor", choices=["auto", "neural", "rules"], default="auto")
    e.set_defaults(func=cmd_extract)

    i = sub.add_parser("ingest", help="extract from a folder of .txt/.pdf and write to Neo4j")
    i.add_argument("path")
    i.add_argument("--extractor", choices=["auto", "neural", "rules"], default="auto")
    i.set_defaults(func=cmd_ingest)

    g = sub.add_parser("ingest-gold", help="write SciERC gold annotations as verified knowledge")
    g.add_argument("--split", default="test", choices=["train", "dev", "test"])
    g.add_argument("--data", default="data/processed")
    g.set_defaults(func=cmd_ingest_gold)

    pr = sub.add_parser("ingest-predictions", help="write model predictions on SciERC to Neo4j")
    pr.add_argument("--split", default="test", choices=["dev", "test"])
    pr.add_argument("--data", default="data/processed")
    pr.add_argument(
        "--predictions",
        default="outputs/pipeline_test.jsonl",
        help="output of scripts/predict_scierc.py",
    )
    pr.set_defaults(func=cmd_ingest_predictions)

    x = sub.add_parser("export-ttl", help="export the Neo4j graph as Turtle")
    x.add_argument("out")
    x.add_argument("--verified-only", action="store_true")
    x.set_defaults(func=cmd_export_ttl)

    a = sub.add_parser("analytics", help="NetworkX analytics on the Neo4j graph")
    a.add_argument("--top-k", type=int, default=10)
    a.set_defaults(func=cmd_analytics)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
