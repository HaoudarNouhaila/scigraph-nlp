"""Experiment 4: quality of the knowledge graph built from predicted extractions.

Compares the graph built from the end-to-end predictions on the SciERC test split
(scripts/predict_scierc.py) with the graph built from the gold annotations of the same
documents, after normalisation and deduplication. Reported quantities:

* graph-level triple precision / recall / F1: a predicted triple is correct when a gold
  triple of the same document has the same relation and the same (type, normalised key)
  for head and tail;
* duplicates: near-duplicate entity pairs reported by the builder, mentions per entity;
* type-signature validity: share of predicted triples whose (head type, relation,
  tail type) signature never occurs in the training annotations;
* structure: self-loops, SHACL conformance of the RDF export;
* provenance: share of triples and mentions that carry a document and a sentence.

Usage::

    python scripts/evaluate_graph.py --predictions outputs/pipeline_test.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scigraph.cli import _scierc_graph
from scigraph.data.scierc import load_split
from scigraph.evaluation.metrics import prf
from scigraph.graph.analytics import triples_from_kg
from scigraph.graph.rdf_export import triples_to_rdf, validate_shacl


def fact_keys(kg) -> set[tuple]:
    """Document-level facts: (doc, head type, head key, relation, tail type, tail key)."""
    out = set()
    for t in kg.triples.values():
        h, tl = kg.entities[t.head_id], kg.entities[t.tail_id]
        for ev in t.evidence:
            key = (ev.doc_id, h.type, h.key, t.relation, tl.type, tl.key)
            if t.relation in ("COMPARE", "CONJUNCTION"):
                a, b = sorted([(h.type, h.key), (tl.type, tl.key)])
                key = (ev.doc_id, *a, t.relation, *b)
            out.add(key)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", default="outputs/pipeline_test.jsonl")
    p.add_argument("--data", default="data/processed")
    p.add_argument("--out", type=Path, default=Path("reports/metrics/graph_quality.json"))
    args = p.parse_args()

    gold_kg = _scierc_graph("test", args.data, None).finalize()
    pred_kg = _scierc_graph("test", args.data, args.predictions).finalize()

    gold_f, pred_f = fact_keys(gold_kg), fact_keys(pred_kg)
    untyped = lambda keys: {(k[0], k[2], k[3], k[5]) for k in keys}  # noqa: E731
    triple_prf = prf(len(gold_f & pred_f), len(pred_f), len(gold_f))
    untyped_prf = prf(
        len(untyped(gold_f) & untyped(pred_f)), len(untyped(pred_f)), len(untyped(gold_f))
    )

    train_sigs = Counter(
        (r.head.label, r.label, r.tail.label)
        for s in load_split(args.data, "train")
        for r in s.relations
    )
    pred_sigs = Counter(
        (pred_kg.entities[t.head_id].type, t.relation, pred_kg.entities[t.tail_id].type)
        for t in pred_kg.triples.values()
    )
    unseen = {
        f"{h} {r} {t}": n for (h, r, t), n in pred_sigs.items() if (h, r, t) not in train_sigs
    }

    def graph_stats(kg) -> dict:
        n_ent = len(kg.entities)
        rows = triples_from_kg(kg)
        conforms, report = validate_shacl(triples_to_rdf(rows))
        return {
            "documents": len(kg.documents),
            "mentions": len(kg.mentions),
            "entities": n_ent,
            "global_entities": sum(e.scope == "global" for e in kg.entities.values()),
            "mentions_per_entity": round(len(kg.mentions) / n_ent, 3) if n_ent else 0,
            "entities_in_several_documents": sum(len(e.doc_ids) > 1 for e in kg.entities.values()),
            "entities_with_aliases": sum(bool(e.aliases) for e in kg.entities.values()),
            "triples": len(kg.triples),
            "triples_supported_by_several_documents": sum(
                t.support > 1 for t in kg.triples.values()
            ),
            "candidate_duplicate_pairs": len(kg.candidate_duplicates),
            "self_loops": sum(t.head_id == t.tail_id for t in kg.triples.values()),
            "triples_with_evidence_pct": round(
                100 * sum(bool(t.evidence) for t in kg.triples.values()) / max(1, len(kg.triples)),
                2,
            ),
            "mentions_with_document_pct": round(
                100 * sum(bool(m.doc_id) for m in kg.mentions.values()) / max(1, len(kg.mentions)),
                2,
            ),
            "shacl_conforms": conforms,
            "shacl_warnings": report.count("Severity: sh:Warning"),
            "shacl_violations": report.count("Severity: sh:Violation"),
        }

    result = {
        "protocol": (
            "Graphs built from SciERC test documents (100 abstracts). Facts are compared per "
            "document after normalisation: (head type, head key, relation, tail type, tail key); "
            "symmetric relations are unordered."
        ),
        "gold_graph": graph_stats(gold_kg),
        "predicted_graph": graph_stats(pred_kg),
        "fact_level": {"typed": triple_prf, "untyped_entities": untyped_prf},
        "predicted_triples_with_signature_unseen_in_train": sum(unseen.values()),
        "predicted_triples_with_signature_unseen_in_train_pct": round(
            100 * sum(unseen.values()) / max(1, len(pred_kg.triples)), 2
        ),
        "unseen_signatures": dict(sorted(unseen.items(), key=lambda kv: -kv[1])[:15]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "protocol"}, indent=2))


if __name__ == "__main__":
    main()
