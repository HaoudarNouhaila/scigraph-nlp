"""Experiment 6: inference and graph construction time on reproducible inputs.

Measures, on the machine running the script:

* NER and RE inference time over the SciERC test split (551 sentences), excluding model
  loading, reported per sentence;
* end-to-end processing time (sentence splitting, NER, RE, graph building) for the
  three sample abstracts in data/samples;
* graph construction time (normalisation, deduplication) for the test split predictions;
* Neo4j ingestion time for that graph (only with --neo4j, which wipes the database).

Usage::

    python scripts/benchmark.py --repeats 3 [--neo4j]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

import torch

from scigraph.cli import _scierc_graph
from scigraph.data.scierc import load_split
from scigraph.models import relation as re_mod
from scigraph.pipeline.extractor import NeuralExtractor, process_text
from scigraph.pipeline.graph_builder import GraphBuilder
from scigraph.training.train_ner import predict as ner_predict


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ner", default="outputs/ner/best")
    p.add_argument("--re", default="outputs/re/best")
    p.add_argument("--predictions", default="outputs/pipeline_test.jsonl")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--neo4j", action="store_true", help="also time ingestion (wipes the DB)")
    p.add_argument("--out", type=Path, default=Path("reports/metrics/performance.json"))
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sents = load_split("data/processed", "test")
    ex = NeuralExtractor(args.ner, args.re, device=device)

    ner_times, re_times = [], []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        ents = ner_predict(
            ex.ner_model, ex.ner_tok, sents, ex.ner_model.config, 60, device, return_scores=True
        )
        ner_times.append(time.perf_counter() - t0)
        spans = [[e for e, _ in s] for s in ents]
        t0 = time.perf_counter()
        inst = re_mod.build_instances(ex.re_tok, sents, ex.re_model.config.max_seq_length, spans)
        with torch.no_grad():
            for b in range(0, len(inst), 64):
                batch = re_mod.collate(inst[b : b + 64], ex.re_tok.pad_token_id)
                batch.pop("labels")
                ex.re_model(**{k: v.to(device) for k, v in batch.items()})
        re_times.append(time.perf_counter() - t0)

    samples = [p.read_text(encoding="utf-8") for p in sorted(Path("data/samples").glob("*.txt"))]
    doc_times = []
    for _ in range(args.repeats):
        for text in samples:
            t0 = time.perf_counter()
            process_text(text, ex, builder=GraphBuilder())
            doc_times.append(time.perf_counter() - t0)

    build_times = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        kg = _scierc_graph("test", "data/processed", args.predictions).finalize()
        build_times.append(time.perf_counter() - t0)

    report = {
        "hardware": {
            "device": device,
            "cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "platform": platform.platform(terse=True),
            "torch": torch.__version__,
        },
        "protocol": (
            f"Median of {args.repeats} runs; model loading excluded; batch size 32 (NER) "
            "and 64 (RE); wall-clock time."
        ),
        "ner": {
            "sentences": len(sents),
            "seconds": round(statistics.median(ner_times), 2),
            "ms_per_sentence": round(1000 * statistics.median(ner_times) / len(sents), 1),
        },
        "re": {
            "candidate_pairs": len(inst),
            "seconds": round(statistics.median(re_times), 2),
            "ms_per_pair": round(1000 * statistics.median(re_times) / max(1, len(inst)), 2),
        },
        "end_to_end_sample_abstracts": {
            "documents": len(samples),
            "median_seconds_per_document": round(statistics.median(doc_times), 3),
        },
        "graph_construction_test_split": {
            "entities": len(kg.entities),
            "triples": len(kg.triples),
            "seconds": round(statistics.median(build_times), 3),
        },
    }
    if args.neo4j:
        from scigraph.graph.neo4j_store import Neo4jStore

        with Neo4jStore() as store:
            store.clear()
            store.create_schema()
            t0 = time.perf_counter()
            counts = store.ingest(kg, status="predicted", source=ex.name)
            report["neo4j_ingestion"] = {
                "written": counts,
                "seconds": round(time.perf_counter() - t0, 2),
            }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
