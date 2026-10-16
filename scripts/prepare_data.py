"""Validate the SciERC splits, convert them to sentence-level JSONL and report statistics.

Usage::

    python scripts/prepare_data.py --raw data/raw/scierc --out data/processed \
        --stats reports/metrics/dataset_statistics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scigraph.data.scierc import (
    SPLITS,
    compute_statistics,
    document_to_sentences,
    read_jsonl,
    write_jsonl,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, default=Path("data/raw/scierc"))
    p.add_argument("--out", type=Path, default=Path("data/processed"))
    p.add_argument("--stats", type=Path, default=Path("reports/metrics/dataset_statistics.json"))
    args = p.parse_args()

    stats = {}
    doc_keys: dict[str, set[str]] = {}
    for split in SPLITS:
        docs = read_jsonl(args.raw / f"{split}.json")
        doc_keys[split] = {d["doc_key"] for d in docs}
        sentences = [s for d in docs for s in document_to_sentences(d)]
        write_jsonl((s.to_dict() for s in sentences), args.out / f"{split}.jsonl")
        stats[split] = compute_statistics(docs)
        print(
            f"{split}: {stats[split]['documents']} docs, {stats[split]['sentences']} sentences, "
            f"{stats[split]['entities']} entities, {stats[split]['relations']} relations"
        )

    leaks = {
        f"{a}-{b}": sorted(doc_keys[a] & doc_keys[b])
        for i, a in enumerate(SPLITS)
        for b in SPLITS[i + 1 :]
    }
    stats["split_document_overlap"] = {k: len(v) for k, v in leaks.items()}
    if any(leaks.values()):
        raise SystemExit(f"Document leakage between splits: {leaks}")

    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Statistics written to {args.stats}")


if __name__ == "__main__":
    main()
