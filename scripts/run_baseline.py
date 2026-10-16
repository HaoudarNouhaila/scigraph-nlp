"""Evaluate the rule-based spaCy baseline on SciERC dev and test.

Relations are evaluated twice: on gold entities (Experiment A) and on the baseline's own
entities (Experiment B), mirroring the neural evaluation protocol.

Usage::

    python scripts/run_baseline.py --data data/processed --out reports/metrics/baseline.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scigraph.baseline.rules import rule_entities, rule_relations
from scigraph.data.scierc import load_split
from scigraph.evaluation.metrics import entity_scores, relation_scores


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    p.add_argument("--out", type=Path, default=Path("reports/metrics/baseline.json"))
    args = p.parse_args()

    results = {}
    for split in ("dev", "test"):
        sents = load_split(args.data, split)
        pred_ents = [rule_entities(s.tokens) for s in sents]
        gold_ents = [s.entities for s in sents]
        gold_rels = [s.relations for s in sents]
        rels_a = [rule_relations(s.tokens, s.entities) for s in sents]
        rels_b = [rule_relations(s.tokens, e) for s, e in zip(sents, pred_ents, strict=True)]
        results[split] = {
            "ner": entity_scores(gold_ents, pred_ents),
            "relations_gold_entities": relation_scores(gold_rels, rels_a),
            "relations_predicted_entities": relation_scores(gold_rels, rels_b),
        }
        r = results[split]
        print(
            f"{split}: NER F1 {r['ner']['micro']['f1']} | "
            f"RE (gold ents) F1 {r['relations_gold_entities']['boundaries']['micro']['f1']} | "
            f"RE (pred ents) F1 {r['relations_predicted_entities']['boundaries']['micro']['f1']}"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Written {args.out}")


if __name__ == "__main__":
    main()
