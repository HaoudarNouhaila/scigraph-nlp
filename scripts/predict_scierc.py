"""Run the trained NER and RE models end to end on a SciERC split and save scored output.

The output (one JSON line per sentence, with entity and relation confidence scores) is
used to populate the graph with predicted knowledge next to the gold (verified) layer.
The script also recomputes Experiment B scores as a reproducibility check.

Usage::

    python scripts/predict_scierc.py --split test --out outputs/pipeline_test.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scigraph.data.scierc import load_split
from scigraph.evaluation.metrics import entity_scores, relation_scores
from scigraph.models import relation as re_mod
from scigraph.models import span_ner as ner_mod
from scigraph.training.train_ner import predict as ner_predict


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--data", default="data/processed")
    p.add_argument("--ner", default="outputs/ner/best")
    p.add_argument("--re", default="outputs/re/best")
    p.add_argument("--out", type=Path, default=Path("outputs/pipeline_test.jsonl"))
    p.add_argument(
        "--decoding",
        choices=["nested", "flat"],
        default="nested",
        help="NER decoding (see scripts/ner_decoding_ablation.py)",
    )
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sents = load_split(args.data, args.split)
    ner_model, ner_tok = ner_mod.load_model(args.ner, device)
    ents = ner_predict(
        ner_model,
        ner_tok,
        sents,
        ner_model.config,
        60,
        device,
        allow_overlap=args.decoding == "nested",
        return_scores=True,
    )
    spans = [[e for e, _ in sent] for sent in ents]

    re_model, re_tok = re_mod.load_model(args.re, device)
    inst = re_mod.build_instances(re_tok, sents, re_model.config.max_seq_length, spans)
    probs = []
    with torch.no_grad():
        for b in range(0, len(inst), 64):
            batch = re_mod.collate(inst[b : b + 64], re_tok.pad_token_id)
            batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            probs.append(torch.softmax(re_model(**batch).float(), -1).cpu())
    merged = re_mod.merge_predictions(inst, torch.cat(probs))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pred_rels = []
    with open(args.out, "w", encoding="utf-8") as f:
        for i, s in enumerate(sents):
            rels = merged.get(i, [])
            pred_rels.append([r for r, _ in rels])
            f.write(
                json.dumps(
                    {
                        "doc_key": s.doc_key,
                        "sent_id": s.sent_id,
                        "entities": [[e.start, e.end, e.label, round(sc, 4)] for e, sc in ents[i]],
                        "relations": [
                            [
                                r.head.start,
                                r.head.end,
                                r.tail.start,
                                r.tail.end,
                                r.label,
                                round(sc, 4),
                            ]
                            for r, sc in rels
                        ],
                    }
                )
                + "\n"
            )
    ner = entity_scores([s.entities for s in sents], spans)["micro"]
    rel = relation_scores([s.relations for s in sents], pred_rels)
    print(
        f"NER F1 {ner['f1']} | RE boundaries F1 {rel['boundaries']['micro']['f1']} | "
        f"RE strict F1 {rel['strict']['micro']['f1']}"
    )
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
