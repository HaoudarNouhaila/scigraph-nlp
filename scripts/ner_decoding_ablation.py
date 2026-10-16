"""Compare nested and non-overlapping decoding of the trained span NER model.

Both decoders use the same span probabilities. ``nested`` keeps every span whose best
label is not O; ``flat`` greedily keeps the highest-scoring spans that do not overlap.
The decoder to use downstream is chosen on the dev split; test scores are reported for
both for transparency.

Usage::

    python scripts/ner_decoding_ablation.py --ner outputs/ner/best
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scigraph.data.scierc import load_split, overlap_kind
from scigraph.evaluation.metrics import entity_scores
from scigraph.models import span_ner as ner_mod
from scigraph.models.span_ner import collate, decode
from scigraph.training.train_ner import featurize


def nested_stats(sents, preds) -> dict:
    gold_pairs = recovered = pred_pairs = pred_pairs_correct = 0
    for s, p in zip(sents, preds, strict=True):
        pset = {(e.start, e.end, e.label) for e in p}
        gset = {(e.start, e.end, e.label) for e in s.entities}
        for i, a in enumerate(s.entities):
            for b in s.entities[i + 1 :]:
                if overlap_kind(a, b) == "nested":
                    gold_pairs += 1
                    recovered += (a.start, a.end, a.label) in pset and (
                        b.start,
                        b.end,
                        b.label,
                    ) in pset
        for i, a in enumerate(p):
            for b in p[i + 1 :]:
                if overlap_kind(a, b):
                    pred_pairs += 1
                    pred_pairs_correct += (a.start, a.end, a.label) in gset and (
                        b.start,
                        b.end,
                        b.label,
                    ) in gset
    return {
        "gold_nested_pairs": gold_pairs,
        "gold_nested_pairs_fully_recovered": recovered,
        "predicted_overlapping_pairs": pred_pairs,
        "predicted_overlapping_pairs_both_correct": pred_pairs_correct,
    }


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ner", default="outputs/ner/best")
    p.add_argument("--out", type=Path, default=Path("reports/metrics/ner_decoding.json"))
    args = p.parse_args()
    model, tok = ner_mod.load_model(args.ner)
    result = {}
    for split in ("dev", "test"):
        sents = load_split("data/processed", split)
        feats = featurize(tok, sents, model.config, 60)
        probs = []
        for b in range(0, len(feats), 32):
            batch = collate(feats[b : b + 32], tok.pad_token_id)
            batch.pop("labels")
            out = torch.softmax(model(**batch).float(), -1)
            probs += [out[k, : len(f.spans)] for k, f in enumerate(feats[b : b + 32])]
        for name, overlap in (("nested", True), ("flat", False)):
            preds = [
                [s for s, _ in decode(f.spans, pr, allow_overlap=overlap)]
                for f, pr in zip(feats, probs, strict=True)
            ]
            scores = entity_scores([s.entities for s in sents], preds)
            result.setdefault(name, {})[split] = {
                "micro": scores["micro"],
                "macro_f1": scores["macro_f1"],
                **nested_stats(sents, preds),
            }
    best = max(("nested", "flat"), key=lambda k: result[k]["dev"]["micro"]["f1"])
    result["selected_on_dev"] = best
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for name in ("nested", "flat"):
        print(
            name,
            {s: result[name][s]["micro"]["f1"] for s in ("dev", "test")},
            {k: v for k, v in result[name]["test"].items() if k.startswith(("gold", "pred"))},
        )
    print("selected on dev:", best)


if __name__ == "__main__":
    main()
