"""Train and evaluate the SciBERT relation classifier with typed entity markers.

Two evaluation settings are produced after training:

* Experiment A (gold entities): candidate pairs are built from annotated entities, so
  the score isolates relation classification.
* Experiment B (predicted entities): candidate pairs are built from the NER model's
  predictions (``ner_predictions_dir``), so errors propagate from NER to RE. This is
  the end-to-end score that matters for knowledge graph construction.

The run is resumable from ``<output_dir>/checkpoints/last.pt``.

Usage::

    python -m scigraph.training.train_re --config configs/re.yaml \
        --set ner_predictions_dir=outputs/ner
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn

from scigraph.data.scierc import load_split, read_jsonl
from scigraph.evaluation.metrics import entity_scores, relation_confusion, relation_scores
from scigraph.models.relation import (
    RE_LABELS,
    RelationConfig,
    RelationModel,
    build_instances,
    collate,
    merge_predictions,
    prepare_tokenizer,
    save_model,
)
from scigraph.schema import Span
from scigraph.training.common import (
    CheckpointManager,
    JsonlLogger,
    backward_and_step,
    get_device,
    grouped_parameters,
    parse_params,
    set_seed,
)

DEFAULTS = {
    "data_dir": "data/processed",
    "output_dir": "outputs/re",
    "ner_predictions_dir": None,
    "encoder_name": "allenai/scibert_scivocab_uncased",
    "max_seq_length": 256,
    "epochs": 10,
    "batch_size": 32,
    "eval_batch_size": 64,
    "encoder_lr": 2e-5,
    "head_lr": 1e-4,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "gradient_accumulation": 1,
    "seed": 42,
    "fp16": True,
    "save_every": 0,
    # Negatives per positive, resampled every epoch (None: use all candidate pairs).
    "negative_ratio": None,
    "max_train_sentences": None,
    "max_eval_sentences": None,
}


@torch.no_grad()
def predict(model, tokenizer, sentences, device, max_length, batch_size=64, entities_override=None):
    model.eval()
    instances = build_instances(tokenizer, sentences, max_length, entities_override)
    all_probs = []
    for b in range(0, len(instances), batch_size):
        batch = collate(instances[b : b + batch_size], tokenizer.pad_token_id)
        batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        all_probs.append(torch.softmax(model(**batch).float(), dim=-1).cpu())
    probs = torch.cat(all_probs) if all_probs else torch.zeros(0, len(RE_LABELS))
    merged = merge_predictions(instances, probs)
    return [[r for r, _ in merged.get(i, [])] for i in range(len(sentences))]


def evaluate(model, tokenizer, sentences, device, max_length, batch_size, entities_override=None):
    preds = predict(model, tokenizer, sentences, device, max_length, batch_size, entities_override)
    gold = [s.relations for s in sentences]
    scores = relation_scores(gold, preds)
    scores["confusion"] = relation_confusion(gold, preds)
    return scores, preds


def load_ner_predictions(path: Path, sentences) -> list[list[Span]]:
    rows = {(r["doc_key"], r["sent_id"]): r for r in read_jsonl(path)}
    out = []
    for s in sentences:
        row = rows.get((s.doc_key, s.sent_id))
        if row is None:
            raise SystemExit(f"Missing NER prediction for {s.doc_key}/{s.sent_id} in {path}")
        out.append([Span(a, b, lab) for a, b, lab in row["entities"]])
    return out


def train(params: dict) -> dict:
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    set_seed(params["seed"])
    device = get_device()
    out = Path(params["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    logger = JsonlLogger(out / "train_log.jsonl")

    train_s = load_split(params["data_dir"], "train")
    dev_s = load_split(params["data_dir"], "dev")
    test_s = load_split(params["data_dir"], "test")
    if params["max_train_sentences"]:
        train_s = train_s[: params["max_train_sentences"]]
    if params["max_eval_sentences"]:
        dev_s = dev_s[: params["max_eval_sentences"]]
        test_s = test_s[: params["max_eval_sentences"]]

    cfg = RelationConfig(
        encoder_name=params["encoder_name"], max_seq_length=params["max_seq_length"]
    )
    tokenizer = prepare_tokenizer(AutoTokenizer.from_pretrained(cfg.encoder_name))
    model = RelationModel(cfg, vocab_size=len(tokenizer)).to(device)

    all_inst = build_instances(tokenizer, train_s, cfg.max_seq_length, training=True)
    label_counts = Counter(RE_LABELS[i.label] for i in all_inst)
    positives = [i for i in all_inst if i.label != 0]
    negatives = [i for i in all_inst if i.label == 0]

    def epoch_instances(epoch: int):
        """All instances, or all positives plus a fresh negative sample for this epoch."""
        if not params["negative_ratio"]:
            return all_inst
        rng = random.Random(params["seed"] * 1000 + epoch)
        k = min(len(negatives), int(params["negative_ratio"] * len(positives)))
        return positives + rng.sample(negatives, k)

    epoch_size = len(epoch_instances(0))
    steps_per_epoch = math.ceil(epoch_size / params["batch_size"])
    accum = max(1, int(params["gradient_accumulation"]))
    # Scheduler steps are optimiser updates; checkpoints and logs count micro-batches.
    total_steps = math.ceil(steps_per_epoch / accum) * params["epochs"]
    optimizer = torch.optim.AdamW(
        grouped_parameters(model, params["encoder_lr"], params["head_lr"], params["weight_decay"])
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(params["warmup_ratio"] * total_steps), total_steps
    )
    use_amp = params["fp16"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    loss_fn = nn.CrossEntropyLoss()

    ckpt = CheckpointManager(out / "checkpoints")
    start_epoch, global_step, best = 0, 0, -1.0
    if ckpt.exists():
        state = ckpt.load(
            model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, device=device
        )
        start_epoch, global_step, best = state["epoch"], state["step"], state["best"]
        logger.log(event="resume", epoch=start_epoch, step=global_step, best_dev_f1=best)
    else:
        logger.log(
            event="start",
            device=str(device),
            train_instances=len(all_inst),
            instances_per_epoch=epoch_size,
            label_counts=dict(label_counts),
            total_steps=total_steps,
        )

    for epoch in range(start_epoch, params["epochs"]):
        model.train()
        train_inst = epoch_instances(epoch)
        order = list(range(len(train_inst)))
        random.Random(params["seed"] + epoch).shuffle(order)
        skip = global_step - epoch * steps_per_epoch
        t0, running, n_running = time.time(), 0.0, 0
        for b_idx in range(steps_per_epoch):
            if b_idx < skip:
                continue
            idx = order[b_idx * params["batch_size"] : (b_idx + 1) * params["batch_size"]]
            batch = collate([train_inst[i] for i in idx], tokenizer.pad_token_id)
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(**batch)
            loss = loss_fn(logits.float(), labels)
            boundary = (b_idx + 1) % accum == 0 or b_idx + 1 == steps_per_epoch
            backward_and_step(
                loss,
                model,
                optimizer,
                scheduler,
                scaler,
                grad_clip=params["grad_clip"],
                accumulation=accum,
                boundary=boundary,
            )
            global_step += 1
            running += loss.item()
            n_running += 1
            if global_step % 100 == 0:
                logger.log(
                    event="step",
                    epoch=epoch,
                    step=global_step,
                    loss=round(running / n_running, 5),
                    elapsed=round(time.time() - t0, 1),
                )
                running, n_running = 0.0, 0
            if params["save_every"] and global_step % params["save_every"] == 0 and boundary:
                ckpt.save(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    step=global_step,
                    best=best,
                )

        dev_scores, _ = evaluate(
            model, tokenizer, dev_s, device, cfg.max_seq_length, params["eval_batch_size"]
        )
        f1 = dev_scores["boundaries"]["micro"]["f1"]
        logger.log(
            event="dev",
            epoch=epoch + 1,
            step=global_step,
            **dev_scores["boundaries"]["micro"],
            epoch_seconds=round(time.time() - t0, 1),
        )
        if f1 > best:
            best = f1
            save_model(model, tokenizer, out / "best")
            logger.log(event="new_best", epoch=epoch + 1, dev_f1=f1)
        ckpt.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch + 1,
            step=global_step,
            best=best,
        )

    return final_evaluation(params, dev_s, test_s, device, logger)


def final_evaluation(params, dev_s, test_s, device, logger) -> dict:
    from scigraph.models.relation import load_model

    out = Path(params["output_dir"])
    model, tokenizer = load_model(out / "best", device=str(device))
    max_len = model.config.max_seq_length
    bs = params["eval_batch_size"]
    results: dict = {"experiment_a_gold_entities": {}, "experiment_b_predicted_entities": {}}
    for name, sents in (("dev", dev_s), ("test", test_s)):
        scores, preds = evaluate(model, tokenizer, sents, device, max_len, bs)
        results["experiment_a_gold_entities"][name] = scores
        _write_predictions(out / f"predictions_gold_entities_{name}.jsonl", sents, preds)

    ner_dir = params["ner_predictions_dir"]
    if ner_dir:
        for name, sents in (("dev", dev_s), ("test", test_s)):
            ents = load_ner_predictions(Path(ner_dir) / f"predictions_{name}.jsonl", sents)
            scores, preds = evaluate(model, tokenizer, sents, device, max_len, bs, ents)
            scores["ner"] = entity_scores([s.entities for s in sents], ents)["micro"]
            results["experiment_b_predicted_entities"][name] = scores
            _write_predictions(out / f"predictions_pred_entities_{name}.jsonl", sents, preds, ents)
    else:
        results["experiment_b_predicted_entities"] = "skipped: ner_predictions_dir not set"

    (out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    summary = {
        "event": "final",
        "A_test_f1": results["experiment_a_gold_entities"]["test"]["boundaries"]["micro"]["f1"],
    }
    if isinstance(results["experiment_b_predicted_entities"], dict):
        b = results["experiment_b_predicted_entities"]["test"]
        summary["B_test_f1_boundaries"] = b["boundaries"]["micro"]["f1"]
        summary["B_test_f1_strict"] = b["strict"]["micro"]["f1"]
    logger.log(**summary)
    return results


def _write_predictions(path: Path, sents, preds, entities=None) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i, (s, rels) in enumerate(zip(sents, preds, strict=True)):
            row = {
                "doc_key": s.doc_key,
                "sent_id": s.sent_id,
                "relations": [
                    [r.head.start, r.head.end, r.tail.start, r.tail.end, r.label] for r in rels
                ],
            }
            if entities is not None:
                row["entities"] = [[e.start, e.end, e.label] for e in entities[i]]
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    train(parse_params(DEFAULTS, "Train the SciBERT relation classifier."))
