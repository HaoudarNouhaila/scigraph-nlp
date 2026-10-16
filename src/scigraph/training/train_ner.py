"""Train and evaluate the span-based SciBERT NER model on SciERC.

The run is resumable: re-launching with the same ``--output-dir`` continues from the
last saved checkpoint (end of the last completed epoch, or the last ``--save-every``
step boundary).

Usage::

    python -m scigraph.training.train_ner --config configs/ner.yaml --set epochs=3
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import torch
from torch import nn

from scigraph.data.scierc import load_split
from scigraph.evaluation.metrics import entity_scores
from scigraph.models.span_ner import (
    NER_LABELS,
    SpanNERConfig,
    SpanNERModel,
    build_features,
    collate,
    decode,
    save_model,
    sentence_contexts,
)
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
    "output_dir": "outputs/ner",
    "encoder_name": "allenai/scibert_scivocab_uncased",
    "max_span_length": 8,
    "context_window": 100,
    "context_words": 60,
    "max_seq_length": 256,
    "epochs": 10,
    "batch_size": 16,
    "encoder_lr": 1e-5,
    "head_lr": 5e-4,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "gradient_accumulation": 1,
    "seed": 42,
    "fp16": True,
    "save_every": 0,
    "allow_overlap": True,
    "max_train_sentences": None,
    "max_eval_sentences": None,
}


def featurize(tokenizer, sentences, cfg: SpanNERConfig, context_words: int):
    ctx = sentence_contexts(sentences, context_words) if cfg.context_window > 0 else None
    feats = []
    for s in sentences:
        left, right = next(ctx) if ctx is not None else (None, None)
        feats.append(build_features(tokenizer, s, cfg, left, right))
    return feats


@torch.no_grad()
def predict(
    model,
    tokenizer,
    sentences,
    cfg,
    context_words,
    device,
    batch_size=32,
    allow_overlap=True,
    return_scores=False,
):
    model.eval()
    feats = featurize(tokenizer, sentences, cfg, context_words)
    preds = []
    for b in range(0, len(feats), batch_size):
        chunk = feats[b : b + batch_size]
        batch = collate(chunk, tokenizer.pad_token_id)
        batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        probs = torch.softmax(model(**batch).float(), dim=-1).cpu()
        for k, f in enumerate(chunk):
            decoded = decode(f.spans, probs[k, : len(f.spans)], allow_overlap=allow_overlap)
            preds.append(decoded if return_scores else [s for s, _ in decoded])
    return preds


def evaluate(model, tokenizer, sentences, cfg, context_words, device, allow_overlap=True):
    preds = predict(
        model, tokenizer, sentences, cfg, context_words, device, allow_overlap=allow_overlap
    )
    return entity_scores([s.entities for s in sentences], preds), preds


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

    cfg = SpanNERConfig(
        encoder_name=params["encoder_name"],
        max_span_length=params["max_span_length"],
        max_seq_length=params["max_seq_length"],
        context_window=params["context_window"],
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.encoder_name)
    model = SpanNERModel(cfg).to(device)
    train_feats = featurize(tokenizer, train_s, cfg, params["context_words"])
    gold_too_long = sum(1 for s in train_s for e in s.entities if e.length > cfg.max_span_length)

    steps_per_epoch = math.ceil(len(train_feats) / params["batch_size"])
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
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

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
            train_sentences=len(train_s),
            dev_sentences=len(dev_s),
            gold_entities_longer_than_max_span=gold_too_long,
            total_steps=total_steps,
        )

    for epoch in range(start_epoch, params["epochs"]):
        model.train()
        order = list(range(len(train_feats)))
        random.Random(params["seed"] + epoch).shuffle(order)
        skip = global_step - epoch * steps_per_epoch  # resume inside an epoch
        t0, running = time.time(), 0.0
        for b_idx in range(steps_per_epoch):
            if b_idx < skip:
                continue
            idx = order[b_idx * params["batch_size"] : (b_idx + 1) * params["batch_size"]]
            batch = collate([train_feats[i] for i in idx], tokenizer.pad_token_id)
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(**batch)
                loss = loss_fn(logits.view(-1, len(NER_LABELS)).float(), labels.view(-1))
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
            if global_step % 50 == 0:
                logger.log(
                    event="step",
                    epoch=epoch,
                    step=global_step,
                    loss=round(running / 50, 5),
                    elapsed=round(time.time() - t0, 1),
                )
                running = 0.0
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
            model, tokenizer, dev_s, cfg, params["context_words"], device, params["allow_overlap"]
        )
        f1 = dev_scores["micro"]["f1"]
        logger.log(
            event="dev",
            epoch=epoch + 1,
            step=global_step,
            **dev_scores["micro"],
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
    from scigraph.models.span_ner import load_model

    out = Path(params["output_dir"])
    model, tokenizer = load_model(out / "best", device=str(device))
    cfg = model.config
    results = {}
    for name, sents in (("dev", dev_s), ("test", test_s)):
        scores, preds = evaluate(
            model, tokenizer, sents, cfg, params["context_words"], device, params["allow_overlap"]
        )
        nested_pred = sum(
            1
            for p in preds
            for i, a in enumerate(p)
            for b in p[i + 1 :]
            if not (a.end < b.start or b.end < a.start)
        )
        scores["predicted_overlapping_pairs"] = nested_pred
        results[name] = scores
        with open(out / f"predictions_{name}.jsonl", "w", encoding="utf-8") as f:
            for s, p in zip(sents, preds, strict=True):
                f.write(
                    json.dumps(
                        {
                            "doc_key": s.doc_key,
                            "sent_id": s.sent_id,
                            "entities": [[e.start, e.end, e.label] for e in p],
                        }
                    )
                    + "\n"
                )
    (out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.log(
        event="final", dev_f1=results["dev"]["micro"]["f1"], test_f1=results["test"]["micro"]["f1"]
    )
    return results


if __name__ == "__main__":
    train(parse_params(DEFAULTS, "Train the span-based SciBERT NER model."))
