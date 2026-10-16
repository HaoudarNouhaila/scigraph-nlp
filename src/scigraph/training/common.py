"""Training utilities: seeding, device selection, resumable checkpoints, logging."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CheckpointManager:
    """Saves and restores the full training state so that interrupted runs can resume.

    The checkpoint (``last.pt``) holds model, optimizer, scheduler and AMP scaler state,
    the epoch and global step, the best dev score so far, and RNG states. It is written
    atomically (temporary file then rename) so that a kill during saving cannot corrupt
    the previous checkpoint.
    """

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "last.pt"

    def exists(self) -> bool:
        return self.path.exists()

    def save(self, *, model, optimizer, scheduler, scaler, epoch, step, best, extra=None):
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "step": step,
            "best": best,
            "extra": extra or {},
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        tmp = self.path.with_suffix(".tmp")
        torch.save(state, tmp)
        os.replace(tmp, self.path)

    def load(self, *, model, optimizer, scheduler, scaler, device) -> dict:
        state = torch.load(self.path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state["scheduler"] is not None:
            scheduler.load_state_dict(state["scheduler"])
        if scaler is not None and state["scaler"] is not None:
            scaler.load_state_dict(state["scaler"])
        rng = state["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"].cpu())
        if rng["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
        return state


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **record) -> None:
        record.setdefault("time", time.strftime("%Y-%m-%dT%H:%M:%S"))
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record))


def grouped_parameters(model, encoder_lr: float, head_lr: float, weight_decay: float = 0.01):
    """Separate learning rates for the pretrained encoder and the task-specific head."""
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = []
    for is_encoder, lr in ((True, encoder_lr), (False, head_lr)):
        for decay in (True, False):
            params = [
                p
                for n, p in model.named_parameters()
                if n.startswith("encoder.") == is_encoder
                and (not any(nd in n for nd in no_decay)) == decay
                and p.requires_grad
            ]
            if params:
                groups.append(
                    {"params": params, "lr": lr, "weight_decay": weight_decay if decay else 0.0}
                )
    return groups


def parse_params(defaults: dict, description: str, argv=None) -> dict:
    """Merge defaults, an optional YAML config and ``--set key=value`` overrides."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", type=Path, default=None, help="YAML file with parameters")
    p.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="override parameters, values are parsed as YAML scalars",
    )
    args = p.parse_args(argv)
    params = dict(defaults)
    if args.config:
        params.update(yaml.safe_load(args.config.read_text(encoding="utf-8")) or {})
    for item in args.set:
        key, _, value = item.partition("=")
        if key not in defaults:
            raise SystemExit(f"Unknown parameter: {key}")
        params[key] = yaml.safe_load(value)
    unknown = set(params) - set(defaults)
    if unknown:
        raise SystemExit(f"Unknown parameters in config: {sorted(unknown)}")
    return params
