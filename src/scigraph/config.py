"""Pipeline and graph settings read from ``configs/graph.yaml``."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

DEFAULTS = {
    "extraction": {
        "ner_threshold": 0.5,
        "re_threshold": 0.5,
        "ner_decoding": "nested",
        "max_sentence_tokens": 120,
    },
    "entity_resolution": {"near_duplicate_jaccard": 0.75},
    "ingestion": {"batch_size": 500},
}


@lru_cache(maxsize=1)
def graph_settings() -> dict:
    """Defaults overridden by the YAML file named in SCIGRAPH_GRAPH_CONFIG, if present."""
    path = Path(os.getenv("SCIGRAPH_GRAPH_CONFIG", "configs/graph.yaml"))
    settings = {k: dict(v) for k, v in DEFAULTS.items()}
    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for section, values in loaded.items():
            if section not in settings:
                raise ValueError(f"Unknown section {section!r} in {path}")
            settings[section].update(values or {})
    return settings
