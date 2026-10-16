#!/usr/bin/env bash
# Train NER then RE on CPU with resumable checkpoints. Re-running resumes both stages.
set -euo pipefail
PY=${PYTHON:-python}
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
$PY -m scigraph.training.train_ner --config configs/ner_cpu.yaml
$PY -m scigraph.training.train_re --config configs/re_cpu.yaml
