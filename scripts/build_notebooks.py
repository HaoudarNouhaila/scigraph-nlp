"""Generate the four project notebooks from source so they stay in sync with the code.

Usage::

    python scripts/build_notebooks.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

OUT = Path("notebooks")

SETUP = r"""# Environment setup: works on Kaggle, Google Colab and a local checkout.
import os, subprocess, sys
from pathlib import Path

IN_KAGGLE = Path("/kaggle/working").exists()
IN_COLAB = "google.colab" in sys.modules
REPO_URL = "https://github.com/HaoudarNouhaila/scigraph-nlp.git"

if IN_KAGGLE or IN_COLAB:
    base = Path("/kaggle/working") if IN_KAGGLE else Path("/content")
    ROOT = base / "scigraph-nlp"
    if not ROOT.exists():
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(ROOT)], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(ROOT)], check=True)
else:
    ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT)
print("project root:", ROOT.name)
"""

DATA = r"""# Download the official SciERC release (once) and build sentence-level splits.
if not Path("data/processed/train.jsonl").exists():
    if not Path("data/raw/scierc/train.json").exists():
        subprocess.run([sys.executable, "scripts/download_scierc.py", "--out", "data/raw/scierc"], check=True)
    subprocess.run([sys.executable, "scripts/prepare_data.py"], check=True)
print(sorted(p.name for p in Path("data/processed").glob("*.jsonl")))
"""

PERSIST = r"""# Persistent output directory so that an interrupted session can resume training.
# Colab: Google Drive. Kaggle: attach the output of a previous run of this notebook as an
# input dataset and it is copied back before training starts.
import shutil
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    OUTPUTS = Path("/content/drive/MyDrive/scigraph-nlp/outputs")
else:
    OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)
if IN_KAGGLE:
    for prev in Path("/kaggle/input").glob("*/outputs"):
        print("restoring checkpoints from", prev)
        shutil.copytree(prev, OUTPUTS, dirs_exist_ok=True)
import torch
print("device:", "cuda" if torch.cuda.is_available() else "cpu",
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
"""


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def nb(cells) -> nbf.NotebookNode:
    n = nbf.v4.new_notebook()
    n.cells = cells
    n.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    }
    return n


def exploration():
    return nb(
        [
            md("""
# 01 - SciERC data exploration

This notebook verifies the SciERC release used by the project and reports its statistics:
documents, sentences, entity and relation distributions, span lengths and nested entities.

Source: Luan et al., *Multi-Task Identification of Entities, Relations, and Coreference for
Scientific Knowledge Graph Construction*, EMNLP 2018, http://nlp.cs.washington.edu/sciIE/.
The official release does not state an explicit licence; see `docs/DATASET.md`.
"""),
            code(SETUP),
            code(DATA),
            code("""
import json
import pandas as pd
stats = json.loads(Path("reports/metrics/dataset_statistics.json").read_text())
splits = ["train", "dev", "test"]
pd.DataFrame({s: {k: stats[s][k] for k in ("documents", "sentences", "tokens", "entities",
              "relations", "coreference_clusters")} for s in splits})
"""),
            md("## Entity and relation types"),
            code("""
ents = pd.DataFrame({s: stats[s]["entity_types"] for s in splits})
rels = pd.DataFrame({s: stats[s]["relation_types"] for s in splits})
display(ents.assign(train_pct=(100 * ents.train / ents.train.sum()).round(1)))
display(rels.assign(train_pct=(100 * rels.train / rels.train.sum()).round(1)))
"""),
            md("## Span lengths and nested entities"),
            code("""
import matplotlib.pyplot as plt
hist = {int(k): v for k, v in stats["train"]["entity_length_histogram"].items()}
plt.figure(figsize=(6, 3))
plt.bar(list(hist), list(hist.values()))
plt.xlabel("entity length (tokens)"); plt.ylabel("count"); plt.title("Train entity lengths")
plt.tight_layout(); plt.show()
for s in splits:
    print(s, "within 8 tokens: %.2f%%" % stats[s]["entities_within_8_tokens_pct"],
          "| overlapping pairs:", stats[s]["overlapping_entity_pairs"],
          "| sentences with overlap:", stats[s]["sentences_with_overlapping_entities"])
"""),
            md("""
All overlapping entity pairs in SciERC are *nested* (one span inside another); there are no
crossing spans and no identical spans with two labels. The span-based NER model scores
every span independently and can therefore output nested entities.
"""),
            md("## Examples"),
            code("""
from scigraph.data.scierc import load_split
train = load_split("data/processed", "train")
nested = [s for s in train if any(a != b and a.start <= b.start and b.end <= a.end
          for a in s.entities for b in s.entities)]
for s in nested[:3]:
    print(" ".join(s.tokens))
    for e in s.entities:
        print(f"   {e.label:20s} {s.text_of(e)}")
    for r in s.relations:
        print(f"   {s.text_of(r.head)} --{r.label}--> {s.text_of(r.tail)}")
    print()
"""),
            md("## Relation direction"),
            code("""
from collections import Counter
c = Counter((r.head.label, r.label, r.tail.label) for s in train for r in s.relations)
pd.DataFrame([(h, r, t, n) for (h, r, t), n in c.most_common(15)],
             columns=["head type", "relation", "tail type", "count"])
"""),
            md("## Rule baseline"),
            code("""
_ = subprocess.run([sys.executable, "scripts/run_baseline.py"], check=True)
"""),
        ]
    )


def train_ner():
    return nb(
        [
            md("""
# 02 - Train the span-based SciBERT NER model

Enumerates all spans of up to 8 tokens per sentence and classifies each into one of the six
SciERC entity types or `O`. Nested entities are supported because spans are classified
independently. Training is resumable: checkpoints (model, optimiser, scheduler, RNG state)
are written to `OUTPUTS/ner/checkpoints/last.pt` and re-running the training cell continues
from the last checkpoint.

The configuration `configs/ner.yaml` targets a GPU (cross-sentence context, 15 epochs);
`configs/ner_cpu.yaml` is the reduced configuration used for the CPU run reported in the
README.
"""),
            code(SETUP),
            code(DATA),
            code(PERSIST),
            code("""
import yaml
CONFIG = "configs/ner.yaml" if torch.cuda.is_available() else "configs/ner_cpu.yaml"
print(yaml.safe_load(Path(CONFIG).read_text()))
"""),
            code("""
# Training (resumes automatically if a checkpoint exists in OUTPUTS/ner).
cmd = [sys.executable, "-m", "scigraph.training.train_ner", "--config", CONFIG,
       "--set", f"output_dir={OUTPUTS / 'ner'}"]
_ = subprocess.run(cmd, check=True)
"""),
            code("""
import json
m = json.loads((OUTPUTS / "ner" / "metrics.json").read_text())
print("dev  F1:", m["dev"]["micro"]["f1"])
print("test P/R/F1:", m["test"]["micro"]["precision"], m["test"]["micro"]["recall"], m["test"]["micro"]["f1"])
import pandas as pd
pd.DataFrame(m["test"]["per_type"]).T
"""),
        ]
    )


def train_re():
    return nb(
        [
            md("""
# 03 - Train the SciBERT relation classifier (typed entity markers)

Each ordered pair of entities in a sentence is encoded with typed markers such as
`<S:Method> ... </S:Method>` and `<O:Task> ... </O:Task>`; the two opening-marker states are
concatenated and classified into the seven relation types or `NO-RELATION`.

* Experiment A: candidate pairs from gold entities (relation classification only).
* Experiment B: candidate pairs from the NER predictions of notebook 02 (end to end).

Run notebook 02 first, so that `OUTPUTS/ner/predictions_{dev,test}.jsonl` exist.
"""),
            code(SETUP),
            code(DATA),
            code(PERSIST),
            code("""
CONFIG = "configs/re.yaml" if torch.cuda.is_available() else "configs/re_cpu.yaml"
cmd = [sys.executable, "-m", "scigraph.training.train_re", "--config", CONFIG,
       "--set", f"output_dir={OUTPUTS / 're'}", f"ner_predictions_dir={OUTPUTS / 'ner'}"]
_ = subprocess.run(cmd, check=True)
"""),
            code("""
import json, pandas as pd
m = json.loads((OUTPUTS / "re" / "metrics.json").read_text())
rows = []
a = m["experiment_a_gold_entities"]["test"]
rows.append(("A: gold entities", "boundaries", *[a["boundaries"]["micro"][k] for k in ("precision", "recall", "f1")]))
b = m["experiment_b_predicted_entities"]
if isinstance(b, dict):
    for crit in ("boundaries", "strict"):
        rows.append(("B: predicted entities", crit, *[b["test"][crit]["micro"][k] for k in ("precision", "recall", "f1")]))
pd.DataFrame(rows, columns=["setting", "criterion", "P", "R", "F1"])
"""),
        ]
    )


def evaluation():
    return nb(
        [
            md("""
# 04 - Evaluation and error analysis

Collects the measured metrics of the baseline and the neural models, draws figures, and
inspects errors on the test split. Every number shown is read from files produced by the
training and evaluation runs.
"""),
            code(SETUP),
            code("""
subprocess.run([sys.executable, "scripts/make_report.py", "--ner", "outputs/ner", "--re", "outputs/re"], check=True)
import json, pandas as pd
summary = json.loads(Path("reports/metrics/summary.json").read_text())
pd.DataFrame(summary["rows"])
"""),
            code("""
from IPython.display import Image, display
for f in sorted(Path("reports/figures").glob("*.png")):
    print(f.name); display(Image(filename=str(f)))
"""),
            md("## NER errors"),
            code("""
from collections import Counter
from scigraph.data.scierc import load_split, read_jsonl
test = load_split("data/processed", "test")
preds = {(r["doc_key"], r["sent_id"]): r["entities"] for r in read_jsonl("outputs/ner/predictions_test.jsonl")}
kinds = Counter()
examples = []
for s in test:
    gold = {(e.start, e.end): e.label for e in s.entities}
    pred = {(a, b): l for a, b, l in preds[(s.doc_key, s.sent_id)]}
    for span, lab in pred.items():
        if span in gold and gold[span] != lab:
            kinds[f"type: {gold[span]} -> {lab}"] += 1
        elif span not in gold:
            overl = any(not (span[1] < g[0] or g[1] < span[0]) for g in gold)
            kinds["boundary error" if overl else "spurious entity"] += 1
            if overl and len(examples) < 8:
                examples.append((" ".join(s.tokens[span[0]:span[1]+1]), lab, [" ".join(s.tokens[g[0]:g[1]+1]) for g in gold if not (span[1] < g[0] or g[1] < span[0])]))
    for span in gold:
        if span not in pred and not any(not (span[1] < p[0] or p[1] < span[0]) for p in pred):
            kinds["missed entity"] += 1
pd.Series(kinds).sort_values(ascending=False).head(15)
"""),
            code("""
pd.DataFrame(examples, columns=["predicted span", "predicted type", "overlapping gold spans"])
"""),
            md("## Relation errors (Experiment B)"),
            code("""
rel_preds = {(r["doc_key"], r["sent_id"]): r for r in read_jsonl("outputs/re/predictions_pred_entities_test.jsonl")}
cause = Counter()
for s in test:
    row = rel_preds[(s.doc_key, s.sent_id)]
    pred_spans = {(a, b) for a, b, _ in row["entities"]}
    pred_rels = {(a, b, c, d, l) for a, b, c, d, l in row["relations"]}
    for r in s.relations:
        key = (r.head.start, r.head.end, r.tail.start, r.tail.end, r.label)
        rev = (r.tail.start, r.tail.end, r.head.start, r.head.end, r.label)
        if key in pred_rels or rev in pred_rels and r.label in ("COMPARE", "CONJUNCTION"):
            cause["correct"] += 1
        elif (r.head.start, r.head.end) not in pred_spans or (r.tail.start, r.tail.end) not in pred_spans:
            cause["missed: argument not found by NER"] += 1
        elif any(p[:4] == key[:4] for p in pred_rels):
            cause["wrong relation type"] += 1
        elif any(p[:4] == rev[:4] for p in pred_rels):
            cause["wrong direction"] += 1
        else:
            cause["missed: pair predicted as NO-RELATION"] += 1
pd.Series(cause)
"""),
        ]
    )


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, builder in (
        ("01_data_exploration.ipynb", exploration),
        ("02_train_scibert_ner.ipynb", train_ner),
        ("03_train_relation_extraction.ipynb", train_re),
        ("04_evaluation.ipynb", evaluation),
    ):
        nbf.write(builder(), OUT / name)
        print("wrote", OUT / name)


if __name__ == "__main__":
    main()
