"""Collect measured metrics into reports/ and draw figures.

Reads only files written by training and evaluation runs; nothing is estimated or
filled in. Missing inputs are reported and skipped.

Usage::

    python scripts/make_report.py --ner outputs/ner --re outputs/re
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METRICS = Path("reports/metrics")
FIGURES = Path("reports/figures")


def read_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def plot_curves(log: list[dict], title: str, out: Path) -> None:
    steps = [r for r in log if r["event"] == "step"]
    dev = [r for r in log if r["event"] == "dev"]
    # Runs logged before the averaging fix divided the first post-epoch window by the full
    # logging interval; those points (window crossing an epoch end) are left out.
    if len(steps) > 1:
        interval = min(b["step"] - a["step"] for a, b in zip(steps, steps[1:], strict=False))
        ends = [d["step"] for d in dev]
        steps = [r for r in steps if not any(r["step"] - interval < e < r["step"] for e in ends)]
    if not steps and not dev:
        return
    fig, ax1 = plt.subplots(figsize=(7, 3.5))
    if steps:
        ax1.plot(
            [r["step"] for r in steps],
            [r["loss"] for r in steps],
            color="#4c72b0",
            label="training loss",
        )
        ax1.set_xlabel("optimisation step")
        ax1.set_ylabel("loss")
    if dev:
        ax2 = ax1.twinx()
        ax2.plot(
            [r["step"] for r in dev], [r["f1"] for r in dev], "o-", color="#c44e52", label="dev F1"
        )
        ax2.set_ylabel("dev micro F1 (%)")
    ax1.set_title(title)
    fig.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_per_type(per_type: dict, title: str, out: Path) -> None:
    labels = list(per_type)
    f1 = [per_type[k]["f1"] for k in labels]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh(labels, f1, color="#55a868")
    for i, v in enumerate(f1):
        ax.text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("F1 (%)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_confusion(conf: dict, out: Path) -> None:
    labels = sorted(set(conf) | {p for row in conf.values() for p in row})
    mat = [[conf.get(g, {}).get(p, 0) for p in labels] for g in labels]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for i, row in enumerate(mat):
        for j, v in enumerate(row):
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold")
    ax.set_title("Relation confusion, test, gold entities")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def recompute_ner(run_dir: Path, logged: dict) -> dict:
    """Recompute NER scores from the saved predictions with the current metric code.

    Adds macro-F1 and checks that the micro scores equal those logged at training time.
    """
    from scigraph.data.scierc import load_split, read_jsonl
    from scigraph.evaluation.metrics import entity_scores
    from scigraph.schema import Span

    out = {}
    for split in ("dev", "test"):
        path = run_dir / f"predictions_{split}.jsonl"
        if not path.exists():
            out[split] = logged[split]
            continue
        sents = load_split("data/processed", split)
        rows = {(r["doc_key"], r["sent_id"]): r for r in read_jsonl(path)}
        pred = [
            [Span(a, b, lab) for a, b, lab, *_ in rows[(s.doc_key, s.sent_id)]["entities"]]
            for s in sents
        ]
        scores = entity_scores([s.entities for s in sents], pred)
        if scores["micro"]["f1"] != logged[split]["micro"]["f1"]:
            raise SystemExit(f"NER {split} F1 mismatch: {scores['micro']} vs logged")
        scores["predicted_overlapping_pairs"] = logged[split].get("predicted_overlapping_pairs")
        out[split] = scores
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ner", type=Path, default=Path("outputs/ner"))
    p.add_argument("--re", type=Path, default=Path("outputs/re"))
    args = p.parse_args()
    METRICS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    rows, notes = [], []
    base = (
        json.loads((METRICS / "baseline.json").read_text())
        if (METRICS / "baseline.json").exists()
        else None
    )

    def row(system: str, task: str, setting: str, block: dict) -> dict:
        """One summary row from a score block with 'micro' and 'macro_f1' entries."""
        return {
            "system": system,
            "task": task,
            "setting": setting,
            **{k: block["micro"][k] for k in ("precision", "recall", "f1")},
            "macro_f1": block.get("macro_f1"),
        }

    if base:
        t = base["test"]
        rows += [
            row("Rule baseline (spaCy)", "NER", "-", t["ner"]),
            row(
                "Rule baseline (spaCy)",
                "RE",
                "A: gold entities",
                t["relations_gold_entities"]["boundaries"],
            ),
            row(
                "Rule baseline (spaCy)",
                "RE",
                "B: predicted entities (boundaries)",
                t["relations_predicted_entities"]["boundaries"],
            ),
            row(
                "Rule baseline (spaCy)",
                "RE",
                "B: predicted entities (strict)",
                t["relations_predicted_entities"]["strict"],
            ),
        ]

    ner_m = args.ner / "metrics.json"
    if ner_m.exists():
        ner = recompute_ner(args.ner, json.loads(ner_m.read_text()))
        (METRICS / "ner_metrics.json").write_text(json.dumps(ner, indent=2), encoding="utf-8")
        rows.append(row("SciBERT span NER", "NER", "-", ner["test"]))
        plot_per_type(
            ner["test"]["per_type"], "NER F1 per entity type (test)", FIGURES / "ner_per_type.png"
        )
        shutil.copy(args.ner / "params.json", METRICS / "ner_params.json")
    else:
        notes.append(f"NER metrics not found at {ner_m}")
    plot_curves(
        read_log(args.ner / "train_log.jsonl"), "Span NER training", FIGURES / "ner_training.png"
    )

    re_m = args.re / "metrics.json"
    if re_m.exists():
        shutil.copy(re_m, METRICS / "re_metrics.json")
        rel = json.loads(re_m.read_text())
        a = rel["experiment_a_gold_entities"]["test"]
        rows.append(row("SciBERT marker RE", "RE", "A: gold entities", a["boundaries"]))
        b = rel["experiment_b_predicted_entities"]
        if isinstance(b, dict):
            rows.append(
                row(
                    "SciBERT NER + RE",
                    "RE",
                    "B: predicted entities (boundaries)",
                    b["test"]["boundaries"],
                )
            )
            rows.append(
                row("SciBERT NER + RE", "RE", "B: predicted entities (strict)", b["test"]["strict"])
            )
        plot_per_type(
            a["boundaries"]["per_type"],
            "Relation F1 per type (test, gold entities)",
            FIGURES / "re_per_type.png",
        )
        plot_confusion(a["confusion"], FIGURES / "re_confusion.png")
        shutil.copy(args.re / "params.json", METRICS / "re_params.json")
    else:
        notes.append(f"RE metrics not found at {re_m}")
    plot_curves(
        read_log(args.re / "train_log.jsonl"),
        "Relation classifier training",
        FIGURES / "re_training.png",
    )
    notes.append("All scores are micro-averaged on the official SciERC test split (551 sentences).")
    summary = {"rows": rows, "notes": notes}
    (METRICS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for r in rows:
        print(
            f"{r['system']:28s} {r['task']:4s} {r['setting']:38s} "
            f"P {r['precision']:6.2f} R {r['recall']:6.2f} F1 {r['f1']:6.2f} "
            f"macro-F1 {r['macro_f1'] if r['macro_f1'] is not None else '-'}"
        )
    for n in notes:
        print("note:", n)


if __name__ == "__main__":
    main()
