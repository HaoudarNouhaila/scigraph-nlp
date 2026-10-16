"""Precision, recall and F1 (micro and macro) for entities and relations.

Entity matching is exact: same sentence, same boundaries, same type.

Two relation criteria are reported, following Taillé et al. (2020):

* ``boundaries``: head and tail boundaries and the relation type are correct;
* ``strict``: additionally, the entity types of head and tail are correct.

Directional relations must match in direction. COMPARE and CONJUNCTION are symmetric
and are matched irrespective of argument order.

Micro scores pool all decisions; macro F1 is the unweighted mean of per-class F1 over the
classes present in the gold annotations, so rare classes weigh as much as frequent ones.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from scigraph.schema import SYMMETRIC_RELATIONS, Relation, Span


def prf(tp: int, n_pred: int, n_gold: int) -> dict[str, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {
        "precision": round(100 * p, 2),
        "recall": round(100 * r, 2),
        "f1": round(100 * f, 2),
        "tp": tp,
        "n_pred": n_pred,
        "n_gold": n_gold,
    }


def macro_f1(per_type: dict[str, dict]) -> float:
    classes = [v for v in per_type.values() if v["n_gold"] > 0]
    return round(sum(v["f1"] for v in classes) / len(classes), 2) if classes else 0.0


def entity_scores(gold: list[list[Span]], pred: list[list[Span]]) -> dict:
    tp = n_pred = n_gold = 0
    per_type: dict[str, Counter] = {}
    for g_sent, p_sent in zip(gold, pred, strict=True):
        g = {(s.start, s.end, s.label) for s in g_sent}
        p = {(s.start, s.end, s.label) for s in p_sent}
        tp += len(g & p)
        n_pred += len(p)
        n_gold += len(g)
        for key, bucket in ((g, "gold"), (p, "pred"), (g & p, "tp")):
            for _, _, label in key:
                per_type.setdefault(label, Counter())[bucket] += 1
    untyped_tp = sum(
        len({(s.start, s.end) for s in g} & {(s.start, s.end) for s in p})
        for g, p in zip(gold, pred, strict=True)
    )
    per = {k: prf(v["tp"], v["pred"], v["gold"]) for k, v in sorted(per_type.items())}
    return {
        "micro": prf(tp, n_pred, n_gold),
        "macro_f1": macro_f1(per),
        "span_only": prf(untyped_tp, n_pred, n_gold),
        "per_type": per,
    }


def _relation_key(r: Relation, strict: bool) -> tuple:
    h = (r.head.start, r.head.end) + ((r.head.label,) if strict else ())
    t = (r.tail.start, r.tail.end) + ((r.tail.label,) if strict else ())
    if r.label in SYMMETRIC_RELATIONS:
        h, t = sorted([h, t])
    return (h, t, r.label)


def relation_scores(gold: Iterable[list[Relation]], pred: Iterable[list[Relation]]) -> dict:
    gold, pred = list(gold), list(pred)
    out = {}
    for name, strict in (("boundaries", False), ("strict", True)):
        tp = n_pred = n_gold = 0
        per_type: dict[str, Counter] = {}
        for g_sent, p_sent in zip(gold, pred, strict=True):
            g = {_relation_key(r, strict) for r in g_sent}
            p = {_relation_key(r, strict) for r in p_sent}
            tp += len(g & p)
            n_pred += len(p)
            n_gold += len(g)
            for keys, bucket in ((g, "gold"), (p, "pred"), (g & p, "tp")):
                for k in keys:
                    per_type.setdefault(k[2], Counter())[bucket] += 1
        per = {k: prf(v["tp"], v["pred"], v["gold"]) for k, v in sorted(per_type.items())}
        out[name] = {"micro": prf(tp, n_pred, n_gold), "macro_f1": macro_f1(per), "per_type": per}
    return out


def relation_confusion(
    gold: list[list[Relation]], pred: list[list[Relation]]
) -> dict[str, dict[str, int]]:
    """Label confusion on unordered entity-boundary pairs.

    A pair present on only one side is counted against ``NO-RELATION``. Direction is
    ignored here; directional errors are visible in :func:`relation_scores`.
    """

    def keyed(rels: list[Relation]) -> dict[tuple, str]:
        return {
            tuple(sorted([(r.head.start, r.head.end), (r.tail.start, r.tail.end)])): r.label
            for r in rels
        }

    matrix: dict[str, Counter] = {}
    for g_sent, p_sent in zip(gold, pred, strict=True):
        g, p = keyed(g_sent), keyed(p_sent)
        for pair in set(g) | set(p):
            matrix.setdefault(g.get(pair, "NO-RELATION"), Counter())[
                p.get(pair, "NO-RELATION")
            ] += 1
    return {k: dict(v) for k, v in sorted(matrix.items())}
