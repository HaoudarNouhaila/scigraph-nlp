import pytest

from scigraph.data.scierc import (
    SciERCFormatError,
    compute_statistics,
    document_to_sentences,
    overlap_kind,
)
from scigraph.evaluation.metrics import entity_scores, relation_confusion, relation_scores
from scigraph.schema import Relation, Span


def test_document_offsets_become_sentence_local(scierc_doc):
    s1, s2 = document_to_sentences(scierc_doc)
    assert s1.text_of(s1.entities[0]) == "neural network"
    assert s2.entities[0] == Span(0, 0, "Metric")  # doc offset 9 -> local 0
    assert s2.relations[0].tail.label == "Generic"


def test_nested_entities_are_preserved(scierc_doc):
    s1 = document_to_sentences(scierc_doc)[0]
    assert Span(6, 7, "Task") in s1.entities and Span(7, 7, "Task") in s1.entities
    assert overlap_kind(Span(6, 7, "Task"), Span(7, 7, "Task")) == "nested"
    assert overlap_kind(Span(0, 2, "X"), Span(2, 4, "X")) == "crossing"
    assert overlap_kind(Span(0, 1, "X"), Span(3, 4, "X")) is None


def test_cross_sentence_span_is_rejected(scierc_doc):
    bad = dict(scierc_doc, ner=[[[7, 9, "Task"]], []], relations=[[], []])
    with pytest.raises(SciERCFormatError):
        document_to_sentences(bad)


def test_statistics(scierc_doc):
    st = compute_statistics([scierc_doc])
    assert st["entities"] == 5 and st["relations"] == 2
    assert st["overlapping_entity_pairs"] == {"nested": 1}
    assert st["entity_types"]["Task"] == 2


def test_entity_scores():
    gold = [[Span(0, 1, "Method"), Span(3, 3, "Task")]]
    pred = [[Span(0, 1, "Method"), Span(3, 3, "Method"), Span(5, 5, "Task")]]
    s = entity_scores(gold, pred)
    assert s["micro"]["tp"] == 1
    assert s["micro"]["precision"] == pytest.approx(33.33, abs=0.01)
    assert s["micro"]["recall"] == 50.0
    assert s["span_only"]["tp"] == 2


def test_relation_scores_direction_and_symmetry():
    a, b = Span(0, 0, "Method"), Span(2, 2, "Task")
    gold = [[Relation(a, b, "USED-FOR"), Relation(a, b, "CONJUNCTION")]]
    # USED-FOR reversed is wrong; CONJUNCTION reversed is right.
    pred = [[Relation(b, a, "USED-FOR"), Relation(b, a, "CONJUNCTION")]]
    s = relation_scores(gold, pred)
    assert s["boundaries"]["micro"]["tp"] == 1


def test_strict_requires_entity_types():
    gold = [[Relation(Span(0, 0, "Method"), Span(2, 2, "Task"), "USED-FOR")]]
    pred = [[Relation(Span(0, 0, "Material"), Span(2, 2, "Task"), "USED-FOR")]]
    s = relation_scores(gold, pred)
    assert s["boundaries"]["micro"]["f1"] == 100.0
    assert s["strict"]["micro"]["f1"] == 0.0


def test_confusion_counts_missing_as_no_relation():
    a, b, c = Span(0, 0, "Method"), Span(2, 2, "Task"), Span(4, 4, "Task")
    gold = [[Relation(a, b, "USED-FOR")]]
    pred = [[Relation(a, b, "PART-OF"), Relation(a, c, "USED-FOR")]]
    m = relation_confusion(gold, pred)
    assert m["USED-FOR"]["PART-OF"] == 1
    assert m["NO-RELATION"]["USED-FOR"] == 1
