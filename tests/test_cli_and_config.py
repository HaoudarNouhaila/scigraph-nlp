import json

import pytest

from scigraph.cli import _scierc_graph, main
from scigraph.config import graph_settings
from scigraph.data.scierc import document_to_sentences, write_jsonl
from scigraph.pipeline.normalize import detokenize


@pytest.fixture
def processed_dir(tmp_path, scierc_doc):
    sents = document_to_sentences(scierc_doc)
    write_jsonl((s.to_dict() for s in sents), tmp_path / "test.jsonl")
    return tmp_path


def test_graph_settings_defaults_and_file():
    s = graph_settings()
    assert 0 < s["extraction"]["ner_threshold"] <= 1
    assert s["ingestion"]["batch_size"] > 0


def test_detokenize_offsets():
    text, offs = detokenize(["a", "model", "-LRB-", "CRF", "-RRB-", ",", "it"])
    assert text == "a model (CRF), it"
    assert [text[a:b] for a, b in offs] == ["a", "model", "(", "CRF", ")", ",", "it"]


def test_gold_graph_from_scierc(processed_dir):
    kg = _scierc_graph("test", str(processed_dir), None).finalize()
    assert len(kg.triples) == 2
    m = next(iter(kg.mentions.values()))
    doc_text = " ".join(e.sentence for t in kg.triples.values() for e in t.evidence)
    assert m.text and m.char_end > m.char_start and doc_text


def test_predicted_graph_uses_scores(processed_dir, tmp_path):
    rows = [
        {
            "doc_key": "TEST-1",
            "sent_id": 0,
            "entities": [[3, 4, "Method", 0.91], [6, 7, "Task", 0.8]],
            "relations": [[3, 4, 6, 7, "USED-FOR", 0.77]],
        },
        {"doc_key": "TEST-1", "sent_id": 1, "entities": [], "relations": []},
    ]
    path = tmp_path / "pred.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    kg = _scierc_graph("test", str(processed_dir), str(path)).finalize()
    t = next(iter(kg.triples.values()))
    assert t.score == pytest.approx(0.77)
    assert {m.score for m in kg.mentions.values()} == {0.91, 0.8}


def test_cli_extract_with_rules(tmp_path, capsys):
    src = tmp_path / "doc.txt"
    src.write_text("We use a neural network for machine translation.", encoding="utf-8")
    out = tmp_path / "out.json"
    main(["extract", str(src), "--out", str(out), "--extractor", "rules"])
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["extractor"] == "rule-baseline" and data["triples"]
