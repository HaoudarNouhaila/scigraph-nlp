from scigraph.baseline.rules import rule_entities, rule_relations, type_for
from scigraph.pipeline.extractor import RuleExtractor, process_pdf, process_text, triples_table
from scigraph.pipeline.graph_builder import GraphBuilder, entity_scope
from scigraph.pipeline.normalize import (
    acronym_matches,
    find_acronyms,
    normalize_mention,
    singularize,
)
from scigraph.pipeline.text import clean_text, split_sentences


def test_normalize_mention():
    assert normalize_mention("The Neural Networks") == "neural network"
    assert normalize_mention("graph - based  methods") == "graph-based method"
    assert normalize_mention("corpus") == "corpus"
    assert singularize("strategies") == "strategy"
    assert singularize("analysis") == "analysis"


def test_acronyms():
    found = find_acronyms(
        "We train a conditional random field (CRF) and a hidden Markov model (HMM)."
    )
    assert found == {"CRF": "conditional random field", "HMM": "hidden Markov model"}
    assert acronym_matches("POS", "part of speech")
    assert not acronym_matches("SVM", "neural network")


def test_clean_text_repairs_pdf_artifacts():
    assert clean_text("seg-\nmentation of\nimages") == "segmentation of images"


def test_split_sentences_keeps_offsets():
    text = "We study parsing. It works well."
    sents = split_sentences(text)
    assert len(sents) == 2
    s = sents[1]
    a, b = s.token_offsets[0]
    assert text[a:b] == s.tokens[0] == "It"


def test_rule_baseline():
    assert type_for("BLEU score") == "Metric"
    assert type_for("Penn Treebank corpus") == "Material"
    toks = ["We", "use", "a", "neural", "network", "for", "machine", "translation", "."]
    ents = rule_entities(toks)
    rels = rule_relations(toks, ents)
    texts = {" ".join(toks[e.start : e.end + 1]) for e in ents}
    assert "neural network" in texts
    assert any(r.label == "USED-FOR" for r in rels)


def test_dedup_is_type_and_scope_aware():
    b = GraphBuilder()
    sent = {
        "index": 0,
        "text": "x",
        "tokens": ["parsing", "parsing", "this", "method"],
        "entities": [
            {"start": 0, "end": 0, "label": "Task", "score": 0.9},
            {"start": 1, "end": 1, "label": "Method", "score": 0.9},
            {"start": 2, "end": 3, "label": "Generic", "score": 0.9},
        ],
        "relations": [{"head": 1, "tail": 0, "label": "USED-FOR", "score": 0.8}],
    }
    b.add_document("d1", {}, "parsing parsing this method", [sent])
    b.add_document("d2", {}, "parsing parsing this method", [sent])
    kg = b.finalize()
    names = sorted((e.name, e.type, e.scope) for e in kg.entities.values())
    # Task and Method 'parsing' are different nodes, each shared across documents;
    # the Generic mention stays document-scoped.
    assert ("parsing", "Task", "global") in names and ("parsing", "Method", "global") in names
    assert sum(1 for n in names if n[1] == "Generic") == 2
    assert len(kg.triples) == 1
    t = next(iter(kg.triples.values()))
    assert t.support == 2 and {e.doc_id for e in t.evidence} == {"d1", "d2"}
    assert entity_scope("d1", "Method", "method") == "d1"


def test_acronym_merges_within_document_only():
    b = GraphBuilder()
    text = "a conditional random field (CRF) . the CRF"
    sent = {
        "index": 0,
        "text": text,
        "tokens": ["a", "conditional", "random", "field", "(", "CRF", ")", ".", "the", "CRF"],
        "entities": [
            {"start": 1, "end": 3, "label": "Method", "score": 1},
            {"start": 9, "end": 9, "label": "Method", "score": 1},
        ],
        "relations": [],
    }
    b.add_document("d1", {}, text, [sent])
    kg = b.finalize()
    assert len(kg.entities) == 1
    assert next(iter(kg.entities.values())).aliases == ["CRF"]


def test_process_text_end_to_end(sample_texts):
    b = GraphBuilder()
    for i, text in enumerate(sample_texts):
        result = process_text(text, RuleExtractor(), title=f"s{i}", builder=b)
    kg = result["graph"]
    assert result["extractor"] == "rule-baseline"
    assert len(kg.documents) == 3 and kg.triples
    for t in kg.triples.values():
        assert t.evidence and t.evidence[0].sentence
    rows = triples_table(kg)
    assert {"head", "relation", "tail", "evidence"} <= set(rows[0])


def test_process_pdf(tmp_path):
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "We use a neural network for machine translation.", fontsize=11)
    page.insert_text((72, 120), "References", fontsize=11)
    page.insert_text((72, 140), "Smith, J. A paper about parsing.", fontsize=11)
    pdf_bytes = doc.tobytes()
    doc.close()
    result = process_pdf(pdf_bytes, RuleExtractor())
    assert result["n_pages"] == 1
    joined = " ".join(s["text"] for s in result["sentences"])
    assert "machine translation" in joined and "Smith" not in joined
