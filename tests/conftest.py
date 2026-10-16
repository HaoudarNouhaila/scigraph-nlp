from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples"


class WordPieceStub:
    """Minimal tokenizer with the subset of the Hugging Face API used by the models.

    Words longer than four characters are split into two pieces, so that subword
    alignment is exercised without downloading a real vocabulary.
    """

    cls_token, sep_token, unk_token, pad_token = "[CLS]", "[SEP]", "[UNK]", "[PAD]"
    pad_token_id = 0

    def __init__(self):
        self.vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}

    def tokenize(self, word: str) -> list[str]:
        w = word.lower()
        return [w[:4], "##" + w[4:]] if len(w) > 4 else [w]

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.setdefault(t, len(self.vocab)) for t in tokens]

    def __len__(self):
        return len(self.vocab)


@pytest.fixture
def stub_tokenizer():
    return WordPieceStub()


@pytest.fixture
def sample_texts() -> list[str]:
    return [p.read_text(encoding="utf-8") for p in sorted(SAMPLES.glob("abstract_*.txt"))]


SCIERC_DOC = {
    "doc_key": "TEST-1",
    "sentences": [
        ["We", "use", "a", "neural", "network", "for", "machine", "translation", "."],
        ["BLEU", "is", "used", "to", "evaluate", "it", "."],
    ],
    "ner": [
        [[3, 4, "Method"], [6, 7, "Task"], [7, 7, "Task"]],
        [[9, 9, "Metric"], [14, 14, "Generic"]],
    ],
    "relations": [
        [[3, 4, 6, 7, "USED-FOR"]],
        [[9, 9, 14, 14, "EVALUATE-FOR"]],
    ],
    "clusters": [],
}


@pytest.fixture
def scierc_doc() -> dict:
    return SCIERC_DOC


def neo4j_available() -> bool:
    """Neo4j tests wipe the target database, so they only run when explicitly enabled
    with SCIGRAPH_NEO4J_TESTS=1 (a dedicated test instance, e.g. the CI service)."""
    if os.getenv("SCIGRAPH_NEO4J_TESTS") != "1":
        return False
    try:
        from scigraph.graph.neo4j_store import Neo4jStore

        s = Neo4jStore()
        s.verify_connectivity()
        s.close()
        return True
    except Exception:  # noqa: BLE001
        return False
