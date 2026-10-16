"""End-to-end extraction: text or PDF -> sentences -> entities -> relations -> graph.

Two extractor back-ends share one interface:

* :class:`NeuralExtractor` loads the trained span NER and relation models;
* :class:`RuleExtractor` uses the spaCy baseline and needs no trained weights.

:func:`load_extractor` returns the neural extractor when model directories exist and
falls back to the rule-based one otherwise; the chosen back-end is always reported in
the output so that results are never presented as coming from a model that was not used.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

from scigraph.config import graph_settings
from scigraph.pipeline.graph_builder import GraphBuilder, KnowledgeGraph
from scigraph.pipeline.text import TokenizedSentence, clean_text, split_sentences
from scigraph.schema import Sentence, Span


class RuleExtractor:
    name = "rule-baseline"

    def extract(self, sentences: list[TokenizedSentence]) -> list[dict]:
        from scigraph.baseline.rules import rule_entities, rule_relations

        out = []
        for s in sentences:
            ents = rule_entities(s.tokens)
            rels = rule_relations(s.tokens, ents)
            out.append(_pack(s, [(e, 1.0) for e in ents], [(r, 1.0) for r in rels]))
        return out


class NeuralExtractor:
    name = "scibert-span-ner+marker-re"

    def __init__(
        self,
        ner_dir: str | Path,
        re_dir: str | Path,
        device: str = "cpu",
        ner_threshold: float | None = None,
        re_threshold: float | None = None,
    ):
        from scigraph.models import relation as re_mod
        from scigraph.models import span_ner as ner_mod

        self.device = device
        self.ner_model, self.ner_tok = ner_mod.load_model(ner_dir, device)
        self.re_model, self.re_tok = re_mod.load_model(re_dir, device)
        cfg = graph_settings()["extraction"]
        self.ner_threshold = cfg["ner_threshold"] if ner_threshold is None else ner_threshold
        self.re_threshold = cfg["re_threshold"] if re_threshold is None else re_threshold
        self.allow_overlap = cfg["ner_decoding"] == "nested"
        self.version = {
            "ner": dir_fingerprint(Path(ner_dir)),
            "re": dir_fingerprint(Path(re_dir)),
        }

    def extract(self, sentences: list[TokenizedSentence]) -> list[dict]:
        import torch

        from scigraph.models.relation import build_instances, collate, merge_predictions
        from scigraph.training.train_ner import predict as ner_predict

        sents = [Sentence("input", s.index, s.tokens) for s in sentences]
        ent_preds = ner_predict(
            self.ner_model,
            self.ner_tok,
            sents,
            self.ner_model.config,
            60,
            self.device,
            allow_overlap=self.allow_overlap,
            return_scores=True,
        )
        ent_preds = [[(e, sc) for e, sc in p if sc >= self.ner_threshold] for p in ent_preds]
        spans = [[e for e, _ in p] for p in ent_preds]
        instances = build_instances(
            self.re_tok, sents, self.re_model.config.max_seq_length, entities_override=spans
        )
        probs = []
        with torch.no_grad():
            for b in range(0, len(instances), 64):
                batch = collate(instances[b : b + 64], self.re_tok.pad_token_id)
                batch.pop("labels")
                batch = {k: v.to(self.device) for k, v in batch.items()}
                probs.append(torch.softmax(self.re_model(**batch).float(), -1).cpu())
        merged = merge_predictions(instances, torch.cat(probs), self.re_threshold) if probs else {}
        return [_pack(s, ent_preds[i], merged.get(i, [])) for i, s in enumerate(sentences)]


def surface(sent: TokenizedSentence, start: int, end: int) -> str:
    """Original text of a token span (keeps 'graph-based' instead of 'graph - based')."""
    if sent.token_offsets:
        a = sent.token_offsets[start][0] - sent.char_start
        b = sent.token_offsets[end][1] - sent.char_start
        return sent.text[a:b]
    return " ".join(sent.tokens[start : end + 1])


def _pack(sent: TokenizedSentence, ents: list[tuple[Span, float]], rels) -> dict:
    index = {(e.start, e.end): k for k, (e, _) in enumerate(ents)}
    relations = []
    for r, score in rels:
        h = index.get((r.head.start, r.head.end))
        t = index.get((r.tail.start, r.tail.end))
        if h is not None and t is not None:
            relations.append({"head": h, "tail": t, "label": r.label, "score": round(score, 4)})
    return {
        "index": sent.index,
        "text": sent.text,
        "char_start": sent.char_start,
        "tokens": sent.tokens,
        "token_offsets": sent.token_offsets,
        "entities": [
            {
                "start": e.start,
                "end": e.end,
                "label": e.label,
                "score": round(sc, 4),
                "text": surface(sent, e.start, e.end),
            }
            for e, sc in ents
        ],
        "relations": relations,
    }


def dir_fingerprint(d: Path) -> str:
    f = d / "model.pt"
    if not f.exists():
        return "missing"
    st = f.stat()
    return hashlib.sha1(f"{f.name}:{st.st_size}:{int(st.st_mtime)}".encode()).hexdigest()[:12]


def load_extractor(ner_dir: str | None = None, re_dir: str | None = None, prefer: str = "auto"):
    ner_dir = ner_dir or os.getenv("SCIGRAPH_NER_DIR", "models/ner")
    re_dir = re_dir or os.getenv("SCIGRAPH_RE_DIR", "models/re")
    have = (Path(ner_dir) / "model.pt").exists() and (Path(re_dir) / "model.pt").exists()
    if prefer == "rules" or (prefer == "auto" and not have):
        return RuleExtractor()
    if not have:
        raise FileNotFoundError(f"Trained models not found in {ner_dir} and {re_dir}")
    return NeuralExtractor(ner_dir, re_dir)


def model_version_string(extractor) -> str:
    """Compact identifier of the model weights used, stored with every relation."""
    version = getattr(extractor, "version", None)
    if not version:
        return extractor.name
    return ",".join(f"{k}:{v}" for k, v in version.items())


def doc_id_for(text: str, source: str = "") -> str:
    return "doc:" + hashlib.sha1((source + "\n" + text).encode("utf-8")).hexdigest()[:16]


def process_text(
    text: str,
    extractor,
    title: str = "",
    source: str = "text",
    doc_id: str | None = None,
    builder: GraphBuilder | None = None,
    page_starts: list[tuple[int, int]] | None = None,
) -> dict:
    """Run the full pipeline on one document and return sentences, triples and graph."""
    t0 = time.time()
    clean = clean_text(text)
    sentences = split_sentences(clean, graph_settings()["extraction"]["max_sentence_tokens"])
    extracted = extractor.extract(sentences)
    if page_starts:
        from scigraph.pipeline.pdf import page_of

        for sent in extracted:
            sent["page"] = page_of(sent["char_start"], page_starts)
    doc_id = doc_id or doc_id_for(clean, source)
    builder = builder or GraphBuilder()
    meta = {
        "title": title,
        "source": source,
        "extractor": extractor.name,
        "model_version": model_version_string(extractor),
    }
    builder.add_document(doc_id, meta, clean, extracted)
    kg = builder.finalize(graph_settings()["entity_resolution"]["near_duplicate_jaccard"])
    return {
        "doc_id": doc_id,
        "extractor": extractor.name,
        "model_version": getattr(extractor, "version", None),
        "n_sentences": len(sentences),
        "sentences": extracted,
        "graph": kg,
        "seconds": round(time.time() - t0, 3),
    }


def process_pdf(
    path_or_bytes, extractor, source: str = "pdf", builder: GraphBuilder | None = None
) -> dict:
    from scigraph.pipeline.pdf import extract_pdf

    pdf = extract_pdf(path_or_bytes)
    result = process_text(
        pdf.text,
        extractor,
        title=pdf.title,
        source=source,
        builder=builder,
        page_starts=pdf.page_starts,
    )
    result["n_pages"] = pdf.n_pages
    return result


def triples_table(kg: KnowledgeGraph) -> list[dict]:
    rows = []
    for t in kg.triples.values():
        h, tl = kg.entities[t.head_id], kg.entities[t.tail_id]
        rows.append(
            {
                "head": h.name,
                "head_type": h.type,
                "relation": t.relation,
                "tail": tl.name,
                "tail_type": tl.type,
                "score": round(t.score, 4),
                "support": t.support,
                "evidence": t.evidence[0].sentence if t.evidence else "",
            }
        )
    return sorted(rows, key=lambda r: (-r["support"], -r["score"]))
