"""Rule-based baseline for entity and relation extraction using spaCy.

Entities: spaCy noun chunks, trimmed of determiners and pronouns, typed with keyword
lexicons on the head noun. Relations: lexical patterns between consecutive entities.
The baseline is deliberately simple and transparent; it provides a lower bound for
the neural models and a fallback extractor when no trained weights are available.
"""

from __future__ import annotations

import re
from functools import lru_cache

from scigraph.schema import Relation, Span

METRIC_HEADS = {
    "accuracy", "precision", "recall", "f1", "f-measure", "f-score", "bleu", "error",
    "rate", "perplexity", "score", "auc", "map", "wer", "measure", "metric", "metrics",
    "performance", "rouge", "correlation", "speed", "efficiency", "robustness",
}
TASK_HEADS = {
    "task", "tasks", "problem", "problems", "recognition", "translation", "parsing",
    "classification", "detection", "segmentation", "retrieval", "generation", "tagging",
    "extraction", "summarization", "understanding", "estimation", "tracking",
    "identification", "prediction", "disambiguation", "resolution", "alignment",
    "learning", "inference", "search", "matching", "analysis", "reconstruction",
    "localization", "reasoning", "answering", "labeling", "synthesis", "acquisition",
}
METHOD_HEADS = {
    "method", "methods", "model", "models", "algorithm", "algorithms", "approach",
    "approaches", "technique", "techniques", "framework", "system", "systems", "network",
    "networks", "classifier", "classifiers", "parser", "parsers", "architecture",
    "procedure", "strategy", "scheme", "formalism", "grammar", "grammars", "tagger",
    "estimator", "kernel", "machine", "machines", "decoder", "encoder", "heuristic",
    "heuristics", "tool", "toolkit", "mechanism", "representation", "representations",
}
MATERIAL_HEADS = {
    "corpus", "corpora", "dataset", "datasets", "data", "database", "treebank", "text",
    "texts", "images", "image", "documents", "sentences", "speech", "lexicon", "videos",
    "collection", "benchmark", "benchmarks", "wordnet", "web", "resources", "resource",
}
GENERIC_PHRASES = {
    "it", "this", "they", "them", "this method", "this approach", "this paper",
    "our method", "our approach", "our system", "the method", "the approach",
    "the system", "the model", "this model", "this technique", "this work",
    "the algorithm", "our model", "this framework", "the framework", "this task",
}
STOP_LEADING = {"a", "an", "the", "this", "these", "that", "those", "our", "their", "its",
                "such", "each", "every", "some", "any", "several", "many", "various", "both"}
PRONOUNS = {"we", "i", "you", "he", "she", "us", "which", "what", "who", "whom"}


@lru_cache(maxsize=1)
def load_spacy(model: str = "en_core_web_sm"):
    import spacy

    try:
        return spacy.load(model)
    except OSError:  # model not installed: fall back to a blank pipeline
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        return nlp


def type_for(phrase: str) -> str:
    low = phrase.lower()
    if low in GENERIC_PHRASES:
        return "Generic"
    head = low.split()[-1]
    if head in METRIC_HEADS:
        return "Metric"
    if head in MATERIAL_HEADS:
        return "Material"
    if head in TASK_HEADS or head.endswith("ing") and len(low.split()) > 1:
        return "Task"
    if head in METHOD_HEADS:
        return "Method"
    return "OtherScientificTerm"


def rule_entities(tokens: list[str]) -> list[Span]:
    """Noun-chunk entities for a pre-tokenised sentence."""
    from spacy.tokens import Doc

    nlp = load_spacy()
    doc = Doc(nlp.vocab, words=tokens)
    for _, proc in nlp.pipeline:
        doc = proc(doc)
    spans: list[Span] = []
    if not doc.has_annotation("DEP"):
        return spans
    for chunk in doc.noun_chunks:
        start, end = chunk.start, chunk.end - 1
        text = " ".join(tokens[start : end + 1]).lower()
        if text in GENERIC_PHRASES:
            spans.append(Span(start, end, "Generic"))
            continue
        while start <= end and (tokens[start].lower() in STOP_LEADING or not tokens[start].isalnum()):
            start += 1
        if start > end or tokens[end].lower() in PRONOUNS or tokens[start].lower() in PRONOUNS:
            continue
        phrase = " ".join(tokens[start : end + 1])
        if not re.search(r"[A-Za-z]", phrase):
            continue
        spans.append(Span(start, end, type_for(phrase)))
    return spans


# Patterns are matched on the lower-cased words strictly between two entities.
RELATION_PATTERNS: list[tuple[str, str, bool]] = [
    # (regex on the gap text, label, reversed: True if the right entity is the head)
    (r"^(such as|including|like)$", "HYPONYM-OF", True),
    (r"^(is|are) (a|an)( \w+)?$", "HYPONYM-OF", False),
    (r"^(and|or|as well as|,)$", "CONJUNCTION", False),
    (r"^(compared (to|with)|versus|vs\.?|outperforms?|than)$", "COMPARE", False),
    (r"^(for|to)$", "USED-FOR", False),
    (r"^(is|are) used (for|to|in)$", "USED-FOR", False),
    (r"^(using|with|based on|by|via|through|by using|uses|use|employs?)$", "USED-FOR", True),
    (r"^(of|in)$", "FEATURE-OF", False),
    (r"^(on|in terms of)$", "EVALUATE-FOR", True),
    (r"^(evaluated on|tested on)$", "EVALUATE-FOR", True),
    (r"^(consists of|contains|includes)$", "PART-OF", True),
    (r"^(is|are) part of$", "PART-OF", False),
]
_COMPILED = [(re.compile(p), lab, rev) for p, lab, rev in RELATION_PATTERNS]


def rule_relations(tokens: list[str], entities: list[Span]) -> list[Relation]:
    ents = sorted(entities, key=lambda e: (e.start, e.end))
    rels: list[Relation] = []
    for left, right in zip(ents, ents[1:]):
        if right.start <= left.end:
            continue
        gap = " ".join(tokens[left.end + 1 : right.start]).lower().strip()
        for pattern, label, reverse in _COMPILED:
            if pattern.match(gap):
                if label == "EVALUATE-FOR" and left.label != "Metric" and right.label != "Material":
                    continue
                head, tail = (right, left) if reverse else (left, right)
                if label == "EVALUATE-FOR" and head.label not in ("Metric", "Material"):
                    head, tail = tail, head
                rels.append(Relation(head, tail, label))
                break
    return rels
