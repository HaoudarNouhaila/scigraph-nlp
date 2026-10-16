"""Span-based named entity recognition on top of a BERT-style encoder (SciBERT).

Every contiguous word span up to ``max_span_length`` words is classified independently
into one of the entity types or ``O``. Because spans are scored independently, nested
and overlapping entities are representable: two spans can both receive a non-``O``
label. A span is represented as ``[h_first_subword(start); h_last_subword(end); w(len)]``
where ``w`` is a learned width embedding (Lee et al., 2017; Zhong and Chen, 2021).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from scigraph.models.encoding import encode_words, pad_batch
from scigraph.schema import ENTITY_TYPES, NO_ENTITY, Sentence, Span

NER_LABELS: tuple[str, ...] = (NO_ENTITY, *ENTITY_TYPES)
NER_LABEL2ID = {label: i for i, label in enumerate(NER_LABELS)}


@dataclass
class SpanNERConfig:
    encoder_name: str = "allenai/scibert_scivocab_uncased"
    max_span_length: int = 8
    width_embedding_dim: int = 150
    hidden_dim: int = 150
    dropout: float = 0.2
    max_seq_length: int = 256
    context_window: int = 100  # subword budget for neighbouring sentences (0 disables)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> SpanNERConfig:
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def enumerate_spans(n_words: int, max_len: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n_words) for j in range(i, min(n_words, i + max_len))]


class SpanNERModel(nn.Module):
    def __init__(self, config: SpanNERConfig, encoder: nn.Module | None = None):
        super().__init__()
        from transformers import AutoModel

        self.config = config
        self.encoder = (
            encoder if encoder is not None else AutoModel.from_pretrained(config.encoder_name)
        )
        hidden = self.encoder.config.hidden_size
        self.width_embedding = nn.Embedding(config.max_span_length + 1, config.width_embedding_dim)
        in_dim = 2 * hidden + config.width_embedding_dim
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, len(NER_LABELS)),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        span_starts: torch.Tensor,  # (B, S) subword index of span start
        span_ends: torch.Tensor,  # (B, S) subword index of span end
        span_widths: torch.Tensor,  # (B, S) width in words
    ) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = self.dropout(hidden)
        dim = hidden.size(-1)
        start_h = torch.gather(hidden, 1, span_starts.unsqueeze(-1).expand(-1, -1, dim))
        end_h = torch.gather(hidden, 1, span_ends.unsqueeze(-1).expand(-1, -1, dim))
        width_h = self.width_embedding(span_widths)
        return self.classifier(torch.cat([start_h, end_h, width_h], dim=-1))


@dataclass
class SpanNERFeatures:
    input_ids: list[int]
    spans: list[tuple[int, int]]  # word-level spans (inclusive)
    span_starts: list[int]
    span_ends: list[int]
    span_widths: list[int]
    labels: list[int]


def build_features(
    tokenizer,
    sentence: Sentence,
    config: SpanNERConfig,
    left_context: list[str] | None = None,
    right_context: list[str] | None = None,
) -> SpanNERFeatures:
    if config.context_window <= 0:
        left_context = right_context = None
    enc = encode_words(
        tokenizer,
        sentence.tokens,
        left_context,
        right_context,
        max_length=config.max_seq_length,
        context_budget=config.context_window,
    )
    gold = {(e.start, e.end): NER_LABEL2ID[e.label] for e in sentence.entities}
    spans = enumerate_spans(len(sentence.tokens), config.max_span_length)
    return SpanNERFeatures(
        input_ids=enc.input_ids,
        spans=spans,
        span_starts=[enc.word_start[i] for i, _ in spans],
        span_ends=[enc.word_end[j] for _, j in spans],
        span_widths=[j - i + 1 for i, j in spans],
        labels=[gold.get(s, 0) for s in spans],
    )


def sentence_contexts(sentences: list[Sentence], window_words: int = 60):
    """Yield (left_context, right_context) words for each sentence, from the same document."""
    for idx, s in enumerate(sentences):
        left: list[str] = []
        right: list[str] = []
        k = idx - 1
        while k >= 0 and sentences[k].doc_key == s.doc_key and len(left) < window_words:
            left = sentences[k].tokens + left
            k -= 1
        k = idx + 1
        while (
            k < len(sentences) and sentences[k].doc_key == s.doc_key and len(right) < window_words
        ):
            right = right + sentences[k].tokens
            k += 1
        yield left[-window_words:], right[:window_words]


def collate(features: list[SpanNERFeatures], pad_id: int) -> dict[str, torch.Tensor]:
    ids, mask = pad_batch([f.input_ids for f in features], pad_id)
    n = max(len(f.spans) for f in features)

    def pad(values: list[int], fill: int) -> list[int]:
        return values + [fill] * (n - len(values))

    return {
        "input_ids": torch.tensor(ids),
        "attention_mask": torch.tensor(mask),
        "span_starts": torch.tensor([pad(f.span_starts, 0) for f in features]),
        "span_ends": torch.tensor([pad(f.span_ends, 0) for f in features]),
        "span_widths": torch.tensor([pad(f.span_widths, 1) for f in features]),
        "labels": torch.tensor([pad(f.labels, -100) for f in features]),
    }


def decode(
    spans: list[tuple[int, int]],
    probs: torch.Tensor,
    allow_overlap: bool = True,
    threshold: float = 0.0,
) -> list[tuple[Span, float]]:
    """Turn span probabilities into entities.

    With ``allow_overlap=True`` every span whose best label is not ``O`` is kept, so
    nested and crossing entities can be produced. With ``allow_overlap=False`` a greedy
    pass keeps the highest-scoring spans that do not overlap any already kept span.
    """
    scores, labels = probs.max(dim=-1)
    cands = [
        (Span(i, j, NER_LABELS[int(lab)]), float(sc))
        for (i, j), sc, lab in zip(spans, scores.tolist(), labels.tolist(), strict=True)
        if lab != 0 and sc >= threshold
    ]
    if allow_overlap:
        return sorted(cands, key=lambda x: (x[0].start, x[0].end))
    kept: list[tuple[Span, float]] = []
    for span, sc in sorted(cands, key=lambda x: -x[1]):
        if all(span.end < k.start or k.end < span.start for k, _ in kept):
            kept.append((span, sc))
    return sorted(kept, key=lambda x: (x[0].start, x[0].end))


def save_model(model: SpanNERModel, tokenizer, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.config.save(out / "span_ner_config.json")
    model.encoder.config.save_pretrained(out / "encoder")
    tokenizer.save_pretrained(out / "encoder")
    torch.save(model.state_dict(), out / "model.pt")


def load_model(model_dir: str | Path, device: str = "cpu") -> tuple[SpanNERModel, object]:
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    d = Path(model_dir)
    config = SpanNERConfig.load(d / "span_ner_config.json")
    encoder = AutoModel.from_config(AutoConfig.from_pretrained(d / "encoder"))
    model = SpanNERModel(config, encoder=encoder)
    model.load_state_dict(torch.load(d / "model.pt", map_location=device))
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(d / "encoder")
    return model, tokenizer
