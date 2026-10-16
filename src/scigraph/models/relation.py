"""Relation classification with typed entity markers on top of SciBERT.

For an ordered candidate pair (head, tail) inside one sentence the input is::

    [CLS] ... <S:Method> neural network </S:Method> ... <O:Task> parsing </O:Task> ... [SEP]

The marker tokens encode the role (subject ``S`` / object ``O``) and the entity type.
The pair representation is the concatenation of the hidden states of the two opening
markers, classified into the seven SciERC relation types plus ``NO-RELATION``
(Zhong and Chen, 2021, "A Frustratingly Easy Approach for Entity and Relation
Extraction"). Because head and tail are marked differently, the model is directional:
(A, B) and (B, A) are distinct instances.

Symmetric relations (COMPARE, CONJUNCTION) are annotated once in SciERC. During training
both directions are used as positives; at inference the two directional scores are
merged and the pair is emitted once with the earlier span first.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from scigraph.models.encoding import pad_batch
from scigraph.schema import (
    ENTITY_TYPES,
    NO_RELATION,
    RELATION_TYPES,
    SYMMETRIC_RELATIONS,
    Relation,
    Sentence,
    Span,
)

RE_LABELS: tuple[str, ...] = (NO_RELATION, *RELATION_TYPES)
RE_LABEL2ID = {label: i for i, label in enumerate(RE_LABELS)}


def marker_tokens() -> list[str]:
    toks = []
    for role in ("S", "O"):
        for t in ENTITY_TYPES:
            toks += [f"<{role}:{t}>", f"</{role}:{t}>"]
    return toks


@dataclass
class RelationConfig:
    encoder_name: str = "allenai/scibert_scivocab_uncased"
    max_seq_length: int = 256
    dropout: float = 0.1
    context_window: int = 0  # PURE uses 100 for SciERC; 0 keeps training cheap

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> RelationConfig:
        return cls(**json.loads(path.read_text(encoding="utf-8")))


class RelationModel(nn.Module):
    def __init__(self, config: RelationConfig, encoder: nn.Module | None = None, vocab_size=None):
        super().__init__()
        from transformers import AutoModel

        self.config = config
        self.encoder = (
            encoder if encoder is not None else AutoModel.from_pretrained(config.encoder_name)
        )
        if (
            vocab_size is not None
            and self.encoder.get_input_embeddings().num_embeddings != vocab_size
        ):
            self.encoder.resize_token_embeddings(vocab_size)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(2 * hidden)
        self.classifier = nn.Linear(2 * hidden, len(RE_LABELS))

    def forward(self, input_ids, attention_mask, head_pos, tail_pos) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        idx = torch.arange(hidden.size(0), device=hidden.device)
        rep = torch.cat([hidden[idx, head_pos], hidden[idx, tail_pos]], dim=-1)
        return self.classifier(self.dropout(self.layer_norm(rep)))


@dataclass
class RelationInstance:
    sentence_index: int
    head: Span
    tail: Span
    label: int
    input_ids: list[int]
    head_pos: int
    tail_pos: int


def build_instance(
    tokenizer,
    tokens: list[str],
    head: Span,
    tail: Span,
    label: str,
    max_length: int,
    sentence_index: int = 0,
) -> RelationInstance:
    pieces: list[str] = [tokenizer.cls_token]
    head_pos = tail_pos = 0
    for w_idx, word in enumerate(tokens):
        if w_idx == head.start:
            head_pos = len(pieces)
            pieces.append(f"<S:{head.label}>")
        if w_idx == tail.start:
            tail_pos = len(pieces)
            pieces.append(f"<O:{tail.label}>")
        pieces.extend(tokenizer.tokenize(word) or [tokenizer.unk_token])
        # Close inner spans first so nested markers are well formed.
        closers = sorted(
            [(s, r) for s, r in ((head, "S"), (tail, "O")) if s.end == w_idx],
            key=lambda x: x[0].start,
            reverse=True,
        )
        for s, role in closers:
            pieces.append(f"</{role}:{s.label}>")
    pieces.append(tokenizer.sep_token)
    if len(pieces) > max_length:
        # Keep a window that contains both markers.
        lo = max(0, min(head_pos, tail_pos) - 8)
        hi = min(len(pieces), lo + max_length - 2)
        if max(head_pos, tail_pos) >= hi:
            lo = max(0, max(head_pos, tail_pos) - (max_length - 10))
            hi = min(len(pieces), lo + max_length - 2)
        body = pieces[max(lo, 1) : hi]
        shift = max(lo, 1) - 1
        pieces = [tokenizer.cls_token, *body, tokenizer.sep_token]
        head_pos -= shift
        tail_pos -= shift
    ids = tokenizer.convert_tokens_to_ids(pieces)
    return RelationInstance(sentence_index, head, tail, RE_LABEL2ID[label], ids, head_pos, tail_pos)


def candidate_pairs(entities: list[Span]) -> list[tuple[Span, Span]]:
    return [(h, t) for h in entities for t in entities if h != t]


def gold_label_map(relations: list[Relation], symmetric_both_ways: bool) -> dict:
    gold: dict[tuple[Span, Span], str] = {}
    for r in relations:
        gold[(r.head, r.tail)] = r.label
        if symmetric_both_ways and r.label in SYMMETRIC_RELATIONS:
            gold.setdefault((r.tail, r.head), r.label)
    return gold


def build_instances(
    tokenizer,
    sentences: list[Sentence],
    max_length: int,
    entities_override: list[list[Span]] | None = None,
    training: bool = False,
) -> list[RelationInstance]:
    """All ordered entity pairs of every sentence.

    With ``entities_override`` (Experiment B), candidate pairs come from predicted
    entities; their gold label is used only when both predicted spans exactly match
    the boundaries of an annotated relation (with matching types).
    """
    out = []
    for idx, s in enumerate(sentences):
        ents = entities_override[idx] if entities_override is not None else s.entities
        gold = gold_label_map(s.relations, symmetric_both_ways=training)
        for h, t in candidate_pairs(ents):
            label = gold.get((h, t), NO_RELATION)
            out.append(build_instance(tokenizer, s.tokens, h, t, label, max_length, idx))
    return out


def collate(batch: list[RelationInstance], pad_id: int) -> dict[str, torch.Tensor]:
    ids, mask = pad_batch([b.input_ids for b in batch], pad_id)
    return {
        "input_ids": torch.tensor(ids),
        "attention_mask": torch.tensor(mask),
        "head_pos": torch.tensor([b.head_pos for b in batch]),
        "tail_pos": torch.tensor([b.tail_pos for b in batch]),
        "labels": torch.tensor([b.label for b in batch]),
    }


def merge_predictions(
    instances: list[RelationInstance], probs: torch.Tensor, threshold: float = 0.0
) -> dict[int, list[tuple[Relation, float]]]:
    """Convert per-instance probabilities into relations per sentence.

    Symmetric labels predicted in either direction are merged into one relation whose
    head is the span that starts first; the score is the maximum of both directions.
    """
    per_sent: dict[int, dict[tuple, tuple[Relation, float]]] = {}
    for inst, p in zip(instances, probs, strict=True):
        score, lab = float(p.max()), int(p.argmax())
        if lab == 0 or score < threshold:
            continue
        label = RE_LABELS[lab]
        head, tail = inst.head, inst.tail
        if label in SYMMETRIC_RELATIONS and (tail.start, tail.end) < (head.start, head.end):
            head, tail = tail, head
        key = (head, tail, label)
        bucket = per_sent.setdefault(inst.sentence_index, {})
        if key not in bucket or bucket[key][1] < score:
            bucket[key] = (Relation(head, tail, label), score)
    return {k: list(v.values()) for k, v in per_sent.items()}


def save_model(model: RelationModel, tokenizer, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.config.save(out / "relation_config.json")
    model.encoder.config.save_pretrained(out / "encoder")
    tokenizer.save_pretrained(out / "encoder")
    torch.save(model.state_dict(), out / "model.pt")


def load_model(model_dir: str | Path, device: str = "cpu"):
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    d = Path(model_dir)
    config = RelationConfig.load(d / "relation_config.json")
    tokenizer = AutoTokenizer.from_pretrained(d / "encoder")
    encoder = AutoModel.from_config(AutoConfig.from_pretrained(d / "encoder"))
    model = RelationModel(config, encoder=encoder, vocab_size=len(tokenizer))
    model.load_state_dict(torch.load(d / "model.pt", map_location=device))
    model.to(device).eval()
    return model, tokenizer


def prepare_tokenizer(tokenizer):
    tokenizer.add_special_tokens({"additional_special_tokens": marker_tokens()})
    return tokenizer
