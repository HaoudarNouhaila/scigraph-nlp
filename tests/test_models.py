import torch

from scigraph.data.scierc import document_to_sentences
from scigraph.models.encoding import encode_words
from scigraph.models.relation import (
    RE_LABEL2ID,
    RelationConfig,
    RelationModel,
    build_instance,
    build_instances,
    marker_tokens,
    merge_predictions,
)
from scigraph.models.relation import (
    collate as re_collate,
)
from scigraph.models.span_ner import (
    NER_LABEL2ID,
    SpanNERConfig,
    SpanNERModel,
    build_features,
    collate,
    decode,
    enumerate_spans,
)
from scigraph.schema import Span


def tiny_encoder(vocab_size=500):
    from transformers import BertConfig, BertModel

    cfg = BertConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=48,
        max_position_embeddings=128,
    )
    torch.manual_seed(0)
    return BertModel(cfg)


def test_encode_words_alignment(stub_tokenizer):
    enc = encode_words(stub_tokenizer, ["neural", "net"], ["left"], ["right"], max_length=32)
    # [CLS] left neur ##al net righ ##t [SEP]
    assert enc.word_start == [2, 4] and enc.word_end == [3, 4]
    assert len(enc.input_ids) == 8


def test_enumerate_spans_respects_max_length():
    spans = enumerate_spans(4, 2)
    assert (0, 1) in spans and (0, 2) not in spans and len(spans) == 7


def test_span_features_label_nested_spans(stub_tokenizer, scierc_doc):
    sent = document_to_sentences(scierc_doc)[0]
    feats = build_features(stub_tokenizer, sent, SpanNERConfig(max_span_length=4, context_window=0))
    labels = dict(zip(feats.spans, feats.labels, strict=True))
    assert labels[(6, 7)] == NER_LABEL2ID["Task"]
    assert labels[(7, 7)] == NER_LABEL2ID["Task"]
    assert labels[(0, 0)] == 0


def test_decode_keeps_or_removes_overlap():
    spans = [(0, 1), (1, 1), (3, 3)]
    probs = torch.tensor(
        [
            [0.1, 0.0, 0.9, 0, 0, 0, 0],  # Method (0,1)
            [0.2, 0.8, 0.0, 0, 0, 0, 0],  # Task (1,1), nested in (0,1)
            [0.9, 0.1, 0.0, 0, 0, 0, 0],  # O
        ]
    )
    nested = decode(spans, probs, allow_overlap=True)
    flat = decode(spans, probs, allow_overlap=False)
    assert [s for s, _ in nested] == [Span(0, 1, "Method"), Span(1, 1, "Task")]
    assert [s for s, _ in flat] == [Span(0, 1, "Method")]


def test_span_ner_forward_and_learning(stub_tokenizer, scierc_doc):
    sents = document_to_sentences(scierc_doc)
    cfg = SpanNERConfig(
        max_span_length=4, context_window=0, hidden_dim=32, width_embedding_dim=8, dropout=0.0
    )
    feats = [build_features(stub_tokenizer, s, cfg) for s in sents]
    model = SpanNERModel(cfg, encoder=tiny_encoder())
    batch = collate(feats, 0)
    labels = batch.pop("labels")
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(15):
        logits = model(**batch)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert logits.shape[:2] == labels.shape
    assert losses[-1] < losses[0] * 0.5


def test_relation_markers_are_typed_and_nested(stub_tokenizer):
    tokens = ["machine", "translation", "model"]
    head = Span(0, 2, "Method")
    tail = Span(0, 1, "Task")
    for t in marker_tokens():
        stub_tokenizer.vocab.setdefault(t, len(stub_tokenizer.vocab))
    inst = build_instance(stub_tokenizer, tokens, head, tail, "USED-FOR", 64)
    inv = {v: k for k, v in stub_tokenizer.vocab.items()}
    pieces = [inv[i] for i in inst.input_ids]
    assert pieces[inst.head_pos] == "<S:Method>"
    assert pieces[inst.tail_pos] == "<O:Task>"
    # inner (tail) span closes before the outer (head) span
    assert pieces.index("</O:Task>") < pieces.index("</S:Method>")
    assert inst.label == RE_LABEL2ID["USED-FOR"]


def test_symmetric_relations_doubled_only_for_training(stub_tokenizer, scierc_doc):
    from scigraph.schema import Relation

    sent = document_to_sentences(scierc_doc)[0]
    a, b = sent.entities[0], sent.entities[1]
    sent.relations = [Relation(a, b, "CONJUNCTION")]
    train = build_instances(stub_tokenizer, [sent], 64, training=True)
    test = build_instances(stub_tokenizer, [sent], 64, training=False)
    assert sum(i.label != 0 for i in train) == 2
    assert sum(i.label != 0 for i in test) == 1


def test_merge_predictions_canonicalises_symmetric(stub_tokenizer, scierc_doc):
    sent = document_to_sentences(scierc_doc)[0]
    insts = build_instances(stub_tokenizer, [sent], 64)
    probs = torch.zeros(len(insts), len(RE_LABEL2ID))
    probs[:, 0] = 1.0
    for k, inst in enumerate(insts):
        if {inst.head, inst.tail} == {sent.entities[0], sent.entities[1]}:
            probs[k] = 0
            probs[k, RE_LABEL2ID["COMPARE"]] = 0.9 if inst.head == sent.entities[1] else 0.7
    merged = merge_predictions(insts, probs)[0]
    assert len(merged) == 1
    rel, score = merged[0]
    assert rel.head == sent.entities[0] and abs(score - 0.9) < 1e-6


def test_relation_model_forward(stub_tokenizer, scierc_doc):
    for t in marker_tokens():
        stub_tokenizer.vocab.setdefault(t, len(stub_tokenizer.vocab))
    sents = document_to_sentences(scierc_doc)
    insts = build_instances(stub_tokenizer, sents, 64)
    model = RelationModel(RelationConfig(), encoder=tiny_encoder(), vocab_size=500)
    batch = re_collate(insts, 0)
    batch.pop("labels")
    out = model(**batch)
    assert out.shape == (len(insts), len(RE_LABEL2ID))
