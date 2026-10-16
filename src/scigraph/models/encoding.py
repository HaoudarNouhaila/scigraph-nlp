"""Word-to-subword alignment shared by the NER and relation models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Encoded:
    input_ids: list[int]
    # For every word of the *target* sentence: index of its first and last subword.
    word_start: list[int]
    word_end: list[int]


def encode_words(
    tokenizer,
    words: list[str],
    left_context: list[str] | None = None,
    right_context: list[str] | None = None,
    max_length: int = 256,
    context_budget: int | None = None,
) -> Encoded:
    """Encode `words` (optionally surrounded by context words) as [CLS] ... [SEP].

    Context words are trimmed symmetrically when the sequence would exceed
    `max_length`; the target sentence itself is truncated only as a last resort.
    """
    left_context = list(left_context or [])
    right_context = list(right_context or [])
    pieces = [tokenizer.tokenize(w) or [tokenizer.unk_token] for w in words]
    n_target = sum(len(p) for p in pieces)
    budget = max_length - 2 - n_target
    if context_budget is not None:
        budget = min(budget, context_budget)
    left_pieces: list[str] = []
    right_pieces: list[str] = []
    if budget > 0:
        lp = [t for w in left_context for t in (tokenizer.tokenize(w) or [tokenizer.unk_token])]
        rp = [t for w in right_context for t in (tokenizer.tokenize(w) or [tokenizer.unk_token])]
        half = budget // 2
        take_left = min(len(lp), max(half, budget - len(rp)))
        take_right = min(len(rp), budget - take_left)
        left_pieces = lp[len(lp) - take_left :] if take_left else []
        right_pieces = rp[:take_right]

    tokens = [tokenizer.cls_token] + left_pieces
    word_start, word_end = [], []
    for p in pieces:
        if len(tokens) + len(p) > max_length - 1:
            # Truncate the sentence: remaining words map to the last valid position.
            pos = len(tokens) - 1
            word_start.append(pos)
            word_end.append(pos)
            continue
        word_start.append(len(tokens))
        tokens.extend(p)
        word_end.append(len(tokens) - 1)
    room = max_length - 1 - len(tokens)
    tokens.extend(right_pieces[: max(room, 0)])
    tokens.append(tokenizer.sep_token)
    return Encoded(tokenizer.convert_tokens_to_ids(tokens), word_start, word_end)


def pad_batch(sequences: list[list[int]], pad_id: int) -> tuple[list[list[int]], list[list[int]]]:
    width = max(len(s) for s in sequences)
    ids = [s + [pad_id] * (width - len(s)) for s in sequences]
    mask = [[1] * len(s) + [0] * (width - len(s)) for s in sequences]
    return ids, mask
