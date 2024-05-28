from __future__ import annotations

import torchtext; torchtext.disable_torchtext_deprecation_warning()
from torchtext.vocab import vocab

from collections import Counter, OrderedDict


class Token:
    UNK = '<unk>'
    PAD = '<pad>'
    BOS = '<bos>'
    EOS = '<eos>'
    DEFAULT = [PAD, UNK, BOS, EOS]


def build_vocab(raw_dataset, minimum_frequency=1, tokenizer=lambda x: x, specials=Token.DEFAULT):
    data = [str(word) for data in raw_dataset for word in tokenizer(data)]
    frequency = Counter(data)
    sorted_by_freq_tuples = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq_tuples)
    return vocab(ordered_dict, min_freq=minimum_frequency, specials=specials, special_first=True)
