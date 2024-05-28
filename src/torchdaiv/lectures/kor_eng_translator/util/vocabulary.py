from __future__ import annotations

import torchtext; torchtext.disable_torchtext_deprecation_warning()
from torchtext.vocab import vocab

from collections import Counter, OrderedDict

import spacy


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


def load_tokenizer(backend=spacy, language=""):
    """
    토큰화 방식: 띄어쓰기 기준으로 토큰화, + => +## 으로 분리
    """
    tokenizer = backend.load(language)
    return lambda x: [tk for token in tokenizer(x) for tk in token.lemma_.replace("+", "+##").split('+')]
