from __future__ import annotations

from typing import Callable
from collections import Counter

import re


def split_string(s):
    return re.findall(r'\b\w+\b|[\.,;!\?\-+*/]', s)


class Vocabulary(dict):
    PAD = "<PAD>"
    UNK = "<UNK>"

    def __init__(self):
        super().__init__({self.PAD: 0, self.UNK: 1})

    def __setitem__(self, key, index):
        if not isinstance(key, str):
            raise ValueError("Key type must be str.")
        super().__setitem__(key, index)

    def update(self, data: list[str] | Vocabulary):
        if isinstance(data, list) or isinstance(data, Vocabulary):
            for key in data:
                if key not in self:
                    self[key] = len(self)
        else:
            raise ValueError("Input type must be list or Vocabulary.")

    def __getitem__(self, item):
        if isinstance(item, str):
            return self[item]
        else:
            raise ValueError("Input type must be str.")

    def get(self, key, default=1):
        return super().get(key, default)


def vocabulary_creator(minimum_frequency=5) -> tuple[Vocabulary, Callable]:
    new_vocab = Vocabulary()

    def convert(data: list[str]) -> list[str]:
        data = [word for line in data for word in split_string(line)]
        frequency = Counter(data)
        frequency_filtered = filter(lambda x: frequency[x] >= minimum_frequency, frequency)
        return list(set(frequency_filtered))

    def to_vocabulary(data: list[str] | Callable):
        def wrapper(dt):
            new_vocab.update(convert(dt))
            return dt

        if isinstance(data, Callable):
            return lambda x: data(wrapper(x))
        else:
            return wrapper(data)

    return new_vocab, to_vocabulary
