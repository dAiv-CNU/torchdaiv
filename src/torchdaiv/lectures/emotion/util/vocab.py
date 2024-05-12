from typing import Callable


class Vocabulary(dict):
    PAD = "<PAD>"
    UNK = "<UNK>"

    def __init__(self):
        super().__init__()
        self.vocab = {0: "<PAD>", 1: "<UNK>"}

    def update(self, data: list | dict):
        if isinstance(data, list):
            for d in data:
                self.update(d)
        elif isinstance(data, dict):
            for k, v in data.items():
                self.update(v)
        else:
            if data not in self.vocab.values():
                self.vocab[len(self.vocab)] = data
        return data

    def get(self, key, default=1):
        return self.vocab[key] if key in self else default


def vocabulary_creator() -> tuple[Vocabulary, Callable]:
    new_vocab = Vocabulary()

    def to_vocabulary(data: list):
        new_vocab.update(data)
        return data

    return new_vocab, to_vocabulary
