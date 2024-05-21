from __future__ import annotations

from torch.utils.data import Dataset
from gensim.models import Word2Vec

from typing import Callable
from collections import Counter

from json import loads
import requests
import re
import os


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


def vocabulary_creator(minimum_frequency=5, tokenizer=split_string) -> tuple[Vocabulary, Callable]:
    new_vocab = Vocabulary()

    if tokenizer is None:
        tokenizer = lambda x: x

    def convert(data: list[str]) -> list[str]:
        data = [word for line in data for word in tokenizer(line)]
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


class KoNLPyTokenizer:
    morphs = lambda data: None
    nouns = lambda data: None
    phrases = lambda data: None

    def __init__(self, java_path=None):
        if java_path is not None:
            if java_path:
                os.environ['JAVA_HOME'] = java_path
            elif 'JAVA_HOME' not in os.environ:
                os.environ['JAVA_HOME'] = input("JAVA_HOME is not specified. Please enter your Java path: ")

        from konlpy.tag import Okt
        self.tokenizer = Okt()

        self.morphs = lambda data: [self.tokenizer.morphs(line) for line in data]
        self.nouns = lambda data: [self.tokenizer.nouns(line) for line in data]
        self.phrases = lambda data: [self.tokenizer.phrases(line) for line in data]

    @classmethod
    def from_pretrained(cls, dataset, *args, transform=None, **kwargs) -> Dataset:
        if transform is cls.morphs:
            transform = lambda data: cls._dl_pretrained(dataset.Pretrained(*args, **kwargs)['morphs'])
        elif transform is cls.nouns:
            transform = lambda data: cls._dl_pretrained(dataset.Pretrained(*args, **kwargs)['nouns'])
        elif transform is cls.phrases:
            transform = lambda data: cls._dl_pretrained(dataset.Pretrained(*args, **kwargs)['phrases'])

        return dataset(*args, transform=transform, **kwargs)

    @staticmethod
    def _dl_pretrained(path):
        if isinstance(path, str):
            return requests.get(path).json()
        else:
            return loads("".join([requests.get(pth).text for pth in path]))


class WordVector(Word2Vec):
    PAD = Vocabulary.PAD
    UNK = Vocabulary.UNK

    def __init__(self, dataset_list, *args, vector_size=100, window=5, min_count=5, workers=4, sg=0, **kwargs):
        """
        :param dataset_list: 학습 시킬 데이터 셋
        :param vector_size: 특징 값 혹은 임베딩 된 벡터의 차원
        :param window: 주변 단어 수
        :param min_count: 단어 최소 빈도 수 / 해당 숫자보다 적은 빈도의 단어들은 사용하지 않음
        :param workers: 학습에 사용되는 프로세스 수
        :param sg: Word2Vec 생성 방식 / 0 : CBOW, 1: Skip-gram
        """
        sentences = []
        for dataset in dataset_list:
            sentences += dataset.data

        sentences = self.cut_off(sentences, min_count)

        super().__init__(
            *args, sentences=sentences, vector_size=vector_size,
            window=window, min_count=1, workers=workers, sg=sg, **kwargs
        )

    @classmethod
    def cut_off(cls, data: list[str], min_count=5):
        """
        :param data: 단어로 나눌 문장 리스트
        :param min_count: 단어 최소 빈도 수 / 해당 숫자보다 적은 빈도의 단어들은 사용하지 않음
        """
        sentences = [word for line in data for word in data]
        frequency = Counter(data)
        cut_off_list = list(filter(lambda x: frequency[x] < min_count, frequency))

        # 빈도수가 min_count보다 작은 단어들은 UNK로 대체
        UNK = cls.UNK
        sentences = [[UNK if word in cut_off_list else word for word in sentence] for sentence in sentences]

        # 각 문장의 최대 길이를 계산
        max_len = max(len(sentence) for sentence in sentences)

        # 문장의 길이를 max_len으로 맞추고 PAD로 채움
        PAD = cls.PAD
        return [sentence + [PAD] * (max_len - len(sentence)) for sentence in sentences]
