from .vocabulary import Vocabulary, split_string

from collections import Counter

import torch
from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_sequence
from matplotlib import pyplot as plt


def to_tensor(vocabulary: Vocabulary | dict, show_graph=False, tokenizer=split_string):
    if tokenizer is None:
        tokenizer = lambda x: x

    def convert_to_tensor(dataset: list[str]):
        tensor_list = [torch.tensor([vocabulary.get(word, 1) for word in tokenizer(data)])/len(vocabulary) for data in dataset]
        length_list = [len(tensor) for tensor in tensor_list]
        frequency = Counter(length_list)
        if show_graph:
            plt.figure(figsize=(6, 3))
            plt.bar(*zip(*sorted(frequency.items())))
            plt.show()
        padded_tensor_list = pad_sequence(tensor_list, batch_first=True)
        return padded_tensor_list
    return convert_to_tensor


def size_to(to=30):
    def shrink(data):
        if isinstance(data, torch.Tensor):
            data = list(data)  # convert to list
        if data[0].size(0) <= to:
            data[0] = pad(data[0], (0, to))
        data = pad_sequence(data, batch_first=True)
        return [tensor[:to] for tensor in data]
    return shrink


def label_to_tensor(label):
    """
    Convert label to tensor

    POSITIVE(1) => [1, 0, 0]
    NEUTRAL(0) => [0, 1, 0]
    NEGATIVE(-1) => [0, 0, 1]

    :param label: label list
    :return: tensor of labels
    """
    def converter(lb):
        label_list = [0.0, 0.0, 0.0]
        label_list[lb*-1 + 1] = 1.0
        return label_list

    return torch.tensor([converter(lb.toint()) for lb in label])
