import torch
from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_sequence

from .vocabulary import Token


def to_tensor(vocabulary, tokenizer=lambda x: x):
    PAD_IDX = vocabulary[Token.PAD]
    UNK_IDX = vocabulary[Token.UNK]

    print(f"Using Special Tokens - PAD_IDX: {PAD_IDX}, UNK_IDX: {UNK_IDX}")

    get = vocabulary.get_stoi().get

    def convert_to_tensor(dataset: list[str]):
        tensor_list = [
            torch.tensor([
                get(word, UNK_IDX) for word in tokenizer(data)
            ], dtype=torch.long)
            for data in dataset
        ]
        padded_tensor_list = pad_sequence(tensor_list, batch_first=True, padding_value=PAD_IDX)
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
