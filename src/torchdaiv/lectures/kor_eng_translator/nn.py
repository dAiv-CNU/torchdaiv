import torch
from torch import optim, nn
from torch.nn import *
import torch.nn.functional as F

from .util import vocabulary

from tqdm.notebook import tqdm

cat = torch.cat


EMBEDDING_SIZE = 256

BOS = vocabulary.Token.BOS
EOS = vocabulary.Token.EOS
PAD = vocabulary.Token.PAD

ko_vocab = None
en_vocab = None
input_vocab = None
word_2_idx = None
idx_2_word = None


def set_vocabulary(ko, en):
    global input_vocab, word_2_idx, idx_2_word, ko_vocab, en_vocab
    ko_vocab = ko
    en_vocab = en
    input_vocab = ko.get_stoi()
    word_2_idx = en.get_stoi()
    idx_2_word = en.get_itos()

    print(len(input_vocab), len(word_2_idx), len(idx_2_word))


class BasicModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cpu"

    def forward(self, *args, **kwargs):
        pass

    def to(self, device, *args, **kwargs):
        self.device = device
        return super().to(device, *args, **kwargs)

    def cpu(self):
        self.device = "cpu"
        return super().cpu()

    def cuda(self, device=None):
        if device is not None:
            self.device = device
        else:
            self.device = "cuda"
        return super().cuda()


class Encoder(BasicModule):
    EMBEDDING_SIZE = EMBEDDING_SIZE

    def __init__(
            self,
            model: nn.Module,
            height: int,
            hidden: int,
            dropout: float = 0.2
    ):

        self.input_vocab = globals()['input_vocab']
        super().__init__()
        self.height = height
        self.hidden = hidden
        self.embedding = nn.Embedding(len(self.input_vocab), self.EMBEDDING_SIZE)
        self.model = model(
            self.EMBEDDING_SIZE,
            hidden,
            num_layers=height,
            batch_first=True,
            dropout=dropout
        )

    def forward(self, x):
        x = self.embedding(x)
        x = F.relu(x)
        _, x = self.model(x)
        return x


class Decoder(BasicModule):
    EMBEDDING_SIZE = 256

    def __init__(
            self,
            model: nn.Module,
            height: int,
            hidden: int,
            dropout: float = 0.2
    ):
        self.idx_2_word = globals()['idx_2_word']
        self.word_2_idx = globals()['word_2_idx']

        super().__init__()
        self.height = height
        self.hidden = hidden
        self.embedding = nn.Embedding(len(self.idx_2_word), self.EMBEDDING_SIZE)
        self.model = model(
            self.EMBEDDING_SIZE,
            hidden,
            num_layers=height,
            batch_first = True,
            dropout=dropout
        )

        vocab_size = len(self.idx_2_word)
        self.fc1 = nn.Linear(hidden,vocab_size//2)
        self.fc2 = nn.Linear(vocab_size//2,vocab_size)

    def forward(self, x, cv=None):
        x = self.embedding(x)
        x = F.relu(x)
        output, last_hidden = self.model(x,cv)
        x = F.relu(self.fc1(output))
        x = self.fc2(x)
        return x, last_hidden

    def generate(self, cv):
        device = self.device

        word = BOS
        cnt = 0
        while word not in (EOS, PAD):
            if cnt > 10:
                break
            x = torch.tensor(self.word_2_idx[word]).unsqueeze(0).unsqueeze(0).to(device)
            x, cv = self(x, cv)
            _, x = torch.max(x.view(-1, len(self.idx_2_word)), dim=1)
            word = self.idx_2_word[x.item()]
            if word not in (EOS, PAD):
                print(word, end=" ")
            cnt += 1


def train_loop(model, dataset, epochs, encoder_optimizer, decoder_optimizer, criterion):
    model.train()
    device = model.device

    EOS_IDX = model.EOS_IDX
    BOS_IDX = model.BOS_IDX

    idx_2_word = globals()['idx_2_word']

    def eos(batch_size):
        # 추가하려는 숫자를 텐서로 변환합니다. 이 경우에는 0을 추가합니다.
        # unsqueeze(0)을 사용하여 차원을 맞춥니다. (batch_size, 1) 형태가 됩니다.
        EOS_TK = torch.tensor(EOS_IDX).unsqueeze(0).expand(batch_size, -1)
        return EOS_TK

    def bos(batch_size):
        BOS_TK = torch.tensor(BOS_IDX).unsqueeze(0).expand(batch_size, -1)
        return BOS_TK

    for i, epoch in enumerate(range(epochs)):
        running_loss = 0.0

        for step, batch in enumerate(dataset):
            korean, english = batch

            batch_size = english.size(0)

            encoder_input = korean
            encoder_input = encoder_input.to(device)

            # torch.cat을 사용하여 각 1차원 텐서에 숫자를 추가합니다.
            decoder_input = torch.cat((bos(batch_size), english), dim=1).to(device)
            decoder_label = torch.cat((english, eos(batch_size)), dim=1).to(device)

            x, _ = model(encoder_input, decoder_input)

            decoder_label = F.one_hot(decoder_label, num_classes=len(idx_2_word)).float()
            loss = criterion(x, decoder_label)
            running_loss += loss.item()
            loss.backward()
            encoder_optimizer.step()
            encoder_optimizer.zero_grad()
            decoder_optimizer.step()
            decoder_optimizer.zero_grad()
        running_loss /= len(dataset)
        print(f"epoch:{i+1}/{epochs} loss: {running_loss}")


def translate(model, message, transform):
    model.eval()

    with torch.no_grad():
        message = transform([message])
        last_hidden = model.encoder(message)
        model.decoder.generate(last_hidden)


class Module(BasicModule):
    def __init__(self):
        super(Module, self).__init__()

        self.EOS_IDX = globals()['en_vocab'][EOS]
        self.BOS_IDX = globals()['en_vocab'][BOS]
        self.PAD_IDX = globals()['ko_vocab'][PAD]

    def init_optimizer(self, lr=0.0001):
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr)

        self.criterion = nn.CrossEntropyLoss()  #ignore_index=PAD_IDX

    def fit(self, dataset, epochs):
        return train_loop(self, dataset, epochs, self.encoder_optimizer, self.decoder_optimizer, self.criterion)

    def translate(self, message, transform):
        return translate(self, message, transform)
