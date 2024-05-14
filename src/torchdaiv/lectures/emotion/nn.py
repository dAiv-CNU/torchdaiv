import torch
from torch import optim, nn
from torch.nn import *

from ...datasets import EmotionDataset

from tqdm.notebook import tqdm


class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.device = "cpu"
        self.train_dataloader = ()
        self.test_dataloader = ()
        self.optimizer = None
        self.criterion = None

    def forward(self, x):
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

    def set_dataloader(self, train_dataloader, test_dataloader):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

    def init_optimizer(self, lr=0.0001):
        self.optimizer = optim.AdamW(self.parameters(), lr=lr)
        self.criterion = CrossEntropyLoss()

    def train(self, epochs, optimizer_init=False, lr=0.0001):
        """
        Train the model

        :param epochs: epochs to train
        :param optimizer_init: if True, initialize optimizer
        :param lr: Set learning rate, only used when optimizer_init is True
        """

        if optimizer_init or self.optimizer is None:
            self.init_optimizer(lr)

        super().train()
        datalen = len(self.train_dataloader)

        for epoch in range(epochs):
            running_acc, running_loss = 0.0, 0.0

            for i, (inputs, labels) in enumerate(tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")):
                self.optimizer.zero_grad()

                inputs: torch.Tensor = inputs.to(self.device)
                labels: torch.Tensor = labels.to(self.device)

                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)

                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                running_acc += (predicted == torch.max(labels, 1)[1]).float().mean().item()

                inputs, labels = inputs.detach(), labels.detach()
                inputs, labels = inputs.cpu(), labels.cpu()

                print(f"\rEpoch [{epoch+1}/{epochs}], Step: [{i+1}/{datalen}], Accuracy: {running_acc/(i+1):.6%}, Loss: {running_loss/(i+1):.8f}", end="")

            print(f"\rEpoch [{epoch+1}/{epochs}], Step: [{datalen}/{datalen}], Accuracy: {running_acc/datalen:.6%}, Loss: {running_loss/datalen:.8f}")

    def evaluate(self):
        self.eval()
        datalen = len(self.test_dataloader)
        running_acc, running_loss = 0.0, 0.0

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(tqdm(self.test_dataloader, desc=f"Performance Test")):
                inputs: torch.Tensor = inputs.to(self.device)
                labels: torch.Tensor = labels.to(self.device)

                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)

                running_loss += self.criterion(outputs, labels).item()
                running_acc += (predicted == torch.max(labels, 1)[1]).float().mean().item()

                inputs, labels = inputs.cpu(), labels.cpu()

                print(f"\rAccuracy: {running_acc/(i+1):.6%}, Loss: {running_loss/(i+1):.8f}", end="")

        print(f"\rAccuracy: {running_acc/datalen:.6%}, Loss: {running_loss/datalen:.8f}")

    def pipeline(self, message: str, transform: list | tuple) -> EmotionDataset.Emotion:
        if not isinstance(transform, list) and not isinstance(transform, tuple):
            transform = [transform]

        message = [message]
        for t in transform:
            message = t(message)
        message = message[0]

        if not isinstance(message, torch.Tensor):
            raise TypeError("Transform function must return torch.Tensor evantually.")

        self.eval()
        with torch.no_grad():
            message: torch.Tensor = message.unsqueeze(0).to(self.device)

            outputs = self(message)
            _, predicted = torch.max(outputs.data, 1)

            message = message.cpu()

            return EmotionDataset.Emotion(predicted.item()*-1+1)
