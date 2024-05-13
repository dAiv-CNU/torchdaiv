import torch
from torch import optim, nn
from torch.nn import *

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
        if optimizer_init or self.optimizer is None:
            self.init_optimizer(lr)

        super().train()
        datalen = len(self.train_dataloader)

        for epoch in tqdm(range(epochs), desc="Running epochs"):
            running_loss = 0.0

            for i, (inputs, labels) in enumerate(tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")):
                self.optimizer.zero_grad()

                outputs = self(inputs.to(self.device))

                loss = self.criterion(outputs, labels.to(self.device))

                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

                print(f"\rEpoch [{epoch+1}/{epochs}], Step: [{i+1}/{datalen}], Loss: {running_loss/(i+1):.4f}", end="")

            print(f"\rEpoch [{epoch+1}/{epochs}], Step: [{datalen}/{datalen}], Loss: {running_loss/datalen:.4f}")

    def evaluate(self):
        self.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.test_dataloader:
                outputs = self(inputs.to(self.device))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(self.device)).sum().item()

        print(f"Accuracy: {100 * correct / total}")
