from tqdm.auto import tqdm
from torch import nn, optim
import torch


class Trainer:
    def __init__(self, model, batched_dataset, lr=1e-3, optimizer=optim.Adam, criterion=nn.MSELoss(), device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.model = model
        self.optimizer = optimizer(model.parameters(), lr=lr)
        self.criterion = criterion
        self.batched_dataset = batched_dataset
        self.device = device
        self.model.to(self.device)

    def __call__(self, *args, **kwargs):
        return self.train(*args, **kwargs)

    def train(self, epochs):
        self.model.train()
        for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
            self.optimizer.zero_grad()

            inputs = self.batched_dataset.to(self.device)

            outputs, latents = self.model(inputs, show=False)
            loss = self.criterion(outputs, inputs)

            loss.backward()
            self.optimizer.step()

            print(f"\rEpoch [{epoch+1:4}/{epochs:4}], Loss: {loss.item():.6f}", end=("" if (epoch+1) % 10 else "\n"))

        self.model.eval()

    def evaluate(self, show=True):
        self.model.eval()
        with torch.no_grad():
            inputs = self.batched_dataset.to(self.device)
            outputs, latents = self.model(inputs, show=show)
            return outputs, latents
