import torch


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

    def save_weights(self, path):
        torch.save(self.state_dict(), path)