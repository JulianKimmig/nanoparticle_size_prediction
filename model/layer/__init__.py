import torch
from torch import nn


class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return torch.flatten(x)


