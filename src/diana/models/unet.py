import torch
from torch import nn


class SinusoidalEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

    def forward(time: torch.Tensor):
        pass

    def computeembeddings(t: torch.Tensor) -> torch.Tensor: ...
