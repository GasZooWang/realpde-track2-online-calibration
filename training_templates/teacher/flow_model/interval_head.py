"""Interval head."""

from torch import nn
import torch


class IntervalHead(nn.Module):

    def __init__(self, width=32, feature_count=28):
        super().__init__()
        self.width = width
        self.feature_count = feature_count
        self.network = nn.Sequential(
            nn.Linear(feature_count, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 2),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, x, damping=1.0):
        raw = self.network(x[..., : self.feature_count])
        return (
            damping * torch.tanh(raw[..., 0]),
            torch.exp(damping * 0.69314718056 * torch.tanh(raw[..., 1])),
        )
