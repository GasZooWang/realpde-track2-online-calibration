"""Temporal interval head."""

from .temporal_interval import TemporalIntervalHead, CHANNELS
from torch import nn
import torch


class CanonicalTemporalHead(TemporalIntervalHead):

    def __init__(self, width=48):
        super().__init__(width)
        self.stem = nn.Conv2d(CHANNELS + 160, width, 1)

    def forward(self, x, global_context):
        n, _, height, width = x.shape
        inp = x[:, :40].reshape(n, 20, 2, height, width)
        center = x[:, 40:80].reshape_as(inp)
        half = x[:, 80:120].reshape_as(inp).exp().clamp_min(0.0001)
        residual = x[:, 120:160].reshape_as(inp)
        mean = inp.mean(1, keepdim=True)
        std = inp.std(1, correction=0, keepdim=True)
        derived = torch.cat(
            [
                (inp - mean) / half,
                (center - mean) / half,
                residual / half,
                std.expand_as(half) / half,
            ],
            1,
        )
        extra = derived.reshape(n, 160, height, width).clamp(-8, 8)
        return super().forward(torch.cat([x, extra], 1), global_context)
