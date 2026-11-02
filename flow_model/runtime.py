"""Mixed-precision affine kernels."""

import torch
from torch import nn


class HalfAffine(nn.Module):

    def __init__(self, layer):
        super().__init__()
        self.layer = layer.half()

    def forward(self, x):
        return self.layer(x.half()).float()


def half_affine_(module):
    for parent in list(module.modules()):
        for name, layer in list(parent.named_children()):
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                setattr(parent, name, HalfAffine(layer))
