"""Distilled student network."""

import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, c, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=dilation, dilation=dilation),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class Student(nn.Module):
    def __init__(self, width=48):
        super().__init__()
        w = width
        self.stem = nn.Conv2d(40, w, 3, padding=1)
        self.b0 = Block(w)
        self.d1 = nn.Conv2d(w, 2 * w, 3, stride=2, padding=1)
        self.b1 = Block(2 * w)
        self.d2 = nn.Conv2d(2 * w, 3 * w, 3, stride=2, padding=1)
        self.b2 = nn.Sequential(Block(3 * w), Block(3 * w, 2))
        self.u1 = nn.Conv2d(3 * w, 2 * w, 1)
        self.r1 = Block(2 * w)
        self.u0 = nn.Conv2d(2 * w, w, 1)
        self.r0 = Block(w)
        self.out = nn.Conv2d(w, 40, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        a = self.b0(F.silu(self.stem(x)))
        b = self.b1(F.silu(self.d1(a)))
        c = self.b2(F.silu(self.d2(b)))
        b = self.r1(
            b
            + F.interpolate(
                self.u1(c), size=b.shape[-2:], mode="bilinear", align_corners=False
            )
        )
        a = self.r0(
            a
            + F.interpolate(
                self.u0(b), size=a.shape[-2:], mode="bilinear", align_corners=False
            )
        )
        return x + self.out(a)


class StudentBackbone(nn.Module):
    def __init__(self, state, device):
        super().__init__()
        self.net = Student(state["width"]).to(device).eval().requires_grad_(False)
        self.net.load_state_dict(state["state_dict"])
        self.half_mode = torch.device(device).type == "cuda"
        if self.half_mode:
            self.net.half().to(memory_format=torch.channels_last)

    def forward(self, x):
        a = x[..., :2].permute(0, 1, 4, 2, 3).flatten(1, 2)
        if self.half_mode:
            a = a.half().contiguous(memory_format=torch.channels_last)
        q = (
            self.net(a)
            .float()
            .reshape(x.shape[0], 20, 2, 32, 64)
            .permute(0, 1, 3, 4, 2)
        )
        return torch.cat((q, torch.zeros_like(q[..., :1])), -1)
