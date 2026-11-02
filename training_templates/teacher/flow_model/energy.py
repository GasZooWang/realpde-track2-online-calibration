"""Energy heads."""

from .temporal_interval import ResidualBlock
from torch import nn
from torch.nn import functional as F
import torch
from .constants import constant_like


def canonical(x):
    n, _, h, w = x.shape
    inp = x[:, :40].reshape(n, 20, 2, h, w)
    center = x[:, 40:80].reshape_as(inp)
    half = x[:, 80:120].reshape_as(inp).exp().clamp_min(0.0001)
    residual = x[:, 120:160].reshape_as(inp)
    mean = inp.mean(1, keepdim=True)
    std = inp.std(1, correction=0, keepdim=True)
    extra = torch.cat(
        [
            (inp - mean) / half,
            (center - mean) / half,
            residual / half,
            std.expand_as(half) / half,
        ],
        1,
    )
    return torch.cat([x, extra.reshape(n, 160, h, w).clamp(-8, 8)], 1)


class EnergyHead(nn.Module):

    def __init__(self, width=64):
        super().__init__()
        self.width = width
        self.stem = nn.Conv2d(333, width, 1)
        self.global_encoder = nn.Sequential(
            nn.Linear(256, 32), nn.SiLU(), nn.Linear(32, 2 * width)
        )
        self.fine = nn.Sequential(ResidualBlock(width, 1), ResidualBlock(width, 2))
        self.coarse = nn.Sequential(ResidualBlock(width, 1), ResidualBlock(width, 2))
        self.out = nn.Conv2d(width, 4, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, g, p):
        return self.forward_prepared(canonical(x), x, g, p)

    def forward_prepared(self, features, x, g, p):
        n, _, h, w = p.shape
        scale = constant_like(p, [0.1, 0.02])[None, None, :, None, None]
        pp = p.reshape(n, 20, 2, h, w) / scale
        pm = pp.mean(1)
        ps = pp.std(1, correction=0)
        ins = x[:, :40].reshape_as(pp).std(1, correction=0)
        z = torch.cat([features, pm, ps, (ins / ps.clamp_min(0.0001)).clamp(0, 8)], 1)
        h = self.stem(z)
        gamma, beta = self.global_encoder(g).chunk(2, -1)
        h = F.silu(
            h * (1 + 0.1 * torch.tanh(gamma[:, :, None, None]))
            + 0.1 * beta[:, :, None, None]
        )
        z = self.fine(h) + F.interpolate(
            self.coarse(F.avg_pool2d(h, 4)),
            size=h.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        mu, lv = self.out(z).chunk(2, 1)
        return (0.1 * torch.tanh(mu), 1.5 * torch.tanh(lv))


class DirectEnergyHead(nn.Module):

    def __init__(self, width=32):
        super().__init__()
        self.stem = nn.Conv2d(78, width, 1)
        self.fine = nn.Sequential(ResidualBlock(width, 1), ResidualBlock(width, 2))
        self.coarse = nn.Sequential(ResidualBlock(width, 1), ResidualBlock(width, 2))
        self.out = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, f):
        z = F.silu(self.stem(f))
        h = self.fine(z) + F.interpolate(
            self.coarse(F.avg_pool2d(z, 4)),
            size=z.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return 4 * torch.tanh(self.out(h))


def energy_features(x, history, pm, pv):
    n, _, h, w = x.shape
    u = constant_like(x, [0.1, 0.02])[None, :, None, None]
    inp = x[:, :40].reshape(n, 20, 2, h, w)
    im = inp.mean(1)
    iv = inp.var(1, correction=0)
    pk = pv.sum(1, keepdim=True)
    ik = (iv * u.square()).sum(1, keepdim=True)

    def pool(z, k):
        return F.avg_pool2d(z, k, stride=1, padding=k // 2, count_include_pad=False)

    ref = 0.5 * (pk + pool(ik, 5)) + 1e-09
    pieces = [
        history,
        pm / u,
        pv.clamp_min(0).sqrt() / u,
        im,
        iv.clamp_min(0).sqrt(),
        inp[:, -5:].mean(1) - inp[:, :5].mean(1),
        ((ik + 1e-10) / (pk + 1e-10)).log(),
        (pk / 0.0001 + 1e-08).log(),
        x[:, 162:166],
    ]
    for k in (3, 7, 15):
        pieces.extend(
            [
                pool(iv, k).clamp_min(0).sqrt(),
                ((pool(pk, k) + 1e-10) / (pk + 1e-10)).log(),
                ((pool(ik, k) + 1e-10) / (pk + 1e-10)).log(),
            ]
        )
    pieces.append(im.mean((2, 3), keepdim=True).expand(-1, -1, h, w))
    pieces.append((inp.reshape(n, 4, 5, 2, h, w).mean(2) - im[:, None]).flatten(1, 2))
    f = torch.cat(pieces, 1).clamp(-12, 12)
    assert f.shape[1] == 78, f.shape
    return (f, ref)


def energy_logscale(delta, pm, pv, reference, strength=1.0):
    pk = pv.sum(1, keepdim=True)
    newk = (pk + strength * delta * reference).clamp_min(1e-12)
    lv = (newk / pk.clamp_min(1e-12)).log().clamp(-3, 3)
    return lv.expand_as(pm)
