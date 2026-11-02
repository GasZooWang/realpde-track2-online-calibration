"""Component energy model."""
from pathlib import Path
import io
import sys
import zipfile

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'dev/algorithm'), str(ROOT / 'dev/shared'), str(ROOT / 'submission')]
from train_wave import Wave, apply
from train_refiner import features


def load_native(device='cuda'):
    state = torch.load(ROOT / 'submission_native/algorithm.pt', map_location=device, weights_only=True)
    net = Wave(196, state['width'], 'joint').to(device).eval().requires_grad_(False)
    net.load_state_dict(state['state_dict'])
    with torch.no_grad():
        net.net.out.weight.mul_(state['strength'])
        net.net.out.bias.mul_(state['strength'])
    if str(device).startswith('cuda'):
        net.half().to(memory_format=torch.channels_last)
    return net


def latent(model, x):
    n = model.net
    a = n.b0(F.silu(n.stem(x.to(n.stem.weight.dtype))))
    b = n.b1(F.silu(n.d1(a)))
    c = n.b2(F.silu(n.d2(b)))
    b = n.r1(b + F.interpolate(n.u1(c), size=b.shape[-2:], mode='bilinear', align_corners=False))
    a = n.r0(a + F.interpolate(n.u0(b), size=a.shape[-2:], mode='bilinear', align_corners=False))
    return a


def redistribute(prediction, d):
    p = prediction.float()
    wave = p - p.mean(1, keepdim=True)
    v = wave.square().mean(1)
    total = v.sum(1, keepdim=True)
    fraction = torch.where(total > 0, v / total.clamp_min(1e-30), torch.zeros_like(v))
    signed = torch.cat((d.float(), -d.float()), 1)
    ratio_delta = (fraction * signed.expm1()).sum(1, keepdim=True)
    log_normalizer = -0.5 * torch.log1p(ratio_delta.clamp_min(-0.999999))
    log_factor = 0.5 * signed + log_normalizer
    return p + wave * log_factor.expm1()[:, None]


class SplitEnergy(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(32, 1, 3, padding=1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return 0.5 * torch.tanh(self.conv(x).float())


@torch.inference_mode()
def prepare(path, net):
    d = torch.load(path, map_location=next(net.parameters()).device, weights_only=False)
    f, s = features(d)
    zs = [latent(net, b) for b in f.half().split(16)]
    z = torch.cat(zs)
    raw = torch.cat([net.net.out(a).float() for a in z.split(16)])
    p, lo, hi = apply(raw, d['prediction'], d['lower'], d['upper'], s, 'joint')
    return dict(latent=z, prediction=p, lower=lo, upper=hi, target=d['target'], input=d['input'])
