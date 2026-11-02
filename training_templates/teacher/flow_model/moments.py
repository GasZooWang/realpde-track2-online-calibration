"""Moment head."""

from torch import nn
import torch
from .constants import constant_like
import torch.nn.functional as F

FEATURE_COUNT = 22


class MomentHead(nn.Module):

    def __init__(self, response_range=1.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_COUNT, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.response_range = float(response_range)

    def forward(self, x):
        z = self.net(x)
        return (
            0.15 * torch.tanh(z[..., 0]),
            self.response_range * torch.tanh(z[..., 1] * (0.5 / self.response_range)),
        )


def smooth(x):
    v = F.pad(x.permute(2, 0, 1)[None], (1, 1, 1, 1), mode="replicate")
    return F.avg_pool2d(v, 3, 1)[0].permute(1, 2, 0)


class MomentContext:

    def __init__(self):
        self.reset()

    def reset(self):
        self.previous_prediction = None
        self.lag_ratio = self.lag_mean = self.ratio_state = self.mean_state = None

    def observe(self, matured_raw):
        if matured_raw is None:
            return
        if self.previous_prediction is None:
            raise ValueError("matured target without previous safe prediction")
        previous = self.previous_prediction
        channel_scale = constant_like(previous, [0.1, 0.02])
        self.lag_ratio = (
            torch.log(matured_raw.var(0, correction=0) + 1e-09)
            - torch.log(previous.var(0, correction=0) + 1e-09)
        ).clamp(-2, 2)
        self.lag_mean = (matured_raw - previous).mean(0) / channel_scale
        self.ratio_state = (
            self.lag_ratio
            if self.ratio_state is None
            else 0.8 * self.ratio_state + 0.2 * self.lag_ratio
        )
        self.mean_state = (
            self.lag_mean
            if self.mean_state is None
            else 0.8 * self.mean_state + 0.2 * self.lag_mean
        )

    def features(self, inp, pred):
        channel_scale = constant_like(pred, [0.1, 0.02])
        pm, pv = (pred.mean(0), pred.var(0, correction=0))
        im, iv = (inp.mean(0), inp.var(0, correction=0))
        ps = (pv + 1e-09).sqrt()
        zeros = torch.zeros_like(pm)
        xv = torch.linspace(-1, 1, 64, device=pred.device)[None, :, None].expand_as(pm)
        yv = torch.linspace(-1, 1, 32, device=pred.device)[:, None, None].expand_as(pm)
        cv = constant_like(pred, [-1.0, 1.0])[None, None].expand_as(pm)
        values = [
            pm / channel_scale,
            ps / channel_scale,
            (im - pm) / channel_scale,
            (inp[-1] - pm) / channel_scale,
            (inp[-5:].mean(0) - inp[:5].mean(0)) / channel_scale,
            torch.log((iv + 1e-09) / (pv + 1e-09)),
            torch.log((inp[:10].var(0, correction=0) + 1e-09) / (pv + 1e-09)),
            torch.log((inp[-10:].var(0, correction=0) + 1e-09) / (pv + 1e-09)),
            (inp[1:] - inp[:-1]).square().mean(0).sqrt() / ps,
            torch.log((smooth(iv) + 1e-09) / (pv + 1e-09)),
            (smooth(im) - im) / channel_scale,
            zeros if self.lag_ratio is None else self.lag_ratio,
            zeros if self.ratio_state is None else self.ratio_state,
            zeros if self.lag_mean is None else self.lag_mean,
            zeros if self.mean_state is None else self.mean_state,
            (inp == 0).float().mean(0),
            smooth((inp == 0).float().mean(0)),
            pv.flip(-1) / (pv.sum(-1, keepdim=True) + 1e-09),
            xv,
            yv,
            cv,
            torch.ones_like(pm) * float(self.previous_prediction is not None),
        ]
        self.previous_prediction = pred.detach().clone()
        return torch.stack(values, -1).clamp(-8, 8)
