"""Multirate history features."""

import torch
from .constants import constant_like


class MomentHistory:
    rates = (0.1, 0.3, 0.7)

    def __init__(self):
        self.reset()

    def reset(self):
        self.ema = None
        self.errors = None
        self.lags = []
        self.total = None
        self.count = 0
        self.pm = self.pv = None

    def features(self, inp, pm, pv, previous_target=None):
        units = constant_like(pm, [0.1, 0.02])[:, None, None]
        im = inp.mean(0)
        iv = inp.var(0, correction=0)
        moment = torch.stack([im, iv])
        self.count += 1
        self.total = (
            moment.clone()
            if self.total is None
            else self.total + (moment - self.total) / self.count
        )
        if self.ema is None:
            self.ema = [moment.clone() for _ in self.rates]
        else:
            self.ema = [
                old + rate * (moment - old) for rate, old in zip(self.rates, self.ema)
            ]
        if self.errors is None:
            self.errors = [torch.zeros_like(moment) for _ in self.rates]
        if previous_target is not None and self.pm is not None:
            tm = previous_target.mean(0)
            tv = previous_target.var(0, correction=0)
            obs = torch.stack(
                [
                    ((tm - self.pm) / units).clamp(-2, 2),
                    (tv.clamp_min(1e-10) / self.pv.clamp_min(1e-10)).log().clamp(-3, 3),
                ]
            )
            self.errors = [
                old + rate * (obs - old) for rate, old in zip(self.rates, self.errors)
            ]

        def relative(s):
            return torch.cat(
                [
                    ((s[0] - pm) / units).clamp(-2, 2),
                    (s[1].clamp_min(1e-10) / pv.clamp_min(1e-10)).log().clamp(-3, 3),
                ],
                0,
            )

        out = []
        for e, r in zip(self.ema, self.errors):
            out.extend([relative(e), r.flatten(0, 1)])
        for lag in range(3):
            out.append(
                relative(self.lags[-1 - lag] if len(self.lags) > lag else moment)
            )
        out.append(relative(self.total))
        self.lags = (self.lags + [moment.clone()])[-3:]
        self.pm = pm.detach().clone()
        self.pv = pv.detach().clone()
        value = torch.cat(out, 0)
        assert value.shape == (40, 32, 64)
        return value
