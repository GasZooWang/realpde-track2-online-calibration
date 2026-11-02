"""Neural interval head."""
from __future__ import annotations

import torch
from torch import nn

FEATURE_NAMES = (
    'center', 'log_half', 'prediction_disagreement', 'input_mean_disagreement',
    'input_last_disagreement', 'input_aligned_disagreement', 'input_std',
    'input_trend', 'lag_signed', 'lag_absolute', 'ema_signed', 'ema_absolute',
    'prediction_std', 'horizon', 'y', 'x', 'channel', 'has_previous',
    'input_zero_fraction', 'half_to_sigma',
)
SIGMA = 0.0563870259


class IntervalHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(len(FEATURE_NAMES), 32), nn.SiLU(),
                                     nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 2))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features, damping=1.0):
        raw = self.network(features)
        shift = damping * torch.tanh(raw[..., 0])
        scale = torch.exp(damping * 0.69314718056 * torch.tanh(raw[..., 1]))
        return shift, scale


class Context:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.reset()

    def reset(self):
        self.previous_center = None
        self.lag = self.lag_abs = self.ema = self.ema_abs = None
        self.initialized = None

    def observe(self, matured_raw):
        if matured_raw is None:
            return
        if self.previous_center is None:
            raise ValueError('matured target without cached previous bounds')
        valid = matured_raw != 0
        residual = matured_raw - self.previous_center
        count = valid.sum(dim=0).clamp_min(1)
        signed = (residual * valid).sum(dim=0) / count
        absolute = (residual.abs() * valid).sum(dim=0) / count
        seen = valid.any(dim=0)
        if self.ema is None:
            self.ema = torch.zeros_like(signed)
            self.ema_abs = torch.zeros_like(absolute)
            self.initialized = torch.zeros_like(seen)
        self.lag = torch.where(seen, signed, self.ema)
        self.lag_abs = torch.where(seen, absolute, self.ema_abs)
        new_ema = torch.where(self.initialized, .8 * self.ema + .2 * signed, signed)
        new_abs = torch.where(self.initialized, .8 * self.ema_abs + .2 * absolute, absolute)
        self.ema = torch.where(seen, new_ema, self.ema)
        self.ema_abs = torch.where(seen, new_abs, self.ema_abs)
        self.initialized |= seen

    def features(self, input_raw, prediction_raw, center_raw, half_raw):
        shape = center_raw.shape
        if tuple(shape) != (20, 32, 64, 2):
            raise ValueError(f'unexpected UV window shape: {shape}')
        half = half_raw.clamp_min(1e-5)
        scale_c = center_raw.new_tensor([.1, .02])
        mean = input_raw.mean(dim=0, keepdim=True)
        std = input_raw.std(dim=0, correction=0, keepdim=True)
        zero_frac = (input_raw == 0).float().mean(dim=0, keepdim=True)
        trend = input_raw[-5:].mean(dim=0) - input_raw[:5].mean(dim=0)
        zeros = torch.zeros_like(mean[0])
        lag = self.lag if self.lag is not None else zeros
        lag_abs = self.lag_abs if self.lag_abs is not None else zeros
        ema = self.ema if self.ema is not None else zeros
        ema_abs = self.ema_abs if self.ema_abs is not None else zeros
        def coord(n, dim):
            sizes = [1, 1, 1, 1]
            sizes[dim] = n
            return torch.linspace(-1, 1, n, device=center_raw.device).reshape(sizes)
        values = [center_raw / scale_c, torch.log(half / scale_c),
                  (prediction_raw-center_raw)/half, (mean-center_raw)/half,
                  (input_raw[-1:]-center_raw)/half, (input_raw-center_raw)/half,
                  std/half, trend/half, lag/half, lag_abs/half,
                  ema/half, ema_abs/half,
                  prediction_raw.std(dim=0, correction=0, keepdim=True)/half,
                  coord(20, 0), coord(32, 1), coord(64, 2), coord(2, 3),
                  torch.ones_like(zeros) * float(self.previous_center is not None),
                  zero_frac, half / SIGMA]
        features = torch.stack([v.expand(shape).clamp(-8, 8) for v in values], dim=-1)
        self.previous_center = center_raw.detach().clone()
        return features
