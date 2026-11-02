"""Interval and moment heads."""

from .interval_context import IntervalContext
from .interval_head import IntervalHead
from .moments import MomentContext, MomentHead
import torch
from .constants import constant_like, physical


class AccuracyTTTModel:

    def __init__(
        self,
        safe,
        interval_state,
        moment_state,
        width=32,
        response_range=1.5,
        feature_count=28,
    ):
        self.safe = safe
        self.device = safe.device
        self.interval = IntervalHead(width, feature_count).to(self.device).eval()
        self.interval.load_state_dict(interval_state, strict=True)
        self.moments = MomentHead(response_range).to(self.device).eval()
        self.moments.load_state_dict(moment_state, strict=True)
        for head in (self.interval, self.moments):
            for p in head.parameters():
                p.requires_grad_(False)
        assert feature_count == 28, "expected 28 interval features"
        self.interval_context = IntervalContext(self.device)
        self.moment_context = MomentContext()

    def reset_ttt_state(self):
        self.safe.reset_ttt_state()
        self.interval_context.reset()
        self.moment_context.reset()

    @torch.inference_mode()
    def ttt_step(self, input_norm, prev_target_norm=None):
        prev_raw = None
        if prev_target_norm is not None:
            prev = torch.as_tensor(
                prev_target_norm, device=self.device, dtype=torch.float32
            )
            prev_raw = physical(prev, self.safe._target_mean, self.safe._target_std, self.safe._target_zero)[
                0, ..., :2
            ]
        self.interval_context.observe(prev_raw)
        self.moment_context.observe(prev_raw)
        pred, info = self.safe.ttt_step(input_norm, prev_target_norm)
        inp = torch.as_tensor(input_norm, device=self.device, dtype=torch.float32)
        inp_raw = physical(inp, self.safe._input_mean, self.safe._input_std, self.safe._input_zero)[0, ..., :2]
        std = self.safe._target_std[0, ..., :2]
        mean = self.safe._target_mean[0, ..., :2]
        p = pred[0, ..., :2] * std + mean
        lower = info["lower"][0, ..., :2] * std + mean
        upper = info["upper"][0, ..., :2] * std + mean
        center = 0.5 * (lower + upper)
        half = 0.5 * (upper - lower)
        ix = self.interval_context.features(inp_raw, p, center, half)
        mx = self.moment_context.features(inp_raw, p)
        shift, scale = self.interval(ix)
        mu, lv = self.moments(mx)
        pm = p.mean(0)
        corrected = pm + mu * constant_like(p, [0.1, 0.02]) + (p - pm) * torch.exp(0.5 * lv)
        result = pred.clone()
        result[0, ..., :2] = (corrected - mean) / std
        lo = info["lower"].clone()
        hi = info["upper"].clone()
        lo[0, ..., :2] = (center + half * shift - half * scale - mean) / std
        hi[0, ..., :2] = (center + half * shift + half * scale - mean) / std
        out = dict(info)
        out["lower"] = lo
        out["upper"] = hi
        return (result, out)
