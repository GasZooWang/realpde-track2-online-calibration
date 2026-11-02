"""Multiscale interval and energy refinement."""

from .canonical_interval import CanonicalTemporalHead
from .energy import EnergyHead, canonical
from .multiscale_interval import MultiscaleIntervalHead
from .temporal_interval import IntervalContext, image, TemporalIntervalHead
import torch
from .constants import constant_like, physical


class SPSTTTModel:

    def __init__(
        self,
        accuracy,
        interval_state,
        width=96,
        energy_state=None,
        energy_width=64,
        energy_strength=1.0,
    ):
        self.accuracy = accuracy
        self.device = accuracy.device
        self.context = IntervalContext()
        self.energy_strength = energy_strength
        cls = (
            MultiscaleIntervalHead
            if any((k.startswith("pyramid.") for k in interval_state))
            else CanonicalTemporalHead
        )
        self.interval = cls(width).to(self.device).eval()
        self.interval.load_state_dict(interval_state, strict=True)
        self.interval.requires_grad_(False)
        self.energy = None
        if energy_state is not None:
            self.energy = EnergyHead(energy_width).to(self.device).eval()
            self.energy.load_state_dict(energy_state, strict=True)
            self.energy.requires_grad_(False)

    def reset_ttt_state(self):
        self.accuracy.reset_ttt_state()
        self.context.reset()

    @torch.inference_mode()
    def ttt_step(self, input_norm, prev_target_norm=None):
        safe = self.accuracy.safe
        previous = None
        if prev_target_norm is not None:
            raw = torch.as_tensor(
                prev_target_norm, device=self.device, dtype=torch.float32
            )
            previous = physical(raw, safe._target_mean, safe._target_std, safe._target_zero)[0, ..., :2]
        self.context.observe(previous)
        prediction, info = self.accuracy.ttt_step(input_norm, prev_target_norm)
        raw = torch.as_tensor(input_norm, device=self.device, dtype=torch.float32)
        inp = physical(raw, safe._input_mean, safe._input_std, safe._input_zero)[0, ..., :2]
        std = safe._target_std[0, ..., :2]
        mean = safe._target_mean[0, ..., :2]
        lower = info["lower"][0, ..., :2] * std + mean
        upper = info["upper"][0, ..., :2] * std + mean
        x, g = self.context.features(inp, 0.5 * (lower + upper), 0.5 * (upper - lower))
        if previous is None:
            return (prediction, info)
        shared = canonical(x[None])
        if isinstance(self.interval, MultiscaleIntervalHead):
            shift, scale = self.interval.forward_prepared(shared, g[None])
        else:
            shift, scale = TemporalIntervalHead.forward(self.interval, shared, g[None])
        shift = shift[0].reshape(20, 2, 32, 64).permute(0, 2, 3, 1)
        scale = scale[0].reshape(20, 2, 32, 64).permute(0, 2, 3, 1)
        half = 0.5 * (info["upper"][0, ..., :2] - info["lower"][0, ..., :2])
        lo = info["lower"].clone()
        hi = info["upper"].clone()
        lo[0, ..., :2] += half * (shift + 1 - scale)
        hi[0, ..., :2] += half * (shift + scale - 1)
        result = dict(info)
        result["lower"] = lo
        result["upper"] = hi
        new_lower = lo[0, ..., :2] * std + mean
        new_upper = hi[0, ..., :2] * std + mean
        delta = 0.5 * (new_lower + new_upper - lower - upper)
        p = prediction[0, ..., :2] * std + mean
        pm = p.mean(0)
        pv = p.var(0, correction=0)
        dm = delta.mean(0)
        q = p - pm + 0.5 * (delta - dm)
        q = q - q.mean(0)
        qv = q.var(0, correction=0)
        q = torch.where(
            (pv > 1e-12) & (qv > 1e-12), q * (pv / qv.clamp_min(1e-12)).sqrt(), p - pm
        )
        corrected = pm + q + 0.5 * dm
        if self.energy is not None:
            mu, lv = self.energy.forward_prepared(
                shared, x[None], g[None], image(p)[None]
            )
            mu = mu[0].permute(1, 2, 0) * self.energy_strength
            lv = lv[0].permute(1, 2, 0) * self.energy_strength
            cm = corrected.mean(0)
            corrected = (
                cm
                + mu * constant_like(p, [0.1, 0.02])
                + (corrected - cm) * torch.exp(0.5 * lv)
            )
        prediction = prediction.clone()
        prediction[0, ..., :2] = (corrected - mean) / std
        return (prediction, result)
