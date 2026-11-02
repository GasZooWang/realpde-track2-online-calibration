"""Forecast and energy correction."""

from .energy import DirectEnergyHead, energy_features, energy_logscale
from .forecast import CausalForecastHead
from .history import MomentHistory
from .temporal_interval import image
import torch
from .constants import constant_like


class Model:

    def __init__(
        self, sps, forecast_state, energy_state, width=32, energy_strength=0.5
    ):
        self.sps = sps
        self.device = sps.device
        self.history = MomentHistory()
        self.captured = {}
        self.head = CausalForecastHead(width).to(self.device).eval()
        self.head.load_state_dict(forecast_state, strict=True)
        self.head.requires_grad_(False)
        self.extra = DirectEnergyHead().to(self.device).eval()
        self.extra.load_state_dict(energy_state, strict=True)
        self.extra.requires_grad_(False)
        self.energy_strength = energy_strength
        features = sps.context.features
        interval = sps.interval.forward_prepared
        head_forward = self.head.forward

        def capture_features(*args):
            x, g = features(*args)
            self.captured.update(x=x, g=g)
            return (x, g)

        def capture_interval(*args):
            shift, scale = interval(*args)
            self.captured.update(shift=shift, scale=scale)
            return (shift, scale)

        def capture_history(*args):
            self.captured["moment_history"] = args[3]
            return head_forward(*args)

        sps.context.features = capture_features
        sps.interval.forward_prepared = capture_interval
        self.head.forward = capture_history

    @torch.inference_mode()
    def ttt_step(self, input_norm, prev_target_norm=None):
        pred, info = self._forecast_step(input_norm, prev_target_norm)
        if prev_target_norm is None:
            return (pred, info)
        safe = self.sps.accuracy.safe
        std = safe._target_std[0, ..., :2]
        mean = safe._target_mean[0, ..., :2]
        p = image(pred[0, ..., :2] * std + mean).reshape(1, 20, 2, 32, 64)
        pm = p.mean(1)
        pv = p.var(1, correction=0)
        f, reference = energy_features(
            self.captured["x"][None], self.captured["moment_history"], pm, pv
        )
        lv = energy_logscale(self.extra(f), pm, pv, reference, self.energy_strength)
        q = pm[:, None] + (p - pm[:, None]) * torch.exp(0.5 * lv[:, None])
        out = pred.clone()
        out[0, ..., :2] = (q[0].permute(0, 2, 3, 1) - mean) / std
        return (out, info)

    def reset_ttt_state(self):
        self.sps.reset_ttt_state()
        self.history.reset()
        self.captured.clear()

    @torch.inference_mode()
    def _forecast_step(self, input_norm, prev_target_norm=None):
        self.captured.clear()
        prediction, info = self.sps.ttt_step(input_norm, prev_target_norm)
        safe = self.sps.accuracy.safe
        std = safe._target_std[0, ..., :2]
        mean = safe._target_mean[0, ..., :2]
        p = prediction[0, ..., :2] * std + mean
        pp = image(p).reshape(20, 2, 32, 64)
        x, g = (self.captured["x"], self.captured["g"])
        units = constant_like(x, [0.1, 0.02])[None, :, None, None]
        inp = x[:40].reshape_as(pp) * units
        previous = None
        if prev_target_norm is not None:
            raw = torch.as_tensor(
                prev_target_norm, device=self.device, dtype=torch.float32
            )
            previous = image(
                (raw * safe._target_std + safe._target_mean)[0, ..., :2]
            ).reshape_as(pp)
        history = self.history.features(
            inp, pp.mean(0), pp.var(0, correction=0), previous
        )
        if previous is None:
            return (prediction, info)
        q, shift, scale = self.head(
            x[None],
            g[None],
            image(p)[None],
            history[None],
            torch.zeros(1, device=self.device, dtype=torch.bool),
            self.captured["shift"],
            self.captured["scale"],
        )
        q = q[0].reshape(20, 2, 32, 64).permute(0, 2, 3, 1)
        shift = shift[0].reshape(20, 2, 32, 64).permute(0, 2, 3, 1)
        scale = scale[0].reshape(20, 2, 32, 64).permute(0, 2, 3, 1)
        result = prediction.clone()
        result[0, ..., :2] = (q - mean) / std
        lo = info["lower"].clone()
        hi = info["upper"].clone()
        half = 0.5 * (hi[0, ..., :2] - lo[0, ..., :2])
        lo[0, ..., :2] += half * (shift + 1 - scale)
        hi[0, ..., :2] += half * (shift + scale - 1)
        updated = dict(info)
        updated["lower"] = lo
        updated["upper"] = hi
        return (result, updated)
