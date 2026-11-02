"""Unified correction head and mixture-density intervals."""

import torch
import warnings
import math
from torch import nn
from torch.nn import functional as F
from .student import Student
from .calibration import INPUT_MEAN, INPUT_STD, TARGET_MEAN, TARGET_STD


DENSITY_SIGMA = 0.0563870259


class DensityNetwork(nn.Module):

    def __init__(self, checkpoint, device):
        super().__init__()
        width = int(checkpoint["width"])
        channels = int(checkpoint["channels"])
        self.net = Student(width)
        self.net.stem = nn.Conv2d(channels, width, 3, padding=1)
        self.net.out = nn.Conv2d(width, 360, 3, padding=1)
        self.load_state_dict(checkpoint["state_dict"], strict=True)
        self.to(device).eval().requires_grad_(False)
        if torch.device(device).type == "cuda":
            self.half().to(memory_format=torch.channels_last)

    def forward(self, value):
        n = self.net
        value = value.to(n.stem.weight.dtype)
        a = n.b0(F.silu(n.stem(value)))
        b = n.b1(F.silu(n.d1(a)))
        c = n.b2(F.silu(n.d2(b)))
        b = n.r1(
            b
            + F.interpolate(
                n.u1(c), size=b.shape[-2:], mode="bilinear", align_corners=False
            )
        )
        a = n.r0(
            a
            + F.interpolate(
                n.u0(b), size=a.shape[-2:], mode="bilinear", align_corners=False
            )
        )
        return n.out(a).float()


def density_distribution(raw):
    batch, _channels, height, width = raw.shape
    value = raw.float().reshape(batch, 9, 20, 2, height, width)
    base = value.new_tensor((-0.5, 0.0, 0.5)).reshape(1, 3, 1, 1, 1, 1)
    mean = base + 2.0 * torch.tanh(value[:, :3])
    std = 0.75 * torch.exp(1.5 * torch.tanh(value[:, 3:6]))
    log_weight = (
        value[:, 6:9]
        + value.new_tensor((-1.0, 0.0, -1.0)).reshape(1, 3, 1, 1, 1, 1)
    ).log_softmax(dim=1)
    return mean, std, log_weight


def density_choose(mean, std, weight, lower, upper, halves):
    center = 0.5 * (lower + upper)
    radius = 0.5 * (upper - lower)
    candidate_centers = [
        torch.zeros_like(mean[:, 0]),
        mean[:, 0],
        mean[:, 1],
        mean[:, 2],
        (mean * weight).sum(dim=1),
    ]
    normal_scale = std * math.sqrt(2.0)
    penalties = [
        torch.exp(-2.0 * half * radius / DENSITY_SIGMA) for half in halves
    ]
    best = torch.full_like(center, -1.0)
    best_lower = lower.clone()
    best_upper = upper.clone()
    for midpoint in candidate_centers:
        for half, interval_penalty in zip(halves, penalties):
            lo = midpoint - half
            hi = midpoint + half
            coverage = (
                0.5
                * (
                    torch.erf((hi[:, None] - mean) / normal_scale)
                    - torch.erf((lo[:, None] - mean) / normal_scale)
                )
                * weight
            ).sum(dim=1)
            utility = coverage * interval_penalty
            take = utility > best
            best = torch.maximum(best, utility)
            best_lower = torch.where(take, center + radius * lo, best_lower)
            best_upper = torch.where(take, center + radius * hi, best_upper)
    return best_lower, best_upper


def density_interval_statistics(lo, hi, mean, std, weight):
    zl = (lo[:, None] - mean) / std
    zh = (hi[:, None] - mean) / std
    mass = (
        0.5
        * (torch.erf(zh / math.sqrt(2.0)) - torch.erf(zl / math.sqrt(2.0)))
        * weight
    ).sum(dim=1)
    density_scale = math.sqrt(2.0 * math.pi) * std
    pl = torch.exp(-0.5 * zl.square()) * weight / density_scale
    ph = torch.exp(-0.5 * zh.square()) * weight / density_scale
    fl = pl.sum(dim=1)
    fh = ph.sum(dim=1)
    dfl = (-zl / std * pl).sum(dim=1)
    dfh = (-zh / std * ph).sum(dim=1)
    return mass.clamp_min(0.0), fl, fh, dfl, dfh


def density_utility(lo, hi, mean, std, weight, penalty):
    zl = (lo[:, None] - mean) / std
    zh = (hi[:, None] - mean) / std
    mass = (
        0.5
        * (torch.erf(zh / math.sqrt(2.0)) - torch.erf(zl / math.sqrt(2.0)))
        * weight
    ).sum(dim=1)
    return mass.clamp_min(0.0) * torch.exp(-penalty * (hi - lo))


def density_refine(mean, std, weight, lower, upper, coarse, steps, rates):
    center = 0.5 * (lower + upper)
    radius = (0.5 * (upper - lower)).clamp_min(1e-7)
    penalty = radius / DENSITY_SIGMA
    coarse_lower, coarse_upper = coarse
    lo = (coarse_lower - center) / radius
    hi = (coarse_upper - center) / radius
    best = None
    for _ in range(steps):
        mass, fl, fh, dfl, dfh = density_interval_statistics(
            lo, hi, mean, std, weight
        )
        if best is None:
            best = mass * torch.exp(-penalty * (hi - lo))
        denominator = mass.clamp_min(1e-7)
        fl_over_mass = fl / denominator
        fh_over_mass = fh / denominator
        gradient_lower = penalty - fl_over_mass
        gradient_upper = fh_over_mass - penalty
        hessian_ll = -dfl / denominator - fl_over_mass.square()
        hessian_hh = dfh / denominator - fh_over_mass.square()
        hessian_lh = fl * fh / denominator.square()
        determinant = hessian_ll * hessian_hh - hessian_lh.square()
        negative_definite = (
            (hessian_ll < -1e-6)
            & (hessian_hh < -1e-6)
            & (determinant > 1e-8)
        )
        safe_determinant = determinant.clamp_min(1e-8)
        newton_lower = (
            -hessian_hh * gradient_lower + hessian_lh * gradient_upper
        ) / safe_determinant
        newton_upper = (
            hessian_lh * gradient_lower - hessian_ll * gradient_upper
        ) / safe_determinant
        bounded_lower = gradient_lower / (gradient_lower.abs() + 1.0)
        bounded_upper = gradient_upper / (gradient_upper.abs() + 1.0)
        delta_lower = torch.where(
            negative_definite, newton_lower, bounded_lower
        ).clamp(-0.75, 0.75)
        delta_upper = torch.where(
            negative_definite, newton_upper, bounded_upper
        ).clamp(-0.75, 0.75)
        origin_lower = lo
        origin_upper = hi
        for rate in rates:
            candidate_lower = origin_lower + rate * delta_lower
            candidate_upper = origin_upper + rate * delta_upper
            allowed = (
                (candidate_upper - candidate_lower >= 1e-4)
                & torch.isfinite(candidate_lower)
                & torch.isfinite(candidate_upper)
            )
            score = density_utility(
                candidate_lower,
                candidate_upper,
                mean,
                std,
                weight,
                penalty,
            )
            take = allowed & (score > best)
            lo = torch.where(take, candidate_lower, lo)
            hi = torch.where(take, candidate_upper, hi)
            best = torch.where(take, score, best)
    return center + radius * lo, center + radius * hi


class Network(nn.Module):
    def __init__(self, checkpoint, device):
        super().__init__()
        self.net = Student(checkpoint["width"])
        self.net.stem = nn.Conv2d(
            checkpoint["channels"], checkpoint["width"], 3, padding=1
        )
        channels = 123 if checkpoint["mode"] in ("joint", "interval") else 43
        if checkpoint.get("component_energy", False):
            if checkpoint["mode"] != "joint" or checkpoint["width"] != 32:
                raise ValueError("Component energy requires a joint head of width 32")
            channels += 1
        self.net.out = nn.Conv2d(checkpoint["width"], channels, 3, padding=1)
        self.load_state_dict(checkpoint["state_dict"])
        self.to(device).eval().requires_grad_(False)
        if torch.device(device).type == "cuda":
            self.half().to(memory_format=torch.channels_last)

    def forward(self, x):
        n = self.net
        x = x.to(n.stem.weight.dtype)
        a = n.b0(F.silu(n.stem(x)))
        b = n.b1(F.silu(n.d1(a)))
        c = n.b2(F.silu(n.d2(b)))
        b = n.r1(
            b
            + F.interpolate(
                n.u1(c), size=b.shape[-2:], mode="bilinear", align_corners=False
            )
        )
        a = n.r0(
            a
            + F.interpolate(
                n.u0(b), size=a.shape[-2:], mode="bilinear", align_corners=False
            )
        )
        return n.out(a).float()


class Core(nn.Module):
    def __init__(self, checkpoint, device):
        super().__init__()
        self.model = Network(checkpoint, device)
        self.density = DensityNetwork(checkpoint["density"], device)
        self.mode = checkpoint["mode"]
        self.strength = checkpoint["strength"]
        self.component_energy = bool(checkpoint.get("component_energy", False))
        self.energy_strength = float(checkpoint.get("energy_strength", 1.0))
        self.density_steps = int(checkpoint["density_steps"])
        self.density_blend = float(checkpoint["density_blend"])
        density_point_strength = checkpoint["density_point_strength"]
        if not isinstance(density_point_strength, (list, tuple)) or len(density_point_strength) != 2:
            raise ValueError("density_point_strength must contain [u, v]")
        self.density_point_strength_u = float(density_point_strength[0])
        self.density_point_strength_v = float(density_point_strength[1])
        if not math.isfinite(self.density_point_strength_u) or not math.isfinite(self.density_point_strength_v):
            raise ValueError("density point strengths must be finite")
        density_point_strength_tensor = torch.tensor(
            [self.density_point_strength_u, self.density_point_strength_v],
            dtype=torch.float32,
            device=device,
        )
        if not torch.isfinite(density_point_strength_tensor).all():
            raise ValueError("density point strengths must be finite in float32")
        self.register_buffer(
            "density_point_strength",
            density_point_strength_tensor.view(1, 1, 2, 1, 1),
        )
        density_halves = checkpoint["density_halves"]
        if not isinstance(density_halves, (list, tuple)) or not density_halves:
            raise ValueError("density_halves must be a non-empty list or tuple")
        self.density_halves = tuple(float(value) for value in density_halves)
        if len(self.density_halves) > 32:
            raise ValueError("density_halves must contain at most 32 values")
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in self.density_halves
        ):
            raise ValueError("density_halves must contain finite positive values")
        if any(
            left >= right
            for left, right in zip(self.density_halves, self.density_halves[1:])
        ):
            raise ValueError("density_halves must be strictly increasing")
        density_halves_tensor = torch.tensor(
            self.density_halves, dtype=torch.float32, device=device
        )
        if not torch.isfinite(density_halves_tensor).all():
            raise ValueError("density_halves must be finite in float32")
        if not bool((density_halves_tensor > 0.0).all()):
            raise ValueError("density_halves must stay positive in float32")
        density_rates = checkpoint["density_rates"]
        if not isinstance(density_rates, (list, tuple)) or not density_rates:
            raise ValueError("density_rates must be a non-empty list or tuple")
        self.density_rates = tuple(float(value) for value in density_rates)
        if len(self.density_rates) > 32:
            raise ValueError("density_rates must contain at most 32 values")
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in self.density_rates
        ):
            raise ValueError("density_rates must contain finite positive values")
        if any(
            left <= right
            for left, right in zip(self.density_rates, self.density_rates[1:])
        ):
            raise ValueError("density_rates must be strictly decreasing")
        density_rates_tensor = torch.tensor(
            self.density_rates, dtype=torch.float32, device=device
        )
        if not torch.isfinite(density_rates_tensor).all():
            raise ValueError("density_rates must be finite in float32")
        if not bool((density_rates_tensor > 0.0).all()):
            raise ValueError("density_rates must stay positive in float32")
        for key, values in [
            ("im", INPUT_MEAN),
            ("ist", INPUT_STD),
            ("tm", TARGET_MEAN),
            ("tst", TARGET_STD),
        ]:
            self.register_buffer(
                key, torch.tensor(values, device=device).view(1, 1, 1, 1, 3)
            )
        self.register_buffer(
            "rates", torch.tensor([0.1, 0.3, 0.7], device=device).view(3, 1, 1, 1, 1)
        )
        self.register_buffer(
            "units", torch.tensor([0.1, 0.02], device=device).view(1, 2, 1, 1)
        )
        self.input_zero = -self.im / self.ist
        self.target_zero = -self.tm / self.tst

    @staticmethod
    def images(value):
        return value[..., :2].permute(0, 1, 4, 2, 3)

    def forward(
        self,
        inp,
        pred,
        lower,
        upper,
        previous,
        first,
        observed,
        old_pm,
        old_pv,
        old_im,
        old_iv,
        old_bias,
        old_k,
    ):
        x = self.images(
            torch.where(inp == self.input_zero, 0.0, inp * self.ist + self.im)
        )
        t = self.images(
            torch.where(
                previous == self.target_zero, 0.0, previous * self.tst + self.tm
            )
        )
        p, l, h = [self.images(v * self.tst + self.tm) for v in (pred, lower, upper)]
        pm = p.mean(1)
        im = x.mean(1)
        iv = x.var(1, correction=0)
        pv = p.var(1, correction=0)
        s = (0.5 * (iv + pv)).clamp_min(2.5e-7).sqrt()
        base_l = l
        base_h = h
        c = (l + h) / 2
        r = (h - l) / 2
        bias = t.mean(1) - old_pm
        k = ((t.var(1, correction=0) + 1e-8) / (old_pv + 1e-8)).log().clamp(-3, 3)
        ei = torch.where(
            first, im[None], old_im * (1 - self.rates) + im[None] * self.rates
        )
        ev = torch.where(
            first, iv[None], old_iv * (1 - self.rates) + iv[None] * self.rates
        )
        eb = torch.where(
            observed, old_bias * (1 - self.rates) + bias[None] * self.rates, old_bias
        )
        ek = torch.where(
            observed, old_k * (1 - self.rates) + k[None] * self.rates, old_k
        )
        pieces = [
            ((x - pm[:, None]) / s[:, None]).flatten(1, 2),
            ((p - pm[:, None]) / s[:, None]).flatten(1, 2),
            ((c - pm[:, None]) / s[:, None]).flatten(1, 2),
            (r.clamp_min(1e-7) / s[:, None]).log().flatten(1, 2),
        ]
        for i in range(3):
            pieces.extend(
                [(ei[i] - pm) / s, (ev[i] + 1e-8).sqrt() / s, eb[i] / s, ek[i]]
            )
        pieces.extend(
            [
                (x[:, -5:].mean(1) - x[:, :5].mean(1)) / s,
                (x == 0).float().mean(1),
                pv.sqrt() / s,
                pm / self.units,
                im / self.units,
                (s / self.units).log(),
            ]
        )
        f = torch.cat(pieces, 1).clamp(-10, 10)
        raw_unscaled = self.model(f)
        raw = raw_unscaled * self.strength
        mu = pm[:, None]
        var = pv[:, None]
        wave = p - mu + s[:, None] * 2 * torch.tanh(raw[:, 3:43]).reshape_as(p)
        wave = wave - wave.mean(1, keepdim=True)
        wv = wave.var(1, correction=0, keepdim=True)
        wave = torch.where(
            (wv > 1e-12) & (var > 1e-12),
            wave * (var / wv.clamp_min(1e-12)).sqrt(),
            p - mu,
        )
        if self.mode == "interval":
            q = p
        elif self.mode == "wave":
            q = mu + wave
        else:
            q = (
                mu
                + 0.5 * s[:, None] * torch.tanh(raw[:, :2])[:, None]
                + wave * (0.75 * torch.tanh(raw[:, 2:3])[:, None]).exp()
            )
        if self.component_energy:
            delta = 0.5 * torch.tanh(raw_unscaled[:, 123:124]) * self.energy_strength
            qwave = q - q.mean(1, keepdim=True)
            qvar = qwave.square().mean(1)
            total = qvar.sum(1, keepdim=True)
            fraction = torch.where(
                total > 0, qvar / total.clamp_min(1e-30), torch.zeros_like(qvar)
            )
            signed = torch.cat((delta, -delta), 1)
            ratio_delta = (fraction * signed.expm1()).sum(1, keepdim=True)
            log_normalizer = -0.5 * torch.log1p(ratio_delta.clamp_min(-0.999999))
            log_factor = 0.5 * signed + log_normalizer
            q = q + qwave * log_factor.expm1()[:, None]
        if self.mode in ("joint", "interval"):
            shift = 0.7 * torch.tanh(raw[:, 43:83]).reshape_as(p)
            width = (0.7 * torch.tanh(raw[:, 83:123])).exp().reshape_as(p)
            l, h = c + r * (shift - width), c + r * (shift + width)
        density_raw = self.density(f)
        density_mean, density_std, density_log_weight = density_distribution(density_raw)
        density_weight = density_log_weight.exp()
        base_center = 0.5 * (base_l + base_h)
        base_radius = 0.5 * (base_h - base_l)
        density_point = base_center + base_radius * (
            density_mean * density_weight
        ).sum(dim=1)
        q = q + self.density_point_strength * (density_point - q).mean(
            dim=1, keepdim=True
        )
        density_coarse = density_choose(
            density_mean,
            density_std,
            density_weight,
            base_l,
            base_h,
            self.density_halves,
        )
        density_lower, density_upper = density_refine(
            density_mean,
            density_std,
            density_weight,
            base_l,
            base_h,
            density_coarse,
            self.density_steps,
            self.density_rates,
        )
        l = l + self.density_blend * (density_lower - l)
        h = h + self.density_blend * (density_upper - h)
        values = []
        for value, original in [(q, pred), (l, lower), (h, upper)]:
            uv = (value.permute(0, 1, 3, 4, 2) - self.tm[..., :2]) / self.tst[..., :2]
            values.append(torch.cat([uv, original[..., 2:]], -1))
        return torch.stack(values), [pm, pv, ei, ev, eb, ek], f


class AlgorithmModel:
    @torch.inference_mode()
    def __init__(self, base, checkpoint, device, fuse=True, submission_dir=None):
        self.base = base
        self.device = torch.device(device)
        self.sps = base.sps
        self.core = Core(checkpoint, self.device)
        self.graph = None
        shape = (1, 20, 32, 64, 3)
        self.inputs = [torch.zeros(shape, device=self.device) for _ in range(5)]
        self.first = torch.ones((), dtype=torch.bool, device=self.device)
        self.observed = torch.zeros((), dtype=torch.bool, device=self.device)
        self.state = [torch.zeros((1, 2, 32, 64), device=self.device) for _ in range(2)]
        self.state.extend(
            [torch.zeros((3, 1, 2, 32, 64), device=self.device) for _ in range(4)]
        )
        self.fused = False
        if self.device.type == "cuda":
            if fuse:
                original = self.core
                try:
                    from .algorithm_ptx import load_algorithm_core
                    self.core = load_algorithm_core(submission_dir, original, self.device)
                    self._body()
                    torch.cuda.synchronize(self.device)
                    self.fused = True
                except Exception as error:
                    self.core = original
                    warnings.warn("Packaged algorithm kernels unavailable: " + str(error)[:240])
                finally:
                    for value in self.state:
                        value.zero_()
                    self.first.fill_(True)
                    self.observed.fill_(False)
            if self.fused:
                try:
                    self._capture()
                except Exception as error:
                    self.graph = None
                    warnings.warn("Algorithm replay unavailable: " + str(error)[:240])
        self.reset_ttt_state()

    def _capture(self):
        current = torch.cuda.current_stream(self.device)
        stream = torch.cuda.Stream(device=self.device)
        try:
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self._body()
            current.wait_stream(stream)
            current.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                self._body()
        finally:
            current.wait_stream(stream)

    def _body(self):
        self.output, new, self.features = self.core(
            *self.inputs, self.first, self.observed, *self.state
        )
        for a, b in zip(self.state, new):
            a.copy_(b)

    @property
    def status(self):
        return dict(
            self.base.status,
            algorithm="temporal_shape_and_moments",
            algorithm_mode=self.core.mode,
            algorithm_graph=self.graph is not None,
            algorithm_fused=self.fused,
            algorithm_backend=getattr(self.core, "backend", "eager"),
            exact_density=True,
            exact_density_width=24,
            exact_density_steps=self.core.density_steps,
            exact_density_blend=self.core.density_blend,
            exact_density_point_strength_u=self.core.density_point_strength_u,
            exact_density_point_strength_v=self.core.density_point_strength_v,
            exact_density_halves=self.core.density_halves,
            exact_density_rates=self.core.density_rates,
        )

    @torch.inference_mode()
    def reset_ttt_state(self):
        self.base.reset_ttt_state()
        self.first.fill_(True)
        self.observed.fill_(False)
        for value in self.state:
            value.zero_()

    @torch.inference_mode()
    def ttt_step(self, input_norm, prev_target_norm=None):
        p, info = self.base.ttt_step(input_norm, prev_target_norm)
        for dst, src in zip(
            self.inputs[:4], [input_norm, p, info["lower"], info["upper"]]
        ):
            dst.copy_(torch.as_tensor(src, device=self.device))
        self.observed.fill_(prev_target_norm is not None)
        if prev_target_norm is not None:
            self.inputs[4].copy_(torch.as_tensor(prev_target_norm, device=self.device))
        if self.graph is None:
            self._body()
        else:
            self.graph.replay()
        self.first.fill_(False)
        output = self.output.clone()
        out = dict(info)
        out["lower"] = output[1]
        out["upper"] = output[2]
        return output[0], out
