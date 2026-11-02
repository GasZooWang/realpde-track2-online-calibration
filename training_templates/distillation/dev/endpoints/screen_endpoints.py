"""Interval endpoint refinement."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "dev/algorithm"), str(ROOT / "dev/shared"), str(ROOT / "submission")]
from metrics import record, summary
from train_density import Density, choose, distribution
from train_refiner import features
from train_wave import Wave, apply

SIGMA = 0.0563870259


def interval_statistics(lo, hi, mu, sd, weight):
    zl, zh = (lo[:, None] - mu) / sd, (hi[:, None] - mu) / sd
    mass = (0.5 * (torch.erf(zh / math.sqrt(2)) - torch.erf(zl / math.sqrt(2))) * weight).sum(1)
    pl = torch.exp(-0.5 * zl.square()) * weight / (math.sqrt(2 * math.pi) * sd)
    ph = torch.exp(-0.5 * zh.square()) * weight / (math.sqrt(2 * math.pi) * sd)
    fl, fh = pl.sum(1), ph.sum(1)
    dfl, dfh = (-zl / sd * pl).sum(1), (-zh / sd * ph).sum(1)
    return mass.clamp_min(0), fl, fh, dfl, dfh


def utility(lo, hi, mu, sd, weight, penalty):
    zl, zh = (lo[:, None] - mu) / sd, (hi[:, None] - mu) / sd
    mass = (0.5 * (torch.erf(zh / math.sqrt(2)) - torch.erf(zl / math.sqrt(2))) * weight).sum(1)
    return mass.clamp_min(0) * torch.exp(-penalty * (hi - lo))


@torch.inference_mode()
def refine(raw, lower, upper, coarse=None, steps=10):
    mu, sd, logw = distribution(raw.float())
    weight = logw.exp()
    center = (lower + upper) / 2
    radius = ((upper - lower) / 2).clamp_min(1e-7)
    penalty = radius / SIGMA
    cl, ch = choose(raw, lower, upper) if coarse is None else coarse
    lo, hi = (cl - center) / radius, (ch - center) / radius
    initial = utility(lo, hi, mu, sd, weight, penalty)
    best = initial.clone()
    for _ in range(steps):
        mass, fl, fh, dfl, dfh = interval_statistics(lo, hi, mu, sd, weight)
        denom = mass.clamp_min(1e-7)
        gl, gh = penalty - fl / denom, fh / denom - penalty
        hll = -dfl / denom - (fl / denom).square()
        hhh = dfh / denom - (fh / denom).square()
        hlh = fl * fh / denom.square()
        det = hll * hhh - hlh.square()
        negative_definite = (hll < -1e-6) & (hhh < -1e-6) & (det > 1e-8)
        nl = (-hhh * gl + hlh * gh) / det.clamp_min(1e-8)
        nh = (hlh * gl - hll * gh) / det.clamp_min(1e-8)
        gl, gh = gl / (gl.abs() + 1), gh / (gh.abs() + 1)
        dl = torch.where(negative_definite, nl, gl).clamp(-0.75, 0.75)
        dh = torch.where(negative_definite, nh, gh).clamp(-0.75, 0.75)
        origin_l, origin_h = lo, hi
        for rate in (1.0, 0.5, 0.25, 0.125):
            ll, hh = origin_l + rate * dl, origin_h + rate * dh
            allowed = (hh - ll >= 1e-4) & torch.isfinite(ll) & torch.isfinite(hh)
            score = utility(ll, hh, mu, sd, weight, penalty)
            take = allowed & (score > best)
            lo, hi = torch.where(take, ll, lo), torch.where(take, hh, hi)
            best = torch.where(take, score, best)
    return center + radius * lo, center + radius * hi, initial, best


def get_model(path, device, density=False):
    state = torch.load(path, map_location=device, weights_only=False)
    model = (Density(state["channels"], state["width"]) if density else
             Wave(state["channels"], state["width"], state["mode"]))
    model.load_state_dict(state["state_dict"])
    return model.to(device).eval().to(memory_format=torch.channels_last), state


def load_paths(args):
    split = json.loads((ROOT / "dev/backbone/split.json").read_text())
    names = split["train50" if args.split == "train10" else args.split]
    if args.split == "train10":
        names = names[:10]
    paths = [ROOT / "dev/shared/cache" / f"{Path(name).stem}.pt" for name in names]
    if args.dense:
        if args.split != "train50":
            raise ValueError("Dense teacher generation is restricted to train50")
        paths += [ROOT / "dev/algorithm/cache_dense" / f"{Path(name).stem}_o{o:02d}.pt"
                  for name in names for o in (5, 10, 15)]
    return paths


@torch.inference_mode()
def synthetic_check():
    torch.manual_seed(73)
    raw = torch.randn(4, 360, 2, 3) * 0.5
    lower = -torch.rand(4, 20, 2, 2, 3) * 0.03
    upper = -lower
    lo, hi, start, final = refine(raw, lower, upper, steps=15)
    assert torch.isfinite(lo).all() and torch.isfinite(hi).all() and (lo <= hi).all()
    assert (final >= start).all(), "The density-model objective must never fall"
    print(json.dumps(dict(status="PASS", elements=lo.numel()), indent=2))


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train10", "train50", "eval25"], default="train10")
    ap.add_argument("--density", type=Path, default=ROOT / "dev/algorithm/reports/density_best.pt")
    ap.add_argument("--point", type=Path, default=ROOT / "dev/algorithm/reports/unified_best.pt")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dense", action="store_true")
    ap.add_argument("--teacher-dir", type=Path)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(2)
    if args.self_test:
        synthetic_check()
        return
    if args.teacher_dir and args.split != "train50":
        raise ValueError("Teacher caches may only be generated from train50")
    if args.teacher_dir:
        args.teacher_dir.mkdir(parents=True, exist_ok=True)
    out = args.output or ROOT / f"dev/endpoints/reports/endpoints_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    net, state = get_model(args.density, args.device, density=True)
    point, point_state = get_model(args.point, args.device)
    rows, model_improvement = {}, []
    began = time.perf_counter()
    for path in load_paths(args):
        d = torch.load(path, map_location=args.device, weights_only=False)
        f, scale = features(d)
        p, t, l, h = [d[k] for k in ("prediction", "target", "lower", "upper")]
        parts = {k: [] for k in ("point", "point_lower", "point_upper", "coarse_lower", "coarse_upper", "refined_lower", "refined_upper")}
        for start in range(0, len(p), args.batch):
            ix = slice(start, start + args.batch)
            with torch.autocast("cuda", dtype=torch.float16, enabled=args.device.startswith("cuda")):
                raw, point_raw = net(f[ix].to(torch.float16 if args.device.startswith("cuda") else torch.float32)), point(f[ix].to(torch.float16 if args.device.startswith("cuda") else torch.float32))
            pp, pl, ph = apply(point_raw, p[ix], l[ix], h[ix], scale[ix], "joint", point_state["strength"])
            cl, ch = choose(raw, l[ix], h[ix])
            rl, rh, before, after = refine(raw, l[ix], h[ix], (cl, ch), args.steps)
            model_improvement.append([float(before.sum()), float(after.sum()), before.numel()])
            for key, value in zip(parts, (pp, pl, ph, cl, ch, rl, rh)):
                parts[key].append(value)
        z = {key: torch.cat(values) for key, values in parts.items()}
        rows.setdefault("student", []).append(record(p, t, l, h))
        rows.setdefault("algorithm_point", []).append(record(z["point"], t, z["point_lower"], z["point_upper"]))
        for origin in ("student", "algorithm_point"):
            q, base_l, base_h = (p, l, h) if origin == "student" else (z["point"], z["point_lower"], z["point_upper"])
            for method in ("coarse", "refined"):
                for strength in (0.25, 0.5, 0.75, 1.0):
                    ll = base_l + strength * (z[method + "_lower"] - base_l)
                    hh = base_h + strength * (z[method + "_upper"] - base_h)
                    rows.setdefault(f"{origin}/{method}/{strength}", []).append(record(q, t, ll, hh))
        if args.teacher_dir:
            torch.save(dict(lower=z["refined_lower"].cpu(), upper=z["refined_upper"].cpu()), args.teacher_dir / path.name)
        scores = {key: summary(value) for key, value in rows.items()}
        best = sorted(scores, key=lambda key: scores[key]["proxy"], reverse=True)[:6]
        print(path.name, "seconds", round(time.perf_counter() - began, 1), best, flush=True)
        out.write_text(json.dumps(dict(split=args.split, paths=[p.name for p in load_paths(args)],
                                       density_sha256=hashlib.sha256(args.density.read_bytes()).hexdigest(),
                                       point_sha256=hashlib.sha256(args.point.read_bytes()).hexdigest(),
                                       steps=args.steps, fixed_time_score=90.167643, rows=rows, scores=scores,
                                       model_utility_sums=model_improvement, seconds=time.perf_counter() - began), indent=2))


if __name__ == "__main__":
    main()
