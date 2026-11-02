"""Metric-aligned fine-tuning."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "dev/algorithm"),
    str(ROOT / "dev/shared"),
    str(ROOT / "submission"),
]
from train_refiner import features, load
from train_wave import Wave, apply, evaluate
from metrics import errors, record, summary


def aggregate_loss(q, target, lo, hi, temperature, hard_forward=True, per_window=False):
    e = errors(q, target)
    accuracy = (0.5 / (0.5 + e) * e.new_tensor([0.5, 0.3, 0.2])).sum(-1)
    valid = target != 0
    soft = torch.sigmoid((target - lo) / temperature) * torch.sigmoid(
        (hi - target) / temperature
    )
    if hard_forward:
        hard = ((target >= lo) & (target <= hi)).to(soft.dtype)
        coverage = hard + soft - soft.detach()
    else:
        coverage = soft
    reward = (coverage * torch.exp(-(hi - lo) / 0.0563870259) * valid).flatten(1).sum(1)
    if per_window:
        sps = 100 * reward * accuracy / valid.flatten(1).sum(1).clamp_min(1)
        scores = torch.cat(
            (100 / (1 + 0.5 * e), q.new_full((len(q), 1), 90.167643), sps[:, None]), 1
        )
        return -(
            (
                scores.clamp_min(1e-4).pow(0.6854)
                * q.new_tensor([0.5, 0.1, 0.1, 0.1, 0.2])
            )
            .sum(1)
            .pow(1 / 0.6854)
        ).mean()
    sps = 100 * (reward * accuracy).sum() / valid.sum().clamp_min(1)
    scores = torch.cat(
        (100 / (1 + 0.5 * e.mean(0)), q.new_tensor([90.167643]), sps.reshape(1))
    )
    return -(
        (scores.clamp_min(1e-4).pow(0.6854) * q.new_tensor([0.5, 0.1, 0.1, 0.1, 0.2]))
        .sum()
        .pow(1 / 0.6854)
    )


def reflect(items, dim, component):
    f, s, p, target, lo, hi = [x.flip(dim) for x in items]
    sign = p.new_ones(2)
    sign[component] = -1
    feature_sign = f.new_ones(196)
    feature_sign[:120] = sign.repeat(60)
    for rate in range(3):
        feature_sign[160 + 8 * rate : 162 + 8 * rate] = sign
        feature_sign[164 + 8 * rate : 166 + 8 * rate] = sign
    feature_sign[184:186] = sign
    feature_sign[190:194] = sign.repeat(2)
    f = f * feature_sign[None, :, None, None]
    component_sign = sign[None, None, :, None, None]
    p, target = p * component_sign, target * component_sign
    a, b = lo * component_sign, hi * component_sign
    return f, s, p, target, torch.minimum(a, b), torch.maximum(a, b)


def anchor_loss(q, lo, hi, reference, scale, lower, upper, valid):
    tq, tl, th = reference
    point = ((q - tq) / scale[:, None]).square().mean()
    radius = ((upper - lower) * 0.5).clamp_min(1e-4)
    boundary = F.smooth_l1_loss(
        (lo - tl) / radius, torch.zeros_like(lo), reduction="none"
    )
    boundary = boundary + F.smooth_l1_loss(
        (hi - th) / radius, torch.zeros_like(hi), reduction="none"
    )
    return 2 * point + (boundary * valid).sum() / valid.sum().clamp_min(1)


def smoke():
    torch.manual_seed(19)
    n = 3
    target = torch.randn(n, 20, 2, 32, 64) * 0.01
    target[0, :, :, :24] = 0
    q = (target + torch.randn_like(target) * 0.002).requires_grad_()
    lo = (q.detach() - 0.006).requires_grad_()
    hi = (q.detach() + 0.007).requires_grad_()
    exact = summary([record(q.detach(), target, lo.detach(), hi.detach())])["proxy"]
    objective = aggregate_loss(q, target, lo, hi, 0.0005)
    assert abs(float(-objective) - exact) < 2e-5, (float(-objective), exact)
    objective.backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in (q, lo, hi))
    f = torch.randn(n, 196, 32, 64)
    scale = torch.rand(n, 2, 32, 64).clamp_min(0.01)
    items = (f, scale, q.detach(), target, lo.detach(), hi.detach())
    for dim, component in ((-1, 0), (-2, 1)):
        once = reflect(items, dim, component)
        twice = reflect(once, dim, component)
        assert all(torch.equal(a, b) for a, b in zip(items, twice))
        assert torch.allclose(
            errors(once[2], once[3])[:, :2],
            errors(items[2], items[3])[:, :2],
            atol=1e-6,
        )
        assert (once[4] <= once[5]).all()
    print(
        json.dumps(
            {
                "status": "PASS",
                "finite_gradients": True,
                "reflection_involution": True,
            },
            indent=2,
        )
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="dev/algorithm/reports/unified_best.pt")
    ap.add_argument("--output", default="dev/aligned/reports/aligned")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=71)
    ap.add_argument("--anchor", type=float, default=0.1)
    ap.add_argument("--anchor-floor", type=float, default=0.02)
    ap.add_argument("--curvature", type=float, default=0.0)
    ap.add_argument("--ema", type=float, default=0.995)
    ap.add_argument(
        "--objective", choices=("aggregate", "per-window"), default="aggregate"
    )
    ap.add_argument("--coverage", choices=("ste", "smooth"), default="ste")
    ap.add_argument("--temperature", type=float, default=0.0005)
    ap.add_argument("--temperature-floor", type=float, default=0.0001)
    ap.add_argument("--temperature-decay", type=float, default=0.94)
    ap.add_argument("--no-flip", action="store_true")
    ap.add_argument("--sparse", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(2)
    if args.smoke:
        smoke()
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA; use --smoke for the CPU contract.")
    torch.manual_seed(args.seed)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    state = torch.load(checkpoint, map_location="cuda", weights_only=False)
    assert state["channels"] == 196 and state["mode"] == "joint"
    model = (
        Wave(196, state["width"], "joint").cuda().to(memory_format=torch.channels_last)
    )
    model.load_state_dict(state["state_dict"])
    with torch.no_grad():
        model.net.out.weight.mul_(float(state.get("strength", 1)))
        model.net.out.bias.mul_(float(state.get("strength", 1)))
    reference = copy.deepcopy(model).eval().requires_grad_(False)
    average = copy.deepcopy(model).eval().requires_grad_(False)
    split = json.loads((ROOT / "dev/backbone/split.json").read_text())
    train = load(split["train50"])
    if not args.sparse:
        for name in split["train50"]:
            for offset in (5, 10, 15):
                path = (
                    ROOT
                    / "dev/algorithm/cache_dense"
                    / f"{Path(name).stem}_o{offset:02d}.pt"
                )
                d = torch.load(path, map_location="cuda", weights_only=False)
                d["features"], d["scale"] = features(d)
                del d["input"]
                d["features"] = d["features"].half()
                train.append(d)
    keys = ["features", "scale", "prediction", "target", "lower", "upper"]
    tr = {}
    for key in keys:
        tr[key] = torch.cat([d.pop(key) for d in train])
    del train
    val = load(split["eval25"])
    baseline = summary(
        [record(d["prediction"], d["target"], d["lower"], d["upper"]) for d in val]
    )
    incumbent = evaluate(reference, val, (1.0,))[1.0]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    best = incumbent["proxy"]
    history = []
    n = len(tr["prediction"])
    started = time.perf_counter()
    print(
        json.dumps(
            {
                "args": vars(args),
                "train_windows": n,
            }
        ),
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(n, device="cuda")
        schedule = 0.5 * (1 + math.cos(math.pi * (epoch - 1) / args.epochs))
        for group in opt.param_groups:
            group["lr"] = args.lr * (0.1 + 0.9 * schedule)
        anchor_weight = args.anchor_floor + (args.anchor - args.anchor_floor) * schedule
        temperature = max(
            args.temperature_floor,
            args.temperature * args.temperature_decay ** (epoch - 1),
        )
        total = 0.0
        for ids in order.split(args.batch_size):
            items = [tr[k][ids] for k in keys]
            for dim, component in ((-1, 0), (-2, 1)):
                if not args.no_flip and bool(torch.rand((), device="cuda") < 0.5):
                    items = reflect(items, dim, component)
            f, scale, p, target, lower, upper = items
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                r0 = reference(f)
                reference_outputs = apply(r0, p, lower, upper, scale, "joint", 1.0)
            with torch.autocast("cuda", dtype=torch.float16):
                raw = model(f)
            q, lo, hi = apply(raw, p, lower, upper, scale, "joint", 1.0)
            objective = aggregate_loss(
                q,
                target,
                lo,
                hi,
                temperature,
                hard_forward=args.coverage == "ste",
                per_window=args.objective == "per-window",
            )
            objective = objective + anchor_weight * anchor_loss(
                q, lo, hi, reference_outputs, scale, lower, upper, target != 0
            )
            if args.curvature:
                delta = (raw[:, 3:43] - r0[:, 3:43]).reshape(-1, 20, 2, 32, 64)
                curvature = delta[:, 2:] - 2 * delta[:, 1:-1] + delta[:, :-2]
                objective = objective + args.curvature * curvature.square().mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(objective).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            with torch.no_grad():
                for avg, param in zip(average.parameters(), model.parameters()):
                    avg.lerp_(param, 1.0 - args.ema)
            total += float(objective.detach()) * len(ids)
        if epoch == 1 or epoch % 2 == 0:
            model.eval()
            scores = {
                "live": evaluate(model, val, (1.0,))[1.0],
                "ema": evaluate(average, val, (1.0,))[1.0],
            }
            selected = max(scores, key=lambda name: scores[name]["proxy"])
            row = dict(
                epoch=epoch,
                seconds=time.perf_counter() - started,
                loss=total / n,
                anchor=anchor_weight,
                temperature=temperature,
                scores=scores,
                selected=selected,
                gain_student=scores[selected]["proxy"] - baseline["proxy"],
                gain_incumbent=scores[selected]["proxy"] - incumbent["proxy"],
            )
            history.append(row)
            print(json.dumps({k: row[k] for k in ("epoch", "loss", "seconds")}), flush=True)
            if scores[selected]["proxy"] > best:
                best = scores[selected]["proxy"]
                chosen = model if selected == "live" else average
                torch.save(
                    dict(
                        state_dict=chosen.state_dict(),
                        channels=196,
                        width=state["width"],
                        mode="joint",
                        strength=1.0,
                        epoch=epoch,
                        scores=scores[selected],
                        objective=args.objective,
                        source=str(checkpoint),
                        source_incumbent=incumbent,
                        validation="eval25",
                        training='train50 with independent offsets',
                        selection=selected,
                    ),
                    out.with_name(out.name + "_best.pt"),
                )
            out.with_suffix(".json").write_text(
                json.dumps(
                    dict(
                        args=vars(args),
                        baseline=baseline,
                        incumbent=incumbent,
                        train_windows=n,
                        history=history,
                    ),
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
