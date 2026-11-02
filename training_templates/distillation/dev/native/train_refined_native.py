"""Native-coordinate refinement."""
import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "dev/aligned"), str(ROOT / "dev/algorithm"), str(ROOT / "dev/shared"), str(ROOT / "submission")]
from train_refiner import features, load
from train_wave import Wave, apply, evaluate
from train_metric_aligned import anchor_loss, reflect
from metrics import record, summary


def frozen(path):
    state = torch.load(path, map_location="cuda", weights_only=False)
    assert state["channels"] == 196 and state["mode"] == "joint"
    model = Wave(196, state["width"], "joint").cuda().to(memory_format=torch.channels_last)
    model.load_state_dict(state["state_dict"])
    with torch.no_grad():
        model.net.out.weight.mul_(float(state.get("strength", 1)))
        model.net.out.bias.mul_(float(state.get("strength", 1)))
    return model.eval().requires_grad_(False), state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "submission_refined/algorithm.pt")
    parser.add_argument("--interval-anchor", type=Path, default=ROOT / "submission_algorithm/algorithm.pt")
    parser.add_argument("--teacher-dir", type=Path, default=ROOT / "dev/refined/cache_refined")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=163)
    parser.add_argument("--flip", action="store_true", help="Matched augmented-data control only")
    parser.add_argument("--output", type=Path, default=ROOT / "dev/native/reports/refined_native")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    point, state = frozen(args.checkpoint)
    teacher, _ = frozen(args.interval_anchor)
    model = copy.deepcopy(point).train().requires_grad_(True)
    average = copy.deepcopy(point)
    split = json.loads((ROOT / "dev/backbone/split.json").read_text())
    assert not set(split["train50"]) & set(split["eval25"])
    paths = [p for name in split["train50"] for p in
             [ROOT / "dev/shared/cache" / f"{Path(name).stem}.pt",
              *[ROOT / "dev/algorithm/cache_dense" / f"{Path(name).stem}_o{o:02d}.pt" for o in (5, 10, 15)]]]
    data = []
    for path in paths:
        d = torch.load(path, map_location="cuda", weights_only=False)
        d["features"], d["scale"] = features(d)
        d["features"] = d["features"].half()
        d["valid"] = d.pop("target") != 0
        del d["input"]
        endpoints = torch.load(args.teacher_dir / path.name, map_location="cuda", weights_only=False)
        d["refined_lower"], d["refined_upper"] = endpoints["lower"], endpoints["upper"]
        data.append(d)
    keys = ("features", "scale", "prediction", "valid", "lower", "upper", "refined_lower", "refined_upper")
    train = {k: torch.cat([d.pop(k) for d in data]) for k in keys}
    del data
    assert len(train["prediction"]) == 8248
    val = load(split["eval25"])
    incumbent = evaluate(point, val, (1.0,))[1.0]
    baseline = summary([record(d["prediction"], d["target"], d["lower"], d["upper"]) for d in val])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    scaler = torch.cuda.amp.GradScaler()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sources = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.checkpoint, args.interval_anchor)}
    report = dict(status="RUNNING", args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  sources=sources, windows=8248, incumbent_refined=incumbent, baseline_student=baseline,
                  training='train50 with independently reset offsets', validation='eval25',
                  interval_teacher="0.5 algorithm + 0.5 refined-density endpoints", history=[])
    print(json.dumps({k: v for k, v in report.items() if k not in ("history", "incumbent_refined", "baseline_student")}), flush=True)
    best, started = incumbent["proxy"], time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for group in opt.param_groups:
            group["lr"] = args.lr * (.1 + .9 * .5 * (1 + math.cos(math.pi * (epoch - 1) / args.epochs)))
        total = 0.0
        order_rng = torch.Generator(device="cuda").manual_seed(args.seed + epoch)
        for ids in torch.randperm(8248, device="cuda", generator=order_rng).split(16):
            f, s, p, valid, lo, hi, rl, rh = [train[k][ids] for k in keys]
            for dim, component in ((-1, 0), (-2, 1)):
                if args.flip and bool(torch.rand((), device="cuda") < .5):
                    f, s, p, _, lo, hi = reflect((f, s, p, torch.zeros_like(p), lo, hi), dim, component)
                    valid = valid.flip(dim)
                    sign = p.new_ones(2)
                    sign[component] = -1
                    a, b = rl.flip(dim) * sign[None, None, :, None, None], rh.flip(dim) * sign[None, None, :, None, None]
                    rl, rh = torch.minimum(a, b), torch.maximum(a, b)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                rpoint, rinterval = point(f), teacher(f)
            with torch.no_grad():
                rq, _, _ = apply(rpoint, p, lo, hi, s, "joint", 1.0)
                _, al, ah = apply(rinterval, p, lo, hi, s, "joint", 1.0)
                tl, th = al + .5 * (rl - al), ah + .5 * (rh - ah)
            with torch.autocast("cuda", dtype=torch.float16):
                raw = model(f)
            q, l, h = apply(raw, p, lo, hi, s, "joint", 1.0)
            loss = anchor_loss(q, l, h, (rq, tl, th), s, lo, hi, valid)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            total += float(loss.detach()) * len(ids)
            with torch.no_grad():
                for avg, param in zip(average.parameters(), model.parameters()):
                    avg.lerp_(param, .005)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            model.eval()
            scores = {k: evaluate(m, val, (1.0,))[1.0] for k, m in (("live", model), ("ema", average))}
            which = max(scores, key=lambda k: scores[k]["proxy"])
            row = dict(epoch=epoch, loss=total / 8248, scores=scores, selection=which,
                       gain_refined=scores[which]["proxy"] - incumbent["proxy"],
                       gain_student=scores[which]["proxy"] - baseline["proxy"], seconds=time.perf_counter() - started)
            report["history"].append(row)
            print(json.dumps({k: row[k] for k in ("epoch", "loss", "selection", "seconds")}), flush=True)
            chosen = model if which == "live" else average
            checkpoint = dict(state_dict=chosen.state_dict(), channels=196, width=state["width"], mode="joint",
                              strength=1.0, epoch=epoch, source_hashes=sources, scores=scores[which], selection=which,
                              objective="native-coordinate distillation with the fixed endpoint mixture")
            if scores[which]["proxy"] > best:
                best = scores[which]["proxy"]
                torch.save(checkpoint, args.output.with_name(args.output.name + "_best.pt"))
            torch.save(checkpoint, args.output.with_name(args.output.name + "_last.pt"))
            report["status"] = "COMPLETE" if epoch == args.epochs else "RUNNING"
            args.output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in sources.items())


if __name__ == "__main__":
    main()
