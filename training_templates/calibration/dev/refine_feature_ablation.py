"""Feature ablation."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from refine_interval_coupling import final_proxy, metric_errors, accuracy
from refine_learned_interval import (
    DEV_IDS,
    FEATURE_NAMES,
    ROOT,
    collect_stats,
    feature_tensor,
    load_data,
    new_model,
    solve_coefficients,
    target_raw,
)
import scoring


OUTPUT = ROOT / "dev" / "reports" / "feature_ablation.json"
SUBSETS = {
    "full8": (0, 1, 2, 3, 4, 5, 6, 7),
    "drop_ridge": (0, 1, 3, 4, 5, 6, 7),
    "drop_base": (0, 1, 2, 4, 5, 6, 7),
    "drop_safe": (0, 2, 3, 4, 5, 6, 7),
    "drop_returned": (0, 1, 2, 3, 5, 6, 7),
    "drop_ridge_base": (0, 1, 4, 5, 6, 7),
    "drop_safe_ridge": (0, 3, 4, 5, 6, 7),
    "returned_inputs": (0, 4, 5, 6, 7),
    "safe_inputs": (0, 1, 5, 6, 7),
    "input_only": (0, 5, 6, 7),
}


def fit_loto(stats: dict[str, np.ndarray], subset: tuple[int, ...]) -> np.ndarray:
    xtx = stats["xtx"]
    xty = stats["xty"][:, 1]
    count = stats["count"]
    total_x = xtx.sum(axis=0)
    total_y = xty.sum(axis=0)
    total_n = count.sum(axis=0)
    bank = []
    ix = np.asarray(subset)
    for trajectory in range(xtx.shape[0]):
        matrix = (total_x - xtx[trajectory])[..., ix[:, None], ix]
        vector = (total_y - xty[trajectory])[..., ix]
        beta = solve_coefficients_subset(
            matrix, vector, total_n - count[trajectory], 1.0e-5
        )
        full = np.zeros((*beta.shape[:-1], len(FEATURE_NAMES)), dtype=np.float32)
        full[..., ix] = beta
        bank.append(full)
    return np.stack(bank)


def solve_coefficients_subset(
    xtx: np.ndarray,
    xty: np.ndarray,
    count: np.ndarray,
    ridge: float,
) -> np.ndarray:
    p = xty.shape[-1]
    matrix = xtx.copy()
    diag = np.arange(p)
    matrix[..., diag[1:], diag[1:]] += ridge * np.maximum(count, 1)[..., None]
    matrix[..., diag, diag] += 1.0e-10
    try:
        beta = np.linalg.solve(matrix, xty[..., None])[..., 0]
    except np.linalg.LinAlgError:
        beta = np.einsum("...pq,...q->...p", np.linalg.pinv(matrix), xty)
    return np.nan_to_num(beta).astype(np.float32)


def expand(beta: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(beta, 8, axis=2), 8, axis=3).transpose(0, 2, 3, 1, 4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--progress-every", type=int, default=300)
    args = parser.parse_args()
    start = time.perf_counter()
    data = load_data()
    stats = collect_stats(data, args.progress_every)
    coefficient_banks = {name: fit_loto(stats, subset) for name, subset in SUBSETS.items()}
    names = list(SUBSETS)
    nc, nt = len(names), len(data.trajectory_ids)
    errors = np.zeros((nc, nt, 3), dtype=np.float64)
    numerator = np.zeros((nc, nt), dtype=np.float64)
    covered = np.zeros((nc, nt), dtype=np.int64)
    valid_count = np.zeros(nt, dtype=np.int64)
    windows = np.zeros(nt, dtype=np.int64)
    base, ridge = new_model(data)
    previous_target = None
    previous_ridge = None
    pixel_state = None
    replay_start = time.perf_counter()
    for index in range(data.targets.shape[0]):
        if data.first[index]:
            ridge.reset_ttt_state()
            previous_target = None
            previous_ridge = None
            pixel_state = None
        returned, info = ridge.ttt_step(data.inputs[index].unsqueeze(0), previous_target)
        lower = info["lower"][..., :2].detach().cpu()
        upper = info["upper"][..., :2].detach().cpu()
        center = 0.5 * (lower + upper)
        half = 0.925 * 0.5 * (upper - lower)
        x = feature_tensor(data=data, index=index, model=ridge, returned=returned, center=center).numpy()
        tr = int(data.trajectory_index[index])
        forecasts = []
        for name in names:
            beta = expand(coefficient_banks[name][tr])
            value = np.einsum("thwcp,thwcp->thwc", x, beta, optimize=True)
            forecasts.append(np.clip(value, -0.75, 0.75).astype(np.float32))

        if previous_target is not None and previous_ridge is not None:
            target_phys = previous_target[..., :2] * data.std_target[..., :2]
            target_dev = target_phys - target_phys.mean(dim=1, keepdim=True)
            previous_dev = previous_ridge - previous_ridge.mean(dim=1, keepdim=True)
            old_var = (previous_dev * data.std_target[..., :2]).square().mean(dim=1)[0]
            target_var = target_dev.square().mean(dim=1)[0]
            obs = 0.5 * torch.log((target_var + 1.0e-8) / (old_var + 1.0e-8)).clamp(math.log(0.5), math.log(2.0))
            pixel_state = obs.clone() if pixel_state is None else 0.8 * pixel_state + 0.2 * obs
        ridge_uv = returned.detach().cpu()[..., :2]
        ridge_dev = ridge_uv - ridge_uv.mean(dim=1, keepdim=True)
        ridge_var = (ridge_dev * data.std_target[..., :2]).square().mean(dim=1)[0]
        input_phys = data.inputs[index].unsqueeze(0)[..., :2] * data.std_input[..., :2]
        input_dev = input_phys - input_phys.mean(dim=1, keepdim=True)
        input_var = input_dev.square().mean(dim=1)[0]
        input_log = 0.5 * torch.log((input_var + 1.0e-8) / (ridge_var + 1.0e-8)).clamp(math.log(0.5), math.log(2.0))
        if data.first[index] or pixel_state is None:
            pixel_scale = torch.ones_like(input_log)
        else:
            pixel_scale = torch.exp(0.35 * (0.75 * pixel_state + 0.25 * input_log)).clamp(0.8, 1.25)

        target_raw_full = target_raw(data, index)
        target_raw_np = target_raw_full.numpy()
        valid = (target_raw_full[..., :2] != 0.0).squeeze(0)
        target_error = (data.targets[index].unsqueeze(0)[..., :2] - center).squeeze(0)
        tight = torch.exp(-2.0 * half.squeeze(0) * data.std_target[..., :2].squeeze(0) / float(scoring.SIGMA_GLOBAL))
        valid_count[tr] += int(valid.sum())
        windows[tr] += 1
        for ci, forecast in enumerate(forecasts):
            forecast_t = torch.from_numpy(forecast).unsqueeze(0)
            prediction = returned.detach().cpu().clone()
            prediction[..., :2].add_(0.8 * forecast_t)
            if not data.first[index]:
                uv = prediction[..., :2]
                mean = uv.mean(dim=1, keepdim=True)
                prediction[..., :2] = mean + pixel_scale[None, None] * (uv - mean)
            raw = (prediction * data.std_target + data.mean_target).numpy().astype(np.float32)
            current_errors = metric_errors(raw, target_raw_np)
            errors[ci, tr] += current_errors
            inside = (target_error - torch.from_numpy(forecast)).abs() <= half.squeeze(0)
            numerator[ci, tr] += float((inside * valid * tight * accuracy(current_errors)).sum(dtype=torch.float64))
            covered[ci, tr] += int((inside & valid).sum())
        previous_target = data.targets[index].unsqueeze(0)
        previous_ridge = ridge_uv
        if args.progress_every and (index + 1) % args.progress_every == 0:
            print(f"ablation_windows={index+1} elapsed={time.perf_counter()-replay_start:.1f}s", flush=True)

    dev = np.asarray([name in DEV_IDS for name in data.trajectory_ids])
    angles = np.asarray([int(Path(name).stem.rsplit("_", 1)[1]) for name in data.trajectory_ids])
    selectors = {"global": np.ones(nt, bool), "dev10": dev, "hold20": ~dev, **{f"angle_{a}": angles == a for a in sorted(set(angles))}}
    results = []
    for ci, name in enumerate(names):
        groups = {}
        for group, selector in selectors.items():
            nwin = int(windows[selector].sum()); nvalid = int(valid_count[selector].sum())
            mean_error = errors[ci, selector].sum(axis=0) / nwin
            rel, tke, mvpe = [float(scoring.score_error(v)) for v in mean_error]
            sps = 100.0 * float(numerator[ci, selector].sum()) / nvalid
            groups[group] = {"rel_l2_score": rel, "tke_score": tke, "mvpe_score": mvpe, "sps_score": sps, "coverage": float(covered[ci, selector].sum()) / nvalid, "final_proxy": final_proxy(rel, tke, mvpe, sps)}
        trajectory_proxy = []
        for tr in range(nt):
            e = errors[ci, tr] / windows[tr]
            r, k, m = [float(scoring.score_error(v)) for v in e]
            s = 100.0 * numerator[ci, tr] / valid_count[tr]
            trajectory_proxy.append(final_proxy(r, k, m, s))
        results.append({"name": name, "feature_indices": list(SUBSETS[name]), "feature_names": [FEATURE_NAMES[i] for i in SUBSETS[name]], "groups": groups, "trajectory_proxy": trajectory_proxy})
    baseline = next(x for x in results if x["name"] == "full8")
    for item in results:
        item["proxy_loss_vs_full8"] = baseline["groups"]["global"]["final_proxy"] - item["groups"]["global"]["final_proxy"]
        delta = np.asarray(item["trajectory_proxy"]) - np.asarray(baseline["trajectory_proxy"])
        item["trajectory_vs_full8"] = {"mean": float(delta.mean()), "worst": float(delta.min()), "wins": int((delta > 0).sum()), "losses": int((delta < 0).sum())}
    results.sort(key=lambda x: x["groups"]["global"]["final_proxy"], reverse=True)
    report = {"schema_version": 1, "experiment": "learned feature ablation", "features": list(FEATURE_NAMES), "results": results, "duration_seconds": time.perf_counter() - start}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
