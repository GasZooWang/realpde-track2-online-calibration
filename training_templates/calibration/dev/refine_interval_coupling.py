"""Interval-prediction coupling."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from refine_learned_interval import (
    DEV_IDS,
    ROOT,
    collect_stats,
    expand_beta,
    feature_tensor,
    fit_and_rank,
    load_data,
    new_model,
    target_raw,
)

import scoring


DEFAULT_OUTPUT = ROOT / "dev" / "reports" / "interval_prediction_coupling.json"
MODEL_KEY = (4, 8, 0.5, 1.0e-5)
CENTER_DAMPINGS = (0.75, 1.00)
PREDICTION_COUPLINGS = (0.00, 0.10, 0.25, 0.50, 0.65, 0.80, 1.00)
WIDTH_SCALE = 0.925
FORECAST_CLIP = 0.75
FINAL_POWER = 0.6854
FINAL_WEIGHTS = (0.5, 0.1, 0.1, 0.1, 0.2)
TIME_SCORE_ASSUMPTION = 84.388239


def final_proxy(rel: float, tke: float, mvpe: float, sps: float) -> float:
    values = (rel, tke, mvpe, TIME_SCORE_ASSUMPTION, sps)
    return float(sum(w * value**FINAL_POWER for w, value in zip(FINAL_WEIGHTS, values)) ** (1.0 / FINAL_POWER))


def metric_errors(pred: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    return (
        float(scoring.rel_l2_per_sample(pred, target, 2)[0]),
        float(scoring.tke_rel_l2_per_sample(pred, target, 2)[0]),
        float(scoring.mvpe_rel_l2_per_sample(pred, target)[0]),
    )


def accuracy(errors: tuple[float, float, float]) -> float:
    factors = [0.5 / (0.5 + max(value, 0.0)) for value in errors]
    return float(0.5 * factors[0] + 0.3 * factors[1] + 0.2 * factors[2])


def summarize(
    errors: np.ndarray,
    sps_num: np.ndarray,
    covered: np.ndarray,
    valid: np.ndarray,
    windows: np.ndarray,
    selector: np.ndarray,
) -> dict[str, float | int]:
    nwin = int(windows[selector].sum())
    nvalid = int(valid[selector].sum())
    mean_errors = errors[selector].sum(axis=0) / nwin
    rel, tke, mvpe = [float(scoring.score_error(value)) for value in mean_errors]
    sps = 100.0 * float(sps_num[selector].sum()) / nvalid
    return {
        "trajectories": int(selector.sum()),
        "windows": nwin,
        "rel_l2_score": rel,
        "tke_score": tke,
        "mvpe_score": mvpe,
        "sps_score": sps,
        "coverage": float(covered[selector].sum()) / nvalid,
        "final_proxy": final_proxy(rel, tke, mvpe, sps),
        "valid_elements": nvalid,
    }


def delta_dict(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    keys = ("rel_l2_score", "tke_score", "mvpe_score", "sps_score", "coverage", "final_proxy")
    return {key: float(candidate[key] - baseline[key]) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    started = time.perf_counter()
    data = load_data()
    stats = collect_stats(data, args.progress_every)
    _selected, coefficients = fit_and_rank(stats, data.trajectory_ids)
    if MODEL_KEY not in coefficients:
        raise RuntimeError(f"LOTO model coefficients missing: {MODEL_KEY}")

    configs: list[tuple[str, float, float, bool]] = [("ridge_exact", 0.0, 0.0, False)]
    configs.extend(
        (f"center{d:g}_pred{p:g}", d, p, False)
        for d in CENTER_DAMPINGS
        for p in PREDICTION_COUPLINGS
    )
    configs.extend(
        [
            ("pixel_tke_only", 0.0, 0.0, True),
            ("joint_center1_pred0.8_pixel_tke", 1.0, 0.8, True),
            ("joint_center0.75_pred0.8_pixel_tke", 0.75, 0.8, True),
            ("joint_center1_pred0.65_pixel_tke", 1.0, 0.65, True),
        ]
    )
    nc = len(configs)
    nt = len(data.trajectory_ids)
    errors = np.zeros((nc, nt, 3), dtype=np.float64)
    sps_num = np.zeros((nc, nt), dtype=np.float64)
    covered = np.zeros((nc, nt), dtype=np.int64)
    valid_count = np.zeros(nt, dtype=np.int64)
    windows = np.zeros(nt, dtype=np.int64)
    base, ridge = new_model(data)
    previous_target = None
    previous_ridge_returned: torch.Tensor | None = None
    pixel_log_ratio_state: torch.Tensor | None = None
    replay_start = time.perf_counter()
    for index in range(data.targets.shape[0]):
        if data.first[index]:
            ridge.reset_ttt_state()
            previous_target = None
            previous_ridge_returned = None
            pixel_log_ratio_state = None
        returned, info = ridge.ttt_step(data.inputs[index].unsqueeze(0), previous_target)
        lower = info["lower"][..., :2].detach().cpu()
        upper = info["upper"][..., :2].detach().cpu()
        center = 0.5 * (lower + upper)
        half = 0.5 * (upper - lower)
        x = feature_tensor(data=data, index=index, model=ridge, returned=returned, center=center).numpy()
        tr = int(data.trajectory_index[index])
        beta = coefficients[MODEL_KEY][tr]
        forecast = np.einsum("thwcp,thwcp->thwc", x, expand_beta(beta), optimize=True)
        forecast = np.clip(forecast, -FORECAST_CLIP, FORECAST_CLIP).astype(np.float32)
        forecast_t = torch.from_numpy(forecast).unsqueeze(0)
        if previous_target is not None and previous_ridge_returned is not None:
            previous_target_physical = previous_target[..., :2] * data.std_target[..., :2]
            previous_target_dev = previous_target_physical - previous_target_physical.mean(dim=1, keepdim=True)
            previous_prediction_dev = previous_ridge_returned - previous_ridge_returned.mean(dim=1, keepdim=True)
            previous_prediction_var = (previous_prediction_dev * data.std_target[..., :2]).square().mean(dim=1)[0]
            previous_target_var = previous_target_dev.square().mean(dim=1)[0]
            observed_log_ratio = 0.5 * torch.log(
                (previous_target_var + 1.0e-8) / (previous_prediction_var + 1.0e-8)
            ).clamp(math.log(0.5), math.log(2.0))
            if pixel_log_ratio_state is None:
                pixel_log_ratio_state = observed_log_ratio.clone()
            else:
                pixel_log_ratio_state.mul_(0.8).add_(observed_log_ratio, alpha=0.2)
        returned_uv = returned.detach().cpu()[..., :2]
        returned_dev = returned_uv - returned_uv.mean(dim=1, keepdim=True)
        current_input_physical = (
            data.inputs[index].unsqueeze(0)[..., :2] * data.std_input[..., :2]
        )
        current_input_dev = current_input_physical - current_input_physical.mean(dim=1, keepdim=True)
        returned_var = (returned_dev * data.std_target[..., :2]).square().mean(dim=1)[0]
        current_input_var = current_input_dev.square().mean(dim=1)[0]
        current_input_log_ratio = 0.5 * torch.log(
            (current_input_var + 1.0e-8) / (returned_var + 1.0e-8)
        ).clamp(math.log(0.5), math.log(2.0))
        if data.first[index] or pixel_log_ratio_state is None:
            pixel_scale = torch.ones_like(current_input_log_ratio)
        else:
            blended_log_ratio = 0.75 * pixel_log_ratio_state + 0.25 * current_input_log_ratio
            pixel_scale = torch.exp(0.35 * blended_log_ratio).clamp(0.8, 1.25)
        target_raw_full = target_raw(data, index)
        target_raw_np = target_raw_full.numpy()
        valid = (target_raw_full[..., :2] != 0.0).squeeze(0)
        target_norm_uv = data.targets[index].unsqueeze(0)[..., :2]
        target_center_error = (target_norm_uv - center).squeeze(0)
        half0 = half.squeeze(0)
        windows[tr] += 1
        valid_count[tr] += int(valid.sum())

        for ci, (_name, center_damping, prediction_coupling, use_pixel_tke) in enumerate(configs):
            candidate_pred = returned.detach().cpu().clone()
            if prediction_coupling != 0.0:
                candidate_pred[..., :2].add_(float(prediction_coupling) * forecast_t)
            if use_pixel_tke and not data.first[index]:
                candidate_uv = candidate_pred[..., :2]
                candidate_mean = candidate_uv.mean(dim=1, keepdim=True)
                candidate_pred[..., :2] = candidate_mean + pixel_scale.unsqueeze(0).unsqueeze(0) * (
                    candidate_uv - candidate_mean
                )
            candidate_raw = (candidate_pred * data.std_target + data.mean_target).numpy().astype(np.float32)
            current_errors = metric_errors(candidate_raw, target_raw_np)
            errors[ci, tr] += current_errors
            current_accuracy = accuracy(current_errors)
            if center_damping == 0.0:
                candidate_half = half0
                total_center_shift = torch.zeros_like(half0)
            else:
                candidate_half = WIDTH_SCALE * half0
                total_damping = prediction_coupling + center_damping * (1.0 - prediction_coupling)
                total_center_shift = float(total_damping) * forecast_t.squeeze(0)
            inside = (target_center_error - total_center_shift).abs() <= candidate_half
            tight = torch.exp(
                -2.0 * candidate_half * data.std_target[..., :2].squeeze(0) / float(scoring.SIGMA_GLOBAL)
            )
            sps_num[ci, tr] += float((inside * valid * tight * current_accuracy).sum(dtype=torch.float64))
            covered[ci, tr] += int((inside & valid).sum())
        previous_target = data.targets[index].unsqueeze(0)
        previous_ridge_returned = returned_uv
        if args.progress_every and (index + 1) % args.progress_every == 0:
            print(f"coupling_windows={index+1} elapsed={time.perf_counter()-replay_start:.1f}s", flush=True)
    if base.index != data.targets.shape[0]:
        raise RuntimeError("coupling replay did not consume full cache")

    names = np.asarray(data.trajectory_ids)
    dev = np.asarray([name in DEV_IDS for name in names])
    angles = np.asarray([int(name.removesuffix(".h5").rsplit("_", 1)[1]) for name in names])
    selectors = {
        "global": np.ones(nt, dtype=bool),
        "dev10": dev,
        "hold20": ~dev,
        **{f"angle_{a}": angles == a for a in sorted(set(angles))},
    }
    baseline_groups = {
        name: summarize(errors[0], sps_num[0], covered[0], valid_count, windows, selector)
        for name, selector in selectors.items()
    }
    baseline_tr = [
        summarize(errors[0], sps_num[0], covered[0], valid_count, windows, np.arange(nt) == tr)
        for tr in range(nt)
    ]
    results = []
    for ci, (name, center_damping, prediction_coupling, use_pixel_tke) in enumerate(configs):
        groups = {
            group_name: summarize(errors[ci], sps_num[ci], covered[ci], valid_count, windows, selector)
            for group_name, selector in selectors.items()
        }
        trajectory = [
            summarize(errors[ci], sps_num[ci], covered[ci], valid_count, windows, np.arange(nt) == tr)
            for tr in range(nt)
        ]
        proxy_delta = np.asarray([trajectory[tr]["final_proxy"] - baseline_tr[tr]["final_proxy"] for tr in range(nt)])
        item = {
            "name": name,
            "center_damping": center_damping,
            "prediction_coupling": prediction_coupling,
            "pixel_tke": use_pixel_tke,
            "bounds_total_damping": prediction_coupling + center_damping * (1.0 - prediction_coupling) if center_damping else 0.0,
            "width_scale": WIDTH_SCALE if center_damping else 1.0,
            "groups": groups,
            "delta": {group_name: delta_dict(groups[group_name], baseline_groups[group_name]) for group_name in groups},
            "robustness": {
                "trajectory_proxy_mean_delta": float(proxy_delta.mean()),
                "trajectory_proxy_std_delta": float(proxy_delta.std()),
                "trajectory_proxy_robust_050": float(proxy_delta.mean() - 0.5 * proxy_delta.std()),
                "trajectory_proxy_worst_delta": float(proxy_delta.min()),
                "trajectory_proxy_wins": int((proxy_delta > 0).sum()),
                "trajectory_proxy_losses": int((proxy_delta < 0).sum()),
            },
            "trajectory_proxy_delta": {str(names[tr]): float(proxy_delta[tr]) for tr in range(nt)},
        }
        results.append(item)
    results.sort(key=lambda item: item["delta"]["global"]["final_proxy"], reverse=True)
    report = {
        "schema_version": 1,
        "experiment": "interval forecast to prediction coupling",
        "status": "completed",
        "data": {"windows": int(windows.sum()), "trajectories": nt, "source": "official public train_real cache"},
        "method": {
            "model_key": list(MODEL_KEY),
            "crossfit": "leave-one-trajectory-out",
            "prediction_couplings": list(PREDICTION_COUPLINGS),
            "center_dampings": list(CENTER_DAMPINGS),
            "bounds_shift": "p + d*(1-p), so bounds follow prediction shift and center remaining forecast",
            "width_scale": WIDTH_SCALE,
            "time_score_assumption": TIME_SCORE_ASSUMPTION,
        },
        "baseline_ridge": baseline_groups,
        "ranked_candidates": results,
        "duration_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"coupling_report={args.output}", flush=True)


if __name__ == "__main__":
    main()
