"""Joint hyperparameter search."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from refine_interval_coupling import (
    FINAL_POWER,
    FINAL_WEIGHTS,
    MODEL_KEY,
    TIME_SCORE_ASSUMPTION,
)
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
from refine_feature_ablation import fit_loto

import scoring


DEFAULT_OUTPUT = ROOT / "dev" / "reports" / "joint_hypergrid.json"
DEFAULT_STATS_CACHE = ROOT / "dev" / "cache" / "sufficient_stats.pt"
FORECAST_CLIP = 0.75
PREDICTION_COUPLINGS = (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2)
PIXEL_PREVIOUS_WEIGHTS = (0.5, 0.75, 1.0)
PIXEL_DAMPINGS = (0.2, 0.35, 0.5, 0.65)
BOUND_DAMPINGS = (0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)
WIDTH_SCALES = (0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1.0, 1.025)


def final_proxy(rel: np.ndarray, tke: np.ndarray, mvpe: np.ndarray, sps: np.ndarray) -> np.ndarray:
    total = (
        FINAL_WEIGHTS[0] * rel**FINAL_POWER
        + FINAL_WEIGHTS[1] * tke**FINAL_POWER
        + FINAL_WEIGHTS[2] * mvpe**FINAL_POWER
        + FINAL_WEIGHTS[3] * TIME_SCORE_ASSUMPTION**FINAL_POWER
        + FINAL_WEIGHTS[4] * sps**FINAL_POWER
    )
    return total ** (1.0 / FINAL_POWER)


def score_from_error(error: np.ndarray) -> np.ndarray:
    return 100.0 / (1.0 + 0.5 * np.maximum(error, 0.0))


def vector_errors(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    b = pred.shape[0]
    diff = (pred[..., :2] - target[..., :2]).reshape(b, -1)
    target_uv = target[..., :2].reshape(1, -1)
    rel = torch.linalg.vector_norm(diff, dim=1) / torch.linalg.vector_norm(target_uv, dim=1).clamp_min(1.0e-8)

    pred_uv = pred[..., :2]
    target_uv5 = target[..., :2]
    pred_dev = pred_uv - pred_uv.mean(dim=1, keepdim=True)
    target_dev = target_uv5 - target_uv5.mean(dim=1, keepdim=True)
    pred_ke = 0.5 * pred_dev.square().mean(dim=1).sum(dim=-1)
    target_ke = 0.5 * target_dev.square().mean(dim=1).sum(dim=-1)
    tke = torch.linalg.vector_norm((pred_ke - target_ke).reshape(b, -1), dim=1)
    tke /= torch.linalg.vector_norm(target_ke.reshape(1, -1), dim=1).clamp_min(1.0e-8)

    probe_center_y = 16
    probe_y = [probe_center_y + 2 * j for j in range(-4, 5)]
    mvpe_parts = []
    for probe_x in (13, 21, 29, 37):
        pp = pred[:, :, probe_y, probe_x, :2].mean(dim=1).reshape(b, -1)
        tt = target[:, :, probe_y, probe_x, :2].mean(dim=1).reshape(1, -1)
        mvpe_parts.append(
            torch.linalg.vector_norm(pp - tt, dim=1)
            / torch.linalg.vector_norm(tt, dim=1).clamp_min(1.0e-8)
        )
    mvpe = torch.stack(mvpe_parts).mean(dim=0)
    return torch.stack((rel, tke, mvpe), dim=1)


def make_prediction_configs(
    couplings: tuple[float, ...] = PREDICTION_COUPLINGS,
    previous_weights: tuple[float, ...] = PIXEL_PREVIOUS_WEIGHTS,
    pixel_dampings: tuple[float, ...] = PIXEL_DAMPINGS,
) -> list[dict[str, float]]:
    pixel = [(0.0, 0.0)] + [
        (previous_weight, damping)
        for previous_weight in previous_weights
        for damping in pixel_dampings
    ]
    return [
        {"prediction_coupling": p, "pixel_previous_weight": q, "pixel_damping": d}
        for p in couplings
        for q, d in pixel
    ]


def make_interval_configs(
    bound_dampings: tuple[float, ...] = BOUND_DAMPINGS,
    width_scales: tuple[float, ...] = WIDTH_SCALES,
) -> list[dict[str, float]]:
    return [{"bounds_total_damping": 0.0, "width_scale": 1.0}] + [
        {"bounds_total_damping": damping, "width_scale": width}
        for damping in bound_dampings
        for width in width_scales
    ]


def group_values(
    error_sum: np.ndarray,
    sps_num: np.ndarray,
    windows: np.ndarray,
    valid: np.ndarray,
    selector: np.ndarray,
) -> dict[str, np.ndarray]:
    nwin = int(windows[selector].sum())
    nvalid = int(valid[selector].sum())
    scores = score_from_error(error_sum[:, selector].sum(axis=1) / nwin)
    sps = 100.0 * sps_num[:, :, selector].sum(axis=2) / nvalid
    rel, tke, mvpe = scores[:, 0, None], scores[:, 1, None], scores[:, 2, None]
    return {
        "rel_l2_score": np.broadcast_to(rel, sps.shape),
        "tke_score": np.broadcast_to(tke, sps.shape),
        "mvpe_score": np.broadcast_to(mvpe, sps.shape),
        "sps_score": sps,
        "final_proxy": final_proxy(rel, tke, mvpe, sps),
    }


def scalar_item(values: dict[str, np.ndarray], pi: int, ii: int) -> dict[str, float]:
    return {key: float(value[pi, ii]) for key, value in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--stats-cache", type=Path, default=DEFAULT_STATS_CACHE)
    parser.add_argument("--forecast-subset", choices=("full8", "input_only", "blend"), default="full8")
    parser.add_argument("--input-only-weight", type=float, default=0.5,
                        help="input-only forecast weight when --forecast-subset=blend")
    parser.add_argument("--focused", action="store_true", help="scan the requested local neighborhood")
    parser.add_argument("--boundary", action="store_true", help="extend the input-only optimum at the p/bounds upper edge")
    parser.add_argument("--prediction-edge", action="store_true", help="confirm the upper prediction-coupling peak")
    args = parser.parse_args()
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    started = time.perf_counter()
    data = load_data()
    if args.stats_cache.exists():
        stats = torch.load(args.stats_cache, map_location="cpu", weights_only=False)
        print(f"distribution_loaded_stats_cache={args.stats_cache}", flush=True)
    else:
        stats = collect_stats(data, args.progress_every)
        args.stats_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(stats, args.stats_cache)
        print(f"saved_stats_cache={args.stats_cache}", flush=True)
    if args.forecast_subset == "full8":
        _selected, coefficients = fit_and_rank(stats, data.trajectory_ids)
        beta_bank = coefficients[MODEL_KEY]
    elif args.forecast_subset == "input_only":
        beta_bank = fit_loto(stats, (0, 5, 6, 7))
    else:
        _selected, coefficients = fit_and_rank(stats, data.trajectory_ids)
        full_bank = coefficients[MODEL_KEY]
        input_bank = fit_loto(stats, (0, 5, 6, 7))
        beta_bank = (1.0 - args.input_only_weight) * full_bank + args.input_only_weight * input_bank

    if args.prediction_edge:
        search_couplings = (0.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6)
        search_previous_weights = (0.5,)
        search_pixel_dampings = (0.35,)
        search_bound_dampings = (1.1, 1.2, 1.3, 1.4)
        search_width_scales = (0.9, 0.925, 0.95)
    elif args.boundary:
        search_couplings = (0.0, 0.9, 1.0, 1.1, 1.2)
        search_previous_weights = (0.5, 0.75)
        search_pixel_dampings = (0.35,)
        search_bound_dampings = (1.1, 1.2, 1.3, 1.4, 1.5)
        search_width_scales = (0.9, 0.925, 0.95, 0.975)
    elif args.focused:
        search_couplings = (0.0, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)
        search_previous_weights = (0.25, 0.5, 0.75)
        search_pixel_dampings = (0.25, 0.35, 0.5)
        search_bound_dampings = (0.9, 1.0, 1.1)
        search_width_scales = (0.9, 0.925, 0.95)
    else:
        search_couplings = PREDICTION_COUPLINGS
        search_previous_weights = PIXEL_PREVIOUS_WEIGHTS
        search_pixel_dampings = PIXEL_DAMPINGS
        search_bound_dampings = BOUND_DAMPINGS
        search_width_scales = WIDTH_SCALES
    pred_configs = make_prediction_configs(search_couplings, search_previous_weights, search_pixel_dampings)
    interval_configs = make_interval_configs(search_bound_dampings, search_width_scales)
    npred, ninterval, ntraj = len(pred_configs), len(interval_configs), len(data.trajectory_ids)
    error_sum = np.zeros((npred, ntraj, 3), dtype=np.float64)
    sps_num = np.zeros((npred, ninterval, ntraj), dtype=np.float64)
    covered = np.zeros((ninterval, ntraj), dtype=np.int64)
    windows = np.zeros(ntraj, dtype=np.int64)
    valid_count = np.zeros(ntraj, dtype=np.int64)

    p_values = torch.tensor([x["prediction_coupling"] for x in pred_configs], dtype=torch.float32)
    q_values = torch.tensor([x["pixel_previous_weight"] for x in pred_configs], dtype=torch.float32)
    pd_values = torch.tensor([x["pixel_damping"] for x in pred_configs], dtype=torch.float32)
    bound_values = torch.tensor([x["bounds_total_damping"] for x in interval_configs], dtype=torch.float32)
    width_values = torch.tensor([x["width_scale"] for x in interval_configs], dtype=torch.float32)

    base, ridge = new_model(data)
    previous_target = None
    previous_ridge_returned = None
    pixel_log_ratio_state = None
    replay_started = time.perf_counter()
    for index in range(data.targets.shape[0]):
        if data.first[index]:
            ridge.reset_ttt_state()
            previous_target = None
            previous_ridge_returned = None
            pixel_log_ratio_state = None
        returned, info = ridge.ttt_step(data.inputs[index].unsqueeze(0), previous_target)
        returned = returned.detach().cpu()
        lower = info["lower"][..., :2].detach().cpu()
        upper = info["upper"][..., :2].detach().cpu()
        center = 0.5 * (lower + upper)
        half = 0.5 * (upper - lower)
        features = feature_tensor(data=data, index=index, model=ridge, returned=returned, center=center).numpy()
        tr = int(data.trajectory_index[index])
        forecast = np.einsum(
            "thwcp,thwcp->thwc", features, expand_beta(beta_bank[tr]), optimize=True
        )
        forecast_t = torch.from_numpy(np.clip(forecast, -FORECAST_CLIP, FORECAST_CLIP).astype(np.float32)).unsqueeze(0)

        if previous_target is not None and previous_ridge_returned is not None:
            prev_target_physical = previous_target[..., :2] * data.std_target[..., :2]
            prev_target_dev = prev_target_physical - prev_target_physical.mean(dim=1, keepdim=True)
            prev_prediction_dev = previous_ridge_returned - previous_ridge_returned.mean(dim=1, keepdim=True)
            prev_prediction_var = (prev_prediction_dev * data.std_target[..., :2]).square().mean(dim=1)[0]
            prev_target_var = prev_target_dev.square().mean(dim=1)[0]
            observed = 0.5 * torch.log((prev_target_var + 1.0e-8) / (prev_prediction_var + 1.0e-8))
            observed = observed.clamp(math.log(0.5), math.log(2.0))
            if pixel_log_ratio_state is None:
                pixel_log_ratio_state = observed.clone()
            else:
                pixel_log_ratio_state.mul_(0.8).add_(observed, alpha=0.2)

        returned_uv = returned[..., :2]
        returned_dev = returned_uv - returned_uv.mean(dim=1, keepdim=True)
        current_input_physical = data.inputs[index].unsqueeze(0)[..., :2] * data.std_input[..., :2]
        current_input_dev = current_input_physical - current_input_physical.mean(dim=1, keepdim=True)
        returned_var = (returned_dev * data.std_target[..., :2]).square().mean(dim=1)[0]
        current_input_var = current_input_dev.square().mean(dim=1)[0]
        current_ratio = 0.5 * torch.log((current_input_var + 1.0e-8) / (returned_var + 1.0e-8))
        current_ratio = current_ratio.clamp(math.log(0.5), math.log(2.0))

        target = target_raw(data, index)
        valid = (target[..., :2] != 0.0).squeeze(0)
        target_error = (data.targets[index].unsqueeze(0)[..., :2] - center).squeeze(0)
        half0 = half.squeeze(0)
        interval_shift = bound_values[:, None, None, None, None] * forecast_t
        interval_half = width_values[:, None, None, None, None] * half0.unsqueeze(0)
        inside = (target_error.unsqueeze(0) - interval_shift).abs() <= interval_half
        tight = torch.exp(
            -2.0 * interval_half * data.std_target[..., :2].squeeze(0).unsqueeze(0)
            / float(scoring.SIGMA_GLOBAL)
        )
        raw_interval_reward = (inside * valid.unsqueeze(0) * tight).sum(dim=(1, 2, 3, 4), dtype=torch.float64)
        covered[:, tr] += (inside & valid.unsqueeze(0)).sum(dim=(1, 2, 3, 4), dtype=torch.int64).numpy()

        accuracy_bank = torch.empty(npred, dtype=torch.float64)
        for start in range(0, npred, args.batch_size):
            stop = min(start + args.batch_size, npred)
            p = p_values[start:stop, None, None, None, None]
            candidate_uv = returned_uv + p * forecast_t
            damping = pd_values[start:stop]
            if not data.first[index] and pixel_log_ratio_state is not None:
                q = q_values[start:stop, None, None, None]
                blended = q * pixel_log_ratio_state.unsqueeze(0) + (1.0 - q) * current_ratio.unsqueeze(0)
                scale = torch.exp(pd_values[start:stop, None, None, None] * blended).clamp(0.8, 1.25)
                candidate_mean = candidate_uv.mean(dim=1, keepdim=True)
                candidate_uv = candidate_mean + scale.unsqueeze(1) * (candidate_uv - candidate_mean)
            candidate = returned.expand(stop - start, -1, -1, -1, -1).clone()
            candidate[..., :2] = candidate_uv
            candidate_raw = candidate * data.std_target + data.mean_target
            batch_errors = vector_errors(candidate_raw, target)
            error_sum[start:stop, tr] += batch_errors.numpy().astype(np.float64)
            factors = 0.5 / (0.5 + batch_errors)
            accuracy_bank[start:stop] = 0.5 * factors[:, 0] + 0.3 * factors[:, 1] + 0.2 * factors[:, 2]
        sps_num[:, :, tr] += torch.outer(accuracy_bank, raw_interval_reward).numpy()
        windows[tr] += 1
        valid_count[tr] += int(valid.sum())
        previous_target = data.targets[index].unsqueeze(0)
        previous_ridge_returned = returned_uv
        if args.progress_every and (index + 1) % args.progress_every == 0:
            print(f"joint_windows={index+1} elapsed={time.perf_counter()-replay_started:.1f}s", flush=True)

    selectors = {"global": np.ones(ntraj, dtype=bool)}
    names = np.asarray(data.trajectory_ids)
    dev = np.asarray([name in DEV_IDS for name in names])
    angles = np.asarray([int(name.removesuffix(".h5").rsplit("_", 1)[1]) for name in names])
    selectors["dev10"] = dev
    selectors["hold20"] = ~dev
    selectors.update({f"angle_{a}": angles == a for a in sorted(set(angles))})
    groups = {
        name: group_values(error_sum, sps_num, windows, valid_count, selector)
        for name, selector in selectors.items()
    }

    baseline_pi = pred_configs.index({"prediction_coupling": 0.0, "pixel_previous_weight": 0.0, "pixel_damping": 0.0})
    baseline_ii = interval_configs.index({"bounds_total_damping": 0.0, "width_scale": 1.0})
    baseline = {name: scalar_item(values, baseline_pi, baseline_ii) for name, values in groups.items()}
    trajectory_selector = [np.arange(ntraj) == tr for tr in range(ntraj)]
    trajectory_values = [group_values(error_sum, sps_num, windows, valid_count, selector) for selector in trajectory_selector]
    baseline_tr = np.asarray([
        trajectory_values[tr]["final_proxy"][baseline_pi, baseline_ii] for tr in range(ntraj)
    ])
    global_delta = groups["global"]["final_proxy"] - baseline["global"]["final_proxy"]
    flat_order = np.argsort(global_delta.ravel())[::-1]

    def candidate_item(flat_index: int) -> dict[str, Any]:
        pi, ii = np.unravel_index(int(flat_index), global_delta.shape)
        trajectory_delta = np.asarray([
            trajectory_values[tr]["final_proxy"][pi, ii] - baseline_tr[tr] for tr in range(ntraj)
        ])
        group_items = {name: scalar_item(values, pi, ii) for name, values in groups.items()}
        delta_items = {
            name: {key: float(value - baseline[name][key]) for key, value in item.items()}
            for name, item in group_items.items()
        }
        return {
            "prediction": pred_configs[pi],
            "interval": interval_configs[ii],
            "groups": group_items,
            "delta": delta_items,
            "robustness": {
                "trajectory_mean_delta": float(trajectory_delta.mean()),
                "trajectory_std_delta": float(trajectory_delta.std()),
                "trajectory_robust_050": float(trajectory_delta.mean() - 0.5 * trajectory_delta.std()),
                "trajectory_worst_delta": float(trajectory_delta.min()),
                "trajectory_wins": int((trajectory_delta > 0).sum()),
                "trajectory_losses": int((trajectory_delta < 0).sum()),
                "all_angles_positive": bool(all(delta_items[f"angle_{a}"]["final_proxy"] > 0 for a in sorted(set(angles)))),
            },
            "trajectory_delta": {str(name): float(value) for name, value in zip(names, trajectory_delta)},
        }

    top_global = [candidate_item(i) for i in flat_order[:100]]
    robust_items = []
    for flat_index in flat_order:
        item = candidate_item(int(flat_index))
        if item["robustness"]["trajectory_losses"] == 0 and item["robustness"]["all_angles_positive"]:
            robust_items.append(item)
        if len(robust_items) >= 100:
            break

    report = {
        "schema_version": 1,
        "experiment": "joint forecast/pixel/interval grid",
        "status": "completed",
        "causality": {
            "crossfit": "leave-one-trajectory-out",
        },
        "data": {"windows": int(windows.sum()), "trajectories": ntraj, "trajectory_ids": list(data.trajectory_ids)},
        "search": {
            "forecast_subset": args.forecast_subset,
            "input_only_weight": args.input_only_weight if args.forecast_subset == "blend" else None,
            "forecast_model_key": list(MODEL_KEY),
            "prediction_couplings": list(search_couplings),
            "pixel_previous_weights": list(search_previous_weights),
            "pixel_dampings": list(search_pixel_dampings),
            "bounds_total_dampings": list(search_bound_dampings),
            "width_scales": list(search_width_scales),
            "prediction_config_count": npred,
            "interval_config_count": ninterval,
            "joint_candidate_count": npred * ninterval,
            "time_score_assumption": TIME_SCORE_ASSUMPTION,
        },
        "baseline_ridge": baseline,
        "top_global": top_global,
        "top_all_trajectory_and_angle_robust": robust_items,
        "duration_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"joint_report={args.output}", flush=True)


if __name__ == "__main__":
    main()
