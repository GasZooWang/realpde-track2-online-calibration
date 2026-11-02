"""Interval width map."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from refine_feature_ablation import fit_loto
from refine_learned_interval import (
    DEV_IDS,
    ROOT,
    feature_tensor,
    load_data,
    new_model,
    target_raw,
)
from refine_joint_grid import (
    DEFAULT_STATS_CACHE,
    score_from_error,
    vector_errors,
)

import scoring


DEFAULT_OUTPUT = ROOT / "dev" / "reports" / "loto_width_map.json"
DEFAULT_DEPLOY = ROOT / "dev" / "cache" / "loto_width_map.pt"
BASE_WIDTH = 0.95
PREDICTION_COUPLING = 1.30
BOUNDS_DAMPING = 1.20
PIXEL_PREVIOUS_WEIGHT = 0.50
PIXEL_STATE_DECAY = 0.80
PIXEL_DAMPING = 0.35
FORECAST_CLIP = 0.75
TIME_SCORE = 85.066771
WIDTH_GRID = np.round(np.arange(0.60, 1.3001, 0.0125), 4)
TIME_BINS = (1, 4, 20)
SPATIAL_PARTITIONS = ((1, 1), (2, 4), (4, 8))
SHRINKAGES = (0.25, 0.50, 0.75, 1.00)


def final_proxy(
    rel: float | np.ndarray,
    tke: float | np.ndarray,
    mvpe: float | np.ndarray,
    sps: float | np.ndarray,
) -> float | np.ndarray:
    power = 0.6854
    total = (
        0.5 * np.asarray(rel) ** power
        + 0.1 * np.asarray(tke) ** power
        + 0.1 * np.asarray(mvpe) ** power
        + 0.1 * TIME_SCORE**power
        + 0.2 * np.asarray(sps) ** power
    )
    return total ** (1.0 / power)


def expand_beta(beta: np.ndarray) -> np.ndarray:

    return np.repeat(np.repeat(beta, 8, axis=2), 8, axis=3).transpose(
        0, 2, 3, 1, 4
    )


def aggregate_groups(
    values: np.ndarray,
    time_bins: int,
    y_bins: int,
    x_bins: int,
) -> np.ndarray:

    nt, time, channels, fine_y, fine_x, candidates = values.shape
    if time % time_bins or fine_y % y_bins or fine_x % x_bins:
        raise ValueError("grouping must divide the fine statistic grid")
    reshaped = values.reshape(
        nt,
        time_bins,
        time // time_bins,
        channels,
        y_bins,
        fine_y // y_bins,
        x_bins,
        fine_x // x_bins,
        candidates,
    )
    return reshaped.sum(axis=(2, 5, 7))


def nearest_grid_indices(values: np.ndarray) -> np.ndarray:
    distance = np.abs(values[..., None] - WIDTH_GRID)
    return distance.argmin(axis=-1)


def expand_group_width(
    grouped: np.ndarray,
    time_bins: int,
    y_bins: int,
    x_bins: int,
) -> np.ndarray:

    result = grouped.repeat(20 // time_bins, axis=0)
    result = result.repeat(4 // y_bins, axis=2)
    result = result.repeat(8 // x_bins, axis=3)
    if result.shape != (20, 2, 4, 8):
        raise RuntimeError(f"unexpected expanded width shape {result.shape}")
    return result


def group_scores(
    error_sum: np.ndarray,
    reward_sum: np.ndarray,
    coverage_count: np.ndarray,
    windows: np.ndarray,
    valid_count: np.ndarray,
    selector: np.ndarray,
) -> dict[str, float | int]:
    nwin = int(windows[selector].sum())
    nvalid = int(valid_count[selector].sum())
    errors = error_sum[selector].sum(axis=0) / max(nwin, 1)
    rel, tke, mvpe = score_from_error(errors)
    sps = 100.0 * float(reward_sum[selector].sum()) / max(nvalid, 1)
    return {
        "trajectories": int(selector.sum()),
        "windows": nwin,
        "valid_elements": nvalid,
        "rel_l2_score": float(rel),
        "tke_score": float(tke),
        "mvpe_score": float(mvpe),
        "sps_score": sps,
        "coverage": float(coverage_count[selector].sum()) / max(nvalid, 1),
        "final_proxy": float(final_proxy(rel, tke, mvpe, sps)),
    }


def subtract(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    keys = (
        "rel_l2_score",
        "tke_score",
        "mvpe_score",
        "sps_score",
        "coverage",
        "final_proxy",
    )
    return {key: float(a[key] - b[key]) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--deploy-output", type=Path, default=DEFAULT_DEPLOY)
    parser.add_argument("--stats-cache", type=Path, default=DEFAULT_STATS_CACHE)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))

    data = load_data()
    if not args.stats_cache.exists():
        raise FileNotFoundError(
            f"missing reference sufficient statistics: {args.stats_cache}"
        )
    stats = torch.load(args.stats_cache, map_location="cpu", weights_only=False)
    beta_bank = fit_loto(stats, (0, 5, 6, 7))
    ntraj = len(data.trajectory_ids)
    nwidth = len(WIDTH_GRID)

    fine_reward = np.zeros(
        (ntraj, 20, 2, 4, 8, nwidth), dtype=np.float64
    )
    fine_covered = np.zeros_like(fine_reward, dtype=np.int64)
    error_sum = np.zeros((ntraj, 3), dtype=np.float64)
    windows = np.zeros(ntraj, dtype=np.int64)
    valid_count = np.zeros(ntraj, dtype=np.int64)

    width_values = torch.tensor(WIDTH_GRID, dtype=torch.float32).reshape(
        nwidth, 1, 1, 1, 1
    )
    base, ridge = new_model(data)
    previous_target = None
    previous_ridge_returned = None
    pixel_state = None
    replay_started = time.perf_counter()

    for index in range(data.targets.shape[0]):
        if data.first[index]:
            ridge.reset_ttt_state()
            previous_target = None
            previous_ridge_returned = None
            pixel_state = None

        returned, info = ridge.ttt_step(
            data.inputs[index].unsqueeze(0), previous_target
        )
        returned = returned.detach().cpu()
        lower = info["lower"][..., :2].detach().cpu()
        upper = info["upper"][..., :2].detach().cpu()
        center = 0.5 * (lower + upper)
        half = 0.5 * (upper - lower)
        features = feature_tensor(
            data=data,
            index=index,
            model=ridge,
            returned=returned,
            center=center,
        ).numpy()
        tr = int(data.trajectory_index[index])
        forecast = np.einsum(
            "thwcp,thwcp->thwc",
            features,
            expand_beta(beta_bank[tr]),
            optimize=True,
        )
        forecast = np.clip(forecast, -FORECAST_CLIP, FORECAST_CLIP).astype(
            np.float32
        )
        forecast_t = torch.from_numpy(forecast).unsqueeze(0)

        if previous_target is not None and previous_ridge_returned is not None:
            previous_target_physical = (
                previous_target[..., :2] * data.std_target[..., :2]
            )
            target_deviation = previous_target_physical - (
                previous_target_physical.mean(dim=1, keepdim=True)
            )
            prediction_deviation = previous_ridge_returned - (
                previous_ridge_returned.mean(dim=1, keepdim=True)
            )
            prediction_variance = (
                prediction_deviation * data.std_target[..., :2]
            ).square().mean(dim=1)[0]
            target_variance = target_deviation.square().mean(dim=1)[0]
            observed = 0.5 * torch.log(
                (target_variance + 1.0e-8)
                / (prediction_variance + 1.0e-8)
            ).clamp(math.log(0.5), math.log(2.0))
            if pixel_state is None:
                pixel_state = observed.clone()
            else:
                pixel_state = (
                    PIXEL_STATE_DECAY * pixel_state
                    + (1.0 - PIXEL_STATE_DECAY) * observed
                )

        returned_uv = returned[..., :2]
        candidate_uv = returned_uv + PREDICTION_COUPLING * forecast_t
        returned_deviation = returned_uv - returned_uv.mean(
            dim=1, keepdim=True
        )
        current_input_physical = (
            data.inputs[index].unsqueeze(0)[..., :2] * data.std_input[..., :2]
        )
        input_deviation = current_input_physical - current_input_physical.mean(
            dim=1, keepdim=True
        )
        returned_variance = (
            returned_deviation * data.std_target[..., :2]
        ).square().mean(dim=1)[0]
        input_variance = input_deviation.square().mean(dim=1)[0]
        current_ratio = 0.5 * torch.log(
            (input_variance + 1.0e-8) / (returned_variance + 1.0e-8)
        ).clamp(math.log(0.5), math.log(2.0))
        if pixel_state is not None:
            blended = (
                PIXEL_PREVIOUS_WEIGHT * pixel_state
                + (1.0 - PIXEL_PREVIOUS_WEIGHT) * current_ratio
            )
            scale = torch.exp(PIXEL_DAMPING * blended).clamp(0.8, 1.25)
            candidate_mean = candidate_uv.mean(dim=1, keepdim=True)
            candidate_uv = candidate_mean + scale[None, None] * (
                candidate_uv - candidate_mean
            )

        candidate = returned.clone()
        candidate[..., :2] = candidate_uv
        target = target_raw(data, index)
        errors = vector_errors(
            candidate * data.std_target + data.mean_target, target
        )[0]
        error_sum[tr] += errors.numpy().astype(np.float64)
        factors = 0.5 / (0.5 + errors)
        accuracy = float(
            0.5 * factors[0] + 0.3 * factors[1] + 0.2 * factors[2]
        )

        target_norm = data.targets[index].unsqueeze(0)[..., :2]
        learned_center = center + BOUNDS_DAMPING * forecast_t
        residual = (target_norm - learned_center).abs().squeeze(0)
        half0 = half.squeeze(0)
        target_raw_uv = target[..., :2]
        valid = (target_raw_uv != 0.0).squeeze(0)
        candidate_half = width_values * half0.unsqueeze(0)
        inside = residual.unsqueeze(0) <= candidate_half
        tight = torch.exp(
            -2.0
            * candidate_half
            * data.std_target[..., :2].squeeze(0).unsqueeze(0)
            / float(scoring.SIGMA_GLOBAL)
        )
        reward = inside * valid.unsqueeze(0) * tight * accuracy

        reward_group = reward.reshape(
            nwidth, 20, 4, 8, 8, 8, 2
        ).sum(dim=(3, 5)).permute(1, 4, 2, 3, 0)
        covered_group = (inside & valid.unsqueeze(0)).reshape(
            nwidth, 20, 4, 8, 8, 8, 2
        ).sum(dim=(3, 5)).permute(1, 4, 2, 3, 0)
        fine_reward[tr] += reward_group.numpy().astype(np.float64)
        fine_covered[tr] += covered_group.numpy().astype(np.int64)
        windows[tr] += 1
        valid_count[tr] += int(valid.sum())

        previous_target = data.targets[index].unsqueeze(0)
        previous_ridge_returned = returned_uv
        if args.progress_every and (index + 1) % args.progress_every == 0:
            print(
                f"width_windows={index + 1} "
                f"elapsed={time.perf_counter() - replay_started:.1f}s",
                flush=True,
            )

    baseline_index = int(np.argmin(np.abs(WIDTH_GRID - BASE_WIDTH)))
    baseline_reward = fine_reward[..., baseline_index].sum(
        axis=(1, 2, 3, 4)
    )
    baseline_covered = fine_covered[..., baseline_index].sum(
        axis=(1, 2, 3, 4)
    )
    names = np.asarray(data.trajectory_ids)
    dev = np.asarray([name in DEV_IDS for name in names])
    angles = np.asarray(
        [int(name.removesuffix(".h5").rsplit("_", 1)[1]) for name in names]
    )
    selectors = {
        "global": np.ones(ntraj, dtype=bool),
        "dev10": dev,
        "hold20": ~dev,
    }
    selectors.update(
        {f"angle_{angle}": angles == angle for angle in sorted(set(angles))}
    )
    baseline_groups = {
        name: group_scores(
            error_sum,
            baseline_reward,
            baseline_covered,
            windows,
            valid_count,
            selector,
        )
        for name, selector in selectors.items()
    }
    baseline_trajectory = np.asarray(
        [
            group_scores(
                error_sum,
                baseline_reward,
                baseline_covered,
                windows,
                valid_count,
                np.arange(ntraj) == tr,
            )["final_proxy"]
            for tr in range(ntraj)
        ]
    )

    results: list[dict[str, Any]] = []
    cached_group_stats: dict[
        tuple[int, int, int], tuple[np.ndarray, np.ndarray]
    ] = {}
    for time_bins in TIME_BINS:
        for y_bins, x_bins in SPATIAL_PARTITIONS:
            key = (time_bins, y_bins, x_bins)
            grouped_reward = aggregate_groups(
                fine_reward, time_bins, y_bins, x_bins
            )
            grouped_covered = aggregate_groups(
                fine_covered, time_bins, y_bins, x_bins
            )
            cached_group_stats[key] = (grouped_reward, grouped_covered)
            total_reward = grouped_reward.sum(axis=0)
            for shrinkage in SHRINKAGES:
                selected_reward = np.zeros(ntraj, dtype=np.float64)
                selected_covered = np.zeros(ntraj, dtype=np.int64)
                width_min = math.inf
                width_max = -math.inf
                width_sum = 0.0
                width_n = 0
                for tr in range(ntraj):
                    train_reward = total_reward - grouped_reward[tr]
                    fitted_index = train_reward.argmax(axis=-1)
                    fitted_width = WIDTH_GRID[fitted_index]
                    effective_width = BASE_WIDTH + shrinkage * (
                        fitted_width - BASE_WIDTH
                    )
                    effective_index = nearest_grid_indices(effective_width)
                    chosen_reward = np.take_along_axis(
                        grouped_reward[tr], effective_index[..., None], axis=-1
                    )[..., 0]
                    chosen_covered = np.take_along_axis(
                        grouped_covered[tr], effective_index[..., None], axis=-1
                    )[..., 0]
                    selected_reward[tr] = chosen_reward.sum()
                    selected_covered[tr] = chosen_covered.sum()
                    selected_width = WIDTH_GRID[effective_index]
                    width_min = min(width_min, float(selected_width.min()))
                    width_max = max(width_max, float(selected_width.max()))
                    width_sum += float(selected_width.sum())
                    width_n += int(selected_width.size)

                groups = {
                    name: group_scores(
                        error_sum,
                        selected_reward,
                        selected_covered,
                        windows,
                        valid_count,
                        selector,
                    )
                    for name, selector in selectors.items()
                }
                deltas = {
                    name: subtract(groups[name], baseline_groups[name])
                    for name in groups
                }
                trajectory_scores = np.asarray(
                    [
                        group_scores(
                            error_sum,
                            selected_reward,
                            selected_covered,
                            windows,
                            valid_count,
                            np.arange(ntraj) == tr,
                        )["final_proxy"]
                        for tr in range(ntraj)
                    ]
                )
                trajectory_delta = trajectory_scores - baseline_trajectory
                results.append(
                    {
                        "config": {
                            "time_bins": time_bins,
                            "y_bins": y_bins,
                            "x_bins": x_bins,
                            "shrinkage": shrinkage,
                        },
                        "groups": groups,
                        "delta_vs_reference": deltas,
                        "width_summary": {
                            "min": width_min,
                            "mean": width_sum / max(width_n, 1),
                            "max": width_max,
                        },
                        "robustness": {
                            "mean": float(trajectory_delta.mean()),
                            "std": float(trajectory_delta.std()),
                            "robust_050": float(
                                trajectory_delta.mean()
                                - 0.5 * trajectory_delta.std()
                            ),
                            "worst": float(trajectory_delta.min()),
                            "wins": int((trajectory_delta > 0.0).sum()),
                            "losses": int((trajectory_delta < 0.0).sum()),
                            "all_angles_positive": bool(
                                all(
                                    deltas[f"angle_{angle}"]["final_proxy"]
                                    > 0.0
                                    for angle in sorted(set(angles))
                                )
                            ),
                        },
                        "trajectory_delta": {
                            str(name): float(value)
                            for name, value in zip(names, trajectory_delta)
                        },
                    }
                )

    results.sort(
        key=lambda item: (
            item["robustness"]["losses"] == 0,
            item["robustness"]["all_angles_positive"],
            item["robustness"]["robust_050"],
            item["delta_vs_reference"]["global"]["final_proxy"],
        ),
        reverse=True,
    )
    best = results[0]
    best_config = best["config"]
    key = (
        int(best_config["time_bins"]),
        int(best_config["y_bins"]),
        int(best_config["x_bins"]),
    )
    grouped_reward, _ = cached_group_stats[key]
    all30_index = grouped_reward.sum(axis=0).argmax(axis=-1)
    all30_width = WIDTH_GRID[all30_index]
    all30_width = BASE_WIDTH + float(best_config["shrinkage"]) * (
        all30_width - BASE_WIDTH
    )
    all30_width = WIDTH_GRID[nearest_grid_indices(all30_width)]
    deploy_width = expand_group_width(
        all30_width.astype(np.float32), *key
    )
    total_grouped_reward = grouped_reward.sum(axis=0)
    loto_width_bank = []
    for tr in range(ntraj):
        train_index = (
            total_grouped_reward - grouped_reward[tr]
        ).argmax(axis=-1)
        train_width = WIDTH_GRID[train_index]
        train_width = BASE_WIDTH + float(best_config["shrinkage"]) * (
            train_width - BASE_WIDTH
        )
        train_width = WIDTH_GRID[nearest_grid_indices(train_width)]
        loto_width_bank.append(
            expand_group_width(train_width.astype(np.float32), *key)
        )
    loto_width_scale = np.stack(loto_width_bank)
    args.deploy_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "width_scale": torch.from_numpy(deploy_width),
            "loto_width_scale": torch.from_numpy(loto_width_scale),
            "trajectory_ids": list(data.trajectory_ids),
            "base_width": BASE_WIDTH,
            "selection": "trajectory LOTO exact-SPS width-map",
            "config": best_config,
            "grid": WIDTH_GRID.tolist(),
        },
        args.deploy_output,
    )

    report = {
        "schema_version": 1,
        "experiment": "learned static width map",
        "status": "completed",
        "causality": {
            "crossfit": 'leave-one-trajectory-out',
            "inference_features": 'none; static width tensor',
        },
        "data": {
            "trajectories": ntraj,
            "windows": int(windows.sum()),
            "trajectory_ids": list(data.trajectory_ids),
        },
        "frozen_reference": {
            "prediction_coupling": PREDICTION_COUPLING,
            "bounds_damping": BOUNDS_DAMPING,
            "pixel_previous_weight": PIXEL_PREVIOUS_WEIGHT,
            "pixel_state_decay": PIXEL_STATE_DECAY,
            "pixel_damping": PIXEL_DAMPING,
            "base_width": BASE_WIDTH,
            "time_score_assumption": TIME_SCORE,
        },
        "search": {
            "width_grid": WIDTH_GRID.tolist(),
            "time_bins": list(TIME_BINS),
            "spatial_partitions": [list(value) for value in SPATIAL_PARTITIONS],
            "shrinkages": list(SHRINKAGES),
            "candidate_count": len(results),
        },
        "baseline_reference": baseline_groups,
        "best": best,
        "ranked_candidates": results,
        "deploy": {
            "path": str(args.deploy_output),
            "shape": list(deploy_width.shape),
            "min": float(deploy_width.min()),
            "mean": float(deploy_width.mean()),
            "max": float(deploy_width.max()),
        },
        "duration_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"width_report={args.output}", flush=True)


if __name__ == "__main__":
    main()
