"""Interval recalibration."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch

from refine_learned_interval import DEV_IDS, load_data, target_raw
from refine_joint_grid import score_from_error, vector_errors


ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "starting_kit"
PROTOTYPE = ROOT / "ridge_calibrated"
DEFAULT_REPORT = ROOT / "dev/reports/final_interval_recalibration.json"
DEFAULT_SCALAR = ROOT / "dev/cache/final_interval_scalar.pt"
DEFAULT_RESIDUAL = ROOT / "dev/cache/final_interval_residual_map.pt"
DEFAULT_STATS = ROOT / "dev/cache/final_interval_stats.pt"
TIME_SCORE = 85.066771
BASE_MULTIPLIER = 1.0

GLOBAL_GRID = np.round(np.arange(0.70, 1.1001, 0.005), 4)
MAP_GRID = np.round(np.arange(0.50, 1.3001, 0.0125), 4)
MULTIPLIER_GRID = np.unique(np.concatenate((GLOBAL_GRID, MAP_GRID, [1.0])))
TIME_BINS = (1, 4, 20)
SPATIAL_PARTITIONS = ((1, 1), (2, 4), (4, 8))
SHRINKAGES = (0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.50, 0.75, 1.00)

sys.path.insert(0, str(KIT))
import scoring


class ReplayBase(torch.nn.Module):

    def __init__(self, predictions: torch.Tensor) -> None:
        super().__init__()
        self.predictions = predictions
        self.index = 0

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.index >= self.predictions.shape[0]:
            raise IndexError("replay consumed too many cached base predictions")
        output = self.predictions[self.index].unsqueeze(0).to(value.device)
        self.index += 1
        return output


def load_submission(root: Path) -> ModuleType:

    for dependency in (
        "causal_ridge",
        "causal_sps",
        "load_baseline",
        "cuda_graph_forward",
    ):
        sys.modules.pop(dependency, None)
    spec = importlib.util.spec_from_file_location(
        "interval_recalibration_prototype", root / "submission.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import submission from {root}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(root))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def payload_tensor(path: Path, key: str) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    tensor = payload.get(key)
    if tensor is None:
        raise KeyError(f"{path} has no {key!r}")
    return torch.as_tensor(tensor).float()


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


def aggregate_groups(
    values: np.ndarray,
    time_bins: int,
    y_bins: int,
    x_bins: int,
) -> np.ndarray:

    nt, time_steps, channels, fine_y, fine_x, candidates = values.shape
    if time_steps % time_bins or fine_y % y_bins or fine_x % x_bins:
        raise ValueError("grouping must divide the fine statistic grid")
    reshaped = values.reshape(
        nt,
        time_bins,
        time_steps // time_bins,
        channels,
        y_bins,
        fine_y // y_bins,
        x_bins,
        fine_x // x_bins,
        candidates,
    )
    return reshaped.sum(axis=(2, 5, 7))


def nearest_indices(values: np.ndarray) -> np.ndarray:
    return np.abs(values[..., None] - MULTIPLIER_GRID).argmin(axis=-1)


def grid_indices(grid: np.ndarray) -> np.ndarray:
    return np.asarray(
        [int(np.argmin(np.abs(MULTIPLIER_GRID - value))) for value in grid],
        dtype=np.int64,
    )


def expand_group_map(
    grouped: np.ndarray,
    time_bins: int,
    y_bins: int,
    x_bins: int,
) -> np.ndarray:
    result = grouped.repeat(20 // time_bins, axis=0)
    result = result.repeat(4 // y_bins, axis=2)
    result = result.repeat(8 // x_bins, axis=3)
    if result.shape != (20, 2, 4, 8):
        raise RuntimeError(f"unexpected residual-map shape {result.shape}")
    return result


def score_group(
    error_sum: np.ndarray,
    reward: np.ndarray,
    covered: np.ndarray,
    windows: np.ndarray,
    valid_count: np.ndarray,
    selector: np.ndarray,
) -> dict[str, float | int]:
    nwin = int(windows[selector].sum())
    nvalid = int(valid_count[selector].sum())
    rel, tke, mvpe = score_from_error(
        error_sum[selector].sum(axis=0) / max(nwin, 1)
    )
    sps = 100.0 * float(reward[selector].sum()) / max(nvalid, 1)
    return {
        "trajectories": int(selector.sum()),
        "windows": nwin,
        "valid_elements": nvalid,
        "rel_l2_score": float(rel),
        "tke_score": float(tke),
        "mvpe_score": float(mvpe),
        "sps_score": sps,
        "coverage": float(covered[selector].sum()) / max(nvalid, 1),
        "final_proxy": float(final_proxy(rel, tke, mvpe, sps)),
    }


def subtract(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    keys = (
        "rel_l2_score",
        "tke_score",
        "mvpe_score",
        "sps_score",
        "coverage",
        "final_proxy",
    )
    return {key: float(candidate[key] - baseline[key]) for key in keys}


def summarize_fixed_multiplier(
    *,
    multiplier: float,
    multiplier_index: int,
    fine_reward: np.ndarray,
    fine_covered: np.ndarray,
    error_sum: np.ndarray,
    windows: np.ndarray,
    valid_count: np.ndarray,
    selectors: dict[str, np.ndarray],
    baseline_groups: dict[str, dict[str, Any]],
    baseline_trajectory: np.ndarray,
) -> dict[str, Any]:
    reward = fine_reward[..., multiplier_index].sum(axis=(1, 2, 3, 4))
    covered = fine_covered[..., multiplier_index].sum(axis=(1, 2, 3, 4))
    groups = {
        name: score_group(
            error_sum, reward, covered, windows, valid_count, selector
        )
        for name, selector in selectors.items()
    }
    deltas = {
        name: subtract(groups[name], baseline_groups[name]) for name in groups
    }
    trajectory = np.asarray(
        [
            score_group(
                error_sum,
                reward,
                covered,
                windows,
                valid_count,
                np.arange(len(windows)) == tr,
            )["final_proxy"]
            for tr in range(len(windows))
        ]
    )
    trajectory_delta = trajectory - baseline_trajectory
    return {
        "multiplier": float(multiplier),
        "groups": groups,
        "delta_vs_current": deltas,
        "robustness": {
            "mean": float(trajectory_delta.mean()),
            "std": float(trajectory_delta.std()),
            "robust_050": float(
                trajectory_delta.mean() - 0.5 * trajectory_delta.std()
            ),
            "worst": float(trajectory_delta.min()),
            "wins": int((trajectory_delta > 1.0e-12).sum()),
            "ties": int((np.abs(trajectory_delta) <= 1.0e-12).sum()),
            "losses": int((trajectory_delta < -1.0e-12).sum()),
            "all_angles_positive": bool(
                all(
                    deltas[key]["final_proxy"] > 0.0
                    for key in deltas
                    if key.startswith("angle_")
                )
            ),
        },
        "trajectory_delta": trajectory_delta.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prototype", type=Path, default=PROTOTYPE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--scalar-output", type=Path, default=DEFAULT_SCALAR)
    parser.add_argument("--residual-output", type=Path, default=DEFAULT_RESIDUAL)
    parser.add_argument("--stats-cache", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))

    data = load_data()
    module = load_submission(args.prototype.resolve())
    replay_base = ReplayBase(data.base)
    learned = payload_tensor(args.prototype / "learned_interval.pt", "coefficients")
    old_width = payload_tensor(args.prototype / "learned_width.pt", "width_scale")
    supervised = payload_tensor(args.prototype / "supervised_tke.pt", "coefficients")
    temporal = payload_tensor(args.prototype / "temporal_history.pt", "native_beta")
    model = module.CausalRidgeTTTModel(
        replay_base,
        "cpu",
        learned,
        old_width,
        supervised,
        temporal,
    )

    ntraj = len(data.trajectory_ids)
    nmult = len(MULTIPLIER_GRID)
    fine_reward = np.zeros(
        (ntraj, 20, 2, 4, 8, nmult), dtype=np.float64
    )
    fine_covered = np.zeros(
        (ntraj, 20, 2, 4, 8, nmult), dtype=np.int64
    )
    error_sum = np.zeros((ntraj, 3), dtype=np.float64)
    windows = np.zeros(ntraj, dtype=np.int64)
    valid_count = np.zeros(ntraj, dtype=np.int64)
    multiplier_tensor = torch.tensor(
        MULTIPLIER_GRID, dtype=torch.float32
    ).reshape(nmult, 1, 1, 1, 1)
    target_std_uv = data.std_target[..., :2]
    previous_target = None
    replay_started = time.perf_counter()

    replay_indices: range | tuple[()] = range(data.targets.shape[0])
    if args.stats_cache.exists():
        cached = torch.load(
            args.stats_cache, map_location="cpu", weights_only=False
        )
        cached_grid = np.asarray(cached.get("multiplier_grid"), dtype=np.float64)
        if np.array_equal(cached_grid, MULTIPLIER_GRID):
            fine_reward = np.asarray(cached["fine_reward"], dtype=np.float64)
            fine_covered = np.asarray(cached["fine_covered"], dtype=np.int64)
            error_sum = np.asarray(cached["error_sum"], dtype=np.float64)
            windows = np.asarray(cached["windows"], dtype=np.int64)
            valid_count = np.asarray(cached["valid_count"], dtype=np.int64)
            replay_indices = ()
            print(f"loaded_stats_cache={args.stats_cache}", flush=True)

    for index in replay_indices:
        if data.first[index]:
            model.reset_ttt_state()
            previous_target = None
        returned, info = model.ttt_step(
            data.inputs[index].unsqueeze(0), previous_target
        )
        returned = returned.detach().cpu()
        lower = info["lower"][..., :2].detach().cpu()
        upper = info["upper"][..., :2].detach().cpu()
        center = 0.5 * (lower + upper)
        half_raw = (0.5 * (upper - lower) * target_std_uv).squeeze(0)
        target = target_raw(data, index)
        target_raw_uv = target[..., :2]
        center_raw_uv = center * target_std_uv + data.mean_target[..., :2]
        residual_raw = (target_raw_uv - center_raw_uv).abs().squeeze(0)
        valid = (target_raw_uv != 0.0).squeeze(0)

        prediction_raw = returned * data.std_target + data.mean_target
        errors = vector_errors(prediction_raw, target)[0]
        tr = int(data.trajectory_index[index])
        error_sum[tr] += errors.numpy().astype(np.float64)
        factors = 0.5 / (0.5 + errors.clamp_min(0.0))
        accuracy = float(
            0.5 * factors[0] + 0.3 * factors[1] + 0.2 * factors[2]
        )

        candidate_half = multiplier_tensor * half_raw.unsqueeze(0)
        inside = residual_raw.unsqueeze(0) <= candidate_half
        tightness = torch.exp(
            -2.0 * candidate_half / float(scoring.SIGMA_GLOBAL)
        )
        reward = inside * valid.unsqueeze(0) * tightness * accuracy
        reward_group = reward.reshape(
            nmult, 20, 4, 8, 8, 8, 2
        ).sum(dim=(3, 5)).permute(1, 4, 2, 3, 0)
        covered_group = (inside & valid.unsqueeze(0)).reshape(
            nmult, 20, 4, 8, 8, 8, 2
        ).sum(dim=(3, 5)).permute(1, 4, 2, 3, 0)
        fine_reward[tr] += reward_group.numpy().astype(np.float64)
        fine_covered[tr] += covered_group.numpy().astype(np.int64)
        windows[tr] += 1
        valid_count[tr] += int(valid.sum())
        previous_target = data.targets[index].unsqueeze(0)

        if args.progress_every and (index + 1) % args.progress_every == 0:
            print(
                f"final_interval_windows={index + 1} "
                f"elapsed={time.perf_counter() - replay_started:.1f}s",
                flush=True,
            )

    if replay_indices and replay_base.index != data.targets.shape[0]:
        raise RuntimeError("cached base replay did not consume the full stream")
    if replay_indices:
        args.stats_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "prototype": str(args.prototype.resolve()),
                "multiplier_grid": MULTIPLIER_GRID,
                "fine_reward": fine_reward,
                "fine_covered": fine_covered,
                "error_sum": error_sum,
                "windows": windows,
                "valid_count": valid_count,
                "trajectory_ids": data.trajectory_ids,
            },
            args.stats_cache,
        )

    names = np.asarray(data.trajectory_ids)
    dev = np.asarray([name in DEV_IDS for name in names])
    angles = np.asarray(
        [int(Path(name).stem.rsplit("_", 1)[1]) for name in names]
    )
    selectors: dict[str, np.ndarray] = {
        "global": np.ones(ntraj, dtype=bool),
        "dev10": dev,
        "hold20": ~dev,
    }
    selectors.update(
        {f"angle_{angle}": angles == angle for angle in sorted(set(angles))}
    )

    baseline_index = int(np.argmin(np.abs(MULTIPLIER_GRID - 1.0)))
    baseline_reward = fine_reward[..., baseline_index].sum(
        axis=(1, 2, 3, 4)
    )
    baseline_covered = fine_covered[..., baseline_index].sum(
        axis=(1, 2, 3, 4)
    )
    baseline_groups = {
        name: score_group(
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
            score_group(
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

    scalar_rows = []
    global_indices = grid_indices(GLOBAL_GRID)
    for multiplier, multiplier_index in zip(GLOBAL_GRID, global_indices):
        scalar_rows.append(
            summarize_fixed_multiplier(
                multiplier=float(multiplier),
                multiplier_index=int(multiplier_index),
                fine_reward=fine_reward,
                fine_covered=fine_covered,
                error_sum=error_sum,
                windows=windows,
                valid_count=valid_count,
                selectors=selectors,
                baseline_groups=baseline_groups,
                baseline_trajectory=baseline_trajectory,
            )
        )

    def best_on(group: str) -> dict[str, Any]:
        return max(
            scalar_rows,
            key=lambda row: (
                row["groups"][group]["final_proxy"],
                -abs(row["multiplier"] - 1.0),
            ),
        )

    dev_selected = best_on("dev10")
    hold_selected = best_on("hold20")
    all_selected = best_on("global")
    bidirectional = {
        "dev10_selects_hold20": {
            "selected_multiplier": dev_selected["multiplier"],
            "selection_delta_dev10": dev_selected["delta_vs_current"]["dev10"],
            "frozen_evaluation_delta_hold20": dev_selected["delta_vs_current"]["hold20"],
            "full_row": dev_selected,
        },
        "hold20_selects_dev10": {
            "selected_multiplier": hold_selected["multiplier"],
            "selection_delta_hold20": hold_selected["delta_vs_current"]["hold20"],
            "frozen_evaluation_delta_dev10": hold_selected["delta_vs_current"]["dev10"],
            "full_row": hold_selected,
        },
    }

    map_candidate_indices = grid_indices(MAP_GRID)
    results: list[dict[str, Any]] = []
    cached_group_stats: dict[
        tuple[int, int, int], tuple[np.ndarray, np.ndarray]
    ] = {}
    for time_bins in TIME_BINS:
        for y_bins, x_bins in SPATIAL_PARTITIONS:
            key = (time_bins, y_bins, x_bins)
            grouped_reward_all = aggregate_groups(
                fine_reward, time_bins, y_bins, x_bins
            )
            grouped_covered_all = aggregate_groups(
                fine_covered, time_bins, y_bins, x_bins
            )
            grouped_reward = grouped_reward_all[..., map_candidate_indices]
            cached_group_stats[key] = (grouped_reward, grouped_reward_all)
            total_reward = grouped_reward.sum(axis=0)
            for shrinkage in SHRINKAGES:
                selected_reward = np.zeros(ntraj, dtype=np.float64)
                selected_covered = np.zeros(ntraj, dtype=np.int64)
                width_values: list[np.ndarray] = []
                for tr in range(ntraj):
                    training_reward = total_reward - grouped_reward[tr]
                    fitted = MAP_GRID[training_reward.argmax(axis=-1)]
                    effective = BASE_MULTIPLIER + shrinkage * (
                        fitted - BASE_MULTIPLIER
                    )
                    effective_index = nearest_indices(effective)
                    chosen_reward = np.take_along_axis(
                        grouped_reward_all[tr],
                        effective_index[..., None],
                        axis=-1,
                    )[..., 0]
                    chosen_covered = np.take_along_axis(
                        grouped_covered_all[tr],
                        effective_index[..., None],
                        axis=-1,
                    )[..., 0]
                    selected_reward[tr] = chosen_reward.sum()
                    selected_covered[tr] = chosen_covered.sum()
                    width_values.append(MULTIPLIER_GRID[effective_index])

                groups = {
                    name: score_group(
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
                        score_group(
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
                stacked_width = np.stack(width_values)
                results.append(
                    {
                        "config": {
                            "time_bins": time_bins,
                            "y_bins": y_bins,
                            "x_bins": x_bins,
                            "shrinkage": shrinkage,
                        },
                        "groups": groups,
                        "delta_vs_current": deltas,
                        "multiplier_summary": {
                            "min": float(stacked_width.min()),
                            "mean": float(stacked_width.mean()),
                            "max": float(stacked_width.max()),
                        },
                        "robustness": {
                            "mean": float(trajectory_delta.mean()),
                            "std": float(trajectory_delta.std()),
                            "robust_050": float(
                                trajectory_delta.mean()
                                - 0.5 * trajectory_delta.std()
                            ),
                            "worst": float(trajectory_delta.min()),
                            "wins": int((trajectory_delta > 1.0e-12).sum()),
                            "ties": int((np.abs(trajectory_delta) <= 1.0e-12).sum()),
                            "losses": int((trajectory_delta < -1.0e-12).sum()),
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
            item["delta_vs_current"]["global"]["final_proxy"],
        ),
        reverse=True,
    )
    best = results[0]
    recommendation_passed = bool(
        best["robustness"]["all_angles_positive"]
        and best["robustness"]["worst"] >= -1.0e-12
    )

    best_config = best["config"]
    best_key = (
        int(best_config["time_bins"]),
        int(best_config["y_bins"]),
        int(best_config["x_bins"]),
    )
    grouped_reward, _ = cached_group_stats[best_key]
    all30_fitted = MAP_GRID[grouped_reward.sum(axis=0).argmax(axis=-1)]
    all30_effective = BASE_MULTIPLIER + float(best_config["shrinkage"]) * (
        all30_fitted - BASE_MULTIPLIER
    )
    all30_effective_index = np.abs(
        all30_effective[..., None] - MAP_GRID
    ).argmin(axis=-1)
    all30_effective = MAP_GRID[all30_effective_index]
    deploy_map = expand_group_map(all30_effective.astype(np.float32), *best_key)

    args.scalar_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "half_width_multiplier": float(all_selected["multiplier"]),
            "selection": 'all30 refit',
            "bidirectional_selected": {
                "dev10": float(dev_selected["multiplier"]),
                "hold20": float(hold_selected["multiplier"]),
            },
            "prototype": str(args.prototype),
        },
        args.scalar_output,
    )
    torch.save(
        {
            "schema_version": 1,
            "residual_multiplier": torch.from_numpy(deploy_map),
            "selection": 'all30 refit of the LOTO-selected family',
            "config": best_config,
            "recommended": recommendation_passed,
            "prototype": str(args.prototype),
        },
        args.residual_output,
    )

    report = {
        "schema_version": 1,
        "experiment": "interval recalibration",
        "status": "completed",
        "causality": {
            "prediction_frozen": True,
            "interval_center_frozen": True,
            "global_scan": 'static scalar; dev10/hold20 split',
            "residual_map": 'leave-one-trajectory-out',
        },
        "data": {
            "source": 'train_real balanced-30 caches',
            "trajectories": ntraj,
            "windows": int(windows.sum()),
            "trajectory_ids": list(data.trajectory_ids),
        },
        "prototype": {
            "path": str(args.prototype),
            "components": [
                "full20 temporal-history forecast",
                "supervised TKE correction",
                "existing learned width map",
            ],
            "existing_width": {
                "shape": list(old_width.shape),
                "min": float(old_width.min()),
                "mean": float(old_width.mean()),
                "max": float(old_width.max()),
            },
            "time_score_assumption": TIME_SCORE,
        },
        "baseline_current": baseline_groups,
        "global_scalar_scan": {
            "grid": GLOBAL_GRID.tolist(),
            "all30_in_sample_best": all_selected,
            "bidirectional_frozen": bidirectional,
            "ranked_by_global": sorted(
                scalar_rows,
                key=lambda row: row["groups"]["global"]["final_proxy"],
                reverse=True,
            ),
        },
        "loto_residual_map": {
            "grid": MAP_GRID.tolist(),
            "time_bins": list(TIME_BINS),
            "spatial_partitions": [list(value) for value in SPATIAL_PARTITIONS],
            "shrinkages": list(SHRINKAGES),
            "candidate_count": len(results),
            "recommendation_gate": {
                "required": "all five angles positive and worst trajectory >= 0",
                "passed": recommendation_passed,
            },
            "best": best,
            "ranked_candidates": results,
        },
        "deploy": {
            "scalar_path": str(args.scalar_output),
            "scalar": float(all_selected["multiplier"]),
            "residual_path": str(args.residual_output),
            "residual_shape": list(deploy_map.shape),
            "residual_min": float(deploy_map.min()),
            "residual_mean": float(deploy_map.mean()),
            "residual_max": float(deploy_map.max()),
        },
        "duration_seconds": time.perf_counter() - started,
        "artifacts": {
            "source_data": [
                "dev/cache/train_real_balanced30.pt",
                "dev/cache/train_real_balanced30_inputs.pt",
            ],
            "source_model": str(args.prototype),
            "generated": [str(args.report), str(args.scalar_output), str(args.residual_output)],
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"final_interval_report={args.report}", flush=True)


if __name__ == "__main__":
    main()
