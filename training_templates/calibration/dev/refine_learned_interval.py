"""Learned interval adapter."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "starting_kit"
RIDGE_DIR = ROOT / "ridge_mean_split"
CACHE = ROOT / "dev" / "cache"
DEFAULT_OUTPUT = ROOT / "dev" / "reports" / "learned_causal_interval.json"
sys.path.insert(0, str(KIT))

import scoring


DEV_IDS = {
    "5025_0.h5", "25425_0.h5", "6300_5.h5", "21600_5.h5",
    "8850_10.h5", "26700_10.h5", "10125_15.h5", "26700_15.h5",
    "3750_20.h5", "25425_20.h5",
}
FEATURE_NAMES = (
    "intercept",
    "safe_minus_center",
    "ridge_minus_center",
    "base_minus_center",
    "returned_minus_center",
    "input_aligned_minus_center",
    "input_last_minus_center",
    "linear_extrapolation_minus_center",
)
P = len(FEATURE_NAMES)
FINE_Y_BINS = 4
FINE_X_BINS = 8
TRAIN_RESPONSE_CLIPS = (0.25, 0.50, 1.00, 4.00)
RIDGES = (1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2)
PARTITIONS = ((1, 1), (2, 4), (4, 8))
DAMPINGS = (0.25, 0.50, 0.75, 1.00, 1.25)
WIDTH_SCALES = (0.80, 0.85, 0.875, 0.90, 0.925, 0.95, 0.975, 1.00, 1.025)
PREDICTION_CLIP = 0.75
TOP_MODELS = 12


class ReplayBase(torch.nn.Module):
    def __init__(self, predictions: torch.Tensor) -> None:
        super().__init__()
        self.predictions = predictions
        self.index = 0

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.index >= self.predictions.shape[0]:
            raise IndexError("ridge replay consumed too many cached predictions")
        result = self.predictions[self.index].unsqueeze(0).to(value.device)
        self.index += 1
        return result


def load_ridge():
    sys.path.insert(0, str(RIDGE_DIR))
    try:
        spec = importlib.util.spec_from_file_location(
            "refine_learned_interval_ridge", RIDGE_DIR / "submission.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot import the ridge model")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


@dataclass(frozen=True)
class ReplayData:
    base: torch.Tensor
    inputs: torch.Tensor
    targets: torch.Tensor
    first: np.ndarray
    simulation_ids: tuple[str, ...]
    trajectory_ids: tuple[str, ...]
    trajectory_index: np.ndarray
    mean_input: torch.Tensor
    std_input: torch.Tensor
    mean_target: torch.Tensor
    std_target: torch.Tensor


def load_data() -> ReplayData:
    base_cache = torch.load(
        CACHE / "train_real_balanced30.pt", map_location="cpu", weights_only=False
    )
    input_cache = torch.load(
        CACHE / "train_real_balanced30_inputs.pt", map_location="cpu", weights_only=False
    )
    if input_cache["simulation_id"] != base_cache["simulation_id"]:
        raise RuntimeError("input and base caches have different stream order")
    ids = tuple(str(x) for x in base_cache["simulation_id"])
    trajectories = tuple(dict.fromkeys(ids))
    mapping = {name: i for i, name in enumerate(trajectories)}
    trajectory_index = np.asarray([mapping[name] for name in ids], dtype=np.int64)
    first = base_cache["trajectory_first"].numpy().astype(bool)
    if int(first.sum()) != len(trajectories) or len(trajectories) != 30:
        raise RuntimeError("expected 30 correctly marked trajectories")
    return ReplayData(
        base=base_cache["base_prediction_norm"].float(),
        inputs=input_cache["input_norm"].float(),
        targets=base_cache["target_norm"].float(),
        first=first,
        simulation_ids=ids,
        trajectory_ids=trajectories,
        trajectory_index=trajectory_index,
        mean_input=input_cache["mean_input"].float().reshape(1, 1, 1, 1, 3),
        std_input=input_cache["std_input"].float().reshape(1, 1, 1, 1, 3),
        mean_target=base_cache["mean_target"].float().reshape(1, 1, 1, 1, 3),
        std_target=base_cache["std_target"].float().reshape(1, 1, 1, 1, 3),
    )


def new_model(data: ReplayData):
    module = load_ridge()
    base = ReplayBase(data.base)
    model = module.CausalRidgeTTTModel(base, "cpu")
    return base, model


def feature_tensor(
    *,
    data: ReplayData,
    index: int,
    model: Any,
    returned: torch.Tensor,
    center: torch.Tensor,
) -> torch.Tensor:
    safe = model._previous_prediction[..., :2].detach().cpu()
    ridge = model._previous_ridge_prediction[..., :2].detach().cpu()
    base = model._previous_base_prediction[..., :2].detach().cpu()
    returned_uv = returned[..., :2].detach().cpu()
    input_raw = (
        data.inputs[index].unsqueeze(0) * data.std_input + data.mean_input
    )[..., :2]
    input_target_norm = (
        input_raw - data.mean_target[..., :2]
    ) / data.std_target[..., :2]
    input_last = input_target_norm[:, -1:].expand_as(safe)

    trend = (
        input_target_norm[:, -5:].mean(dim=1, keepdim=True)
        - input_target_norm[:, :5].mean(dim=1, keepdim=True)
    )
    horizon = torch.arange(1, 21, dtype=safe.dtype).reshape(1, 20, 1, 1, 1) / 15.0
    linear = input_last + horizon * trend

    result = torch.empty((*safe.shape[1:], P), dtype=torch.float32)
    center0 = center.squeeze(0)
    result[..., 0] = 1.0
    result[..., 1] = safe.squeeze(0) - center0
    result[..., 2] = ridge.squeeze(0) - center0
    result[..., 3] = base.squeeze(0) - center0
    result[..., 4] = returned_uv.squeeze(0) - center0
    result[..., 5] = input_target_norm.squeeze(0) - center0
    result[..., 6] = input_last.squeeze(0) - center0
    result[..., 7] = linear.squeeze(0) - center0
    if not torch.isfinite(result).all():
        raise RuntimeError("non-finite learned interval feature")
    return result


def target_raw(data: ReplayData, index: int) -> torch.Tensor:
    return (
        data.targets[index].unsqueeze(0) * data.std_target + data.mean_target
    ).to(torch.float32)


def window_accuracy(pred_raw: np.ndarray, tgt_raw: np.ndarray) -> tuple[float, tuple[float, float, float]]:
    errors = (
        float(scoring.rel_l2_per_sample(pred_raw, tgt_raw, 2)[0]),
        float(scoring.tke_rel_l2_per_sample(pred_raw, tgt_raw, 2)[0]),
        float(scoring.mvpe_rel_l2_per_sample(pred_raw, tgt_raw)[0]),
    )
    factors = [0.5 / (0.5 + max(value, 0.0)) for value in errors]
    return float(0.5 * factors[0] + 0.3 * factors[1] + 0.2 * factors[2]), errors


def fine_stats_for_window(
    x: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t, h, w, c, p = x.shape
    if (t, h, w, c, p) != (20, 32, 64, 2, P):
        raise ValueError(f"unexpected feature shape {x.shape}")
    bh, bw = h // FINE_Y_BINS, w // FINE_X_BINS
    xr = x.reshape(t, FINE_Y_BINS, bh, FINE_X_BINS, bw, c, p)
    vr = valid.reshape(t, FINE_Y_BINS, bh, FINE_X_BINS, bw, c)
    yr = y.reshape(t, FINE_Y_BINS, bh, FINE_X_BINS, bw, c)
    xv = xr * vr[..., None]
    xtx = np.einsum("tYhXwcp,tYhXwcq->tcYXpq", xv, xr, optimize=True)
    xty = np.empty((len(TRAIN_RESPONSE_CLIPS), t, c, FINE_Y_BINS, FINE_X_BINS, p), dtype=np.float64)
    for j, clip in enumerate(TRAIN_RESPONSE_CLIPS):
        yc = np.clip(yr, -clip, clip) * vr
        xty[j] = np.einsum("tYhXwcp,tYhXwc->tcYXp", xr, yc, optimize=True)
    y2 = np.einsum("tYhXwc,tYhXwc->tcYX", yr * vr, yr, optimize=True)
    count = vr.sum(axis=(2, 4)).transpose(0, 3, 1, 2)
    return xtx, xty, y2, count


def collect_stats(data: ReplayData, progress_every: int) -> dict[str, np.ndarray]:
    ntraj = len(data.trajectory_ids)
    xtx = np.zeros((ntraj, 20, 2, FINE_Y_BINS, FINE_X_BINS, P, P), dtype=np.float64)
    xty = np.zeros((ntraj, len(TRAIN_RESPONSE_CLIPS), 20, 2, FINE_Y_BINS, FINE_X_BINS, P), dtype=np.float64)
    y2 = np.zeros((ntraj, 20, 2, FINE_Y_BINS, FINE_X_BINS), dtype=np.float64)
    count = np.zeros((ntraj, 20, 2, FINE_Y_BINS, FINE_X_BINS), dtype=np.int64)
    base, model = new_model(data)
    previous_target = None
    start = time.perf_counter()
    for index in range(data.targets.shape[0]):
        if data.first[index]:
            model.reset_ttt_state()
            previous_target = None
        returned, info = model.ttt_step(data.inputs[index].unsqueeze(0), previous_target)
        lower = info["lower"][..., :2].detach().cpu()
        upper = info["upper"][..., :2].detach().cpu()
        center = 0.5 * (lower + upper)
        x = feature_tensor(data=data, index=index, model=model, returned=returned, center=center)
        tgt_norm = data.targets[index, ..., :2].numpy()
        center_np = center.squeeze(0).numpy()
        response = tgt_norm - center_np
        tgt_raw_uv = target_raw(data, index)[..., :2]
        valid = (tgt_raw_uv != 0.0).squeeze(0).numpy()
        values = fine_stats_for_window(x.numpy(), response, valid)
        tr = int(data.trajectory_index[index])
        xtx[tr] += values[0]
        xty[tr] += values[1]
        y2[tr] += values[2]
        count[tr] += values[3]
        previous_target = data.targets[index].unsqueeze(0)
        if progress_every and (index + 1) % progress_every == 0:
            print(f"stats_windows={index + 1} elapsed={time.perf_counter()-start:.1f}s", flush=True)
    if base.index != data.targets.shape[0]:
        raise RuntimeError("stats pass did not consume full base cache")
    return {"xtx": xtx, "xty": xty, "y2": y2, "count": count}


def aggregate_partition(array: np.ndarray, y_bins: int, x_bins: int) -> np.ndarray:
    fy, fx = FINE_Y_BINS, FINE_X_BINS
    if fy % y_bins or fx % x_bins:
        raise ValueError("requested partition does not divide fine partition")
    y_factor, x_factor = fy // y_bins, fx // x_bins
    shape = array.shape
    reshaped = array.reshape(*shape[:3], y_bins, y_factor, x_bins, x_factor, *shape[5:])
    return reshaped.sum(axis=(4, 6))


def solve_coefficients(
    xtx_train: np.ndarray,
    xty_train: np.ndarray,
    count_train: np.ndarray,
    ridge: float,
) -> np.ndarray:
    matrix = xtx_train.copy()
    diagonal = np.arange(P)
    penalty = ridge * np.maximum(count_train, 1)[..., None]
    matrix[..., diagonal[1:], diagonal[1:]] += penalty
    matrix[..., diagonal, diagonal] += 1.0e-10
    try:
        beta = np.linalg.solve(matrix, xty_train[..., None])[..., 0]
    except np.linalg.LinAlgError:
        beta = np.einsum("...pq,...q->...p", np.linalg.pinv(matrix), xty_train)
    beta = np.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0)
    return beta


def fit_and_rank(stats: dict[str, np.ndarray], trajectory_ids: tuple[str, ...]) -> tuple[list[dict[str, Any]], dict[tuple[Any, ...], np.ndarray]]:
    models: list[dict[str, Any]] = []
    coefficients: dict[tuple[Any, ...], np.ndarray] = {}
    for y_bins, x_bins in PARTITIONS:
        xtx = aggregate_partition(stats["xtx"], y_bins, x_bins)
        xty_all = aggregate_partition(stats["xty"].transpose(0, 2, 3, 4, 5, 1, 6), y_bins, x_bins)
        xty = xty_all.transpose(0, 5, 1, 2, 3, 4, 6)
        y2 = aggregate_partition(stats["y2"], y_bins, x_bins)
        count = aggregate_partition(stats["count"], y_bins, x_bins)
        total_xtx = xtx.sum(axis=0)
        total_xty = xty.sum(axis=0)
        total_count = count.sum(axis=0)
        for clip_index, clip in enumerate(TRAIN_RESPONSE_CLIPS):
            for ridge in RIDGES:
                key = (y_bins, x_bins, clip, ridge)
                betas = []
                trajectory_mse = []
                trajectory_rmse = []
                for tr in range(len(trajectory_ids)):
                    beta = solve_coefficients(
                        total_xtx - xtx[tr],
                        total_xty[clip_index] - xty[tr, clip_index],
                        total_count - count[tr],
                        ridge,
                    )
                    betas.append(beta.astype(np.float32))
                    cross = np.einsum("...p,...p->...", beta, xty[tr, -1])
                    quad = np.einsum("...p,...pq,...q->...", beta, xtx[tr], beta)
                    sse = float(np.sum(y2[tr] - 2.0 * cross + quad))
                    n = int(count[tr].sum())
                    mse = max(sse / max(n, 1), 0.0)
                    trajectory_mse.append(mse)
                    trajectory_rmse.append(math.sqrt(mse))
                beta_bank = np.stack(betas)
                coefficients[key] = beta_bank
                mse = np.asarray(trajectory_mse)
                dev = np.asarray([name in DEV_IDS for name in trajectory_ids])
                angles = np.asarray([int(name.removesuffix(".h5").rsplit("_", 1)[1]) for name in trajectory_ids])
                models.append({
                    "key": list(key),
                    "partition": [y_bins, x_bins],
                    "response_clip": clip,
                    "ridge": ridge,
                    "oof_mse": float(mse.mean()),
                    "oof_rmse": float(np.sqrt(mse.mean())),
                    "trajectory_rmse_mean": float(np.mean(trajectory_rmse)),
                    "trajectory_rmse_std": float(np.std(trajectory_rmse)),
                    "dev10_rmse": float(np.sqrt(mse[dev].mean())),
                    "hold20_rmse": float(np.sqrt(mse[~dev].mean())),
                    "angle_rmse": {str(a): float(np.sqrt(mse[angles == a].mean())) for a in sorted(set(angles))},
                })
    models.sort(key=lambda item: (item["oof_mse"], item["partition"][0] * item["partition"][1]))
    selected: list[dict[str, Any]] = []
    for partition in PARTITIONS:
        group = [m for m in models if tuple(m["partition"]) == partition]
        selected.extend(group[:2])
    for model in models:
        if model not in selected:
            selected.append(model)
        if len(selected) >= TOP_MODELS:
            break
    return selected, coefficients


def expand_beta(beta: np.ndarray, height: int = 32, width: int = 64) -> np.ndarray:
    t, c, yb, xb, p = beta.shape
    expanded = np.repeat(np.repeat(beta, height // yb, axis=2), width // xb, axis=3)
    return expanded.transpose(0, 2, 3, 1, 4)


def group_summary(
    numerator: np.ndarray,
    covered: np.ndarray,
    valid: np.ndarray,
    baseline_numerator: np.ndarray,
    baseline_covered: np.ndarray,
    selector: np.ndarray,
) -> dict[str, float | int]:
    denominator = int(valid[selector].sum())
    score = 100.0 * float(numerator[selector].sum()) / denominator
    baseline = 100.0 * float(baseline_numerator[selector].sum()) / denominator
    coverage = float(covered[selector].sum()) / denominator
    baseline_coverage = float(baseline_covered[selector].sum()) / denominator
    return {
        "trajectories": int(selector.sum()),
        "valid_elements": denominator,
        "sps": score,
        "baseline_sps": baseline,
        "delta": score - baseline,
        "coverage": coverage,
        "baseline_coverage": baseline_coverage,
        "coverage_delta": coverage - baseline_coverage,
    }


def evaluate(
    data: ReplayData,
    selected_models: list[dict[str, Any]],
    coefficients: dict[tuple[Any, ...], np.ndarray],
    progress_every: int,
    explicit_candidates: list[tuple[dict[str, Any], float, float]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates: list[tuple[dict[str, Any], float, float]] = (
        explicit_candidates
        if explicit_candidates is not None
        else [
            (model, damping, 1.0)
            for model in selected_models
            for damping in DAMPINGS
        ]
    )
    nc = len(candidates)
    nt = len(data.trajectory_ids)
    numerator = np.zeros((nc, nt), dtype=np.float64)
    covered = np.zeros((nc, nt), dtype=np.int64)
    baseline_numerator = np.zeros(nt, dtype=np.float64)
    baseline_covered = np.zeros(nt, dtype=np.int64)
    valid_count = np.zeros(nt, dtype=np.int64)
    prediction_error_sum = np.zeros(3, dtype=np.float64)
    base, model = new_model(data)
    previous_target = None
    start = time.perf_counter()
    for index in range(data.targets.shape[0]):
        if data.first[index]:
            model.reset_ttt_state()
            previous_target = None
        returned, info = model.ttt_step(data.inputs[index].unsqueeze(0), previous_target)
        lower = info["lower"][..., :2].detach().cpu()
        upper = info["upper"][..., :2].detach().cpu()
        center = 0.5 * (lower + upper)
        half = 0.5 * (upper - lower)
        x = feature_tensor(data=data, index=index, model=model, returned=returned, center=center).numpy()
        tr = int(data.trajectory_index[index])
        target_raw_full = target_raw(data, index)
        target_raw_uv = target_raw_full[..., :2]
        valid = (target_raw_uv != 0.0).squeeze(0)
        valid_np = valid.numpy()
        target_norm_uv = data.targets[index].unsqueeze(0)[..., :2]
        center_error = (target_norm_uv - center).squeeze(0)
        half0 = half.squeeze(0)
        returned_raw = (
            returned.detach().cpu() * data.std_target + data.mean_target
        ).numpy().astype(np.float32)
        accuracy, errors = window_accuracy(returned_raw, target_raw_full.numpy())
        prediction_error_sum += errors
        tightness = torch.exp(-2.0 * half0 * data.std_target[..., :2].squeeze(0) / float(scoring.SIGMA_GLOBAL))
        inside0 = center_error.abs() <= half0
        reward0 = inside0.to(torch.float32) * tightness * valid.to(torch.float32) * accuracy
        baseline_numerator[tr] += float(reward0.sum(dtype=torch.float64))
        baseline_covered[tr] += int((inside0 & valid).sum())
        valid_count[tr] += int(valid.sum())

        prediction_bank = []
        forecast_cache: dict[tuple[Any, ...], np.ndarray] = {}
        for candidate_model, damping, _width_scale in candidates:
            key = tuple(candidate_model["key"])
            forecast = forecast_cache.get(key)
            if forecast is None:
                beta = coefficients[key][tr]
                expanded = expand_beta(beta)
                forecast = np.einsum("thwcp,thwcp->thwc", x, expanded, optimize=True)
                forecast = np.clip(forecast, -PREDICTION_CLIP, PREDICTION_CLIP)
                forecast_cache[key] = forecast
            prediction_bank.append(float(damping) * forecast)
        shift = torch.from_numpy(np.stack(prediction_bank)).to(torch.float32)
        width_scale = torch.tensor(
            [value[2] for value in candidates], dtype=torch.float32
        ).reshape(-1, 1, 1, 1, 1)
        candidate_half = half0.unsqueeze(0) * width_scale
        inside = (center_error.unsqueeze(0) - shift).abs() <= candidate_half
        candidate_tightness = torch.exp(
            -2.0
            * candidate_half
            * data.std_target[..., :2].squeeze(0).unsqueeze(0)
            / float(scoring.SIGMA_GLOBAL)
        )
        reward = (
            inside.to(torch.float32)
            * candidate_tightness
            * valid.unsqueeze(0).to(torch.float32)
            * accuracy
        )
        numerator[:, tr] += reward.sum(dim=(1, 2, 3, 4), dtype=torch.float64).numpy()
        covered[:, tr] += (inside & valid.unsqueeze(0)).sum(dim=(1, 2, 3, 4), dtype=torch.int64).numpy()
        previous_target = data.targets[index].unsqueeze(0)
        if progress_every and (index + 1) % progress_every == 0:
            print(f"eval_windows={index + 1} elapsed={time.perf_counter()-start:.1f}s", flush=True)
    if base.index != data.targets.shape[0]:
        raise RuntimeError("evaluation pass did not consume full cache")

    names = np.asarray(data.trajectory_ids)
    dev = np.asarray([name in DEV_IDS for name in names])
    angles = np.asarray([int(name.removesuffix(".h5").rsplit("_", 1)[1]) for name in names])
    baseline_sps = 100.0 * baseline_numerator.sum() / valid_count.sum()
    baseline_trajectory = 100.0 * baseline_numerator / valid_count
    results: list[dict[str, Any]] = []
    for ci, (candidate_model, damping, width_scale) in enumerate(candidates):
        trajectory_sps = 100.0 * numerator[ci] / valid_count
        delta = trajectory_sps - baseline_trajectory
        item = {
            "name": f"zone{candidate_model['partition'][0]}x{candidate_model['partition'][1]}_clip{candidate_model['response_clip']:g}_ridge{candidate_model['ridge']:g}_damp{damping:g}_width{width_scale:g}",
            "model": candidate_model,
            "damping": damping,
            "width_scale": width_scale,
            "global": group_summary(numerator[ci], covered[ci], valid_count, baseline_numerator, baseline_covered, np.ones(nt, dtype=bool)),
            "dev10": group_summary(numerator[ci], covered[ci], valid_count, baseline_numerator, baseline_covered, dev),
            "hold20": group_summary(numerator[ci], covered[ci], valid_count, baseline_numerator, baseline_covered, ~dev),
            "angles": {str(a): group_summary(numerator[ci], covered[ci], valid_count, baseline_numerator, baseline_covered, angles == a) for a in sorted(set(angles))},
            "robustness": {
                "mean_delta": float(delta.mean()),
                "std_delta": float(delta.std()),
                "robust_025": float(delta.mean() - 0.25 * delta.std()),
                "robust_050": float(delta.mean() - 0.50 * delta.std()),
                "best_delta": float(delta.max()),
                "worst_delta": float(delta.min()),
                "wins": int((delta > 0).sum()),
                "losses": int((delta < 0).sum()),
            },
            "trajectory_delta": {name: float(value) for name, value in zip(names.tolist(), delta.tolist())},
        }
        results.append(item)
    results.sort(key=lambda item: item["global"]["delta"], reverse=True)
    baseline = {
        "sps": float(baseline_sps),
        "coverage": float(baseline_covered.sum() / valid_count.sum()),
        "rel_l2_score": float(scoring.score_error(prediction_error_sum[0] / data.targets.shape[0])),
        "tke_score": float(scoring.score_error(prediction_error_sum[1] / data.targets.shape[0])),
        "mvpe_score": float(scoring.score_error(prediction_error_sum[2] / data.targets.shape[0])),
        "valid_elements": int(valid_count.sum()),
    }
    return results, baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    data = load_data()
    print(f"neural_loaded windows={len(data.simulation_ids)} trajectories={len(data.trajectory_ids)}", flush=True)
    stats = collect_stats(data, args.progress_every)
    selected, coefficients = fit_and_rank(stats, data.trajectory_ids)
    print("neural_selected_models", flush=True)
    for item in selected:
        print(json.dumps(item["key"]), flush=True)
    results, baseline = evaluate(data, selected, coefficients, args.progress_every)
    seeds = [results[0]]
    seeds.append(max(results, key=lambda x: x["robustness"]["robust_050"]))
    all_win = [x for x in results if x["robustness"]["losses"] == 0]
    if all_win:
        seeds.append(max(all_win, key=lambda x: x["global"]["delta"]))
    unique_seeds: list[dict[str, Any]] = []
    seen_seed: set[tuple[Any, ...]] = set()
    for seed in seeds:
        marker = (tuple(seed["model"]["key"]), seed["damping"])
        if marker not in seen_seed:
            unique_seeds.append(seed)
            seen_seed.add(marker)
    refined_bank = [
        (seed["model"], float(seed["damping"]), float(width))
        for seed in unique_seeds
        for width in WIDTH_SCALES
    ]
    refined_results, refined_baseline = evaluate(
        data,
        selected,
        coefficients,
        args.progress_every,
        explicit_candidates=refined_bank,
    )
    for key in baseline:
        if isinstance(baseline[key], float) and abs(baseline[key] - refined_baseline[key]) > 1.0e-8:
            raise RuntimeError(f"baseline drift between center and width passes: {key}")
    report = {
        "schema_version": 1,
        "experiment": "learned causal interval residual forecast",
        "status": "completed",
        "data": {
            "source": 'train_real cohort caches',
            "windows": len(data.simulation_ids),
            "trajectories": len(data.trajectory_ids),
            "trajectory_ids": list(data.trajectory_ids),
        },
        "causality": {
            "features": list(FEATURE_NAMES),
            "labels": 'training statistics and evaluation',
            "crossfit": 'leave-one-trajectory-out',
        },
        "search": {
            "fine_partition": [FINE_Y_BINS, FINE_X_BINS],
            "partitions": [list(x) for x in PARTITIONS],
            "response_clips": list(TRAIN_RESPONSE_CLIPS),
            "ridges": list(RIDGES),
            "dampings": list(DAMPINGS),
            "width_scales": list(WIDTH_SCALES),
            "prediction_clip": PREDICTION_CLIP,
            "selected_model_count": len(selected),
            "evaluated_candidate_count": len(results),
        },
        "baseline_ridge": baseline,
        "selected_by_oof_mse": selected,
        "top_global": results[:20],
        "top_robust_050": sorted(results, key=lambda x: x["robustness"]["robust_050"], reverse=True)[:20],
        "all_candidates": results,
        "width_refinement_seeds": [seed["name"] for seed in unique_seeds],
        "width_refinement_top_global": refined_results[:20],
        "width_refinement_top_robust_050": sorted(refined_results, key=lambda x: x["robustness"]["robust_050"], reverse=True)[:20],
        "width_refinement_all_candidates": refined_results,
        "duration_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"neural_report={args.output}", flush=True)


if __name__ == "__main__":
    main()
