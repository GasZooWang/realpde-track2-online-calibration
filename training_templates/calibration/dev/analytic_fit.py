"""Analytic calibration fit."""

from __future__ import annotations
import math
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCORING_RELATIVE_PATH = "starting_kit/scoring.py"
INTERVAL_FEATURE_INDICES = np.asarray((0, 5, 6, 7), dtype=np.int64)
INTERVAL_FEATURE_NAMES = (
    "intercept",
    "input_aligned_minus_center",
    "input_last_minus_center",
    "linear_extrapolation_minus_center",
)
INTERVAL_RIDGE = 1.0e-5
INTERVAL_RESPONSE_CLIP_INDEX = 1
TEMPORAL_FAMILY = "same_full20"
TEMPORAL_PARTITION = (2, 4)
TEMPORAL_RIDGE = 1.0e-4
PREDICTION_COUPLING = 1.30
BOUNDS_TOTAL_DAMPING = 1.20
FORECAST_CLIP = 0.75
BASE_WIDTH = 0.95
WIDTH_GRID = np.round(np.arange(0.60, 1.3001, 0.0125), 4)
WIDTH_CONFIG = {
    "time_bins": 20,
    "y_bins": 4,
    "x_bins": 8,
    "shrinkage": 1.0,
}
PIXEL_PREVIOUS_WEIGHT = 0.50
PIXEL_STATE_DECAY = 0.80
PIXEL_DAMPING = 0.35
TKE_CANDIDATE = {
    "channel_mode": "per_channel",
    "zones": "zone16x32",
    "features": "full",
    "weight": "uniform",
    "damping": 0.80,
}
TKE_RIDGE = 0.005
TKE_STATE_DECAY = 0.80
TKE_RESPONSE_CLIP = 0.60
TKE_PREDICTION_CLIP = 0.35
TKE_SCALE_BOUNDS = (0.75, 4.0 / 3.0)
TKE_EPS = 1.0e-10
TKE_FEATURE_NAMES = (
    "intercept",
    "current_input_over_prediction_log_scale",
    "prediction_log_energy_level",
    "input_late_over_early_log_scale",
    "matured_previous_target_ratio_ema",
    "matured_state_available",
)


def fit_interval_and_loto(
    stats: Mapping[str, np.ndarray], names: Sequence[str]
) -> tuple[dict[str, Any], np.ndarray]:
    from refine_feature_ablation import (
        fit_loto,
        solve_coefficients_subset,
    )

    xtx_full = np.asarray(stats["xtx"]).sum(axis=0)
    xty_full = np.asarray(stats["xty"])[
        :, INTERVAL_RESPONSE_CLIP_INDEX
    ].sum(axis=0)
    count = np.asarray(stats["count"]).sum(axis=0)
    ix = INTERVAL_FEATURE_INDICES
    xtx = xtx_full[..., ix[:, None], ix]
    xty = xty_full[..., ix]
    beta = solve_coefficients_subset(
        xtx, xty, count, INTERVAL_RIDGE
    ).astype(np.float32)
    if beta.shape != (20, 2, 4, 8, 4) or not np.isfinite(beta).all():
        raise RuntimeError(f"invalid fixed learned_interval beta: {beta.shape}")
    loto = fit_loto(dict(stats), tuple(int(value) for value in ix))
    if loto.shape != (len(names), 20, 2, 4, 8, 8):
        raise RuntimeError(f"invalid learned_interval LOTO bank: {loto.shape}")
    if not np.isfinite(loto).all():
        raise RuntimeError("learned_interval LOTO bank is non-finite")
    payload = {
        "schema_version": 1,
        "source": 'fixed-family refit',
        "trajectory_ids": list(names),
        "selection": 'input-only family',
        "partition": (4, 8),
        "response_clip": 0.5,
        "ridge": INTERVAL_RIDGE,
        "feature_names": INTERVAL_FEATURE_NAMES,
        "coefficient_layout": (
            "(target_horizon, channel, y_bin, x_bin, feature)"
        ),
        "coefficients": torch.from_numpy(beta),
    }
    return payload, loto


def fit_temporal_from_records(
    records: Sequence[Mapping[str, Any]], names: Sequence[str]
) -> dict[str, Any]:
    from refine_history_forecast import (
        effective_max_beta,
        expand_to_fine,
        make_families,
        solve_coefficients,
        transform_stats,
    )

    family = next(
        value for value in make_families() if value.name == TEMPORAL_FAMILY
    )
    total_xx: np.ndarray | None = None
    total_xy: np.ndarray | None = None
    total_count: np.ndarray | None = None
    for record in records:
        xtx, xty, _y2, count = transform_stats(
            dict(record), family, TEMPORAL_PARTITION
        )
        if xtx.shape[0] != 1:
            raise RuntimeError("temporal statistic shard must contain one trajectory")
        total_xx = xtx[0].copy() if total_xx is None else total_xx + xtx[0]
        total_xy = xty[0].copy() if total_xy is None else total_xy + xty[0]
        total_count = (
            count[0].copy() if total_count is None else total_count + count[0]
        )
    assert total_xx is not None and total_xy is not None and total_count is not None
    native = solve_coefficients(
        total_xx, total_xy, total_count, TEMPORAL_RIDGE
    )
    fine = expand_to_fine(native, TEMPORAL_PARTITION)
    effective = effective_max_beta(fine, family)
    if native.shape != (20, 2, 2, 4, 22) or not np.isfinite(native).all():
        raise RuntimeError(f"invalid temporal native beta: {native.shape}")
    if effective.shape != (20, 2, 4, 8, 42) or not np.isfinite(effective).all():
        raise RuntimeError(f"invalid temporal effective beta: {effective.shape}")
    return {
        "schema_version": 1,
        "method": "same_full20 family refit",
        "feature_system": "[1,-center,same_history20,cross_history20]",
        "family": family.name,
        "family_description": family.description,
        "partition": list(TEMPORAL_PARTITION),
        "ridge": TEMPORAL_RIDGE,
        "response_clip": 0.5,
        "forecast_clip": FORECAST_CLIP,
        "transform": torch.from_numpy(family.transform.astype(np.float32)),
        "native_beta": torch.from_numpy(native),
        "effective_beta_fine4x8": torch.from_numpy(effective),
        "trajectory_ids": list(names),
        "joint_calibration": {
            "prediction_coupling": PREDICTION_COUPLING,
            "bounds_total_damping": BOUNDS_TOTAL_DAMPING,
            "width_scale": BASE_WIDTH,
            "pixel_previous_weight": PIXEL_PREVIOUS_WEIGHT,
            "pixel_state_decay": PIXEL_STATE_DECAY,
            "pixel_damping": PIXEL_DAMPING,
        },
        "selection": 'fixed family and hyperparameters',
    }


def collect_width_reward(data: Any, loto_beta: np.ndarray) -> dict[str, np.ndarray]:

    from refine_width_map import expand_beta
    from refine_learned_interval import (
        feature_tensor,
        new_model,
        target_raw,
    )
    from refine_joint_grid import vector_errors

    scoring = sys.modules.get("scoring")
    if scoring is None or Path(str(getattr(scoring, "__file__", ""))).resolve() != (
        ROOT / SCORING_RELATIVE_PATH
    ).resolve():
        raise RuntimeError("scoring module is not loaded")

    nwidth = len(WIDTH_GRID)
    fine_reward = np.zeros((20, 2, 4, 8, nwidth), dtype=np.float64)
    fine_covered = np.zeros_like(fine_reward, dtype=np.int64)
    width_values = torch.tensor(WIDTH_GRID, dtype=torch.float32).reshape(
        nwidth, 1, 1, 1, 1
    )
    base, ridge = new_model(data)
    previous_target = None
    previous_ridge_returned = None
    pixel_state = None
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
        forecast = np.einsum(
            "thwcp,thwcp->thwc",
            features,
            expand_beta(loto_beta),
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
            pixel_state = (
                observed.clone()
                if pixel_state is None
                else PIXEL_STATE_DECAY * pixel_state
                + (1.0 - PIXEL_STATE_DECAY) * observed
            )

        returned_uv = returned[..., :2]
        candidate_uv = returned_uv + PREDICTION_COUPLING * forecast_t
        returned_deviation = returned_uv - returned_uv.mean(
            dim=1, keepdim=True
        )
        current_input_physical = (
            data.inputs[index].unsqueeze(0)[..., :2]
            * data.std_input[..., :2]
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
        factors = 0.5 / (0.5 + errors)
        accuracy = float(
            0.5 * factors[0] + 0.3 * factors[1] + 0.2 * factors[2]
        )

        target_norm = data.targets[index].unsqueeze(0)[..., :2]
        learned_center = center + BOUNDS_TOTAL_DAMPING * forecast_t
        residual = (target_norm - learned_center).abs().squeeze(0)
        half0 = half.squeeze(0)
        valid = (target[..., :2] != 0.0).squeeze(0)
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
        fine_reward += reward_group.numpy().astype(np.float64)
        fine_covered += covered_group.numpy().astype(np.int64)
        previous_target = data.targets[index].unsqueeze(0)
        previous_ridge_returned = returned_uv
    if base.index != data.targets.shape[0]:
        raise RuntimeError("width replay did not consume the complete base cache")
    return {"reward": fine_reward, "covered": fine_covered}


def fit_width(
    rewards: np.ndarray,
    covered: np.ndarray,
    names: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_shape = (len(names), 20, 2, 4, 8, len(WIDTH_GRID))
    if rewards.shape != expected_shape or rewards.dtype != np.float64:
        raise RuntimeError(f"invalid width reward bank: {rewards.shape}")
    if covered.shape != expected_shape or covered.dtype != np.int64:
        raise RuntimeError(f"invalid width coverage bank: {covered.shape}")
    if (
        not np.isfinite(rewards).all()
        or (rewards < 0).any()
        or (covered < 0).any()
        or (np.diff(covered, axis=-1) < 0).any()
        or (rewards > covered.astype(np.float64) + 1.0e-9).any()
    ):
        raise RuntimeError("invalid numeric width reward/coverage bank")
    if not (covered.sum(axis=(0, -1)) > 0).all():
        raise RuntimeError("width estimator has an uncovered deployment cell")
    if not (rewards.sum(axis=(0, -1)) > 0).all():
        raise RuntimeError("width estimator has a zero-reward deployment cell")
    total = rewards.sum(axis=0)
    if not np.isfinite(total).all():
        raise RuntimeError("non-finite aggregate width reward")
    deploy_index = total.argmax(axis=-1)
    deploy = WIDTH_GRID[deploy_index].astype(np.float32)
    loto_rows = []
    for trajectory in range(len(names)):
        held_out = total - rewards[trajectory]
        if not (held_out.sum(axis=-1) > 0).all():
            raise RuntimeError("width LOTO estimator has a zero-reward cell")
        loto_index = held_out.argmax(axis=-1)
        loto_rows.append(WIDTH_GRID[loto_index].astype(np.float32))
    loto = np.stack(loto_rows)
    full_payload = {
        "width_scale": torch.from_numpy(deploy),
        "loto_width_scale": torch.from_numpy(loto),
        "trajectory_ids": list(names),
        "base_width": BASE_WIDTH,
        "selection": (
            '20x4x8 shrinkage1 SPS map'
        ),
        "config": dict(WIDTH_CONFIG),
        "grid": WIDTH_GRID.tolist(),
        "training_anchor": (
            "fixed input-only four-feature trajectory-LOTO forecast"
        ),
        "covered_count": int(covered.sum()),
    }
    package_payload = {
        key: full_payload[key]
        for key in ("width_scale", "base_width", "selection", "config", "grid")
    }
    return full_payload, package_payload


def collect_tke_sufficient_stats(
    data: Any, temporal_effective: np.ndarray
) -> dict[str, np.ndarray]:

    from refine_learned_interval import new_model, target_raw
    from refine_history_forecast import (
        expand_effective,
        max_feature_tensor,
    )

    effective = np.asarray(temporal_effective, dtype=np.float32)
    if effective.shape != (20, 2, 4, 8, 42):
        raise RuntimeError(f"invalid temporal effective tensor: {effective.shape}")
    expanded_beta = expand_effective(effective[None])[0]
    entities, features = 16 * 32 * 2, len(TKE_FEATURE_NAMES)
    xtx = np.zeros((entities, features, features), dtype=np.float64)
    xty = np.zeros((entities, features), dtype=np.float64)
    count = np.zeros(entities, dtype=np.float64)

    base, ridge = new_model(data)
    previous_target = None
    previous_ridge_returned = None
    pixel_state = None
    supervised_channel_state: np.ndarray | None = None
    for index in range(data.targets.shape[0]):
        if data.first[index]:
            ridge.reset_ttt_state()
            previous_target = None
            previous_ridge_returned = None
            pixel_state = None
            supervised_channel_state = None
        returned, info = ridge.ttt_step(
            data.inputs[index].unsqueeze(0), previous_target
        )
        returned = returned.detach().cpu()
        center = 0.5 * (
            info["lower"][..., :2].detach().cpu()
            + info["upper"][..., :2].detach().cpu()
        )
        xmax = max_feature_tensor(data, index, center).numpy()
        forecast = np.einsum(
            "thwcm,thwcm->thwc", xmax, expanded_beta, optimize=True
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
            prediction_variance0 = (
                prediction_deviation * data.std_target[..., :2]
            ).square().mean(dim=1)[0]
            target_variance0 = target_deviation.square().mean(dim=1)[0]
            observed = 0.5 * torch.log(
                (target_variance0 + 1.0e-8)
                / (prediction_variance0 + 1.0e-8)
            ).clamp(math.log(0.5), math.log(2.0))
            pixel_state = (
                observed.clone()
                if pixel_state is None
                else PIXEL_STATE_DECAY * pixel_state
                + (1.0 - PIXEL_STATE_DECAY) * observed
            )

        returned_uv = returned[..., :2]
        candidate_uv = returned_uv + PREDICTION_COUPLING * forecast_t
        returned_deviation = returned_uv - returned_uv.mean(
            dim=1, keepdim=True
        )
        input_physical_t = (
            data.inputs[index].unsqueeze(0)[..., :2]
            * data.std_input[..., :2]
        )
        input_deviation_t = input_physical_t - input_physical_t.mean(
            dim=1, keepdim=True
        )
        returned_variance = (
            returned_deviation * data.std_target[..., :2]
        ).square().mean(dim=1)[0]
        input_variance = input_deviation_t.square().mean(dim=1)[0]
        current_ratio = 0.5 * torch.log(
            (input_variance + 1.0e-8) / (returned_variance + 1.0e-8)
        ).clamp(math.log(0.5), math.log(2.0))
        if pixel_state is not None:
            blended = (
                PIXEL_PREVIOUS_WEIGHT * pixel_state
                + (1.0 - PIXEL_PREVIOUS_WEIGHT) * current_ratio
            )
            pixel_scale = torch.exp(PIXEL_DAMPING * blended).clamp(0.8, 1.25)
            candidate_mean = candidate_uv.mean(dim=1, keepdim=True)
            candidate_uv = candidate_mean + pixel_scale[None, None] * (
                candidate_uv - candidate_mean
            )

        prediction = (
            candidate_uv * data.std_target[..., :2]
            + data.mean_target[..., :2]
        ).numpy()[0].astype(np.float32)
        target = target_raw(data, index)[..., :2].numpy()[0].astype(np.float32)
        current_input = (
            data.inputs[index:index + 1, ..., :2] * data.std_input[..., :2]
            + data.mean_input[..., :2]
        ).numpy()[0].astype(np.float32)
        prediction_mean = prediction.mean(axis=0, keepdims=True)
        target_mean = target.mean(axis=0, keepdims=True)
        input_mean = current_input.mean(axis=0, keepdims=True)
        prediction_dev = prediction - prediction_mean
        target_dev = target - target_mean
        input_dev = current_input - input_mean
        vp = np.mean(prediction_dev * prediction_dev, axis=0)
        vt = np.mean(target_dev * target_dev, axis=0)
        vi = np.mean(input_dev * input_dev, axis=0)
        channel_early = np.mean(input_dev[:5] * input_dev[:5], axis=0)
        channel_late = np.mean(input_dev[-5:] * input_dev[-5:], axis=0)
        channel_y = np.clip(
            0.5 * np.log((vt + TKE_EPS) / (vp + TKE_EPS)),
            -TKE_RESPONSE_CLIP,
            TKE_RESPONSE_CLIP,
        )
        x = np.empty((32, 64, 2, features), dtype=np.float32)
        x[..., 0] = 1.0
        x[..., 1] = np.clip(
            0.5 * np.log((vi + TKE_EPS) / (vp + TKE_EPS)), -0.8, 0.8
        )
        x[..., 2] = np.clip(
            np.log((vp + TKE_EPS) / 1.0e-4) / 3.0, -1.0, 1.0
        )
        x[..., 3] = np.clip(
            0.5 * np.log(
                (channel_late + TKE_EPS) / (channel_early + TKE_EPS)
            ),
            -1.0,
            1.0,
        )
        x[..., 4] = (
            0.0 if supervised_channel_state is None else supervised_channel_state
        )
        x[..., 5] = 0.0 if supervised_channel_state is None else 1.0

        x64 = x.astype(np.float64, copy=False)
        channel_y64 = channel_y.astype(np.float64, copy=False)
        pixel_xx = np.einsum("hwcf,hwcg->hwcfg", x64, x64, optimize=True)
        pixel_xy = np.einsum(
            "hwcf,hwc->hwcf", x64, channel_y64, optimize=True
        )
        zone_xx = pixel_xx.reshape(16, 2, 32, 2, 2, features, features).sum(
            axis=(1, 3)
        )
        zone_xy = pixel_xy.reshape(16, 2, 32, 2, 2, features).sum(
            axis=(1, 3)
        )
        xtx += zone_xx.reshape(entities, features, features)
        xty += zone_xy.reshape(entities, features)
        count += 4.0

        supervised_channel_state = (
            channel_y.copy()
            if supervised_channel_state is None
            else TKE_STATE_DECAY * supervised_channel_state
            + (1.0 - TKE_STATE_DECAY) * channel_y
        )
        previous_target = data.targets[index].unsqueeze(0)
        previous_ridge_returned = returned_uv
    if base.index != data.targets.shape[0]:
        raise RuntimeError("TKE replay did not consume the complete base cache")
    return {"xtx": xtx, "xty": xty, "count": count}


def fit_tke(
    records: Sequence[Mapping[str, np.ndarray]], names: Sequence[str]
) -> dict[str, Any]:
    total_xx = np.sum([np.asarray(value["xtx"]) for value in records], axis=0)
    total_xy = np.sum([np.asarray(value["xty"]) for value in records], axis=0)
    total_count = np.sum(
        [np.asarray(value["count"]) for value in records], axis=0
    )
    if (
        total_xx.shape != (1024, 6, 6)
        or total_xy.shape != (1024, 6)
        or total_count.shape != (1024,)
        or not np.isfinite(total_xx).all()
        or not np.isfinite(total_xy).all()
        or not np.isfinite(total_count).all()
        or (total_count <= 0).any()
    ):
        raise RuntimeError("invalid aggregate TKE sufficient statistics")
    features = len(TKE_FEATURE_NAMES)
    penalty = np.eye(features, dtype=np.float64) * TKE_RIDGE
    penalty[0, 0] = TKE_RIDGE * 0.05
    coefficients = np.zeros((16 * 32 * 2, features), dtype=np.float64)
    for entity in range(coefficients.shape[0]):
        matrix = total_xx[entity] / max(total_count[entity], 1.0) + penalty
        vector = total_xy[entity] / max(total_count[entity], 1.0)
        coefficients[entity] = np.linalg.solve(matrix, vector)
    coefficients = coefficients.astype(np.float32)
    if coefficients.shape != (1024, 6) or not np.isfinite(coefficients).all():
        raise RuntimeError(f"invalid supervised TKE coefficients: {coefficients.shape}")
    return {
        "schema_version": 1,
        "source": "fixed-family refit on deployed full20 temporal anchor",
        "candidate": dict(TKE_CANDIDATE),
        "candidate_name": "per_channel__zone16x32__full__uniform__d0.80",
        "feature_names": TKE_FEATURE_NAMES,
        "feature_indices": (0, 1, 2, 3, 4, 5),
        "zone_layout": (16, 32),
        "state_decay": TKE_STATE_DECAY,
        "prediction_clip": TKE_PREDICTION_CLIP,
        "scale_bounds": TKE_SCALE_BOUNDS,
        "ridge": TKE_RIDGE,
        "coefficients": torch.from_numpy(coefficients),
        "trajectory_ids": tuple(names),
        "selection": 'fixed family and damping',
    }
