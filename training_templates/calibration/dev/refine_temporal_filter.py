"""Temporal filter."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from refine_interval_recalibration import (
    PROTOTYPE,
    ReplayBase,
    load_submission,
    payload_tensor,
    score_group,
    subtract,
)
from refine_learned_interval import DEV_IDS, load_data, target_raw
from refine_joint_grid import vector_errors

import scoring


ROOT = Path(__file__).resolve().parents[1]
ANCHOR_CACHE = ROOT / "dev/cache/temporal_filter_anchor.pt"
STATS_CACHE = ROOT / "dev/cache/temporal_filter_stats.pt"
REPORT = ROOT / "dev/reports/temporal_filter.json"
ARTIFACT = ROOT / "dev/cache/temporal_filter.pt"
TIME_SCORE = 85.066771
FINAL_POWER = 0.6854
TARGET_GAIN = 0.04
FINE_PARTITION = (4, 8)
PARTITIONS = ((1, 1), (2, 4), (4, 8))
DAMPINGS = (0.25, 0.50, 0.75, 1.00)
EXPECTED_BASELINE = {
    "rel_l2_score": 95.97332674479573,
    "tke_score": 78.16875714776761,
    "mvpe_score": 96.45360450822537,
    "sps_score": 51.095957870402685,
    "final_proxy": 83.51802512196883,
}


@dataclass(frozen=True)
class FilterSpec:
    family: str
    ridge: float

    @property
    def name(self) -> str:
        return f"{self.family}__ridge{self.ridge:g}"


COMPACT_SPECS = (
    FilterSpec("scalar", 0.0),
    FilterSpec("lag1_symmetric", 1.0e-3),
    FilterSpec("toeplitz3", 1.0e-3),
    FilterSpec("toeplitz3", 1.0e-2),
    FilterSpec("toeplitz5", 1.0e-3),
    FilterSpec("toeplitz5", 1.0e-2),
    FilterSpec("dct_diag4", 1.0e-3),
    FilterSpec("dct_diag8", 1.0e-3),
    FilterSpec("dct_diag20", 1.0e-3),
)
FULL_SPECS = (
    FilterSpec("full20", 1.0e-3),
    FilterSpec("full20", 1.0e-2),
)
QUICK_SPECS = (
    FilterSpec("scalar", 0.0),
    FilterSpec("lag1_symmetric", 1.0e-3),
    FilterSpec("toeplitz3", 1.0e-3),
    FilterSpec("dct_diag4", 1.0e-3),
    FilterSpec("dct_diag8", 1.0e-3),
    FilterSpec("dct_diag20", 1.0e-3),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_signature(root: Path) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for name in (
        "submission.py",
        "learned_interval.pt",
        "learned_width.pt",
        "supervised_tke.pt",
        "temporal_history.pt",
    ):
        stat = (root / name).stat()
        result[name] = [int(stat.st_size), int(stat.st_mtime_ns)]
    return result


def dct_non_dc() -> np.ndarray:
    t = np.arange(20, dtype=np.float64)
    k = np.arange(20, dtype=np.float64)[:, None]
    basis = np.sqrt(2.0 / 20.0) * np.cos(np.pi * (t + 0.5) * k / 20.0)
    basis[0] = math.sqrt(1.0 / 20.0)
    if not np.allclose(basis @ basis.T, np.eye(20), atol=1.0e-12):
        raise RuntimeError("DCT basis is not orthonormal")
    return basis[1:]


DCT = dct_non_dc()


def final_proxy(rel: float, tke: float, mvpe: float, sps: float) -> float:
    total = (
        0.5 * rel**FINAL_POWER
        + 0.1 * tke**FINAL_POWER
        + 0.1 * mvpe**FINAL_POWER
        + 0.1 * TIME_SCORE**FINAL_POWER
        + 0.2 * sps**FINAL_POWER
    )
    return float(total ** (1.0 / FINAL_POWER))


def anchor_model(data: Any) -> tuple[ReplayBase, Any]:
    module = load_submission(PROTOTYPE.resolve())
    base = ReplayBase(data.base)
    model = module.CausalRidgeTTTModel(
        base,
        "cpu",
        payload_tensor(PROTOTYPE / "learned_interval.pt", "coefficients"),
        payload_tensor(PROTOTYPE / "learned_width.pt", "width_scale"),
        payload_tensor(PROTOTYPE / "supervised_tke.pt", "coefficients"),
        payload_tensor(PROTOTYPE / "temporal_history.pt", "native_beta"),
    )
    return base, model


def build_anchor_cache(data: Any, output: Path, progress_every: int) -> dict[str, Any]:
    n = data.targets.shape[0]
    prediction = torch.empty((n, 20, 32, 64, 2), dtype=torch.float32)
    error = np.empty((n, 3), dtype=np.float64)
    sps_kernel = np.empty(n, dtype=np.float64)
    covered = np.empty(n, dtype=np.int64)
    valid_count = np.empty(n, dtype=np.int64)
    base, model = anchor_model(data)
    previous_target = None
    started = time.perf_counter()
    for index in range(n):
        if data.first[index]:
            model.reset_ttt_state()
            previous_target = None
        returned, info = model.ttt_step(
            data.inputs[index].unsqueeze(0), previous_target
        )
        returned = returned.detach().cpu()
        prediction[index].copy_(returned[0, ..., :2])
        target = target_raw(data, index)
        prediction_raw = returned * data.std_target + data.mean_target
        errors = vector_errors(prediction_raw, target)[0]
        error[index] = errors.numpy().astype(np.float64)

        lower = info["lower"][..., :2].detach().cpu()
        upper = info["upper"][..., :2].detach().cpu()
        center = 0.5 * (lower + upper)
        half_raw = 0.5 * (upper - lower) * data.std_target[..., :2]
        target_uv = target[..., :2]
        target_norm_uv = data.targets[index].unsqueeze(0)[..., :2]
        valid = target_uv != 0.0
        inside = (target_norm_uv - center).abs() <= 0.5 * (upper - lower)
        tightness = torch.exp(-2.0 * half_raw / float(scoring.SIGMA_GLOBAL))
        kernel = inside * valid * tightness
        sps_kernel[index] = float(kernel.sum(dtype=torch.float64))
        covered[index] = int((inside & valid).sum())
        valid_count[index] = int(valid.sum())
        previous_target = data.targets[index].unsqueeze(0)
        if progress_every and (index + 1) % progress_every == 0:
            print(
                f"temporal_anchor_windows={index + 1} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    if base.index != n:
        raise RuntimeError("anchor replay did not consume the complete base cache")
    result = {
        "schema_version": 1,
        "source": 'calibrated replay',
        "artifact_signature": artifact_signature(PROTOTYPE.resolve()),
        "simulation_id": data.simulation_ids,
        "trajectory_first": torch.from_numpy(data.first.copy()),
        "prediction_norm_uv": prediction,
        "baseline_error": torch.from_numpy(error),
        "fixed_bounds_sps_kernel": torch.from_numpy(sps_kernel),
        "fixed_bounds_covered": torch.from_numpy(covered),
        "valid_count": torch.from_numpy(valid_count),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    print(f"saved_anchor_cache={output}", flush=True)
    return result


def load_or_build_anchor(
    data: Any, output: Path, rebuild: bool, progress_every: int
) -> dict[str, Any]:
    signature = artifact_signature(PROTOTYPE.resolve())
    if output.exists() and not rebuild:
        value = torch.load(output, map_location="cpu", weights_only=False)
        if (
            value.get("artifact_signature") == signature
            and tuple(value.get("simulation_id", ())) == data.simulation_ids
        ):
            print(f"loaded_anchor_cache={output}", flush=True)
            return value
        print("anchor_cache_signature_mismatch_rebuilding", flush=True)
    return build_anchor_cache(data, output, progress_every)


def toeplitz5_features(samples: np.ndarray) -> np.ndarray:
    padded = np.pad(samples, ((0, 0), (2, 2)), mode="edge")
    features = np.stack(
        [padded[:, offset:offset + 20] for offset in range(5)], axis=-1
    )
    features -= features.mean(axis=1, keepdims=True)
    return features


def collect_stats(
    data: Any, anchor: Mapping[str, Any], output: Path, rebuild: bool
) -> dict[str, Any]:
    signature = artifact_signature(PROTOTYPE.resolve())
    if output.exists() and not rebuild:
        cached = torch.load(output, map_location="cpu", weights_only=False)
        if (
            cached.get("artifact_signature") == signature
            and tuple(cached.get("trajectory_ids", ())) == data.trajectory_ids
        ):
            print(f"loaded_filter_stats={output}", flush=True)
            return cached

    ntraj = len(data.trajectory_ids)
    toe_xx = np.zeros((ntraj, 2, 4, 8, 5, 5), dtype=np.float64)
    toe_xy = np.zeros((ntraj, 2, 4, 8, 5), dtype=np.float64)
    dct_xx = np.zeros((ntraj, 2, 4, 8, 19, 19), dtype=np.float64)
    dct_xy = np.zeros((ntraj, 2, 4, 8, 19, 19), dtype=np.float64)
    sample_count = np.zeros((ntraj, 2, 4, 8), dtype=np.int64)
    prediction = torch.as_tensor(anchor["prediction_norm_uv"]).numpy()
    target = data.targets[..., :2].numpy()
    started = time.perf_counter()
    for tr in range(ntraj):
        selector = data.trajectory_index == tr
        x_window = prediction[selector]
        y_window = target[selector]
        x_window = x_window - x_window.mean(axis=1, keepdims=True)
        y_window = y_window - y_window.mean(axis=1, keepdims=True)
        for y_zone in range(4):
            ys = slice(8 * y_zone, 8 * (y_zone + 1))
            for x_zone in range(8):
                xs = slice(8 * x_zone, 8 * (x_zone + 1))
                for channel in range(2):
                    x = x_window[:, :, ys, xs, channel]
                    y = y_window[:, :, ys, xs, channel]
                    x = np.moveaxis(x, 1, -1).reshape(-1, 20).astype(
                        np.float64, copy=False
                    )
                    y = np.moveaxis(y, 1, -1).reshape(-1, 20).astype(
                        np.float64, copy=False
                    )
                    features = toeplitz5_features(x).reshape(-1, 5)
                    response = y.reshape(-1)
                    toe_xx[tr, channel, y_zone, x_zone] = features.T @ features
                    toe_xy[tr, channel, y_zone, x_zone] = features.T @ response
                    xm = x @ DCT.T
                    ym = y @ DCT.T
                    dct_xx[tr, channel, y_zone, x_zone] = xm.T @ xm
                    dct_xy[tr, channel, y_zone, x_zone] = xm.T @ ym
                    sample_count[tr, channel, y_zone, x_zone] = x.shape[0]
        print(
            f"temporal_stats_trajectory={tr + 1}/{ntraj} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    result = {
        "schema_version": 1,
        "source": "calibrated/target deviation sufficient statistics",
        "artifact_signature": signature,
        "trajectory_ids": data.trajectory_ids,
        "fine_partition": FINE_PARTITION,
        "toeplitz5_xtx": toe_xx,
        "toeplitz5_xty": toe_xy,
        "dct_xtx": dct_xx,
        "dct_xty": dct_xy,
        "sample_count": sample_count,
        "dct_non_dc": DCT,
        "toeplitz_boundary": "edge replication followed by temporal projection",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    print(f"saved_filter_stats={output}", flush=True)
    return result


def aggregate_partition(value: np.ndarray, partition: tuple[int, int]) -> np.ndarray:
    y_bins, x_bins = partition
    if 4 % y_bins or 8 % x_bins:
        raise ValueError(f"invalid partition {partition}")
    shape = value.shape
    grouped = value.reshape(
        shape[0], shape[1], y_bins, 4 // y_bins,
        x_bins, 8 // x_bins, *shape[4:]
    )
    return grouped.sum(axis=(3, 5))


def toe_transform(family: str) -> tuple[np.ndarray, np.ndarray]:
    if family == "scalar":
        transform = np.zeros((5, 1), dtype=np.float64)
        transform[2, 0] = 1.0
        prior = np.asarray([1.0])
    elif family == "lag1_symmetric":
        transform = np.zeros((5, 2), dtype=np.float64)
        transform[2, 0] = 1.0
        transform[1, 1] = transform[3, 1] = 0.5
        prior = np.asarray([1.0, 0.0])
    elif family == "toeplitz3":
        transform = np.zeros((5, 3), dtype=np.float64)
        transform[1:4] = np.eye(3)
        prior = np.asarray([0.0, 1.0, 0.0])
    elif family == "toeplitz5":
        transform = np.eye(5, dtype=np.float64)
        prior = np.asarray([0.0, 0.0, 1.0, 0.0, 0.0])
    else:
        raise ValueError(f"not a Toeplitz family: {family}")
    return transform, prior


def regularized_solve(
    xx: np.ndarray, xy: np.ndarray, ridge: float, prior: np.ndarray
) -> np.ndarray:
    p = xx.shape[-1]
    scale = np.trace(xx, axis1=-2, axis2=-1) / max(p, 1)
    penalty = ridge * np.maximum(scale, 1.0e-12)
    matrix = xx.copy()
    diagonal = np.arange(p)
    matrix[..., diagonal, diagonal] += penalty[..., None] + 1.0e-12
    rhs = xy + penalty[..., None] * prior
    try:
        return np.linalg.solve(matrix, rhs[..., None])[..., 0]
    except np.linalg.LinAlgError:
        return np.einsum("...ij,...j->...i", np.linalg.pinv(matrix), rhs)


def fit_coefficients(
    stats: Mapping[str, Any],
    spec: FilterSpec,
    partition: tuple[int, int],
    held_out: int | None,
) -> np.ndarray:
    if spec.family in {"scalar", "lag1_symmetric", "toeplitz3", "toeplitz5"}:
        xx_all = aggregate_partition(
            np.asarray(stats["toeplitz5_xtx"]), partition
        )
        xy_all = aggregate_partition(
            np.asarray(stats["toeplitz5_xty"]), partition
        )
        xx = xx_all.sum(axis=0)
        xy = xy_all.sum(axis=0)
        if held_out is not None:
            xx = xx - xx_all[held_out]
            xy = xy - xy_all[held_out]
        transform, prior = toe_transform(spec.family)
        xx = np.einsum("ip,...ij,jq->...pq", transform, xx, transform)
        xy = np.einsum("ip,...i->...p", transform, xy)
        return regularized_solve(xx, xy, spec.ridge, prior).astype(np.float32)

    xx_all = aggregate_partition(np.asarray(stats["dct_xtx"]), partition)
    xy_all = aggregate_partition(np.asarray(stats["dct_xty"]), partition)
    xx = xx_all.sum(axis=0)
    xy = xy_all.sum(axis=0)
    if held_out is not None:
        xx = xx - xx_all[held_out]
        xy = xy - xy_all[held_out]
    if spec.family.startswith("dct_diag"):
        requested = int(spec.family.removeprefix("dct_diag"))
        individual = min(requested, 19)
        diagonal_xx = np.diagonal(xx, axis1=-2, axis2=-1)
        diagonal_xy = np.diagonal(xy, axis1=-2, axis2=-1)
        reference = np.maximum(diagonal_xx.mean(axis=-1, keepdims=True), 1.0e-12)
        penalty = spec.ridge * reference
        gain = (diagonal_xy + penalty) / (diagonal_xx + penalty)
        if individual < 19:
            tail_xx = diagonal_xx[..., individual:].sum(axis=-1, keepdims=True)
            tail_xy = diagonal_xy[..., individual:].sum(axis=-1, keepdims=True)
            tail_penalty = penalty * (19 - individual)
            tail = (tail_xy + tail_penalty) / (tail_xx + tail_penalty)
            gain[..., individual:] = tail
        return gain.astype(np.float32)
    if spec.family == "full20":
        identity = np.eye(19, dtype=np.float64)
        scale = np.trace(xx, axis1=-2, axis2=-1) / 19.0
        penalty = spec.ridge * np.maximum(scale, 1.0e-12)
        matrix = xx.copy()
        diagonal = np.arange(19)
        matrix[..., diagonal, diagonal] += penalty[..., None] + 1.0e-12
        rhs = xy + penalty[..., None, None] * identity
        try:
            beta = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            beta = np.einsum("...ij,...jk->...ik", np.linalg.pinv(matrix), rhs)
        return beta.astype(np.float32)
    raise ValueError(f"unknown family {spec.family}")


def fit_loto_bank(
    stats: Mapping[str, Any], spec: FilterSpec, partition: tuple[int, int]
) -> np.ndarray:
    rows = [
        fit_coefficients(stats, spec, partition, held_out=tr)
        for tr in range(len(stats["trajectory_ids"]))
    ]
    result = np.stack(rows)
    if not np.isfinite(result).all():
        raise RuntimeError(f"non-finite coefficient bank for {spec}/{partition}")
    return result


def expand_zones(value: np.ndarray, partition: tuple[int, int]) -> np.ndarray:
    y_bins, x_bins = partition
    expanded = value.repeat(32 // y_bins, axis=1).repeat(64 // x_bins, axis=2)
    return np.moveaxis(expanded, 0, 2)


def apply_filter(
    deviation: np.ndarray,
    spec: FilterSpec,
    coefficients: np.ndarray,
    partition: tuple[int, int],
) -> np.ndarray:
    if spec.family in {"scalar", "lag1_symmetric", "toeplitz3", "toeplitz5"}:
        transform, _prior = toe_transform(spec.family)
        flat = np.moveaxis(deviation, 1, -1).reshape(-1, 20)
        features5 = toeplitz5_features(flat)
        features5 = features5.reshape(*np.moveaxis(deviation, 1, -1).shape, 5)
        features5 = np.moveaxis(features5, -2, 1)
        features = np.einsum("bthwcf,fp->bthwcp", features5, transform)
        beta = expand_zones(coefficients, partition)
        result = np.einsum("bthwcp,hwcp->bthwc", features, beta, optimize=True)
    else:
        modes = np.einsum("kt,bthwc->bkhwc", DCT, deviation, optimize=True)
        if spec.family.startswith("dct_diag"):
            gain = expand_zones(coefficients, partition)
            filtered_modes = modes * np.moveaxis(gain, -1, 0)[None]
        elif spec.family == "full20":
            beta = expand_zones(coefficients, partition)
            filtered_modes = np.einsum(
                "bihwc,hwcio->bohwc", modes, beta, optimize=True
            )
        else:
            raise ValueError(spec.family)
        result = np.einsum("kt,bkhwc->bthwc", DCT, filtered_modes, optimize=True)
    result = result.astype(np.float32, copy=False)
    result -= result.mean(axis=1, keepdims=True)
    return result


def polynomial_errors(
    anchor_raw: np.ndarray,
    target_raw_uv: np.ndarray,
    correction_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    e0 = anchor_raw - target_raw_uv
    rel = np.stack(
        (
            np.sum(e0 * e0, axis=(1, 2, 3, 4), dtype=np.float64),
            2.0 * np.sum(e0 * correction_raw, axis=(1, 2, 3, 4), dtype=np.float64),
            np.sum(
                correction_raw * correction_raw,
                axis=(1, 2, 3, 4), dtype=np.float64,
            ),
        ),
        axis=-1,
    )
    target_norm = np.sqrt(
        np.sum(target_raw_uv * target_raw_uv, axis=(1, 2, 3, 4), dtype=np.float64)
    )
    rel = rel / np.maximum(target_norm[:, None] ** 2, 1.0e-16)

    anchor_dev = anchor_raw - anchor_raw.mean(axis=1, keepdims=True)
    target_dev = target_raw_uv - target_raw_uv.mean(axis=1, keepdims=True)
    correction = correction_raw - correction_raw.mean(axis=1, keepdims=True)
    anchor_ke = 0.5 * np.mean(
        np.sum(anchor_dev * anchor_dev, axis=-1), axis=1
    )
    target_ke = 0.5 * np.mean(
        np.sum(target_dev * target_dev, axis=-1), axis=1
    )
    linear = np.mean(np.sum(anchor_dev * correction, axis=-1), axis=1)
    quadratic = 0.5 * np.mean(
        np.sum(correction * correction, axis=-1), axis=1
    )
    base = anchor_ke - target_ke
    tke = np.stack(
        (
            np.sum(base * base, axis=(1, 2), dtype=np.float64),
            2.0 * np.sum(base * linear, axis=(1, 2), dtype=np.float64),
            np.sum(
                linear * linear + 2.0 * base * quadratic,
                axis=(1, 2), dtype=np.float64,
            ),
            2.0 * np.sum(linear * quadratic, axis=(1, 2), dtype=np.float64),
            np.sum(quadratic * quadratic, axis=(1, 2), dtype=np.float64),
        ),
        axis=-1,
    )
    tke_denom = np.sqrt(
        np.sum(target_ke * target_ke, axis=(1, 2), dtype=np.float64)
    )
    tke = tke / np.maximum(tke_denom[:, None] ** 2, 1.0e-16)
    mean_error = float(
        np.max(np.abs(correction_raw.mean(axis=1)))
    )
    return rel, tke, mean_error


def evaluate_candidate(
    data: Any,
    anchor: Mapping[str, Any],
    stats: Mapping[str, Any],
    spec: FilterSpec,
    partition: tuple[int, int],
    dampings: Sequence[float],
    selectors: Mapping[str, np.ndarray],
    baseline_groups: Mapping[str, Mapping[str, Any]],
    baseline_trajectory: np.ndarray,
) -> list[dict[str, Any]]:
    ntraj = len(data.trajectory_ids)
    nd = len(dampings)
    error_sum = np.zeros((nd, ntraj, 3), dtype=np.float64)
    reward = np.zeros((nd, ntraj), dtype=np.float64)
    windows = np.zeros(ntraj, dtype=np.int64)
    valid = np.zeros(ntraj, dtype=np.int64)
    covered = np.zeros(ntraj, dtype=np.int64)
    bank = fit_loto_bank(stats, spec, partition)
    prediction = torch.as_tensor(anchor["prediction_norm_uv"]).numpy()
    sps_kernel = torch.as_tensor(anchor["fixed_bounds_sps_kernel"]).numpy()
    anchor_covered = torch.as_tensor(anchor["fixed_bounds_covered"]).numpy()
    anchor_valid = torch.as_tensor(anchor["valid_count"]).numpy()
    baseline_error = torch.as_tensor(anchor["baseline_error"]).numpy()
    std = data.std_target[..., :2].numpy().reshape(1, 1, 1, 1, 2)
    mean = data.mean_target[..., :2].numpy().reshape(1, 1, 1, 1, 2)
    max_mean_error = 0.0
    for tr in range(ntraj):
        selector = data.trajectory_index == tr
        anchor_norm = prediction[selector]
        anchor_mean = anchor_norm.mean(axis=1, keepdims=True)
        anchor_dev = anchor_norm - anchor_mean
        filtered = apply_filter(anchor_dev, spec, bank[tr], partition)
        correction_raw = (filtered - anchor_dev) * std
        anchor_raw = anchor_norm * std + mean
        target_uv = data.targets[selector, ..., :2].numpy() * std + mean
        rel_poly, tke_poly, mean_error = polynomial_errors(
            anchor_raw, target_uv, correction_raw
        )
        max_mean_error = max(max_mean_error, mean_error)
        mvpe = baseline_error[selector, 2]
        for di, damping in enumerate(dampings):
            d = float(damping)
            rel_squared = rel_poly[:, 0] + d * (
                rel_poly[:, 1] + d * rel_poly[:, 2]
            )
            tke_squared = (
                tke_poly[:, 0]
                + d * tke_poly[:, 1]
                + d * d * tke_poly[:, 2]
                + d**3 * tke_poly[:, 3]
                + d**4 * tke_poly[:, 4]
            )
            rel_error = np.sqrt(np.maximum(rel_squared, 0.0))
            tke_error = np.sqrt(np.maximum(tke_squared, 0.0))
            errors = np.stack((rel_error, tke_error, mvpe), axis=-1)
            error_sum[di, tr] = errors.sum(axis=0)
            factors = 0.5 / (0.5 + np.maximum(errors, 0.0))
            accuracy = (
                0.5 * factors[:, 0]
                + 0.3 * factors[:, 1]
                + 0.2 * factors[:, 2]
            )
            reward[di, tr] = np.sum(
                sps_kernel[selector] * accuracy, dtype=np.float64
            )
        windows[tr] = int(selector.sum())
        valid[tr] = int(anchor_valid[selector].sum())
        covered[tr] = int(anchor_covered[selector].sum())

    rows: list[dict[str, Any]] = []
    for di, damping in enumerate(dampings):
        groups = {
            name: score_group(
                error_sum[di], reward[di], covered, windows, valid, selector
            )
            for name, selector in selectors.items()
        }
        deltas = {
            name: subtract(groups[name], baseline_groups[name]) for name in groups
        }
        trajectory_score = np.asarray(
            [
                score_group(
                    error_sum[di], reward[di], covered, windows, valid,
                    np.arange(ntraj) == tr,
                )["final_proxy"]
                for tr in range(ntraj)
            ],
            dtype=np.float64,
        )
        trajectory_delta = trajectory_score - baseline_trajectory
        rows.append(
            {
                "name": (
                    f"{spec.name}__zone{partition[0]}x{partition[1]}"
                    f"__d{float(damping):g}"
                ),
                "filter": asdict(spec),
                "partition": list(partition),
                "damping": float(damping),
                "groups": groups,
                "delta": deltas,
                "robustness": {
                    "trajectory_mean": float(trajectory_delta.mean()),
                    "trajectory_std": float(trajectory_delta.std()),
                    "robust_050": float(
                        trajectory_delta.mean() - 0.5 * trajectory_delta.std()
                    ),
                    "worst": float(trajectory_delta.min()),
                    "best": float(trajectory_delta.max()),
                    "wins": int((trajectory_delta > 1.0e-12).sum()),
                    "ties": int((np.abs(trajectory_delta) <= 1.0e-12).sum()),
                    "losses": int((trajectory_delta < -1.0e-12).sum()),
                    "all_angles_positive": bool(
                        all(
                            deltas[f"angle_{angle}"]["final_proxy"] > 0.0
                            for angle in (0, 5, 10, 15, 20)
                        )
                    ),
                },
                "trajectory_delta": {
                    name: float(value)
                    for name, value in zip(data.trajectory_ids, trajectory_delta)
                },
                "max_temporal_mean_change_raw": max_mean_error * float(damping),
            }
        )
    return rows


def baseline_summary(
    data: Any,
    anchor: Mapping[str, Any],
    selectors: Mapping[str, np.ndarray],
) -> tuple[dict[str, dict[str, Any]], np.ndarray]:
    ntraj = len(data.trajectory_ids)
    error_window = torch.as_tensor(anchor["baseline_error"]).numpy()
    kernel = torch.as_tensor(anchor["fixed_bounds_sps_kernel"]).numpy()
    covered_window = torch.as_tensor(anchor["fixed_bounds_covered"]).numpy()
    valid_window = torch.as_tensor(anchor["valid_count"]).numpy()
    error = np.zeros((ntraj, 3), dtype=np.float64)
    reward = np.zeros(ntraj, dtype=np.float64)
    covered = np.zeros(ntraj, dtype=np.int64)
    valid = np.zeros(ntraj, dtype=np.int64)
    windows = np.zeros(ntraj, dtype=np.int64)
    factors = 0.5 / (0.5 + np.maximum(error_window, 0.0))
    accuracy = 0.5 * factors[:, 0] + 0.3 * factors[:, 1] + 0.2 * factors[:, 2]
    for tr in range(ntraj):
        selector = data.trajectory_index == tr
        error[tr] = error_window[selector].sum(axis=0)
        reward[tr] = np.sum(kernel[selector] * accuracy[selector])
        covered[tr] = int(covered_window[selector].sum())
        valid[tr] = int(valid_window[selector].sum())
        windows[tr] = int(selector.sum())
    groups = {
        name: score_group(error, reward, covered, windows, valid, selector)
        for name, selector in selectors.items()
    }
    trajectory = np.asarray(
        [
            score_group(
                error, reward, covered, windows, valid,
                np.arange(ntraj) == tr,
            )["final_proxy"]
            for tr in range(ntraj)
        ]
    )
    return groups, trajectory


def choose_candidate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    admissible = [
        row for row in rows
        if row["delta"]["global"]["final_proxy"] >= TARGET_GAIN
        and row["robustness"]["all_angles_positive"]
        and row["robustness"]["worst"] >= -1.0e-12
    ]
    pool = admissible or list(rows)
    return max(
        pool,
        key=lambda row: (
            row in admissible,
            row["robustness"]["robust_050"],
            row["delta"]["global"]["final_proxy"],
            row["robustness"]["worst"],
        ),
    )


def export_deploy(
    path: Path,
    stats: Mapping[str, Any],
    selected: Mapping[str, Any],
    trajectory_ids: Sequence[str],
) -> dict[str, Any]:
    spec = FilterSpec(**selected["filter"])
    partition = tuple(int(x) for x in selected["partition"])
    coefficients = fit_coefficients(stats, spec, partition, held_out=None)
    payload = {
        "schema_version": 1,
        "method": "mean-preserving temporal filter on the calibrated final prediction",
        "family": spec.family,
        "ridge": spec.ridge,
        "partition": list(partition),
        "damping": float(selected["damping"]),
        "coefficients": torch.from_numpy(coefficients),
        "coefficient_layout": (
            "(channel,ybin,xbin,param) for scalar/Toeplitz/DCT; "
            "(channel,ybin,xbin,input_mode,output_mode) for full20"
        ),
        "dct_non_dc": torch.from_numpy(DCT.astype(np.float32)),
        "toeplitz_offsets": [-2, -1, 0, 1, 2],
        "toeplitz_boundary": "edge replication, then subtract feature time mean",
        "mean_contract": "filter deviation, recenter, restore the calibrated mean",
        "bounds_contract": "return calibrated lower/upper unchanged",
        "trajectory_ids": list(trajectory_ids),
        "training": 'all-30 refit after LOTO validation',
        "conditional_validation": selected,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        "coefficient_shape": list(coefficients.shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-cache", type=Path, default=ANCHOR_CACHE)
    parser.add_argument("--stats-cache", type=Path, default=STATS_CACHE)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--rebuild-anchor", action="store_true")
    parser.add_argument("--rebuild-stats", action="store_true")
    parser.add_argument("--include-full", action="store_true")
    parser.add_argument(
        "--quick-screen",
        action="store_true",
        help=(
            "evaluate one representative ridge per compact family; intended "
            "for early closure when the gate is already missed"
        ),
    )
    args = parser.parse_args()
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    started = time.perf_counter()
    data = load_data()
    anchor = load_or_build_anchor(
        data, args.anchor_cache, args.rebuild_anchor, args.progress_every
    )
    stats = collect_stats(data, anchor, args.stats_cache, args.rebuild_stats)

    names = np.asarray(data.trajectory_ids)
    dev = np.asarray([name in DEV_IDS for name in names])
    angles = np.asarray([int(Path(name).stem.rsplit("_", 1)[1]) for name in names])
    selectors: dict[str, np.ndarray] = {
        "global": np.ones(len(names), dtype=bool),
        "dev10": dev,
        "hold20": ~dev,
    }
    selectors.update(
        {f"angle_{value}": angles == value for value in (0, 5, 10, 15, 20)}
    )
    baseline_groups, baseline_trajectory = baseline_summary(data, anchor, selectors)
    for key, expected in EXPECTED_BASELINE.items():
        actual = float(baseline_groups["global"][key])
        if abs(actual - expected) > 2.0e-5:
            raise RuntimeError(
                f"calibrated baseline mismatch: {key}"
            )

    specs = list(QUICK_SPECS if args.quick_screen else COMPACT_SPECS)
    if args.include_full:
        specs.extend(FULL_SPECS)
    rows: list[dict[str, Any]] = []
    evaluation_started = time.perf_counter()
    for spec in specs:
        for partition in PARTITIONS:
            evaluated = evaluate_candidate(
                data, anchor, stats, spec, partition, DAMPINGS,
                selectors, baseline_groups, baseline_trajectory,
            )
            rows.extend(evaluated)
            best = max(
                evaluated,
                key=lambda row: row["delta"]["global"]["final_proxy"],
            )
            print(
                f"calibrated_temporal {spec.name} zone={partition} "
                f"best={best['name']} "
                f"elapsed={time.perf_counter() - evaluation_started:.1f}s",
                flush=True,
            )

    selected = choose_candidate(rows)
    gate = {
        "target_fixed_time_final_gain": TARGET_GAIN,
        "global_gain_pass": bool(
            selected["delta"]["global"]["final_proxy"] >= TARGET_GAIN
        ),
        "all_angles_positive_pass": bool(
            selected["robustness"]["all_angles_positive"]
        ),
        "worst_trajectory_nonnegative_pass": bool(
            selected["robustness"]["worst"] >= -1.0e-12
        ),
    }
    gate["passed"] = bool(
        gate["global_gain_pass"]
        and gate["all_angles_positive_pass"]
        and gate["worst_trajectory_nonnegative_pass"]
    )
    deploy = None
    if gate["passed"]:
        deploy = export_deploy(
            args.artifact, stats, selected, data.trajectory_ids
        )

    ranked = sorted(
        rows,
        key=lambda row: (
            row["robustness"]["all_angles_positive"],
            row["robustness"]["worst"] >= -1.0e-12,
            row["robustness"]["robust_050"],
            row["delta"]["global"]["final_proxy"],
        ),
        reverse=True,
    )
    report = {
        "schema_version": 1,
        "experiment": "calibrated final-prediction temporal phase/shape filter",
        "status": "completed",
        "anchor": {
            "prototype": str(PROTOTYPE),
            "artifact_signature": artifact_signature(PROTOTYPE.resolve()),
            "groups": baseline_groups,
            "time_score_frozen": TIME_SCORE,
        },
        "protocol": {
            "offline_label": "current target deviation",
            "inference_inputs": ["calibrated final prediction"],
            "coefficient_validation": (
                'leave-one-trajectory-out'
            ),
            "temporal_mean": "preserved per window/pixel/channel",
            "bounds": 'calibrated lower/upper unchanged',
            "coverage": 'unchanged',
        },
        "search": {
            "specs": [asdict(spec) for spec in specs],
            "partitions": [list(value) for value in PARTITIONS],
            "dampings": list(DAMPINGS),
            "include_full20": args.include_full,
            "quick_screen": args.quick_screen,
            "early_stop_rule": (
                "do not expand to full20 when compact global gain < +0.04 or "
                "compact worst trajectory < 0"
            ),
            "candidate_rows": len(rows),
            "dct_interpretation": (
                "4/8 are individual lowest non-DC gains plus one shared tail gain; "
                "20 means all 19 non-DC gains because DC/mean is frozen"
            ),
        },
        "selected": selected,
        "recommendation_gate": gate,
        "top_ranked": ranked[:20],
        "all_candidates": rows,
        "deploy_artifact": deploy,
        "duration_seconds": time.perf_counter() - started,
        "artifacts": {
            "source": 'balanced-30 caches and calibrated model',
            "generated": [
                str(args.anchor_cache), str(args.stats_cache), str(args.report)
            ] + ([str(args.artifact)] if deploy is not None else []),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"temporal_report={args.report}", flush=True)


if __name__ == "__main__":
    main()
