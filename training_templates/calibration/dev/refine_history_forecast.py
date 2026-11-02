"""Temporal history forecast."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from refine_feature_ablation import fit_loto as fit_reference_loto
from refine_learned_interval import (
    DEV_IDS,
    ROOT,
    expand_beta as expand_reference_beta,
    feature_tensor as reference_feature_tensor,
    load_data,
    new_model,
    target_raw,
)
from refine_joint_grid import (
    FINAL_POWER,
    FINAL_WEIGHTS,
    TIME_SCORE_ASSUMPTION,
    vector_errors,
)

import scoring


DEFAULT_OUTPUT = ROOT / "dev" / "reports" / "temporal_history_nested.json"
DEFAULT_STATS_CACHE = ROOT / "dev" / "cache" / "temporal_history_stats.pt"
DEFAULT_DEPLOY = ROOT / "dev" / "cache" / "temporal_history_deploy.pt"
REFERENCE_STATS_CACHE = ROOT / "dev" / "cache" / "sufficient_stats.pt"

T = 20
H = 32
W = 64
C = 2
FINE_Y = 4
FINE_X = 8
MAX_P = 42
RESPONSE_CLIP = 0.5
FORECAST_CLIP = 0.75

PREDICTION_COUPLING = 1.30
BOUNDS_TOTAL_DAMPING = 1.20
WIDTH_SCALE = 0.95
PIXEL_PREVIOUS_WEIGHT = 0.50
PIXEL_STATE_DECAY = 0.80
PIXEL_DAMPING = 0.35

PARTITIONS = ((1, 1), (2, 4), (4, 8))
RIDGES = (1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2)


@dataclass(frozen=True)
class Family:
    name: str
    description: str
    transform: np.ndarray

    @property
    def p(self) -> int:
        return int(self.transform.shape[-1])


@dataclass(frozen=True)
class Candidate:
    family: str
    partition: tuple[int, int]
    ridge: float

    @property
    def key(self) -> str:
        return f"{self.family}|{self.partition[0]}x{self.partition[1]}|r{self.ridge:g}"


def final_proxy(
    rel: np.ndarray | float,
    tke: np.ndarray | float,
    mvpe: np.ndarray | float,
    sps: np.ndarray | float,
    time_score: float = TIME_SCORE_ASSUMPTION,
) -> np.ndarray:
    rel = np.asarray(rel)
    tke = np.asarray(tke)
    mvpe = np.asarray(mvpe)
    sps = np.asarray(sps)
    total = (
        FINAL_WEIGHTS[0] * rel**FINAL_POWER
        + FINAL_WEIGHTS[1] * tke**FINAL_POWER
        + FINAL_WEIGHTS[2] * mvpe**FINAL_POWER
        + FINAL_WEIGHTS[3] * time_score**FINAL_POWER
        + FINAL_WEIGHTS[4] * sps**FINAL_POWER
    )
    return total ** (1.0 / FINAL_POWER)


def score_from_error(error: np.ndarray) -> np.ndarray:
    return 100.0 / (1.0 + 0.5 * np.maximum(error, 0.0))


def dct_basis(k: int) -> np.ndarray:
    j = np.arange(T, dtype=np.float64) + 0.5
    n = np.arange(k, dtype=np.float64)[:, None]
    basis = np.cos(math.pi * n * j[None] / T)
    basis[0] *= math.sqrt(1.0 / T)
    if k > 1:
        basis[1:] *= math.sqrt(2.0 / T)
    check = basis @ basis.T
    if not np.allclose(check, np.eye(k), atol=1.0e-12):
        raise RuntimeError("DCT basis is not orthonormal")
    return basis


def base_transform(columns: int) -> np.ndarray:
    return np.zeros((T, MAX_P, columns), dtype=np.float64)


def make_families() -> list[Family]:
    families: list[Family] = []

    a = base_transform(4)
    for horizon in range(T):
        a[horizon, 0, 0] = 1.0
        a[horizon, 1, 1:] = 1.0
        a[horizon, 2 + horizon, 1] = 1.0
        a[horizon, 2 + (T - 1), 2] = 1.0
        a[horizon, 2 + (T - 1), 3] += 1.0
        scale = float(horizon + 1) / 15.0
        a[horizon, 2 + 15 : 2 + 20, 3] += scale / 5.0
        a[horizon, 2 + 0 : 2 + 5, 3] -= scale / 5.0
    families.append(Family("reference_four", "reference four-feature anchor", a))

    for k in (4, 8, 12):
        basis = dct_basis(k)
        a = base_transform(2 + k)
        a[:, 0, 0] = 1.0
        a[:, 1, 1] = 1.0
        a[:, 2:22, 2:] = basis.T[None]
        families.append(Family(
            f"same_dct{k}",
            f"same-channel {k}-mode DCT history plus explicit center",
            a,
        ))

    a = base_transform(22)
    a[:, 0, 0] = 1.0
    a[:, 1, 1] = 1.0
    for j in range(T):
        a[:, 2 + j, 2 + j] = 1.0
    families.append(Family(
        "same_full20",
        "same-channel complete 20-frame history plus explicit center",
        a,
    ))

    for k in (4, 8):
        basis = dct_basis(k)
        a = base_transform(2 + 2 * k)
        a[:, 0, 0] = 1.0
        a[:, 1, 1] = 1.0
        a[:, 2:22, 2 : 2 + k] = basis.T[None]
        a[:, 22:42, 2 + k :] = basis.T[None]
        families.append(Family(
            f"both_dct{k}",
            f"same/cross-channel {k}-mode DCT histories plus explicit center",
            a,
        ))

    basis8 = dct_basis(8)
    a = base_transform(30)
    a[:, 0, 0] = 1.0
    a[:, 1, 1] = 1.0
    for j in range(T):
        a[:, 2 + j, 2 + j] = 1.0
    a[:, 22:42, 22:] = basis8.T[None]
    families.append(Family(
        "same_full20_cross_dct8",
        "full same-channel history plus 8-mode cross-channel history",
        a,
    ))

    a = base_transform(42)
    for j in range(MAX_P):
        a[:, j, j] = 1.0
    families.append(Family(
        "both_full40",
        "complete same/cross-channel 20-frame histories plus explicit center",
        a,
    ))
    return families


def max_feature_tensor(data: Any, index: int, center: torch.Tensor) -> torch.Tensor:
    input_raw = (
        data.inputs[index].unsqueeze(0) * data.std_input + data.mean_input
    )[..., :2]
    input_target_norm = (
        input_raw - data.mean_target[..., :2]
    ) / data.std_target[..., :2]
    history = input_target_norm.squeeze(0).permute(1, 2, 3, 0).contiguous()
    result = torch.empty((T, H, W, C, MAX_P), dtype=torch.float32)
    result[..., 0] = 1.0
    result[..., 1] = -center.squeeze(0)
    result[..., 2:22] = history.unsqueeze(0)
    result[..., 22:42] = history[..., [1, 0], :].unsqueeze(0)
    if not torch.isfinite(result).all():
        raise RuntimeError("non-finite maximal temporal feature")
    return result


def fine_stats_for_window(
    x: torch.Tensor,
    response: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bh, bw = H // FINE_Y, W // FINE_X
    xr = (
        x.reshape(T, FINE_Y, bh, FINE_X, bw, C, MAX_P)
        .permute(0, 5, 1, 3, 2, 4, 6)
        .reshape(T, C, FINE_Y, FINE_X, bh * bw, MAX_P)
    )
    vr = (
        valid.reshape(T, FINE_Y, bh, FINE_X, bw, C)
        .permute(0, 5, 1, 3, 2, 4)
        .reshape(T, C, FINE_Y, FINE_X, bh * bw)
    )
    yr = (
        response.reshape(T, FINE_Y, bh, FINE_X, bw, C)
        .permute(0, 5, 1, 3, 2, 4)
        .reshape(T, C, FINE_Y, FINE_X, bh * bw)
    )
    xv = xr * vr[..., None]
    xtx = torch.matmul(xv.transpose(-1, -2), xr)
    yc = yr.clamp(-RESPONSE_CLIP, RESPONSE_CLIP) * vr
    xty = torch.matmul(xr.transpose(-1, -2), yc[..., None])[..., 0]
    y2 = ((yr * vr) * yr).sum(dim=-1, dtype=torch.float64)
    count = vr.sum(dim=-1, dtype=torch.int64)
    return xtx.numpy(), xty.numpy(), y2.numpy(), count.numpy()


def collect_stats(data: Any, progress_every: int) -> dict[str, Any]:
    ntraj = len(data.trajectory_ids)
    xtx = np.zeros(
        (ntraj, T, C, FINE_Y, FINE_X, MAX_P, MAX_P), dtype=np.float32
    )
    xty = np.zeros((ntraj, T, C, FINE_Y, FINE_X, MAX_P), dtype=np.float32)
    y2 = np.zeros((ntraj, T, C, FINE_Y, FINE_X), dtype=np.float64)
    count = np.zeros((ntraj, T, C, FINE_Y, FINE_X), dtype=np.int64)
    base, model = new_model(data)
    previous_target = None
    started = time.perf_counter()
    for index in range(data.targets.shape[0]):
        if data.first[index]:
            model.reset_ttt_state()
            previous_target = None
        returned, info = model.ttt_step(data.inputs[index].unsqueeze(0), previous_target)
        del returned
        center = 0.5 * (
            info["lower"][..., :2].detach().cpu()
            + info["upper"][..., :2].detach().cpu()
        )
        x = max_feature_tensor(data, index, center)
        response = data.targets[index, ..., :2] - center.squeeze(0)
        valid = (target_raw(data, index)[..., :2] != 0.0).squeeze(0)
        values = fine_stats_for_window(x, response, valid)
        tr = int(data.trajectory_index[index])
        xtx[tr] += values[0]
        xty[tr] += values[1]
        y2[tr] += values[2]
        count[tr] += values[3]
        previous_target = data.targets[index].unsqueeze(0)
        if progress_every and (index + 1) % progress_every == 0:
            print(
                f"temporal_stats_windows={index+1} "
                f"elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )
    if base.index != data.targets.shape[0]:
        raise RuntimeError("temporal stats replay did not consume full base cache")
    return {
        "schema_version": 1,
        "feature_system": "[1,-center,same_history20,cross_history20]",
        "response_clip": RESPONSE_CLIP,
        "trajectory_ids": list(data.trajectory_ids),
        "xtx": xtx,
        "xty": xty,
        "y2": y2,
        "count": count,
    }


def aggregate_partition(array: np.ndarray, yb: int, xb: int) -> np.ndarray:
    if FINE_Y % yb or FINE_X % xb:
        raise ValueError("partition must divide the 4x8 fine grid")
    yf, xf = FINE_Y // yb, FINE_X // xb
    shape = array.shape
    reshaped = array.reshape(
        *shape[:3], yb, yf, xb, xf, *shape[5:]
    )
    return reshaped.sum(axis=(4, 6))


def transform_stats(
    stats: dict[str, Any], family: Family, partition: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    yb, xb = partition
    xtx0 = aggregate_partition(stats["xtx"], yb, xb)
    xty0 = aggregate_partition(stats["xty"], yb, xb)
    y2 = aggregate_partition(stats["y2"], yb, xb)
    count = aggregate_partition(stats["count"], yb, xb)
    ntraj = xtx0.shape[0]
    p = family.p
    xtx = np.empty((ntraj, T, C, yb, xb, p, p), dtype=np.float64)
    xty = np.empty((ntraj, T, C, yb, xb, p), dtype=np.float64)
    for horizon in range(T):
        a = family.transform[horizon]
        xtx[:, horizon] = np.einsum(
            "acYXmn,mp,nq->acYXpq",
            xtx0[:, horizon].astype(np.float64),
            a,
            a,
            optimize=True,
        )
        xty[:, horizon] = np.einsum(
            "acYXm,mp->acYXp",
            xty0[:, horizon].astype(np.float64),
            a,
            optimize=True,
        )
    return xtx, xty, y2, count


def solve_coefficients(
    xtx: np.ndarray,
    xty: np.ndarray,
    count: np.ndarray,
    ridge: float,
) -> np.ndarray:
    matrix = xtx.copy()
    p = xty.shape[-1]
    diagonal = np.arange(p)
    penalty = ridge * np.maximum(count, 1)[..., None]
    if p > 1:
        matrix[..., diagonal[1:], diagonal[1:]] += penalty
    matrix[..., diagonal, diagonal] += 1.0e-10
    try:
        beta = np.linalg.solve(matrix, xty[..., None])[..., 0]
    except np.linalg.LinAlgError:
        beta = np.einsum("...pq,...q->...p", np.linalg.pinv(matrix), xty)
    return np.nan_to_num(beta).astype(np.float32)


def validation_rmse(
    beta: np.ndarray,
    xtx: np.ndarray,
    xty: np.ndarray,
    y2: np.ndarray,
    count: np.ndarray,
) -> float:
    cross = np.einsum("...p,...p->...", beta, xty, optimize=True)
    quad = np.einsum("...p,...pq,...q->...", beta, xtx, beta, optimize=True)
    sse = float(np.sum(y2 - 2.0 * cross + quad))
    return math.sqrt(max(sse / max(int(count.sum()), 1), 0.0))


def split_cv_candidate(
    transformed: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    split_indices: np.ndarray,
    ridge: float,
) -> dict[str, Any]:
    xtx, xty, y2, count = transformed
    total_x = xtx[split_indices].sum(axis=0)
    total_y = xty[split_indices].sum(axis=0)
    total_n = count[split_indices].sum(axis=0)
    rmses = []
    for tr in split_indices:
        beta = solve_coefficients(
            total_x - xtx[tr], total_y - xty[tr], total_n - count[tr], ridge
        )
        rmses.append(validation_rmse(beta, xtx[tr], xty[tr], y2[tr], count[tr]))
    values = np.asarray(rmses, dtype=np.float64)
    return {
        "trajectory_rmse": values,
        "mean_rmse": float(values.mean()),
        "std_rmse": float(values.std()),
        "robust_050_rmse": float(values.mean() + 0.5 * values.std()),
        "worst_rmse": float(values.max()),
    }


def expand_to_fine(beta: np.ndarray, partition: tuple[int, int]) -> np.ndarray:
    yb, xb = partition
    return np.repeat(
        np.repeat(beta, FINE_Y // yb, axis=2), FINE_X // xb, axis=3
    )


def effective_max_beta(beta: np.ndarray, family: Family) -> np.ndarray:
    return np.einsum(
        "tmp,tcyxp->tcyxm", family.transform, beta, optimize=True
    ).astype(np.float32)


def fit_effective_for_train(
    transformed: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    train_indices: np.ndarray,
    candidate: Candidate,
    family: Family,
) -> np.ndarray:
    xtx, xty, _y2, count = transformed
    beta = solve_coefficients(
        xtx[train_indices].sum(axis=0),
        xty[train_indices].sum(axis=0),
        count[train_indices].sum(axis=0),
        candidate.ridge,
    )
    return effective_max_beta(expand_to_fine(beta, candidate.partition), family)


def fit_effective_banks(
    stats: dict[str, Any],
    family: Family,
    dev_choice: Candidate,
    hold_choice: Candidate,
    dev_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ntraj = stats["xtx"].shape[0]
    all_indices = np.arange(ntraj)
    dev_indices = all_indices[dev_mask]
    hold_indices = all_indices[~dev_mask]
    cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    def transformed(candidate: Candidate):
        if candidate.partition not in cache:
            cache[candidate.partition] = transform_stats(stats, family, candidate.partition)
        return cache[candidate.partition]

    cv_bank = np.zeros((ntraj, T, C, FINE_Y, FINE_X, MAX_P), dtype=np.float32)
    outer_bank = np.zeros_like(cv_bank)
    for tr in all_indices:
        if dev_mask[tr]:
            cv_choice, cv_pool = dev_choice, dev_indices
            outer_choice = hold_choice
        else:
            cv_choice, cv_pool = hold_choice, hold_indices
            outer_choice = dev_choice
        cv_bank[tr] = fit_effective_for_train(
            transformed(cv_choice), cv_pool[cv_pool != tr], cv_choice, family
        )
        outer_bank[tr] = fit_effective_for_train(
            transformed(outer_choice), all_indices[all_indices != tr], outer_choice, family
        )
    return cv_bank, outer_bank


def select_family_candidates(
    stats: dict[str, Any],
    families: list[Family],
    dev_mask: np.ndarray,
    progress: bool,
) -> tuple[
    dict[str, dict[str, Candidate]],
    dict[str, list[dict[str, Any]]],
    dict[str, tuple[np.ndarray, np.ndarray]],
]:
    ntraj = stats["xtx"].shape[0]
    indices = np.arange(ntraj)
    dev_indices, hold_indices = indices[dev_mask], indices[~dev_mask]
    choices: dict[str, dict[str, Candidate]] = {}
    search_report: dict[str, list[dict[str, Any]]] = {}
    banks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for family_index, family in enumerate(families):
        family_started = time.perf_counter()
        candidates = (
            [Candidate(family.name, (4, 8), 1.0e-5)]
            if family.name == "reference_four"
            else [
                Candidate(family.name, partition, ridge)
                for partition in PARTITIONS
                for ridge in RIDGES
            ]
        )
        by_partition = {
            partition: transform_stats(stats, family, partition)
            for partition in sorted({candidate.partition for candidate in candidates})
        }
        rows = []
        for candidate in candidates:
            transformed = by_partition[candidate.partition]
            dev = split_cv_candidate(transformed, dev_indices, candidate.ridge)
            hold = split_cv_candidate(transformed, hold_indices, candidate.ridge)
            rows.append({
                "candidate": candidate,
                "dev10": dev,
                "hold20": hold,
                "consensus_robust_rmse": max(
                    dev["robust_050_rmse"], hold["robust_050_rmse"]
                ),
            })
        dev_best = min(
            rows,
            key=lambda row: (
                row["dev10"]["robust_050_rmse"],
                row["dev10"]["mean_rmse"],
                row["candidate"].partition[0] * row["candidate"].partition[1],
                row["candidate"].ridge,
            ),
        )["candidate"]
        hold_best = min(
            rows,
            key=lambda row: (
                row["hold20"]["robust_050_rmse"],
                row["hold20"]["mean_rmse"],
                row["candidate"].partition[0] * row["candidate"].partition[1],
                row["candidate"].ridge,
            ),
        )["candidate"]
        choices[family.name] = {"dev10": dev_best, "hold20": hold_best}
        serial_rows = []
        for row in rows:
            serial_rows.append({
                "candidate": {
                    "key": row["candidate"].key,
                    "partition": list(row["candidate"].partition),
                    "ridge": row["candidate"].ridge,
                },
                "dev10": {
                    key: value for key, value in row["dev10"].items()
                    if key != "trajectory_rmse"
                },
                "hold20": {
                    key: value for key, value in row["hold20"].items()
                    if key != "trajectory_rmse"
                },
                "consensus_robust_rmse": row["consensus_robust_rmse"],
            })
        search_report[family.name] = serial_rows
        del by_partition
        banks[family.name] = fit_effective_banks(
            stats, family, dev_best, hold_best, dev_mask
        )
        if progress:
            print(
                f"temporal_family={family.name} p={family.p} "
                f"dev_choice={dev_best.key} hold_choice={hold_best.key} "
                f"elapsed={time.perf_counter()-family_started:.1f}s",
                flush=True,
            )
    return choices, search_report, banks


def expand_effective(beta: np.ndarray) -> np.ndarray:
    expanded = np.repeat(np.repeat(beta, H // FINE_Y, axis=3), W // FINE_X, axis=4)
    return expanded.transpose(0, 1, 3, 4, 2, 5)


def group_values(
    error_sum: np.ndarray,
    sps_num: np.ndarray,
    covered: np.ndarray,
    windows: np.ndarray,
    valid: np.ndarray,
    selector: np.ndarray,
) -> dict[str, np.ndarray]:
    nwin = int(windows[selector].sum())
    nvalid = int(valid[selector].sum())
    scores = score_from_error(error_sum[:, selector].sum(axis=1) / nwin)
    sps = 100.0 * sps_num[:, selector].sum(axis=1) / nvalid
    coverage = covered[:, selector].sum(axis=1) / nvalid
    return {
        "rel_l2_score": scores[:, 0],
        "tke_score": scores[:, 1],
        "mvpe_score": scores[:, 2],
        "sps_score": sps,
        "coverage": coverage,
        "final_proxy": final_proxy(scores[:, 0], scores[:, 1], scores[:, 2], sps),
        "final_proxy_current_time": final_proxy(
            scores[:, 0], scores[:, 1], scores[:, 2], sps, time_score=85.066771
        ),
    }


def scalar_values(values: dict[str, np.ndarray], index: int) -> dict[str, float]:
    return {key: float(value[index]) for key, value in values.items()}


def replay_exact(
    data: Any,
    families: list[Family],
    banks: dict[str, tuple[np.ndarray, np.ndarray]],
    progress_every: int,
) -> dict[str, Any]:
    family_names = [family.name for family in families]
    config_names = ["ridge_exact", "reference_external"]
    config_names += [f"cv::{name}" for name in family_names]
    config_names += [f"outer::{name}" for name in family_names]
    nconfig = len(config_names)
    ntraj = len(data.trajectory_ids)
    errors = np.zeros((nconfig, ntraj, 3), dtype=np.float64)
    sps_num = np.zeros((nconfig, ntraj), dtype=np.float64)
    covered = np.zeros((nconfig, ntraj), dtype=np.int64)
    windows = np.zeros(ntraj, dtype=np.int64)
    valid_count = np.zeros(ntraj, dtype=np.int64)

    reference_stats = torch.load(REFERENCE_STATS_CACHE, map_location="cpu", weights_only=False)
    reference_beta = fit_reference_loto(reference_stats, (0, 5, 6, 7))

    cv_stack = np.stack([banks[name][0] for name in family_names])
    outer_stack = np.stack([banks[name][1] for name in family_names])
    base, ridge = new_model(data)
    previous_target = None
    previous_ridge_returned = None
    pixel_state = None
    reference_forecast_max_diff = 0.0
    started = time.perf_counter()
    for index in range(data.targets.shape[0]):
        if data.first[index]:
            ridge.reset_ttt_state()
            previous_target = None
            previous_ridge_returned = None
            pixel_state = None
        returned, info = ridge.ttt_step(data.inputs[index].unsqueeze(0), previous_target)
        returned = returned.detach().cpu()
        lower = info["lower"][..., :2].detach().cpu()
        upper = info["upper"][..., :2].detach().cpu()
        center = 0.5 * (lower + upper)
        half = 0.5 * (upper - lower)
        xmax = max_feature_tensor(data, index, center).numpy()
        tr = int(data.trajectory_index[index])

        cv_beta = expand_effective(cv_stack[:, tr])
        outer_beta = expand_effective(outer_stack[:, tr])
        cv_forecasts = np.einsum(
            "thwcm,fthwcm->fthwc", xmax, cv_beta, optimize=True
        )
        outer_forecasts = np.einsum(
            "thwcm,fthwcm->fthwc", xmax, outer_beta, optimize=True
        )
        reference_x = reference_feature_tensor(
            data=data, index=index, model=ridge, returned=returned, center=center
        ).numpy()
        external_reference = np.einsum(
            "thwcp,thwcp->thwc",
            reference_x,
            expand_reference_beta(reference_beta[tr]),
            optimize=True,
        )
        reference_family_index = family_names.index("reference_four")
        reference_forecast_max_diff = max(
            reference_forecast_max_diff,
            float(np.max(np.abs(external_reference - outer_forecasts[reference_family_index]))),
        )
        forecast_bank = np.concatenate(
            (external_reference[None], cv_forecasts, outer_forecasts), axis=0
        )
        forecast_bank = np.clip(
            forecast_bank, -FORECAST_CLIP, FORECAST_CLIP
        ).astype(np.float32)
        forecast_t = torch.from_numpy(forecast_bank)

        if previous_target is not None and previous_ridge_returned is not None:
            target_phys = previous_target[..., :2] * data.std_target[..., :2]
            target_dev = target_phys - target_phys.mean(dim=1, keepdim=True)
            previous_dev = previous_ridge_returned - previous_ridge_returned.mean(
                dim=1, keepdim=True
            )
            previous_var = (
                previous_dev * data.std_target[..., :2]
            ).square().mean(dim=1)[0]
            target_var = target_dev.square().mean(dim=1)[0]
            observed = 0.5 * torch.log(
                (target_var + 1.0e-8) / (previous_var + 1.0e-8)
            ).clamp(math.log(0.5), math.log(2.0))
            if pixel_state is None:
                pixel_state = observed.clone()
            else:
                pixel_state.mul_(PIXEL_STATE_DECAY).add_(
                    observed, alpha=1.0 - PIXEL_STATE_DECAY
                )
        returned_uv = returned[..., :2]
        returned_dev = returned_uv - returned_uv.mean(dim=1, keepdim=True)
        returned_var = (
            returned_dev * data.std_target[..., :2]
        ).square().mean(dim=1)[0]
        input_phys = data.inputs[index].unsqueeze(0)[..., :2] * data.std_input[..., :2]
        input_dev = input_phys - input_phys.mean(dim=1, keepdim=True)
        input_var = input_dev.square().mean(dim=1)[0]
        current_ratio = 0.5 * torch.log(
            (input_var + 1.0e-8) / (returned_var + 1.0e-8)
        ).clamp(math.log(0.5), math.log(2.0))
        if data.first[index] or pixel_state is None:
            pixel_scale = torch.ones_like(current_ratio)
        else:
            blended = (
                PIXEL_PREVIOUS_WEIGHT * pixel_state
                + (1.0 - PIXEL_PREVIOUS_WEIGHT) * current_ratio
            )
            pixel_scale = torch.exp(PIXEL_DAMPING * blended).clamp(0.8, 1.25)

        target = target_raw(data, index)
        valid = (target[..., :2] != 0.0).squeeze(0)
        target_error = (data.targets[index].unsqueeze(0)[..., :2] - center).squeeze(0)
        half0 = half.squeeze(0)
        raw_ridge = returned * data.std_target + data.mean_target
        ridge_error = vector_errors(raw_ridge, target)[0]
        errors[0, tr] += ridge_error.numpy().astype(np.float64)
        ridge_accuracy = (
            0.5 * (0.5 / (0.5 + ridge_error[0]))
            + 0.3 * (0.5 / (0.5 + ridge_error[1]))
            + 0.2 * (0.5 / (0.5 + ridge_error[2]))
        )
        ridge_inside = target_error.abs() <= half0
        ridge_tight = torch.exp(
            -2.0 * half0 * data.std_target[..., :2].squeeze(0)
            / float(scoring.SIGMA_GLOBAL)
        )
        sps_num[0, tr] += float(
            (ridge_inside * valid * ridge_tight).sum(dtype=torch.float64) * ridge_accuracy
        )
        covered[0, tr] += int((ridge_inside & valid).sum())

        nforecast = forecast_t.shape[0]
        candidate = returned.expand(nforecast, -1, -1, -1, -1).clone()
        candidate_uv = returned_uv + PREDICTION_COUPLING * forecast_t
        if not data.first[index]:
            candidate_mean = candidate_uv.mean(dim=1, keepdim=True)
            candidate_uv = candidate_mean + pixel_scale.unsqueeze(0).unsqueeze(0) * (
                candidate_uv - candidate_mean
            )
        candidate[..., :2] = candidate_uv
        candidate_raw = candidate * data.std_target + data.mean_target
        candidate_errors = vector_errors(candidate_raw, target)
        errors[1:, tr] += candidate_errors.numpy().astype(np.float64)
        factors = 0.5 / (0.5 + candidate_errors)
        accuracy = 0.5 * factors[:, 0] + 0.3 * factors[:, 1] + 0.2 * factors[:, 2]
        interval_half = WIDTH_SCALE * half0
        inside = (
            target_error.unsqueeze(0)
            - BOUNDS_TOTAL_DAMPING * forecast_t
        ).abs() <= interval_half.unsqueeze(0)
        tight = torch.exp(
            -2.0 * interval_half * data.std_target[..., :2].squeeze(0)
            / float(scoring.SIGMA_GLOBAL)
        )
        reward = (inside * valid.unsqueeze(0) * tight.unsqueeze(0)).sum(
            dim=(1, 2, 3, 4), dtype=torch.float64
        )
        sps_num[1:, tr] += (reward * accuracy).numpy()
        covered[1:, tr] += (
            inside & valid.unsqueeze(0)
        ).sum(dim=(1, 2, 3, 4), dtype=torch.int64).numpy()
        windows[tr] += 1
        valid_count[tr] += int(valid.sum())
        previous_target = data.targets[index].unsqueeze(0)
        previous_ridge_returned = returned_uv
        if progress_every and (index + 1) % progress_every == 0:
            print(
                f"temporal_exact_windows={index+1} "
                f"elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )
    if base.index != data.targets.shape[0]:
        raise RuntimeError("exact temporal replay did not consume full base cache")

    names = np.asarray(data.trajectory_ids)
    dev = np.asarray([name in DEV_IDS for name in names])
    angles = np.asarray([
        int(name.removesuffix(".h5").rsplit("_", 1)[1]) for name in names
    ])
    selectors = {
        "global": np.ones(ntraj, dtype=bool),
        "dev10": dev,
        "hold20": ~dev,
        **{f"angle_{angle}": angles == angle for angle in sorted(set(angles))},
    }
    groups_raw = {
        group: group_values(errors, sps_num, covered, windows, valid_count, selector)
        for group, selector in selectors.items()
    }
    groups = {
        config: {
            group: scalar_values(values, ci)
            for group, values in groups_raw.items()
        }
        for ci, config in enumerate(config_names)
    }

    best_from_dev = max(
        family_names,
        key=lambda name: groups[f"cv::{name}"]["dev10"]["final_proxy"],
    )
    best_from_hold = max(
        family_names,
        key=lambda name: groups[f"cv::{name}"]["hold20"]["final_proxy"],
    )
    selected_errors = np.empty((ntraj, 3), dtype=np.float64)
    selected_sps = np.empty(ntraj, dtype=np.float64)
    selected_covered = np.empty(ntraj, dtype=np.int64)
    for tr in range(ntraj):
        family = best_from_hold if dev[tr] else best_from_dev
        ci = config_names.index(f"outer::{family}")
        selected_errors[tr] = errors[ci, tr]
        selected_sps[tr] = sps_num[ci, tr]
        selected_covered[tr] = covered[ci, tr]
    selected_group_raw = {
        group: group_values(
            selected_errors[None], selected_sps[None], selected_covered[None],
            windows, valid_count, selector,
        )
        for group, selector in selectors.items()
    }
    selected_groups = {
        group: scalar_values(value, 0) for group, value in selected_group_raw.items()
    }
    baseline_ci = config_names.index("reference_external")
    trajectory_delta = []
    for tr in range(ntraj):
        selector = np.arange(ntraj) == tr
        candidate = group_values(
            selected_errors[None], selected_sps[None], selected_covered[None],
            windows, valid_count, selector,
        )["final_proxy"][0]
        baseline = group_values(
            errors[baseline_ci : baseline_ci + 1],
            sps_num[baseline_ci : baseline_ci + 1],
            covered[baseline_ci : baseline_ci + 1],
            windows, valid_count, selector,
        )["final_proxy"][0]
        trajectory_delta.append(float(candidate - baseline))
    trajectory_delta_np = np.asarray(trajectory_delta)
    selected_delta = {
        group: {
            key: float(value - groups["reference_external"][group][key])
            for key, value in item.items()
        }
        for group, item in selected_groups.items()
    }
    return {
        "config_names": config_names,
        "groups": groups,
        "nested_selected": {
            "family_selected_on_dev10_for_hold20": best_from_dev,
            "family_selected_on_hold20_for_dev10": best_from_hold,
            "groups": selected_groups,
            "delta_vs_reference": selected_delta,
            "trajectory_delta": {
                str(name): float(value)
                for name, value in zip(names, trajectory_delta_np)
            },
            "robustness": {
                "wins": int((trajectory_delta_np > 0).sum()),
                "losses": int((trajectory_delta_np < 0).sum()),
                "mean_delta": float(trajectory_delta_np.mean()),
                "std_delta": float(trajectory_delta_np.std()),
                "robust_050": float(
                    trajectory_delta_np.mean() - 0.5 * trajectory_delta_np.std()
                ),
                "worst_delta": float(trajectory_delta_np.min()),
                "worst_trajectory": str(names[int(np.argmin(trajectory_delta_np))]),
                "all_angles_positive": bool(all(
                    selected_delta[f"angle_{angle}"]["final_proxy"] > 0
                    for angle in sorted(set(angles))
                )),
            },
        },
        "validation": {
            "reference_max_forecast_difference_max_feature_vs_external": reference_forecast_max_diff,
            "windows": int(windows.sum()),
            "valid_elements": int(valid_count.sum()),
        },
        "raw_arrays": {
            "errors": errors,
            "sps_num": sps_num,
            "covered": covered,
            "windows": windows,
            "valid_count": valid_count,
        },
    }


def deploy_consensus(
    families: list[Family],
    choices: dict[str, dict[str, Candidate]],
    search_report: dict[str, list[dict[str, Any]]],
    exact: dict[str, Any],
) -> tuple[Family, Candidate, dict[str, Any]]:
    anchor_dev = exact["groups"]["cv::reference_four"]["dev10"]["final_proxy"]
    anchor_hold = exact["groups"]["cv::reference_four"]["hold20"]["final_proxy"]
    ranking = []
    for family in families:
        dev_score = exact["groups"][f"cv::{family.name}"]["dev10"]["final_proxy"]
        hold_score = exact["groups"][f"cv::{family.name}"]["hold20"]["final_proxy"]
        ranking.append({
            "family": family.name,
            "dev10_delta_vs_reference_cv": dev_score - anchor_dev,
            "hold20_delta_vs_reference_cv": hold_score - anchor_hold,
            "worst_split_delta": min(dev_score - anchor_dev, hold_score - anchor_hold),
            "mean_split_delta": 0.5 * (
                dev_score - anchor_dev + hold_score - anchor_hold
            ),
        })
    ranking.sort(
        key=lambda item: (item["worst_split_delta"], item["mean_split_delta"]),
        reverse=True,
    )
    selected_family = next(
        family for family in families if family.name == ranking[0]["family"]
    )
    rows = search_report[selected_family.name]
    dev_order = sorted(
        range(len(rows)), key=lambda i: rows[i]["dev10"]["robust_050_rmse"]
    )
    hold_order = sorted(
        range(len(rows)), key=lambda i: rows[i]["hold20"]["robust_050_rmse"]
    )
    dev_rank = {index: rank for rank, index in enumerate(dev_order)}
    hold_rank = {index: rank for rank, index in enumerate(hold_order)}
    best_index = min(
        range(len(rows)),
        key=lambda i: (
            max(dev_rank[i], hold_rank[i]),
            dev_rank[i] + hold_rank[i],
            rows[i]["consensus_robust_rmse"],
        ),
    )
    row = rows[best_index]
    candidate = Candidate(
        selected_family.name,
        tuple(row["candidate"]["partition"]),
        float(row["candidate"]["ridge"]),
    )
    return selected_family, candidate, {"family_ranking": ranking, "selected_row": row}


def export_deploy(
    path: Path,
    stats: dict[str, Any],
    family: Family,
    candidate: Candidate,
    exact: dict[str, Any],
    selection: dict[str, Any],
) -> None:
    transformed = transform_stats(stats, family, candidate.partition)
    all_indices = np.arange(stats["xtx"].shape[0])
    xtx, xty, _y2, count = transformed
    native = solve_coefficients(
        xtx.sum(axis=0), xty.sum(axis=0), count.sum(axis=0), candidate.ridge
    )
    fine = expand_to_fine(native, candidate.partition)
    effective = effective_max_beta(fine, family)
    payload = {
        "schema_version": 1,
        "method": "all-30 input-only temporal history residual forecast",
        "feature_system": "[1,-center,same_history20,cross_history20]",
        "family": family.name,
        "family_description": family.description,
        "partition": list(candidate.partition),
        "ridge": candidate.ridge,
        "response_clip": RESPONSE_CLIP,
        "forecast_clip": FORECAST_CLIP,
        "transform": torch.from_numpy(family.transform.astype(np.float32)),
        "native_beta": torch.from_numpy(native),
        "effective_beta_fine4x8": torch.from_numpy(effective),
        "trajectory_ids": list(stats["trajectory_ids"]),
        "joint_calibration": {
            "prediction_coupling": PREDICTION_COUPLING,
            "bounds_total_damping": BOUNDS_TOTAL_DAMPING,
            "width_scale": WIDTH_SCALE,
            "pixel_previous_weight": PIXEL_PREVIOUS_WEIGHT,
            "pixel_state_decay": PIXEL_STATE_DECAY,
            "pixel_damping": PIXEL_DAMPING,
        },
        "nested_selected_validation": exact["nested_selected"],
        "deploy_selection": selection,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Candidate):
        return {
            "key": value.key,
            "family": value.family,
            "partition": list(value.partition),
            "ridge": value.ridge,
        }
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stats-cache", type=Path, default=DEFAULT_STATS_CACHE)
    parser.add_argument("--deploy-output", type=Path, default=DEFAULT_DEPLOY)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--rebuild-stats", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    started = time.perf_counter()
    data = load_data()
    families = make_families()
    if args.stats_cache.exists() and not args.rebuild_stats:
        stats = torch.load(args.stats_cache, map_location="cpu", weights_only=False)
        if tuple(stats["trajectory_ids"]) != tuple(data.trajectory_ids):
            raise RuntimeError("temporal stats cache trajectory order mismatch")
        print(f"temporal_loaded_stats={args.stats_cache}", flush=True)
    else:
        stats = collect_stats(data, args.progress_every)
        args.stats_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(stats, args.stats_cache)
        print(f"temporal_saved_stats={args.stats_cache}", flush=True)
    names = np.asarray(data.trajectory_ids)
    dev_mask = np.asarray([name in DEV_IDS for name in names])
    choices, search_report, banks = select_family_candidates(
        stats, families, dev_mask, progress=True
    )
    exact = replay_exact(data, families, banks, args.progress_every)
    deploy_family, deploy_candidate, deploy_selection = deploy_consensus(
        families, choices, search_report, exact
    )
    nested = exact["nested_selected"]
    robust = nested["robustness"]
    robust_gain = (
        nested["delta_vs_reference"]["global"]["final_proxy"] > 0.0
        and nested["delta_vs_reference"]["dev10"]["final_proxy"] > 0.0
        and nested["delta_vs_reference"]["hold20"]["final_proxy"] > 0.0
        and robust["all_angles_positive"]
        and robust["worst_delta"] > -0.05
    )
    deploy_written = False
    if robust_gain:
        export_deploy(
            args.deploy_output, stats, deploy_family, deploy_candidate,
            exact, deploy_selection,
        )
        deploy_written = True
    raw_arrays = exact.pop("raw_arrays")
    del raw_arrays
    report = {
        "schema_version": 1,
        "experiment": "nested temporal-history forecast",
        "status": "completed",
        "data": {
            "source": 'train_real balanced-30 caches',
            "windows": len(data.simulation_ids),
            "trajectories": len(data.trajectory_ids),
            "trajectory_ids": list(data.trajectory_ids),
        },
        "causality": {
            "capacity_selection": (
                'nested cross-validation'
            ),
        },
        "feature_families": [
            {"name": family.name, "description": family.description, "p": family.p}
            for family in families
        ],
        "search": {
            "partitions": [list(value) for value in PARTITIONS],
            "ridges": list(RIDGES),
            "response_clip": RESPONSE_CLIP,
            "family_choices": choices,
            "residual_cv": search_report,
        },
        "joint_calibration_reference": {
            "prediction_coupling": PREDICTION_COUPLING,
            "bounds_total_damping": BOUNDS_TOTAL_DAMPING,
            "width_scale": WIDTH_SCALE,
            "forecast_clip": FORECAST_CLIP,
            "pixel_previous_weight": PIXEL_PREVIOUS_WEIGHT,
            "pixel_state_decay": PIXEL_STATE_DECAY,
            "pixel_damping": PIXEL_DAMPING,
            "time_score_assumption": TIME_SCORE_ASSUMPTION,
        },
        "exact": exact,
        "deploy": {
            "robust_gain_gate_passed": robust_gain,
            "selected_family": deploy_family.name,
            "selected_candidate": deploy_candidate,
            "selection": deploy_selection,
            "artifact_written": deploy_written,
            "artifact": str(args.deploy_output) if deploy_written else None,
        },
        "duration_seconds": time.perf_counter() - started,
        "artifacts": {
            "source": 'train_real caches',
            "crossfit": "complementary nested trajectory-level validation",
            "generated": [str(args.output), str(args.stats_cache)]
            + ([str(args.deploy_output)] if deploy_written else []),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_safe(report), indent=2) + "\n")
    print(f"temporal_report={args.output}", flush=True)


if __name__ == "__main__":
    main()
