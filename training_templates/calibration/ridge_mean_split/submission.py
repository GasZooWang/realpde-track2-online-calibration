"""Submission entry point."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import torch

from causal_ridge import CausalRidgeCalibrator
from causal_sps import CausalSPSHedgeController
from load_baseline import load_baseline


EXPECTED_SHAPE = (1, 20, 32, 64, 3)
RIDGE = 0.01
FORGETTING_FACTOR = 1.0
TARGET_STD = (0.0968104079366, 0.0159636437893, 1.0)
INPUT_STD = (0.0968056544662, 0.0159606840461, 1.0)
PRIOR_HALF_WIDTH_PHYSICAL = (0.010, 0.008, 0.0)
INTERVAL_FLOOR_RATIO = 0.25
INTERVAL_ERROR_SCALE = 1.25
INTERVAL_CAP_RATIO = 2.0
INTERVAL_EMA_DECAY = 0.8
INTERVAL_HORIZON_DAMPING = 0.5
INTERVAL_HORIZON_MIN = 0.75
INTERVAL_HORIZON_MAX = 1.35
INTERVAL_HORIZON_EPSILON = 1e-8
INTERVAL_CENTER_DECAY = 0.875
INTERVAL_CENTER_ERROR_SCALE = (0.50, 0.25)
INTERVAL_CENTER_RIDGE_SCALE = (-0.30, -0.50)
INTERVAL_CENTER_TKE_SCALE = (-1.60, -1.20)
INTERVAL_CENTER_WIDTH_SCALE = (0.95, 0.95)
PREDICTION_MEAN_DECAY = 0.875
PREDICTION_MEAN_STATE_SCALE = (0.60, 0.27)
PREDICTION_MEAN_DISAGREEMENT_SCALE = (-0.525, -0.6375)
NORMALIZED_ZERO_TARGET = (-1.6006808280944824, 0.03243580833077431)
TKE_PREVIOUS_BLEND = 0.25
TKE_INPUT_BLEND = 0.75
TKE_DAMPING = 0.5
TKE_SCALE_MIN = 0.7
TKE_SCALE_MAX = 1.5
TKE_VARIANCE_EPSILON = 1e-8


class CausalRidgeTTTModel:

    def __init__(self, base: torch.nn.Module, device: str) -> None:
        self.device = torch.device(device)
        self.base = base.to(self.device).eval()
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.calibrator = CausalRidgeCalibrator(
            time_steps=EXPECTED_SHAPE[1],
            observed_channels=2,
            ridge=RIDGE,
            forgetting_factor=FORGETTING_FACTOR,
        )
        self.interval_controller = CausalSPSHedgeController()
        self._previous_base_prediction: Optional[torch.Tensor] = None
        self._previous_ridge_prediction: Optional[torch.Tensor] = None
        self._previous_prediction: Optional[torch.Tensor] = None
        self._interval_scale_state: Optional[torch.Tensor] = None
        self._interval_horizon_state: Optional[torch.Tensor] = None
        self._interval_center_state: Optional[torch.Tensor] = None
        self._interval_center_initialized: Optional[torch.Tensor] = None
        self._prediction_mean_state: Optional[torch.Tensor] = None
        self._previous_interval_candidates: Optional[torch.Tensor] = None
        self._target_std = torch.tensor(
            TARGET_STD, device=self.device, dtype=torch.float32
        ).reshape(1, 1, 1, 1, 3)
        self._input_std = torch.tensor(
            INPUT_STD, device=self.device, dtype=torch.float32
        ).reshape(1, 1, 1, 1, 3)
        self._prior_half_width = torch.tensor(
            PRIOR_HALF_WIDTH_PHYSICAL,
            device=self.device,
            dtype=torch.float32,
        ).reshape(1, 1, 1, 1, 3)
        self._interval_center_error_scale_norm = torch.tensor(
            INTERVAL_CENTER_ERROR_SCALE,
            device=self.device,
            dtype=torch.float32,
        ).reshape(1, 1, 1, 1, 2) / self._target_std[..., :2]
        self._interval_center_ridge_scale = torch.tensor(
            INTERVAL_CENTER_RIDGE_SCALE,
            device=self.device,
            dtype=torch.float32,
        ).reshape(1, 1, 1, 1, 2)
        self._interval_center_tke_scale = torch.tensor(
            INTERVAL_CENTER_TKE_SCALE,
            device=self.device,
            dtype=torch.float32,
        ).reshape(1, 1, 1, 1, 2)
        self._interval_center_width_scale = torch.tensor(
            INTERVAL_CENTER_WIDTH_SCALE,
            device=self.device,
            dtype=torch.float32,
        ).reshape(1, 1, 1, 1, 2)
        self._prediction_mean_state_scale = torch.tensor(
            PREDICTION_MEAN_STATE_SCALE,
            device=self.device,
            dtype=torch.float32,
        ).reshape(1, 1, 1, 1, 2)
        self._prediction_mean_disagreement_scale = torch.tensor(
            PREDICTION_MEAN_DISAGREEMENT_SCALE,
            device=self.device,
            dtype=torch.float32,
        ).reshape(1, 1, 1, 1, 2)
        self._normalized_zero_target = torch.tensor(
            NORMALIZED_ZERO_TARGET,
            device=self.device,
            dtype=torch.float32,
        ).reshape(1, 1, 1, 1, 2)

    def reset_ttt_state(self) -> None:
        self.base.eval()
        self.calibrator.reset()
        self.interval_controller.reset()
        self._previous_base_prediction = None
        self._previous_ridge_prediction = None
        self._previous_prediction = None
        self._interval_scale_state = None
        self._interval_horizon_state = None
        self._interval_center_state = None
        self._interval_center_initialized = None
        self._prediction_mean_state = None
        self._previous_interval_candidates = None

    def ttt_step(
        self,
        input_norm: torch.Tensor,
        prev_target_norm: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        input_tensor = torch.as_tensor(input_norm, device=self.device)
        if tuple(input_tensor.shape) != EXPECTED_SHAPE:
            raise ValueError(
                f"input shape {tuple(input_tensor.shape)} != {EXPECTED_SHAPE}"
            )

        adapt_loss: Optional[float] = None
        interval_scale: Optional[torch.Tensor] = None
        previous_energy_scale = torch.ones(
            (input_tensor.shape[0], 1, 1, 1, 2),
            device=self.device,
            dtype=input_tensor.dtype,
        )
        if prev_target_norm is not None:
            if (
                self._previous_base_prediction is None
                or self._previous_ridge_prediction is None
                or self._previous_prediction is None
                or self._previous_interval_candidates is None
            ):
                raise RuntimeError("previous target was provided without a causal cache")
            previous_target = torch.as_tensor(prev_target_norm, device=self.device)
            if tuple(previous_target.shape) != EXPECTED_SHAPE:
                raise ValueError(
                    f"previous target shape {tuple(previous_target.shape)} "
                    f"!= {EXPECTED_SHAPE}"
                )
            previous_error_physical = (
                previous_target - self._previous_prediction
            ) * self._target_std
            prediction_mean_observation = previous_error_physical[
                ..., :2
            ].mean(dim=1, keepdim=True).detach()
            if self._prediction_mean_state is None:
                self._prediction_mean_state = prediction_mean_observation.clone()
            else:
                self._prediction_mean_state = (
                    PREDICTION_MEAN_DECAY * self._prediction_mean_state
                    + (1.0 - PREDICTION_MEAN_DECAY)
                    * prediction_mean_observation
                )
            self.interval_controller.update(
                previous_error_physical[..., :2],
                self._previous_interval_candidates,
            )
            valid_center_observation = (
                previous_target[..., :2] != self._normalized_zero_target
            )
            center_observation = previous_error_physical[..., :2].detach()
            if self._interval_center_state is None:
                self._interval_center_state = torch.where(
                    valid_center_observation,
                    center_observation,
                    torch.zeros_like(center_observation),
                )
                self._interval_center_initialized = (
                    valid_center_observation.clone()
                )
            else:
                if self._interval_center_initialized is None:
                    raise RuntimeError("interval center initialization is missing")
                center_update = torch.where(
                    self._interval_center_initialized,
                    INTERVAL_CENTER_DECAY * self._interval_center_state
                    + (1.0 - INTERVAL_CENTER_DECAY) * center_observation,
                    center_observation,
                )
                self._interval_center_state = torch.where(
                    valid_center_observation,
                    center_update,
                    self._interval_center_state,
                )
                self._interval_center_initialized = (
                    self._interval_center_initialized
                    | valid_center_observation
                )
            interval_observation = torch.sqrt(
                (previous_error_physical * previous_error_physical).mean(
                    dim=1, keepdim=True
                )
            )
            if self._interval_scale_state is None:
                self._interval_scale_state = interval_observation
            else:
                self._interval_scale_state = (
                    INTERVAL_EMA_DECAY * self._interval_scale_state
                    + (1.0 - INTERVAL_EMA_DECAY) * interval_observation
                )
            interval_scale = self._interval_scale_state
            horizon_observation = torch.sqrt(
                (previous_error_physical[..., :2] ** 2).mean(
                    dim=(2, 3), keepdim=True
                )
            )
            horizon_mean = horizon_observation.mean(dim=1, keepdim=True)
            horizon_observation = torch.where(
                horizon_mean > INTERVAL_HORIZON_EPSILON,
                horizon_observation
                / horizon_mean.clamp_min(INTERVAL_HORIZON_EPSILON),
                torch.ones_like(horizon_observation),
            )
            if self._interval_horizon_state is None:
                self._interval_horizon_state = horizon_observation
            else:
                self._interval_horizon_state = (
                    INTERVAL_EMA_DECAY * self._interval_horizon_state
                    + (1.0 - INTERVAL_EMA_DECAY) * horizon_observation
                )
            previous_prediction_physical = (
                self._previous_ridge_prediction[..., :2]
                * self._target_std[..., :2]
            )
            previous_target_physical = (
                previous_target[..., :2] * self._target_std[..., :2]
            )
            prediction_variance = previous_prediction_physical.var(
                dim=1, unbiased=False
            ).mean(dim=(1, 2))
            target_variance = previous_target_physical.var(
                dim=1, unbiased=False
            ).mean(dim=(1, 2))
            previous_energy_scale = torch.sqrt(
                (target_variance + TKE_VARIANCE_EPSILON)
                / (prediction_variance + TKE_VARIANCE_EPSILON)
            ).reshape(-1, 1, 1, 1, 2)
            adapt_loss = self.calibrator.update(
                self._previous_base_prediction, previous_target
            )

        with torch.no_grad():
            current_base_prediction = self.base(input_tensor)
            ridge_prediction = self.calibrator.correct(current_base_prediction)
            current_input_physical = (
                input_tensor[..., :2] * self._input_std[..., :2]
            )
            ridge_prediction_physical = (
                ridge_prediction[..., :2] * self._target_std[..., :2]
            )
            input_variance = current_input_physical.var(
                dim=1, unbiased=False
            ).mean(dim=(1, 2))
            ridge_variance = ridge_prediction_physical.var(
                dim=1, unbiased=False
            ).mean(dim=(1, 2))
            input_energy_scale = torch.sqrt(
                (input_variance + TKE_VARIANCE_EPSILON)
                / (ridge_variance + TKE_VARIANCE_EPSILON)
            ).reshape(-1, 1, 1, 1, 2)
            raw_tke_scale = torch.exp(
                TKE_PREVIOUS_BLEND * torch.log(previous_energy_scale)
                + TKE_INPUT_BLEND * torch.log(input_energy_scale)
            )
            tke_scale = (
                1.0 + TKE_DAMPING * (raw_tke_scale - 1.0)
            ).clamp(TKE_SCALE_MIN, TKE_SCALE_MAX)
            velocity = ridge_prediction[..., :2]
            temporal_mean = velocity.mean(dim=1, keepdim=True)
            prediction = ridge_prediction.clone()
            prediction[..., :2] = temporal_mean + tke_scale * (
                velocity - temporal_mean
            )

            if interval_scale is None:
                base_half_width_physical = self._prior_half_width
            else:
                base_half_width_physical = torch.minimum(
                    INTERVAL_FLOOR_RATIO * self._prior_half_width
                    + INTERVAL_ERROR_SCALE * interval_scale,
                    INTERVAL_CAP_RATIO * self._prior_half_width,
                )

            if self._interval_horizon_state is not None:
                horizon_factor_uv = (
                    1.0
                    + INTERVAL_HORIZON_DAMPING
                    * (self._interval_horizon_state - 1.0)
                ).clamp(INTERVAL_HORIZON_MIN, INTERVAL_HORIZON_MAX)
                horizon_factor = torch.ones(
                    (*horizon_factor_uv.shape[:-1], 3),
                    device=self.device,
                    dtype=input_tensor.dtype,
                )
                horizon_factor[..., :2] = horizon_factor_uv
                base_half_width_physical = (
                    base_half_width_physical * horizon_factor
                )

            selected_half_width_uv, interval_candidates = (
                self.interval_controller.select(
                    base_half_width_physical[..., :2]
                )
            )
            half_width_physical = torch.zeros(
                (*selected_half_width_uv.shape[:-1], 3),
                device=self.device,
                dtype=input_tensor.dtype,
            )
            half_width_physical[..., :2] = selected_half_width_uv

            center_shift_norm: Optional[torch.Tensor] = None
            if self._interval_center_state is not None:
                current_ridge_correction_norm = (
                    ridge_prediction[..., :2]
                    - current_base_prediction[..., :2]
                )
                current_tke_correction_norm = (
                    prediction[..., :2] - ridge_prediction[..., :2]
                )
                center_shift_norm = (
                    self._interval_center_error_scale_norm
                    * self._interval_center_state
                    + self._interval_center_ridge_scale
                    * current_ridge_correction_norm
                    + self._interval_center_tke_scale
                    * current_tke_correction_norm
                )
                half_width_physical[..., :2] = (
                    half_width_physical[..., :2]
                    * self._interval_center_width_scale
                )

            half_width_norm = half_width_physical / self._target_std
            lower = prediction - half_width_norm
            upper = prediction + half_width_norm
            if center_shift_norm is not None:
                lower[..., :2].add_(center_shift_norm)
                upper[..., :2].add_(center_shift_norm)

            returned_prediction = prediction.clone()
            if self._prediction_mean_state is not None:
                disagreement_mean_physical = (
                    (
                        prediction[..., :2]
                        - current_base_prediction[..., :2]
                    )
                    * self._target_std[..., :2]
                ).mean(dim=1, keepdim=True)
                prediction_mean_shift_physical = (
                    self._prediction_mean_state_scale
                    * self._prediction_mean_state
                    + self._prediction_mean_disagreement_scale
                    * disagreement_mean_physical
                )
                returned_prediction[..., :2].add_(
                    prediction_mean_shift_physical / self._target_std[..., :2]
                )

        self._previous_base_prediction = current_base_prediction.detach()
        self._previous_ridge_prediction = ridge_prediction.detach()
        self._previous_prediction = prediction.detach()
        self._previous_interval_candidates = interval_candidates
        return returned_prediction, {
            "adapt_loss": adapt_loss,
            "lower": lower,
            "upper": upper,
        }


def get_ttt_model(submission_dir: str, device: str) -> CausalRidgeTTTModel:
    checkpoint_path = os.path.join(submission_dir, "model.pth")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
    base, _metadata = load_baseline(
        "cno", checkpoint_path, device=device, strict=True
    )
    return CausalRidgeTTTModel(base, device)


__all__ = ["CausalRidgeTTTModel", "get_ttt_model"]
