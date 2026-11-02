"""Submission entry point."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as functional

from causal_ridge import CausalRidgeCalibrator
from causal_sps import CausalSPSHedgeController
from cuda_graph_forward import maybe_build_cuda_graph_forward_borrowed
from load_baseline import load_baseline


EXPECTED_SHAPE = (1, 20, 32, 64, 3)
RIDGE = 0.01
FORGETTING_FACTOR = 1.0
TARGET_STD = (0.0968104079366, 0.0159636437893, 1.0)
INPUT_STD = (0.0968056544662, 0.0159606840461, 1.0)
INPUT_MEAN = (0.15496085584163666, -0.0005139928543940187, 0.0)
TARGET_MEAN = (0.15496256947517395, -0.0005177936982363462, 0.0)
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
LEARNED_PREDICTION_DAMPING = 1.30
LEARNED_BOUNDS_DAMPING = 1.20
LEARNED_WIDTH_SCALE = 0.95
LEARNED_FORECAST_CLIP = 0.75
PIXEL_TKE_PREVIOUS_WEIGHT = 0.50
PIXEL_TKE_STATE_DECAY = 0.80
PIXEL_TKE_DAMPING = 0.35
PIXEL_TKE_SCALE_MIN = 0.80
PIXEL_TKE_SCALE_MAX = 1.25
PIXEL_TKE_LOG_MIN = -0.6931471805599453
PIXEL_TKE_LOG_MAX = 0.6931471805599453
SUPERVISED_TKE_STATE_DECAY = 0.80
SUPERVISED_TKE_STATE_CLIP = 0.60
SUPERVISED_TKE_RESPONSE_CLIP = 0.35
SUPERVISED_TKE_DAMPING = 0.80
SUPERVISED_TKE_SCALE_MIN = 0.75
SUPERVISED_TKE_SCALE_MAX = 1.3333333333333333
SUPERVISED_TKE_EPSILON = 1.0e-10


class CausalRidgeTTTModel:

    def __init__(
        self,
        base: torch.nn.Module,
        device: str,
        learned_coefficients: Optional[torch.Tensor] = None,
        learned_width_scale: Optional[torch.Tensor] = None,
        supervised_tke_coefficients: Optional[torch.Tensor] = None,
        temporal_history_coefficients: Optional[torch.Tensor] = None,
    ) -> None:
        self.device = torch.device(device)
        self.base = base.to(self.device).eval()
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        (
            self._base_forward,
            self._cuda_graph_runner,
            self._cuda_graph_initial_status,
        ) = maybe_build_cuda_graph_forward_borrowed(
            self.base,
            self.device,
            EXPECTED_SHAPE,
            dtype=torch.float32,
        )
        self.calibrator = CausalRidgeCalibrator(
            time_steps=EXPECTED_SHAPE[1],
            observed_channels=2,
            ridge=RIDGE,
            forgetting_factor=FORGETTING_FACTOR,
            device=self.device,
            dtype=torch.float32,
        )
        self.interval_controller = CausalSPSHedgeController()
        self._previous_base_prediction: Optional[torch.Tensor] = None
        self._previous_ridge_prediction: Optional[torch.Tensor] = None
        self._previous_ridge_variance: Optional[torch.Tensor] = None
        self._previous_prediction: Optional[torch.Tensor] = None
        self._interval_scale_state: Optional[torch.Tensor] = None
        self._interval_horizon_state: Optional[torch.Tensor] = None
        self._interval_center_state: Optional[torch.Tensor] = None
        self._interval_center_initialized: Optional[torch.Tensor] = None
        self._prediction_mean_state: Optional[torch.Tensor] = None
        self._previous_interval_candidates: Optional[torch.Tensor] = None
        self._previous_ridge_variance_map: Optional[torch.Tensor] = None
        self._pixel_tke_log_ratio_state: Optional[torch.Tensor] = None
        self._supervised_tke_ratio_state: Optional[torch.Tensor] = None
        self._previous_supervised_tke_variance: Optional[torch.Tensor] = None
        self._target_std = torch.tensor(
            TARGET_STD, device=self.device, dtype=torch.float32
        ).reshape(1, 1, 1, 1, 3)
        self._input_std = torch.tensor(
            INPUT_STD, device=self.device, dtype=torch.float32
        ).reshape(1, 1, 1, 1, 3)
        self._input_mean = torch.tensor(
            INPUT_MEAN, device=self.device, dtype=torch.float32
        ).reshape(1, 1, 1, 1, 3)
        self._target_mean = torch.tensor(
            TARGET_MEAN, device=self.device, dtype=torch.float32
        ).reshape(1, 1, 1, 1, 3)
        self._input_to_target_scale = (
            self._input_std[..., :2] / self._target_std[..., :2]
        )
        self._input_to_target_offset = (
            (self._input_mean[..., :2] - self._target_mean[..., :2])
            / self._target_std[..., :2]
        )
        self._learned_horizon = (
            torch.arange(1, 21, device=self.device, dtype=torch.float32)
            .reshape(1, 20, 1, 1, 1)
            / 15.0
        )
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
        if learned_coefficients is None:
            self._learned_coefficients = None
            self._learned_folded_coefficients = None
        else:
            coefficients = torch.as_tensor(
                learned_coefficients, device=self.device, dtype=torch.float32
            )
            if tuple(coefficients.shape) != (20, 2, 4, 8, 4):
                raise ValueError(
                    "learned interval coefficients must have shape "
                    f"(20,2,4,8,4), got {tuple(coefficients.shape)}"
                )
            coefficients = coefficients.permute(0, 2, 3, 1, 4)
            coefficients = coefficients.repeat_interleave(8, dim=1)
            coefficients = coefficients.repeat_interleave(8, dim=2)
            self._learned_coefficients = coefficients.unsqueeze(0).contiguous()
            coefficient_0 = self._learned_coefficients[..., 0]
            coefficient_1 = self._learned_coefficients[..., 1]
            coefficient_2 = self._learned_coefficients[..., 2]
            coefficient_3 = self._learned_coefficients[..., 3]
            self._learned_folded_coefficients = torch.stack(
                (
                    coefficient_0,
                    coefficient_1,
                    coefficient_2 + coefficient_3,
                    coefficient_3 * self._learned_horizon,
                    -(coefficient_1 + coefficient_2 + coefficient_3),
                ),
                dim=-1,
            ).contiguous()

        if learned_width_scale is None:
            self._learned_width_scale = None
        else:
            width_scale = torch.as_tensor(
                learned_width_scale, device=self.device, dtype=torch.float32
            )
            if tuple(width_scale.shape) != (20, 2, 4, 8):
                raise ValueError(
                    "learned width scale must have shape (20,2,4,8), got "
                    f"{tuple(width_scale.shape)}"
                )
            width_scale = width_scale.permute(0, 2, 3, 1)
            width_scale = width_scale.repeat_interleave(8, dim=1)
            width_scale = width_scale.repeat_interleave(8, dim=2)
            self._learned_width_scale = width_scale.unsqueeze(0).contiguous()

        if supervised_tke_coefficients is None:
            self._supervised_tke_coefficients = None
        else:
            tke_coefficients = torch.as_tensor(
                supervised_tke_coefficients,
                device=self.device,
                dtype=torch.float32,
            )
            if tuple(tke_coefficients.shape) != (1024, 6):
                raise ValueError(
                    "supervised TKE coefficients must have shape (1024,6), got "
                    f"{tuple(tke_coefficients.shape)}"
                )
            tke_coefficients = tke_coefficients.reshape(16, 32, 2, 6)
            tke_coefficients = tke_coefficients.repeat_interleave(2, dim=0)
            tke_coefficients = tke_coefficients.repeat_interleave(2, dim=1)
            self._supervised_tke_coefficients = (
                tke_coefficients.unsqueeze(0).unsqueeze(0).contiguous()
            )

        if temporal_history_coefficients is None:
            self._temporal_history_coefficients = None
        else:
            history_coefficients = torch.as_tensor(
                temporal_history_coefficients,
                device=self.device,
                dtype=torch.float32,
            )
            if tuple(history_coefficients.shape) != (20, 2, 2, 4, 22):
                raise ValueError(
                    "temporal-history coefficients must have shape "
                    "(20,2,2,4,22), got "
                    f"{tuple(history_coefficients.shape)}"
                )
            history_coefficients = history_coefficients.permute(
                0, 2, 3, 1, 4
            )
            history_coefficients = history_coefficients.repeat_interleave(
                16, dim=1
            )
            history_coefficients = history_coefficients.repeat_interleave(
                16, dim=2
            )
            self._temporal_history_coefficients = (
                history_coefficients.unsqueeze(0).contiguous()
            )

    def reset_ttt_state(self) -> None:
        self.base.eval()
        self.calibrator.reset()
        self.interval_controller.reset()
        self._previous_base_prediction = None
        self._previous_ridge_prediction = None
        self._previous_ridge_variance = None
        self._previous_prediction = None
        self._interval_scale_state = None
        self._interval_horizon_state = None
        self._interval_center_state = None
        self._interval_center_initialized = None
        self._prediction_mean_state = None
        self._previous_interval_candidates = None
        self._previous_ridge_variance_map = None
        self._pixel_tke_log_ratio_state = None
        self._supervised_tke_ratio_state = None
        self._previous_supervised_tke_variance = None

    @property
    def cuda_graph_status(self) -> str:
        if self._cuda_graph_runner is None:
            return self._cuda_graph_initial_status
        return self._cuda_graph_runner.status

    @torch.inference_mode()
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
                or self._previous_ridge_variance is None
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
            if self._previous_ridge_variance_map is None:
                raise RuntimeError("pixel-TKE previous variance is missing")
            previous_target_physical_uv = (
                previous_target[..., :2] * self._target_std[..., :2]
            )
            previous_target_deviation = (
                previous_target_physical_uv
                - previous_target_physical_uv.mean(dim=1, keepdim=True)
            )
            previous_target_variance = previous_target_deviation.square().mean(dim=1)[0]
            observed_pixel_log_ratio = 0.5 * torch.log(
                (previous_target_variance + TKE_VARIANCE_EPSILON)
                / (
                    self._previous_ridge_variance_map
                    + TKE_VARIANCE_EPSILON
                )
            ).clamp(PIXEL_TKE_LOG_MIN, PIXEL_TKE_LOG_MAX)
            if self._pixel_tke_log_ratio_state is None:
                self._pixel_tke_log_ratio_state = observed_pixel_log_ratio.detach().clone()
            else:
                self._pixel_tke_log_ratio_state = (
                    PIXEL_TKE_STATE_DECAY * self._pixel_tke_log_ratio_state
                    + (1.0 - PIXEL_TKE_STATE_DECAY) * observed_pixel_log_ratio
                ).detach()
            if self._supervised_tke_coefficients is not None:
                if self._previous_supervised_tke_variance is None:
                    raise RuntimeError(
                        "supervised-TKE previous anchor variance is missing"
                    )
                supervised_observation = (
                    0.5
                    * torch.log(
                        (previous_target_variance + SUPERVISED_TKE_EPSILON)
                        / (
                            self._previous_supervised_tke_variance
                            + SUPERVISED_TKE_EPSILON
                        )
                    )
                ).clamp(
                    -SUPERVISED_TKE_STATE_CLIP,
                    SUPERVISED_TKE_STATE_CLIP,
                )
                if self._supervised_tke_ratio_state is None:
                    self._supervised_tke_ratio_state = (
                        supervised_observation.detach().clone()
                    )
                else:
                    self._supervised_tke_ratio_state = (
                        SUPERVISED_TKE_STATE_DECAY
                        * self._supervised_tke_ratio_state
                        + (1.0 - SUPERVISED_TKE_STATE_DECAY)
                        * supervised_observation
                    ).detach()
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
            previous_target_physical = (
                previous_target[..., :2] * self._target_std[..., :2]
            )
            prediction_variance = self._previous_ridge_variance
            target_variance = previous_target_physical.var(
                dim=1, unbiased=False
            ).mean(dim=(1, 2))
            previous_energy_scale = torch.sqrt(
                (target_variance + TKE_VARIANCE_EPSILON)
                / (prediction_variance + TKE_VARIANCE_EPSILON)
            ).reshape(-1, 1, 1, 1, 2)
            self.calibrator.update(self._previous_base_prediction, previous_target)
            adapt_loss = None

        with torch.no_grad():
            current_base_prediction = self._base_forward(input_tensor)
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

            ridge_returned_prediction = prediction
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
                ridge_returned_prediction = prediction + functional.pad(
                    prediction_mean_shift_physical / self._target_std[..., :2],
                    (0, 1),
                )
            returned_prediction = ridge_returned_prediction.clone()

            if self._learned_coefficients is not None:
                interval_center = prediction[..., :2]
                if center_shift_norm is not None:
                    interval_center = interval_center + center_shift_norm
                current_input_target_norm = (
                    input_tensor[..., :2] * self._input_to_target_scale
                    + self._input_to_target_offset
                )
                if self._temporal_history_coefficients is None:
                    coefficients = self._learned_folded_coefficients
                    input_last = current_input_target_norm[:, -1:].expand_as(
                        interval_center
                    )
                    input_trend = (
                        current_input_target_norm[:, -5:].mean(
                            dim=1, keepdim=True
                        )
                        - current_input_target_norm[:, :5].mean(
                            dim=1, keepdim=True
                        )
                    )
                    learned_forecast = coefficients[..., 0]
                    learned_forecast = (
                        learned_forecast
                        + coefficients[..., 1] * current_input_target_norm
                    )
                    learned_forecast = (
                        learned_forecast + coefficients[..., 2] * input_last
                    )
                    learned_forecast = (
                        learned_forecast + coefficients[..., 3] * input_trend
                    )
                    learned_forecast = (
                        learned_forecast
                        + coefficients[..., 4] * interval_center
                    )
                else:
                    history_coefficients = (
                        self._temporal_history_coefficients
                    )
                    learned_forecast = history_coefficients[..., 0]
                    learned_forecast = (
                        learned_forecast
                        - history_coefficients[..., 1] * interval_center
                    )
                    learned_forecast = learned_forecast + torch.einsum(
                        "thwci,bihwc->bthwc",
                        history_coefficients[0, ..., 2:],
                        current_input_target_norm,
                    )
                learned_forecast = learned_forecast.clamp(
                    -LEARNED_FORECAST_CLIP, LEARNED_FORECAST_CLIP
                )
                returned_prediction[..., :2].add_(
                    LEARNED_PREDICTION_DAMPING * learned_forecast
                )
                if self._learned_width_scale is None:
                    learned_half_width = (
                        half_width_norm[..., :2] * LEARNED_WIDTH_SCALE
                    )
                else:
                    learned_half_width = (
                        half_width_norm[..., :2]
                        * self._learned_width_scale
                    )
                learned_center = (
                    interval_center
                    + LEARNED_BOUNDS_DAMPING * learned_forecast
                )
                lower[..., :2] = learned_center - learned_half_width
                upper[..., :2] = learned_center + learned_half_width

            current_input_deviation = (
                current_input_physical
                - current_input_physical.mean(dim=1, keepdim=True)
            )
            current_input_variance_map = (
                current_input_deviation.square().mean(dim=1)[0]
            )
            ridge_velocity = ridge_returned_prediction[..., :2]
            ridge_mean = ridge_velocity.mean(dim=1, keepdim=True)
            ridge_deviation = ridge_velocity - ridge_mean
            ridge_variance_map = (
                ridge_deviation * self._target_std[..., :2]
            ).square().mean(dim=1)[0]
            pixel_scale = torch.ones_like(current_input_variance_map)
            if self._pixel_tke_log_ratio_state is not None:
                current_input_log_ratio = 0.5 * torch.log(
                    (current_input_variance_map + TKE_VARIANCE_EPSILON)
                    / (ridge_variance_map + TKE_VARIANCE_EPSILON)
                ).clamp(PIXEL_TKE_LOG_MIN, PIXEL_TKE_LOG_MAX)
                blended_pixel_log_ratio = (
                    PIXEL_TKE_PREVIOUS_WEIGHT
                    * self._pixel_tke_log_ratio_state
                    + (1.0 - PIXEL_TKE_PREVIOUS_WEIGHT)
                    * current_input_log_ratio
                )
                pixel_scale = torch.exp(
                    PIXEL_TKE_DAMPING * blended_pixel_log_ratio
                ).clamp(PIXEL_TKE_SCALE_MIN, PIXEL_TKE_SCALE_MAX)

            coupled_velocity = returned_prediction[..., :2]
            anchor_mean = coupled_velocity.mean(dim=1, keepdim=True)
            anchor_deviation = pixel_scale[None, None] * (
                coupled_velocity - anchor_mean
            )
            anchor_variance = (
                anchor_deviation * self._target_std[..., :2]
            ).square().mean(dim=1)
            returned_prediction[..., :2] = anchor_mean + anchor_deviation
            if self._supervised_tke_coefficients is not None:
                supervised_input_variance = (
                    current_input_variance_map.unsqueeze(0)
                )
                input_early_energy = (
                    current_input_deviation[:, :5].square().mean(dim=1)
                )
                input_late_energy = (
                    current_input_deviation[:, -5:].square().mean(dim=1)
                )
                input_over_prediction = (
                    0.5
                    * torch.log(
                        (
                            supervised_input_variance
                            + SUPERVISED_TKE_EPSILON
                        )
                        / (anchor_variance + SUPERVISED_TKE_EPSILON)
                    )
                ).clamp(-0.8, 0.8)
                prediction_energy_level = (
                    torch.log(
                        (anchor_variance + SUPERVISED_TKE_EPSILON)
                        / 1.0e-4
                    )
                    / 3.0
                ).clamp(-1.0, 1.0)
                input_late_over_early = (
                    0.5
                    * torch.log(
                        (input_late_energy + SUPERVISED_TKE_EPSILON)
                        / (input_early_energy + SUPERVISED_TKE_EPSILON)
                    )
                ).clamp(-1.0, 1.0)
                coefficient = self._supervised_tke_coefficients
                response = coefficient[..., 0]
                response = response + coefficient[..., 1] * (
                    input_over_prediction[:, None]
                )
                response = response + coefficient[..., 2] * (
                    prediction_energy_level[:, None]
                )
                response = response + coefficient[..., 3] * (
                    input_late_over_early[:, None]
                )
                if self._supervised_tke_ratio_state is not None:
                    response = response + coefficient[..., 4] * (
                        self._supervised_tke_ratio_state[None, None]
                    )
                    response = response + coefficient[..., 5]
                response = response.clamp(
                    -SUPERVISED_TKE_RESPONSE_CLIP,
                    SUPERVISED_TKE_RESPONSE_CLIP,
                )
                supervised_scale = torch.exp(
                    SUPERVISED_TKE_DAMPING * response
                ).clamp(
                    SUPERVISED_TKE_SCALE_MIN,
                    SUPERVISED_TKE_SCALE_MAX,
                )
                returned_prediction[..., :2] = (
                    anchor_mean
                    + supervised_scale * anchor_deviation
                )

        self._previous_base_prediction = current_base_prediction.detach()
        self._previous_ridge_prediction = ridge_prediction.detach()
        self._previous_ridge_variance = ridge_variance.detach()
        self._previous_prediction = prediction.detach()
        self._previous_interval_candidates = interval_candidates
        self._previous_ridge_variance_map = ridge_variance_map.detach()
        self._previous_supervised_tke_variance = (
            anchor_variance[0].detach()
        )
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
    learned_path = os.path.join(submission_dir, "learned_interval.pt")
    if not os.path.isfile(learned_path):
        raise FileNotFoundError(f"missing learned adapter: {learned_path}")
    learned_payload = torch.load(
        learned_path, map_location=device, weights_only=False
    )
    learned_coefficients = learned_payload.get("coefficients")
    if learned_coefficients is None:
        raise KeyError("learned adapter has no coefficients tensor")
    width_path = os.path.join(submission_dir, "learned_width.pt")
    if not os.path.isfile(width_path):
        raise FileNotFoundError(f"missing learned width map: {width_path}")
    width_payload = torch.load(
        width_path, map_location=device, weights_only=False
    )
    learned_width_scale = width_payload.get("width_scale")
    if learned_width_scale is None:
        raise KeyError("learned width map has no width_scale tensor")
    supervised_tke_path = os.path.join(
        submission_dir, "supervised_tke.pt"
    )
    if not os.path.isfile(supervised_tke_path):
        raise FileNotFoundError(
            f"missing supervised TKE coefficients: {supervised_tke_path}"
        )
    supervised_tke_payload = torch.load(
        supervised_tke_path, map_location=device, weights_only=False
    )
    supervised_tke_coefficients = supervised_tke_payload.get(
        "coefficients"
    )
    if supervised_tke_coefficients is None:
        raise KeyError(
            "supervised TKE payload has no coefficients tensor"
        )
    temporal_history_path = os.path.join(
        submission_dir, "temporal_history.pt"
    )
    if not os.path.isfile(temporal_history_path):
        raise FileNotFoundError(
            f"missing temporal-history coefficients: {temporal_history_path}"
        )
    temporal_history_payload = torch.load(
        temporal_history_path, map_location=device, weights_only=False
    )
    temporal_history_coefficients = temporal_history_payload.get(
        "native_beta"
    )
    if temporal_history_coefficients is None:
        raise KeyError(
            "temporal-history payload has no native_beta tensor"
        )
    return CausalRidgeTTTModel(
        base,
        device,
        learned_coefficients,
        learned_width_scale,
        supervised_tke_coefficients,
        temporal_history_coefficients,
    )


__all__ = ["CausalRidgeTTTModel", "get_ttt_model"]
