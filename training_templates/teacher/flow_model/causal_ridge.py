"""Recursive ridge calibrator."""

from __future__ import annotations
from typing import Optional
import torch


class CausalRidgeCalibrator:

    def __init__(
        self,
        *,
        time_steps: int = 20,
        observed_channels: int = 2,
        ridge: float = 0.01,
        forgetting_factor: float = 1.0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if time_steps <= 0:
            raise ValueError("time_steps must be positive")
        if observed_channels <= 0:
            raise ValueError("observed_channels must be positive")
        if ridge <= 0.0:
            raise ValueError("ridge must be positive")
        if not 0.0 < forgetting_factor <= 1.0:
            raise ValueError("forgetting_factor must be in (0, 1]")
        self.time_steps = int(time_steps)
        self.observed_channels = int(observed_channels)
        self.ridge = float(ridge)
        self.forgetting_factor = float(forgetting_factor)
        self.device = torch.device(device)
        self.dtype = dtype
        self.output_dimension = self.time_steps * self.observed_channels
        self.feature_dimension = self.output_dimension + 1
        self.reset()

    def reset(self) -> None:
        shape = (self.feature_dimension, self.feature_dimension)
        self._gram = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self._cross = torch.zeros(
            (self.feature_dimension, self.output_dimension),
            dtype=self.dtype,
            device=self.device,
        )
        self._weights = torch.zeros_like(self._cross)
        self._ridge_matrix = self.ridge * torch.eye(
            self.feature_dimension, dtype=self.dtype, device=self.device
        )
        self._ones: Optional[torch.Tensor] = None

    def _local_matrix(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 5 or value.shape[1] != self.time_steps:
            raise ValueError(
                "prediction must have shape (batch, 20, height, width, channels)"
            )
        if value.shape[-1] < self.observed_channels:
            raise ValueError("prediction does not contain the observed channels")
        selected = value[..., : self.observed_channels]
        return selected.permute(0, 2, 3, 1, 4).reshape(-1, self.output_dimension)

    def update(
        self, previous_base_prediction: torch.Tensor, previous_target: torch.Tensor
    ) -> None:
        if previous_base_prediction.shape != previous_target.shape:
            raise ValueError("previous prediction and target shapes must match")
        base = (
            self._local_matrix(previous_base_prediction)
            .detach()
            .to(device=self.device, dtype=self.dtype)
        )
        target = (
            self._local_matrix(previous_target)
            .detach()
            .to(device=self.device, dtype=self.dtype)
        )
        sample_count = int(base.shape[0])
        if self._ones is None or self._ones.shape[0] != sample_count:
            self._ones = torch.ones(
                (sample_count, 1), dtype=self.dtype, device=self.device
            )
        design = torch.cat((self._ones, base), dim=1)
        residual = target - base
        scale = 1.0 / float(sample_count)
        candidate_gram = self.forgetting_factor * self._gram + scale * (
            design.transpose(0, 1) @ design
        )
        candidate_cross = self.forgetting_factor * self._cross + scale * (
            design.transpose(0, 1) @ residual
        )
        system = candidate_gram + self._ridge_matrix
        factor, _info = torch.linalg.cholesky_ex(system, check_errors=False)
        candidate_weights = torch.cholesky_solve(candidate_cross, factor)
        self._gram = candidate_gram
        self._cross = candidate_cross
        self._weights = candidate_weights
        return None

    def correct(self, base_prediction: torch.Tensor) -> torch.Tensor:
        local = self._local_matrix(base_prediction)
        ones = torch.ones((local.shape[0], 1), device=local.device, dtype=local.dtype)
        design = torch.cat((ones, local), dim=1)
        residual = design @ self._weights
        residual = residual.reshape(
            base_prediction.shape[0],
            base_prediction.shape[2],
            base_prediction.shape[3],
            self.time_steps,
            self.observed_channels,
        ).permute(0, 3, 1, 2, 4)
        corrected = base_prediction.clone()
        corrected[..., : self.observed_channels] = (
            corrected[..., : self.observed_channels] + residual
        )
        return corrected


__all__ = ["CausalRidgeCalibrator"]
