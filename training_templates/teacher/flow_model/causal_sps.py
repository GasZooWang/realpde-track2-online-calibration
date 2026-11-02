"""Hedge interval-width controller."""

from __future__ import annotations
from typing import Optional, Sequence, Tuple
import torch

SPS_SIGMA_GLOBAL = 0.0563870259
DEFAULT_MULTIPLIERS = (0.55, 0.7, 0.85, 1.0, 1.2, 1.45, 1.75)
DEFAULT_INITIAL_WEIGHTS = (0.02, 0.06, 0.14, 0.57, 0.14, 0.05, 0.02)
DEFAULT_MAX_HALF_WIDTH = (0.04, 0.032)


class CausalSPSHedgeController:

    def __init__(
        self,
        *,
        multipliers: Sequence[float] = DEFAULT_MULTIPLIERS,
        initial_weights: Sequence[float] = DEFAULT_INITIAL_WEIGHTS,
        learning_rate: float = 2.0,
        memory: float = 0.98,
        max_half_width: Sequence[float] = DEFAULT_MAX_HALF_WIDTH,
    ) -> None:
        if len(multipliers) < 2:
            raise ValueError("at least two interval experts are required")
        if len(multipliers) != len(initial_weights):
            raise ValueError("multipliers and initial_weights must align")
        if any((value <= 0.0 for value in multipliers)):
            raise ValueError("interval multipliers must be positive")
        if any((value <= 0.0 for value in initial_weights)):
            raise ValueError("initial expert weights must be positive")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 < memory <= 1.0:
            raise ValueError("memory must be in (0, 1]")
        if len(max_half_width) != 2 or any((value <= 0.0 for value in max_half_width)):
            raise ValueError("max_half_width must contain positive u/v values")
        weight_sum = float(sum(initial_weights))
        self.multipliers = tuple((float(value) for value in multipliers))
        self.initial_weights = tuple(
            (float(value) / weight_sum for value in initial_weights)
        )
        self.learning_rate = float(learning_rate)
        self.memory = float(memory)
        self.max_half_width = tuple((float(value) for value in max_half_width))
        self.reset()

    def reset(self) -> None:
        self._log_weights: Optional[torch.Tensor] = None
        self._multiplier_tensor: Optional[torch.Tensor] = None
        self._maximum_tensor: Optional[torch.Tensor] = None
        self._updates = 0

    def _initialize(self, base_half_width: torch.Tensor) -> None:
        if base_half_width.ndim != 5 or base_half_width.shape[-1] != 2:
            raise ValueError("base_half_width must have shape (B, T, H, W, 2)")
        batch, _time, height, width, channels = base_half_width.shape
        initial = torch.tensor(
            self.initial_weights,
            device=base_half_width.device,
            dtype=base_half_width.dtype,
        ).log()
        self._log_weights = (
            initial.reshape(-1, 1, 1, 1, 1)
            .expand(-1, batch, height, width, channels)
            .clone()
        )
        self._multiplier_tensor = torch.tensor(
            self.multipliers, device=base_half_width.device, dtype=base_half_width.dtype
        ).reshape(-1, 1, 1, 1, 1, 1)
        self._maximum_tensor = torch.tensor(
            self.max_half_width,
            device=base_half_width.device,
            dtype=base_half_width.dtype,
        ).reshape(1, 1, 1, 1, 1, 2)

    def update(
        self, previous_error_physical: torch.Tensor, previous_candidates: torch.Tensor
    ) -> None:
        if previous_error_physical.ndim != 5:
            raise ValueError("previous_error_physical must be a 5-D tensor")
        if previous_error_physical.shape[-1] != 2:
            raise ValueError("the SPS controller expects the u/v channels")
        if previous_candidates.ndim != 6:
            raise ValueError("previous_candidates must be a 6-D tensor")
        if previous_candidates.shape[0] != len(self.multipliers):
            raise ValueError("candidate bank size changed within a trajectory")
        if previous_candidates.shape[1] != previous_error_physical.shape[0]:
            raise ValueError("candidate and residual batches do not match")
        if previous_candidates.shape[-1] != previous_error_physical.shape[-1]:
            raise ValueError("candidate and residual channels do not match")
        for candidate_size, residual_size in zip(
            previous_candidates.shape[3:5], previous_error_physical.shape[2:4]
        ):
            if candidate_size not in (1, residual_size):
                raise ValueError(
                    "candidate and residual spatial shapes are not broadcastable"
                )
        if previous_candidates.shape[2] not in (1, previous_error_physical.shape[1]):
            raise ValueError("candidate time dimension is not broadcastable")
        if self._log_weights is None:
            self._initialize(previous_candidates[0])
        absolute_error = previous_error_physical.abs().unsqueeze(0)
        inside = absolute_error <= previous_candidates
        penalty = torch.exp(-2.0 * previous_candidates / SPS_SIGMA_GLOBAL)
        reward = (inside.to(previous_candidates.dtype) * penalty).mean(dim=2)
        advantage = reward - reward.mean(dim=0, keepdim=True)
        updated = self.memory * self._log_weights + self.learning_rate * advantage
        self._log_weights = updated.clamp(-16.0, 4.0)
        self._updates += 1

    def select(
        self, base_half_width: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if base_half_width.ndim != 5 or base_half_width.shape[-1] != 2:
            raise ValueError("base_half_width must have shape (B, T, H, W, 2)")
        if self._log_weights is None:
            self._initialize(base_half_width)
        candidates = torch.minimum(
            self._multiplier_tensor * base_half_width.unsqueeze(0), self._maximum_tensor
        )
        if self._updates == 0:
            selected = base_half_width
        else:
            weights = torch.softmax(self._log_weights, dim=0).unsqueeze(2)
            selected = (weights * candidates).sum(dim=0)
        return (selected, candidates.detach())


__all__ = ["CausalSPSHedgeController", "SPS_SIGMA_GLOBAL"]
