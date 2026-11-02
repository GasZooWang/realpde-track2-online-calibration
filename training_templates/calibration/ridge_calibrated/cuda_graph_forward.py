"""CUDA graph forward pass."""

from __future__ import annotations

import os
from typing import Callable, Optional, Sequence, Tuple

import torch


CUDA_GRAPH_ENV = "REALPDE_CUDA_GRAPH"
_ENABLED_MODES = frozenset(("1", "auto", "on", "true", "yes"))
_DISABLED_MODES = frozenset(("0", "eager", "off", "false", "no"))


def _normalized_mode(mode: Optional[str]) -> str:
    value = os.environ.get(CUDA_GRAPH_ENV, "off") if mode is None else mode
    return str(value).strip().lower()


class FrozenCUDAGraphForward:

    def __init__(
        self,
        base: torch.nn.Module,
        device: torch.device | str,
        expected_shape: Sequence[int],
        *,
        dtype: torch.dtype = torch.float32,
        warmup_steps: int = 3,
    ) -> None:
        self.base = base
        self.device = torch.device(device)
        self.expected_shape = tuple(int(value) for value in expected_shape)
        self.dtype = dtype
        self.warmup_steps = max(1, int(warmup_steps))
        self._capture_autocast_enabled = bool(torch.is_autocast_enabled())
        self._graph = None
        self._static_input: Optional[torch.Tensor] = None
        self._static_output: Optional[torch.Tensor] = None
        self._active = False
        self._status = "capture-not-attempted"

        if self.device.type != "cuda":
            self._status = "disabled-non-cuda-device"
            return
        if not torch.cuda.is_available():
            self._status = "disabled-cuda-unavailable"
            return
        if self.base.training:
            self._status = "disabled-base-is-training"
            return
        if any(parameter.requires_grad for parameter in self.base.parameters()):
            self._status = "disabled-base-is-not-frozen"
            return
        if torch.cuda.is_current_stream_capturing():
            self._status = "disabled-parent-stream-is-capturing"
            return

        try:
            self._capture_and_verify()
        except Exception as error:
            self._deactivate(
                "capture-failed:"
                f"{type(error).__name__}:{str(error).replace(chr(10), ' ')[:240]}"
            )

    @property
    def active(self) -> bool:
        return self._active

    @property
    def status(self) -> str:
        return self._status

    def _capture_and_verify(self) -> None:
        with torch.cuda.device(self.device), torch.no_grad():
            element_count = 1
            for extent in self.expected_shape:
                element_count *= extent
            self._static_input = torch.linspace(
                -0.5,
                0.5,
                element_count,
                device=self.device,
                dtype=self.dtype,
            ).reshape(self.expected_shape)
            current_stream = torch.cuda.current_stream(self.device)
            capture_stream = torch.cuda.Stream(device=self.device)
            capture_stream.wait_stream(current_stream)
            with torch.cuda.stream(capture_stream):
                warmup_output = None
                for _ in range(self.warmup_steps):
                    warmup_output = self.base(self._static_input)
                if warmup_output is None:
                    raise RuntimeError("warmup produced no output")
                eager_probe_output = warmup_output.detach().clone()
            current_stream.wait_stream(capture_stream)
            del warmup_output

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                graph,
                stream=capture_stream,
                capture_error_mode="global",
            ):
                static_output = self.base(self._static_input)

            self._graph = graph
            self._static_output = static_output

            graph.replay()
            if (
                tuple(static_output.shape) != tuple(eager_probe_output.shape)
                or static_output.dtype != eager_probe_output.dtype
                or not torch.equal(static_output, eager_probe_output)
            ):
                raise RuntimeError("captured output failed exact eager equivalence")

            self._active = True
            self._status = "active-exact-probe-passed"

    def _deactivate(self, reason: str) -> None:
        graph = self._graph
        self._active = False
        self._status = reason

        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.synchronize(self.device)
            except Exception:
                pass
        if graph is not None:
            try:
                graph.reset()
            except Exception:
                pass
        self._graph = None
        self._static_input = None
        self._static_output = None

    def close(self) -> None:
        self._deactivate("disabled-explicitly")

    def __call__(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self._replay(input_tensor, clone_output=True)

    def replay_borrowed(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self._replay(input_tensor, clone_output=False)

    def _replay(
        self,
        input_tensor: torch.Tensor,
        *,
        clone_output: bool,
    ) -> torch.Tensor:
        if not self._active:
            return self.base(input_tensor)
        if (
            tuple(input_tensor.shape) != self.expected_shape
            or input_tensor.device != self.device
            or input_tensor.dtype != self.dtype
            or (
                self.device.type == "cuda"
                and bool(torch.is_autocast_enabled())
                != self._capture_autocast_enabled
            )
        ):
            return self.base(input_tensor)

        graph = self._graph
        static_input = self._static_input
        static_output = self._static_output
        if graph is None or static_input is None or static_output is None:
            self._deactivate("runtime-missing-static-state")
            return self.base(input_tensor)

        try:
            if self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    static_input.copy_(input_tensor)
                    graph.replay()
                    return static_output.clone() if clone_output else static_output
            else:
                static_input.copy_(input_tensor)
                graph.replay()
                return static_output.clone() if clone_output else static_output
        except Exception as error:
            self._deactivate(
                "replay-failed:"
                f"{type(error).__name__}:{str(error).replace(chr(10), ' ')[:240]}"
            )
            return self.base(input_tensor)


def maybe_build_cuda_graph_forward(
    base: torch.nn.Module,
    device: torch.device | str,
    expected_shape: Sequence[int],
    *,
    dtype: torch.dtype = torch.float32,
    mode: Optional[str] = None,
) -> Tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Optional[FrozenCUDAGraphForward],
    str,
]:

    normalized_mode = _normalized_mode(mode)
    if normalized_mode in _DISABLED_MODES:
        return base, None, f"disabled-by-mode:{normalized_mode}"
    if normalized_mode not in _ENABLED_MODES:
        return base, None, f"disabled-invalid-mode:{normalized_mode}"

    resolved_device = torch.device(device)
    if resolved_device.type != "cuda":
        return base, None, "disabled-non-cuda-device"
    if not torch.cuda.is_available():
        return base, None, "disabled-cuda-unavailable"

    runner = FrozenCUDAGraphForward(
        base,
        resolved_device,
        expected_shape,
        dtype=dtype,
    )
    if not runner.active:
        return base, None, runner.status
    return runner, runner, runner.status


def maybe_build_cuda_graph_forward_borrowed(
    base: torch.nn.Module,
    device: torch.device | str,
    expected_shape: Sequence[int],
    *,
    dtype: torch.dtype = torch.float32,
    mode: Optional[str] = None,
) -> Tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Optional[FrozenCUDAGraphForward],
    str,
]:
    forward, runner, status = maybe_build_cuda_graph_forward(
        base,
        device,
        expected_shape,
        dtype=dtype,
        mode=mode,
    )
    if runner is None:
        return forward, None, status
    return runner.replay_borrowed, runner, status


__all__ = [
    "CUDA_GRAPH_ENV",
    "FrozenCUDAGraphForward",
    "maybe_build_cuda_graph_forward",
    "maybe_build_cuda_graph_forward_borrowed",
]
