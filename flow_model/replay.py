"""CUDA graph replay."""

import warnings
import torch
from .outputs import pack_outputs


SAFE_FIELDS = (
    "_previous_base_prediction",
    "_previous_ridge_prediction",
    "_previous_ridge_variance",
    "_previous_prediction",
    "_interval_scale_state",
    "_interval_horizon_state",
    "_interval_center_state",
    "_interval_center_initialized",
    "_prediction_mean_state",
    "_previous_interval_candidates",
    "_previous_anchor_variance_map",
    "_pixel_tke_log_ratio_state",
    "_supervised_tke_ratio_state",
    "_previous_supervised_tke_variance",
)


def get(root, path):
    for key in path:
        root = root[key] if isinstance(key, int) else getattr(root, key)
    return root


class RecurrentState:
    def __init__(self, model):
        self.model = model
        safe = ("sps", "accuracy", "safe")
        self.paths = [safe + (key,) for key in SAFE_FIELDS]
        for child, fields in (
            ("calibrator", ("_gram", "_cross", "_weights", "_ones", "_ridge_matrix")),
            (
                "interval_controller",
                ("_log_weights", "_multiplier_tensor", "_maximum_tensor"),
            ),
        ):
            self.paths.extend(safe + (child, key) for key in fields)
        for path in (
            ("sps", "accuracy", "interval_context"),
            ("sps", "accuracy", "moment_context"),
            ("sps", "context"),
        ):
            self.paths.extend(
                path + (key,)
                for key, value in vars(get(model, path)).items()
                if isinstance(value, torch.Tensor)
            )
        self.paths.extend(("history", key) for key in ("pm", "pv", "total"))
        for key in ("ema", "errors", "lags"):
            if len(getattr(model.history, key)) != 3:
                raise ValueError("History is not ready for mature replay")
            self.paths.extend(("history", key, i) for i in range(3))
        self.paths.append(("history", "count"))

    def read(self):
        result = []
        for path in self.paths:
            value = get(self.model, path)
            if path == ("history", "count") and isinstance(value, int):
                value = torch.tensor(float(value), device=self.model.device)
            if not isinstance(value, torch.Tensor):
                raise ValueError("Uninitialized state: " + str(path))
            result.append(value)
        return result

    def write(self, values, clone=False):
        for path, value in zip(self.paths, values):
            owner = get(self.model, path[:-1])
            value = value.clone() if clone else value
            if isinstance(path[-1], int):
                owner[path[-1]] = value
            else:
                setattr(owner, path[-1], value)


class ReplayModel:
    def __init__(self, eager):
        self.eager = eager
        self.device = eager.device
        self.sps = eager.sps
        self.graph = None
        self.state = None
        self.steps = 0
        self.bound = False
        self.disabled = self.device.type != "cuda"
        self.status = {"mode": "eager" if self.disabled else "pending"}
        self.core = None

    def _body(self):
        if self.core is None:
            self.state.write(self.buffers, clone=True)
            self.output = self.eager.ttt_step(self.input, self.previous)
            next_state = self.state.read()
        else:
            base = self.sps.accuracy.safe.base(self.input)
            values = self.core(self.input, self.previous, base, *self.buffers)
            self.output = (
                values[0],
                {"lower": values[1], "upper": values[2], "adapt_loss": None},
            )
            next_state = values[3:]
        for dst, src in zip(self.buffers, next_state):
            dst.copy_(src)

    def _configure(self, inp, previous):
        self.state = RecurrentState(self.eager)
        saved = [v.clone() for v in self.state.read()]
        updates = self.sps.accuracy.safe.interval_controller._updates
        self.buffers = [v.clone() for v in saved]
        self.spec = [(v.shape, v.dtype, v.device) for v in saved]
        self.input = inp.clone()
        self.previous = previous.clone()
        current = torch.cuda.current_stream(self.device)
        stream = torch.cuda.Stream(device=self.device)
        try:
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self._body()
            current.wait_stream(stream)
            current.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                self._body()
            current.wait_stream(stream)
            self.status = {
                "mode": "cuda_graph",
                "state_tensors": len(saved),
                "compiled": self.core is not None,
                "core_backend": getattr(
                    self.core,
                    "backend",
                    "inductor" if self.core is not None else "eager",
                ),
            }
        finally:
            current.wait_stream(stream)
            self.state.write(saved)
            self.sps.accuracy.safe.interval_controller._updates = updates
            self.eager.captured.clear()
            self.bound = False

    @torch.inference_mode()
    def reset_ttt_state(self):
        self.eager.reset_ttt_state()
        self.steps = 0
        self.bound = False

    @torch.inference_mode()
    def ttt_step(self, input_norm, prev_target_norm=None):
        inp = torch.as_tensor(input_norm, device=self.device)
        eligible = (
            not self.disabled
            and self.steps >= 3
            and prev_target_norm is not None
            and inp.shape == (1, 20, 32, 64, 3)
            and inp.dtype == torch.float32
        )
        if eligible:
            previous = torch.as_tensor(prev_target_norm, device=self.device)
            eligible = previous.shape == inp.shape and previous.dtype == torch.float32
        if eligible and self.graph is None:
            try:
                self._configure(inp, previous)
            except Exception as error:
                self.disabled = True
                self.graph = None
                self.status = {"mode": "eager", "reason": str(error)[:240]}
                warnings.warn("CUDA replay unavailable: " + str(error)[:240])
                eligible = False
        if eligible and not self.bound:
            try:
                current = self.state.read()
                eligible = [(v.shape, v.dtype, v.device) for v in current] == self.spec
            except ValueError:
                eligible = False
            if eligible:
                for dst, src in zip(self.buffers, current):
                    dst.copy_(src)
                self.bound = True
        if eligible:
            self.input.copy_(inp)
            self.previous.copy_(previous)
            self.graph.replay()
            self.state.write(self.buffers)
            self.sps.accuracy.safe.interval_controller._updates += 1
            pred, info = self.output
            result = pack_outputs(pred, info)
        else:
            self.bound = False
            result = pack_outputs(*self.eager.ttt_step(input_norm, prev_target_norm))
        self.steps += 1
        return result
