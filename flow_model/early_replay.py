"""Startup graph replay."""

import warnings
import torch
from .replay import SAFE_FIELDS, get
from .outputs import pack_outputs


def clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, list):
        return [clone_tree(v) for v in value]
    return value


def tree_structure(value):
    if isinstance(value, list):
        return [tree_structure(v) for v in value]
    return value


def copy_tree(dst, src):
    if isinstance(dst, torch.Tensor):
        dst.copy_(src)
    elif isinstance(dst, list):
        for a, b in zip(dst, src):
            copy_tree(a, b)


def signature(value):
    if isinstance(value, torch.Tensor):
        return (value.shape, value.dtype, value.device)
    if isinstance(value, list):
        return tuple(signature(v) for v in value)
    return value


class StartupState:
    def __init__(self, model):
        self.model = model
        safe = ("sps", "accuracy", "safe")
        self.paths = [safe + (key,) for key in SAFE_FIELDS]
        for child, fields in (
            ("calibrator", ("_gram", "_cross", "_weights", "_ones", "_ridge_matrix")),
            (
                "interval_controller",
                ("_log_weights", "_multiplier_tensor", "_maximum_tensor", "_updates"),
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
                if value is None or isinstance(value, torch.Tensor)
            )
        self.paths.extend(
            ("history", key)
            for key in ("pm", "pv", "total", "ema", "errors", "lags", "count")
        )

    def read(self):
        return [get(self.model, path) for path in self.paths]

    def write(self, values, clone=False):
        for path, value in zip(self.paths, values):
            setattr(
                get(self.model, path[:-1]),
                path[-1],
                clone_tree(value) if clone else tree_structure(value),
            )


class StartupGraph:
    def __init__(self, model, state, inp, previous):
        self.model = model
        self.state = state
        saved = [clone_tree(v) for v in state.read()]
        self.inputs = [clone_tree(v) for v in saved]
        self.spec = signature(self.inputs)
        self.inp = inp.clone()
        self.previous = previous.clone() if previous is not None else None
        current = torch.cuda.current_stream(inp.device)
        stream = torch.cuda.Stream(device=inp.device)
        try:
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self.body()
            current.wait_stream(stream)
            current.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                self.body()
            current.wait_stream(stream)
        finally:
            current.wait_stream(stream)
            state.write(saved)
            model.captured.clear()

    def body(self):
        self.state.write(self.inputs, clone=True)
        self.output = self.model.ttt_step(self.inp, self.previous)
        self.outputs = self.state.read()

    def run(self, inp, previous):
        values = self.state.read()
        if signature(values) != self.spec:
            return None
        copy_tree(self.inputs, values)
        self.inp.copy_(inp)
        if previous is not None:
            self.previous.copy_(previous)
        self.graph.replay()
        self.state.write(self.outputs)
        pred, info = self.output
        return pack_outputs(pred, info)


class EarlyReplay:
    def __init__(self, replay):
        self.replay = replay
        self.eager = replay.eager
        self.sps = replay.sps
        self.device = replay.device
        self.kernels = []
        if self.device.type == "cuda":
            try:
                self.eager.reset_ttt_state()
                state = StartupState(self.eager)
                inp = torch.zeros((1, 20, 32, 64, 3), device=self.device)
                previous = torch.zeros_like(inp)
                for i in range(3):
                    prior = None if i == 0 else previous
                    self.kernels.append(StartupGraph(self.eager, state, inp, prior))
                    self.eager.ttt_step(inp, prior)
            except Exception as error:
                self.kernels = []
                warnings.warn("Startup replay unavailable: " + str(error)[:240])
            finally:
                self.reset_ttt_state()

    @property
    def status(self):
        return dict(self.replay.status, startup_graphs=len(self.kernels))

    def reset_ttt_state(self):
        self.replay.reset_ttt_state()

    @torch.inference_mode()
    def ttt_step(self, input_norm, prev_target_norm=None):
        i = self.replay.steps
        if self.kernels and i < 3 and ((i == 0) == (prev_target_norm is None)):
            inp = torch.as_tensor(input_norm, device=self.device)
            previous = (
                None
                if prev_target_norm is None
                else torch.as_tensor(prev_target_norm, device=self.device)
            )
            if (
                inp.shape == (1, 20, 32, 64, 3)
                and inp.dtype == torch.float32
                and (
                    previous is None
                    or (previous.shape == inp.shape and previous.dtype == inp.dtype)
                )
            ):
                result = self.kernels[i].run(inp, previous)
                if result is not None:
                    self.replay.steps += 1
                    self.replay.bound = False
                    return result
        return self.replay.ttt_step(input_norm, prev_target_norm)
