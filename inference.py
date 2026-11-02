"""Inference entry point."""

import multiprocessing as _multiprocessing
import os as _os

if _multiprocessing.current_process().daemon:
    _os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    try:
        import torch._inductor.config as _inductor_config
        _inductor_config.compile_threads = 1
    except ImportError:
        pass

from flow_model.accuracy import AccuracyTTTModel
from flow_model.calibration import CausalRidgeTTTModel
from flow_model.model import Model
from flow_model.runtime import half_affine_
from flow_model.student import StudentBackbone
from flow_model.optimized import optimize
from pathlib import Path
from flow_model.sps import SPSTTTModel
import torch


def _build_predictor(submission_dir: str, device: str):
    root = Path(submission_dir)
    weights = torch.load(root / "adapters.pt", map_location="cpu", weights_only=True)
    base = StudentBackbone(
        torch.load(root / "model.pth", map_location="cpu", weights_only=True), device
    )
    safe = CausalRidgeTTTModel(
        base,
        device,
        weights["interval_coefficients"],
        weights["width_scale"],
        weights["tke_coefficients"],
        weights["history_coefficients"],
    )
    head = weights["accuracy"]
    accuracy = AccuracyTTTModel(
        safe,
        head["interval_state_dict"],
        head["moment_state_dict"],
        head["width"],
        head["response_range"],
        head["feature_count"],
    )
    head = weights["multiscale"]
    sps = SPSTTTModel(
        accuracy,
        head["interval_state"],
        head["width"],
        head["energy_state"],
        head["energy_width"],
        head["energy_strength"],
    )
    forecast = weights["forecast"]
    energy = weights["energy"]
    model = Model(
        sps,
        forecast["state_dict"],
        energy["state_dict"],
        forecast["width"],
        energy["strength"],
    )
    if torch.device(device).type == "cuda":
        for module in (
            accuracy.interval,
            accuracy.moments,
            sps.interval,
            sps.energy,
            model.head,
            model.extra,
        ):
            half_affine_(module)
    return model


from flow_model.algorithm import AlgorithmModel


class _SharedDeviceInputModel:

    def __init__(self, model, device):
        self._model = model
        self._device = torch.device(device)

    def __getattr__(self, name):
        return getattr(self._model, name)

    def reset_ttt_state(self):
        return self._model.reset_ttt_state()

    def ttt_step(self, input_norm, prev_target_norm=None):
        current = torch.as_tensor(input_norm, device=self._device)
        previous = (
            None
            if prev_target_norm is None
            else torch.as_tensor(prev_target_norm, device=self._device)
        )
        return self._model.ttt_step(current, previous)


def get_ttt_model(submission_dir: str, device: str):
    if torch.device(device).type == "cuda":
        torch.backends.cudnn.benchmark = True
    base = optimize(_build_predictor(submission_dir, device), submission_dir)
    checkpoint = torch.load(
        Path(submission_dir) / "algorithm.pt", map_location="cpu", weights_only=True
    )
    model = AlgorithmModel(base, checkpoint, device, submission_dir=submission_dir)
    return _SharedDeviceInputModel(model, device)
