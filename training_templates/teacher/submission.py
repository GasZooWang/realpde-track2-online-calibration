"""Teacher submission entry point."""

from flow_model.accuracy import AccuracyTTTModel
from flow_model.calibration import CausalRidgeTTTModel
from flow_model.cno import CNO3d
from flow_model.model import Model
from flow_model.runtime import InferenceBackbone
from flow_model.replay import ReplayModel
from pathlib import Path
from flow_model.sps import SPSTTTModel
import torch


def get_ttt_model(submission_dir: str, device: str):
    root = Path(submission_dir)
    weights = torch.load(root / "adapters.pt", map_location="cpu", weights_only=True)
    base = CNO3d(in_dim=3, out_dim=3, out_dim_mult=1, in_size=32, N_layers=3).to(device)
    base.load_state_dict(
        torch.load(root / "model.pth", map_location="cpu", weights_only=True),
        strict=True,
    )
    base.eval()
    base = InferenceBackbone(base, device)
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
    return ReplayModel(Model(
        sps,
        forecast["state_dict"],
        energy["state_dict"],
        forecast["width"],
        energy["strength"],
    ))
