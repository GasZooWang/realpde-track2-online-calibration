"""Submission entry point."""
from pathlib import Path
import torch
from safe_submission import get_ttt_model as get_safe_model
from neural_wrapper import NeuralTTTModel

def get_ttt_model(submission_dir: str, device: str):
    safe = get_safe_model(submission_dir, device)
    payload = torch.load(Path(submission_dir)/"neural_heads.pt", map_location="cpu", weights_only=True)
    return NeuralTTTModel(safe, payload["interval_state_dict"], payload["moment_state_dict"],
                          rich_interval=payload["rich_interval"])
