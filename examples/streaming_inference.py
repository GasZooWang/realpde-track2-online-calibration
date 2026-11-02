"""Streaming inference example."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def predict_trajectory(output_dir, windows, device='cuda'):
    import solution
    model = solution.get_ttt_model(str(output_dir), device)
    model.reset_ttt_state()
    for index, (input_norm, previous_revealed_target) in enumerate(windows):
        if index == 0 and previous_revealed_target is not None:
            raise ValueError('The first window must not have a previous target')
        prediction, info = model.ttt_step(input_norm, previous_revealed_target)
        yield prediction, info['lower'], info['upper']
