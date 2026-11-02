"""Runtime optimization."""

import time
import warnings
import torch
from .compiled_core import compile_core
from .early_replay import EarlyReplay
from .replay import ReplayModel
from .ptx_runtime import load_portable_core


@torch.inference_mode()
def optimize(model, submission_dir=None):
    replay = ReplayModel(model)
    started = time.perf_counter()
    if model.device.type == "cuda":
        try:
            inp = torch.zeros((1, 20, 32, 64, 3), device=model.device)
            for i in range(4):
                model.ttt_step(inp, inp if i else None)
            replay.core = load_portable_core(submission_dir, model.device)
            if replay.core is None:
                replay.core = compile_core(model, inp, inp)
            replay._configure(inp, inp)
        except Exception as error:
            replay.core = None
            replay.graph = None
            replay.state = None
            replay.status = {"mode": "pending", "compile_fallback": str(error)[:240]}
            warnings.warn("Correction fusion unavailable: " + str(error)[:240])
        finally:
            replay.reset_ttt_state()
    result = EarlyReplay(replay)
    result.initialization_seconds = time.perf_counter() - started
    return result
