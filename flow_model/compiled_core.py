"""Fused recurrent core."""

import torch
from torch import nn
from .replay import RecurrentState


class FunctionalCore(nn.Module):
    def __init__(self, model, state):
        super().__init__()
        self.model = model
        self.state = state
        s = model.sps
        self.heads = nn.ModuleList(
            [
                s.accuracy.interval,
                s.accuracy.moments,
                s.interval,
                s.energy,
                model.head,
                model.extra,
            ]
        )

    def forward(self, inp, previous, base, *state):
        self.state.write(state, clone=True)
        safe = self.model.sps.accuracy.safe
        original = safe._base_forward
        safe._base_forward = lambda x: base
        try:
            pred, info = self.model.ttt_step(inp, previous)
        finally:
            safe._base_forward = original
        return (pred, info["lower"], info["upper"], *self.state.read())


@torch.no_grad()
def compile_core(model, inp, previous):
    from torch.fx.experimental.proxy_tensor import make_fx

    inp = inp.clone()
    previous = previous.clone()

    state = RecurrentState(model)
    saved = tuple(x.clone() for x in state.read())
    safe = model.sps.accuracy.safe
    updates = safe.interval_controller._updates
    try:
        base = safe.base(inp)
        args = (inp, previous, base, *saved)
        core = FunctionalCore(model, state).eval()
        graph = make_fx(core)(*args)
        compiled = torch.compile(graph, fullgraph=True)
        for _ in range(3):
            compiled(*args)
        torch.cuda.synchronize(inp.device)
        return compiled
    finally:
        state.write(saved)
        safe.interval_controller._updates = updates
        model.captured.clear()
