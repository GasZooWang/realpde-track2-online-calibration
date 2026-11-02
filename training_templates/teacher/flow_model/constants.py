"""Device constants."""
from functools import lru_cache
import torch


@lru_cache(maxsize=32)
def _constant(device, dtype, values):
    return torch.tensor(values, device=device, dtype=dtype)


def constant_like(tensor, values):
    return _constant(tensor.device, tensor.dtype, tuple(values))
