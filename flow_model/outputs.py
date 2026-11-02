"""Output buffers."""

import torch


def pack_outputs(prediction, info):
    packed = torch.stack((prediction, info["lower"], info["upper"]))
    result = dict(info)
    result["lower"] = packed[1]
    result["upper"] = packed[2]
    return packed[0], result
