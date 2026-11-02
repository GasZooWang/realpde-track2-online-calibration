"""Packaged PTX for the unified head."""

import hashlib
import json
from pathlib import Path
import sys

import torch

from .ptx_runtime import PortableCore


class PackagedAlgorithm:
    backend = "packaged_ptx"

    def __init__(self, path, reference, device):
        self.portable = PortableCore(path, device)
        for name in (
            "mode", "density_steps", "density_blend", "density_point_strength_u",
            "density_point_strength_v", "density_halves", "density_rates",
        ):
            setattr(self, name, getattr(reference, name))

    def __call__(self, *inputs):
        values = self.portable(*inputs)
        return values[0], list(values[1:7]), values[7]


def load_algorithm_core(submission_dir, reference, device):
    if submission_dir is None:
        raise ValueError("algorithm PTX requires the submission directory")
    if not sys.platform.startswith("linux") or torch.__version__.split("+")[0] != "2.2.2":
        raise RuntimeError("algorithm PTX requires Linux and PyTorch 2.2.2")
    if torch.cuda.get_device_capability(device) not in ((8, 0), (8, 6), (8, 7), (8, 9)):
        raise RuntimeError("algorithm PTX requires a supported Ampere or Ada GPU")
    root = Path(submission_dir)
    path = root / "flow_model" / "algorithm_portable"
    manifest = json.loads((path / "manifest.json").read_text())
    for name, expected in manifest["source_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError("algorithm PTX source mismatch: " + name)
    for name, expected in manifest["payload_sha256"].items():
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != expected:
            raise ValueError("algorithm PTX payload mismatch: " + name)
    return PackagedAlgorithm(path, reference, device)
