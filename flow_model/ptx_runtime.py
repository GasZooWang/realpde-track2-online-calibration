"""Packaged PTX runtime."""

import ctypes as C
import json
from pathlib import Path
import types
import hashlib
import sys
import warnings
import torch


class Driver:
    def __init__(self):
        self.lib = C.CDLL("libcuda.so.1")
        self.lib.cuModuleLoadData.argtypes = [C.POINTER(C.c_void_p), C.c_void_p]
        self.lib.cuModuleGetFunction.argtypes = [
            C.POINTER(C.c_void_p),
            C.c_void_p,
            C.c_char_p,
        ]
        self.lib.cuFuncSetAttribute.argtypes = [C.c_void_p, C.c_int, C.c_int]
        self.lib.cuLaunchKernel.argtypes = [
            C.c_void_p,
            *([C.c_uint] * 7),
            C.c_void_p,
            C.POINTER(C.c_void_p),
            C.c_void_p,
        ]
        self.lib.cuModuleUnload.argtypes = [C.c_void_p]

    @staticmethod
    def check(code, operation):
        if code:
            raise RuntimeError(f"CUDA driver {operation} failed: {code}")


class Kernel:
    def __init__(self, driver, path, meta):
        self.driver = driver
        self.meta = meta
        self.module = C.c_void_p()
        self.function = C.c_void_p()
        ptx = C.create_string_buffer(Path(path).read_bytes() + b"\0")
        driver.check(
            driver.lib.cuModuleLoadData(C.byref(self.module), C.cast(ptx, C.c_void_p)),
            "load PTX",
        )
        driver.check(
            driver.lib.cuModuleGetFunction(
                C.byref(self.function), self.module, meta["function"].encode()
            ),
            "get function",
        )
        if meta["shared"] > 49152:
            driver.check(
                driver.lib.cuFuncSetAttribute(self.function, 8, meta["shared"]),
                "set shared memory",
            )

    def run(self, *args, grid, stream):
        dims = grid(self.meta["grid_meta"]) if callable(grid) else grid
        scalar = {
            "i32": C.c_int32,
            "i64": C.c_int64,
            "u32": C.c_uint32,
            "u64": C.c_uint64,
            "fp32": C.c_float,
            "fp64": C.c_double,
            "i1": C.c_bool,
        }
        values = []
        for index, kind in self.meta["parameters"]:
            value = args[index]
            values.append(
                C.c_void_p(value.data_ptr())
                if kind.startswith("*")
                else scalar[kind](value)
            )
        pointers = (C.c_void_p * len(values))(
            *[C.cast(C.byref(v), C.c_void_p) for v in values]
        )
        self.driver.check(
            self.driver.lib.cuLaunchKernel(
                self.function,
                *map(int, dims),
                self.meta["warps"] * 32,
                1,
                1,
                self.meta["shared"],
                C.c_void_p(stream),
                pointers,
                None,
            ),
            "launch",
        )


class PortableCore:
    backend = "packaged_ptx"

    def __init__(self, path, device):
        path = Path(path)
        self.device = torch.device(device)
        if self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.driver = Driver()
        mode = (
            json.loads((path / "mode.json").read_text())
            if (path / "mode.json").exists()
            else {}
        )
        self.whole_model = mode.get("whole_model", False)
        manifest = json.loads((path / "kernels.json").read_text())
        with torch.cuda.device(self.device):
            self.kernels = {
                name: Kernel(self.driver, path / "ptx" / (name + ".ptx"), meta)
                for name, meta in manifest.items()
            }
            self.weights = torch.load(
                path / "weights.pt", map_location=self.device, weights_only=True
            )
        self.module = types.ModuleType("packaged_flow_core")
        self.module.kernels = self.kernels
        self.module.DEVICE_INDEX = self.device.index
        exec(
            compile((path / "core.py").read_text(), str(path / "core.py"), "exec"),
            self.module.__dict__,
        )

    def __call__(self, *args):
        with torch.cuda.device(self.device):
            return self.module.call(self.weights + list(args))


def load_portable_core(submission_dir, device):
    if submission_dir is None or torch.device(device).type != "cuda":
        return None
    if (
        not sys.platform.startswith("linux")
        or torch.__version__.split("+")[0] != "2.2.2"
    ):
        return None
    if torch.cuda.get_device_capability(device) not in ((8, 0), (8, 6), (8, 7), (8, 9)):
        return None
    root = Path(submission_dir)
    path = root / "flow_model" / "portable"
    if not (path / "manifest.json").is_file():
        return None
    try:
        manifest = json.loads((path / "manifest.json").read_text())
        for name, expected in manifest["source_sha256"].items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
                raise ValueError("Packaged kernel source changed: " + name)
        return PortableCore(path, device)
    except Exception as error:
        warnings.warn("Packaged correction kernels unavailable: " + str(error)[:240])
        return None
