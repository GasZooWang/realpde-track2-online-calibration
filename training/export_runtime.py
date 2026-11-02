"""Kernel export."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import torch
from torch import nn
from torch.fx.experimental.proxy_tensor import make_fx
from training.common import PACKAGE, digest, write_json


@torch.inference_mode()
def trace(root, kind, device):
    root = Path(root)
    shape = (1, 20, 32, 64, 3)
    if kind == "base":
        from inference import _build_predictor
        from flow_model.replay import RecurrentState
        from flow_model.compiled_core import FunctionalCore
        model = _build_predictor(str(root), device)
        inp = torch.zeros(shape, device=device)
        previous = torch.zeros_like(inp)
        for step in range(4):
            model.ttt_step(inp, previous if step else None)
        state = RecurrentState(model)
        values = [x.clone() for x in state.read()]
        base = model.sps.accuracy.safe.base(inp)
        inputs = [inp, previous, base, *values]
        core = FunctionalCore(model, state).eval()
    else:
        from flow_model.algorithm import Core

        class FlatCore(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.core = model

            def forward(self, *values):
                prediction, state, features = self.core(*values)
                return prediction, *state, features

        checkpoint = torch.load(root / "algorithm.pt", map_location="cpu", weights_only=True)
        core = FlatCore(Core(checkpoint, device).eval()).eval()
        inputs = [torch.zeros(shape, device=device) for _ in range(5)]
        inputs += [torch.ones((), dtype=torch.bool, device=device),
                   torch.zeros((), dtype=torch.bool, device=device)]
        inputs += [torch.zeros((1, 2, 32, 64), device=device) for _ in range(2)]
        inputs += [torch.zeros((3, 1, 2, 32, 64), device=device) for _ in range(4)]
    graph = make_fx(core)(*inputs)
    tensors = list(graph.parameters()) + list(graph.buffers())
    schema = {
        "program_sha256": hashlib.sha256(str(graph.graph).encode()).hexdigest(),
        "tensor_layout": [{"shape": list(x.shape), "stride": list(x.stride()), "dtype": str(x.dtype)} for x in tensors],
        "input_layout": [{"shape": list(x.shape), "stride": list(x.stride()), "dtype": str(x.dtype)} for x in inputs],
        "torch": torch.__version__.split("+")[0], "kind": kind,
    }
    return graph, tensors, inputs, schema


@torch.inference_mode()
def emit(root, kind, device, schema_path=None):
    if torch.__version__.split("+")[0] != "2.2.2":
        raise RuntimeError("Portable kernels require PyTorch 2.2.2")
    root = Path(root)
    graph, tensors, inputs, schema = trace(root, kind, device)
    if schema_path:
        write_json(schema_path, schema)
        return schema
    expected = json.loads((PACKAGE / "training" / (kind + "_kernel_schema.json")).read_text())
    if schema != expected:
        raise RuntimeError(f"{kind} trace does not match the kernel schema")
    name = "portable" if kind == "base" else "algorithm_portable"
    source = PACKAGE / "flow_model" / name
    destination = root / "flow_model" / name
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        if path.is_file() and path.suffix in (".json", ".py", ".ptx"):
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.resolve() != target.resolve():
                shutil.copy2(path, target)
    torch.save([x.detach().cpu() for x in tensors], destination / "weights.pt")
    manifest = json.loads((destination / "manifest.json").read_text())
    for path in manifest["source_sha256"]:
        manifest["source_sha256"][path] = digest(root / path)
    for group in ("files", "payload_sha256"):
        if group in manifest:
            for path in manifest[group]:
                manifest[group][path] = digest(destination / path)
    write_json(destination / "manifest.json", manifest)
    from flow_model.ptx_runtime import PortableCore
    runtime = PortableCore(destination, device)
    reference, actual = graph(*inputs), runtime(*inputs)
    torch.cuda.synchronize()
    if len(reference) != len(actual):
        raise RuntimeError("Portable output arity differs")
    differences = [float((a.float()-b.float()).abs().max()) for a, b in zip(reference, actual)]
    if not all(torch.isfinite(x).all() for x in actual):
        raise RuntimeError("Nonfinite portable-core output")
    if max(differences) > .02:
        raise RuntimeError(f"Unexpected portable/eager discrepancy: {differences}")
    write_json(root / "training_logs" / (kind + "_kernel_binding.json"), {
        "status": "PASS", "fresh_checkpoint_sha256": digest(root / ("adapters.pt" if kind == "base" else "algorithm.pt")),
        "weights_sha256": digest(destination / "weights.pt"), "tensor_count": len(tensors),
        "program_sha256": schema["program_sha256"], "synthetic_max_abs_differences": differences})
    return differences


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--kind", choices=("base", "algorithm"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--write-schema", type=Path)
    args = parser.parse_args()
    print(emit(args.root, args.kind, args.device, args.write_schema), flush=True)
