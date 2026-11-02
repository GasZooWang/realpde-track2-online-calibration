"""Training utilities."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PACKAGE = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def run(root, relative, arguments=()):
    root = Path(root)
    if root.resolve() == PACKAGE and os.environ.get("TRAINING_DISPATCH_LOG_DIR"):
        log_root = Path(os.environ["TRAINING_DISPATCH_LOG_DIR"])
    else:
        log_root = root / "run_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    name = Path(relative).stem
    suffix = "_" + "_".join(str(x).replace("/", "_")[-60:] for x in arguments)
    path = log_root / (name + suffix[:180] + ".log")
    command = [sys.executable, "-u", str(PACKAGE / "training/run_stage.py"),
               str(root / relative), *map(str, arguments)]
    python_path = str(PACKAGE) + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    environment = dict(os.environ, PYTHONPATH=python_path,
                       TORCHINDUCTOR_COMPILE_THREADS="1", OMP_NUM_THREADS="4",
                       MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    started = time.monotonic()
    print(json.dumps({"stage": relative, "arguments": list(map(str, arguments))}), flush=True)
    with path.open("w") as log:
        process = subprocess.Popen(command, cwd=root, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        code = process.wait()
    if code:
        raise RuntimeError(f"Training stage failed ({code}): {relative}; log: {path}")
    with (log_root / "stages.jsonl").open("a") as log:
        log.write(json.dumps({"command": command, "seconds": time.monotonic() - started,
                              "log": path.name, "sha256": digest(path)}) + "\n")


def clean_checkpoint(source, destination):
    import torch
    state = torch.load(source, map_location="cpu", weights_only=False)
    keep = ("width", "channels", "mode", "strength", "epoch", "component_energy", "energy_strength")
    result = {key: state[key] for key in keep if key in state}
    result["state_dict"] = {key: value.detach().cpu().contiguous()
                            for key, value in state["state_dict"].items()}
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, destination)
    return result
