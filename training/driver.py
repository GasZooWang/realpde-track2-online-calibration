"""Training pipeline."""
import contextlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import torch
from training.common import PACKAGE, digest, run, write_json
from training.data import prepare


class Tee:
    def __init__(self, stream, file):
        self.stream, self.file = stream, file

    def write(self, text):
        self.stream.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()


def collect_logs(work, output):
    destination = Path(output) / "training_logs"
    for family in ("calibration", "distillation"):
        root = work / family
        folder = root / "run_logs"
        if folder.is_dir():
            for path in folder.rglob("*"):
                if path.is_file() and path.suffix in (".log", ".jsonl"):
                    target = destination / family / path.relative_to(root)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)


def train(data_dir, output_dir, device):
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA GPU")
    if torch.__version__.split("+")[0] != "2.2.2":
        raise RuntimeError('Training requires PyTorch 2.2.2')
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('output_dir must be empty')
    if digest(PACKAGE / "official_assets/sim_real_cno.pth") != "82e842928a25dbf5a74c4e336bdd28e89bcf40e68bb8cdd213547f1246af4f61":
        raise ValueError("Official CNO checkpoint checksum mismatch")
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    os.environ["TRAINING_RESOURCE_DIR"] = str(output / "training_logs/resources")
    os.environ["TRAINING_DISPATCH_LOG_DIR"] = str(output / "training_logs/dispatch")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    work = output / "_training"
    work.mkdir()
    for family in ("calibration", "distillation"):
        shutil.copytree(PACKAGE / "training_templates" / family, work / family)
    shutil.copytree(PACKAGE / "flow_model", output / "flow_model")
    started = time.monotonic()
    report = {"status": "RUNNING", "submission_id": 940980,
              "device": str(device), "python": platform.python_version(), "torch": str(torch.__version__),
              "torch_cuda": torch.version.cuda,
              "official_checkpoint_sha256": digest(PACKAGE / "official_assets/sim_real_cno.pth"),
              "source_sha256": {str(p.relative_to(PACKAGE)): digest(p) for p in PACKAGE.rglob("*.py") if "__pycache__" not in p.parts}}
    write_json(output / "TRAINING_STATUS.json", report)
    log_path = output / "train.log"
    with log_path.open("w") as log, contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
        try:
            prepare(data_dir, work)
            run(PACKAGE, "training/data.py", [data_dir, work, device])
            for directory in (work / "calibration").glob("ridge_*"):
                shutil.copy2(PACKAGE / "official_assets/sim_real_cno.pth", directory / "model.pth")
            run(PACKAGE, "training/calibration.py", [work / "calibration", output, device])
            collect_logs(work, output)
            shutil.rmtree(work / "calibration/dev/cache")
            run(PACKAGE, "training/distillation.py", [work / "distillation", output, device])
            run(PACKAGE, "training/export_runtime.py", ["--root", output, "--kind", "algorithm", "--device", device])
            collect_logs(work, output)
            for name in ("model.pth", "adapters.pt", "algorithm.pt"):
                checkpoint = torch.load(output / name, map_location="cpu", weights_only=True)
                def finite(value):
                    if isinstance(value, torch.Tensor):
                        return bool(torch.isfinite(value).all())
                    if isinstance(value, dict):
                        return all(finite(x) for x in value.values())
                    if isinstance(value, (list, tuple)):
                        return all(finite(x) for x in value)
                    return True
                if not finite(checkpoint):
                    raise RuntimeError(f"Nonfinite checkpoint: {name}")
            shutil.rmtree(work)
            report.update(status="COMPLETE", wall_seconds=time.monotonic()-started,
                          generated_weights={name: digest(output / name) for name in ("model.pth", "adapters.pt", "algorithm.pt")})
            write_json(output / "TRAINING_COMPLETE.json", report)
            write_json(output / "TRAINING_STATUS.json", report)
            print(json.dumps({"status": "COMPLETE", "seconds": report["wall_seconds"]}), flush=True)
        except BaseException as error:
            collect_logs(work, output)
            report.update(status="FAILED", wall_seconds=time.monotonic()-started,
                          error=f"{type(error).__name__}: {error}")
            write_json(output / "TRAINING_STATUS.json", report)
            traceback.print_exc()
            raise
