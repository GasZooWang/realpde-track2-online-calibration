"""Stage runner."""
import json
import os
from pathlib import Path
import runpy
import sys
import time
import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

script = Path(sys.argv[1]).resolve()
sys.argv = sys.argv[1:]
sys.path.insert(0, str(script.parent))
started = time.monotonic()
status = "FAILED"
try:
    runpy.run_path(str(script), run_name="__main__")
    status = "COMPLETE"
finally:
    record = {"script": str(script), "arguments": sys.argv[1:], "status": status,
              "seconds": time.monotonic() - started}
    folder = os.environ.get("TRAINING_RESOURCE_DIR")
    if folder:
        destination = Path(folder)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / f"{script.stem}_{os.getpid()}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"stage_resources": record}), flush=True)
