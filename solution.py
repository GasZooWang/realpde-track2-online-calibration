"""Competition entry points."""


def train(data_dir, output_dir, device):
    from training.driver import train as run
    return run(data_dir, output_dir, device)


def get_ttt_model(output_dir, device):
    from pathlib import Path
    required = ("model.pth", "adapters.pt", "algorithm.pt", "TRAINING_COMPLETE.json")
    missing = [name for name in required if not (Path(output_dir) / name).is_file()]
    if missing:
        raise FileNotFoundError("Incomplete training output: " + ", ".join(missing))
    from inference import get_ttt_model as load
    return load(str(output_dir), device)
