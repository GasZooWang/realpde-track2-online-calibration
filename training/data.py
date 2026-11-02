"""Training data."""
from pathlib import Path
import json
import h5py
import numpy as np
import torch
from training.common import write_json


def stats(data_dir):
    mean_in, mean_target, std_in, std_target = torch.load(
        Path(data_dir) / "mean_std_real.pt", map_location="cpu", weights_only=False)
    values = [torch.as_tensor(x).float().reshape(1, 1, 1, 1, 3)
              for x in (mean_in, std_in, mean_target, std_target)]
    values[1] = torch.where(values[1] == 0, torch.ones_like(values[1]), values[1])
    values[3] = torch.where(values[3] == 0, torch.ones_like(values[3]), values[3])
    return values


def read_velocity(path):
    with h5py.File(path, "r") as handle:
        group = handle if "u" in handle else handle["measured_data"]
        u = np.asarray(group["u"][:, ::2, ::2], dtype=np.float32)
        v = np.asarray(group["v"][:, ::2, ::2], dtype=np.float32)
    if u.shape != v.shape or u.ndim != 3 or u.shape[1:] != (32, 64):
        raise ValueError(f"Unexpected official grid in {path}: {u.shape}, {v.shape}")
    if not np.isfinite(u).all() or not np.isfinite(v).all():
        raise ValueError(f"Nonfinite input in {path}")
    return np.stack((u, v), -1)


def prepare(data_dir, work):
    data_dir, work = Path(data_dir).resolve(), Path(work)
    folder = data_dir / "train_real"
    files = sorted(p for p in folder.glob("*.h5") if p.name != "7575_0.h5")
    expected = json.loads((Path(__file__).parent / "recipe.json").read_text())["all81"]
    if [p.name for p in files] != sorted(expected):
        raise ValueError("train_real must contain the 81 released trajectories, excluding 7575_0.h5")
    from flow_model.calibration import INPUT_MEAN, INPUT_STD, TARGET_MEAN, TARGET_STD
    normalizer = stats(data_dir)
    for actual, reference in zip(normalizer, (INPUT_MEAN, INPUT_STD, TARGET_MEAN, TARGET_STD)):
        if not torch.allclose(actual.reshape(3), torch.tensor(reference), atol=1e-7, rtol=1e-6):
            raise ValueError("mean_std_real.pt does not match the expected normalizer")
    destination = work / "distillation/dev/public_data"
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for path in files:
        velocity = read_velocity(path)
        np.savez(destination / (path.stem + ".npz"), velocity=velocity)
        records.append({"name": path.name, "frames": len(velocity),
                        "windows": len(range(0, len(velocity) - 39, 20))})
    write_json(work / "data_manifest.json", {"source": "released train_real",
                "excluded": ["7575_0.h5"],
                "spatial_subsample": 2, "pressure": "zero", "records": records})
    return files, normalizer


@torch.inference_mode()
def cache_cno(data_dir, work, device):
    from flow_model.cno import CNO3d
    from training.common import PACKAGE
    work = Path(work)
    recipe = json.loads((Path(__file__).parent / "recipe.json").read_text())
    model = CNO3d(in_dim=3, out_dim=3, out_dim_mult=1, in_size=32, N_layers=3).to(device).eval()
    checkpoint = torch.load(PACKAGE / "official_assets/sim_real_cno.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.requires_grad_(False)
    im, ist, tm, tst = [x.to(device) for x in stats(data_dir)]
    cache = work / "calibration/dev/cache"
    shards = cache / "train_real_c45_shards"
    shards.mkdir(parents=True, exist_ok=True)
    d30 = []
    all_records = {}
    for name in sorted(set(recipe["d30"]) | set(recipe["c45"])):
        velocity = np.load(work / "distillation/dev/public_data" / (Path(name).stem + ".npz"))["velocity"]
        raw = torch.from_numpy(velocity).to(device)
        raw = torch.cat((raw, torch.zeros_like(raw[..., :1])), -1)
        starts = list(range(0, len(raw) - 39, 20))
        inputs = torch.stack([(raw[s:s+20] - im[0]) / ist[0] for s in starts])
        targets = torch.stack([(raw[s+20:s+40] - tm[0]) / tst[0] for s in starts])
        predictions = torch.cat([model(x).float().cpu() for x in inputs.split(8)])
        payload = dict(format_version=3, input_norm=inputs.cpu(), target_norm=targets.cpu(),
                       base_prediction_norm=predictions, simulation_id=[name]*len(starts),
                       trajectory_first=torch.arange(len(starts)) == 0,
                       time_id=torch.tensor(starts, dtype=torch.int64),
                       mean_input=im.cpu(), std_input=ist.cpu(), mean_target=tm.cpu(), std_target=tst.cpu(),
                       source='RealPDE competition data')
        if name in recipe["d30"]:
            d30.append(payload)
        else:
            torch.save(payload, shards / (Path(name).stem + ".pt"))
        all_records[name] = len(starts)
        print("CNO cache", name, len(starts), flush=True)
    fields = ("input_norm", "target_norm", "base_prediction_norm", "trajectory_first", "time_id")
    combined = {key: torch.cat([d[key] for d in d30]) for key in fields}
    combined.update({key: d30[0][key] for key in ("mean_input", "std_input", "mean_target", "std_target", "source")})
    combined["simulation_id"] = sum([d["simulation_id"] for d in d30], [])
    combined["format_version"] = 2
    torch.save({key: value for key, value in combined.items() if key != "input_norm"}, cache / "train_real_balanced30.pt")
    torch.save({key: value for key, value in combined.items() if key in ("input_norm", "simulation_id", "time_id", "mean_input", "std_input", "source")} | {"format_version": 1}, cache / "train_real_balanced30_inputs.pt")
    write_json(cache / "cno_manifest.json", {"precision": "float32", "files": all_records})


if __name__ == "__main__":
    import sys
    cache_cno(sys.argv[1], sys.argv[2], sys.argv[3])
