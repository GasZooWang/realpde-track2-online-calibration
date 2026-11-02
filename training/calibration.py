"""Calibration and head training."""
from pathlib import Path
import json
import shutil
import sys
import zipfile
import numpy as np
import torch
from training.common import PACKAGE, run, write_json


def fit_analytic(root):
    root = Path(root)
    sys.path[:0] = [str(root / "dev"), str(root / "starting_kit")]
    import refine_learned_interval as interval
    import refine_history_forecast as temporal
    import analytic_fit as fixed
    from cached_data import load_data
    from prepare_c45 import as_data
    data = load_data()
    names = list(data.trajectory_ids)
    torch.set_num_threads(4)
    interval_stats = interval.collect_stats(data, 100)
    learned, loto = fixed.fit_interval_and_loto(interval_stats, names)
    temporal_stats = temporal.collect_stats(data, 100)
    records = [{k: temporal_stats[k][i:i+1] for k in ("xtx", "xty", "y2", "count")}
               for i in range(len(names))]
    history = fixed.fit_temporal_from_records(records, names)
    del temporal_stats, records
    width_records = []
    tke_records = []
    for i, name in enumerate(names):
        ids = np.flatnonzero(np.asarray(data.simulation_ids) == name)
        payload = dict(input_norm=data.inputs[ids], target_norm=data.targets[ids],
                       base_prediction_norm=data.base[ids],
                       trajectory_first=torch.from_numpy(data.first[ids]),
                       simulation_id=[name]*len(ids),
                       **{k: getattr(data, k) for k in ("mean_input", "std_input", "mean_target", "std_target")})
        shard = as_data(payload)
        width_records.append(fixed.collect_width_reward(shard, loto[i]))
        tke_records.append(fixed.collect_tke_sufficient_stats(shard, history["effective_beta_fine4x8"].numpy()))
        print("analytic width/TKE", i + 1, len(names), name, flush=True)
    _, width = fixed.fit_width(np.stack([r["reward"] for r in width_records]),
                              np.stack([r["covered"] for r in width_records]), names)
    tke = fixed.fit_tke(tke_records, names)
    prototype = root / "ridge_calibrated"
    for name, state in (("learned_interval.pt", learned), ("learned_width.pt", width),
                        ("supervised_tke.pt", tke), ("temporal_history.pt", history)):
        torch.save(state, prototype / name)
    write_json(root / "dev/reports/analytic_fit.json", {
        "status": "COMPLETE", "trajectories": names, "interval_partition": [4, 8],
        "interval_ridge": 1e-5, "history_family": "same_full20", "history_partition": [2, 4],
        "history_ridge": 1e-4, "width_grid": fixed.WIDTH_GRID.tolist(), "tke_ridge": .005})


def neural_weights(root):
    cache = root / "dev/cache"
    interval = torch.load(cache / "neural_interval_rich/head.pt", map_location="cpu", weights_only=False)
    moment = torch.load(cache / "neural_moments_head/head.pt", map_location="cpu", weights_only=False)
    destination = root / "ridge_neural"
    torch.save(dict(interval_state_dict=interval["state_dict"], moment_state_dict=moment["state_dict"]),
               destination / "neural_heads.pt")
    with zipfile.ZipFile(root / "ridge_neural_heads.zip", "w") as archive:
        archive.write(destination / "neural_heads.pt", "neural_heads.pt")


def compact_weights(root):
    cache = root / "dev/cache"
    interval = torch.load(cache / "compact_interval/best32_fit50_f28.pt", map_location="cpu", weights_only=False)
    moment = torch.load(cache / "wide_moments/best.pt", map_location="cpu", weights_only=False)
    destination = root / "ridge_compact"
    destination.mkdir(exist_ok=True)
    for name in ("accuracy_wrapper.py", "compact_interval.py", "wide_moments.py"):
        shutil.copy2(root / "dev" / name, destination / name)
    torch.save(dict(interval_state_dict=interval["state_dict"], moment_state_dict=moment["state_dict"],
                    width=32, response_range=1.5, feature_count=28), destination / "neural_heads.pt")
    with zipfile.ZipFile(root / "ridge_compact_heads.zip", "w") as archive:
        archive.write(destination / "neural_heads.pt", "neural_heads.pt")


def assemble(root, output):
    cache = root / "dev/cache"
    prototype = root / "ridge_calibrated"
    load = lambda p: torch.load(p, map_location="cpu", weights_only=False)
    interval = load(cache / "multiscale_interval_cache/best_pyramid96.pt")
    energy = load(cache / "energy_head_cache/best_energy64.pt")
    forecast = load(cache / "forecast_head_cache/best_joint32.pt")
    direct = load(cache / "energy_network_cache/energy_nn.pt")
    state = dict(
        interval_coefficients=load(prototype / "learned_interval.pt")["coefficients"],
        width_scale=load(prototype / "learned_width.pt")["width_scale"],
        tke_coefficients=load(prototype / "supervised_tke.pt")["coefficients"],
        history_coefficients=load(prototype / "temporal_history.pt")["native_beta"],
        accuracy=load(root / "ridge_compact/neural_heads.pt"),
        multiscale=dict(interval_state=interval["state_dict"], width=96,
                        energy_state=energy["state_dict"], energy_width=64, energy_strength=.5),
        forecast=dict(state_dict=forecast["state_dict"], width=32),
        energy=dict(state_dict=direct["state_dict"], width=32, strength=float(direct["strength"])))
    torch.save(state, Path(output) / "adapters.pt")


def train(root, output, device):
    root = Path(root)
    run(PACKAGE, "training/calibration.py", ("analytic", root))
    source = root / "ridge_neural"
    prototype = root / "ridge_calibrated"
    for name in ("learned_interval.pt", "learned_width.pt", "supervised_tke.pt", "temporal_history.pt"):
        shutil.copy2(prototype / name, source / name)
    stages = [
        ("refine_neural_interval_rich.py", ["prepare"]),
        ("refine_neural_interval_rich.py", ["train", "--epochs", 20, "--device", device]),
        ("refine_neural_moments.py", ["prepare"]),
        ("refine_neural_moments.py", ["train", "--epochs", 20, "--device", device]),
    ]
    for name, arguments in stages:
        run(root, "dev/" + name, arguments)
    neural_weights(root)
    for name, arguments in [
        ("accuracy_screen.py", []),
        ("prepare_compact_interval.py", ["prepare"]),
        ("train_compact_interval.py", ["--epochs", 60, "--width", 32]),
        ("prepare_c45.py", []),
        ("train_compact_interval.py", ["--epochs", 40, "--width", 32, "--extra-c45", "--compact28"]),
        ("train_wide_moments.py", ["--epochs", 30, "--response-range", 1.5]),
    ]:
        run(root, "dev/" + name, arguments)
    compact_weights(root)
    for name, arguments in [
        ("prepare_temporal_interval.py", []),
        ("train_temporal_interval.py", ["--epochs", 30, "--steps", 160, "--batch", 4, "--width", 48, "--tag", "full48"]),
        ("train_canonical_interval.py", ["--epochs", 40, "--steps", 120, "--batch", 16, "--width", 96, "--tag", "canonical96", "--device", device]),
        ("train_multiscale_interval.py", ["--epochs", 50, "--stop-after", 25, "--steps", 120, "--batch", 12, "--device", device]),
        ("prepare_joint.py", []),
        ("train_energy_head.py", ["--epochs", 40, "--stop-after", 20, "--steps", 100, "--batch", 12, "--device", device]),
        ("prepare_distribution.py", []),
        ("prepare_moments.py", []),
        ("train_moment_head.py", ["--history", "--tag", "history32", "--epochs", 60, "--steps", 120, "--batch", 8, "--seed", 180911, "--device", device, "--max-seconds", 12000]),
        ("prepare_forecast.py", []),
        ("train_forecast_head.py", ["--tag", "joint32", "--epochs", 60, "--steps", 120, "--batch", 8, "--seed", 180912, "--resume", "dev/cache/moment_head_cache/best_history32.pt", "--device", device, "--max-seconds", 12000]),
        ("prepare_energy.py", ["--device", device, "--max-seconds", 12000]),
        ("train_energy_network.py", ["--epochs", 30, "--steps", 100, "--batch", 16, "--device", device, "--max-seconds", 12000]),
    ]:
        run(root, "dev/" + name, arguments)
    assemble(root, output)


if __name__ == "__main__":
    if sys.argv[1] == "analytic":
        fit_analytic(sys.argv[2])
    else:
        train(sys.argv[1], sys.argv[2], sys.argv[3])
