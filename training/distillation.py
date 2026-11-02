"""Student distillation and unified head training."""
import json
import importlib.util
from pathlib import Path
import shutil
import sys
import numpy as np
import torch
from training.common import PACKAGE, clean_checkpoint, run, write_json


def build(variant, device, root):
    context = json.loads((Path(root) / "training_context.json").read_text())
    if variant == "teacher":
        teacher_root = Path(root) / "dev/backbone/baseline/submission"
        name = "training_teacher_flow_model"
        specification = importlib.util.spec_from_file_location(
            name, teacher_root / "flow_model/__init__.py",
            submodule_search_locations=[str(teacher_root / "flow_model")])
        namespace_package = importlib.util.module_from_spec(specification)
        sys.modules[name] = namespace_package
        specification.loader.exec_module(namespace_package)
        namespace = {}
        source = (teacher_root / "submission.py").read_text().replace("from flow_model.", "from " + name + ".")
        exec(compile(source, str(teacher_root / "submission.py"), "exec"), namespace)
        return namespace["get_ttt_model"](str(teacher_root), device)
    if variant != "student":
        raise ValueError(variant)
    from inference import _build_predictor
    from flow_model.optimized import optimize
    return optimize(_build_predictor(context["output_dir"], device), context["output_dir"])


@torch.inference_mode()
def cache_student(root, device):
    from flow_model.calibration import INPUT_MEAN, INPUT_STD, TARGET_MEAN, TARGET_STD
    root = Path(root)
    model = build("student", device, root)
    im, ist, tm, tst = [torch.tensor(x, device=device).reshape(1, 1, 1, 1, 3)
                        for x in (INPUT_MEAN, INPUT_STD, TARGET_MEAN, TARGET_STD)]
    split = json.loads((root / "dev/backbone/split.json").read_text())
    standard = root / "dev/shared/cache"
    dense = root / "dev/algorithm/cache_dense"
    standard.mkdir(parents=True, exist_ok=True)
    dense.mkdir(parents=True, exist_ok=True)
    records = {}
    for path in sorted((root / "dev/public_data").glob("*.npz")):
        velocity = torch.from_numpy(np.load(path)["velocity"]).to(device)
        raw = torch.cat((velocity, torch.zeros_like(velocity[..., :1])), -1)
        offsets = (0, 5, 10, 15) if path.stem + ".h5" in split["train50"] else (0,)
        for offset in offsets:
            model.reset_ttt_state()
            previous = None
            values = {key: [] for key in ("input", "target", "prediction", "lower", "upper")}
            for start in range(offset, len(raw)-39, 20):
                inp = (raw[start:start+20][None]-im)/ist
                prediction, info = model.ttt_step(inp, previous)
                for key, x in (("prediction", prediction), ("lower", info["lower"]), ("upper", info["upper"])):
                    values[key].append((x*tst+tm)[0, ..., :2].permute(0, 3, 1, 2).cpu().contiguous())
                target = raw[start+20:start+40][None]
                values["input"].append(raw[start:start+20, ..., :2].permute(0, 3, 1, 2).cpu().contiguous())
                values["target"].append(target[0, ..., :2].permute(0, 3, 1, 2).cpu().contiguous())
                previous = (target-tm)/tst
            values = {key: torch.stack(x) for key, x in values.items()}
            name = path.stem + (f"_o{offset:02d}" if offset else "")
            torch.save(values, (dense if offset else standard) / (name + ".pt"))
            records[name] = {"windows": len(values["input"]), "offset": offset, "stride": 20}
        print("Student causal cache", path.stem, len(records), flush=True)
    write_json(standard / "manifest.json", {"status": "COMPLETE", "files": records,
                "source": 'student and eight adapters', "causal": True})


def fold(source, destination, point_gain=1.0, interval_gain=1.0):
    state = clean_checkpoint(source, destination)
    strength = float(state.get("strength", 1.0))
    for name in ("net.out.weight", "net.out.bias"):
        state["state_dict"][name][:43].mul_(strength * point_gain)
        state["state_dict"][name][43:].mul_(strength * interval_gain)
    state["strength"] = 1.0
    torch.save(state, destination)
    return state


def energy_and_density(root, output):
    load = lambda path: torch.load(path, map_location="cpu", weights_only=False)
    state = load(root / "submission_native/algorithm.pt")
    energy = load(root / "dev/component_energy/reports/best.pt")
    tensors = {key: value.detach().cpu().float().contiguous() for key, value in state["state_dict"].items()}
    tensors["net.out.weight"] = torch.cat((tensors["net.out.weight"], energy["state_dict"]["conv.weight"].cpu()), 0)
    tensors["net.out.bias"] = torch.cat((tensors["net.out.bias"], energy["state_dict"]["conv.bias"].cpu()), 0)
    density = load(root / "dev/algorithm/reports/density_best.pt")
    recipe = json.loads((PACKAGE / "training/recipe.json").read_text())["density"]
    final = dict(state_dict=tensors, width=32, channels=196, mode="joint", strength=1.0,
                 component_energy=True, energy_strength=float(energy["strength"]),
                 density=dict(channels=196, width=24,
                              state_dict={key: x.detach().cpu().float().contiguous() for key, x in density["state_dict"].items()}),
                 density_steps=recipe["steps"], density_blend=recipe["blend"],
                 density_point_strength=recipe["point_strength"], density_halves=recipe["halves"],
                 density_rates=recipe["rates"])
    torch.save(final, Path(output) / "algorithm.pt")


def train(root, output, device):
    root, output = Path(root), Path(output)
    write_json(root / "training_context.json", {"output_dir": str(output)})
    (root / "dev/evaluate.py").write_text(
        "from pathlib import Path\nfrom training.distillation import build as _build\n"
        "ROOT=Path(__file__).resolve().parents[1]\n"
        "def build(variant, device): return _build(variant, device, ROOT)\n")
    for path in ("dev/backbone/reports", "dev/algorithm/reports", "dev/refined/reports", "dev/native/reports", "dev/component_energy/reports"):
        (root / path).mkdir(parents=True, exist_ok=True)
    teacher_root = root / "dev/backbone/baseline/submission"
    shutil.copytree(PACKAGE / "training_templates/teacher", teacher_root)
    checkpoint = torch.load(PACKAGE / "official_assets/sim_real_cno.pth", map_location="cpu", weights_only=False)
    torch.save(checkpoint["model_state_dict"], teacher_root / "model.pth")
    shutil.copy2(output / "adapters.pt", teacher_root / "adapters.pt")
    run(root, "dev/backbone/train_student.py", ["--width", 48, "--epochs", 80])
    run(root, "dev/backbone/train_student.py", ["--width", 48, "--epochs", 200, "--augment", "--resume", "dev/backbone/reports/student_48_best.pt"])
    clean_checkpoint(root / "dev/backbone/reports/student_48_aug_best.pt", output / "model.pth")
    for name in ("distill_cache.pt", "distill_aug_cache.pt"):
        (root / "dev/backbone" / name).unlink(missing_ok=True)
    run(PACKAGE, "training/export_runtime.py", ["--root", output, "--kind", "base", "--device", device])
    run(PACKAGE, "training/distillation.py", ["cache", root, device])
    for mode, width, epochs, extra in (("joint", 32, 60, []), ("moments", 24, 50, []), ("interval", 24, 40, ["--hard"])):
        run(root, "dev/algorithm/train_wave.py", ["--mode", mode, "--width", width, "--epochs", epochs, "--seed", 41, "--dense", "--flip", *extra])
    run(root, "dev/algorithm/train_density.py", ["--epochs", 30])
    run(root, "dev/algorithm/cache_teacher.py")
    run(root, "dev/algorithm/train_unified.py")
    clean_checkpoint(root / "dev/algorithm/reports/unified_best.pt", root / "submission_algorithm/algorithm.pt")
    shutil.rmtree(root / "dev/algorithm/cache_teacher")
    run(root, "dev/endpoints/screen_endpoints.py", ["--split", "train50", "--dense", "--steps", 10, "--teacher-dir", "dev/refined/cache_refined"])
    run(root, "dev/refined/train_refined_student.py", ["--epochs", 35, "--mix", .5])
    candidate = root / "dev/refined/reports/refined_student_best.pt"
    if not candidate.exists():
        candidate = root / "submission_algorithm/algorithm.pt"
    fold(candidate, root / "submission_refined/algorithm.pt", point_gain=1.5, interval_gain=1.0)
    run(root, "dev/native/train_refined_native.py", ["--epochs", 35, "--lr", 1e-4, "--seed", 163])
    candidate = root / "dev/native/reports/refined_native_best.pt"
    if not candidate.exists():
        candidate = root / "dev/native/reports/refined_native_last.pt"
    fold(candidate, root / "submission_native/algorithm.pt")
    shutil.rmtree(root / "dev/refined/cache_refined")
    shutil.rmtree(root / "dev/algorithm/cache_dense")
    run(root, "dev/component_energy/train.py", ["--epochs", 20])
    energy_and_density(root, output)


if __name__ == "__main__":
    if sys.argv[1] == "cache":
        cache_student(sys.argv[2], sys.argv[3])
    else:
        train(sys.argv[1], sys.argv[2], sys.argv[3])
