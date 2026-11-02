# Test-Time Adaptation via Fast Online Calibration

**NeurIPS 2026 RealPDE Competition · Track 2 (LTTTA)**

This repository implements a streaming forecaster that adapts at test time without fine-tuning. A distilled convolutional predictor remains frozen, while each revealed target is used to update lightweight online calibrators in closed form. A conditional Gaussian mixture then converts the calibrated forecast into prediction intervals optimized for the interval score. Our team placed in the **top 10 in the final evaluation**. See the [official results](https://realpdecompetition.github.io/results.html).

**Team:** Junru Wang and Shuhong Wang, Xi'an Jiaotong University  

## Contents


- [Method](#method)
- [Results](#results)
- [Quick start](#quick-start)
  - [1. Download the official initialization checkpoint](#1-download-the-official-initialization-checkpoint)
  - [2. Prepare the training data](#2-prepare-the-training-data)
  - [3. Train](#3-train)
- [Inference](#inference)
- [Repository layout](#repository-layout)
- [Citation](#citation)
- [License and attribution](#license-and-attribution)

## Method

- **Distilled.** A width-48 convolutional student is distilled from the official CNO and replaces it at inference.
- **Online calibration.** Each revealed target updates the calibration state through recursive ridge regression for trajectory-specific bias correction, moving averages of residual and energy (TKE) statistics, and pixel-wise Hedge updates over seven interval-width experts. Test-time adaptation is gradient-free.
- **Learned calibration heads.** Compact interval, moment, energy, and temporal heads read the calibration state and refine the forecast's mean, temporal shape, and fluctuation energy.
- **Prediction intervals.** A three-component Gaussian mixture predicts the target distribution. Interval endpoints maximize the expected score, coverage probability × exp(−width penalty), through a candidate search followed by a Newton step.
- **Fast execution.** fp16 networks, packaged PTX kernels, and CUDA-graph replay.

## Results

**Final leaderboard:** Final-Top-10, total score **82.6271 ± 0.0280** (mean ± std over 15 official evaluations). See the [official results](https://realpdecompetition.github.io/results.html).

**[Main Development leaderboard](https://www.codabench.org/competitions/17385/#/results-tab)** (Codabench submission 940980):

| Metric | Score |
|---|---:|
| Rel L2 | 94.702621 |
| TKE | 79.906061 |
| MVPE | 94.331524 |
| Time | 92.331864 |
| SPS | 44.127993 |
| Overall | **81.935842** |

## Quick start

Requirements: Linux x86_64, an NVIDIA Ampere or Ada GPU with at least 48 GB of memory, Python 3.10, and PyTorch 2.2.2 with CUDA 12.1.

### 1. Download the official initialization checkpoint

```bash
python3 scripts/download_assets.py
```

This fetches the official CNO checkpoint from the [official data release](https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDE-Competition-Data) and verifies its SHA256.

### 2. Prepare the training data

Download the data and normalizer from the official release and arrange them as follows:

```text
data/
├── train_real/
└── mean_std_real.pt
```

`train_real/` holds the 81 trajectories listed in `training/recipe.json` (without `7575_0.h5`), and `mean_std_real.pt` is the official normalizer.

### 3. Train

With Docker, built on the official competition image:

```bash
docker build --platform linux/amd64 -t realpde-track2 .
mkdir -p output
docker run --rm --network none --gpus '"device=0"' --shm-size 32g \
  -v "$(pwd)/data:/data:ro" \
  -v "$(pwd)/output:/output" \
  realpde-track2 \
  python scripts/train.py --data-dir /data --output-dir /output --device cuda
```

Without Docker:

```bash
python -m pip install -r requirements.txt
python scripts/check_environment.py
python scripts/train.py --data-dir data --output-dir output --device cuda
```

Use an empty output directory. Add `--check-only` to validate the data layout and checkpoint without training. Training writes `model.pth`, `adapters.pt`, `algorithm.pt`, the runtime kernels, and logs to the output directory.

## Inference

```python
import solution

model = solution.get_ttt_model("output", "cuda")
model.reset_ttt_state()
prediction, info = model.ttt_step(input_norm, prev_target_norm)
lower, upper = info["lower"], info["upper"]
```

`input_norm` is a normalized float32 tensor of shape `(1, 20, 32, 64, 3)`. `prev_target_norm` is `None` for the first window and the revealed target of the previous window afterwards. Channels are `u, v, p`, with a zero pressure channel. Call `reset_ttt_state()` at the start of each trajectory. See [examples/streaming_inference.py](examples/streaming_inference.py).

## Repository layout

```text
solution.py                 training and inference entry points
inference.py                inference model assembly
flow_model/                 predictor, online calibration, intervals, packaged PTX
training/                   training pipeline and recipe
training_templates/
├── calibration/            calibration and head training stages
├── distillation/           student and interval-head distillation stages
└── teacher/                teacher model for distillation
scripts/                    checkpoint download, environment check, training CLI
examples/                   streaming inference example
official_assets/            official checkpoint metadata
licenses/                   third-party licenses
```

## Citation

See [CITATION.cff](CITATION.cff). Please also cite [RealPDEBench](https://github.com/AI4Science-WestlakeU/RealPDEBench).

## License and attribution

Our code is released under **CC BY-NC 4.0** ([LICENSE](LICENSE)). Code adapted from RealPDEBench keeps its CC BY-NC 4.0 terms, and NVIDIA-derived parts of the CNO implementation keep the NVIDIA StyleGAN3 license. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
