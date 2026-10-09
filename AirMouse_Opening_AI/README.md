# NIDAR AirMouse Opening Perception

Research code for identifying **visible physical openings** in indoor scenes using YOLO instance segmentation.

> **Research release. V3 is not ready for live deployment.** The trained V3 baseline passed software checks but failed external-image semantic tests: screens/windows were predicted as openings and real doorways were missed. See [model card](docs/MODEL_CARD.md) and [release findings](docs/RELEASE_FINDINGS.md). No flight-safety or autonomous-navigation claim is made.

## What the model does

Current class: **`0 = candidate_opening`**.

A candidate is a visible free-space aperture through a wall, partition, doorway, gate or broken wall that connects spaces. Closed doors, windows, mirrors, screens, closed glass and furniture gaps are excluded. Multiple and nested openings are separate instances. Stereo depth and geometry must later determine actual clearance.

Older scripts and labels use `traversable_opening` and sometimes broader door/window semantics. They are retained as historical research tools; do not merge those labels with V3 without re-review. See [dataset specification](docs/DATASET.md).

## Team setup

**Start here: [TEAM_SETUP.md](docs/TEAM_SETUP.md)** for clone, installation, automatic artifact setup, inference and troubleshooting.

## Combined opening + human perception

The project now exposes one shared-frame pipeline using the unchanged hybrid human
detector from [RajGoyal01/NIDAR](https://github.com/RajGoyal01/NIDAR) and the local
opening segmentation checkpoint. See [integration setup, contract and measured tests](docs/PERCEPTION_INTEGRATION.md).
It produces opening masks and person boxes with the same frame ID/timestamp, plus
explicit branch errors and stale-frame status. It does not merge weights, count
unique survivors, or issue flight commands.

### Environment setup

Python 3.14.0 was used locally. Create an environment in a fresh clone:

```bat
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
```

Install PyTorch/torchvision using the command for your machine from the [official PyTorch selector](https://pytorch.org/get-started/locally/). For CPU-only inspection/inference:

```bat
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe airmouse.py doctor
```

For GPU use, choose the CUDA wheel supported by your driver using the official selector, then install `requirements.txt`. The observed training environment used torch 2.14.0+cu130 and torchvision 0.29.0+cu130 on an RTX 4060 Laptop GPU. `requirements-observed.txt` records direct versions, not a complete dependency lock. Fresh dependency installation on other machines is not yet verified.

The source checks do not need datasets, weights, CUDA or Ultralytics:

```bat
.venv\Scripts\python.exe -m pip install Pillow numpy
.venv\Scripts\python.exe tools\check_source.py
.venv\Scripts\python.exe -B -m unittest discover -s tests
.venv\Scripts\python.exe -B -m unittest discover -s annotation_tools
```

## Predict on a local image

Supply a **trusted, separately obtained candidate_opening checkpoint** and your own image. The experimental opening checkpoint is available as a versioned GitHub release asset; datasets are not distributed. Missing model files fail explicitly; the CLI does not download a substitute model.

```bat
.venv\Scripts\python.exe airmouse.py predict --model "weights\candidate_opening.pt" --source "my_images\room.jpg" --output "output\trial_01" --device 0
```

Use `--device cpu` without CUDA. `--source` can also be a directory of images (nonrecursive). Default confidence is 0.25 and image size is 640. Every run requires a new output directory.

Output: annotated preview images and `predictions.json` containing the model hash, input hashes, confidences and normalized polygon coordinates. Predictions are proposals, not ground truth or drone-clearance decisions. The V3 checkpoint is suitable for reproducing research failures, not live promotion.

## Project map

| Path | Purpose |
|---|---|
| `airmouse.py` | Portable environment check and image inference entry point. |
| `train_v3.py` | Exact V3 experimental recipe; requires private local annotation artifacts. |
| `test_v3_release.py` | Exact external test recipe; requires recorded review/image assets. |
| `annotation_tools/` | Legacy local annotation/review tooling and existing unit tests. |
| `tools/` | Source parsing and Git release-content checks. |
| `tests/` | CLI boundary tests. |
| `docs/` | Dataset specification, measured model evidence and GitHub handoff. |
| `train_v2*.py`, `build_*.py`, `live_*.py` | Historical workflows; some contain machine-specific paths or start work immediately. Read before running. |
| `datasets/`, `runs/`, `weights/` | Local-only inputs/artifacts, ignored by Git. |

## Training and reproducibility

The published code contains the experiment recipe and measured findings. It is **not a self-contained training-data release**. Follow [dataset restoration requirements](docs/DATASET.md) before running `train_v3.py --prepare`; training fails if required files are absent and never invents data. The frozen V3 recipe reserves whole visual groups: 58 train / 8 validation / 9 test, with 37 annotations excluded pending review.

Do not use the exposed external challenge images as an unseen benchmark after incorporating them into training. Establish a newly frozen, human-reviewed real-image holdout for future experiments.

## Current pipeline

```text
Local source data -> audit / review -> explicit polygons -> grouped split
    -> pretrained YOLO11n-seg -> training -> synthetic evaluation
    -> external-image release testing -> V3: DO NOT PROMOTE
    -> future real-data correction and independent evaluation
```

## GitHub release

See [GITHUB_RELEASE.md](docs/GITHUB_RELEASE.md) for exact Windows CMD commands, exclusions and the final pre-push check. The included GitHub Actions workflow runs source/unit/release-content checks, not GPU training or model-accuracy evaluation. A workflow file existing locally is not evidence of a successful hosted CI run.

## Licenses and external assets

The team repository already contains an MIT license for its own code. Public visibility is not a blanket license for code, datasets or model weights. Ultralytics has its own [license terms](https://www.ultralytics.com/license), and other dependencies/assets keep their respective terms. Third-party images and recordings are not included. The opening checkpoint is published separately as a research release asset; upstream model terms still apply. Review [THIRD_PARTY.md](docs/THIRD_PARTY.md) before distribution beyond this source-only research repository.
