# Team setup: combined opening and human perception

## What you receive

One pipeline runs the opening segmentation model and the upstream hybrid person detector on the same image. The person detector uses two networks, so this integration runs three networks sequentially. Results contain opening polygons, person boxes, frame identity, freshness and branch health.

**Research only:** opening baseline V3 failed external-image quality checks (screens/windows can be mistaken for openings; real openings can be missed). Software success does not establish prediction correctness, clearance, survivor identity or flight readiness.

## 1. Requirements

Git, internet for initial setup, and Python 3.14 (the locally tested version). CPU execution is supported by configuration; measured integration runs used an NVIDIA RTX 4060 Laptop GPU. No training dataset is required to infer on your own images. Fresh dependency installation on a second machine and live camera hardware have not been verified.

From **Windows CMD**:

```bat
git clone https://github.com/Raghav-Kumar01/NIDAR.git
cd NIDAR\AirMouse_Opening_AI
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe tools\setup_team.py --device cpu
.venv\Scripts\python.exe airmouse.py doctor
```

For NVIDIA GPU use, install the CUDA-enabled PyTorch build compatible with your driver using the official PyTorch installation selector instead of the CPU command, then run setup with `--device 0`. The recorded environment is in `requirements-observed.txt`; it is an observation, not a complete lockfile.

Setup is deliberately separate from pip installation. It downloads the exact opening baseline from release `airmouse-v0.1.0-research`, verifies its SHA-256, clones the human repository at commit `0d112b4b1bb28e8e1511a9e994a3de78e3182356`, verifies its model hashes, and creates an ignored local config. Existing modified artifacts/configs are preserved and mismatches fail explicitly.

Opening checkpoint SHA-256:
`af4fc589a206cf2aaab7f4fb38ca396769a468f36eb1fa00544dddb591e2944b`

## 2. Run your own image, folder, video or webcam

Replace the source path with an existing file. Always use a new output directory.

```bat
.venv\Scripts\python.exe run_perception.py --config config\perception.local.json --source "C:\Data\room.jpg" --output output\image_01
.venv\Scripts\python.exe run_perception.py --config config\perception.local.json --source "C:\Data\images" --output output\images_01 --max-frames 100
.venv\Scripts\python.exe run_perception.py --config config\perception.local.json --source "C:\Data\walkthrough.mp4" --output output\video_01 --max-frames 100 --show
.venv\Scripts\python.exe run_perception.py --config config\perception.local.json --camera-index 0 --output output\camera_01 --seconds 30 --max-frames 300 --show
```

Webcam mode is implemented but hardware unverified. ESC closes a displayed run. Output includes `frames.jsonl`, annotated frames, `provenance.json` and `summary.json`. Read `status` and `branches` before consuming candidates: `degraded` means a branch failed; `stale` discards observations. Zero detections do not prove no person/opening exists.

## 3. Configuration

Edit ignored `config/perception.local.json`. Paths resolve relative to the config directory.

| Key | Default | Meaning |
|---|---|---|
| human_repository | ../.external/nidar-human | Pinned upstream checkout |
| opening_model | ../weights/candidate_opening.pt | Verified research checkpoint |
| device | cpu | Change to `0` for first CUDA GPU |
| opening_confidence | 0.25 | Opening score threshold, not correctness probability |
| human_confidence | 0.2 | Upstream hybrid detector primary threshold |
| max_age_ms | 10000 | Offline inspection budget; not a flight-control deadline |

Setup preserves an existing local config, including its device. Change the file explicitly to switch devices after initial setup. Consumers needing recent frames must select a suitable tighter age limit and measure latency on their hardware.

## 4. Engineering checks

```bat
.venv\Scripts\python.exe tools\check_source.py
.venv\Scripts\python.exe -B -m unittest discover -s tests
.venv\Scripts\python.exe -B -m unittest discover -s annotation_tools
.venv\Scripts\python.exe tools\check_release.py
```

These checks verify code/contracts and release contents, not detection accuracy. Hosted CI runs these checks without loading model weights.

## 5. Troubleshooting

- `py -3.14` missing: install Python 3.14 and reopen CMD.
- Setup download failure: check internet access and the repository release asset; rerun setup. Do not substitute another checkpoint under the same filename.
- Existing human checkout mismatch: preserve your changes and use a separate clean checkout at the pinned revision; update the local config. Setup never resets your work.
- CUDA unavailable: use `cpu` or install a compatible CUDA PyTorch build; `airmouse.py doctor` reports the environment.
- Stale output: inspect timing in `summary.json`/`frames.jsonl`; offline use can increase `max_age_ms`. Increasing it does not make processing faster.
- `degraded` output: inspect the per-branch error type and model/environment compatibility. Do not count a failed branch as a negative prediction.
- Output folder exists: choose a new folder; outputs are protected from overwriting.
- No camera: close other applications using it, check OS permissions, and try the correct camera index.

## Important files and how to explain them

`tools/setup_team.py` takes a device choice, retrieves pinned artifacts, validates hashes, and writes a local configuration. It uses only the standard library and Git; it does not train or execute models. Interview explanation: "Our setup reproduces the tested source and weights instead of silently choosing newer models."

`perception.py` adapts both model systems into one versioned frame result. `run_perception.py` feeds images/video/camera frames into it and saves inspectable results. See `PERCEPTION_INTEGRATION.md` for their inputs, outputs, timing semantics and limitations.

The root MIT license applies to the team's own code; external software and model terms remain applicable. See `THIRD_PARTY.md`. Datasets, recordings, training outputs and environments are excluded from Git.
