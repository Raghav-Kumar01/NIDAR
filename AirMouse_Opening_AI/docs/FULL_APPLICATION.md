# Full combined perception application

This supersedes the detector-only integration scope of release v0.1.0. It connects the existing opening model to Raj's hybrid person detector, BoT-SORT tracking, temporal verification, OSNet Re-ID, persistent SQLite survivor manager, and local dashboard. All observe the same decoded frame. It is a laptop perception application; depth, mapping and flight control remain outside this application.

## Windows CMD setup

From `NIDAR\AirMouse_Opening_AI` with Python 3.14 and Git installed:

```bat
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements-full.txt
.venv\Scripts\python.exe tools\setup_team.py --device cpu
```

For GPU execution, install the CUDA-enabled PyTorch wheel compatible with your driver instead of the CPU wheel. The application accepts `--device 0` to use the first CUDA GPU. Existing users should install `requirements-full.txt` and rerun setup; setup now verifies OSNet as well as the two human YOLO checkpoints.

`lap==0.5.13` is deliberately used in this integration: the upstream 0.5.12 pin lacks a Python 3.14 Windows wheel and attempted a C++ source build. The compatible wheel was installed and the relevant upstream tracking/identity/dashboard regressions were run. No upstream algorithm thresholds were changed to compensate.

## Launch the complete application

System webcam, GPU:

```bat
.venv\Scripts\python.exe run_full_perception.py --source 0 --backend dshow --device 0 --output output\full_camera_01
```

CPU fallback: replace `--device 0` with `--device cpu`. Open **http://127.0.0.1:8765** in a browser. The dashboard is local-only. Press **Ctrl+C in the terminal** to save the mission and stop. Add `--native-preview` for an OpenCV window; Q/Esc closes that preview and ends the run.

Recorded video, bounded run:

```bat
.venv\Scripts\python.exe run_full_perception.py --source "C:\Data\walkthrough.mp4" --device 0 --output output\full_video_01 --seconds 60
```

Phone camera:

```bat
.venv\Scripts\python.exe run_full_perception.py --source "http://PHONE_IP:8080/video" --transport mjpeg --device 0 --output output\full_phone_01
```

Substitute your actual camera address. The runner does not change phone settings. Choose a fresh output folder for each launch. Use `--dashboard-port 8766` if 8765 is occupied. Starting on an occupied port fails explicitly.

Resume a saved mission:

```bat
.venv\Scripts\python.exe run_full_perception.py --source 0 --device 0 --output output\resumed_01 --resume-mission --mission-db output\full_camera_01\mission.sqlite3
```

The previous mission must be closed. Resume validates its schema and Re-ID settings; incompatible missions fail. A browser refresh preserves the mission. **New Mission / Reset Demo** asks for confirmation, closes/preserves the old database and creates a new mission; identity and opening frame history are cleared. The runtime prints the new database location.

## Dashboard and outputs

- One preview: person/identity labels plus cyan opening polygons.
- Opening branch status and candidate count; experimental model warning.
- Current tracked people, pending identities, unique identity **estimate**, and accepted duplicate events.
- Temporary track IDs remain distinct from persistent S identities.
- Searchable identity register, event filters, report export and confirmed mission reset.
- `frames.jsonl`: shared sequence, connection, reception timestamp, openings, detections, tracking, verification, Re-ID decisions, managed counts and validity.
- `provenance.json`: model hashes, pinned human revision, configuration and runtime arguments.
- `mission.sqlite3`: managed identities, appearance embeddings and audit trail; local only, excluded from Git.
- `mission_report.json`: final read-only summary without embeddings or images.

An opening failure is explicitly marked and does not become a successful empty prediction. Stale output is discarded. Human-pipeline failures stop the application and mark the dashboard unavailable. The full application's default freshness limit is **1 second**, using Raj's runtime; the detector-only config's `max_age_ms` is not used here. CPU hardware may need a larger `--stale-after` for offline diagnosis, but increasing the limit does not make inference faster.

## Architecture and ownership

```text
Camera / video -> latest decoded frame + sequence + connection
    -> opening segmentation
    -> Raj's general-person + posture-specialist fusion
    -> BoT-SORT -> temporal verification -> selective OSNet
    -> appearance matching -> survivor manager -> SQLite
    -> same-frame combined overlay + dashboard + JSONL
```

`run_full_perception.py` verifies dependencies/artifacts and launches full mode. It accepts the existing upstream camera/runtime options; the full identity pipeline and dashboard are enabled by default. Human model-path overrides are rejected because they would invalidate the checked artifact provenance.

`full_runtime.py` is an attributed integration copy of the pinned upstream orchestration loop, extended with the opening hook, shared-frame export and final report. Raj's detector, tracking, verification, Re-ID, persistence and dashboard algorithms remain imported from his unchanged checkout. `full_missions.py` adapts mission rotation so new databases stay inside this application's output directory. When upgrading upstream, compare its `main.py` with this integration copy and rerun the regressions; this is an explicit maintenance obligation.

`full_openings.py` owns only opening inference, polygon validation/drawing, frame/connection freshness and combined event export. It does not assign human identities or make safety decisions.

The earlier `run_perception.py` remains available as the lightweight detector-only API with its original schema. Full mode is a separate explicit entry point, so existing consumers are not silently given different semantics.

## What “complete” means here

All requested perception stages are connected in one application. First setup still downloads pinned dependencies and model artifacts; a GitHub source ZIP is **not an offline model bundle**. No training datasets are required for inference on your own camera/video.

The opening checkpoint remains `experimental_release_failed`; screens/windows and missed real doorways are known problems. Identity counts are appearance-based estimates, not guaranteed unique people or medical assessments. This work does not improve model accuracy through training or establish field identity accuracy. See the phase report for measured evidence and remaining limits.

## Repeat the engineering checks

From the package folder after setup:

```bat
.venv\Scripts\python.exe tools\check_source.py
.venv\Scripts\python.exe -B -m unittest discover -s tests
.venv\Scripts\python.exe -B -m unittest discover -s annotation_tools
.venv\Scripts\python.exe -m pip install "httpx>=0.28,<1"
.venv\Scripts\python.exe tools\check_full_application.py
```

HTTPX is used by the dashboard tests. The full check verifies the pinned checkout and artifacts, runs 130 selected upstream regressions and two combined-application fixtures. Fixtures do not measure model accuracy. The GitHub source workflow runs the source and 32 lightweight tests; the full 132-test run and GPU/video evidence in [FULL_APPLICATION_REPORT.md](FULL_APPLICATION_REPORT.md) were obtained locally.
