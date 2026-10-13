# Phase completion report: full combined perception application

Date: 2026-10-13. Scope: connect opening segmentation with Raj's full human-perception application and provide a reproducible team handoff.

**Status: COMPLETE for the requested software integration phase.** Implementation checkpoint: 100% of the listed integration features connected. Overall autonomous-drone completion is not quantifiable from this repository: no agreed hardware/flight acceptance checklist exists here. Field validation remains incomplete.

The friend's audit was correct about v0.1.0: it connected opening and human detection but did not connect Raj's tracker, Re-ID, survivor manager or dashboard. This phase corrects that scope. A source ZIP still needs the documented first-run setup to fetch Raj's pinned source and weights.

## 1. What we built

- IMPLEMENTED: shared-frame opening segmentation plus general-person/posture-specialist fusion.
- IMPLEMENTED: BoT-SORT, temporal verification, selective OSNet inference, appearance matching, SQLite identity management.
- IMPLEMENTED: combined preview, local dashboard, identity register, mission resume, confirmed reset, JSONL and final report.
- IMPLEMENTED: pinned upstream revision and three human checkpoint checksums verified before launch; opening checkpoint hash recorded in provenance and validated by setup.
- EXPERIMENTAL: all model-derived opening candidates and identity estimates. The opening checkpoint retains `experimental_release_failed`.
- NOT IMPLEMENTED in this application: stereo clearance, metric mapping, navigation, flight control or guaranteed unique-person counting.

## 2. Important files created

| File | Purpose, inputs and outputs | Important interface / judge explanation |
|---|---|---|
| `run_full_perception.py` | Config + camera arguments + verified model/source artifacts -> full application and provenance | `prepare`, `main`: checks the exact upstream version and weights before launching both systems. |
| `full_runtime.py` | Decoded camera frames -> both inference branches, tracker, verification, Re-ID, manager, dashboard and report | `run`: integrates the opening hook into the attributed pinned upstream loop. Algorithms remain owned by Raj. |
| `full_openings.py` | Opening adapter + frame snapshot -> validated polygons, overlay and combined JSONL | `OpeningHook`: keeps masks tied to the correct frame/connection, suppresses stale output and records branch failures. |
| `full_missions.py` | Existing mission/tracker/verifier + output directory -> replacement mission and fresh state | `rotate_mission`: preserves the old database and starts a separate mission inside the current output folder. |
| `requirements-full.txt` | Runtime dependency requirements -> reproducible dependency constraints | Adds full tracker/dashboard dependencies to the existing detector requirements. |
| `tests/test_full_openings.py` | Explicit synthetic fixtures -> eight regression tests | Tests frame isolation, failures, freshness, reset and source-frame immutability. |
| `tools/check_full_application.py` | Installed dependencies + verified config -> 132-test result | Runs selected upstream tests plus combined-loop and mission-rotation fixtures; does not measure accuracy. |
| `docs/FULL_APPLICATION.md` | Team instructions -> setup, launch, resume and repeatable checks | The authoritative full-application guide. |
| `docs/FULL_APPLICATION_REPORT.md` | Collected engineering evidence -> this scope and validation record | Explains precisely what was and was not verified. |

This design preserves the existing lightweight API and avoids rewriting teammate-owned algorithms. The integration loop is copied and attributed because upstream does not expose an opening-stage plugin interface. This creates an explicit maintenance cost when upgrading upstream.

## 3. Files modified

- `tools/setup_team.py`: verifies the OSNet checkpoint alongside the two human YOLO checkpoints.
- Package and root `README.md`: point teammates to the full application and explain estimate limitations.
- `docs/TEAM_SETUP.md`, `docs/PERCEPTION_INTEGRATION.md`: identify the earlier detector-only scope and link the full guide.
- `docs/THIRD_PARTY.md`: records ownership, upstream attribution and separate OSNet weight terms.
- `release_manifest.txt`: includes the new source, tests and documentation.
- `.github/workflows/airmouse-perception.yml`: installs headless OpenCV for lightweight integration tests.

## 4. Tests performed

| Test | Expected | Actual |
|---|---|---|
| Source parse | All Python source parses | PASS: 49 Python files |
| Lightweight unit suite | Existing API plus opening-hook invariants | PASS: 19 tests |
| Annotation utility regressions | Existing tools unchanged | PASS: 13 tests |
| Selected upstream + full integration checks | Tracking, verification, Re-ID, manager, dashboard, rotation and shared-frame export | PASS: 132 tests in 44.325 seconds locally |
| Dependency consistency | No unsatisfied installed requirements | PASS: `pip check` |
| Setup artifact validation | Pinned human revision and all three hashes; opening baseline hash | PASS; existing local configuration preserved |
| Real model recorded-video run | Both branches and downstream human stages produce output and close cleanly | PASS: 518 consumed frames, all `ok` |
| Resume recorded mission | Preserve mission ID and identity estimate | PASS: 1,671 frames, same mission ID, estimate remained 1 |
| Browser confirmed reset | Preserve old DB, create different mission, continue processing | PASS: new mission shown in dashboard, both databases retained |
| SQLite integrity after reset | Both DBs structurally readable | PASS: `integrity_check = ok` for both |

The full upstream test discovery was not a clean pass: an initial unqualified discovery command produced relative-import errors and lacked optional COCO showcase assets. The reproducible helper runs the 130 applicable module-qualified tests plus two new integration tests. We do not claim the entire upstream test suite passed.

The original `lap==0.5.12` installation failed on Windows/Python 3.14 because it required a C++ build toolchain. `lap==0.5.13` supplied a wheel; the tests above passed with it. This deliberate dependency difference is documented.

## 5. Actual measured results

Local environment: Windows, Python 3.14.0, PyTorch 2.14.0+cu130, OpenCV 5.0.0, NVIDIA GeForce RTX 4060 Laptop GPU. These describe this machine, not a portable performance guarantee.

Recorded input: local `datasets/v3_pilot/videos/Recording 01.mp4`. It has no adjudicated person-identity or opening ground truth for these runs and is not an independent accuracy benchmark. The latest-frame consumer can skip camera frames; counts below are consumed records, not the total input video frames.

| Run | Consumed records | Valid / stale | Measured outputs |
|---|---:|---|---|
| `output/full_validation_01` | 518 | 518 / 0 | 130 frames with opening candidates; 275 with person candidates; 82 with encoded Re-ID crops; final estimate 1, duplicate-event count 1 |
| `output/full_validation_resume` | 1,671 | 1,670 / 1 | Mission `55667fa5-b93e-47ff-adbd-77b24653bfcd` retained; final estimate 1, cumulative duplicate-event count 3 |
| `output/full_reset_ui` | 1,536 | 1,534 / 2 | Browser reset produced mission `996b8ed6-9e3d-4cf8-b277-02ac261ce858`; final new-mission estimate 1, duplicate-event count 1 |

All three runs recorded clean shutdowns. Stale records are explicit unavailable observations, not empty successful detections. Local evidence is retained in ignored `output/` and `release_checks/`; databases, embeddings, videos and model weights are excluded from the Git source release. No precision, recall, identity accuracy, calibrated uncertainty or field-performance claim is supported by these runs.

## 6. What you should have learned

**Easy explanation:** the detector notices people, the tracker follows a person between nearby frames, Re-ID compares appearance when someone returns, and the database remembers the mission. Your model marks visible opening candidates on that same picture. The dashboard displays these outputs together.

**Technical explanation:** the camera worker supplies a sequence ID, connection ID and monotonic reception time. Opening inference precedes the human branch on the same raw frame. BoT-SORT consumes boxes; temporal verification filters unstable evidence; OSNet selectively encodes eligible crops; the matcher and manager apply existing identity rules before persistence. A frame is exportable only while its connection matches and its age is within the freshness bound. The default bound is one second. This is local post-decode freshness, not measured camera-to-display latency.

Sequential inference is simple and preserves frame correspondence; it adds both branches' inference costs. The latest-frame capture avoids building a growing queue, at the cost of skipped frames. More complex concurrent inference should only be considered after a controlled latency benchmark establishes the need. Raising the freshness bound changes acceptance age, not inference speed.

No new loss function or model-training algorithm was introduced. Confidence is a model score, not proof of correctness. A temporary track ID is not a persistent identity. Persistent identities remain appearance-based estimates. `flight_ready: false` remains mandatory.

## 7. Questions you should be able to answer

- **Are the networks merged?** No; their outputs are integrated into one frame-scoped application. They keep their existing weights and responsibilities.
- **Why use Re-ID after tracking?** Tracking follows local temporal continuity; appearance matching helps associate an eligible returning observation with stored references.
- **Why not count every new track as a new survivor?** Occlusion/reconnection can create new track IDs for the same person.
- **Why can the app run correctly while predictions are wrong?** Execution tests verify wiring and state behavior; accuracy requires independent labeled examples and suitable metrics.
- **Does an opening polygon prove safe flight?** No. Depth and vehicle geometry are still required.
- **Why pin upstream?** To make the tested algorithms and checkpoint provenance reproducible and avoid silent changes.

## 8. Current implemented pipeline

```text
Camera/video -> latest decoded frame, sequence, connection, reception time
   -> opening segmentation --------------------------+
   -> hybrid human detector -> BoT-SORT              |
      -> temporal verification -> selective OSNet    |
      -> appearance matching -> survivor manager    |
      -> SQLite mission database -------------------+
   -> same-frame overlay + local dashboard + JSONL + final report

FUTURE: stereo depth -> clearance/geometry -> mapping/navigation/flight
```

## 9. What is still left

Second-machine full setup verification; full-mode live webcam/phone validation; independent opening/identity accuracy evaluation; controlled throughput/latency and resource profiling; multi-person occlusion, re-entry, clothing similarity and lighting trials. The known opening-model quality failures remain unresolved.

## 10. Next recommended phase

Run a documented, consented live-camera acceptance session with manually adjudicated identities and openings, followed by targeted model/data correction. Do not expand to autonomous flight on the basis of this software integration check. This next phase has not been implemented here.

## 11. Blockers and risks

- The opening model remains an experimental checkpoint with known false positives and missed openings.
- Appearance-based identity association can merge different people or split one person; no field-accuracy claim exists.
- Full runtime installation was exercised in the existing project environment, not a second clean machine. Setup artifact checking used an isolated checkout on this machine.
- GPU/video execution does not validate live camera drivers, phone networking or drone hardware.
- The integration copy of upstream orchestration must be reviewed when the pin changes.
- Dependencies and weights require network access on first setup; the source ZIP is not an offline bundle.
- Local mission DBs retain appearance embeddings; keep these private and out of Git.
- GitHub CI checks source and lightweight regressions; the full GPU/video and 132-test evidence is local.

## 12. Git checkpoint

Target repository: `Raghav-Kumar01/NIDAR`, existing isolated `AirMouse_Opening_AI/` directory.

Suggested branch: `main` for the explicitly requested team publication (or `integration/full-perception` for future reviewed changes).

Commit message: `Connect full survivor pipeline and dashboard to opening perception`.

Only reviewed source/docs/workflow files belong in the commit. Release checks must pass on the exact index before pushing; no datasets, weights, databases, videos or machine-local configs should be staged. The terminal handoff supplies the actual commit and CI result after publication.
