# Combined NIDAR perception: opening masks and human candidates

## Phase completion report

**Date:** 2026-10-09. **Phase:** Integrate existing opening and human ML inference.

**Status: COMPLETE** for shared-frame inference API, offline image/video execution and engineering contract tests. Live-camera hardware validation is not performed. Opening-model semantic release remains FAILED. Overall drone completion is not assigned an unsupported percentage.

## 1. What was built

One input BGR image enters both existing model systems. One structured result contains opening masks, human boxes, frame identity, reception timestamp, branch health and timing. The upstream human system itself runs a COCO primary and posture specialist, so combined hybrid mode executes three YOLO networks sequentially.

The friend repository was cloned outside this repository and remains unchanged:

- URL: https://github.com/RajGoyal01/NIDAR.git
- Tested/pinned commit: `0d112b4b1bb28e8e1511a9e994a3de78e3182356`
- Package: `Raj Gupta/ML/nidar_survivor_demo`
- Reused: `HybridPersonDetector`, `DetectorConfig`, and `PhoneStream` for optional webcam input.
- General and specialist checkpoint hashes are verified against the upstream model manifest before loading.

The upstream detection module supports `person` and `person_candidate`; the hybrid adapter retains its upstream fusion policy. We did not replace it with a generic person-only call or rewrite its specialist logic.

Implemented: combined API, shared preview, JSONL, provenance, image folders, recorded video, bounded optional webcam mode. Not integrated in this phase: BoT-SORT, temporal verification, Re-ID, persistent survivor manager, existing dashboard, ROS 2, stereo geometry or flight control. Those remain separate layers and are not simulated.

## 2. Files and responsibilities

| File | Inputs / outputs / important parts |
|---|---|
| `perception.py` | `load_pipeline()` validates config, pinned external checkout and hashes; wraps existing models. `PerceptionPipeline.process()` returns a `FrameResult` for one BGR image. `validate_observations()` rejects malformed output. |
| `run_perception.py` | CLI source handling, JSONL/provenance output and combined drawing. Uses upstream latest-frame capture for webcam input. |
| `config/perception.example.json` | Relative checkout/checkpoint paths, thresholds, device and maximum reception age. |
| `tests/test_perception.py` | Seven fake-adapter contract tests covering shared inputs, nested masks, branch failure, invalid output and stale frames. These do not measure model accuracy. |

Why this design: preserve ownership and trained-model behavior, keep camera acquisition separate, and give planning one stable interface. Sequential inference avoids concurrent GPU contention and makes branch timing/failure attribution straightforward. Parallel scheduling or a jointly trained network adds complexity and needs measured justification.

Interview explanation: "We compose the existing models behind adapters. Both observe the same frame, and the project consumes one versioned result without assuming that an opening is safe or a detected person is a new survivor."

## 3. Files modified and ownership

README links this integration; source CI now installs NumPy for contract tests; release manifest includes the new source/config/documentation. Existing training code, application model paths, raw inputs and all upstream tracked files remain unchanged. No remote push was performed.

## 4. Tests and actual evidence

| Test | Expected | Actual | Result |
|---|---|---|---|
| Shared input / mutation isolation | Both adapters see original same pixels | Tested with mutating fake adapter | PASS |
| One branch failure | Other branch survives; degraded is explicit | Tested | PASS |
| Already stale frame | Both models skipped | Tested | PASS |
| Result expires during inference | No usable observations emitted | Tested | PASS |
| Invalid box | Explicit error branch | Tested | PASS |
| Invalid frame / future timestamp | Rejected | Tested | PASS |
| Nested opening instances | Remain separate | Tested | PASS |
| Recorded image frames | Both real adapters execute without invalid outputs | 87/87 status ok | PASS execution |
| Recorded video decode path | Process bounded video stream | 10/10 status ok | PASS execution |
| Live webcam | Actual camera acquisition | Not run | UNVERIFIED |

Existing CLI boundary tests and annotation tests remain in the suite. Same-environment compatibility was checked with Python 3.14.0, PyTorch 2.14.0+cu130, Ultralytics 8.4.163, OpenCV 5.0.0.93, NumPy 2.5.3. Upstream's full application pins other versions; we imported only its detector/capture layer, not its dashboard/identity stack.

## 5. Actual measured results

On 87 existing recorded frames using RTX 4060 Laptop GPU:

- 87 frames with status `ok`; no degraded or stale output.
- 68 frames contained person candidates; 12 contained opening candidates.
- Median combined adapter time: 58.912 ms; p95: 73.555 ms.
- Timing includes sequential model calls, input copies and CPU result transfer; excludes decode, drawing and disk output, and includes the first cold opening call.

The separate ten-frame recorded-video smoke run had median 167.030 ms and p95
651.634 ms, with all ten frames structurally valid and zero opening/person outputs
in those initial views. This short run includes startup effects and demonstrates
timing variability; no guaranteed real-time FPS is claimed.

These are measured execution/output counts, not precision, recall or unique survivor counts. Several video scenes contain partial first-person limbs, posters, targets or occlusions; detector boxes on those require semantic review. The opening V3 model's prior false-positive/missed-opening findings remain unchanged. A valid merged result does not mean its observations are correct.

Local outputs: `output/combined_perception_smoke/` (87 frames), `output/combined_video_smoke/` (10 video frames). Each contains `frames.jsonl`, `provenance.json`, `summary.json` and combined previews. These generated images and local data stay ignored by Git.

## 6. Easy and technical explanations

Easy: two observers look at the same photo. One outlines openings, the other boxes people. We put both answers in one envelope labelled with the photo number. A failed observer is marked failed; silence is not treated as proof that the room is empty.

Technical: input is nonempty `uint8` BGR `[H,W,3]`, integer `frame_id`, and `received_at` from Python's monotonic clock. Both adapters receive copies of the original frame. Opening coordinates are normalized `[0,1]`; person boxes use original-image pixels `(x1,y1,x2,y2)`. Convert normalized x/y to pixels with `x_px=x_norm*W`, `y_px=y_norm*H`: x=0.5 at W=1280 is 640 pixels. Image-space overlap does not prove equal depth or physical obstruction.

The result schema is version `1.0`:

```text
frame_id, received_at_monotonic, width, height
status: ok | degraded | stale
branches: openings/humans -> ok | error:<type> | skipped_stale
opening_candidates: [{confidence, polygon_normalized}]
person_candidates: [{confidence, bbox_xyxy_pixels}]
timing_ms: openings, humans, total_ms, age_ms
opening_model_status: experimental_release_failed
identity_scope: frame_only_no_unique_survivor_count
flight_ready: false
```

Time is reception/decode time, not hardware sensor-exposure time. Stale input skips inference; output that becomes stale during inference discards observations. A successful branch may survive the other branch's exception with overall `degraded`. Staleness is a consumer guard, not hard real-time preemption of a running GPU operation. Read the status fields before using observations.

This interface is new and additive; existing modules retain their contracts. It can be imported directly:

```python
from pathlib import Path
import time
from perception import load_pipeline

pipeline, provenance = load_pipeline(Path('config/perception.example.json'))
# frame_bgr is supplied by your existing capture owner.
received_at = time.monotonic()  # assign immediately after acquisition/decode
result = pipeline.process(frame_bgr, frame_id=1, received_at=received_at)
payload = result.to_dict()
```

Do not start another camera reader when embedding into an existing application; pass its latest captured frame into this API. The current adapter does not invent persistent person IDs or subtract human boxes from opening masks.

## 7. Questions to answer

- Is this one trained model? No; one perception service composes three existing networks in hybrid mode.
- Why keep masks and people in separate arrays? Their class namespaces, geometry and consumers differ; both upstream class IDs happen to be zero.
- Can person box count become survivor count? No; that requires tracking, temporal verification and identity management.
- Does a person inside an opening box prove blockage? No; image-space overlap lacks depth.
- Why no automatic deployment? V3 failed semantic release checks; integration success does not repair its predictions.

## 8. Pipeline

```text
Image / recorded video / optional latest webcam frame
    -> frame ID + monotonic reception time
    -> opening YOLO segmentation
    -> upstream general-person + posture-specialist hybrid detector
    -> validate outputs / freshness / branch health
    -> FrameResult + one overlay + JSONL
    -> future project consumers (tracking, depth, mapping, planning)
```

## 9. Setup and run (Windows CMD)

Use [TEAM_SETUP.md](TEAM_SETUP.md), the authoritative instructions for the nested team release. The original measurements above predate this packaging. The helper installs artifacts under `.external/` and `weights/`; the default config uses CPU with a 10-second freshness budget for offline inspection. GPU and real-time consumers must choose and measure their own budget.

## 10. Next recommended phase

Review both branches on representative drone-camera footage, correct opening-model failures, then connect this contract to the project's chosen tracking/depth consumers. Live hardware and dashboard integration need separate acceptance. No future phase was automatically implemented.

## 11. Risks / limitations

Opening model remains semantically unsuitable for live promotion. Human detections are body/person candidates, not medical status or unique identity. Combined sequential latency is substantially greater than opening-only latency. CPU execution and live camera operation were not measured. No target-drone hardware, clearance, ROS or flight-control integration was tested. External source/license ownership remains with the friend repository.

## 12. Git checkpoint

Changes belong to the AirMouse source release; the external checkout is unchanged and not staged inside this repository. Generated previews and upstream checkpoints remain outside the staged source set. No remote commit/push performed.

Suggested commit: `Integrate pinned human detector with opening segmentation using shared-frame contract`.

Run source/unit checks and the staged release audit before committing. Preserve the model's failed-release status in any demonstration or README.
