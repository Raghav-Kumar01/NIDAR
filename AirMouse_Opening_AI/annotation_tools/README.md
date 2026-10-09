# Full-dataset annotation assistance

> **Legacy semantics:** this package documents an earlier annotation phase in which
> open windows could be candidates. The current V3 class `candidate_opening`
> excludes windows. See `../docs/DATASET.md`. Do not bulk-merge these historical
> reviews into V3 or treat existing proposals as current ground truth.

Target: class 0 is the visible free-space region inside an opening. Physical
flight clearance cannot be established from an RGB annotation. Open windows may
be candidates for a drone; a door or window object label alone is insufficient.
Exclude visible panels, frames, glass and obstacles from the opening polygon.
Use ambiguous when the image does not establish the target clearly.

## What this package does

`full_dataset.py` inventories all RGB sources in the supplied raw directory,
the existing MonoRaceGate/V1 images, pilot frames and the batch-1 annotation
export. Depth maps and source annotation masks are assets, not extra RGB
training examples. The package reads original images in place. Every source
path and SHA-256 is recorded; exact file copies become aliases of one record.
Byte deduplication does not detect all re-encoded copies or similar video frames.

Existing usable class-0 annotations and prior curated negatives are retained
as editable proposals. Invalid source labels remain recorded as repair needs.
Source door/window object labels are referenced but never silently renamed into
the target class. Previous ambiguous/rejected curation decisions stay on hold.

Remaining images receive predictions from the local V2B model. These are
unverified pseudo-labels from a model trained on earlier data, not independent
ground truth. The model can reproduce its own mistakes and miss novel openings.
Confidence is not correctness. No detection means review required, not negative.
Low-confidence proposals are intentionally included at 0.15 for editing. Record
confidence and the checkpoint hash are retained with package configuration.

## CMD commands (from the project folder)

```bat
.venv\Scripts\python.exe -B annotation_tools\full_dataset.py build --raw F:\AirMouseRaw --output datasets\annotation_full_v1
.venv\Scripts\python.exe -B annotation_tools\full_dataset.py predict --output datasets\annotation_full_v1
.venv\Scripts\python.exe -B annotation_tools\full_dataset.py review --output datasets\annotation_full_v1
.venv\Scripts\python.exe -B annotation_tools\full_dataset.py report --output datasets\annotation_full_v1
.venv\Scripts\python.exe -B -m unittest discover -s annotation_tools
```

Build refuses an existing output directory. Predict is resumable and rejects
a changed inference configuration. Review opens at http://127.0.0.1:8766;
it binds only to localhost. The viewer reads only inventoried image IDs.
Review POSTs require a per-session token. Predictions stay in proposal_labels.
Only explicit positive/negative review saves a file into reviewed_labels.
Rejected and ambiguous reviews do not create training labels. A saved review
cannot be overwritten through the viewer. A positive requires finite in-range
vertices and nonzero area. This is mechanical validation, not proof of correct
semantics, boundary accuracy, topology or complete object coverage.

## Outputs and provenance

- inventory.json: counts, excluded assets and target definition.
- records/: image paths/hashes, source annotations, aliases, proposals,
  confidence, previous category and repair notes.
- proposal_labels/: YOLO polygon suggestions; NEVER bulk-use as final labels.
- reviews/: explicit review decisions with time and source image hash.
- reviewed_labels/: reviewed YOLO positives or explicitly empty negatives.
- inference_config.json: checkpoint hash, library versions and settings.
- last_run.json: measured inference-run wall time and processed images.
- status.json: snapshot of current processing and review counts; refresh with report.

No train/validation/test split or training YAML is emitted. First finish target
review and resolve source/scene/video grouping. Keep related frames and derived
copies together, including the Roboflow export and its raw parent images. Reserve
a genuinely independent test source before fitting or tuning a model. The
existing V1/V2 splits have already participated in development; adding new random
splits does not make those samples an independent evaluation set.

Training on every available raw image is not a requirement for a good model.
Use relevant reviewed examples, useful negatives and source diversity. Ground
truth review and a trustworthy holdout remain required after proposal generation.
