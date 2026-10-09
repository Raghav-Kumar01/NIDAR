# Dataset contract and local restoration

## Current target

Class 0 is `candidate_opening`: a visible physical free-space aperture through a wall, partition, door frame, gate or broken wall connecting the current space to another space. It does not mean the drone fits.

Include open/doorless passages, narrow or distant visible openings, partial image-edge openings and partially obstructed apertures. Follow visible free-space boundaries, exclude obstacles, and keep nested openings separate. Exclude windows, closed doors/glass, mirrors, screens, shadows, cabinets, shelves and furniture gaps.

An image without a valid instance receives an empty TXT. One YOLO instance row is `0 x1 y1 ... xn yn`, with x/width and y/height in [0,1]. For example, x=400 in an 800-wide image becomes 0.5. Format validity does not establish semantic correctness. A simple contour cannot exactly represent arbitrary holes/disconnected components.

## Historical V3 artifacts (not distributed)

```text
datasets/
  v3_synthetic_raw/images/                  # 132 audited raw images
  v3_synthetic_audit/
    inventory.json                        # source paths, hashes, dimensions
    dataset_audit.csv                      # selection and scene groups
  v3_synthetic_clean/images/               # 112 KEEP copies
  v3_annotation_pilot/                     # 15 user-approved pilot images
  v3_synthetic_annotated/
    images/                               # 112 unchanged copies
    labels/                               # 110 positive, 2 negative; 244 instances
    annotation_report.csv                 # filename, count, confidence, review, approval, group
    source_manifest.csv
  v3_training_baseline/                    # generated only once by train_v3.py --prepare
```

The annotation report fields used by training are `filename`, `number_of_openings`, `annotation_confidence`, `review_required`, `notes`, `human_approved`, `audit_index`, `scene_group_id`. Values `yes`, `true` or `1` in `review_required` quarantine the image. Missing approval is not created by training. Fifteen pilot images were approved; other accepted-for-experiment labels remain provisional.

For the exact historical recipe, 75 eligible images are reserved by group: modern_rooms -> val8; pipe -> test9; all other eligible groups -> train58. The exporter rejects unexpected counts and refuses existing destinations. It records image and label hashes and checks group/hash separation. Restore authorized artifacts in the documented locations to reproduce; otherwise do not run the historical recipe. Counts and group rules are deliberately tied to that experiment, not a general dataset splitter.

The external test additionally requires `annotation_full_v1/reviews`, `records`, the referenced original image files, and `v3_pilot/frames_manifest.csv` plus frames. Records contain machine-local absolute paths and must be restored/remapped on a new machine. The full external test is not runnable from a source-only clone without those assets.

## Existing data risks

- Older door/window/traversable labels do not all match today's class definition.
- Review found windows marked positive and apparent open doorways marked negative in older records.
- Do not bulk-import old pseudo-labels or regard no detection as a true negative.
- Keep scene/video/derived copies together across splits. Audited V3 group IDs are conservative visual families, not proven generator identity.
- V3 training used just one negative image; hard negatives and real-camera data need expansion.

No dataset downloads, licensing claims, automatic synthetic samples or empty placeholder training data are supplied by the repository.
