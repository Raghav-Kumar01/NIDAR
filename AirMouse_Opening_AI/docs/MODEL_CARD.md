# Model card: V3 experimental baseline

**Decision: DO NOT PROMOTE TO LIVE APPLICATION.** Local training completed 2026-10-09; external semantic testing failed. This is not a deployed autonomy model.

| Item | Observed value |
|---|---|
| Architecture | YOLO11n instance segmentation |
| Target | 0 = candidate_opening |
| Initialization | Generic pretrained yolo11n-seg.pt |
| Split | 58 train / 8 val / 9 test; 37 quarantined |
| Training | 80 epochs, best logged fitness epoch 77 |
| GPU | RTX 4060 Laptop, 8 GB |
| Training-call duration | 261.865 seconds including setup/final validation |
| Best checkpoint bytes | 5,998,308 |
| SHA-256 | af4fc589a206cf2aaab7f4fb38ca396769a468f36eb1fa00544dddb591e2944b |

AdamW lr0=0.001, cosine decay, batch8, imgsz640, seed42, patience20. Separate nested masks (`overlap_mask=False`), mask_ratio2, no mosaic/mixup/copy-paste. See `train_v3.py` for all settings. Exact GPU determinism across other platforms is not established.

## Measured synthetic held-out results

Nine pipe-environment images, eleven annotated instances, zero negative images:

| Mask metric | Value |
|---|---:|
| Precision | 0.994792 |
| Recall | 0.727273 |
| mAP50 | 0.778674 |
| mAP50-95 | 0.630570 |

These measure agreement with provisional synthetic reference masks. Validation used eight modern-room images; model selection did not use the test. The holdouts are tiny environment families, not a real-world benchmark. Checkpoint loading and fast inference do not imply semantic suitability.

## External findings

Tested 50 historical-reviewed images and 87 recorded frames; inspected 137 source/prediction pairs. Screens received opening confidences up to 0.942; windows and a wall poster also triggered. Several clear real doorways were missed at 0.25 and 0.15 confidence. Historical labels conflict with current semantics, so their aggregate comparison is not accepted as verified real accuracy.

Laptop batch-one timing after ten warmups and 100 synchronized calls: median9.595 ms, p95 13.696 ms, mean10.198 ms. Includes preprocessing/inference/postprocessing, excludes camera/decode/render. Target-drone latency, calibration, physical clearance and flight outcomes are unmeasured.

## Intended use and limits

Use for controlled research and failure reproduction. RGB opening masks propose candidates; stereo geometry must estimate clearance. No live-controller integration, drone testing or deployment approval is provided. Weights are intentionally excluded from Git; request/use only authorized trusted local artifacts.
