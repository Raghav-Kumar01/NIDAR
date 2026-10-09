# External release findings, 2026-10-09

The V3 checkpoint failed semantic release review. The local run artifacts are retained under ignored `runs/airmouse/v3_release_review`; this source-only summary carries the adverse results without republishing external images or machine-local records.

- 137 external inputs processed: 50 historical-reviewed images and 87 correlated frames from three recordings.
- Model hash matched the training report. All tested masks were finite and normalized; class mapping was correct.
- 75 frozen split image/label pairs matched their hashes. No exact overlap with external inputs; no near-duplicate candidates at 64-bit pHash Hamming distance <=6. This heuristic does not prove scene independence.
- Laptop-screen false positives included confidences 0.889, 0.932, 0.942 and 0.770. A window was predicted at 0.648. These are scores, not calibrated correctness.
- Seven documented clear doorway/passage examples were missed at both application thresholds (0.25 and 0.15). This selected failure subset is not a random recall estimate.
- Recorded frame output counts: 10/35, 0/17 and 2/35 frames with predictions. These are not recall because trusted frame-level ground truth is absent. A wall poster was falsely selected.
- Old records had windows/closed doors marked positive and apparent open doorways marked negative. The apparent historical TP was a window. Do not use that comparison as current-class accuracy.
- Darkening, Gaussian blur and JPEG diagnostics on nine synthetic images produced any-detection counts 7/9 in each condition; this is not a mask-robustness certificate.
- No model/data mutation, retraining, live application replacement, remote push or deployment occurred during testing.

## Release distinction

The **code repository** can document a failed experiment honestly and remain useful. The **V3 checkpoint** has not met the requirements for live use. A source release does not reverse that model verdict.

## Next experiment

Adjudicate inconsistent labels, collect/approve representative real openings and hard negatives, and freeze independent scene/video-separated evaluation before a new versioned run. Do not repeatedly tune on these now-exposed challenge examples while presenting them as an unseen test. New training and deployment are separate phases.
