"""Prepare a frozen grouped V3 baseline and train/evaluate it without source writes."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform
import shutil
import time

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'datasets/v3_synthetic_annotated'
DATA = ROOT / 'datasets/v3_training_baseline'
RUN = ROOT / 'runs/airmouse/opening_seg_v3_baseline'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    """Copy eligible annotations and reserve entire visual groups before fitting."""
    import yaml
    if DATA.exists():
        raise FileExistsError(f'Refusing to overwrite frozen dataset: {DATA}')
    rows = list(csv.DictReader((SOURCE / 'annotation_report.csv').open(encoding='utf-8-sig')))
    manifest = []
    groups = {'train': set(), 'val': set(), 'test': set()}
    hashes = {'train': set(), 'val': set(), 'test': set()}
    counts = {s: {'images': 0, 'instances': 0, 'negatives': 0} for s in groups}
    for row in rows:
        group = row['scene_group_id']
        reserved = 'val' if group == 'SG_modern_rooms' else 'test' if group == 'SG_pipe' else 'train'
        split = 'quarantine' if row['review_required'].lower() in ('yes', 'true', '1') else reserved
        image = SOURCE / 'images' / row['filename']
        label = SOURCE / 'labels' / (image.stem + '.txt')
        record = dict(row, split=split, reserved_split=reserved,
                      image_sha256=digest(image), label_sha256=digest(label))
        manifest.append(record)
        if split == 'quarantine':
            continue
        for kind, src in [('images', image), ('labels', label)]:
            target = DATA / kind / split / src.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            assert digest(src) == digest(target)
        groups[split].add(group)
        hashes[split].add(record['image_sha256'])
        counts[split]['images'] += 1
        count = len(label.read_text().splitlines())
        counts[split]['instances'] += count
        counts[split]['negatives'] += int(count == 0)
    for a, b in [('train','val'), ('train','test'), ('val','test')]:
        assert not groups[a] & groups[b], 'Scene group leakage'
        assert not hashes[a] & hashes[b], 'Exact duplicate leakage'
    assert [counts[s]['images'] for s in ('train','val','test')] == [58,8,9]
    with (DATA / 'split_manifest.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
        writer.writeheader(); writer.writerows(manifest)
    config = dict(path=DATA.as_posix(), train='images/train', val='images/val', test='images/test', names={0:'candidate_opening'})
    (DATA / 'data.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    report = dict(counts=counts, groups={k:sorted(v) for k,v in groups.items()}, quarantined=37,
                  group_overlap=False, exact_hash_overlap=False,
                  source='V3 synthetic explicit visual annotations; only pilot has user approval',
                  limitation='Visual scene groups are conservative, not verified generator identities. Validation and test are small environment holdouts, not real-world accuracy.')
    (DATA / 'split_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def verify_frozen():
    rows = list(csv.DictReader((DATA / 'split_manifest.csv').open()))
    for row in rows:
        if row['split'] == 'quarantine':
            continue
        for kind, name, key in [('images', row['filename'], 'image_sha256'),
                                ('labels', Path(row['filename']).stem + '.txt', 'label_sha256')]:
            assert digest(DATA / kind / row['split'] / name) == row[key], f'Frozen data changed: {name}'


def train():
    import torch
    import ultralytics
    from ultralytics import YOLO
    assert torch.cuda.is_available(), 'This run requires the verified CUDA GPU'
    if RUN.exists():
        raise FileExistsError(f'Refusing to overwrite run: {RUN}')
    verify_frozen()
    options = dict(data=str(DATA / 'data.yaml'), epochs=80, patience=20, imgsz=640,
        batch=8, device=0, workers=0, seed=42, deterministic=True,
        optimizer='AdamW', lr0=0.001, lrf=0.01, weight_decay=0.0005,
        warmup_epochs=3, cos_lr=True, overlap_mask=False, mask_ratio=2,
        mosaic=0.0, mixup=0.0, copy_paste=0.0, degrees=5.0, translate=0.05,
        scale=0.2, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5,
        hsv_h=0.01, hsv_s=0.3, hsv_v=0.3, amp=True, cache=False,
        project=str(RUN.parent), name=RUN.name, exist_ok=False, plots=True)
    checkpoint = ROOT / 'yolo11n-seg.pt'
    metadata = dict(status='configured', python=platform.python_version(), torch=torch.__version__,
        ultralytics=ultralytics.__version__, gpu=torch.cuda.get_device_name(0),
        initialization=str(checkpoint), initialization_sha256=digest(checkpoint),
        options=options, interpretation='Experimental synthetic-only baseline with provisional labels')
    (DATA / 'experiment.json').write_text(json.dumps(metadata, indent=2))
    start = time.perf_counter()
    model = YOLO(str(checkpoint))
    model.train(**options)
    training_seconds = time.perf_counter() - start
    best = RUN / 'weights/best.pt'
    assert best.exists()
    # Test is accessed only after model selection has finished on validation.
    test = YOLO(str(best)).val(data=str(DATA / 'data.yaml'), split='test', imgsz=640,
        batch=8, device=0, workers=0, overlap_mask=False, mask_ratio=2,
        project=str(RUN), name='heldout_test', plots=True)
    result = dict(status='completed', training_seconds=training_seconds,
        best_checkpoint=str(best), best_sha256=digest(best),
        test_metrics={k:float(v) for k,v in test.results_dict.items()},
        test_speed_ms_per_image={k:float(v) for k,v in test.speed.items()},
        limitations='Nine synthetic pipe-environment test images; provisional labels; no real drone or safety validation.')
    verify_frozen()
    (RUN / 'completion_report.json').write_text(json.dumps(result, indent=2))
    metadata['status'] = 'completed'
    metadata['completion_report'] = str(RUN / 'completion_report.json')
    (DATA / 'experiment.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        train()
