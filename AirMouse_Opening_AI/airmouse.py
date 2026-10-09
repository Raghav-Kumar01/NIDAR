"""Portable, local-file inference for the experimental candidate-opening model."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}


def existing_file(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f'Local file not found: {path}')
    return path


def input_images(source: str) -> list[Path]:
    path = Path(source).expanduser().resolve()
    files = [path] if path.is_file() else sorted(path.iterdir()) if path.is_dir() else []
    files = [p for p in files if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if not files:
        raise ValueError('Source must be a local image or a directory containing supported images.')
    return files


def doctor() -> int:
    report: dict = {'python': sys.version.split()[0], 'dependencies': {},
                    'model_status': 'experimental; V3 failed external semantic release checks'}
    missing = []
    for name in ('ultralytics', 'torch', 'torchvision', 'opencv-python', 'numpy', 'Pillow', 'PyYAML'):
        try:
            report['dependencies'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
    report['missing'] = missing
    try:
        import torch
        report['cuda_available'] = torch.cuda.is_available()
        report['device'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    except ImportError:
        report['cuda_available'] = False
    print(json.dumps(report, indent=2))
    return int(bool(missing))


def predict(args: argparse.Namespace) -> int:
    # Validate local paths before importing YOLO; never silently download a missing checkpoint.
    model_path = existing_file(args.model)
    sources = input_images(args.source)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise ValueError(f'Output already exists; choose a new folder: {output}')
    if not 0 < args.conf < 1:
        raise ValueError('Confidence must be between 0 and 1, exclusive.')
    if args.imgsz < 32 or args.imgsz % 32:
        raise ValueError('Image size must be a positive multiple of 32 (at least 32).')
    import cv2
    import numpy as np
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    if model.task != 'segment' or model.names != {0: 'candidate_opening'}:
        raise ValueError('Expected a segmentation checkpoint with class 0 = candidate_opening. Legacy models have different semantics.')
    output.mkdir(parents=True)
    print('EXPERIMENTAL: output is an opening proposal, not a clearance or flight-safety decision.')
    rows = []
    for i, path in enumerate(sources):
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f'Unreadable image: {path.name}')
        result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        instances = []
        if len(result.boxes):
            if result.masks is None or len(result.masks) != len(result.boxes):
                raise ValueError('Model returned detections without matching masks.')
            for conf, polygon in zip(result.boxes.conf.cpu().tolist(), result.masks.xyn):
                if not np.isfinite(polygon).all() or not ((polygon >= 0) & (polygon <= 1)).all():
                    raise ValueError('Model returned invalid normalized mask coordinates.')
                instances.append({'class_id': 0, 'confidence': conf, 'polygon_normalized': polygon.tolist()})
        preview = f'{i:05d}_{path.stem}.jpg'
        if not cv2.imwrite(str(output / preview), result.plot()):
            raise OSError('Could not save prediction preview.')
        rows.append({'filename': path.name, 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                     'preview': preview, 'instances': instances})
    report = {'status': 'experimental_predictions', 'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
              'class_name': 'candidate_opening', 'conf': args.conf, 'imgsz': args.imgsz, 'images': rows}
    (output / 'predictions.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Saved {len(rows)} image predictions to {output}')
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('doctor', help='Report installed dependencies and CUDA availability.')
    p = commands.add_parser('predict', help='Predict on local images using an explicitly supplied checkpoint.')
    p.add_argument('--model', required=True, help='Trusted local .pt checkpoint; never use untrusted pickle weights.')
    p.add_argument('--source', required=True, help='Local image or nonrecursive image directory.')
    p.add_argument('--output', required=True, help='New output directory; existing outputs are never overwritten.')
    p.add_argument('--device', default='cpu', help='cpu or CUDA device index, e.g. 0.')
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--imgsz', type=int, default=640)
    args = parser.parse_args(argv)
    try:
        return doctor() if args.command == 'doctor' else predict(args)
    except (ValueError, OSError, ImportError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
