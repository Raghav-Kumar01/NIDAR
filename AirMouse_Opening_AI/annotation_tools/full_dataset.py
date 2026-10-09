"""Local annotation proposals and explicit review. Never trains or edits sources."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import secrets
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

PROJECT = Path(__file__).resolve().parents[1]
EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(polygons: list) -> None:
    """Only class-0 polygons with finite normalized coordinates and nonzero area."""
    if not isinstance(polygons, list):
        raise ValueError('Polygons must be a list')
    for poly in polygons:
        if not isinstance(poly, list) or len(poly) < 3:
            raise ValueError('Each polygon needs at least three vertices')
        for point in poly:
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError('Each vertex needs x and y')
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                   not math.isfinite(v) or not 0 <= v <= 1 for v in point):
                raise ValueError('Coordinates must be finite values in [0, 1]')
        area = abs(sum(a[0]*b[1]-b[0]*a[1]
                       for a, b in zip(poly, poly[1:] + poly[:1]))) / 2
        if area < 1e-8:
            raise ValueError('Polygon has zero area')


def read_label(path: Path) -> list:
    polygons = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        values = list(map(float, line.split()))
        if values[0] != 0 or len(values) < 7 or (len(values)-1) % 2:
            raise ValueError('Expected class-0 segmentation polygon')
        polygons.append([values[i:i+2] for i in range(1, len(values), 2)])
    validate(polygons)
    return polygons


def yolo_text(polygons: list) -> str:
    validate(polygons)
    return ''.join('0 ' + ' '.join(f'{v:.8f}' for p in poly for v in p) + '\n'
                   for poly in polygons)


def write_json(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def inventory(raw: Path, out: Path) -> list:
    from PIL import Image
    if out.exists():
        raise ValueError('Output exists. Use predict/review to resume; never rebuild in place.')
    if not raw.is_dir():
        raise ValueError(f'Raw directory missing: {raw}')
    curated = PROJECT/'datasets/curated_v2'
    with (curated/'provenance.csv').open(newline='', encoding='utf-8') as f:
        provenance = list(csv.DictReader(f))
    by_hash = {r['sha256']: r for r in provenance}
    items = []
    excluded = Counter()
    for p in sorted(raw.rglob('*')):
        if not p.is_file() or p.suffix.lower() not in EXTS:
            continue
        rel = p.relative_to(raw)
        if 'Depth' in rel.parts:
            excluded['depth_maps'] += 1
            continue
        if 'Annotations' in rel.parts:
            excluded['source_masks'] += 1
            continue
        label = p.parent.parent/'labels'/f'{p.stem}.txt'
        if rel.parts[0] == '04_door_window_seg':
            label = p.parent.parent/'Annotations'/p.name
        items.append((p, rel.parts[0], label if label.exists() else None, None))
    legacy = PROJECT/'datasets/airmouse_openings'
    for p in sorted((legacy/'images').rglob('*')):
        if p.is_file() and p.suffix.lower() in EXTS:
            label = legacy/'labels'/p.parent.name/f'{p.stem}.txt'
            items.append((p, 'MonoRaceGate', label, label))
    for p in sorted((PROJECT/'datasets/v3_pilot/frames').glob('*')):
        if p.is_file() and p.suffix.lower() in EXTS:
            items.append((p, 'v3_pilot', None, None))
    for p in sorted((curated/'annotations_work/batch1_export').rglob('*')):
        if p.is_file() and p.suffix.lower() in EXTS and p.parent.name == 'images':
            label = p.parent.parent/'labels'/f'{p.stem}.txt'
            items.append((p, 'batch1_export', label, label))
    records = {}
    for index, (p, source, source_label, target_label) in enumerate(items):
        sha = digest(p)
        alias = {'path': str(p), 'source': source,
                 'source_annotation': str(source_label) if source_label else ''}
        if sha in records:
            records[sha]['aliases'].append(alias)
            continue
        row = by_hash.get(sha, {})
        with Image.open(p) as image:
            image.load()
            width, height = image.size
        rec = {'id': sha, 'sha256': sha, 'image': str(p), 'source': source,
               'width': width, 'height': height, 'aliases': [alias],
               'prior_category': row.get('category', ''),
               'proposal_origin': '', 'proposal_state': 'pending',
               'review_status': 'unreviewed', 'polygons': [], 'confidence': [],
               'notes': [], 'split': 'unassigned'}
        if source == 'v3_pilot':
            rec['group_hint'] = p.name.split('_mp4_')[0]
        else:
            # No unsupported claim that this is a verified scene/video grouping.
            rec['group_hint'] = 'requires_source_scene_grouping'
        if row.get('category') == 'negative':
            rec.update(proposal_origin='prior_curated_negative', proposal_state='existing_label')
        elif row.get('category') in {'rejected', 'ambiguous'}:
            rec.update(proposal_origin='prior_curated_' + row['category'],
                       proposal_state='prior_review_hold')
        elif target_label and target_label.exists():
            try:
                poly = read_label(target_label)
                if not poly:
                    raise ValueError('Empty source label is not verified absence')
                rec.update(polygons=poly, proposal_origin=str(target_label),
                           proposal_state='existing_label')
            except ValueError as exc:
                rec['notes'].append(f'Source label needs repair: {exc}')
        records[sha] = rec
        if (index+1) % 1000 == 0:
            print(f'Inventoried {index+1}/{len(items)} RGB files', flush=True)
    out.mkdir(parents=True)
    for d in ['records', 'proposal_labels', 'reviewed_labels', 'reviews']:
        (out/d).mkdir()
    for rec in records.values():
        write_json(out/'records'/f"{rec['id']}.json", rec)
        if rec['proposal_state'] == 'existing_label':
            (out/'proposal_labels'/f"{rec['id']}.txt").write_text(yolo_text(rec['polygons']), encoding='utf-8')
    summary = {'rgb_files_considered': len(items), 'unique_exact_file_hashes': len(records),
               'duplicate_file_copies': len(items)-len(records), 'excluded_assets': dict(excluded),
               'unique_by_source': dict(Counter(r['source'] for r in records.values())),
               'target': '0 = visible free-space opening; physical flight safety is not an image label',
               'all_splits_unassigned': True, 'training_ready': False}
    write_json(out/'inventory.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    return list(records.values())


def predict(out: Path, model_path: Path, batch: int, device: str) -> None:
    import torch
    import ultralytics
    from ultralytics import YOLO
    records = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((out/'records').glob('*.json'))]
    pending = [r for r in records if r['proposal_state'] in {'pending', 'inference_error'}]
    if not model_path.is_file():
        raise ValueError('Checkpoint missing; no model download will be attempted')
    config = {'checkpoint': str(model_path.resolve()), 'sha256': digest(model_path),
              'imgsz': 640, 'conf': 0.15, 'iou': 0.5, 'batch': batch,
              'device': device, 'torch': torch.__version__, 'ultralytics': ultralytics.__version__,
              'purpose': 'UNVERIFIED proposal generation, never ground truth',
              'confidence_is_not_correctness': True}
    cfg = out/'inference_config.json'
    if cfg.exists() and json.loads(cfg.read_text(encoding='utf-8')) != config:
        raise ValueError('Resume configuration differs; use a separate output package')
    write_json(cfg, config)
    model = YOLO(str(model_path))
    if model.task != 'segment' or model.names != {0: 'traversable_opening'}:
        raise ValueError(f'Wrong task/classes: {model.task}, {model.names}')
    start = time.perf_counter()
    for offset in range(0, len(pending), batch):
        chunk = pending[offset:offset+batch]
        # A failed inference batch stops; successful preceding batches are resumable.
        results = model.predict([r['image'] for r in chunk], imgsz=640, conf=0.15,
                                iou=0.5, device=device, batch=batch, verbose=False,
                                save=False, save_txt=False, stream=False)
        if len(results) != len(chunk):
            raise RuntimeError('Inference output count mismatch')
        for rec, result in zip(chunk, results):
            polygons, confidences = [], []
            if result.masks is not None:
                for contour, box in zip(result.masks.xyn, result.boxes):
                    poly = contour.tolist()
                    try:
                        validate([poly])
                    except ValueError as exc:
                        rec['notes'].append(f'Discarded invalid model polygon: {exc}')
                        continue
                    polygons.append(poly)
                    confidences.append(float(box.conf.item()))
            rec.update(polygons=polygons, confidence=confidences,
                       proposal_origin='model_v2b_unverified',
                       proposal_state='predicted' if polygons else 'no_prediction_requires_review')
            write_json(out/'records'/f"{rec['id']}.json", rec)
            if polygons:
                (out/'proposal_labels'/f"{rec['id']}.txt").write_text(yolo_text(polygons), encoding='utf-8')
        done = offset + len(chunk)
        if offset == 0 or done % (batch*25) == 0 or done == len(pending):
            print(f'Proposals processed {done}/{len(pending)}; elapsed {time.perf_counter()-start:.1f}s', flush=True)
    write_json(out/'last_run.json', {'processed': len(pending), 'elapsed_seconds': time.perf_counter()-start})


def review_payload(status: str, polygons: list) -> None:
    if status not in {'positive', 'negative', 'ambiguous', 'reject'}:
        raise ValueError('Choose a review decision')
    validate(polygons)
    if status == 'positive' and not polygons:
        raise ValueError('Positive needs at least one polygon')
    if status != 'positive' and polygons:
        raise ValueError('Clear polygons for negative/ambiguous/reject')


def serve(out: Path, port: int) -> None:
    token = secrets.token_urlsafe(32)
    records = {p.stem: json.loads(p.read_text(encoding='utf-8')) for p in (out/'records').glob('*.json')}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, data: bytes, mime='application/json', code=200):
            self.send_response(code)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == '/':
                html = (Path(__file__).with_name('review.html')).read_text(encoding='utf-8')
                self.respond(html.replace('__TOKEN__', token).encode(), 'text/html; charset=utf-8')
            elif path == '/queue':
                current = [json.loads((out/'records'/f'{key}.json').read_text(encoding='utf-8'))
                           for key in records]
                self.respond(json.dumps([{'id': r['id'], 'source': r['source'],
                                         'state': r['proposal_state']} for r in current
                                        if not (out/'reviews'/f"{r['id']}.json").exists()]).encode())
            elif path.startswith('/record/') and path.split('/')[-1] in records:
                self.respond((out/'records'/f"{path.split('/')[-1]}.json").read_bytes())
            elif path.startswith('/image/') and path.split('/')[-1] in records:
                p = Path(records[path.split('/')[-1]]['image'])
                self.respond(p.read_bytes(), 'image/png' if p.suffix.lower() == '.png' else 'image/jpeg')
            else:
                self.respond(b'{}', code=404)

        def do_POST(self):
            if self.path != '/review' or self.headers.get('X-Review-Token') != token:
                self.respond(b'{"error":"Unauthorized"}', code=403)
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size < 2000000:
                    raise ValueError('Invalid payload size')
                data = json.loads(self.rfile.read(size))
                rec = records[data['id']]
                status, polygons = data['status'], data['polygons']
                review_payload(status, polygons)
                if digest(Path(rec['image'])) != rec['sha256']:
                    raise ValueError('Image changed after inventory')
                target = out/'reviews'/f"{rec['id']}.json"
                if target.exists():
                    raise ValueError('Review already exists; overwrite is disabled')
                label = out/'reviewed_labels'/f"{rec['id']}.txt"
                if status in {'positive', 'negative'}:
                    label.write_text(yolo_text(polygons), encoding='utf-8')
                write_json(target, {'id': rec['id'], 'status': status, 'polygons': polygons,
                                    'reviewed_utc_epoch': time.time(), 'method': 'explicit_browser_review',
                                    'split': 'unassigned', 'image_sha256': rec['sha256']})
                self.respond(b'{"ok":true}')
            except (ValueError, KeyError, TypeError) as exc:
                self.respond(json.dumps({'error': str(exc)}).encode(), code=400)
    print(f'Annotation review: http://127.0.0.1:{port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()


def report(out: Path) -> dict:
    records = [json.loads(p.read_text(encoding='utf-8')) for p in (out/'records').glob('*.json')]
    reviews = [json.loads(p.read_text(encoding='utf-8')) for p in (out/'reviews').glob('*.json')]
    methods = Counter(r.get('method', 'unknown') for r in reviews)
    reviewed_ids = {r['id'] for r in reviews}
    by_method = {}
    for review in reviews:
        method = review.get('method', 'unknown')
        by_method.setdefault(method, Counter())[review['status']] += 1
    result = {'records': len(records),
              'proposal_states': dict(Counter(r['proposal_state'] for r in records)),
              'proposal_polygon_rows': sum(len(r['polygons']) for r in records),
              'saved_decisions': len(reviews),
              'pending_decisions': sum(r['id'] not in reviewed_ids for r in records),
              'explicit_reviews': methods['explicit_browser_review'] + methods['ai_visual_review'],
              'human_browser_reviews': methods['explicit_browser_review'],
              'ai_visual_reviews': methods['ai_visual_review'],
              'decision_methods': dict(methods),
              'statuses_by_method': {k: dict(v) for k, v in by_method.items()},
              'decision_statuses': dict(Counter(r['status'] for r in reviews)),
              'candidate_label_files': len(list((out/'reviewed_labels').glob('*.txt'))),
              'pending_by_source': dict(Counter(r['source'] for r in records if r['id'] not in reviewed_ids)),
              'semantic_accuracy_measured': False,
              'train_val_test_split': 'UNASSIGNED: requires scene/video grouping and holdout review',
              'training_dataset_exported': False}
    write_json(out/'status.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build', 'predict', 'review', 'report'])
    parser.add_argument('--raw', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=PROJECT/'runs/airmouse/opening_seg_v2b/weights/best.pt')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--device', default='0')
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    if args.action == 'build':
        if args.raw is None:
            parser.error('build requires --raw')
        inventory(args.raw.resolve(), args.output.resolve())
    elif not (args.output/'inventory.json').exists():
        parser.error('Output is not an existing annotation package')
    elif args.action == 'predict':
        predict(args.output, args.model, args.batch, args.device)
        report(args.output)
    elif args.action == 'review':
        serve(args.output, args.port)
    else:
        report(args.output)


if __name__ == '__main__':
    main()
