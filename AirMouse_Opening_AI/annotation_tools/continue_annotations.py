"""Reuse matching target labels and record individually inspected AI decisions."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from PIL import Image, ImageDraw
from full_dataset import review_payload, yolo_text, validate


def save(root, rec, status, polygons, method, reason, evidence):
    decision = root/'reviews'/f"{rec['id']}.json"
    label = root/'reviewed_labels'/f"{rec['id']}.txt"
    if decision.exists() or label.exists():
        return False
    review_payload(status, polygons)
    if hashlib.sha256(Path(rec['image']).read_bytes()).hexdigest() != rec['sha256']:
        raise ValueError('Image changed after inventory')
    data = {'id': rec['id'], 'status': status, 'polygons': polygons,
            'reviewed_utc_epoch': time.time(), 'method': method,
            'reviewer': 'Codex' if method == 'ai_visual_review' else 'source_annotation_reuse',
            'reason': reason, 'evidence': evidence, 'split': 'unassigned',
            'image_sha256': rec['sha256'], 'independent_human_validation': False}
    with decision.open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    if status in {'positive', 'negative'}:
        with label.open('x', encoding='utf-8') as f:
            f.write(yolo_text(polygons))
    return True


def clip(poly):
    """Sutherland-Hodgman polygon intersection with the normalized image rectangle."""
    for axis, bound, sign in [(0, 0., 1), (0, 1., -1), (1, 0., 1), (1, 1., -1)]:
        result = []
        if not poly:
            return []
        previous = poly[-1]
        for current in poly:
            a = sign*(previous[axis]-bound) >= 0
            b = sign*(current[axis]-bound) >= 0
            if a != b:
                fraction = (bound-previous[axis])/(current[axis]-previous[axis])
                point = [previous[k]+fraction*(current[k]-previous[k]) for k in (0, 1)]
                point[axis] = bound
                result.append(point)
            if b:
                result.append(current)
            previous = current
        poly = result
    # Remove consecutive repeated vertices introduced at image corners.
    cleaned = []
    for p in poly:
        p = [min(1., max(0., x)) for x in p]
        if not cleaned or p != cleaned[-1]:
            cleaned.append(p)
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    return cleaned


def reuse(root):
    counts = Counter()
    for f in sorted((root/'records').glob('*.json')):
        r = json.loads(f.read_text(encoding='utf-8'))
        if (root/'reviews'/f.name).exists():
            continue
        state = r['proposal_state']
        if state == 'existing_label':
            status = 'positive' if r['polygons'] else 'negative'
            if save(root, r, status, r['polygons'], 'existing_target_annotation_import',
                    'Reused existing target annotation or previously reviewed negative. '
                    'No new semantic ground-truth verification is claimed.', r['proposal_origin']):
                counts[f'imported_{status}'] += 1
        elif state == 'prior_review_hold':
            status = 'reject' if r['prior_category'] == 'rejected' else 'ambiguous'
            if save(root, r, status, [], 'existing_curation_decision_import',
                    'Preserved existing curation exclusion; no training label generated.',
                    'curated_v2/provenance.csv'):
                counts[f'imported_{status}'] += 1
        elif r['source'] == 'MonoRaceGate':
            label = Path(r['aliases'][0]['source_annotation'])
            lines = label.read_text(encoding='utf-8').splitlines() if label.is_file() else []
            if not any(line.strip() for line in lines):
                continue  # Absence of source annotation is not a negative label.
            polygons = []
            for line in lines:
                values = list(map(float, line.split()))
                if not values:
                    continue
                if values[0] != 0 or len(values) < 7 or (len(values)-1) % 2:
                    raise ValueError(f'Invalid source gate polygon: {label}')
                polygon = clip([values[i:i+2] for i in range(1, len(values), 2)])
                if len(polygon) >= 3:
                    validate([polygon])
                    polygons.append(polygon)
            if polygons and save(root, r, 'positive', polygons, 'source_gate_polygon_clipped',
                                 'Intersected source inner-gate polygon with image bounds. '
                                 'Source geometric annotation reused; no new semantic review.', str(label)):
                counts['clipped_gate_positive'] += 1
    print(json.dumps(dict(counts), indent=2))


def sheet(root, name, source, count):
    work = root/'ai_annotation_work'
    work.mkdir(exist_ok=True)
    selected = []
    for f in sorted((root/'records').glob('*.json')):
        if (root/'reviews'/f.name).exists():
            continue
        r = json.loads(f.read_text(encoding='utf-8'))
        if not source or r['source'] == source:
            selected.append(r)
        if len(selected) == count:
            break
    if not selected:
        print('No pending records for source')
        return
    columns = 8 if count > 16 else 4
    cell = 192 if count > 16 else 320
    height = cell + 24
    image = Image.new('RGB', (columns*cell, ((len(selected)+columns-1)//columns)*height), (20,25,32))
    draw = ImageDraw.Draw(image)
    for i, r in enumerate(selected):
        with Image.open(r['image']) as original:
            thumb = original.convert('RGB')
            thumb.thumbnail((cell-8, cell-8))
        x,y = i%columns*cell, i//columns*height
        image.paste(thumb, (x,y))
        draw.text((x,y+cell), f'{i}: {r["source"]}', fill='white')
    with (work/f'{name}.json').open('x', encoding='utf-8') as f:
        json.dump(selected, f, indent=2)
    image.save(work/f'{name}.jpg')
    print(str(work/f'{name}.jpg'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['reuse','sheet','save'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--name')
    p.add_argument('--source', default='')
    p.add_argument('--count', type=int, default=32)
    p.add_argument('--decisions', type=Path)
    args = p.parse_args()
    if args.action == 'reuse':
        reuse(args.output)
    elif args.action == 'sheet':
        sheet(args.output, args.name, args.source, args.count)
    else:
        batch = json.loads((args.output/'ai_annotation_work'/f'{args.name}.json').read_text(encoding='utf-8'))
        data = json.loads(args.decisions.read_text(encoding='utf-8'))
        count = Counter()
        for item in data:
            r = batch[item['index']]
            if save(args.output, r, item['status'], item.get('polygons', []),
                    'ai_visual_review', item['reason'], f'{args.name}.jpg'):
                count[item['status']] += 1
        print(json.dumps(dict(count)))


if __name__ == '__main__':
    main()
