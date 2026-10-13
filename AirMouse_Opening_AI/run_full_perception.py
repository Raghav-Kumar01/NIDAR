"""Combined openings + Raj's full tracking/Re-ID/mission/dashboard application."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def prepare(config_path):
    from perception import UPSTREAM_COMMIT, digest
    path = config_path.resolve()
    cfg = json.loads(path.read_text(encoding='utf-8'))
    repo = (path.parent / cfg['human_repository']).resolve()
    model = (path.parent / cfg['opening_model']).resolve()
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError('Human repository revision differs from tested pin')
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise ValueError('Human repository contains modified tracked files')
    ml = repo / 'Raj Gupta/ML'
    manifest = json.loads((ml / 'models/manifest.json').read_text())
    required = {'models/yolo11n.pt', 'models/nidar-person-posture-specialist-yolo11n.pt', 'models/osnet_x0_25_msmt17.pth'}
    entries = {row['path']: row for row in manifest['models']}
    for name in required:
        if digest(ml / name) != entries[name]['sha256']:
            raise ValueError('Human model checksum mismatch: ' + name)
    if not model.is_file():
        raise ValueError('Opening checkpoint missing; run tools/setup_team.py')
    sys.path.insert(0, str(ml))
    import nidar_survivor_demo
    if Path(nidar_survivor_demo.__file__).resolve().parent != ml / 'nidar_survivor_demo':
        raise ValueError('Unexpected human package already loaded')
    return cfg, model, ml, {'human_commit': commit, 'human_model_hashes': {n: entries[n]['sha256'] for n in sorted(required)},
                            'opening_sha256': digest(model), 'config': cfg}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog='Remaining options are forwarded to Raj\'s runtime: --source, --device, --seconds, --backend, --dashboard-port, --resume-mission, --mission-db. Dashboard and full identity pipeline are enabled by default.')
    parser.add_argument('--config', type=Path, default=ROOT / 'config/perception.local.json')
    parser.add_argument('--output', type=Path, required=True, help='New output directory; never overwrites an existing run')
    parser.add_argument('--native-preview', action='store_true', help='Show OpenCV preview in addition to the dashboard')
    args, forwarded = parser.parse_known_args(argv)
    cfg, model, ml, provenance = prepare(args.config)
    # Check all CLI constraints before loading weights or creating a mission.
    from nidar_survivor_demo.config import arguments
    runtime_args = ['--dashboard', '--hybrid-detector', '--device', str(cfg.get('device', 'cpu')),
                    '--model', str(ml / 'models/yolo11n.pt'), '--confidence', '0.1',
                    '--mission-db', str(args.output.resolve() / 'mission.sqlite3')]
    if not args.native_preview:
        runtime_args.append('--headless')
    runtime_args += forwarded
    camera, options = arguments(runtime_args)
    from nidar_survivor_demo.tracking import TrackerConfig
    if options.confidence > TrackerConfig.load(options.tracker_config).track_low_thresh:
        raise ValueError('Full tracking requires confidence at or below track_low_thresh')
    if args.output.exists():
        raise FileExistsError('Choose a new --output folder')
    import torch
    torch.set_num_threads(options.cpu_threads)
    expected_models = (ml / 'models/yolo11n.pt', ml / 'models/nidar-person-posture-specialist-yolo11n.pt',
                       ml / 'models/osnet_x0_25_msmt17.pth')
    if tuple(p.resolve() for p in (options.model, options.specialist_model, options.reid_model)) != expected_models:
        raise ValueError('Full release requires verified model paths; model overrides would invalidate provenance')
    from perception import OpeningAdapter
    from full_openings import OpeningHook
    from full_runtime import run
    device = ('0' if torch.cuda.is_available() else 'cpu') if options.device == 'auto' else options.device
    adapter = OpeningAdapter(model, device, cfg['opening_confidence'])
    args.output.mkdir(parents=True)
    provenance.update(runtime_args=runtime_args, opening_status='experimental_release_failed',
                      identity_scope='mission_appearance_estimate', flight_ready=False)
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    with (args.output / 'frames.jsonl').open('x', encoding='utf-8') as output:
        return run(runtime_args, opening_hook=OpeningHook(adapter, output), report_path=args.output / 'mission_report.json')


if __name__ == '__main__':
    raise SystemExit(main())
