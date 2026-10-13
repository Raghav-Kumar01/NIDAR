"""Fetch the pinned research artifacts without installing or executing models."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '0d112b4b1bb28e8e1511a9e994a3de78e3182356'
SHA256 = 'af4fc589a206cf2aaab7f4fb38ca396769a468f36eb1fa00544dddb591e2944b'
URL = 'https://github.com/Raghav-Kumar01/NIDAR/releases/download/airmouse-v0.1.0-research/candidate_opening.pt'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup(device='cpu'):
    model = ROOT / 'weights/candidate_opening.pt'
    model.parent.mkdir(exist_ok=True)
    if not model.exists():
        temporary = model.with_suffix('.download')
        try:
            with urllib.request.urlopen(URL, timeout=120) as response, temporary.open('wb') as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            if digest(temporary) != SHA256:
                raise ValueError('Downloaded opening model checksum mismatch')
            temporary.replace(model)
        finally:
            temporary.unlink(missing_ok=True)
    if digest(model) != SHA256:
        raise ValueError('Existing opening model differs from published baseline; it was not overwritten')
    repo = ROOT / '.external/nidar-human'
    repo.parent.mkdir(exist_ok=True)
    if not repo.exists():
        subprocess.run(['git', 'clone', 'https://github.com/RajGoyal01/NIDAR.git', str(repo)], check=True)
        subprocess.run(['git', '-C', str(repo), 'checkout', '--detach', COMMIT], check=True)
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    if revision != COMMIT or dirty:
        raise ValueError('Existing human checkout is modified or has another revision; it was not reset')
    ml = repo / 'Raj Gupta/ML'
    manifest = json.loads((ml / 'models/manifest.json').read_text())
    entries = {entry['path']: entry for entry in manifest['models']}
    for name in ('models/yolo11n.pt', 'models/nidar-person-posture-specialist-yolo11n.pt', 'models/osnet_x0_25_msmt17.pth'):
        if digest(ml / name) != entries[name]['sha256']:
            raise ValueError('Human checkpoint checksum mismatch: ' + name)
    config = json.loads((ROOT / 'config/perception.example.json').read_text())
    config['device'] = device
    target = ROOT / 'config/perception.local.json'
    if target.exists():
        print('Preserved existing local config; check its paths/device before running.')
    else:
        target.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    print('ARTIFACT CHECKS PASSED. This verifies files, not model accuracy or hardware inference.')
    print('Config:', target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', '0'], default='cpu')
    args = parser.parse_args()
    setup(args.device)
