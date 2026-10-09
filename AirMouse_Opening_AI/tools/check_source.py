"""Parse release source without importing scripts that train or open cameras."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
files = list(ROOT.glob('*.py'))
for folder in ('annotation_tools', 'tests', 'tools'):
    files.extend((ROOT/folder).glob('*.py'))
for path in files:
    ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path.relative_to(ROOT)))
print(f'PASS: parsed {len(files)} Python files without executing workflows.')
