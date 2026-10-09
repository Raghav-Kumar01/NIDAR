"""Check the exact Git index for private artifacts, large files and credential patterns."""
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    paths=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    paths=[p for p in paths if p]
    if not paths:
        raise SystemExit('FAIL: no files in Git index; explicitly stage the release files first.')
    forbidden_dirs={'.venv','venv','datasets','runs','weights','output','release_checks','__pycache__','.idea','.vscode'}
    forbidden_ext={'.pt','.pth','.onnx','.engine','.zip','.mp4','.avi','.mov','.png','.jpg','.jpeg','.pem','.key','.pfx','.pyc'}
    # Pattern values are never printed. This is a bounded heuristic, not a credential-security guarantee.
    patterns=[r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',r'gh[pousr]_[A-Za-z0-9]{30,}',
              r'github_pat_[A-Za-z0-9_]{30,}',r'AKIA[A-Z0-9]{16}',r'sk-[A-Za-z0-9_-]{30,}']
    failures=[];total=0
    for name in paths:
        path=Path(name)
        if any(part in forbidden_dirs for part in path.parts) or path.suffix.lower() in forbidden_ext or path.name.startswith('.env'):
            failures.append((name,'forbidden artifact'))
        data=subprocess.check_output(['git','show',f':./{name}'],cwd=ROOT)
        total+=len(data)
        if len(data)>1_000_000:failures.append((name,'file exceeds source-release size cap'))
        text=data.decode('utf-8',errors='replace')
        if any(re.search(pattern,text) for pattern in patterns):failures.append((name,'possible credential'))
    for name,reason in failures:print(f'FAIL: {name}: {reason}')
    if failures:return 1
    print(f'PASS: {len(paths)} indexed source/documentation files, {total} bytes; no forbidden artifacts or matched credential patterns.')
    return 0


if __name__=='__main__':raise SystemExit(main())
