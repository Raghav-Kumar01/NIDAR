> Historical preparation notes from the standalone workspace. For this published team repository, follow [TEAM_SETUP.md](TEAM_SETUP.md). The code is now nested under AirMouse_Opening_AI and the research checkpoint is distributed through GitHub Releases.

# GitHub handoff (Windows CMD)

## Prepared scope

Source-only research repository. V3 is explicitly experimental and failed external semantic release checks. No checkpoint is promoted to the application. Raw data, videos, images, weights, local environments, inference outputs and private records remain ignored.

The local repository uses branch `main`. `release_manifest.txt` lists the exact source/documentation files selected for the first commit. `tools/check_release.py` inspects the Git index, not just the working tree, for forbidden artifacts, files over 1 MB and common credential patterns. This bounded scan is not a complete security guarantee.

## Inspect and commit

From this project directory:

```bat
git status --short
git diff --cached --stat
git diff --cached --check
.venv\Scripts\python.exe tools\check_release.py
git commit -m "Prepare AirMouse research repository with reproducible checks and V3 findings"
```

The prepared files are staged. If you modify files afterward, stage only the reviewed paths and rerun checks. Avoid `git add .` for dataset/model projects. If Git asks for identity, configure your own name/email locally; no identity is invented by this preparation.

## Create the GitHub destination and push

Create an **empty** repository in your own GitHub account, with no generated README, license or .gitignore. Choose visibility according to the team's sharing policy. Replace the URL placeholder below with the URL GitHub gives you:

```bat
git remote add origin https://github.com/YOUR_ACCOUNT/AirMouse_Opening_AI.git
git push -u origin main
```

If `origin` already exists, inspect `git remote -v` before making changes. Do not overwrite another destination or force-push. Authentication remains in Git/GitHub's normal credential flow; never paste tokens into source or URLs.

After pushing, inspect the GitHub file list and the **Source checks** workflow result. Local validation is not proof that hosted CI ran. The workflow only validates source and unit tests; it cannot evaluate unpublished datasets, GPU training, live cameras or drone performance.

## What stays local

- `datasets/`, `runs/`, `weights/`, `output/`, `release_checks/`
- `.venv/`, caches, logs and archives
- `.env*`, private keys and local configuration
- pretrained and trained model files, recordings and external-image galleries

Verify examples with:

```bat
git check-ignore .venv datasets runs weights yolo11n-seg.pt
```

Do not attach model files or datasets to a GitHub Release without a separate provenance/rights and artifact review. `docs/THIRD_PARTY.md` records that no owner license choice has been made. This source preparation does not automatically authorize redistribution of external assets.

## What a fresh clone can do

It can read the evidence, run source/unit checks, check dependency availability, and run the portable image CLI when supplied a trusted checkpoint and local images. It cannot train or reproduce external-image accuracy without restoring the separate authorized assets described in `DATASET.md`. Legacy scripts may have local paths or immediate side effects; the top-level README identifies those workflows explicitly.
