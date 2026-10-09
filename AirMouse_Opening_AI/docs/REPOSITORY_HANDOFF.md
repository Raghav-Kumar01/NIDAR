> Historical preparation notes from the standalone workspace. For this published team repository, follow [TEAM_SETUP.md](TEAM_SETUP.md). The code is now nested under AirMouse_Opening_AI and the research checkpoint is distributed through GitHub Releases.

# Repository preparation phase report

Date: 2026-10-09.

**Phase:** Source-only GitHub research release preparation.

**Status:** COMPLETE for local source packaging and verification. No remote, commit or push has been created. The V3 model remains rejected for live promotion.

**Project completion:** Repository preparation is complete for the documented source-only scope. Overall drone/model readiness is not assigned an invented percentage; semantic model failures remain open.

## 1. What we built

Completed: local Git repository on main; explicit staged-file manifest; ignores for local data/models/secrets; portable image inference CLI; dependency instructions and observed versions; model/data/release documentation; local checks and a hosted CI workflow definition.

Experimental: V3 candidate-opening model, with its failure evidence preserved in public text summaries.

Not implemented: model retraining/improvement, data/weight distribution, fresh-machine dependency installation, remote GitHub CI execution, remote repository creation or push.

## 2. Important files created

- `README.md`: entry point, scope, Windows setup, inference, model status and legacy caveats.
- `airmouse.py`: `doctor()` reports environment; `predict()` validates explicit local paths, verifies the class/task contract, writes previews and normalized polygon JSON in a new destination. No missing-checkpoint auto-download or overwrite. Inputs: trusted weights and local images. Outputs: preview images plus provenance/prediction JSON.
- `requirements.txt`, `requirements-observed.txt`: dependency bounds and observed direct versions. Neither is a complete transitive cross-platform lock.
- `.gitignore`, `.gitattributes`: exclusions and source text normalization.
- `tests/test_cli.py`: four meaningful boundary tests for missing input, nonrecursive filtering, empty sources and preserving existing output.
- `tools/check_source.py`: parse-only checks avoid accidentally running top-level camera/training scripts.
- `tools/check_release.py`: audits exact indexed bytes for forbidden assets, large files and common credential patterns.
- `.github/workflows/checks.yml`: source, unit and index checks on push/PR. No model training or external assets in CI.
- `docs/`: dataset contract, model card, adverse findings, third-party status, GitHub commands and this report.
- `release_manifest.txt`: exact files selected for staging, including this report.

Why this design: preserve historical workflows, add one explicit portable entry point, and keep heavy/private artifacts outside source control. No new application architecture or model algorithm was introduced.

Interview explanation: "The repository contains reproducible code and honest measured evidence; authorized data and weights are supplied separately. Source release readiness does not imply model deployment readiness."

## 3. Existing files modified

- `annotation_tools/README.md`: clearly marks its historical window semantics as incompatible with the current V3 definition without changing old review records.
- `prepare_yolo_dataset.py`: removed trailing spaces from one print statement to pass Git whitespace checks; no behavior change.
- Existing source images, model binaries, training results and application model paths were preserved.

## 4. Tests performed

| Test | Expected | Actual | Result |
|---|---|---|---|
| Parse source safely | No syntax errors | 39 Python files | PASS |
| Existing annotation tests | All pass | 13/13 | PASS |
| New CLI boundary tests | All pass | 4/4 | PASS |
| Environment doctor | Dependencies reported, CUDA detected locally | No missing direct dependencies; RTX 4060 Laptop | PASS |
| Local CLI inference | Explicit trusted checkpoint/image, new output | Preview and JSON created | PASS |
| Source-only export | Tests/CLI independent of original working directory | All checks and one GPU inference passed | PASS |
| Staged whitespace | No errors | One legacy trailing-space issue fixed; clean afterward | PASS |
| Staged content | No data/weights/environment/credential-pattern matches | Source/documentation only | PASS bounded scan |
| Ignore coverage | Local artifacts excluded | .venv, datasets, runs, weights, pretrained .pt ignored | PASS |

Source-only testing exported the Git index and used the existing Python environment, with absolute paths to external local assets for the inference smoke test. It is not a fresh dependency-install test or hosted CI result. Logs are retained locally in ignored `release_checks/verification.json`.

## 5. Actual results

17 unit tests pass, 39 source files parse, portable inference runs from the source-only export, and the exact staged release is checked. Model accuracy was not improved or remeasured in this packaging phase. Previous external semantic failures remain documented.

## 6. What you should have learned

Easy: GitHub stores the project's instructions and code. Large research inputs and private machine files stay local. A working upload does not fix a model's mistakes.

Technical: Git stages a snapshot of file contents in its index. The release checker reads that snapshot with `git show :path`, so checking working files alone cannot hide a previously staged forbidden artifact. Ignore rules affect untracked files; they do not remove already tracked content. Exact staging and an index audit are both needed.

## 7. Questions you should be able to answer

- What is the public entry point? `airmouse.py doctor` and `airmouse.py predict`.
- Why are weights absent? They are large external artifacts with separate provenance/rights and a failed live-use verdict.
- Can a fresh clone train immediately? No. Restore authorized datasets/manifests and weights first; the exact historical recipe has explicit data requirements.
- Did GitHub CI pass? Not yet; only local equivalent checks ran.
- Is the repository permissively licensed? No project-wide license was selected by the owner; dependency and asset terms still apply.

## 8. Current pipeline

```text
Existing research source -> portable CLI + documentation
    -> unit/source checks -> exact Git index -> source-only export smoke
    -> owner commit -> owner chooses GitHub destination -> push -> hosted CI
```

The last four steps remain user actions; they were not executed by this phase.

## 9. What remains

Create the initial commit, choose/create the GitHub destination and visibility, push and inspect hosted CI. Select a project license if granting reuse rights. Model improvement and real-world validation remain separate work.

## 10. Next recommended phase

Publish this documented research repository using `GITHUB_RELEASE.md`, then perform the planned data correction/model-improvement phase in a new versioned experiment. Do not promote V3 to the drone application.

## 11. Risks and limitations

Legacy scripts contain local paths and immediate side effects; read before running. Training/evaluation assets are excluded and not auto-provisioned. The dependency set was tested in the existing environment, not on a newly provisioned machine. The credential scan is heuristic. No blanket asset redistribution permission or project license is inferred.

## 12. Git checkpoint

Current branch: `main` (new local repository). Selected source/docs are staged; no remote or commit yet.

Suggested commit: `Prepare AirMouse research repository with reproducible checks and V3 findings`.

The audited staged set is ready for this source-only commit. Rerun `git diff --cached --check` and `python tools/check_release.py` if staging changes before commit. Do not use force-push or bulk-add local artifacts.
