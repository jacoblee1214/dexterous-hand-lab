# Private repository preparation plan — `dexterous-hand-lab`

## Audit result

The project directory was not a Git work tree on 2026-09-08. It therefore had no
branch, remote, commit history or historical secret exposure to inspect. No
credential-like filename or high-confidence key/token/password pattern was found
in the current project files; values were not printed during the audit.

Excluding the virtual environment and test cache, the working tree is about
243 MB: approximately 194 MB of assets and 45 MB of experiments. Generated Python
caches are present locally. Raw grasp-trial reconstructions occupy about 38 MB.
The two largest individual files are 16.7 MB STL files, so no present file exceeds
GitHub's 100 MB per-file limit, but the binary asset collection is still large
and poorly suited to ordinary Git deltas.

Machine-specific absolute user-home paths occur in old, locally preserved
experiment JSON/JSONL metadata and formerly in the README provenance statement.
The README is now portable. Original logs and grasp summaries were not rewritten
because that would alter the historical record; those files and large trial data
are ignored for the proposed snapshot.

## Proposed tracked tree

```text
README.md
requirements*.txt
backend/               Python server, schemas, config and docs
simulation/            source code, object XML and joint/grasp configuration
sensors/               approved right/left mounts and scalar calibration
robot_data/            common state plus simulation/replay/hardware interfaces
reconstruction/        tactile-only algorithm
evaluation/            one-way evaluation code
vision/                calibrated RGB configuration and vision-only baseline
ui/                    dashboard plus licensed vendored Three.js subset
tests/                 scientific, provider and browser integration tests
docs/                  architecture, validation and status documents
experiments/
  reconstruction_config.yaml
  README.md
  vision_baseline/.../{experiment_config,camera_calibration,
                       algorithm_summary,evaluation_summary}.json
  vision_baseline/.../algorithm/*/result.json
  vision_baseline/.../evaluation/*/metrics.json
```

The source-controlled experiment portion keeps definitions, calibration versions,
small per-condition outputs and summaries. It excludes raw recordings, large
point-by-point trial files and rendered evidence; all remain untouched locally.

Generated `simulation/hand.xml`, `simulation/hand_left.xml` and `mjmodel.xml` are
ignored because `python -m simulation.model_builder --hand right|left` recreates
them from source assets/configuration.

## Ignored local content

The root `.gitignore` covers virtual environments, Python/test/editor caches,
local `.env`/credential/key files, generated MJCF/runtime data, log/frame/video
caches, raw JSONL streams, experiment images, trial reconstruction/evaluation
payloads and configuration backup directories. It does **not** broadly ignore
all experiments, YAML, calibration or assets. Existing ignored files are not
deleted.

## Asset ownership and license review — excluded from first snapshot

| Asset group | Size | Current evidence | Recommendation |
| --- | ---: | --- | --- |
| `assets/V6_Force_R` | 28 MB | `package.xml` says BSD; author/maintainer are TODO; no license text | Obtain owner/source and redistribution confirmation |
| `assets/V6_Force_L` | 27 MB | Same incomplete package metadata; no license text | Obtain owner/source and redistribution confirmation |
| `assets/left_hand` | 117 MB | Legacy URDF/STLs; no included license/notice | Do not upload until explicitly authorized |
| `assets/meshes`, `assets/urdf` | 23 MB | Apparent older/duplicate right-hand copies; provenance not documented | Keep local; decide canonical status after ownership review |
| `ui/vendor/three` | 2.1 MB | Three.js 0.180.0, MIT license included | Safe to track with `LICENSE` and `README.md` preserved |

The first private snapshot excludes every asset group in this table because
ownership/license approval has not been supplied. The files remain unchanged in
the local working directory. The source code currently uses
`assets/V6_Force_R` and `assets/V6_Force_L` as canonical runtime packages, so a
fresh checkout must obtain them out of band using `docs/asset_manifest.md`.

If inclusion is authorized later, Git LFS is recommended for `*.STL`
(case-insensitive variants if introduced), because there are many binary meshes
even though each is below 100 MB. Configure LFS before the first asset commit;
do not rewrite this initial history or substitute fake geometry.

There is also no root project license. A private repository can remain unlicensed,
but public redistribution or collaboration terms should not be inferred. Select
a project license only after ownership is settled.

## Dependency and reproducibility notes

Runtime requirements are declared in `requirements.txt`: MuJoCo 3.x, NumPy,
PyYAML, pytest, websockets and Pillow. `requirements-validation.txt` adds
Playwright; Chromium installation is a separate documented command. Version
ranges are reproducible at the supported-major level, not a fully locked Python
environment. A future snapshot may add a generated lock/constraints file after
the target OS/Python matrix is chosen.

Required hand assets must appear at:

```text
assets/V6_Force_R/urdf/V6_Force_R.urdf
assets/V6_Force_R/meshes/*.STL
assets/V6_Force_L/urdf/V6_Force_L.urdf
assets/V6_Force_L/meshes/*.STL
```

Their recorded source-package SHA-256 identifiers are listed in
`docs/asset_manifest.md`. No network download location is known, so an authorized
owner must provide these exact packages if they are excluded.

## Proposed local Git commands — not executed

For the approved asset-excluded first snapshot:

```bash
cd /path/to/tactile_shape_reconstruction
git init -b main

git add .gitignore README.md requirements*.txt \
  backend simulation sensors robot_data reconstruction evaluation vision ui tests docs experiments
git status --short
git diff --cached --stat
git diff --cached --check
git commit -m "chore: prepare dexterous hand research platform snapshot"
git tag v0.1.0-rgb-baseline
```

The ignored `assets` directories remain unstaged. Tag creation is optional and
should occur only after the commit contents are approved. At preparation time no
remote creation, tag or push had been performed.

Proposed later remote steps, requiring explicit user approval and a confirmed
private repository URL:

```bash
git remote add origin <PRIVATE_REPOSITORY_URL>
git push -u origin main
git push origin v0.1.0-rgb-baseline
```

## Remaining decisions before the first push

1. Confirm ownership before any later V6/legacy asset commit; configure Git LFS
   first if STL inclusion is eventually approved.
2. Choose a project code/data license or explicitly keep the snapshot unlicensed.
3. Approve the selected small experiment results and the exclusion/archive policy
   for raw logs, screenshots and trial reconstructions.
4. Review the complete staged list before committing and pushing the supplied
   private GitHub remote.
