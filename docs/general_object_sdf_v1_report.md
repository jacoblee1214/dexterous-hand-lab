# Research Milestone 4 report

## Frozen experiment

The protocol was frozen before the first result at
`experiments/general_object/protocol_v1.yaml`. It contains 40 static attempts:
five object instances, two declared fixture orientations/occlusion levels, and
0/2/4/8 active scalar sensors. Development contains sphere and cylinder;
validation contains rounded box; held-out test contains cuboid and asymmetric L.
There is no training in this baseline, but the geometry split prevents future
test tuning. Seeds, calibrated RGB, packed masks, named joint/tactile inputs,
finite patches, calibration metadata, fixture transforms, evaluator-only shape,
pose and surface data are recorded. The generator does not force all 18 sensors
to contact.

The frozen protocol SHA-256 is
`1ffa07c0aea978a7602cd5eecb490941ef6e174ef2995d92949ba55c69c79861` and the
solver/configuration SHA-256 is
`fbfd1c4454e54278571722f3b10a0ea9e238fc292c1e5e7161f87f9e2a6a1969`.
The masks are controlled simulator-generated inputs, not the output of a
validated learned segmenter. This is a controlled analytic simulation dataset,
not a claim of MuJoCo contact realism or real-image generalization.

The five proposed methods are RGB-only, tactile-only, RGB plus representative
tactile, RGB plus finite-patch tactile, and RGB plus reliability-aware tactile.
The perfect-contact method is separately marked oracle and excluded from the
proposed input boundary. All 40 attempts were evaluated, including zero-touch
failures. Tactile-only was invalid for all ten zero-touch cases and valid for
30/40; RGB-containing methods were valid for 40/40.

## Quantitative result

Metrics below are millimetres and means over all valid attempts unless stated.

| Method | Valid | Chamfer-L1 | Visible error | Hand-occluded error | Contact-region error | F-score @5 mm |
|---|---:|---:|---:|---:|---:|---:|
| RGB only | 40/40 | 48.68 | 19.91 | 25.70 | 21.91 | 0.000 |
| Tactile only | 30/40 | 32.44 | 21.54 | 15.96 | 4.79 | 0.159 |
| RGB + representative | 40/40 | 43.76 | 16.18 | 17.18 | 4.77 | 0.044 |
| RGB + finite patch | 40/40 | 44.56 | 16.45 | 18.21 | 6.75 | 0.030 |
| RGB + reliability tactile | 40/40 | 45.99 | 17.88 | 20.49 | 13.51 | 0.018 |
| Perfect-contact oracle | 40/40 | 43.82 | 16.38 | 17.24 | 4.87 | 0.044 |

On the 16 held-out cuboid/L cases, RGB-only hand-occluded error was 26.61 mm.
Representative tactile reduced it to 18.02 mm (8.59 mm mean improvement),
finite patch to 19.09 mm (7.53 mm), and reliability-aware tactile to 21.24 mm
(5.38 mm). Each improved 12 paired cases, worsened zero, and tied the four
zero-sensor cases. Thus sparse touch improved the RGB-occluded surface in this
controlled dataset when contact was present. It did not solve global shape:
absolute errors and 5 mm F-scores remain poor.

## Sparsity and reliability findings

Mean hand-occluded errors across all geometries were:

| Active sensors / fingers | RGB | Representative | Finite patch | Reliability-aware |
|---|---:|---:|---:|---:|
| 0 / 0 | 25.70 | 25.70 | 25.70 | 25.70 |
| 2 / 2 | 25.70 | 21.91 | 22.98 | 25.53 |
| 4 / 4 | 25.70 | 13.67 | 15.81 | 22.08 |
| 8 / 4 | 25.70 | 7.44 | 8.36 | 8.66 |

Counts 0/2/4 correspond to 100/75/50% dropout relative to the eight-sensor
condition. Two reliability-weighted contacts improved only 0.17 mm, which is
not a practically meaningful gain at this grid resolution. More spatially
distributed contacts helped strongly up to eight sensors in this dataset; no
claim is made beyond that range. Repeated temporal observations were not tested
because this milestone forbids cross-pose accumulation and keeps one latest
sample per sensor.

The fixed representative constraint outperformed finite-patch and conservative
reliability weighting. This does not validate the representative point as exact.
It shows that the current coarse grid, sampled patch minimum, Gaussian update and
coverage weight are too conservative. The oracle was also marginally worse than
representative on some aggregates, so it is an input-information upper bound,
not a performance ceiling for this approximate decoder.

## Visibility and failure diagnosis

Evaluator-only GT points are partitioned using a z-buffer before hand masking:
front-most unblocked points are `visible_to_rgb`, front-most hand-blocked points
are `occluded_by_hand`, and the rest are `other_unobserved`. Metrics are reported
for all three plus the contact neighborhood; whole-object Chamfer is secondary.

Supported conclusions:

- one calibrated silhouette leaves object depth poorly constrained;
- touch improves local contact and occluded-region geometry when at least two
  observations are present, with much larger gains at four/eight sensors;
- tactile-only reconstructs contacted neighborhoods but has worse visible and
  other-unobserved coverage and fails with zero contacts;
- finite-patch uncertainty preserves physical honesty but loses accuracy against
  the exact-point approximation in this coarse implementation;
- reliability weighting prevents sparse uncertain touch from dominating, but
  also suppresses useful corrections at two/four sensors.

Unresolved hypotheses include better continuous patch likelihoods, orientation
regularization, learned feature/query decoding, predicted depth priors, denser
grids, multiple synchronized cameras, calibrated real contact uncertainty, and
measured object pose tracking. None was tuned on held-out test truth.

## Compute and dashboard

The SDF is 55,296 bytes per float32 grid. Its transparent decoder has zero
trainable parameters; the frozen ResNet-18 frontend has 11,176,512 parameters
and approximately 1.814 GFLOPs at 224x224. On the recorded system, feature
encoding averaged 817.69 ms and SDF reconstruction averaged 11.61 ms on CPU;
GPU/NPU deployment was not attempted.

Reproduce the frozen run from the repository root after installing the optional
research dependencies and placing the verified torchvision checkpoint locally:

```bash
pip install -r requirements-research.txt
python -m general_object.experiment \
  --feature-weights /path/to/resnet18-5c106cde.pth
```

The runner refuses a checkpoint unless its full SHA-256 is
`5c106cde386e87d4033832f2996f5493238eda96ccf559d1d62760c4de0613f8` and never
downloads weights implicitly.

The existing four quadrants remain. Bottom-right can switch between the sphere
A–E baselines and General Object SDF v1. The latter shows clean RGB input,
its separate object/hand-occluded/unreliable visibility map,
method/status/confidence, tactile constraint type, a draggable/zoomable predicted
surface and contact points. GT/oracle data are absent from default WebSocket
state and appear only after the explicit evaluation toggle.

This is simulation evidence under a declared static fixture assumption. It is
not real-hardware validation, free-object tracking, regrasp, VLM fusion, or a
deployment result.

## Verification and baseline preservation

The branch was created from parent commit
`1841d1cfd00481866fa0708e359bf8d07bad1071`. No tracked file under the prior
`experiments/fusion/` result tree changed, and the sphere A–E reconstruction
implementations were not modified. `vision/calibration.py` gained only an exact
declared-pose calibration constructor reused by this branch.

Final automated verification on Linux with `MUJOCO_GL=egl`:

- scientific/unit/integration suite: **188 passed, 3 skipped**; the skips are the
  explicitly opt-in browser tests;
- real Chromium + HTTP + WebSocket suite: **3 passed**;
- generated input audit: 40/40 cases contain a 20-name joint snapshot, all
  tactile records contain a homogeneous `T_object_from_sensor`, and zero-contact
  cases remain present;
- default dashboard-state audit: general-object evaluation surface and the
  perfect-contact oracle are absent until `Show Ground Truth` is enabled.

The browser result is an automated visual artifact, not a claim of manual
scientific acceptance. Its local path is
`experiments/general_object/object_sdf_v1_20260910/browser/general-object-sdf-dashboard.png`;
screenshots remain ignored by the repository artifact policy.
