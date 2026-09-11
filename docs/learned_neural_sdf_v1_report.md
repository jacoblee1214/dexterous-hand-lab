# Research Milestone 5 — Learned Visuo-Tactile Neural SDF v1

## Outcome

This is a **preliminary single-seed** result, not a solved-reconstruction claim.
On the frozen 96-case held-out test, L4 reliability-aware finite-patch fusion
reduced mean hand-occluded surface error from 5.586 mm (true learned RGB-only)
to 5.193 mm. The paired mean change was -0.393 mm (bootstrap 95% CI
[-0.634, -0.147] mm), and 58.3% of cases improved. On the 78 contact-present
cases, the paired change was -0.368 mm (95% CI [-0.648, -0.084] mm), with 53.8%
of cases improving.

The result is mixed. L1–L3 all degraded hidden-surface error, and L4 did not
generalize that improvement to the completely unseen asymmetric-L family. A
shuffled tactile sample worsened the 24³ control, but scalar-only shuffling had
almost no effect. The network demonstrably responds more to contact geometry
than scalar magnitude; evidence that it uses scalar pressure meaningfully is
absent in v1.

## Reproducibility boundary

- branch: `research/learned-visuotactile-neural-sdf-v1`
- parent: `4978129441fa06306a1ffd1a2453bbe2715dd66f`
- protocol: `experiments/neural_sdf/protocol_v1.yaml`, SHA-256
  `ea303d1930af0645ff605e61ae464884ac670b5494181ac46e6e65ddcef734cb`
- manifest: `experiments/neural_sdf/dataset_manifest_v1.json`, SHA-256
  `4ca2073f8f41f912d0a916672e1d3edb4439e599de11b5b9ea625e1ce4988423`
- model/training config SHA-256:
  `3f8ba5731eb0d9205e0f50de5c0fe31ad16735e2c98dbe3a78a17a4ae403c41e`
- frozen ResNet-18 weights SHA-256:
  `5c106cde386e87d4033832f2996f5493238eda96ccf559d1d62760c4de0613f8`
- training: Python 3.10.19, PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128,
  CUDA build 12.8 but unavailable; CPU execution
- simulator/test: Python 3.12.3, MuJoCo 3.12.0

The 84 instances produce 432 unique pose/observation configurations, 432 RGB
frames, and 1,280 declared tactile observations: train 48 instances/288 samples,
validation 12/48, test 24/96. There are 10 count/pattern contact configurations
(0/1/2/4/8 sensors × clustered/diverse). These are procedural contact
configurations, not claims of 432 physically executed grasps. Eight asymmetric-L
test instances are a family holdout. The older controlled 40-case General Object
SDF v1 benchmark and all sphere/Fusion v2 historical files remain separate and
unchanged.

## Primary held-out test

Values are mean (median), in mm except completeness/F-score. Every method was
valid on 96/96 attempts. Contact error is defined only on contact-present cases.

| Model | Whole Chamfer-L1 | Visible | Hand-occluded | Contact region | Completeness @5 mm | F-score @5 mm |
|---|---:|---:|---:|---:|---:|---:|
| L0 RGB-only | 14.295 (13.502) | 5.967 (5.713) | 5.586 (5.785) | 6.111 (5.539) | 0.431 | 0.421 |
| L1 RGB + global touch | 16.112 (15.189) | 6.817 (6.131) | 6.071 (5.838) | 6.420 (5.886) | 0.382 | 0.390 |
| L2 representative local | 16.241 (15.631) | 7.169 (6.456) | 5.984 (5.813) | 5.521 (4.788) | 0.386 | 0.380 |
| L3 finite-patch local | 15.830 (14.712) | 7.254 (6.458) | 6.584 (6.480) | 5.345 (4.792) | 0.375 | 0.382 |
| L4 reliability finite-patch | **13.880 (13.275)** | **5.723 (5.426)** | **5.193 (5.165)** | **5.024 (4.496)** | **0.455** | **0.435** |

L0 hand-occluded mean has individual bootstrap 95% CI [5.296, 5.909] mm;
L4 has [4.913, 5.497] mm. The paired CI above is the appropriate direct
comparison because all methods use identical case IDs.

### Contact-present and no-contact

| Subset | Cases | L0 occluded mean (median) | L4 occluded mean (median) | Paired L4-L0 mean, 95% CI |
|---|---:|---:|---:|---:|
| Contact present | 78 | 5.570 (5.759) | 5.202 (5.076) | -0.368 [-0.648, -0.084] mm |
| No contact | 18 | 5.657 (5.961) | 5.155 (5.277) | -0.502 [-0.831, -0.111] mm |

The no-contact difference is not instantaneous tactile benefit: L0 and L4 are
separately optimized heads. It reflects different training trajectories and
inductive effects. L4 receives no fabricated tactile observation in those
cases. A regression test verifies that L0 is exactly invariant to tactile
values, geometry, validity, and coverage.

## Generalization

| Held-out family | Cases | L0 occluded | L4 occluded | L0 contact | L4 contact |
|---|---:|---:|---:|---:|---:|
| Asymmetric L (unseen family) | 32 | **3.763** | 3.953 | 3.892 | **3.497** |
| Cuboid (unseen instances/range) | 16 | 7.258 | **6.159** | 7.271 | **5.430** |
| Cylinder | 16 | **6.203** | 6.221 | 7.873 | **6.789** |
| Rounded box | 16 | 6.195 | **5.892** | 5.875 | **4.455** |
| Sphere | 16 | 6.337 | **4.981** | 7.860 | **6.476** |

The unseen-family result is the key limitation: L4 improves the L-object contact
neighborhood but worsens its broader hand-occluded surface by 0.189 mm. The v1
result therefore supports limited held-out-instance benefit, not arbitrary
geometry generalization.

The older geometric 40-case benchmark reported cuboid/L occluded errors of
26.61 mm RGB-only, 18.02 mm representative, 19.09 mm finite patch, and 21.24 mm
reliability-aware. Those values use a different frozen dataset/protocol and are
external context, not a paired comparison to this learned test.

## Mandatory tactile controls

Controls use the same 96 cases at 24³ to keep the controlled sweep practical.

| L4 tactile input | Occluded error | Contact error | Paired occluded change vs normal |
|---|---:|---:|---:|
| Normal | 5.616 | 5.431 | — |
| Zero | 5.591 | 6.295 | -0.024 mm, CI [-0.222, 0.158] |
| Different-sample shuffled | 5.857 | 6.145 | +0.241 mm, CI [-0.034,  0.555] |
| Position shuffled | 5.735 | 6.214 | +0.119 mm, CI [-0.000, 0.250] |
| Scalar shuffled | 5.619 | 5.434 | +0.003 mm, CI [-0.006, 0.013] |

Invalid geometry consistently worsens contact error, and full shuffled touch
worsens mean hidden error, so L4 is not simply ignoring tactile geometry.
However, the occluded-error confidence intervals for individual controls include
zero, zero touch does not hurt that metric, and scalar shuffle is effectively
unchanged. This is weak rather than definitive negative-control evidence.

## Sparsity, extraction, and robustness

Selected-resolution L0→L4 occluded error by active sensor count was: 0 sensors
5.657→5.155 mm (18 cases), 1 sensor 5.899→5.477 (18), 2 sensors
5.300→5.442 (19; worse), 4 sensors 5.606→4.857 (23), and 8 sensors
5.481→5.114 (18). Clustered contacts were 5.623→5.120 mm; diverse contacts
5.550→5.267 mm. The complete machine-readable curves also group by finger count
and spatial-coverage bin. There is no monotonic “more touch always helps” result.

The validation-only extraction study found RGB/L4 occluded error of
5.510/5.316 mm at 24³ and 5.026/4.872 mm at 48³. Higher extraction density
reduced discretization error, so 48³ was fixed before held-out evaluation.

At 24³, L4 normal occluded/contact errors were 5.616/5.431 mm. Sensor-mount
+3/-2/+1 mm gave 5.614/5.554; the declared 2 mm joint-pose proxy gave
5.614/5.441; 25/50/75% tactile dropout gave occluded 5.645/5.670/5.749 and
contact 5.590/6.029/6.128 mm. Incorrect contact geometry (+20/-15 mm) increased
contact error to 6.988 mm. Camera 4 mm/2° gave 5.588/5.478. Two-pixel mask
erosion unexpectedly improved the means to 5.310/5.400; this is retained as an
unresolved distribution/resolution effect, not presented as robustness proof.
The “joint bias” is explicitly a static synthetic FK consequence proxy, not a
full encoder-to-FK perturbation.

## Compute and training discipline

Each head has 27,889 trainable parameters. The frozen ResNet portion used here
has 683,072 parameters and an estimated 555 MFLOP per RGB frame. Best checkpoint
sizes are 321–353 kB. A 48³ field contains 110,592 queries; vectorized CPU
evaluation measured mean 3.46 s per field and about 31.9k SDF queries/s. Peak
evaluation-process RSS was 1.10 GB; GPU allocation was zero. A separate
one-epoch L4 profile with the exact batch/query settings measured 0.94 GB peak
RSS and 20.25 s; it was not used for checkpoint selection.

All heads used seed 51001, AdamW, lr 1e-3, batch 16, 128 queries/sample, five
epochs, and the frozen loss/dropout schedule. Best validation epochs were L0=3,
L1=4, L2=1, L3=3, L4=4. Best and last checkpoints are retained locally but
excluded from Git as generated model binaries. One seed is computationally
practical for this first CPU milestone, so no across-seed uncertainty is claimed.
No separate small/large capacity sweep was run; L0–L4 deliberately keep equal
head capacity, and the measured CPU cost made another capacity axis impractical
for this preliminary run.

## Dashboard and limitations

The existing four-quadrant browser dashboard now includes “Learned Neural SDF
v1” with L0–L4 selection, clean input RGB/masks, active sensor count, reliability,
checkpoint, runtime, and an independently rotatable/zoomable predicted surface.
The backend removes evaluation surface data and even GT diagnostic labels from
WebSocket state unless `Show Ground Truth` is explicitly enabled.

Remaining sim-only assumptions are procedural RGB/input masks, synthetic scalar
response/contact estimates, simulated kinematics, analytic training SDF, and the
declared static fixture/object frame. There is no measured pose tracker, real
sensor mapping/calibration validation, real RGB segmentation validation,
temporal fusion, reorientation, regrasp, motor control, VLM, or edge-deployment
validation in this milestone.

Machine-readable results are in
`experiments/neural_sdf/learned_v1_20260911/`; the exact architecture and loss
are documented in `docs/learned_neural_sdf_v1_architecture.md`.

## Verification

- simulation suite with `MUJOCO_GL=egl`: 193 passed, 4 skipped;
- PyTorch neural contract/device suite in the training environment: 5 passed;
- real headless Chromium + HTTP + WebSocket dashboard suite: 3 passed.

The browser run loaded build `learned-neural-sdf-v1-20260911.1`, exercised the
actual server state, selected learned L0/L4 views, rendered the held-out SDF
surface with GT disabled, and captured a dashboard screenshot. No claim of
real-hardware or physical manual validation is made.

After this frozen CPU result was recorded, device selection was changed to
`--device auto`. On the host (outside the restricted development sandbox),
PyTorch detected the NVIDIA GeForce RTX 5070 Ti Laptop GPU and completed an
actual CUDA neural-SDF forward pass. `--device cuda` now rejects silent CPU
fallback. A complete five-epoch L0 smoke training ran on the RTX 5070 Ti in
12.64 s versus 55.64 s for the corresponding cached-feature CPU training
(about 4.4× faster), with the same best epoch and numerically equivalent
validation result. These checks do not relabel or overwrite the CPU result table
above.
