# Camera and geometric-fusion explainability

## Two cameras with different scientific roles

The **Research RGB Camera** is the fixed `research_rgb_camera` compiled into the
MuJoCo model from `vision/camera_calibration.yaml`. It produces the RGB input used
by segmentation and fusion. Its renderer disables sensor sites, collision hulls,
contact markers, fitted spheres, and every debug overlay. Its pose cannot be
changed from the dashboard.

The **Debug Camera** is the free MuJoCo camera used only for the top-left
engineering image. Left drag, right drag, wheel, and Reset Camera change this
view, but never alter the research camera, physics, `qpos`, or reconstruction
input. The optional **Research camera frame/frustum** checkbox adds a calibrated
camera model to this debug scene only. The frustum length (0.12 m) is merely a
drawing length; its rays come from the exact intrinsics. No marker enters the
clean research RGB stream.

## Exact fixed-camera calibration

Calibration version: `research-rgb-camera-v1`; camera ID:
`external_rgb_01`; optical frame: `research_rgb_camera_optical`; parent:
MuJoCo `world`; time base: `simulation_seconds`.

The YAML pose is position `[0.30, -0.38, 0.27]` m and MuJoCo `xyaxes`
`[0.78, 0.62, 0, -0.29, 0.36, 0.89]`. MuJoCo normalizes these axes. The actual
compiled calibration is:

```text
resolution = 640 x 480
vertical FOV = 50.0 degrees
fx = fy = 514.6816609222941 px
cx = 319.5 px, cy = 239.5 px
distortion = none, coefficients = [0, 0, 0, 0, 0]

T_world_from_camera_cv =
[[ 0.7828232548,  0.2868137687, -0.5522007007,  0.3000000000],
 [ 0.6222441256, -0.3608302252,  0.6947041074, -0.3800000000],
 [ 0.0000000000, -0.8874341726, -0.4609344740,  0.2700000000],
 [ 0.0000000000,  0.0000000000,  0.0000000000,  1.0000000000]]
```

`T_world_from_camera_cv` maps a point in the CV optical frame into world. Its
inverse, displayed by the live state as `T_camera_cv_from_world`, maps world into
the camera for projection. CV axes are +X right, +Y down, +Z forward; MuJoCo
camera axes are +X right, +Y up, -Z forward. The conversion is exactly
`diag(1,-1,-1)`. The dashboard reads values computed from the compiled camera,
not duplicated approximations.

## A–D algorithms

All modes receive the same immutable `FusionObservation`: clean RGB predicted
mask, known-background/unknown-occluded masks, `K`,
`T_camera_cv_from_world`, named synchronized joints, named scalar tactile
constraints and calibration. Simulator object/contact truth, segmentation IDs,
depth, and object mesh are absent. The current sphere setting supplies a **known
radius of 0.040 m** (80 mm diameter); only center `c in R^3` is estimated.

For each visible mask-boundary ray `d_j`, the RGB residual is the calibrated
tangent distance between the ray and the candidate sphere, standardized by
`sigma_rgb = 2 px`. Candidate intrusion into known background is also penalized;
unknown/hand-occluded pixels are not negative evidence.

For representative contact `p_i`:

```text
e_tactile_i(c) = (||p_i - c|| - r) / sigma_i
```

For a finite patch with sampled world points `P_i`, radial interval
`[a_i,b_i] = [min_p ||p-c||, max_p ||p-c||]`, the residual is zero when
`r in [a_i,b_i]`, otherwise the signed distance from `r` to the interval divided
by its finite-patch uncertainty.

| Mode | Inputs and residuals | Initialization | Additional validity |
|---|---|---|---|
| A RGB-only | RGB tangent rays + known-background intrusion | visibility-aware RGB cone (or an existing reliable RGB center) | segmentation OK, >=20 visible boundary samples, angular coverage >=0.20 |
| B Tactile representative | representative radial residuals | fixed-radius algebraic point center, rank 3 | >=4 contacts, >=3 sensors, >=2 fingers, spread >=0.015 m, tactile singular value >=0.08, condition <=1000 |
| C Fusion representative | A + B | reliable RGB center, otherwise observable tactile algebraic center | both RGB and tactile validity |
| D Fusion finite-patch | A + finite-patch interval residual | same as C | both RGB and tactile validity |

Every modality is standardized, divided by the square root of its sample count,
and multiplied by `sqrt(lambda)`. `lambda_rgb=lambda_tactile=1.0` are fixed global
constants. The objective is therefore
`lambda_rgb mean(rho(e_rgb)) + lambda_tactile mean(rho(e_tactile))`, with Huber
loss threshold 2.5 standardized units. The deterministic bounded-step
Gauss-Newton/IRLS solver reports its joint Jacobian condition, minimum singular
value, iterations, and covariance-derived center standard deviation.

A physical sensor has a finite surface but returns one scalar. It does not say
where within that surface contact occurred, so the representative point is an
approximation and even the finite-patch model is a feasible-region model, not a
measured contact coordinate. Likewise, scalar pressure/normal response is not a
three-axis force vector and supplies no independently measured world direction.

The dashboard shows exact residuals in standardized units and pixels/metres,
weights, contributing sensor/finger IDs, spatial spread, tactile and joint
conditioning, uncertainty, initialization, and a specific missing-information
reason for invalid estimates.

## Correct paired reanalysis of the 18 attempts

`python -m fusion.reanalysis` joins algorithm and evaluation records by the
`trial_XX_condition` case ID. All-attempt success is A 18/18 (100%); B, C, and D
are each 6/18 (33.3%). The intersection where all four modes are valid contains
six exact IDs (three strong-occlusion and three controlled-mask-degradation):

| Mode | Mean center error on the same six cases |
|---|---:|
| A RGB-only | 78.934 mm |
| B tactile representative | 16.251 mm |
| C fusion representative | 14.831 mm |
| D fusion finite-patch | 14.326 mm |

This six-case aggregate is dominated by intentionally degraded masks. On the
three ordinary strong-occlusion pairs, A is 7.890 mm, B 16.251 mm, C 11.774 mm,
and D 9.598 mm. C therefore degrades by 3.883 mm and D by 1.708 mm relative to
RGB. The degradation is preserved, not hidden by the aggregate.

Supported by the records: D lowers both strong-occlusion tactile inconsistency
and center degradation relative to C, which implicates within-patch contact
ambiguity as a contributor. Coverage is unchanged, initializations succeed, and
conditions are finite, so gross rank loss or initialization failure is not the
primary explanation in those runs. Under controlled mask removal, RGB error is
149.979 mm while C/D are 17.888/19.053 mm, so touch is useful for that different
failure regime.

Unresolved: these 18 records cannot independently isolate mount, encoder,
pressure, or camera calibration error. Equal standardized modality scaling may
still be mismatched under occlusion, but weights were not tuned from held-out
truth. Exact representative-contact error would require evaluation-only contact
truth and cannot be inferred from the scalar algorithm input. See
`paired_reanalysis.json` for every paired row and diagnostic aggregate.

## Sensitivity interpretation

Run `MUJOCO_GL=egl python -m fusion.sensitivity`. It repeats the same strong-
occlusion simulation protocol three times and perturbs one input class at a time.
Level zero is immutable; `fusion/config.yaml` and weights are never changed.
Center truth is read only after solving. Output reports accuracy, failure rate,
and condition per level/method in a separate `sensitivity_v1_20260909` directory.

Because no hardware accuracy specifications were supplied, every nonzero range
is labeled `EXPLORATORY_NOT_HARDWARE_JUSTIFIED`: mount 0–5 mm/0–5 degrees,
encoder bias or noise 0–1 degree, indentation calibration 0–20%, channel dropout
0–75%, focal scale 0–5%, camera extrinsics 0–5 mm and 0–2 degrees, mask removal
0–40%, and encoder/tactile mismatch 0–50 ms. These are stress-test ranges, not
claims about the real device.

Actual three-trial endpoints from `sensitivity_summary.json` are below. Values are
mean center error for A/C/D; `FAIL` means all three tactile-dependent attempts
were rejected by the explicit observability gate. Directional perturbations can
occasionally reduce an already biased baseline, so these are sensitivity
measurements, not monotonic calibration curves.

| Exploratory endpoint | A RGB | C representative fusion | D finite-patch fusion | Failure/conditioning observation |
|---|---:|---:|---:|---|
| Mount translation 5 mm | 7.89 mm | 8.71 mm | 12.51 mm | 0% failed; C/D condition 1.9/2.2 |
| Mount orientation 5 deg | 7.89 mm | 11.91 mm | 9.87 mm | 0% failed; C/D condition 1.8/1.6 |
| Encoder bias 1 deg | 7.89 mm | 11.77 mm | 9.31 mm | 0% failed |
| Encoder noise 1 deg std | 7.89 mm | 12.04 mm | 9.29 mm | 0% failed |
| Scalar calibration error 20% | 7.89 mm | 8.90 mm | 7.81 mm | 0% failed; deterministic mixed-sign perturbation |
| Tactile channel dropout 50% | 7.89 mm | FAIL | FAIL | 100% tactile/fusion failure from inadequate coverage |
| Focal-length error +5% | 30.18 mm | 12.79 mm | 8.91 mm | 0% failed; RGB condition 39.1 |
| Extrinsic error 5 mm + 2 deg | 18.10 mm | 38.51 mm | 18.52 mm | 0% failed; conditioning does not reveal calibration correctness |
| Additional mask removal 40% | 141.92 mm | 17.93 mm | 19.37 mm | 0% failed at the current visibility gate |
| Encoder/tactile offset 50 ms | 7.89 mm | 11.81 mm | 9.60 mm | 0% failed; near-settled grasp is weakly sensitive |

The supported sim-to-real risk is therefore not conditioning alone: extrinsic
and mask errors can remain numerically well-conditioned while being
geometrically wrong. Channel dropout is caught as invalid. The timestamp test is
weak at a settled final grasp and must be repeated during measured motion on
hardware; it does not justify a 50 ms tolerance.
