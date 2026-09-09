# Research Milestone 2: Geometric Visuo-Tactile Fusion v1

## Scope and boundary

This milestone estimates only the center of a sphere whose radius is declared in
advance. The `sphere_40mm` experiment means **radius = 0.040 m** and **diameter =
0.080 m**. Radius error is therefore not reported as an estimated quantity.

The algorithm receives a `FusionObservation`: calibrated RGB/predicted mask,
visible-background and unknown/hand-occluded classes, source timestamps, named
joint state, estimated scalar-tactile contact geometry with uncertainty, sensor
mount geometry, and the declared radius. It cannot receive MuJoCo object pose,
contact position/normal, segmentation IDs, object geometry, or depth. Those are
available only to `fusion/evaluation.py` after estimation.

Historical tactile observations are combined only in the fixed-pose diagnostic.
Every tactile record retains its own measurement timestamp and the joint state
interpolated at that timestamp. Free/moving-object data accepts only an
instantaneous synchronized sample, so samples from different object poses cannot
be silently mixed.

## Geometry and objective

The RGB term reuses the calibrated perspective tangent-ray geometry from the
vision-only baseline. Boundaries next to pixels classified as unknown/hand-
occluded are excluded. Candidate silhouette intrusion is penalized only in known
background; occluded pixels are never treated as free space.

For representative tactile point `p_i`, center `c`, and fixed radius `r`:

```text
e_touch_i = (||p_i - c|| - r) / sigma_i
```

The finite-patch variant samples the calibrated rectangular or fingertip sensing
surface and uses zero residual when any point in its feasible radial interval can
lie on the sphere. It is deliberately distinct from the representative-point
model. Inactive sensors provide no free-space constraint.

Both groups are normalized by sample count before applying the global weights in
`fusion/config.yaml`:

```text
J(c) = lambda_rgb mean(rho(e_rgb)) + lambda_touch mean(rho(e_touch))
```

The deterministic three-parameter Gauss-Newton/IRLS solver uses a Huber loss,
finite-difference Jacobian, bounded steps, line search, SVD conditioning, and an
approximate covariance. It initializes from a reliable existing RGB center, then
a visibility-aware RGB cone estimate, or (when sufficiently distributed) a
fixed-radius tactile algebraic estimate. It never initializes from truth.

## Methods and dashboard

- A: RGB only
- B: tactile only, representative contact points
- C: RGB plus representative contact points
- D: RGB plus finite sensor-patch feasibility

The browser keeps the same four quadrants. The bottom-right panel exposes the
four statuses and lets the operator inspect one method; the reconstruction view
overlays RGB (blue), tactile (orange), and selected fusion (green) estimates.
The clean RGB image is unchanged. Ground truth remains behind the pre-existing
evaluation toggle and is not embedded in fusion output.

Launch with the existing command:

```bash
MUJOCO_GL=egl python -m backend.dashboard_server
```

The matching frontend/backend build is
`geometric-fusion-v1-20260909.1`.

## Reproducible experiment

Run:

```bash
MUJOCO_GL=egl python -m fusion.experiment \
  --output experiments/fusion/geometric_v1_20260909 --trials 3
```

The matrix contains 18 attempts: three repeats of clear/no-touch, partial hand
occlusion, stronger grasp occlusion, controlled RGB-mask degradation, three-point
sparse touch, and single-finger insufficient distribution. Algorithm JSON is in
`algorithm/`; MuJoCo pose/mask metrics are in a separate `evaluation/` tree.
Raw RGB/masks and replay streams are retained locally under the repository asset
policy and are not Git-tracked.

Measured results from the committed 2026-09-09 run:

| Method | Valid / 18 | Valid center error mean | Failure rate | Mean runtime (all attempts) |
|---|---:|---:|---:|---:|
| A RGB only | 18 | 30.78 mm | 0.0% | 3.84 ms |
| B tactile representative | 6 | 16.25 mm | 66.7% | 3.84 ms |
| C fusion representative | 6 | 14.83 mm | 66.7% | 14.77 ms |
| D fusion finite patch | 6 | 14.33 mm | 66.7% | 37.21 ms |

Across the six cases where RGB and fusion were both valid, C reduced mean center
error by 64.10 mm and D by 64.61 mm relative to RGB. This aggregate improvement
is dominated by controlled mask degradation. Under the undegraded stronger-grasp
condition, RGB was about 7.89 mm while C was 11.20–12.19 mm and D was
9.46–9.72 mm: fusion **degraded** the center estimate there. Partial, sparse, and
single-finger conditions correctly failed the tactile/fusion observability gate.
Thus the milestone demonstrates a useful geometric fusion baseline under a
specific segmentation failure, not a universal improvement claim.

Exact per-attempt metrics, failure statuses, residuals, sensor/finger counts,
coverage, conditioning, covariance, runtime, valid-only aggregates, and
all-attempt failure rates are in
`experiments/fusion/geometric_v1_20260909/evaluation_summary.json` and its
per-condition files.
