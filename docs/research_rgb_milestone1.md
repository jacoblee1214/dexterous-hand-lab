# Calibrated RGB observation and vision-only baseline

## Scope and architecture

This milestone adds one observation path beside the existing tactile path. It
does not fuse the paths:

```text
MuJoCo state ──> fixed research camera ──> clean RGB ──> blue mask
                                                    └─> known-radius sphere-center fit

joint state + 18 tactile scalars + sensor calibration ──> existing tactile reconstruction

MuJoCo object pose/segmentation ──> evaluation files only
```

The fixed `research_rgb_camera` is independent of the movable dashboard/debug
camera. `vision/research_camera.py` hides sensor sites, collision hulls and all
reconstruction overlays before RGB acquisition. It never requests depth.
MuJoCo segmentation IDs and object pose are generated only after the RGB
algorithm has run and are returned in a separate evaluation object.

`CameraObservation` carries the acquisition timestamp, frame/camera IDs,
dimensions, external RGB reference, inline numeric intrinsics, versioned
intrinsic/extrinsic references, source and time base. JPEG bytes use a bounded
one-frame stream buffer and are not embedded in ordinary JSON states. Simulation
publishes the observation through `MuJoCoRobotDataProvider`. Replay retains the
recorded metadata/timestamp and decodes the referenced archived JPEG. The
read-only `HardwareRobotDataProvider.accept_camera_observation()` is the future
camera entry point; no transport or synthetic hardware data was added.

## Calibration and coordinate frames

The reproducible source configuration is `vision/camera_calibration.yaml`.
MuJoCo 3.12 compiles a 640 × 480 fixed camera with vertical field of view 50°.
The pinhole matrix is derived from the compiled `model.cam_fovy`, not from a
screenshot:

```text
fx = fy = 0.5 * 480 / tan(50° / 2) = 514.6816609223 px
cx = 319.5 px, cy = 239.5 px
distortion = none, coefficients = [0, 0, 0, 0, 0]
```

Transform notation is `T_A_from_B`: it maps a homogeneous point expressed in B
to A. The common geometry chain is:

```text
p_world = T_world_from_link · T_link_from_sensor · p_sensor
p_camera_cv = T_camera_cv_from_world · p_world
pixel = K · (x/z, y/z, 1)
```

The hand base is `Palm`, coincident with the model world frame in this fixed-base
experiment. Link poses are world transforms from MuJoCo; each sensor mounting
transform remains link-local in `sensors/sensor_config.yaml`. Object pose is also
expressed in world, but is evaluation-only. For a moving real object, tactile
points must not be accumulated in an object frame until a timestamp-valid
measured object pose exists.

MuJoCo camera axes are +X right, +Y up, -Z forward. Computer-vision axes are +X
right, +Y down, +Z forward, so
`R_cv_from_mujoco = diag(1, -1, -1)`. The actual compiled camera pose is read
from `data.cam_xpos`/`data.cam_xmat`, converted numerically, and stored in the
experiment calibration JSON as both `T_world_from_camera_cv` and its inverse.
Tests verify transform inversion, known 3D projections, round trips, and the
forward-axis conversion.

## Transparent baselines

The segmentation baseline thresholds the deliberately blue experimental sphere,
keeps the largest four-connected region and reports component size, boundary
coverage, confidence, status and processing time. This assumption is controlled
and does not generalize to arbitrary objects or lighting.

For a declared known radius, boundary pixels become calibrated camera rays. The
solver minimizes the ray-to-center tangent distance minus the radius using
robust iteratively reweighted Gauss–Newton. It therefore uses the perspective
tangent cone rather than assuming that the projected outline is a circle whose
center equals the projected 3D center. The supplied radius is labeled
`provided_known_radius_prior`, never as an estimate. A synthetic perspective
test recovers center coordinates within 0.8 mm.

With an unknown radius, one calibrated monocular silhouette only constrains
viewing direction and angular size. The implementation returns
`SCALE_AMBIGUOUS`, with metric radius and center depth left null.

Estimated tactile contact positions may be projected through the same world to
camera transform. The browser draws these as a separate optional canvas overlay;
they never alter the archived clean RGB.

## Reproducible 40 mm experiment and validation

Run:

```bash
MUJOCO_GL=egl python -m vision.experiment \
  --output experiments/vision_baseline/my_run
```

The committed run is `experiments/vision_baseline/rgb_milestone1_20260908_run1`.
It records four controlled conditions: centered reference, +20 mm world-X
offset, development approach, and settled development grasp. No approved taught
pose was present, so the existing `sphere_40mm` automatic preset is explicitly
labeled as a development configuration. Inputs and predictions are under
`inputs/` and `algorithm/`; MuJoCo masks, poses and errors are only under
`evaluation/`. `observations.jsonl` records synchronized joint/tactile/camera
metadata with stable RGB references and original simulation timestamps.

Observed results (these include failures, not a selected favorable frame):

| condition | visibility | mask IoU | center error | silhouette residual | status |
| --- | ---: | ---: | ---: | ---: | --- |
| centered reference | 0.991 | 0.942 | 2.65 mm | 2.37 px | estimated |
| +20 mm X offset | 0.993 | 0.917 | n/a | 5.24 px | unreliable |
| approach occlusion | 0.741 | 0.649 | n/a | 89.05 px | unreliable |
| settled-grasp occlusion | 0.722 | 0.632 | n/a | 15.40 px | unreliable |

All four segmentations returned a mask; three of four geometric fits correctly
failed the current reliability gate, for a 75% fit failure rate. Mean mask IoU
was 0.785. Radius error is intentionally absent because radius was supplied.
Per-condition processing times and exact floating-point values are in
`algorithm_summary.json` and `evaluation_summary.json`.

## Dashboard and unresolved limitations

Launch the existing dashboard command and select **Research RGB Camera** in the
top-left View selector. **MuJoCo Debug Render** remains available; the other
three quadrants and the existing tactile reconstruction are unchanged. The
bottom-right vision card shows segmentation status, known-radius label, center,
residual, processing time, unknown-radius ambiguity and optional evaluation IoU.

Current limitations are deliberate and visible:

- color segmentation is lighting/material-specific and is degraded by similarly
  colored highlights and hand occlusion;
- a single unknown-radius RGB image has no metric scale;
- the known-radius outline fit does not infer missing silhouette under occlusion;
- simulation evaluation pose/masks are not available to the algorithm;
- there is no RGB/tactile fusion, object tracking, multi-view acquisition,
  learned prior or physical camera calibration yet.

The next fusion milestone needs a timestamped `CameraObservation`, its decoded
RGB/mask, `K`, `T_world_from_camera_cv`, estimated tactile contacts in world with
their timestamps/covariances, and an explicit measured object-pose history if
observations are transformed into a moving object frame. It must define clock
alignment uncertainty, observation association, occlusion handling and separate
evaluation-only ground truth before combining either modality.
