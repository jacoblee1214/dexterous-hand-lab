# Contact-point and sphere reconstruction

This package implements the ground-truth-free first reconstruction stage:

```text
timestamp + joint state + scalar sensor values + fixed mounts + calibration
  -> sensor FK -> pressure/indentation -> estimated 3D contact points
```

`ReconstructionInput` is the enforced runtime boundary. The package does not
import MuJoCo or the evaluation package and cannot receive simulator contact
positions, contact normals, object poses, object meshes, or collision geometry.
MuJoCo ground truth is consumed only by the separate top-level `evaluation/`
package for one-way quantitative comparison.

`TemporalTactileObservation` stores the accepted timestamp, sensor/finger/link
IDs, joint-state snapshot, sensor pose/direction, scalar value, calibrated
pressure/indentation, and common-frame estimated point. It contains no ground
truth. `ContactPointBuffer` applies configurable pressure, time, capacity, and
per-sensor spatial-distance filters while reporting raw/accepted/duplicate
counts. `sphere_fitting.py` then consumes only these accepted observations. It
uses algebraic least squares followed by robust geometric least squares for the
explicit objective `sum((norm(point - center) - radius)^2)`, and refuses a
confident result when configurable point/channel/finger/spatial coverage checks
fail. Manual fitting is the default; optional auto fitting is throttled by newly
accepted point count.

The sphere result intentionally has no GT fields. MuJoCo center/radius access and
estimated-vs-GT surface metrics live only in `evaluation/sphere_evaluator.py`.
General free-form reconstruction remains outside this milestone.
`ContactPointReconstructor`
accepts an optional calibration-model factory, so a hardware-calibrated
scalar-to-pressure/indentation implementation can replace the default synthetic
model without changing FK or projection code.
