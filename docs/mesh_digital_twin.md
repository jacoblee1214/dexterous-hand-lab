# Browser mesh digital twin fallback

This Three.js view-only digital twin remains selectable as an explicit fallback
and debugging mode. `MuJoCo Render`, documented in
`docs/mujoco_dashboard_render.md`, is the default for simulation and replay.
The mesh fallback is the default for future hardware without a MuJoCo render
source. The browser never reads or writes `qpos` and does not perform forward
kinematics.

## Data and asset flow

At each synchronized tactile snapshot, `MuJoCoRobotDataProvider` copies current
generalized coordinates into a separate `MjData`, calls position kinematics, and
publishes named world poses for `Palm` and the 20 child links. This does not step
or mutate the live simulation. The common state records each position,
WXYZ quaternion, parent name, and source timestamp.

`VisualAssets` parses the provider's URDF and exposes an allowlisted manifest at
`/robot-assets/manifest.json`. The manifest maps 21 link names to the original
STL files and includes URDF visual origins/scales plus all 18 calibrated sensor
mounts. Only resolved `.stl` files beneath the project's `assets/` directory are
served. Unknown asset paths return 404.

The browser loads each STL once, leaves its vertices in the supplied link frame,
and parents it to the corresponding named link group. Every incoming state only
updates those groups' streamed world transforms. Sensor surfaces are children of
the same groups. Their link-local center and U/V/normal matrix is applied
directly so calibration values are not silently re-orthogonalized.

The scene layers are:

- original right-hand STL visuals;
- the active sphere object;
- cyan finite sensor surfaces, changing to red from the same `active` value used
  by numerical telemetry;
- yellow current estimated contacts and violet accumulated tactile points;
- cyan estimated sphere and optional green evaluation-only ground truth.

Orbit, pan, and zoom modify only the Three.js camera. They never send robot
commands. An adaptive render interval retains the original high-resolution CAD
while preventing slow/software WebGL renderers from starving telemetry.

## Replay and compatibility

New recordings store named link poses, and replay forwards them unchanged to the
same viewer. Older 1.0.0 recordings without that additive field use Python FK as
a compatibility fallback. Reconstruction consumes joint state, scalar tactile
measurements, and calibration only; display poses cannot influence its result.

Three.js is vendored under `ui/vendor/three/` so the dashboard needs no CDN or
network access. Its upstream license is retained in that directory.

## Optional MuJoCo comparison view

Run:

```bash
python -m backend.dashboard_server --debug-native-viewer
```

This opens MuJoCo's passive native viewer against the same live simulation model
and data. It is disabled by default, simulation-only, and exists solely for
engineering comparison. It does not replace or feed the browser renderer.

## Validation

Normal tests verify the exact original asset mapping, named pose timestamps,
MuJoCo pose equality, lack of simulation mutation, sensor transforms, replay,
legacy fallback, reconstruction isolation, and debug-view default. Opt-in
Playwright tests load the real STL files in Chromium, compare every served file
hash with its URDF source, compare rendered link/sensor transforms with the
stream, exercise view-only camera controls, verify active/overlay presentation,
and verify stale/disconnected behavior.
