# Dexterous Hand Lab

Dexterous Hand Lab is a research platform for dexterous robotic-hand simulation,
grasping, sparse scalar tactile sensing, calibrated RGB perception and shape
reconstruction. Its motivating question is whether inexpensive, spatially sparse
tactile observations can improve RGB reconstruction in regions hidden by the
hand. The current research snapshot provides controlled right-hand sphere
baselines and infrastructure for future real-hardware observation; it is not a
finished arbitrary-object or multimodal reconstruction system.

The platform contains right- and left-hand MuJoCo foundations, while the
validated reconstruction/dashboard experiment is currently the right hand: 20
independently actuated joints and the approved 18 physical scalar tactile
channels. MuJoCo owns physics, actuation and contacts. The browser displays the
actual MuJoCo STL render plus a separate calibrated research RGB stream, while
the reconstruction packages retain explicit algorithm/evaluation boundaries.

## What is included

- Preserved copies of the supplied `V6_Force_R` and `V6_Force_L` packages under
  `assets/V6_Force_R/` and
  `assets/V6_Force_L/`.
- Generated right- and left-hand MJCF models with 20 independently controlled
  hinge joints per hand.
- Exactly 18 physical sensing regions per hand: Link02/Link03 rectangular pads
  plus one broad curved fingertip region on each of five fingers, and three
  rectangular palm pads. There are no Link01 or Link04 pressure channels.
- Explicit link-local surface centers, U/V tangent axes, dimensions, outward
  normals, patch boundaries, direction arrows, IDs, and coordinate-frame overlays.
- A sensor-mount calibration viewer that edits those values directly while the
  supplied STL is visible and persists them without modifying STL/URDF assets.
- Explicit sphere, cylinder, and box selection. No object is active by default.
- Free, dynamic objects and collidable hand hulls, with a dedicated collision
  verification scene.
- Pure joint-state plus mount-transform sensor FK, isolated from MuJoCo contact
  ground truth.
- Configurable contact-force → scalar output → pressure/indentation emulation.
  Contacts outside a finite sensor region produce no sensor reading, and the
  reconstruction-facing reading type exposes no MuJoCo contact point or normal.
- A strict reconstruction boundary, articulated sensor FK, configurable signed
  indentation projection, a bounded/deduplicated temporal point buffer, and a
  separate one-way MuJoCo ground-truth evaluator for the right-hand sphere
  experiment.
- Two-stage sphere reconstruction (algebraic initialization plus robust
  geometric least squares), configurable observability checks, manual/auto fit,
  and separate estimated-sphere/GT-sphere overlays.
- Four URDF-validated, actuator-only sphere-grasp presets, staged grasp controls,
  persistent user-tuned presets, and an 18-channel live numerical sensor table.
- A versioned named-joint/named-sensor provider boundary with deterministic
  joint/tactile timestamp interpolation, a simulation adapter, documented
  read-only real-hardware ingestion interface, and record/replay support.
- A browser-embedded MuJoCo offscreen render of the actual 21 right-hand STL
  visuals, sphere, 18 calibrated sensor surfaces and reconstruction overlays,
  plus a selectable Three.js mesh fallback for hardware/debug use.
- A separate fixed, calibrated clean RGB observation stream, transparent
  blue-sphere segmentation, perspective known-radius center baseline, explicit
  unknown-radius scale ambiguity, archived replay frames, and optional projected
  tactile-contact diagnostics. See `docs/research_rgb_milestone1.md`.
- A deterministic known-radius geometric RGB/tactile fusion baseline with strict
  ground-truth isolation, visibility-aware silhouette residuals, representative
  and finite-patch tactile variants, explicit failure states, and a four-method
  dashboard comparison. The live UI now exposes exact camera calibration,
  residuals, weights, coverage, conditioning, uncertainty, initialization, and
  validity reasons. See `docs/research_fusion_milestone2.md` and
  `docs/camera_fusion_explainability.md`.

The invisible hand collision geometry uses provisional convex hulls generated
from the supplied high-resolution visual meshes. It prevents obvious hand/object
tunneling, but it is not calibrated tactile-surface geometry.

## Setup and automated checks

Use Python 3.10 or newer (the current snapshot was validated on Python 3.12).
Linux EGL is the tested headless rendering backend. Run from this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m simulation.model_builder --hand right
python -m simulation.model_builder --hand left
python -m simulation.mujoco_sim --hand right --headless-check
python -m simulation.mujoco_sim --hand left --headless-check
python -m simulation.mujoco_sim --hand right --idle-diagnostics
python -m simulation.mujoco_sim --hand left --idle-diagnostics
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` prevents unrelated ROS pytest plugins from
being injected on robotics workstations.

## Interactive viewer

Launch either hand with no object:

```bash
python -m simulation.mujoco_sim --hand right
python -m simulation.mujoco_sim --hand left
```

Select an initial object explicitly when wanted:

```bash
python -m simulation.mujoco_sim --hand right --object sphere
python -m simulation.mujoco_sim --hand right --object cylinder
python -m simulation.mujoco_sim --hand left --object box
```

Hand controls:

- `O`: open all joints to the reference pose
- `C`: close all joints toward the grasp target
- `H`: reset the hand to its reference pose
- `1`–`5`: toggle one finger between open and flexed targets
- `J` / `K`: select the previous/next one of all 20 joints and print its name,
  current angle, and experimental range
- `,` / `.`: decrease/increase the selected joint target within its range

Normal interactive and sensor-verification launches also open a separate actuator
control panel containing Open/Close/Reset buttons, five continuous finger-flexion
sliders, and an advanced scrollable panel for all 20 joints. Every interactive
control changes `data.ctrl` for a MuJoCo position actuator. Interactive code does
not continuously assign hand `qpos`; direct poses are confined to deterministic
joint/FK checks.

Object controls:

- `S`, `Y`, `B`: activate and reset the sphere, cylinder, or box
- `N`: remove the active object from the workspace
- `R`: reset the active object position
- arrow keys: move the active object in X/Y
- `PgUp` / `PgDn`: move the active object in Z

Sensor overlays:

- `V`: show/hide finite sensor surfaces and rectangular boundaries
- `P`: show/hide representative surface centers
- `D`: show/hide outward sensing-direction arrows
- `I`: show/hide sensor IDs
- `F`: show/hide sensor coordinate frames
- `[` / `]`: select a sensor and print its live surface transform and direction
- `G`: report that ground-truth visualization is unavailable outside the
  reconstruction experiment

Inactive surfaces are cyan and active pressure channels turn red. The selected
sensor report includes area, world pose/direction, normal contact force, scalar
output, pressure, and indentation. Calibration parameters (effective area,
virtual stiffness, activation threshold, force/pressure saturation, noise, and
region tolerance) are in `sensors/pressure_config.yaml`; noise defaults to zero.
Activation uses a configurable lower release threshold and short dropout hold so
a persistent contact does not blink off because of one solver timestep.

## Right-hand tactile sphere reconstruction

Run the current reconstruction milestone with exactly one sphere:

```bash
python -m simulation.mujoco_sim --hand right --reconstruction-experiment
```

This mode automatically spawns the sphere. Cylinder, box, and object removal
commands are disabled so the experiment stays within the milestone scope. Use
the normal actuator controls to articulate the hand and the object arrow/Page
Up/Page Down controls to position the sphere.

Reconstruction controls:

- `V`: show/hide pressure sensor surfaces
- `X`: enable/disable the red active-sensor highlight
- `E`: show/hide current estimated 3D contacts (yellow)
- `M`: show/hide accumulated estimated contacts (violet)
- `G`: show/hide MuJoCo ground-truth contacts and translucent green GT sphere
  (evaluation only, hidden by default)
- `A`: start/pause estimated-point accumulation
- `Z`: clear all violet accumulated points and pause accumulation; press `A` to
  resume. The current yellow contact remains visible while a sensor is active.
- `T`: cycle accumulated-point coloring through uniform, finger ID, sensor ID,
  and observation time
- `L`: save the ground-truth-free reconstruction run and a separate evaluation
  report
- `U`: fit a sphere from accepted tactile points
- `Q`: reset only the fitted sphere while retaining tactile points
- `W`: show/hide the translucent blue estimated sphere
- `0`: switch between `MANUAL FIT` (default) and throttled `AUTO FIT`
- `6` / `7` / `8` / `9`: select a 30/40/50/60 mm sphere, reset its pose, and
  clear incompatible point/reconstruction/evaluation buffers
- `Space`: run the configured staged `Sphere Grasp`

The control window adds `Sphere Grasp`, individual Stage 1–4 buttons, and a
preset-name field with `Save Current Grasp Pose`, `Load Named Preset`, and
`Reload YAML`. The supplied `sphere_30mm` through `sphere_60mm` targets live in
`simulation/grasp_config_right.yaml`, outside `mujoco_sim.py`, and are validated
against all 20 right-hand URDF joint limits. Both the first thumb joint
`joint_00` and the larger secondary opposition rotation `joint_01` use negative
targets.

The window uses separate `Grasp + Joint Control` and `Live Sensor Telemetry`
tabs. All 20 named joint sliders are retained immediately below the grasp
controls and synchronize to the active actuator targets. In particular,
`joint_00`/`joint_01` tune thumb opposition and `joint_02`/`joint_03` tune the
inward thumb curl. Adjusting these sliders and pressing `Save Current Grasp Pose`
saves exactly that 20-joint target vector.

Below those controls, the live table shows all 18 physical channels with sensor,
finger, parent link, active state, scalar value, normal force, pressure, and
indentation. Selecting a row shows its world/base position, sensing direction,
and current estimated 3D contact. The compact summary reports current active
sensors/fingers, accepted points, rejected duplicates, unique contributing
sensors/fingers, spatial spread, and fit status. Red highlighting and numerical
`ACTIVE` use exactly the same sensor state.

The selected-sensor report includes its scalar value, estimated pressure,
estimated indentation, FK sensor position/direction, and estimated contact
position. Ground-truth position and per-point error appear only while `G` is
enabled. The terminal reports count, mean, median, RMSE, and 95th percentile for
Euclidean, normal-direction, and tangential-plane errors separately. Evaluation
outputs never feed back into the reconstruction algorithm.

The reconstruction viewer starts with a free camera. Left-drag orbits,
right-drag pans, and the mouse wheel zooms. Sensor surfaces start visible, so the
first `V` press hides them and the next press shows them again. The `X` active
layer is independent: with normal surfaces hidden, active channels still appear
as solid red patches. The separate hand-control window also includes equivalent
display/accumulation buttons, live ON/OFF and point-count status, plus orbit,
zoom, and reset-view buttons; these work even when keyboard focus is not on the
MuJoCo viewer. The panel also shows raw observations, accepted points, rejected
duplicates, and currently active channel IDs.

Accumulation starts paused. A repeatable run is:

1. Choose the sphere radius with `6`–`9`; this also selects its canonical preset.
2. Press `Space` or `Sphere Grasp` and inspect each live telemetry row. Stage
   buttons can be used instead when debugging activation order.
3. Press `Z` to clear the previous run, then `A` to start accumulation.
4. Repeat or tune the grasp and collect spatially diverse contacts. Activating
   all 18 channels simultaneously is neither required nor desirable.
5. Press `A` to pause, then `U` to fit. `INSUFFICIENT_DATA` requests more input;
   `POOR_SPATIAL_COVERAGE` means the points are too local or degenerate to draw
   a confident sphere.
6. Inspect the blue estimate; optionally press `G` for evaluation-only GT.
7. Press `L` to save.

Choose and position the sphere before accumulating one run. Object pose is
deliberately unavailable to reconstruction, so moving the free sphere while a
single world-frame point cloud is accumulating mixes different object centers.
Changing radius clears the run automatically; after manually repositioning it,
press `Z` before accumulating a new run.

Detailed thumb coordinates, staged behavior, preset persistence, and telemetry
semantics are documented in
[`docs/right_sphere_grasp_and_telemetry.md`](docs/right_sphere_grasp_and_telemetry.md).

## Four-quadrant real-time dashboard

The integrated browser dashboard runs the same reconstruction pipeline through
the configured robot-data provider without opening MuJoCo's engineering viewer:

```bash
python -m backend.dashboard_server --open-browser
```

Alternatively omit `--open-browser` and visit `http://127.0.0.1:8000`. The four
live regions are Robot Hand Digital Twin, Live Sensor Telemetry, Sensor
Time-Series, and Object Reconstruction. The header identifies the active data
source. For simulation and replay, `MuJoCo Render` is the default top-left mode:
Python uses MuJoCo's EGL offscreen renderer and sends bounded, compressed MJPEG
frames containing the actual model visuals, sphere, calibrated sensor surfaces,
active highlights, current/accumulated contacts and valid estimated sphere. The
selectable `Mesh Digital Twin` remains available as an explicit fallback/debug
view and is the default for future hardware without a MuJoCo render source. All browser buttons send versioned named commands over
`ws://127.0.0.1:8765`; JavaScript contains no physics, pressure conversion, FK,
point accumulation, fitting, or evaluation implementation.

Left-drag, right-drag and the mouse wheel send explicit orbit, pan and zoom
camera commands to the backend MuJoCo camera. `Reset Camera` restores its
reference view. This movable Debug Camera is distinct from the fixed Research
RGB Camera. The optional research-camera frame/frustum is drawn only into the
debug scene; exact compiled intrinsics/extrinsics and transform direction appear
in the dashboard calibration panel. The collapsed joint section retains sliders for all 20 named
actuator targets; these send normal `set_joint_target` commands and never write
`qpos` from the browser.

That section also separates `Fixed-object contact diagnostic` from
`Free-object grasp (supported)`. Fixed mode holds the sphere at its known pose
for repeatable sensor/reconstruction checks and is never presented as a free
grasp. Free mode enables gravity and all MuJoCo contacts while a small visible
support holds the initially placed free-joint sphere. Reference plus four
stage buttons, unique named grasp save/load, and the full joint
target/actual/velocity/actuator-force table remain alongside the 20 sliders.
Every accepted named grasp save first backs up the YAML in
`simulation/grasp_backups/`; duplicate names are rejected.

Named trials are recorded with `Start Named Trial`, `Finish + Save Trial`, and
`Repeat Initial Conditions`. Reconstruction-side samples and evaluation-only
MuJoCo object/contact data are written to separate files under
`experiments/grasp_trials/<experiment-name>/`, while `summary.json` reports the
number and distribution of completed trials. See
[`docs/right_sphere_grasp_and_telemetry.md`](docs/right_sphere_grasp_and_telemetry.md)
for thumb joint roles, mode boundaries, motion segmentation, and saved fields.

The top-left controls open/grasp/reset the hand, select 30–60 mm spheres, and
start/pause physics. The reconstruction controls independently start/pause
accumulation, reset points, fit/reset the sphere, and toggle its estimate. GT is
off by default; enabling `Show Ground Truth` adds only the evaluation payload,
overlay, and errors. Time-series scope can follow the selected sensor, its
finger, or the currently active channels. Stream/history rates and bounds are in
`backend/dashboard_config.yaml`; the versioned contracts are
`backend/state_schema_v1.json` and `backend/control_schema_v1.json`.

The dashboard and reconstruction code consume only synchronized common state;
simulation arrays, contacts, and indices stay inside the simulation provider.
Record and replay the exact common stream without frontend changes:

```bash
python -m backend.dashboard_server --record-log experiments/common_state_stream.jsonl
python -m backend.dashboard_server --source replay --replay-log experiments/common_state_stream.jsonl
```

`python -m backend.dashboard_server --source hardware` opens a read-only
`REAL ROBOT · DISCONNECTED` dashboard awaiting a future transport adapter. It
generates no measurements and disables all real motor commands. Connection,
timestamp interpolation and stale-data handling are enforced in both provider
and dashboard. The camera observation schema is ready for a future real RGB
transport, but no hardware camera driver or fabricated image source exists.
Sensor mounting/calibration remains independent of simulation contact geometry. See
[`docs/robot_data_provider.md`](docs/robot_data_provider.md).

### Calibrated research RGB baseline

Choose `Research RGB Camera` in the top-left view selector to see the fixed,
clean 640×480 observation. It is independent of the movable MuJoCo Debug Render
and contains no sensor colors, contact points, reconstruction overlays, IDs or
depth. `vision/camera_calibration.yaml` records the camera/frame IDs, field of
view, zero-distortion model, fixed pose and calibration version; focal lengths
and world/camera transforms are derived from the compiled MuJoCo 3.12 camera.

The transparent vision-only baseline segments the deliberately blue sphere and
fits its 3D center with a perspective tangent-cone formulation when radius is a
declared prior. It never reports that supplied radius as estimated. With radius
unknown, it returns `SCALE_AMBIGUOUS` instead of inventing metric depth or scale.
The committed 40 mm run evaluates four controlled conditions: all masks were
produced, but three center fits were `FIT_UNRELIABLE`; mean IoU was 0.785 and the
successful centered condition had 2.65 mm center error. See
[docs/research_rgb_milestone1.md](docs/research_rgb_milestone1.md).

The native MuJoCo view remains an optional, separate engineering comparison
tool and is disabled by default:

```bash
python -m backend.dashboard_server --debug-native-viewer
```

It shares the live MuJoCo model/data but is never streamed into or used as the
primary dashboard renderer. The debug option is simulation-only. Mesh asset,
pose, calibration, replay, and rendering boundaries are documented in
[`docs/mujoco_dashboard_render.md`](docs/mujoco_dashboard_render.md) and
[`docs/mesh_digital_twin.md`](docs/mesh_digital_twin.md).

The installed Linux backend was verified with MuJoCo EGL. Resolution, bounded
render rate and JPEG quality are configured under `mujoco_render` in
`backend/dashboard_config.yaml`. Run the standalone render proof with:

```bash
python -m backend.mujoco_offscreen \
  --output experiments/mujoco_offscreen_proof.png
```

Automated Chromium checks are available separately from the scientific tests:

```bash
pip install -r requirements-validation.txt
python -m playwright install chromium
RUN_BROWSER_TESTS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_dashboard_browser.py
```

These exercise the browser scene, four synchronized panels, commands, telemetry,
fitting and stale/disconnected presentation. They are not manual collision or
sensor-mount acceptance. The native engineering viewer remains available via
`python -m simulation.mujoco_sim --hand right --reconstruction-experiment`.
Results and the current insufficient-data limitation of the 40 mm grasp are
documented in [docs/dashboard_validation.md](docs/dashboard_validation.md).

If non-default ports are used, start the backend with `--http-port` and
`--websocket-port`, then open the page with `?wsPort=<websocket-port>`.

The default reconstruction output is `experiments/right_sphere_run.json`; the
separate evaluation output is `experiments/right_sphere_run.evaluation.json`.
Choose another reconstruction path with:

```bash
python -m simulation.mujoco_sim --hand right --reconstruction-experiment \
  --reconstruction-output experiments/my_run.json
```

Point-buffer and sphere-fit settings are in
`experiments/reconstruction_config.yaml`:
`minimum_pressure_threshold` (Pa), `minimum_time_between_points` (seconds),
`maximum_point_count`, and `minimum_spatial_distance_between_points` (metres).
Spatial duplicate checks are performed per sensor so simultaneous contacts from
different fingers remain distinguishable in the common base/world frame.
The sphere section configures minimum points/sensors/fingers, spread, covariance
eigenvalue and condition gates, robust loss/scale, iteration limit, and new-point
count per auto refit. These are adjustable engineering gates, not scientific
constants.

The reconstruction JSON contains the estimated sphere result and coverage but no
MuJoCo sphere parameters. Only `.evaluation.json` contains GT center/radius,
center error, absolute/relative radius error, and estimated-point distances to
the estimated and GT sphere surfaces.

The reconstruction input is limited to timestamp, right-hand joint state,
scalar sensor values, the fixed sensor-mount file, and pressure/indentation
calibration. The `reconstruction/` package does not import MuJoCo and receives no
contact position/normal, object pose/mesh, or collision geometry. The synthetic
conversion and signed projection (`contact_projection_sign`, default `+1`) are
configured in `sensors/pressure_config.yaml` and can later be replaced by a
hardware calibration model.

### Scalar-sensor limitation

A finite-area scalar pressure sensor measures scalar normal loading but does not
directly identify the tangential contact location within its sensing surface.
The current representative estimate is the calibrated sensor reference position
plus estimated indentation along its calibrated sensing direction. MuJoCo
ground truth is never used to correct the tangential estimate.

The approved right-hand registry contains 18 channels: Link02, Link03, and tip
on each finger plus three palm sensors. Five Link04 channels are intentionally
absent by explicit user approval. Experiment startup prints the full registry
table and aborts on a config/runtime mismatch; see
[`docs/right_sensor_registry.md`](docs/right_sensor_registry.md).

## Sensor verification

The sensor-verification scene intentionally starts with no sphere or other
object. Move the hand interactively and check every surface from several camera
angles:

```bash
python -m simulation.mujoco_sim --hand right --sensor-verification
python -m simulation.mujoco_sim --hand left --sensor-verification
```

Each hand has exactly 13 rectangular pads plus five curved tip regions. A tip has
a representative outermost center for FK and an area-coordinate API for selecting
different points on its ellipsoid cap later; it is not modeled as one permanently
fixed contact point. The exact schema, convention, and calibration caveat are in
[`docs/sensor_mount_assumptions.md`](docs/sensor_mount_assumptions.md).

## Sensor-mount calibration

Use the calibration scene to correct a mount directly against the rendered STL.
It never spawns an object:

```bash
python -m simulation.mujoco_sim --hand right --sensor-mount-calibration
python -m simulation.mujoco_sim --hand left --sensor-mount-calibration
```

The selected surface is magenta. The calibration panel provides link and sensor
selection, parent-link XYZ center placement in 0.5 mm increments, XYZ orientation
adjustment in 2° increments, width/height adjustment, `Flip Normal`, and explicit
YAML saving. The MuJoCo window uses a free camera: left-drag rotates around the
hand, right-drag pans, and the mouse wheel zooms. The calibration panel also has
orbit, zoom, `Focus selected sensor`, and `Reset view` buttons. Rectangular and
`fingertip_surface` records are independently selectable. Keyboard equivalents
for sensor editing are printed at launch.

Saving writes the matching `sensors/sensor_config*.yaml`. The live calibration
overlay uses the edited mount directly, so model recompilation is not needed to
see an edit during that session. A normal launch regenerates the MJCF from the
saved file.

## Collision verification

These scenes explicitly spawn a free sphere and print its position, object-contact
count, and minimum contact distance while you close or move the hand:

```bash
python -m simulation.mujoco_sim --hand right --collision-verification
python -m simulation.mujoco_sim --hand left --collision-verification
```

The sphere is not fixed: contact can translate and rotate it. The cylinder and box
use the same dynamic free-body path. Inactive objects are transparent,
non-colliding, and parked outside the workspace.

## Idle stability diagnostics

Run the gravity-on and gravity-off audit with no object or motion command:

```bash
python -m simulation.mujoco_sim --hand right --idle-diagnostics
python -m simulation.mujoco_sim --hand left --idle-diagnostics
```

The audit initializes every actuator target from its actual joint position, runs
each scenario for five seconds, logs per-joint qpos/qvel/control/force and all
requested generalized forces, reports contacts and maximum contact force, then
prints the concise acceptance table. The updated CAD meshes overlap around their
mechanical pivots, so parent-child and palm/base-sleeve collision pairs are
explicitly excluded; external object collisions and unrelated link collisions
remain active. Automated PASS does not replace the required visual no-twitch
check.

The measured before/after diagnosis and exact offending link pairs are recorded
in [`docs/idle_diagnosis.md`](docs/idle_diagnosis.md).

## Deterministic joint verification

Test minimum, reference, midpoint, and maximum poses while auditing that only the
selected joint's downstream chain moves:

```bash
python -m simulation.mujoco_sim --hand right --joint-test --joint joint_11
python -m simulation.mujoco_sim --hand left --joint-test --joint joint_11
python -m simulation.mujoco_sim --hand right --joint-test --headless-check
python -m simulation.mujoco_sim --hand left --joint-test --headless-check
```

Omit `--joint` to test all 20 joints. Right-hand ranges/reference targets live in
`simulation/joint_config_right.yaml`; left-hand values live in
`simulation/joint_config_left.yaml`.

## Updated hand assets

The newly supplied source packages are copied in full to:

- `assets/V6_Force_R/` (`urdf/V6_Force_R.urdf`, `meshes/`, ROS files)
- `assets/V6_Force_L/` (`urdf/V6_Force_L.urdf`, `meshes/`, ROS files)

The copied source URDF/STL files are unchanged. Both updated models contain 21
links and 20 revolute joints named `joint_00` through `joint_43`, and the builder
preserves their valid source limits. Generated models are `simulation/hand.xml`
and `simulation/hand_left.xml`; sensor catalogs remain independent.

## Sensor configuration

Each entry in `sensors/sensor_config.yaml` or
`sensors/sensor_config_left.yaml` explicitly specifies:

- sensor/finger ID and parent link
- `sensor_type`: `rectangular` or `fingertip_surface`
- link-local `center_xyz`
- link-local unit `surface_u_axis` and `surface_v_axis`
- explicit `outward_normal = normalize(cross(U, V))`
- surface `width`, `height`, and thickness
- three ellipsoid radii for a fingertip region

No mount field is derived at runtime from a joint origin, link origin, joint
axis, or mesh bounding box. The checked-in right-hand centers were seeded from
the visible palmar surfaces and then manually calibrated by the user. Left-hand
centers/frames were mirrored into the updated left mesh-local coordinates and
remain deferred for later manual approval. The transforms are not measured CAD
values.
After editing either file, regenerate the corresponding model with
`python -m simulation.model_builder --hand right|left`.

## Supplied-asset discrepancies

The source assets are preserved unchanged under `assets/`.

1. The updated URDFs contain valid lower/upper/effort/velocity values, so the
   builder no longer overrides their ranges. Joint signs are rotations about
   differently oriented local axes and are not treated as semantic guarantees
   of flexion or thumb opposition; approve the visible direction in Teach Grasp
   Pose mode.
2. Mesh paths use ROS package URIs. The builder resolves
   them to the preserved local mesh directory.
3. The URDFs contain no actuators, sensor frames, sensor dimensions, or verified
   mount transforms. These are explicit simulation-only additions.
4. High-resolution visual meshes are reused as provisional convex collision
   hulls. They must not be interpreted as calibrated pressure surfaces.

## Acceptance status

Automated checks cover both model loads, all 40 downstream-chain audits, 18
surface transforms per hand, explicit U/V/normal invariants, idle stability,
calibration persistence, actuator-only controls, scalar API isolation, finite
sensor-region activation/rejection, default no-object state, dynamic sphere
collision, the strict reconstruction boundary, signed projection formula,
articulated sensor pose, active/inactive point generation, evaluator separation,
normal/tangential metrics, activation hysteresis, temporal accumulation,
pressure/time/spatial filtering, multi-finger records, maximum buffer size,
strict reconstruction serialization, separate evaluation serialization, and
buffer reset. They also cover four complete URDF-limited grasp presets, negative
right-thumb opposition, actuator-only staging, named-preset save/reload, all-18
channel telemetry, active-state agreement, and a settling multi-sensor 40 mm
grasp. The current automated suite reports 147 passed tests with three opt-in
browser tests skipped by default; those three Chromium/HTTP/WebSocket tests also
pass when run explicitly. Manual visual acceptance remains open: confirm the posture and red/table
agreement in the GUI, explore the sphere with several configurations, fit it,
and save a representative run before marking this milestone complete.
Dashboard checks additionally cover the versioned state groups, bounded series
selection, named-command validation, actuator-only browser commands, default GT
omission, optional evaluation exposure, and exactly four primary UI regions.
The browser also provides a user-taught grasp workflow with immutable named
poses, separately stored actuator/actual vectors, optional ordered waypoints,
bounded actuator-only reproduction, stop control, and target-versus-actual
diagnostics. See `docs/user_taught_grasp.md`. Final thumb/finger direction and
posture approval still require manual visual acceptance. The local `grasp`
taught pose is preserved, but its final physical grasp quality, collision and
penetration have not been scientifically validated.

## Known limitations and next direction

- Reconstruction is sphere-only. Cylinder, box, free-form surface and arbitrary
  object reconstruction are not implemented.
- The tactile baseline uses a representative point along each calibrated sensor
  normal; a scalar pad does not reveal tangential position within its area.
- Collision hulls and sensor mounts are engineering approximations, not measured
  hardware geometry. Right mounts were manually calibrated; left mounts remain
  pending visual approval.
- The color-based RGB mask is a controlled experimental assumption. In the
  four-condition 40 mm run, three of four known-radius fits were unreliable under
  offset/occlusion effects.
- The current RGB+tactile implementation is a geometric known-radius sphere
  baseline, not a final/general fusion method. There is no vision-based object
  tracker, learned reconstruction, or actual hardware transport/motor control.
- CAD/URDF/STL ownership and redistribution terms must be confirmed before any
  remote repository includes `assets/`; see
  [docs/repository_snapshot_plan.md](docs/repository_snapshot_plan.md).

The current milestone stops at explainability, paired evaluation, exploratory
sim-to-real sensitivity, and a read-only real-hardware protocol. It does not
start neural fusion or real motor control.

## Layout

```text
tactile_shape_reconstruction/
├── assets/                 # preserved right- and left-hand source assets
├── simulation/             # MJCF generators, models, runner, object definitions
├── sensors/                # right/left surface catalogs and shared FK
├── reconstruction/         # ground-truth-free contacts and sphere fitting
├── evaluation/             # one-way MuJoCo ground-truth comparison and metrics
├── vision/                 # calibrated RGB observation and transparent baseline
├── robot_data/             # simulation/replay/read-only hardware common state
├── backend/                # versioned WebSocket state/control server
├── ui/                     # four-quadrant browser dashboard
├── tests/                  # model, transforms, controls, and collision tests
└── experiments/            # temporal buffer configuration and saved sphere runs
```
