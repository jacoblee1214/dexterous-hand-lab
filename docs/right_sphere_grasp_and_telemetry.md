# Right-hand sphere grasp and live telemetry

The repeatable grasp configuration is `simulation/grasp_config_right.yaml`.
It contains explicit 20-joint actuator targets and sphere spawn positions for
30, 40, 50, and 60 mm radii. Targets are checked against the limits in the
right-hand source URDF whenever the experiment starts or reloads the file.

## Verified right-hand joint mapping

The source of truth is `assets/V6_Force_R/urdf/V6_Force_R.urdf`; the source
URDF is not modified to compensate for controller signs. All 20 revolute joints
rotate about local `+Z`, but each joint frame has a different URDF origin/RPY,
so equal coordinate signs do not imply equal physical world motion. Limits and
the `0.38 N·m` effort value are imported into MuJoCo as joint control ranges and
actuator force ranges.

| Joint family | Parent → child | Physical role | URDF limit (rad) | Grasp sign |
|---|---|---|---:|---|
| `joint_00` | Palm → thumb_0 | thumb base opposition/placement | -0.0349 … 1.6581 | small negative |
| `joint_01` | thumb_0 → thumb_1 | primary opposition rotation toward palm | -1.6581 … 1.6581 | negative |
| `joint_02` | thumb_1 → thumb_2 | proximal thumb flexion/curl | -0.1745 … 1.3090 | positive |
| `joint_03` | thumb_2 → thumb_3 | distal thumb flexion | -0.1745 … 1.3963 | positive |
| `joint_10` | Palm → index_0 | index spread/placement | -0.3840 … 1.3090 | preset-specific |
| `joint_11..13` | index chain | proximal-to-distal flexion | see URDF | positive closure |
| `joint_20` | Palm → middle_0 | middle spread/placement | -1.1345 … 1.1345 | preset-specific |
| `joint_21..23` | middle chain | proximal-to-distal flexion | see URDF | positive closure |
| `joint_30` | Palm → ring_0 | ring spread/placement | -1.3090 … 0.3840 | preset-specific |
| `joint_31..33` | ring chain | proximal-to-distal flexion | see URDF | positive closure |
| `joint_40` | Palm → pinky_0 | little-finger spread/placement | -1.3090 … 0.4363 | preset-specific |
| `joint_41..43` | pinky chain | proximal-to-distal flexion | see URDF | positive closure |

The first right-thumb joint has only about two degrees of available negative
travel. The observed inward opposition therefore uses a small negative
`joint_00` placement plus a larger negative `joint_01` rotation. `joint_02` and
`joint_03` then curl the proximal and distal thumb inward. This mapping was
checked against the URDF axes/frames and rendered kinematics; the final visual
direction still requires the user's manual acceptance in the live dashboard.

## Right thumb convention

The first right-thumb joint, `joint_00` (`Palm -> thumb_0`), has the URDF range
`[-0.0349066, 1.6580628]` radians. Every sphere preset commands its available
negative position. The following `joint_01` (`thumb_0 -> thumb_1`) has the wider
`[-1.6580628, 1.6580628]` range and is also negative in every preset to supply
the larger secondary opposition rotation. The loader rejects a right-hand
sphere preset when either configured opposition target is non-negative or any
of its 20 targets lies outside the URDF limit.

## Execution

`Sphere Grasp` first commands the open/reference targets and resets the sphere.
After a configured reference interval it applies four actuator-only stages,
making five observable phases including phase 0:

0. open/reference posture;
1. thumb opposition;
2. index and middle approach;
3. ring and little approach;
4. smooth final 20-joint compliant closure, then velocity-based settling.

No stage writes hand `qpos`. MuJoCo contacts remain enabled and constrain the
actuator-driven motion. The four stage buttons allow the same sequence to be
inspected manually.

The wide control window is split into `Grasp + Joint Control` and
`Live Sensor Telemetry` tabs. All 20 joint sliders remain directly below the
grasp controls. Their positions follow the live MuJoCo actuator targets, so the
thumb can be refined with `joint_00` through `joint_03` after applying a preset.
The supplied presets use stronger positive `joint_02`/`joint_03` flexion in the
final stage so the opposed thumb curls inward over the sphere instead of only
rotating at its base.

The browser preset-name field supports `Save Current Grasp Pose` and
`Load Named Grasp Pose`. Saving records the complete current `data.ctrl` target
vector, current sphere radius, and configured spawn position. A non-empty unique
name is mandatory; an existing name is rejected instead of overwritten. Before
every accepted save, the YAML is copied into `simulation/grasp_backups/`.
A saved right-hand pose must retain negative thumb opposition.

## Fixed diagnostic versus free grasp

`Fixed-object contact diagnostic` is the initial reconstruction mode. The
sphere is explicitly held at its configured pose so sensor force, contact
projection, and spatial coverage can be studied repeatably. It must never be
reported as a successful free grasp.

`Free-object grasp (supported)` enables gravity on the sphere and uses the
otherwise inactive cylinder object as a small, visible kinematic support. The
sphere remains a true free joint and all hand/sphere/support contacts stay
enabled. If its evaluation-only MuJoCo pose moves more than 2 mm or 5 degrees
during accumulation, the current point segment and fit are reset. That pose is
used only to reject mixed-frame accumulation and is never fed to contact
projection or sphere fitting. A future measured tracker implements
`robot_data.object_pose.ObjectPoseTracker`; no fabricated tracking is present.

## Named repeatable trials

Open `Individual Joint Targets & Grasp Presets`, choose a mode/radius/preset,
enter an experiment name, and use `Start Named Trial`. Execute the staged grasp,
then `Finish + Save Trial`. `Repeat Initial Conditions` restores the same mode,
radius, preset, reference pose, and object spawn for another numbered trial.

Outputs are written below `experiments/grasp_trials/<name>/`:

- `trial_NNN.reconstruction.json`: versions, synchronized timestamps, joint
  target/position/velocity, actuator force diagnostics, all 18 scalar tactile
  channels, accepted estimated contacts, and reconstruction result;
- `trial_NNN.evaluation.json`: MuJoCo object pose, penetration/contact diagnostics,
  and ground-truth evaluation, physically separate from reconstruction data;
- `summary.json`: trial count and distributions of accepted points, contributing
  sensors/fingers, and settling time.

## Telemetry interpretation

The live table contains every approved physical channel and displays sensor ID,
finger, parent link, active state, scalar output, raw normal force, pressure, and
estimated indentation. Selecting a row also shows its current world/base sensor
position, sensing direction, and estimated contact point.

Red surface highlighting and the table's `ACTIVE` state use the same
`ScalarSensorReading.active` property, so they cannot disagree. Scalar output,
pressure, and indentation return to zero after the configured release/hold
logic. Raw normal force can briefly be nonzero while the channel is inactive
when it is below the activation threshold; this is intentional and makes the
threshold behavior visible.

The compact summary distinguishes current contacts from accumulated diversity.
It reports current active channels/fingers as well as accepted point count,
unique contributing sensors/fingers, rejected duplicates, spatial spread, and
sphere reconstruction status. A useful experiment seeks spatial diversity, not
simultaneous activation of all 18 channels.
