# User-taught right-hand grasp

The dashboard now treats the user's visible MuJoCo posture as the authority. It
does not infer thumb opposition from a joint-coordinate sign. The original
right URDF, approved 18-channel sensor registry, pressure calibration, and
reconstruction code are unchanged.

## Exact 20-joint mapping

Every source joint rotates about its URDF-local `+Z` axis and maps one-to-one to
the named MuJoCo position actuator `servo_<joint name>`. Positive motion is the
right-hand-rule rotation about that local `+Z`; negative motion is its exact
inverse. Since each child frame has a different orientation, the sign alone
must not be called “flexion” or “opposition.” Use the individual slider and the
MuJoCo Render to approve the visible physical direction.

| Joint | Parent → child | Physical role to inspect | Axis | Limits (rad) | Actuator |
|---|---|---|---|---:|---|
| `joint_00` | Palm → thumb_0 | thumb base placement/sweep | 0 0 1 | -0.0349066 … 1.6580628 | `servo_joint_00` |
| `joint_01` | thumb_0 → thumb_1 | thumb opposition/sweep | 0 0 1 | -1.6580628 … 1.6580628 | `servo_joint_01` |
| `joint_02` | thumb_1 → thumb_2 | thumb proximal bend | 0 0 1 | -0.1745329 … 1.3089969 | `servo_joint_02` |
| `joint_03` | thumb_2 → thumb_3 | thumb distal bend | 0 0 1 | -0.1745329 … 1.3962634 | `servo_joint_03` |
| `joint_10` | Palm → index_0 | index base spread | 0 0 1 | -0.3839724 … 1.3089969 | `servo_joint_10` |
| `joint_11` | index_0 → index_1 | index proximal bend | 0 0 1 | -0.0698132 … 1.5707963 | `servo_joint_11` |
| `joint_12` | index_1 → index_2 | index middle bend | 0 0 1 | -0.1745329 … 1.3962634 | `servo_joint_12` |
| `joint_13` | index_2 → index_3 | index distal bend | 0 0 1 | -0.1745329 … 1.3962634 | `servo_joint_13` |
| `joint_20` | Palm → middle_0 | middle base spread | 0 0 1 | -1.1344640 … 1.1344640 | `servo_joint_20` |
| `joint_21` | middle_0 → middle_1 | middle proximal bend | 0 0 1 | -0.0698132 … 1.5707963 | `servo_joint_21` |
| `joint_22` | middle_1 → middle_2 | middle middle bend | 0 0 1 | -0.1745329 … 1.3962634 | `servo_joint_22` |
| `joint_23` | middle_2 → middle_3 | middle distal bend | 0 0 1 | -0.1745329 … 1.3962634 | `servo_joint_23` |
| `joint_30` | Palm → ring_0 | ring base spread | 0 0 1 | -1.3089969 … 0.3839724 | `servo_joint_30` |
| `joint_31` | ring_0 → ring_1 | ring proximal bend | 0 0 1 | -0.0698132 … 1.5707963 | `servo_joint_31` |
| `joint_32` | ring_1 → ring_2 | ring middle bend | 0 0 1 | -0.1745329 … 1.3962634 | `servo_joint_32` |
| `joint_33` | ring_2 → ring_3 | ring distal bend | 0 0 1 | -0.1745329 … 1.3962634 | `servo_joint_33` |
| `joint_40` | Palm → pinky_0 | little-finger base spread | 0 0 1 | -1.3089969 … 0.4363323 | `servo_joint_40` |
| `joint_41` | pinky_0 → pinky_1 | little-finger proximal bend | 0 0 1 | -0.0698132 … 1.5707963 | `servo_joint_41` |
| `joint_42` | pinky_1 → pinky_2 | little-finger middle bend | 0 0 1 | -0.1745329 … 1.3962634 | `servo_joint_42` |
| `joint_43` | pinky_2 → pinky_3 | little-finger distal bend | 0 0 1 | -0.1745329 … 1.3962634 | `servo_joint_43` |

All actuators retain the URDF effort-derived range `[-0.38, 0.38] N·m`. The
dashboard joint table exposes the parent/child/axis in the joint-name tooltip.

## Teach workflow

1. Start `python -m backend.dashboard_server --open-browser`, expand
   **Individual Joint Targets & Grasp Presets**, and enable **Teach Grasp Pose**.
2. Select 30/40/50/60 mm, choose either no-object pose reproduction or the
   separately labelled supported free-sphere test, and reset to reference.
3. Move all 20 sliders while watching the actual MuJoCo image. Pause/resume is
   available from the existing simulation button.
4. Enter a unique name and waypoint role. Choose **Actual measured joints** only
   after `settled=YES`, or explicitly choose **Current actuator targets**.
5. Save. Loading a taught pose only selects it and causes no motion.
6. Enter an optional comma-separated sequence such as
   `my_open,my_thumb,my_approach,my_final`, set per-waypoint duration and joint
   tolerance, then execute. **Stop Taught Motion** immediately ends the
   trajectory and commands an actuator hold at the measured pose.

Each immutable saved record contains both actuator and measured vectors, the
selected reproduction vector, radius, role, UTC time, full URDF SHA-256, joint
configuration SHA-256, and grasp configuration version. Existing names are
rejected, and every write first backs up the full configuration.

Execution changes actuator targets only. Smooth interpolation is additionally
rate-limited to 0.8 rad/s, clamped by the URDF limits, and retains the existing
actuator force limits and contacts. It never assigns hand `qpos`.

The target/actual table reports taught target, current actuator target, actual
angle, error, velocity, force, and a measured cause category. `REPRODUCED` is
only a no-object pose result when all joint errors remain inside the selected
tolerance and velocity remains below the settling threshold. A sphere contact
test remains a separate physical-grasp evaluation and is never promoted to a
pose-reproduction result.

## Preserved backups

- `simulation/grasp_backups/grasp_config_right.yaml.20260908T0945-pre-teach-in.bak`
- `simulation/grasp_backups/joint_config_right.yaml.20260908T0945-pre-teach-in.bak`

Manual approval of the final user-taught posture remains required.
