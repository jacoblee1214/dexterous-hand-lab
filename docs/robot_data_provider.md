# Simulation / real-robot data boundary

The dashboard and reconstruction orchestration consume only the versioned
`CommonRobotState` defined in `robot_data/common.py`. Simulator model/data
arrays, contact IDs, geometry IDs, and actuator indices are confined to
`MuJoCoRobotDataProvider`.

## Common state

Every common state is keyed by named joints and named sensor IDs and contains:

- the tactile sample timestamp;
- joint positions and velocities interpolated at that timestamp;
- timestamped tactile channels with scalar value, pressure, active state, and
  optional force and indentation;
- optional timestamped, named link poses for display-only digital-twin replay;
- optional named joint targets, object/grasp presentation metadata, and camera
  availability metadata.

The JSON contract is `robot_data/common_state_schema_v1.json`. Additive fields
retain compatibility with existing 1.0.0 logs. `synchronization` preserves the
original bracketing encoder timestamps, tactile timestamp, and time-base name.
The optional camera channel carries calibrated image references (see below).

`TimestampSynchronizer` retains timestamped joint samples and linearly
interpolates both position and velocity to obtain `q(t_sensor)`. It refuses to
extrapolate outside the buffered interval. Simulation samples pass through this
same boundary even though their streams currently have equal timestamps.

## Provider responsibilities

`RobotDataProvider` exposes a provider description, a current timestamp,
`step()`, `read_common_state()`, and `send_command()`. Robot commands are named:

- `open_hand`
- `grasp`
- `reset`
- `set_joint_target`
- `set_grasp_preset`

`MuJoCoRobotDataProvider` translates these commands to simulation actuator
targets. `HardwareRobotDataProvider` accepts timestamped named measurements
through `accept_joint_state()` and `accept_tactile_state()`. There is no transport
implementation. The future `read_joint_state()` and `read_tactile_state()` I/O
hooks still raise `NOT_IMPLEMENTED`. Every hardware `send_command()` and
`send_joint_targets()` call raises `HardwareReadOnlyError` (code `READ_ONLY`),
including when receive data is streaming. The backend also rejects motor
commands before invoking the provider, and the UI disables those controls.

Sensor mounting remains in `sensors/sensor_config*.yaml`: parent link, explicit
link-local center and U/V/normal basis (the mounting transform), dimensions, and
sensor type. Scalar-to-pressure/indentation model parameters remain in
`sensors/pressure_config.yaml`. Neither format requires simulation contact
geometry, so a hardware adapter can publish the same calibrated IDs.

## Source selection, recording, and replay

The default in `backend/dashboard_config.yaml` is `source: simulation`.

Record a common-state stream while running the normal dashboard:

```bash
python -m backend.dashboard_server \
  --source simulation \
  --record-log experiments/common_state_stream.jsonl
```

Replay it through the unchanged dashboard and reconstruction pipeline:

```bash
python -m backend.dashboard_server \
  --source replay \
  --replay-log experiments/common_state_stream.jsonl
```

The JSON Lines log starts with a versioned provider/calibration description and
then stores common-state records. Keep the referenced URDF and sensor mounting
YAML unchanged alongside the log for repeatability; mounting files are not
embedded in this format. Object metadata is display-only and is not passed to
reconstruction. Ground-truth contacts are not recorded in this stream.

Simulation records each named link's world position and WXYZ quaternion at the
same tactile snapshot timestamp. Replay uses recorded joint state to pose a
renderer-only MuJoCo model for its default RGB view and also forwards recorded
link poses unchanged to the selectable browser mesh fallback. Older recordings
without `link_poses` remain readable and use the documented Python FK fallback
for that mesh view. Link poses are presentation data only; the reconstruction
adapter deliberately ignores them.

Replay processes each recorded sample, including intervals shorter than the
dashboard telemetry period. The live server schedules replay using recorded
timestamp intervals; `step()` advances exactly one sample for deterministic
offline evaluation. Reset rewinds time and clears accumulation/history gates.
Replay may reset playback but cannot actuate a robot.

Run the unconfigured real-robot dashboard with:

```bash
python -m backend.dashboard_server --source hardware
```

It displays `REAL ROBOT · DISCONNECTED`, empty measurements, and disabled motor
controls. It does not create a zero-pose robot or simulated pressure data.

## Read-only ingestion contract and connection states

The default description is loaded directly from the right-hand URDF, sensor
mounts and scalar calibration files. No simulator model is built or imported.
The approved registry is enforced by exact names: five fingers each have Link02,
Link03 and Tip, plus Palm01/Palm02/Palm03 (18 total). Link01/Link04 channels are
excluded. All 20 measured joints must be named; missing values never default to
zero. Sensor mount centers, dimensions, finite regions and sensing normals are
unchanged. Existing scalar calibration is preserved; these virtual calibration
coefficients still require physical validation before quantitative hardware use.

A future transport adapter must construct the provider with an explicit
`time_base` and `source_time_now` callable, call `begin_connection()`, and supply
validated `JointStateSample` and `TactileStateSample` objects to the accept hooks.
These are in-process interfaces, not a new wire protocol. Schedule accept hooks
on the dashboard's owning event-loop thread; use `loop.call_soon_threadsafe` to
marshal receive-thread callbacks. A tactile scan currently requires all 18
channels at one source timestamp; independently sampled channels require an
explicit future scan assembly contract, not relabeling reception timestamps.

| State | Meaning and presentation |
| --- | --- |
| DISCONNECTED | No configured transport, or explicit disconnect; data is unavailable. |
| CONNECTING | Connection begun; awaiting measurements/bracketing encoder samples. |
| STREAMING | Complete named measurements with valid synchronization and freshness. |
| STALE | Source measurement age or independent receive watchdog exceeds the limit. |
| ERROR | Invalid registry, timestamp, overflow, interpolation gap or transport error. |

Invalid input is rejected and flagged ERROR; restart with `begin_connection()`
after correcting the cause. Fresh valid samples can recover STALE. A new
connection session clears derived points/fit/history and pauses accumulation so
clock epochs and experiments cannot mix. If recording already contains samples,
it closes the old log and displays a message requesting a new log for the new
epoch; it never appends backwards timestamps to the existing stream.
The default maximum measurement age is 0.5 s and the maximum encoder bracket
gap is 0.1 s, both constructor parameters that must be chosen for actual hardware.
The pending tactile queue and encoder buffer are bounded. No extrapolation or
wrapping-angle interpolation is performed (the supplied hand has bounded
revolute coordinates). Future continuous encoders must be unwrapped upstream.

The dashboard stops reconstructing and appending history while data is invalid.
Previously received values may remain visible as explicitly labeled historical
data, with dimmed canvases and no live ACTIVE highlights. A separate browser
watchdog flags a silent WebSocket after 1.5 s; socket connectivity alone never
means the hand is streaming. All four panels share `snapshot.timestamp` and
`snapshot.valid`. Plot samples retain their individual acquisition timestamps;
the final plotted sample can precede the current snapshot because plots are
sampled at their own configured rate.

## Time-base requirements

All joint and tactile timestamps must be source measurement times in seconds in
the same explicitly named monotonic time domain. `source_time_now()` supplies
the current time in that domain using a known clock mapping, not the packet's
arrival time. A separate local monotonic clock measures receive liveness. Delayed
packets with old measurement times are rejected even if they just arrived.
Clock offsets/drift, ticks-to-seconds conversion, reboot/reset behavior and
uncertainty must be specified by the hardware integration. Future timestamps,
duplicates, backwards samples, out-of-buffer interpolation and excessive
bracket gaps are rejected. Relative timestamp tolerance is disabled, including
for large absolute epoch values. Simulation uses `simulation_seconds`; replay
retains original source timing and does not compare it to today's wall clock.

## Camera extension (no fusion)

`CameraObservation` supports `timestamp`, `frame_id`, `camera_id`, `source`,
`image_width`, `image_height`, `rgb_reference`, optional `depth_reference`,
`intrinsics`, versioned intrinsic/extrinsic calibration references and
`time_base`. References identify external image files/assets rather than putting
uncompressed pixels into state JSON. The calibrated research-camera path decodes
those archived RGB files during replay; unrelated providers may leave them as
metadata only. Intrinsics contain width/height, fx/fy/cx/cy and an explicit
distortion model (with coefficients where needed). Extrinsics identify a
versioned camera-to-robot transform with named frames. Image references require
explicit calibration and timing. Old availability-only camera records remain
readable but are insufficient for geometric use.

The hardware `accept_camera_observation()` hook checks the declared clock domain
and measurement age. Its observation retains its own timestamp when attached to
a later common state; attaching it does not establish synchronization for fusion.
Depth stays absent when not supplied. For future unregistered depth, separate
depth intrinsics, units and RGB/depth registration must be supplied before use.

## Exact information required from the real hand

1. Actual SDK/API specification, ROS topic/message definitions, or serial/CAN
   packet specification; read subscription/open/close/error behavior and an
   example recorded joint/tactile capture.
2. Mapping of each hardware encoder to the 20 URDF joint names; units, sign,
   zero offsets, mechanical reference pose, encoder wrap behavior, and whether
   velocities are measured or derived.
3. Mapping of every pressure input to the approved 18 sensor IDs; raw units,
   offsets/gains, scale, saturation, activation semantics, noise/filter behavior,
   effective area and scalar-to-pressure/indentation calibration.
4. Source timestamp fields for every joint state, tactile scan and optional
   camera frame; clock domains, tick frequency, epoch/reset/wrap behavior,
   synchronization method, clock-mapping uncertainty, rates, latency and jitter.
5. Robot base frame and mounting/calibration revision, plus acceptable maximum
   age and interpolation gap. Optional cameras additionally need image formats,
   intrinsic calibration and camera-to-robot extrinsics (and depth units if used).
6. Motor command protocol, safety limits, watchdogs and emergency-stop behavior
   for a **separately approved future control milestone**. None of these enable
   motor output in this read-only milestone.
