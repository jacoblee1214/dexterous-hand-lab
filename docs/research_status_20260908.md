# Research status snapshot — 2026-09-08

This document distinguishes implemented software from scientific or manual
acceptance. The proposed repository name is `dexterous-hand-lab`.

## Completed implementation milestones

- Preserved V6 right/left URDF and STL packages and generated MuJoCo models with
  20 named hinge joints per hand.
- Defined finite tactile regions and pressure/indentation emulation. The approved
  right-hand registry has exactly 18 scalar channels: Link02, Link03 and Tip for
  five fingers, plus three palm sensors; Link04 channels remain excluded.
- Added actuator-only interactive controls, per-joint controls, named staged
  grasps and immutable Teach Grasp Pose save/load/reproduction support.
- Isolated tactile reconstruction from MuJoCo ground truth: synchronized joint
  state plus scalar tactile channels and calibration produce estimated contacts,
  bounded temporal points and robust sphere fitting. Evaluation is one-way and
  stored separately.
- Added the four-quadrant browser dashboard with actual MuJoCo STL rendering,
  18-channel telemetry, time-series data, tactile contacts and reconstruction.
- Added named `Simulation`, `Replay` and read-only future `Hardware` providers,
  timestamp interpolation, stale/error states and external camera metadata.
- Added a separate calibrated fixed RGB camera, clean JPEG stream, controlled
  blue-object segmentation, perspective known-radius sphere-center fitting,
  explicit unknown-radius scale ambiguity, archived RGB replay and optional
  projected tactile-contact diagnostics.

## Latest verification actually run

On 2026-09-08, with MuJoCo EGL:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
147 passed, 3 skipped in 30.03 s

RUN_BROWSER_TESTS=1 ... .venv/bin/python -m pytest -q tests/test_dashboard_browser.py
3 passed in 16.41 s
```

The three normal-suite skips are the opt-in Chromium checks. They were executed
separately and passed. The live local server was also checked at ports 8000/8765:
build `rgb-baseline-20260908.1`, research MJPEG HTTP 200 with multipart JPEG,
latest RGB HTTP 200 at 640×480, matching UI/API identifiers and no Chromium page
errors. This establishes software/browser integration, not human visual approval
of every physical behavior.

## Scientific observations

The selected 40 mm RGB experiment contains four controlled conditions. Mean mask
IoU is 0.785. The centered reference fit passed with 2.65 mm center error and
2.37 px silhouette residual. The +20 mm offset, approach occlusion and settled
development grasp produced `FIT_UNRELIABLE`; the observed center-fit failure
rate is therefore 75%. Radius error is not reported because radius was supplied
as a prior. This baseline is not evidence of arbitrary-object generalization.

Existing tactile sphere outputs demonstrate the pipeline and its explicit
`INSUFFICIENT_DATA`, `POOR_SPATIAL_COVERAGE` and valid-result gates. Sparse local
contacts and scalar-pad tangential ambiguity remain real scientific limitations;
tests do not prove shape accuracy on hardware.

## Approved, preserved and pending

| Item | Current status |
| --- | --- |
| Right-hand 18-channel registry; no Link04 | User-approved and enforced |
| Right sensor mounting YAML | User-calibrated local configuration preserved |
| Left sensor mounts | Implemented/mirrored; manual calibration still pending |
| Local taught pose named `grasp` | Saved and backed up; final physical-grasp acceptance not established |
| 20-joint actuator and replay behavior | Automated integration verified |
| Browser Research RGB and Vision-only card | Actual Chromium integration verified |
| Collision hull fidelity/no-penetration | Provisional hulls; manual multi-view acceptance pending |
| Free-object reliable grasp | Engineering experiments exist; repeatable physical quality pending |
| Real hand/camera connection | Interface only; protocol, calibration and hardware unavailable |
| Final RGB+tactile fusion | Not implemented |

## Next research milestone

The next scientific step is RGB plus sparse tactile fusion. It requires a common
timestamp domain, calibrated `K` and `T_world_from_camera_cv`, per-contact spatial
uncertainty, explicit image/contact association, occlusion handling and a
measured object-pose history before moving-object observations share an object
frame. MuJoCo object pose, contact geometry and masks must remain evaluation-only.
No cylinder/box/free-form reconstruction, neural implicit method or real motor
control should be inferred from this snapshot.
