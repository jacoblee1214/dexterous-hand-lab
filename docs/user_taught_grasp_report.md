# User-taught grasp milestone report

Date: 2026-09-08

## Implemented and verified

- Build: `teach-grasp-20260908.1`
- Full automated suite: `138 passed, 3 skipped`
- Opt-in Chromium/HTTP/WebSocket suite: `3 passed`
- Live server: PID `2526051`, ports `127.0.0.1:8000` and `127.0.0.1:8765`
- MuJoCo MJPEG endpoint: HTTP 200, multipart JPEG frames confirmed
- Browser evidence: `experiments/user-taught-grasp-dashboard.png`

The normal suite skips the three opt-in browser cases; those cases were also
run explicitly with Chromium and passed.

The verified workflow preserves all 20 named joint sliders and adds immutable
taught-pose save/load, explicit actuator-versus-measured save source, four
waypoint roles, ordered bounded interpolation, a user stop command, no-object
pose validation, separate supported-sphere physical validation, and per-joint
target/actual/error/velocity/force/cause telemetry.

The checked-in grasp file contains only the four existing automatic presets.
No user-approved pose was fabricated or saved by the automated tests. Test
writes used temporary copies.

## Manual acceptance pending

The user must still teach and approve the intended pose. Completion cannot be
claimed until the saved joint vector matches that visible posture, the no-object
reproduction is accepted, the sphere approach is observed, and thumb/finger
directions, oscillation, collision, and penetration are manually approved.
