# Digital twin milestone validation — 2026-09-07

Status: data interface and automated dashboard integration verified. Manual
collision, grasp quality and sensor mounting acceptance remain pending.

## Automated evidence

```bash
RUN_BROWSER_TESTS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
```

Result: **118 passed in 21.13 s** (the original 88 tests preserved, 28 new data
integration/timestamp/calibration checks, and 2 actual Chromium checks).
The normal suite without `RUN_BROWSER_TESTS=1` skips the two browser tests.
Browser validation requires `requirements-validation.txt` and Chromium installed
with `python -m playwright install chromium`.

The simulation test executes the current right-hand 40 mm sphere grasp for 2,600
physics steps (5.2 s), records common states, and checks every replay sample.
It verifies named actuator targets, 20 joint measurements, all 18 tactile
channels, contact projection, temporal accumulation, and fitting. Contact lists,
accumulated points and sphere-fit output are identical between simulation and
replay. Changing display-only object metadata does not change reconstruction.

Actual Chromium tests start local HTTP/WebSocket servers and interact with the
browser's radius, grasp, accumulation, sensor selection, fit, pause and camera
rotation/zoom controls. They verify four panels share one snapshot timestamp,
18 telemetry rows are present, the selected contact sensor has nonzero history,
and JavaScript emits no runtime errors. Source status fixtures test CONNECTING,
STALE and ERROR presentation. A real server shutdown tests DISCONNECTED. A
separate actual hardware-source server shows an empty, read-only dashboard with
no invented measurements. These tests do not represent a real hand connection.

Headless browser screenshots were inspected for layout/status presentation.
This is not manual interactive collision or tactile calibration verification.

## Current experiment limitation

A deterministic 40 mm grasp run produced 5 accepted points from 3 sensors across
2 fingers after 5.2 s. The fitter returned **INSUFFICIENT_DATA** (the unchanged
configuration requires at least 8 points). No valid sphere estimate was produced
in that run. Browser runs can collect slightly different counts depending on
when accumulation starts; they also correctly display the insufficient-data
result. A successful fitted sphere from this physical grasp has not been claimed.

The scientific synthetic-cloud tests still cover valid sphere fitting, poor
coverage, degeneracy and separation from evaluation ground truth. No thresholds,
sensor mounts, grasp presets or physics settings were changed to force a fit.

## Read-only hardware and timing

Tests verify exact named registries, encoder interpolation at tactile timestamps,
separate receive/measurement clocks, stale source packets, watchdog expiry,
future/backwards/duplicate timestamps, unavailable interpolation, excessive
encoder gaps, camera calibration requirements, source epoch restart, and
rejection of every motor command at both the provider and dashboard boundary.
Replacing an approved sensor with an excluded Link04 sensor is rejected even
when the total remains 18. A device session restart clears derived state and
closes any old-epoch recording before new measurements are shown.

Connection to an actual hand remains pending its real communication interface,
encoder/scalar mappings, calibration and clock specifications. Exact required
inputs and adapter hooks are listed in [robot_data_provider.md](robot_data_provider.md).
Real motor output, vision fusion and additional object reconstruction methods
remain outside this milestone.
