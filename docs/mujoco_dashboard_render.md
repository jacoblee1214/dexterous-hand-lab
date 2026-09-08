# MuJoCo dashboard rendering

`MuJoCo Render` is the default top-left mode for simulation and replay. The
underlying RGB image is produced by MuJoCo's supported Python offscreen renderer,
not by browser geometry, screen capture, VNC, or a native viewer window.

## Rendering pipeline

On Linux the default backend is EGL, verified with the installed MuJoCo 3.12.0
package and a directly created EGL context. The backend, resolution, rate and
JPEG quality are explicit in
`backend/dashboard_config.yaml`:

```yaml
mujoco_render:
  mode: mujoco
  backend: egl
  width: 640
  height: 480
  fps: 20
  jpeg_quality: 85
```

Simulation rendering shares the authoritative `MjModel` but owns a dedicated
`MjData`. On the single dashboard asyncio thread, `mj_copyData` snapshots the
live state before scene update and rendering. Physics never runs concurrently
with that copy/render operation, and the renderer never advances its data. This
avoids a drifting second simulation and avoids concurrent `mjData` access.

Replay owns a renderer-only copy of the same generated MuJoCo visual model. It
never steps physics: every frame overwrites named joint coordinates, sphere pose,
radius and time from the recorded common state, then calls `mj_forward` solely
to produce visual transforms. Future real hardware continues to use the separate
mesh twin unless an actual camera or an explicitly approved visualization source
is provided.

MuJoCo creates the hand, sphere and floor pixels from the model's real visual
geometries. The same MuJoCo scene receives calibrated finite sensor surfaces,
red active-channel surfaces, yellow current estimated contacts, violet
accumulated tactile points, and a valid estimated sphere. These overlays use
dashboard reconstruction outputs but do not feed information back into physics
or reconstruction.

## Bounded MJPEG transport

Frames are JPEG-compressed and published through
`/mujoco-render/stream.mjpg`. `LatestFrameBuffer` holds exactly one frame: a new
frame replaces the old one and slow clients skip stale frames instead of
building latency. Each multipart frame includes sequence, simulation timestamp,
wall rendering timestamp and source headers. The synchronized WebSocket state
also publishes this metadata and the renderer status.

If context creation or rendering fails, the state reports `ERROR` with the real
exception. The browser displays that message over the MuJoCo view and does not
silently switch to the mesh fallback.

## Controls and debug viewer

Left drag, right drag and wheel gestures send explicit `render_camera` commands
for orbit, pan and zoom. `Reset Camera` resets only `MjvCamera`. Browser camera
commands never write generalized coordinates or actuator targets.

The separate native viewer remains opt-in:

```bash
python -m backend.dashboard_server --debug-native-viewer
```

The dashboard does not need this window. To switch explicitly to the old mesh
fallback, use the top-left `View` selector or start with `--render-mode mesh`.

## Standalone proof

Before dashboard integration, the renderer can be validated independently:

```bash
python -m backend.mujoco_offscreen \
  --output experiments/mujoco_offscreen_proof.png
```

The command checks for all 21 MuJoCo hand visual geoms and the sphere geom,
rejects blank/near-uniform RGB output, and saves the rendered image.
