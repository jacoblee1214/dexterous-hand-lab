# Four-quadrant browser dashboard

`index.html`, `styles.css`, and `app.js` implement a dependency-free browser UI
for the Python WebSocket backend. The two custom canvases render named 3D state
from the backend; the browser never runs physics, sensor emulation, kinematics,
or reconstruction algorithms.

The source badge is populated from the provider-neutral state (`SIMULATION`,
`REAL ROBOT`, or replay). Hand buttons send generic commands such as `grasp` and
`set_grasp_preset`; the frontend has no source-specific control path.

Start it from the repository root with:

```bash
python -m backend.dashboard_server --open-browser
```

Without `--open-browser`, visit `http://127.0.0.1:8000`. Ground truth is absent
from state messages until the user explicitly enables it.

The status banner distinguishes source health from WebSocket connectivity.
Invalid/stale data is labeled historical, canvas displays dim, live ACTIVE
highlights clear, and hardware motor buttons remain disabled. Missing optional
force/indentation samples are omitted from plots instead of plotted as zero.
All four quadrants carry the same snapshot timestamp for integration validation.
