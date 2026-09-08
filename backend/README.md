# Real-time dashboard backend

`dashboard_server.py` owns provider-neutral orchestration, FK contact projection,
the temporal point buffer, sphere fitter, and WebSocket presentation. The
simulation and scalar sensor emulator are isolated behind
`MuJoCoRobotDataProvider`; no simulator array or index crosses into the
dashboard or reconstruction core.

```bash
python -m backend.dashboard_server --open-browser
```

Defaults are `http://127.0.0.1:8000` and `ws://127.0.0.1:8765`. Rates and bounded
history sizes are configured in `dashboard_config.yaml`. The contracts are
documented by `state_schema_v1.json` and `control_schema_v1.json`.

The backend accepts only named generic hand commands. The selected provider
translates them to its actuator target mechanism. Evaluation GT is omitted from
outgoing state unless `show_ground_truth` has been enabled. Provider selection,
record/replay, synchronization, read-only hardware ingestion, and the camera extension are
documented in [`../docs/robot_data_provider.md`](../docs/robot_data_provider.md).

`--source hardware` serves a disconnected read-only dashboard until an actual
receive adapter supplies timestamped data. Connection health and snapshot
validity apply to all four quadrants; old measurements never receive a live
status. See [validation results](../docs/dashboard_validation.md).
