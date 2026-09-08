import json
from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("mujoco")

from backend.dashboard_server import (
    CONTROL_SCHEMA_VERSION,
    DASHBOARD_BUILD_ID,
    STATE_SCHEMA_VERSION,
    DashboardSimulation,
    load_dashboard_config,
)


ROOT = Path(__file__).parents[1]


@pytest.fixture
def engine():
    return DashboardSimulation()


def message(command, **parameters):
    return {
        "schema_version": CONTROL_SCHEMA_VERSION,
        "command": command,
        "parameters": parameters,
        "request_id": "test",
    }


def test_dashboard_config_has_distinct_positive_logical_rates():
    config = load_dashboard_config()
    assert config.physics_hz == 500
    assert config.visualization_hz == 30
    assert config.sensor_telemetry_hz == 25
    assert config.plots_hz == 15
    assert config.maximum_samples_per_sensor > 0
    assert config.render_mode == "mujoco"
    assert config.render_backend == "egl"
    assert config.render_fps == 20


def test_versioned_state_has_four_panel_data_and_hides_gt_by_default(engine):
    state = engine.state()
    assert state["schema_version"] == STATE_SCHEMA_VERSION
    assert state["build"] == {"backend": DASHBOARD_BUILD_ID}
    assert state["message_type"] == "state"
    assert state["source"] == {"kind": "simulation", "display_name": "SIMULATION"}
    assert state["common_state_schema_version"] == "1.0.0"
    for group in (
        "simulation",
        "joints",
        "hand",
        "sensors",
        "current_estimated_contacts",
        "accumulated_points",
        "time_series",
        "sphere_reconstruction",
        "evaluation",
        "render",
    ):
        assert group in state
    assert len(state["joints"]) == 20
    assert len(state["sensors"]) == 18
    assert state["evaluation"] == {"enabled": False}
    assert "ground_truth" not in json.dumps(state).lower()
    json.dumps(state, allow_nan=False)


def test_render_camera_command_is_explicit_and_does_not_touch_qpos(engine):
    calls = []
    engine.render_controller = type("Camera", (), {"camera_command": lambda _, values: calls.append(values)})()
    before = engine.simulation.data.qpos.copy()
    acknowledgement = engine.handle_command(message("render_camera", action="orbit", azimuth_delta_deg=12))
    assert acknowledgement["command"] == "render_camera"
    assert calls == [{"action": "orbit", "azimuth_delta_deg": 12}]
    np.testing.assert_array_equal(engine.simulation.data.qpos, before)


def test_frontend_grasp_and_open_commands_change_targets_not_hand_qpos(engine):
    displaced = {
        name: 0.2 * engine.simulation.demonstration_target(index)
        for index, name in enumerate(engine.simulation.joint_names)
    }
    engine.simulation.set_joint_positions(displaced)
    qpos_before = engine.simulation.data.qpos[:20].copy()
    engine.handle_command(message("open_hand"))
    np.testing.assert_allclose(engine.simulation.data.qpos[:20], qpos_before)
    engine.handle_command(message("sphere_grasp"))
    np.testing.assert_allclose(engine.simulation.data.qpos[:20], qpos_before)
    assert engine.grasp.running


def test_generic_named_joint_and_grasp_commands_reach_simulation_provider(engine):
    engine.handle_command(message("set_joint_target", joint_name="joint_10", target_rad=0.3))
    target = next(item for item in engine.state()["joints"] if item["name"] == "joint_10")
    assert target["target_rad"] == pytest.approx(0.3)
    acknowledgement = engine.handle_command(message("grasp"))
    assert acknowledgement["command"] == "grasp"
    assert engine.grasp.running
    engine.handle_command(message("set_grasp_preset", radius_m=0.04))
    assert engine.objects.sphere_radius == pytest.approx(0.04)


def test_dashboard_commands_control_radius_accumulation_fit_and_pause(engine):
    engine.handle_command(message("set_sphere_radius", radius_m=0.04))
    assert engine.objects.sphere_radius == pytest.approx(0.04)
    assert engine.grasp.selected_preset_name == "sphere_40mm"
    engine.handle_command(message("start_accumulation"))
    assert engine.point_buffer.accumulating
    engine.handle_command(message("pause_accumulation"))
    assert not engine.point_buffer.accumulating
    engine.handle_command(message("fit_sphere"))
    assert engine.sphere_session.result is not None
    assert engine.state()["sphere_reconstruction"]["status"] == "INSUFFICIENT_DATA"
    engine.handle_command(message("reset_fit"))
    assert engine.sphere_session.result is None
    engine.handle_command(message("pause_simulation"))
    simulation_time = engine.simulation.data.time
    engine.step()
    assert engine.simulation.data.time == simulation_time


def test_gt_appears_only_through_optional_evaluation_payload(engine):
    reconstruction_before = engine.state()["sphere_reconstruction"]
    engine.handle_command(message("show_ground_truth", enabled=True))
    state = engine.state()
    assert state["evaluation"]["enabled"] is True
    assert state["evaluation"]["ground_truth_sphere"]["radius_m"] == pytest.approx(0.03)
    assert state["sphere_reconstruction"] == reconstruction_before
    engine.handle_command(message("show_ground_truth", enabled=False))
    assert engine.state()["evaluation"] == {"enabled": False}


def test_sensor_selection_and_time_series_scope_are_explicit(engine):
    sensor_id = "Finger03_Link03_Sensor"
    engine.handle_command(message("select_sensor", sensor_id=sensor_id))
    engine.handle_command(message("set_time_series_scope", scope="single_finger"))
    for _ in range(30):
        engine.step()
    engine._sample_history()
    state = engine.state()
    assert state["selected_sensor_id"] == sensor_id
    assert state["time_series"]["scope"] == "single_finger"
    assert {series["finger_id"] for series in state["time_series"]["series"]} == {"Finger03"}
    assert all(len(series["samples"]) <= state["time_series"]["maximum_samples_per_sensor"] for series in state["time_series"]["series"])


def test_bad_schema_unknown_commands_and_unknown_sensors_are_rejected(engine):
    with pytest.raises(ValueError, match="schema_version"):
        engine.handle_command({"schema_version": "0", "command": "open_hand"})
    with pytest.raises(ValueError, match="Unknown dashboard command"):
        engine.handle_command(message("overwrite_qpos", value=1))
    with pytest.raises(ValueError, match="Unknown sensor_id"):
        engine.handle_command(message("select_sensor", sensor_id="not-a-sensor"))


def test_browser_ui_defines_exactly_four_primary_quadrants():
    html = (ROOT / "ui" / "index.html").read_text(encoding="utf-8")
    assert html.count('class="quadrant ') == 4
    for title in (
        "Robot Hand Digital Twin",
        "Live Sensor Telemetry",
        "Sensor Time-Series",
        "Object Reconstruction",
    ):
        assert title in html
    javascript = (ROOT / "ui" / "app.js").read_text(encoding="utf-8")
    for forbidden in ("mj_step", "least_squares", "fit_sphere("):
        assert forbidden not in javascript
