"""Actual MuJoCo RGB rendering, overlay and latest-frame transport checks."""

from dataclasses import replace
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import pytest

from backend.dashboard_server import DashboardSimulation
from backend.mujoco_offscreen import LatestFrameBuffer, MuJoCoOffscreenRenderer, RenderedFrame
from robot_data.mujoco_provider import MuJoCoRobotDataProvider


@pytest.fixture
def rendering():
    provider = MuJoCoRobotDataProvider()
    engine = DashboardSimulation(provider=provider)
    renderer = MuJoCoOffscreenRenderer(provider, width=640, height=480)
    try:
        yield provider, engine, renderer
    finally:
        renderer.close()
        engine.close()


def test_actual_mujoco_rgb_contains_hand_sphere_and_sensor_surfaces(rendering):
    provider, engine, renderer = rendering
    before = provider.simulation.data.qpos.copy()
    rgb = renderer.render_rgb(engine)
    names = [
        renderer.mujoco.mj_id2name(renderer.model, renderer.mujoco.mjtObj.mjOBJ_GEOM, index)
        for index in range(renderer.model.ngeom)
    ]
    assert sum(bool(name and name.startswith("visual_")) for name in names) == 21
    assert "sphere_object_geom" in names
    assert rgb.shape == (480, 640, 3) and rgb.dtype == np.uint8
    assert rgb.std() > 20
    assert np.count_nonzero((rgb[:, :, 1] > 170) & (rgb[:, :, 2] > 170)) > 1000
    assert renderer.data is not provider.simulation.data
    np.testing.assert_array_equal(provider.simulation.data.qpos, before)


def test_active_channel_changes_same_mujoco_sensor_surface_red(rendering):
    _, engine, renderer = rendering
    inactive = renderer.render_rgb(engine)
    common = engine.current_common_state
    channels = list(common.tactile_channels)
    channels[0] = replace(channels[0], active=True)
    engine.current_common_state = replace(common, tactile_channels=tuple(channels))
    active = renderer.render_rgb(engine)
    red = lambda image: np.count_nonzero(
        (image[:, :, 0] > 180) & (image[:, :, 1] < 80) & (image[:, :, 2] < 80)
    )
    assert red(active) > red(inactive) + 500


def test_camera_commands_change_muJoCo_frame_and_validate_input(rendering):
    _, engine, renderer = rendering
    before = renderer.render_rgb(engine)
    renderer.camera_command({"action": "orbit", "azimuth_delta_deg": 25, "elevation_delta_deg": 10})
    after = renderer.render_rgb(engine)
    assert np.mean(np.abs(before.astype(float) - after.astype(float))) > 2
    renderer.camera_command({"action": "pan", "horizontal": .1, "vertical": -.1})
    renderer.camera_command({"action": "zoom", "factor": .8})
    renderer.camera_command({"action": "reset"})
    with pytest.raises(ValueError, match="Camera action"):
        renderer.camera_command({"action": "overwrite_qpos"})


def test_latest_frame_buffer_drops_old_frames_and_reports_metadata():
    buffer = LatestFrameBuffer()
    assert buffer.metadata()["status"] == "STARTING"
    first = RenderedFrame(1, 1.0, 2.0, "simulation", b"first")
    second = RenderedFrame(2, 1.1, 2.1, "simulation", b"second")
    buffer.publish(first)
    buffer.publish(second)
    assert buffer.wait_after(-1, timeout=0) is second
    assert buffer.wait_after(2, timeout=0) is None
    assert buffer.metadata()["frame_sequence"] == 2
    buffer.fail("render failed")
    assert buffer.metadata()["status"] == "ERROR"
    assert buffer.metadata()["error"] == "render failed"
