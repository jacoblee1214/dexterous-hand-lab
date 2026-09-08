from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from sensors.sensor_kinematics import load_sensor_mounts
from simulation.model_builder import HAND_VARIANTS
from simulation.mujoco_sim import (
    HandSimulation,
    OverlayState,
    apply_calibration_camera_command,
    apply_calibration_command,
    reset_calibration_camera,
)
from simulation.sensor_calibration import SensorMountCalibration


ROOT = Path(__file__).resolve().parents[1]


def test_manual_calibration_edits_and_persists_explicit_surface_axes(tmp_path: Path) -> None:
    source = ROOT / "sensors" / "sensor_config.yaml"
    target = tmp_path / "sensor_config.yaml"
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    calibration = SensorMountCalibration(load_sensor_mounts(target), target)
    state = OverlayState()
    calibration.select_sensor("Finger03_Link02_Sensor")
    original = calibration.selected

    apply_calibration_command(calibration, state, ("move_center", 1, 0.0005))
    apply_calibration_command(calibration, state, ("rotate", 2, 2.0))
    apply_calibration_command(calibration, state, ("resize", "width", 0.0005))
    before_flip = calibration.selected.outward_normal.copy()
    apply_calibration_command(calibration, state, ("flip_normal",))
    apply_calibration_command(calibration, state, ("save",))

    reloaded = load_sensor_mounts(target)[original.sensor_id]
    assert reloaded.center_xyz[1] == original.center_xyz[1] + 0.0005
    assert reloaded.width == original.width + 0.0005
    np.testing.assert_allclose(reloaded.outward_normal, -before_flip, atol=1e-8)
    np.testing.assert_allclose(
        np.cross(reloaded.surface_u_axis, reloaded.surface_v_axis),
        reloaded.outward_normal,
        atol=1e-8,
    )
    raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert raw["schema_version"] == 3
    record = next(item for item in raw["sensors"] if item["sensor_id"] == original.sensor_id)
    assert "center_xyz" in record
    assert "surface_u_axis" in record
    assert "surface_v_axis" in record
    assert "outward_normal" in record
    assert "orientation_rpy" not in record


def test_calibration_camera_can_orbit_zoom_reset_and_focus_selected_sensor() -> None:
    simulation = HandSimulation(HAND_VARIANTS["right"].output)
    mounts = load_sensor_mounts(HAND_VARIANTS["right"].sensors)
    calibration = SensorMountCalibration(mounts, HAND_VARIANTS["right"].sensors)
    calibration.select_sensor("Finger03_Link02_Sensor")
    camera = SimpleNamespace(
        type=-1,
        fixedcamid=42,
        lookat=np.zeros(3),
        distance=1.0,
        azimuth=0.0,
        elevation=0.0,
    )

    reset_calibration_camera(camera, "right")
    assert camera.type == 0
    assert camera.fixedcamid == -1
    assert camera.azimuth == 135.0
    initial_distance = camera.distance
    apply_calibration_camera_command(
        camera, simulation, calibration, "right", ("camera", "orbit_right")
    )
    apply_calibration_camera_command(
        camera, simulation, calibration, "right", ("camera", "orbit_up")
    )
    apply_calibration_camera_command(
        camera, simulation, calibration, "right", ("camera", "zoom_in")
    )
    assert camera.azimuth == 145.0
    assert camera.elevation == -15.0
    assert camera.distance < initial_distance

    apply_calibration_camera_command(
        camera, simulation, calibration, "right", ("camera", "focus_selected")
    )
    mount = calibration.selected
    body_id = simulation.mujoco.mj_name2id(
        simulation.model, simulation.mujoco.mjtObj.mjOBJ_BODY, mount.parent_link
    )
    expected = simulation.data.xpos[body_id] + simulation.data.xmat[body_id].reshape(
        3, 3
    ) @ mount.center_xyz
    np.testing.assert_allclose(camera.lookat, expected)
    assert camera.distance <= 0.16
