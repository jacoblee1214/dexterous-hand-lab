from dataclasses import replace
from pathlib import Path
from queue import Queue
import shutil

import numpy as np
import pytest


mujoco = pytest.importorskip("mujoco")

from reconstruction.contact_point_buffer import ContactPointBuffer
from reconstruction.sphere_fitting import SphereReconstructionSession
from sensors.pressure_emulator import PressureSensorEmulator
from sensors.sensor_kinematics import load_sensor_mounts
from simulation.grasp_presets import GraspPresetLibrary, SphereGraspController
from simulation.model_builder import HAND_VARIANTS
from simulation.mujoco_sim import (
    HandSimulation,
    ObjectController,
    OverlayState,
    PRESSURE_CONFIG,
    RIGHT_GRASP_CONFIG,
    finger_joint_groups,
    service_hand_control_panel,
)


def make_grasp_system(config_path=RIGHT_GRASP_CONFIG):
    simulation = HandSimulation(HAND_VARIANTS["right"].output)
    objects = ObjectController(simulation, HAND_VARIANTS["right"].sphere_position)
    objects.activate("sphere")
    library = GraspPresetLibrary(config_path, HAND_VARIANTS["right"].urdf)
    controller = SphereGraspController(simulation, objects, library)
    controller.select_for_radius(objects.sphere_radius)
    return simulation, objects, library, controller


def test_required_presets_have_20_targets_with_negative_thumb_opposition():
    _, _, library, _ = make_grasp_system()
    assert library.opposition_joint == "joint_00"
    assert library.secondary_opposition_joint == "joint_01"
    for radius_mm in (30, 40, 50, 60):
        preset = library.presets[f"sphere_{radius_mm}mm"]
        assert len(preset.joint_targets) == 20
        assert preset.joint_targets[library.opposition_joint] < 0
        assert preset.joint_targets[library.secondary_opposition_joint] < 0
        for joint, target in preset.joint_targets.items():
            lower, upper = library.joint_limits[joint]
            assert lower <= target <= upper


def test_staged_grasp_changes_actuator_targets_without_overwriting_hand_qpos():
    simulation, _, library, controller = make_grasp_system()
    hand_qpos_before = simulation.data.qpos[:20].copy()
    controller.start()
    np.testing.assert_allclose(simulation.data.qpos[:20], hand_qpos_before)
    opposition_actuator = mujoco.mj_name2id(
        simulation.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "servo_joint_00"
    )
    thumb_flex_actuator = mujoco.mj_name2id(
        simulation.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "servo_joint_02"
    )
    assert simulation.data.ctrl[opposition_actuator] == pytest.approx(
        simulation.reference_positions()["joint_00"]
    )
    while simulation.data.time <= library.stage_duration_seconds:
        simulation.step()
        controller.update()
    assert simulation.data.ctrl[opposition_actuator] == pytest.approx(
        controller.selected_preset.joint_targets["joint_00"]
    )
    assert simulation.data.ctrl[opposition_actuator] < 0
    assert simulation.data.ctrl[thumb_flex_actuator] == pytest.approx(
        simulation.reference_positions()["joint_02"]
    )
    controller.apply_through_stage(4)
    assert simulation.data.ctrl[thumb_flex_actuator] == pytest.approx(
        controller.selected_preset.joint_targets["joint_02"]
    )
    assert not controller.running
    assert len(library.stages) == 4


def test_current_20_joint_target_vector_can_be_saved_and_reloaded(tmp_path):
    config = tmp_path / "grasp_config_right.yaml"
    shutil.copyfile(RIGHT_GRASP_CONFIG, config)
    simulation, _, _, controller = make_grasp_system(config)
    targets = controller.selected_preset.joint_targets
    simulation.set_actuator_targets(dict(targets))
    simulation.set_actuator_targets({"joint_01": -0.72, "joint_02": 0.88})
    saved = controller.save_current_targets("my_30mm_grasp")
    assert len(saved.joint_targets) == 20
    assert saved.joint_targets["joint_01"] == pytest.approx(-0.72)
    controller.reload()
    loaded = controller.select_named("my_30mm_grasp")
    assert loaded.joint_targets["joint_02"] == pytest.approx(0.88)


class PanelCapture:
    def __init__(self):
        self.commands = Queue()
        self.last_state = None

    def publish(self, state):
        self.last_state = state


def test_live_telemetry_reports_every_channel_and_matches_active_state():
    simulation, objects, _, controller = make_grasp_system()
    mounts = load_sensor_mounts(HAND_VARIANTS["right"].sensors)
    pressure = PressureSensorEmulator(simulation, mounts, PRESSURE_CONFIG)
    readings = pressure.read()
    sensor_id = "Finger01_Link02_Sensor"
    readings[sensor_id] = replace(
        readings[sensor_id],
        normal_contact_force_n=0.5,
        scalar_output_n=0.5,
        pressure_pa=1200.0,
        indentation_m=0.001,
    )
    panel = PanelCapture()
    state = OverlayState(selected_sensor=list(mounts).index(sensor_id))
    service_hand_control_panel(
        panel,
        simulation,
        finger_joint_groups("right", mounts),
        reconstruction_state=state,
        point_buffer=ContactPointBuffer(),
        sensor_readings=readings,
        sphere_session=SphereReconstructionSession(),
        object_controller=objects,
        mounts=mounts,
        current_estimates=[],
        grasp_controller=controller,
    )
    summary = panel.last_state["reconstruction"]
    assert len(panel.last_state["joint_targets"]) == 20
    assert summary["physical_sensor_count"] == 18
    assert summary["currently_active_count"] == 1
    assert summary["unique_active_sensor_count"] == 1
    assert summary["active_finger_count"] == 1
    assert len(summary["sensor_telemetry"]) == 18
    selected = next(row for row in summary["sensor_telemetry"] if row["sensor_id"] == sensor_id)
    assert selected["active"] is True
    assert selected["scalar_value_n"] == pytest.approx(0.5)
    assert selected["normal_force_n"] == pytest.approx(0.5)
    assert selected["pressure_pa"] == pytest.approx(1200.0)
    assert selected["indentation_m"] == pytest.approx(0.001)
    assert len(selected["position_xyz"]) == 3
    assert len(selected["sensing_direction_xyz"]) == 3


def test_40mm_grasp_reaches_multiple_sensor_families_and_settles():
    simulation, objects, _, controller = make_grasp_system()
    objects.set_sphere_radius(0.040)
    controller.select_for_radius(0.040)
    mounts = load_sensor_mounts(HAND_VARIANTS["right"].sensors)
    pressure = PressureSensorEmulator(simulation, mounts, PRESSURE_CONFIG)
    controller.start()
    seen = set()
    sphere_geom = objects._ids("sphere")[2]
    minimum_distance = 0.0
    for _ in range(2600):
        simulation.step()
        controller.update()
        seen.update(
            sensor_id
            for sensor_id, reading in pressure.read().items()
            if reading.active
        )
        distances = [
            simulation.data.contact[index].dist
            for index in range(simulation.data.ncon)
            if sphere_geom
            in (
                simulation.data.contact[index].geom1,
                simulation.data.contact[index].geom2,
            )
        ]
        if distances:
            minimum_distance = min(minimum_distance, min(distances))
    contributing_fingers = {mounts[sensor_id].finger_id for sensor_id in seen}
    assert len(seen) >= 3
    assert len(contributing_fingers) >= 2
    assert minimum_distance > -0.003
    assert np.max(np.abs(simulation.data.qvel)) < 0.01
