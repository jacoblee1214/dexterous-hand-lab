from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest


mujoco = pytest.importorskip("mujoco")

from sensors.pressure_emulator import (
    PressureSensorEmulator,
    ScalarSensorReading,
    load_pressure_config,
)
from sensors.sensor_kinematics import load_sensor_mounts
from evaluation.contact_evaluator import (
    collect_mujoco_ground_truth,
    evaluate_contact_points,
)
from reconstruction.contact_projection import ContactPointReconstructor
from simulation.model_builder import HAND_VARIANTS
from simulation.mujoco_sim import (
    HandSimulation,
    ObjectController,
    PRESSURE_CONFIG,
    make_reconstruction_input,
)


def _place_sphere_at_sensor(
    simulation: HandSimulation,
    objects: ObjectController,
    mount,
    offset_in_surface: np.ndarray,
) -> None:
    body_id = mujoco.mj_name2id(
        simulation.model, mujoco.mjtObj.mjOBJ_BODY, mount.parent_link
    )
    rotation = simulation.data.xmat[body_id].reshape(3, 3)
    surface_point = simulation.data.xpos[body_id] + rotation @ (
        mount.center_xyz + offset_in_surface
    )
    normal = rotation @ mount.outward_normal
    _, qpos_address, _ = objects._ids("sphere")
    # The test sphere radius is 30 mm; 29 mm creates a controlled 1 mm overlap.
    simulation.data.qpos[qpos_address : qpos_address + 3] = (
        surface_point + 0.029 * normal
    )
    simulation.data.qvel[:] = 0.0
    mujoco.mj_forward(simulation.model, simulation.data)


def test_pressure_configuration_creates_exactly_18_scalar_channels() -> None:
    mounts = load_sensor_mounts(HAND_VARIANTS["right"].sensors)
    configs = load_pressure_config(PRESSURE_CONFIG, mounts)
    assert len(configs) == len(mounts) == 18
    assert all(config.noise_standard_deviation_n == 0.0 for config in configs.values())
    assert {field.name for field in fields(ScalarSensorReading)} == {
        "sensor_id",
        "parent_link",
        "sensor_type",
        "surface_area_m2",
        "normal_contact_force_n",
        "scalar_output_n",
        "pressure_pa",
        "indentation_m",
    }  # No MuJoCo contact point/normal leaks into the reconstruction-facing API.


def test_activation_hysteresis_holds_brief_solver_contact_dropouts() -> None:
    simulation = HandSimulation(HAND_VARIANTS["right"].output)
    mounts = load_sensor_mounts(HAND_VARIANTS["right"].sensors)
    emulator = PressureSensorEmulator(simulation, mounts, PRESSURE_CONFIG)
    sensor_id = "Finger03_Link02_Sensor"

    assert emulator._stabilize_force(sensor_id, 0.06, 0.0) == pytest.approx(0.06)
    assert emulator._stabilize_force(sensor_id, 0.0, 0.02) == pytest.approx(0.06)
    assert emulator._stabilize_force(sensor_id, 0.03, 0.04) == pytest.approx(0.03)
    assert emulator._stabilize_force(sensor_id, 0.0, 0.08) == pytest.approx(0.03)
    assert emulator._stabilize_force(sensor_id, 0.0, 0.10) == 0.0
    assert emulator._stabilize_force(sensor_id, 0.06, 2.0) == pytest.approx(0.06)
    assert emulator._stabilize_force(sensor_id, 0.0, 0.0) == 0.0


def test_contact_on_patch_activates_only_corresponding_scalar_channel() -> None:
    simulation = HandSimulation(HAND_VARIANTS["right"].output)
    mounts = load_sensor_mounts(HAND_VARIANTS["right"].sensors)
    objects = ObjectController(simulation, HAND_VARIANTS["right"].sphere_position)
    objects.activate("sphere")
    emulator = PressureSensorEmulator(simulation, mounts, PRESSURE_CONFIG)
    sensor_id = "Finger03_Link02_Sensor"
    mount = mounts[sensor_id]
    _place_sphere_at_sensor(simulation, objects, mount, np.zeros(3))

    simulation.step()
    readings = emulator.read()
    active = [reading for reading in readings.values() if reading.active]
    assert [reading.sensor_id for reading in active] == [sensor_id]
    assert active[0].normal_contact_force_n > 0
    assert active[0].scalar_output_n > 0
    assert active[0].pressure_pa > 0
    assert active[0].indentation_m > 0


def test_right_sphere_scalar_pipeline_generates_and_evaluates_a_3d_point() -> None:
    variant = HAND_VARIANTS["right"]
    simulation = HandSimulation(variant.output)
    mounts = load_sensor_mounts(variant.sensors)
    objects = ObjectController(simulation, variant.sphere_position)
    objects.activate("sphere")
    emulator = PressureSensorEmulator(simulation, mounts, PRESSURE_CONFIG)
    sensor_id = "Finger03_Link02_Sensor"
    _place_sphere_at_sensor(simulation, objects, mounts[sensor_id], np.zeros(3))

    simulation.step()
    readings = emulator.read()
    frame = make_reconstruction_input(simulation, emulator, variant.sensors, readings)
    estimated = ContactPointReconstructor(variant.urdf).reconstruct(frame)
    ground_truth = collect_mujoco_ground_truth(simulation, mounts, emulator.configs)
    report = evaluate_contact_points(estimated, ground_truth)

    assert [point.sensor_id for point in estimated] == [sensor_id]
    assert report.metrics.count == 1
    assert np.isfinite(report.metrics.mean_error_m)


def test_contact_on_same_link_but_outside_patch_produces_no_signal() -> None:
    simulation = HandSimulation(HAND_VARIANTS["right"].output)
    mounts = load_sensor_mounts(HAND_VARIANTS["right"].sensors)
    objects = ObjectController(simulation, HAND_VARIANTS["right"].sphere_position)
    objects.activate("sphere")
    emulator = PressureSensorEmulator(simulation, mounts, PRESSURE_CONFIG)
    mount = mounts["Finger03_Link02_Sensor"]
    outside_point = mount.center_xyz + mount.surface_u_axis * (
        mount.width / 2 + emulator.configs[mount.sensor_id].region_tolerance_m + 0.001
    )
    assert not emulator.point_is_in_region(
        mount,
        outside_point,
        emulator.configs[mount.sensor_id].region_tolerance_m,
    )


@pytest.mark.parametrize("hand", ["right", "left"])
def test_every_physical_sensor_center_belongs_to_its_own_finite_region(hand: str) -> None:
    mounts = load_sensor_mounts(HAND_VARIANTS[hand].sensors)
    for sensor_id, mount in mounts.items():
        simulation = HandSimulation(HAND_VARIANTS[hand].output)
        emulator = PressureSensorEmulator(simulation, mounts, PRESSURE_CONFIG)
        assert emulator.point_is_in_region(
            mount, mount.center_xyz, emulator.configs[sensor_id].region_tolerance_m
        )


@pytest.mark.parametrize("hand", ["right", "left"])
def test_idle_reference_pose_settles_without_self_contact(hand: str) -> None:
    simulation = HandSimulation(HAND_VARIANTS[hand].output)
    objects = ObjectController(simulation, HAND_VARIANTS[hand].sphere_position)
    objects.activate(None)
    np.testing.assert_allclose(simulation.data.ctrl, simulation.data.qpos[:20], atol=0)
    max_contacts = 0
    for _ in range(1500):
        simulation.step()
        max_contacts = max(max_contacts, simulation.data.ncon)
    assert max_contacts == 0
    assert np.max(np.abs(simulation.data.qvel[:20])) < 1e-3
