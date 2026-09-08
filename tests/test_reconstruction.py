from dataclasses import fields
import math
from pathlib import Path

import numpy as np
import pytest

from evaluation.contact_evaluator import (
    GroundTruthContactPoint,
    evaluate_contact_points,
)
from reconstruction.contact_point_buffer import (
    ContactPointBuffer,
    PointBufferConfig,
)
from reconstruction.contact_projection import (
    ContactPointReconstructor,
    project_contact_point,
)
from reconstruction.estimated_contact import EstimatedContactPoint
from reconstruction.reconstruction_input import ReconstructionInput
from reconstruction.temporal_observation import TemporalTactileObservation
from sensors.pressure_emulator import load_pressure_config
from sensors.sensor_kinematics import KinematicTree, load_sensor_mounts
from simulation.model_builder import HAND_VARIANTS


ROOT = Path(__file__).resolve().parents[1]
RIGHT = HAND_VARIANTS["right"]
SENSOR_ID = "Finger03_Link02_Sensor"


def _calibration() -> dict[str, dict[str, float]]:
    mounts = load_sensor_mounts(RIGHT.sensors)
    configs = load_pressure_config(ROOT / "sensors" / "pressure_config.yaml", mounts)
    return {
        sensor_id: {
            "effective_sensor_area_m2": config.effective_sensor_area_m2,
            "virtual_stiffness_n_per_m": config.virtual_stiffness_n_per_m,
            "minimum_activation_force_n": config.minimum_activation_force_n,
            "saturation_pressure_pa": config.saturation_pressure_pa,
            "contact_projection_sign": config.contact_projection_sign,
        }
        for sensor_id, config in configs.items()
    }


def _frame(
    timestamp: float,
    sensor_values: dict[str, float],
    joint_state: dict[str, float] | None = None,
) -> ReconstructionInput:
    return ReconstructionInput(
        timestamp=timestamp,
        joint_state=joint_state or {},
        sensor_values=sensor_values,
        sensor_config_path=str(RIGHT.sensors),
        calibration_config=_calibration(),
    )


def _estimate(position: tuple[float, float, float]) -> EstimatedContactPoint:
    return EstimatedContactPoint(
        timestamp=1.0,
        sensor_id=SENSOR_ID,
        finger_id="Finger03",
        parent_link="middle_1",
        sensor_position=(0.0, 0.0, 0.0),
        sensor_sensing_direction=(0.0, 0.0, 1.0),
        scalar_signal=1.0,
        estimated_pressure=1000.0,
        estimated_indentation=0.001,
        estimated_contact_position=position,
    )


def _observation(
    position: tuple[float, float, float],
    *,
    timestamp: float = 1.0,
    sensor_id: str = SENSOR_ID,
    finger_id: str = "Finger03",
    pressure: float = 1000.0,
) -> TemporalTactileObservation:
    return TemporalTactileObservation(
        timestamp=timestamp,
        sensor_id=sensor_id,
        finger_id=finger_id,
        parent_link="middle_1",
        joint_state_snapshot={"joint_20": 0.1},
        sensor_position=(0.0, 0.0, 0.0),
        sensor_sensing_direction=(0.0, 0.0, 1.0),
        scalar_sensor_value=1.0,
        estimated_pressure=pressure,
        estimated_indentation=0.001,
        estimated_contact_position=position,
    )


def test_reconstruction_boundary_has_exact_schema_and_no_ground_truth_dependency() -> None:
    assert {field.name for field in fields(ReconstructionInput)} == {
        "timestamp",
        "joint_state",
        "sensor_values",
        "sensor_config_path",
        "calibration_config",
    }
    assert {field.name for field in fields(EstimatedContactPoint)} == {
        "timestamp",
        "sensor_id",
        "finger_id",
        "parent_link",
        "sensor_position",
        "sensor_sensing_direction",
        "scalar_signal",
        "estimated_pressure",
        "estimated_indentation",
        "estimated_contact_position",
    }
    combined_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "reconstruction").glob("*.py"))
    ).lower()
    assert "import mujoco" not in combined_source
    assert "data.contact" not in combined_source
    assert "contact.pos" not in combined_source
    assert "contact.frame" not in combined_source
    assert "from evaluation" not in combined_source
    assert "import evaluation" not in combined_source

    invalid = _calibration()
    invalid[SENSOR_ID]["object_pose"] = 0.0
    with pytest.raises(ValueError, match="forbidden/unknown"):
        ReconstructionInput(
            timestamp=0.0,
            joint_state={},
            sensor_values={},
            sensor_config_path=str(RIGHT.sensors),
            calibration_config=invalid,
        )


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_projection_is_exactly_position_plus_signed_indentation_normal(sign: float) -> None:
    position = np.array([0.25, -0.5, 1.25])
    direction = np.array([0.0, 3.0, 4.0])
    indentation = 0.02
    expected = position + sign * indentation * direction / np.linalg.norm(direction)
    np.testing.assert_allclose(
        project_contact_point(position, direction, indentation, sign),
        expected,
        rtol=0.0,
        atol=1e-15,
    )


def test_reconstructed_sensor_pose_follows_articulated_parent_link() -> None:
    reconstructor = ContactPointReconstructor(RIGHT.urdf)
    rest = reconstructor.reconstruct(_frame(0.0, {SENSOR_ID: 1.0}, {"joint_20": 0.0}))[0]
    moved_joint_state = {"joint_20": 0.35, "joint_21": 0.4}
    moved = reconstructor.reconstruct(
        _frame(0.1, {SENSOR_ID: 1.0}, moved_joint_state)
    )[0]

    mounts = load_sensor_mounts(RIGHT.sensors)
    expected = KinematicTree.from_urdf(RIGHT.urdf).sensor_poses(
        mounts, moved_joint_state
    )[SENSOR_ID]
    assert not np.allclose(rest.sensor_position, moved.sensor_position)
    np.testing.assert_allclose(moved.sensor_position, expected.position, atol=1e-12)
    np.testing.assert_allclose(
        moved.sensor_sensing_direction, expected.sensing_direction, atol=1e-12
    )


def test_inactive_sensor_generates_no_estimated_contact() -> None:
    reconstructor = ContactPointReconstructor(RIGHT.urdf)
    assert reconstructor.reconstruct(_frame(0.0, {SENSOR_ID: 0.0})) == []


def test_one_active_sensor_generates_exactly_one_point_per_timestep() -> None:
    reconstructor = ContactPointReconstructor(RIGHT.urdf)
    first = reconstructor.reconstruct(_frame(1.0, {SENSOR_ID: 2.0}))
    second = reconstructor.reconstruct(_frame(2.0, {SENSOR_ID: 2.0}))
    assert len(first) == len(second) == 1
    assert first[0].timestamp == 1.0
    assert second[0].timestamp == 2.0
    assert first[0].sensor_id == SENSOR_ID
    parameters = _calibration()[SENSOR_ID]
    assert first[0].estimated_pressure == pytest.approx(
        2.0 / parameters["effective_sensor_area_m2"]
    )
    assert first[0].estimated_indentation == pytest.approx(
        2.0 / parameters["virtual_stiffness_n_per_m"]
    )


def test_multiple_active_scalar_channels_generate_multiple_estimated_points() -> None:
    sensor_values = {
        "Finger02_Link02_Sensor": 1.0,
        "Finger03_Link02_Sensor": 2.0,
        "Palm01_Sensor": 3.0,
    }
    contacts = ContactPointReconstructor(RIGHT.urdf).reconstruct(
        _frame(3.0, sensor_values)
    )
    assert len(contacts) == 3
    assert {contact.sensor_id for contact in contacts} == set(sensor_values)


def test_scalar_calibration_model_is_hardware_swappable() -> None:
    class HardwareCalibration:
        def estimate(self, scalar_signal: float) -> tuple[float, float]:
            assert scalar_signal == 1.25
            return 4321.0, 0.004

    requested_sensor_ids = []

    def factory(sensor_id, parameters):
        requested_sensor_ids.append(sensor_id)
        assert "contact_projection_sign" in parameters
        return HardwareCalibration()

    contact = ContactPointReconstructor(
        RIGHT.urdf, calibration_model_factory=factory
    ).reconstruct(_frame(4.0, {SENSOR_ID: 1.25}))[0]
    assert requested_sensor_ids == [SENSOR_ID]
    assert contact.estimated_pressure == 4321.0
    assert contact.estimated_indentation == 0.004


def test_ground_truth_evaluator_is_one_way_and_reports_required_metrics() -> None:
    estimated = [_estimate((1.0, 0.0, 0.0)), _estimate((2.0, 0.0, 0.0))]
    truth = [GroundTruthContactPoint(1.0, SENSOR_ID, (0.0, 0.0, 0.0))]
    report = evaluate_contact_points(estimated, truth)
    assert [error.euclidean_error_m for error in report.errors] == [1.0, 2.0]
    assert report.metrics.count == 2
    assert report.metrics.mean_error_m == pytest.approx(1.5)
    assert report.metrics.median_error_m == pytest.approx(1.5)
    assert report.metrics.rmse_m == pytest.approx(math.sqrt(2.5))
    assert report.metrics.percentile_95_error_m == pytest.approx(1.95)
    assert not any(
        field.name.startswith("ground_truth")
        for field in fields(EstimatedContactPoint)
    )


def test_point_buffer_accumulates_multiple_timesteps_and_reset_clears_all() -> None:
    point_buffer = ContactPointBuffer(
        PointBufferConfig(
            minimum_time_between_points=0.0,
            maximum_point_count=3,
            minimum_spatial_distance_between_points=0.0,
        )
    )
    point_buffer.start()
    point_buffer.add([_observation((1.0, 0.0, 0.0), timestamp=1.0)])
    point_buffer.add([_observation((2.0, 0.0, 0.0), timestamp=2.0)])
    assert len(point_buffer.points) == 2
    point_buffer.pause()
    point_buffer.add([_observation((3.0, 0.0, 0.0), timestamp=3.0)])
    assert len(point_buffer.points) == 2
    point_buffer.reset()
    assert point_buffer.points == ()


def test_simulator_adapter_exposes_only_the_strict_reconstruction_inputs() -> None:
    pytest.importorskip("mujoco")
    from sensors.pressure_emulator import PressureSensorEmulator
    from simulation.mujoco_sim import (
        HandSimulation,
        PRESSURE_CONFIG,
        make_reconstruction_input,
    )

    simulation = HandSimulation(RIGHT.output)
    mounts = load_sensor_mounts(RIGHT.sensors)
    pressure = PressureSensorEmulator(simulation, mounts, PRESSURE_CONFIG)
    frame = make_reconstruction_input(
        simulation, pressure, RIGHT.sensors, pressure.read()
    )
    assert set(frame.joint_state) == set(simulation.joint_names)
    assert set(frame.sensor_values) == set(mounts)
    assert all(isinstance(value, float) for value in frame.sensor_values.values())
    assert all(set(values) == {
        "effective_sensor_area_m2",
        "virtual_stiffness_n_per_m",
        "minimum_activation_force_n",
        "saturation_pressure_pa",
        "contact_projection_sign",
    } for values in frame.calibration_config.values())
