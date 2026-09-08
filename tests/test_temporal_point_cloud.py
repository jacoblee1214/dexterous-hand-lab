from dataclasses import fields
import json
import math

import pytest

from evaluation.contact_evaluator import (
    GroundTruthContactPoint,
    evaluate_contact_points,
)
from evaluation.experiment_io import save_evaluation_dataset
from reconstruction.contact_point_buffer import (
    ContactPointBuffer,
    PointBufferConfig,
)
from reconstruction.estimated_contact import EstimatedContactPoint
from reconstruction.experiment_io import save_reconstruction_dataset
from reconstruction.temporal_observation import TemporalTactileObservation


def observation(
    timestamp: float,
    position: tuple[float, float, float],
    *,
    sensor_id: str = "Finger02_Link02_Sensor",
    finger_id: str = "Finger02",
    pressure: float = 1000.0,
) -> TemporalTactileObservation:
    return TemporalTactileObservation(
        timestamp=timestamp,
        sensor_id=sensor_id,
        finger_id=finger_id,
        parent_link="index_1",
        joint_state_snapshot={"joint_10": 0.2, "joint_11": 0.4},
        sensor_position=(0.0, 0.0, 0.0),
        sensor_sensing_direction=(0.0, 0.0, 1.0),
        scalar_sensor_value=1.0,
        estimated_pressure=pressure,
        estimated_indentation=0.2,
        estimated_contact_position=position,
    )


def estimate(position: tuple[float, float, float]) -> EstimatedContactPoint:
    return EstimatedContactPoint(
        timestamp=1.0,
        sensor_id="Finger02_Link02_Sensor",
        finger_id="Finger02",
        parent_link="index_1",
        sensor_position=(0.0, 0.0, 0.0),
        sensor_sensing_direction=(0.0, 0.0, 1.0),
        scalar_signal=1.0,
        estimated_pressure=1000.0,
        estimated_indentation=0.2,
        estimated_contact_position=position,
    )


def test_normal_and_tangential_error_decomposition() -> None:
    report = evaluate_contact_points(
        [estimate((0.0, 0.0, 0.2))],
        [
            GroundTruthContactPoint(
                1.0, "Finger02_Link02_Sensor", (0.3, 0.4, 0.5)
            )
        ],
    )
    error = report.errors[0]
    assert error.euclidean_error_m == pytest.approx(math.sqrt(0.34))
    assert error.normal_direction_error_m == pytest.approx(0.3)
    assert error.tangential_plane_error_m == pytest.approx(0.5)
    assert report.metrics.euclidean.mean_m == pytest.approx(math.sqrt(0.34))
    assert report.metrics.normal_direction.mean_m == pytest.approx(0.3)
    assert report.metrics.tangential_plane.mean_m == pytest.approx(0.5)


def test_temporal_and_spatial_duplicates_are_rejected_and_counted() -> None:
    point_buffer = ContactPointBuffer(
        PointBufferConfig(
            minimum_time_between_points=0.1,
            minimum_spatial_distance_between_points=0.01,
        )
    )
    point_buffer.start()
    point_buffer.add([observation(0.0, (0.0, 0.0, 0.0))])
    point_buffer.add([observation(0.05, (0.1, 0.0, 0.0))])
    point_buffer.add([observation(0.2, (0.005, 0.0, 0.0))])
    point_buffer.add([observation(0.3, (0.02, 0.0, 0.0))])
    assert len(point_buffer.points) == 2
    assert point_buffer.statistics.raw_observation_count == 4
    assert point_buffer.statistics.accepted_point_count == 2
    assert point_buffer.statistics.rejected_duplicate_count == 2


def test_pressure_threshold_rejects_low_pressure_observation() -> None:
    point_buffer = ContactPointBuffer(
        PointBufferConfig(
            minimum_pressure_threshold=500.0,
            minimum_time_between_points=0.0,
            minimum_spatial_distance_between_points=0.0,
        )
    )
    point_buffer.start()
    point_buffer.add([observation(0.0, (0.0, 0.0, 0.0), pressure=499.0)])
    point_buffer.add([observation(0.1, (0.0, 0.0, 0.0), pressure=500.0)])
    assert len(point_buffer.points) == 1
    assert point_buffer.statistics.rejected_pressure_count == 1


def test_multiple_fingers_accumulate_in_one_common_coordinate_frame() -> None:
    point_buffer = ContactPointBuffer(
        PointBufferConfig(
            minimum_time_between_points=0.0,
            minimum_spatial_distance_between_points=0.01,
        )
    )
    point_buffer.start()
    point_buffer.add(
        [
            observation(0.0, (0.1, 0.2, 0.3)),
            observation(
                0.0,
                (0.1, 0.2, 0.3),
                sensor_id="Finger04_Link02_Sensor",
                finger_id="Finger04",
            ),
        ]
    )
    assert {point.finger_id for point in point_buffer.points} == {
        "Finger02",
        "Finger04",
    }
    assert all(
        point.estimated_contact_position == (0.1, 0.2, 0.3)
        for point in point_buffer.points
    )


def test_maximum_point_count_evicts_oldest_points() -> None:
    point_buffer = ContactPointBuffer(
        PointBufferConfig(
            minimum_time_between_points=0.0,
            maximum_point_count=2,
            minimum_spatial_distance_between_points=0.0,
        )
    )
    point_buffer.start()
    for index in range(3):
        point_buffer.add(
            [observation(float(index), (float(index), 0.0, 0.0))]
        )
    assert [point.timestamp for point in point_buffer.points] == [1.0, 2.0]
    assert point_buffer.statistics.accepted_point_count == 2


def test_reconstruction_record_and_saved_dataset_have_no_ground_truth_fields(
    tmp_path,
) -> None:
    field_names = {field.name for field in fields(TemporalTactileObservation)}
    assert not any(
        forbidden in name
        for name in field_names
        for forbidden in ("ground_truth", "object_pose", "object_mesh", "contact_normal")
    )
    point_buffer = ContactPointBuffer(
        PointBufferConfig(
            minimum_time_between_points=0.0,
            minimum_spatial_distance_between_points=0.0,
        )
    )
    point_buffer.start()
    point_buffer.add([observation(0.0, (0.1, 0.2, 0.3))])
    reconstruction_path = save_reconstruction_dataset(
        tmp_path / "run.json",
        point_buffer.points,
        point_buffer.config,
        point_buffer.statistics,
    )
    payload = json.loads(reconstruction_path.read_text(encoding="utf-8"))
    serialized = json.dumps(payload).lower()
    assert all(
        forbidden not in serialized
        for forbidden in ("ground_truth", "object_pose", "object_mesh", "contact_normal")
    )
    assert payload["observations"][0]["joint_state_snapshot"]["joint_10"] == 0.2

    report = evaluate_contact_points(
        [estimate((0.0, 0.0, 0.2))],
        [GroundTruthContactPoint(1.0, "Finger02_Link02_Sensor", (0.0, 0.0, 0.3))],
    )
    evaluation_path = save_evaluation_dataset(
        tmp_path / "run.evaluation.json", report.errors
    )
    evaluation_payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
    assert "ground_truth_position" in evaluation_payload["errors"][0]

    empty_path = save_evaluation_dataset(tmp_path / "empty.evaluation.json", [])
    empty_payload = json.loads(empty_path.read_text(encoding="utf-8"))
    assert empty_payload["metrics"]["euclidean"]["mean_m"] is None
