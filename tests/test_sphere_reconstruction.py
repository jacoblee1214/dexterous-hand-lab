import json
from pathlib import Path

import numpy as np
import pytest

from evaluation.sphere_evaluator import (
    SphereGroundTruth,
    evaluate_sphere_reconstruction,
)
from evaluation.experiment_io import save_evaluation_dataset
from reconstruction.contact_point_buffer import ContactPointBuffer, PointBufferConfig
from reconstruction.experiment_io import save_reconstruction_dataset
from reconstruction.sphere_fitting import (
    SphereFittingConfig,
    SphereReconstructionSession,
    SphereReconstructionStatus,
    fit_sphere,
)
from reconstruction.temporal_observation import TemporalTactileObservation
from sensors.registry_audit import (
    APPROVED_RIGHT_SENSOR_COUNT,
    audit_right_sensor_registry,
    require_approved_right_sensor_registry,
)
from sensors.sensor_kinematics import load_sensor_mounts


def observation(position, index, *, sensor=None, finger=None):
    return TemporalTactileObservation(
        timestamp=0.01 * index,
        sensor_id=sensor or f"sensor_{index % 4}",
        finger_id=finger or f"finger_{index % 3}",
        parent_link="link",
        joint_state_snapshot={},
        sensor_position=tuple(position),
        sensor_sensing_direction=(0.0, 0.0, 1.0),
        scalar_sensor_value=1.0,
        estimated_pressure=100.0,
        estimated_indentation=0.001,
        estimated_contact_position=tuple(position),
    )


def sphere_observations(center=(0.02, -0.01, 0.08), radius=0.04, noise=0.0):
    directions = []
    golden = np.pi * (3.0 - np.sqrt(5.0))
    for index in range(48):
        z = 1.0 - 2.0 * (index + 0.5) / 48
        radial = np.sqrt(1.0 - z * z)
        directions.append((radial * np.cos(index * golden), radial * np.sin(index * golden), z))
    points = np.asarray(center) + radius * np.asarray(directions)
    if noise:
        points += np.random.default_rng(1234).normal(0.0, noise, points.shape)
    return [observation(point, index) for index, point in enumerate(points)]


def test_exact_sphere_recovers_center_and_radius():
    result = fit_sphere(sphere_observations())
    assert result.status is SphereReconstructionStatus.VALID_RECONSTRUCTION
    assert result.estimated_center_xyz == pytest.approx((0.02, -0.01, 0.08), abs=1e-9)
    assert result.estimated_radius == pytest.approx(0.04, abs=1e-9)
    assert result.fit_residual_rmse < 1e-10


def test_deterministic_noise_has_bounded_error():
    result = fit_sphere(sphere_observations(noise=0.0002))
    assert np.linalg.norm(np.asarray(result.estimated_center_xyz) - (0.02, -0.01, 0.08)) < 0.0002
    assert abs(result.estimated_radius - 0.04) < 0.0002


def test_insufficient_and_degenerate_coverage_are_not_valid():
    insufficient = fit_sphere(sphere_observations()[:4])
    assert insufficient.status is SphereReconstructionStatus.INSUFFICIENT_DATA
    line = [observation((0.002 * index, 0.0, 0.0), index) for index in range(12)]
    poor = fit_sphere(line)
    assert poor.status is SphereReconstructionStatus.POOR_SPATIAL_COVERAGE


def test_gt_changes_evaluation_but_not_reconstruction():
    samples = sphere_observations()
    result = fit_sphere(samples)
    first = evaluate_sphere_reconstruction(
        result, samples, SphereGroundTruth((0.02, -0.01, 0.08), 0.04)
    )
    second = evaluate_sphere_reconstruction(
        result, samples, SphereGroundTruth((0.02, -0.01, 0.08), 0.06)
    )
    assert result.estimated_radius == pytest.approx(0.04)
    assert first.radius_error_m == pytest.approx(0.0, abs=1e-9)
    assert second.radius_error_m == pytest.approx(0.02, abs=1e-9)


def test_buffer_integration_and_session_reset():
    samples = sphere_observations()
    buffer = ContactPointBuffer(PointBufferConfig(minimum_spatial_distance_between_points=0.0))
    buffer.start()
    assert len(buffer.add(samples)) == len(samples)
    session = SphereReconstructionSession()
    assert session.fit(buffer.points).status is SphereReconstructionStatus.VALID_RECONSTRUCTION
    session.reset()
    assert session.result is None


def test_reconstruction_json_has_result_and_no_ground_truth(tmp_path):
    samples = tuple(sphere_observations())
    config = PointBufferConfig(minimum_spatial_distance_between_points=0.0)
    buffer = ContactPointBuffer(config)
    buffer.start()
    buffer.add(list(samples))
    output = save_reconstruction_dataset(
        tmp_path / "run.json", buffer.points, config, buffer.statistics, fit_sphere(samples)
    )
    payload = json.loads(output.read_text())
    assert payload["sphere_reconstruction"]["estimated_radius"] == pytest.approx(0.04)
    assert "ground_truth" not in output.read_text().lower()


def test_reconstruction_module_has_no_simulator_geometry_dependency():
    source = (Path(__file__).parents[1] / "reconstruction" / "sphere_fitting.py").read_text()
    for forbidden in ("import mujoco", "geom_size", "geom_xpos", "ground_truth"):
        assert forbidden not in source.lower()


def test_approved_right_sensor_registry_is_exactly_18_and_mismatch_fails():
    root = Path(__file__).parents[1]
    mounts = load_sensor_mounts(root / "sensors" / "sensor_config.yaml")
    audit = audit_right_sensor_registry(mounts, mounts.keys())
    assert APPROVED_RIGHT_SENSOR_COUNT == 18
    assert audit.approved
    assert len(audit.intentionally_absent) == 5
    with pytest.raises(RuntimeError, match="approved 18-channel"):
        require_approved_right_sensor_registry(mounts, tuple(mounts)[:-1])


def test_evaluation_file_alone_contains_sphere_ground_truth(tmp_path):
    samples = sphere_observations()
    result = fit_sphere(samples)
    truth = SphereGroundTruth((0.02, -0.01, 0.08), 0.04)
    evaluation = evaluate_sphere_reconstruction(result, samples, truth)
    output = save_evaluation_dataset(tmp_path / "run.evaluation.json", [], truth, evaluation)
    payload = json.loads(output.read_text())
    assert payload["sphere_ground_truth"]["radius"] == pytest.approx(0.04)
    assert payload["sphere_evaluation"]["radius_error_m"] == pytest.approx(0.0, abs=1e-9)
