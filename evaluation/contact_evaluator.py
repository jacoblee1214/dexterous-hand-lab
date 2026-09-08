"""One-way MuJoCo ground-truth evaluation for estimated tactile contacts.

This module is intentionally outside ``reconstruction``. Ground-truth contact
positions are collected here and are used only to compute/report errors; no
value produced here is fed back into contact projection.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from reconstruction.estimated_contact import EstimatedContactPoint, Vector3
from sensors.pressure_emulator import PressureSensorEmulator, SensorSignalConfig
from sensors.sensor_kinematics import SensorMount


@dataclass(frozen=True)
class GroundTruthContactPoint:
    timestamp: float
    sensor_id: str
    position: Vector3


@dataclass(frozen=True)
class ContactPointError:
    timestamp: float
    sensor_id: str
    estimated_position: Vector3
    ground_truth_position: Vector3
    sensor_position: Vector3
    sensor_sensing_direction: Vector3
    euclidean_error_m: float
    normal_direction_error_m: float
    tangential_plane_error_m: float


@dataclass(frozen=True)
class ErrorStatistics:
    count: int
    mean_m: float
    median_m: float
    rmse_m: float
    percentile_95_m: float


@dataclass(frozen=True)
class EvaluationMetrics:
    euclidean: ErrorStatistics
    normal_direction: ErrorStatistics
    tangential_plane: ErrorStatistics

    @property
    def count(self) -> int:
        return self.euclidean.count

    # Compatibility accessors for the original Euclidean-only report API.
    @property
    def mean_error_m(self) -> float:
        return self.euclidean.mean_m

    @property
    def median_error_m(self) -> float:
        return self.euclidean.median_m

    @property
    def rmse_m(self) -> float:
        return self.euclidean.rmse_m

    @property
    def percentile_95_error_m(self) -> float:
        return self.euclidean.percentile_95_m


@dataclass(frozen=True)
class EvaluationReport:
    errors: tuple[ContactPointError, ...]
    metrics: EvaluationMetrics


def collect_mujoco_ground_truth(
    simulation,
    mounts: Mapping[str, SensorMount],
    configs: Mapping[str, SensorSignalConfig],
) -> list[GroundTruthContactPoint]:
    """Collect assigned contact positions for evaluation/debug display only."""
    mujoco, model, data = simulation.mujoco, simulation.model, simulation.data
    body_ids = {
        sensor_id: mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, mount.parent_link
        )
        for sensor_id, mount in mounts.items()
    }
    result = []
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        contact_bodies = {
            int(model.geom_bodyid[contact.geom1]),
            int(model.geom_bodyid[contact.geom2]),
        }
        for sensor_id, mount in mounts.items():
            body_id = body_ids[sensor_id]
            if body_id not in contact_bodies:
                continue
            rotation = data.xmat[body_id].reshape(3, 3)
            point_local = rotation.T @ (contact.pos - data.xpos[body_id])
            if PressureSensorEmulator.point_is_in_region(
                mount, point_local, configs[sensor_id].region_tolerance_m
            ):
                result.append(
                    GroundTruthContactPoint(
                        timestamp=float(data.time),
                        sensor_id=sensor_id,
                        position=tuple(float(value) for value in contact.pos),
                    )
                )
    return result


def evaluate_contact_points(
    estimated: list[EstimatedContactPoint],
    ground_truth: list[GroundTruthContactPoint],
) -> EvaluationReport:
    """Compare each estimate with its nearest same-channel GT contact."""
    by_sensor: dict[str, list[GroundTruthContactPoint]] = {}
    for point in ground_truth:
        by_sensor.setdefault(point.sensor_id, []).append(point)
    errors = []
    for estimate in estimated:
        candidates = by_sensor.get(estimate.sensor_id, [])
        if not candidates:
            continue
        estimated_position = np.asarray(estimate.estimated_contact_position)
        truth = min(
            candidates,
            key=lambda candidate: np.linalg.norm(
                estimated_position - np.asarray(candidate.position)
            ),
        )
        error = float(
            np.linalg.norm(estimated_position - np.asarray(truth.position))
        )
        sensing_direction = np.asarray(estimate.sensor_sensing_direction, dtype=float)
        sensing_direction /= np.linalg.norm(sensing_direction)
        sensor_to_truth = (
            np.asarray(truth.position, dtype=float)
            - np.asarray(estimate.sensor_position, dtype=float)
        )
        normal_error = abs(
            float(np.dot(sensor_to_truth, sensing_direction))
            - estimate.estimated_indentation
        )
        delta = np.asarray(truth.position, dtype=float) - estimated_position
        tangential_error = float(
            np.linalg.norm(
                delta - np.dot(delta, sensing_direction) * sensing_direction
            )
        )
        errors.append(
            ContactPointError(
                timestamp=estimate.timestamp,
                sensor_id=estimate.sensor_id,
                estimated_position=estimate.estimated_contact_position,
                ground_truth_position=truth.position,
                sensor_position=estimate.sensor_position,
                sensor_sensing_direction=estimate.sensor_sensing_direction,
                euclidean_error_m=error,
                normal_direction_error_m=normal_error,
                tangential_plane_error_m=tangential_error,
            )
        )
    return EvaluationReport(tuple(errors), summarize_contact_errors(errors))


def summarize_contact_errors(
    errors: list[ContactPointError] | tuple[ContactPointError, ...],
) -> EvaluationMetrics:
    return EvaluationMetrics(
        euclidean=_statistics(
            [sample.euclidean_error_m for sample in errors]
        ),
        normal_direction=_statistics(
            [sample.normal_direction_error_m for sample in errors]
        ),
        tangential_plane=_statistics(
            [sample.tangential_plane_error_m for sample in errors]
        ),
    )


def _statistics(values_input: list[float]) -> ErrorStatistics:
    values = np.asarray(values_input, dtype=float)
    if values.size:
        return ErrorStatistics(
            count=int(values.size),
            mean_m=float(np.mean(values)),
            median_m=float(np.median(values)),
            rmse_m=float(np.sqrt(np.mean(values**2))),
            percentile_95_m=float(np.percentile(values, 95)),
        )
    return ErrorStatistics(0, math.nan, math.nan, math.nan, math.nan)
