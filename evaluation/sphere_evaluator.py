"""Evaluation-only comparison against MuJoCo sphere ground truth."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from evaluation.contact_evaluator import ErrorStatistics
from reconstruction.sphere_fitting import (
    SphereReconstructionResult,
    SphereReconstructionStatus,
)
from reconstruction.temporal_observation import TemporalTactileObservation


@dataclass(frozen=True)
class SphereGroundTruth:
    center_xyz: tuple[float, float, float]
    radius: float


@dataclass(frozen=True)
class SphereEvaluationResult:
    center_error_m: float
    radius_error_m: float
    relative_radius_error: float
    point_to_estimated_surface: ErrorStatistics
    point_to_ground_truth_surface: ErrorStatistics


def collect_mujoco_sphere_ground_truth(
    simulation, geom_name: str = "sphere_object_geom"
) -> SphereGroundTruth:
    """Read sphere parameters from MuJoCo; call only from evaluation/UI code."""
    geom_id = simulation.mujoco.mj_name2id(
        simulation.model, simulation.mujoco.mjtObj.mjOBJ_GEOM, geom_name
    )
    if geom_id < 0:
        raise KeyError(f"Unknown MuJoCo sphere geometry: {geom_name}")
    return SphereGroundTruth(
        tuple(float(value) for value in simulation.data.geom_xpos[geom_id]),
        float(simulation.model.geom_size[geom_id, 0]),
    )


def _surface_statistics(values: np.ndarray) -> ErrorStatistics:
    if values.size == 0:
        return ErrorStatistics(0, math.nan, math.nan, math.nan, math.nan)
    return ErrorStatistics(
        int(values.size),
        float(np.mean(values)),
        float(np.median(values)),
        float(np.sqrt(np.mean(values**2))),
        float(np.percentile(values, 95)),
    )


def evaluate_sphere_reconstruction(
    reconstruction: SphereReconstructionResult,
    observations: Iterable[TemporalTactileObservation],
    ground_truth: SphereGroundTruth,
) -> SphereEvaluationResult | None:
    """Compare without feeding any evaluation value back into reconstruction."""
    if (
        reconstruction.status is not SphereReconstructionStatus.VALID_RECONSTRUCTION
        or reconstruction.estimated_center_xyz is None
        or reconstruction.estimated_radius is None
    ):
        return None
    points = np.asarray(
        [sample.estimated_contact_position for sample in observations], dtype=float
    ).reshape((-1, 3))
    estimated_center = np.asarray(reconstruction.estimated_center_xyz, dtype=float)
    truth_center = np.asarray(ground_truth.center_xyz, dtype=float)
    distance_to_estimated = np.abs(
        np.linalg.norm(points - estimated_center, axis=1)
        - reconstruction.estimated_radius
    )
    distance_to_truth = np.abs(
        np.linalg.norm(points - truth_center, axis=1) - ground_truth.radius
    )
    radius_error = abs(reconstruction.estimated_radius - ground_truth.radius)
    return SphereEvaluationResult(
        center_error_m=float(np.linalg.norm(estimated_center - truth_center)),
        radius_error_m=float(radius_error),
        relative_radius_error=float(radius_error / ground_truth.radius),
        point_to_estimated_surface=_surface_statistics(distance_to_estimated),
        point_to_ground_truth_surface=_surface_statistics(distance_to_truth),
    )
