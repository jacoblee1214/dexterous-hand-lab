"""Ground-truth-free robust sphere fitting from accepted tactile points only."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

from reconstruction.temporal_observation import TemporalTactileObservation


class SphereReconstructionStatus(str, Enum):
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    POOR_SPATIAL_COVERAGE = "POOR_SPATIAL_COVERAGE"
    VALID_RECONSTRUCTION = "VALID_RECONSTRUCTION"


@dataclass(frozen=True)
class SphereFittingConfig:
    minimum_point_count: int = 8
    minimum_unique_sensor_count: int = 3
    minimum_unique_finger_count: int = 2
    minimum_spatial_spread_m: float = 0.015
    minimum_covariance_eigenvalue_m2: float = 1.0e-7
    maximum_covariance_condition: float = 1.0e4
    robust_loss: str = "soft_l1"
    robust_loss_scale_m: float = 0.002
    maximum_iterations: int = 100
    auto_fit_minimum_new_points: int = 5

    def __post_init__(self) -> None:
        if min(
            self.minimum_point_count,
            self.minimum_unique_sensor_count,
            self.minimum_unique_finger_count,
            self.maximum_iterations,
            self.auto_fit_minimum_new_points,
        ) <= 0:
            raise ValueError("Sphere fitting integer thresholds must be positive")
        if self.robust_loss not in {"linear", "soft_l1", "huber"}:
            raise ValueError("robust_loss must be linear, soft_l1, or huber")
        if (
            not all(
                math.isfinite(value)
                for value in (
                    self.minimum_spatial_spread_m,
                    self.minimum_covariance_eigenvalue_m2,
                    self.maximum_covariance_condition,
                    self.robust_loss_scale_m,
                )
            )
            or
            self.minimum_spatial_spread_m < 0
            or self.minimum_covariance_eigenvalue_m2 < 0
            or self.maximum_covariance_condition <= 0
            or self.robust_loss_scale_m <= 0
        ):
            raise ValueError("Sphere fitting thresholds must be non-negative")


@dataclass(frozen=True)
class SpatialCoverageMetrics:
    axis_extent_xyz_m: tuple[float, float, float]
    spatial_spread_m: float
    covariance_eigenvalues_m2: tuple[float, float, float]


@dataclass(frozen=True)
class SphereReconstructionResult:
    timestamp: float
    status: SphereReconstructionStatus
    number_of_input_points: int
    number_of_unique_sensors: int
    number_of_unique_fingers: int
    estimated_center_xyz: tuple[float, float, float] | None
    estimated_radius: float | None
    fit_residual_mean: float | None
    fit_residual_median: float | None
    fit_residual_rmse: float | None
    fit_residual_p95: float | None
    spatial_coverage_metrics: SpatialCoverageMetrics
    condition_metric: float | None


def load_sphere_fitting_config(path: str | Path) -> SphereFittingConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError("Sphere reconstruction config must use schema_version 1")
    values = raw.get("sphere_reconstruction", {})
    return SphereFittingConfig(**values)


def _coverage(points: np.ndarray) -> tuple[SpatialCoverageMetrics, float]:
    if len(points) == 0:
        return SpatialCoverageMetrics((0.0, 0.0, 0.0), 0.0, (0.0, 0.0, 0.0)), math.inf
    extent = np.ptp(points, axis=0)
    spread = float(np.linalg.norm(extent))
    if len(points) < 2:
        eigenvalues = np.zeros(3)
    else:
        covariance = np.cov(points, rowvar=False, bias=True)
        eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    smallest = float(eigenvalues[0])
    condition = math.inf if smallest <= np.finfo(float).eps else float(eigenvalues[-1] / smallest)
    return (
        SpatialCoverageMetrics(
            tuple(float(value) for value in extent),
            spread,
            tuple(float(value) for value in eigenvalues),
        ),
        condition,
    )


def _algebraic_initialization(points: np.ndarray) -> tuple[np.ndarray, float]:
    origin = np.mean(points, axis=0)
    centered = points - origin
    design = np.column_stack((2.0 * centered, np.ones(len(points))))
    solution, _, rank, _ = np.linalg.lstsq(
        design, np.sum(centered * centered, axis=1), rcond=None
    )
    if rank < 4:
        raise np.linalg.LinAlgError("Sphere initialization is rank deficient")
    center_relative = solution[:3]
    radius_squared = float(solution[3] + np.dot(center_relative, center_relative))
    if not math.isfinite(radius_squared) or radius_squared <= 0:
        raise np.linalg.LinAlgError("Sphere initialization produced an invalid radius")
    return origin + center_relative, math.sqrt(radius_squared)


def _robust_weights(residuals: np.ndarray, loss: str, scale: float) -> np.ndarray:
    normalized = np.abs(residuals) / scale
    if loss == "linear":
        return np.ones_like(residuals)
    if loss == "huber":
        return np.where(normalized <= 1.0, 1.0, 1.0 / np.maximum(normalized, 1e-15))
    return 1.0 / np.sqrt(1.0 + normalized * normalized)


def _refine_geometrically(
    points: np.ndarray,
    center: np.ndarray,
    radius: float,
    config: SphereFittingConfig,
) -> tuple[np.ndarray, float]:
    """Deterministic geometric least squares, with a NumPy fallback."""
    parameters = np.r_[center, math.log(radius)]
    try:
        from scipy.optimize import least_squares
    except ImportError:
        least_squares = None
    if least_squares is not None:
        def residual_function(values: np.ndarray) -> np.ndarray:
            return np.linalg.norm(points - values[:3], axis=1) - math.exp(float(values[3]))

        optimized = least_squares(
            residual_function,
            parameters,
            loss=config.robust_loss,
            f_scale=config.robust_loss_scale_m,
            max_nfev=config.maximum_iterations,
            method="trf",
        )
        return optimized.x[:3].copy(), math.exp(float(optimized.x[3]))

    # Equivalent deterministic robust IRLS/Gauss-Newton fallback keeps SciPy optional.
    for _ in range(config.maximum_iterations):
        center = parameters[:3]
        radius = math.exp(float(parameters[3]))
        delta = points - center
        distances = np.linalg.norm(delta, axis=1)
        safe_distances = np.maximum(distances, 1e-12)
        residuals = distances - radius
        jacobian = np.column_stack((-delta / safe_distances[:, None], -np.full(len(points), radius)))
        weights = _robust_weights(residuals, config.robust_loss, config.robust_loss_scale_m)
        root_weights = np.sqrt(weights)
        weighted_jacobian = jacobian * root_weights[:, None]
        weighted_residuals = residuals * root_weights
        step, _, _, _ = np.linalg.lstsq(weighted_jacobian, -weighted_residuals, rcond=None)
        # A bounded step prevents a poorly observed patch from exploding.
        step[:3] = np.clip(step[:3], -0.05, 0.05)
        step[3] = float(np.clip(step[3], -0.5, 0.5))
        parameters += step
        if float(np.linalg.norm(step)) < 1e-11:
            break
    return parameters[:3].copy(), math.exp(float(parameters[3]))


def fit_sphere(
    observations: Iterable[TemporalTactileObservation],
    config: SphereFittingConfig | None = None,
    *,
    timestamp: float | None = None,
) -> SphereReconstructionResult:
    """Fit using only accepted estimated points and their sensor/finger metadata."""
    config = config or SphereFittingConfig()
    samples = tuple(observations)
    points = np.asarray(
        [sample.estimated_contact_position for sample in samples], dtype=float
    ).reshape((-1, 3))
    sensor_count = len({sample.sensor_id for sample in samples})
    finger_count = len({sample.finger_id for sample in samples})
    coverage, condition = _coverage(points)
    result_time = float(
        max((sample.timestamp for sample in samples), default=0.0)
        if timestamp is None
        else timestamp
    )
    enough = (
        len(samples) >= config.minimum_point_count
        and sensor_count >= config.minimum_unique_sensor_count
        and finger_count >= config.minimum_unique_finger_count
    )
    observable = (
        coverage.spatial_spread_m >= config.minimum_spatial_spread_m
        and coverage.covariance_eigenvalues_m2[0]
        >= config.minimum_covariance_eigenvalue_m2
        and condition <= config.maximum_covariance_condition
    )
    status = (
        SphereReconstructionStatus.INSUFFICIENT_DATA
        if not enough
        else SphereReconstructionStatus.VALID_RECONSTRUCTION
        if observable
        else SphereReconstructionStatus.POOR_SPATIAL_COVERAGE
    )
    center = None
    radius = None
    statistics: tuple[float | None, ...] = (None, None, None, None)
    if enough:
        try:
            center_array, radius_value = _algebraic_initialization(points)
            center_array, radius_value = _refine_geometrically(
                points, center_array, radius_value, config
            )
            absolute_residuals = np.abs(np.linalg.norm(points - center_array, axis=1) - radius_value)
            center = tuple(float(value) for value in center_array)
            radius = float(radius_value)
            statistics = (
                float(np.mean(absolute_residuals)),
                float(np.median(absolute_residuals)),
                float(np.sqrt(np.mean(absolute_residuals**2))),
                float(np.percentile(absolute_residuals, 95)),
            )
        except (np.linalg.LinAlgError, FloatingPointError, OverflowError):
            status = SphereReconstructionStatus.POOR_SPATIAL_COVERAGE
    return SphereReconstructionResult(
        result_time,
        status,
        len(samples),
        sensor_count,
        finger_count,
        center,
        radius,
        *statistics,
        coverage,
        condition if math.isfinite(condition) else None,
    )


class SphereReconstructionSession:
    """Manual-first fit lifecycle with throttled point-count-based auto fitting."""

    def __init__(self, config: SphereFittingConfig | None = None) -> None:
        self.config = config or SphereFittingConfig()
        self.auto_fit = False
        self.result: SphereReconstructionResult | None = None
        self._last_fit_point_count = 0

    @property
    def mode(self) -> str:
        return "AUTO FIT" if self.auto_fit else "MANUAL FIT"

    def fit(self, observations: Iterable[TemporalTactileObservation]) -> SphereReconstructionResult:
        samples = tuple(observations)
        self.result = fit_sphere(samples, self.config)
        self._last_fit_point_count = len(samples)
        return self.result

    def maybe_auto_fit(self, observations: Iterable[TemporalTactileObservation]) -> SphereReconstructionResult | None:
        samples = tuple(observations)
        if self.auto_fit and len(samples) - self._last_fit_point_count >= self.config.auto_fit_minimum_new_points:
            return self.fit(samples)
        return None

    def toggle_auto_fit(self) -> None:
        self.auto_fit = not self.auto_fit

    def reset(self) -> None:
        self.result = None
        self._last_fit_point_count = 0
