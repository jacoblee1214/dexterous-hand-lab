"""Deterministic known-radius sphere solvers using RGB and/or scalar touch.

This module deliberately has no simulator import and no object-pose argument.
MuJoCo truth belongs only in the evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable, Mapping

import numpy as np

from fusion.config import load_fusion_config
from fusion.observation import FusionObservation, TactileContactConstraint


def _boundary(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(np.asarray(mask, dtype=bool), 1, constant_values=False)
    center = padded[1:-1, 1:-1]
    edges = []
    for neighbor, du, dv in (
        (padded[1:-1, 2:], 0.5, 0.0),
        (padded[1:-1, :-2], -0.5, 0.0),
        (padded[2:, 1:-1], 0.0, 0.5),
        (padded[:-2, 1:-1], 0.0, -0.5),
    ):
        rows, columns = np.nonzero(center & ~neighbor)
        if len(rows):
            edges.append(np.column_stack((columns + du, rows + dv)))
    return np.concatenate(edges).astype(float) if edges else np.empty((0, 2))


def _nearest_mask(mask: np.ndarray, pixels_uv: np.ndarray, margin: int = 2) -> np.ndarray:
    height, width = mask.shape
    answer = np.zeros(len(pixels_uv), dtype=bool)
    for index, (u, v) in enumerate(pixels_uv):
        column, row = int(round(u)), int(round(v))
        r0, r1 = max(0, row-margin), min(height, row+margin+1)
        c0, c1 = max(0, column-margin), min(width, column+margin+1)
        answer[index] = bool(mask[r0:r1, c0:c1].any())
    return answer


def _pixel_rays(K: np.ndarray, pixels_uv: np.ndarray) -> np.ndarray:
    rays = np.column_stack((
        (pixels_uv[:, 0] - K[0, 2]) / K[0, 0],
        (pixels_uv[:, 1] - K[1, 2]) / K[1, 1],
        np.ones(len(pixels_uv)),
    ))
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    return rays


@dataclass(frozen=True)
class VisionConstraintSet:
    boundary_rays_cv: np.ndarray
    background_rays_cv: np.ndarray
    status: str
    visible_boundary_count: int
    visible_angular_coverage: float


def build_vision_constraints(observation: FusionObservation, config: Mapping) -> VisionConstraintSet:
    settings = config["rgb_visibility"]
    boundary = _boundary(observation.predicted_object_mask)
    if len(boundary):
        boundary = boundary[~_nearest_mask(observation.unknown_occluded_mask, boundary)]
    maximum = int(settings["maximum_boundary_samples"])
    if len(boundary) > maximum:
        boundary = boundary[np.linspace(0, len(boundary)-1, maximum).astype(int)]
    coverage = 0.0
    if len(boundary) >= 2:
        center = boundary.mean(axis=0)
        angles = np.mod(np.arctan2(boundary[:, 1]-center[1], boundary[:, 0]-center[0]), 2*np.pi)
        coverage = float(len(np.unique(np.floor(angles/(2*np.pi)*36).astype(int))) / 36.0)
    rows, columns = np.nonzero(observation.known_background_mask)
    stride = max(1, int(settings["background_sample_stride_pixels"]))
    background = np.column_stack((columns[::stride], rows[::stride])).astype(float)
    maximum_background = int(settings["maximum_background_samples"])
    if len(background) > maximum_background:
        background = background[np.linspace(0, len(background)-1, maximum_background).astype(int)]
    reliable = (
        observation.segmentation_status == "SEGMENTATION_OK"
        and len(boundary) >= int(settings["minimum_visible_boundary_samples"])
        and coverage >= float(settings["minimum_visible_angular_coverage"])
    )
    status = "VALID" if reliable else "INSUFFICIENT_VISIBLE_SILHOUETTE"
    return VisionConstraintSet(
        _pixel_rays(observation.intrinsic_matrix, boundary),
        _pixel_rays(observation.intrinsic_matrix, background),
        status, len(boundary), coverage,
    )


def representative_tactile_residual(
    center_world_m: np.ndarray, constraint: TactileContactConstraint, radius_m: float
) -> float:
    return (float(np.linalg.norm(constraint.representative_point()-center_world_m)) - radius_m) / constraint.position_sigma_m


def finite_patch_tactile_residual(
    center_world_m: np.ndarray, constraint: TactileContactConstraint, radius_m: float
) -> float:
    """Distance to the patch's feasible radial interval, zero when feasible."""
    radial = np.linalg.norm(constraint.patch_points_world() - center_world_m[None, :], axis=1)
    minimum, maximum = float(radial.min()), float(radial.max())
    if radius_m < minimum:
        return (minimum-radius_m) / constraint.finite_patch_sigma_m
    if radius_m > maximum:
        return (maximum-radius_m) / constraint.finite_patch_sigma_m
    return 0.0


def _center_camera(observation: FusionObservation, center_world: np.ndarray) -> np.ndarray:
    return (observation.camera_cv_from_world @ np.r_[center_world, 1.0])[:3]


def _vision_initial(observation: FusionObservation, constraints: VisionConstraintSet):
    """Cone-axis initialization from visible RGB boundary rays only."""
    rays = constraints.boundary_rays_cv
    if constraints.status != "VALID" or len(rays) < 3:
        return None
    axis = rays.mean(axis=0)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        return None
    axis /= norm
    angles = np.arccos(np.clip(rays@axis, -1.0, 1.0))
    angular_radius = float(np.median(angles))
    if not 1e-5 < angular_radius < math.pi/2:
        return None
    center_cv = axis*(observation.known_radius_m/math.sin(angular_radius))
    if center_cv[2] <= observation.known_radius_m:
        return None
    world_from_camera = np.linalg.inv(observation.camera_cv_from_world)
    return (world_from_camera @ np.r_[center_cv, 1.0])[:3]


def _vision_residual(
    observation: FusionObservation, constraints: VisionConstraintSet, center_world: np.ndarray,
    config: Mapping,
) -> np.ndarray:
    center = _center_camera(observation, center_world)
    radius = observation.known_radius_m
    sigma_px = float(config["uncertainty"]["vision_boundary_sigma_px"])
    focal = math.sqrt(observation.intrinsic_matrix[0, 0]*observation.intrinsic_matrix[1, 1])
    depth_scale = focal / max(center[2], radius*0.05) / sigma_px
    parts = []
    if len(constraints.boundary_rays_cv):
        projection = constraints.boundary_rays_cv @ center
        perpendicular = np.sqrt(np.maximum(0.0, float(center@center)-projection*projection))
        parts.append((perpendicular-radius)*depth_scale)
    if len(constraints.background_rays_cv):
        projection = constraints.background_rays_cv @ center
        perpendicular = np.sqrt(np.maximum(0.0, float(center@center)-projection*projection))
        intrusion = np.maximum(0.0, radius-perpendicular)
        intrusion[projection <= 0] = 0.0
        parts.append(intrusion*depth_scale)
    return np.concatenate(parts) if parts else np.empty(0)


def _active_constraints(observation: FusionObservation) -> tuple[TactileContactConstraint, ...]:
    return tuple(item for item in observation.tactile_constraints if item.activation_valid)


def _tactile_init(contacts: tuple[TactileContactConstraint, ...]) -> tuple[np.ndarray | None, int]:
    if len(contacts) < 4:
        return None, 0
    points = np.asarray([item.representative_point() for item in contacts])
    p0 = points[0]
    matrix = 2.0*(points[1:]-p0)
    values = np.sum(points[1:]*points[1:], axis=1)-float(p0@p0)
    rank = int(np.linalg.matrix_rank(matrix, tol=1e-9))
    if rank < 3:
        return None, rank
    return np.linalg.lstsq(matrix, values, rcond=None)[0], rank


def _tactile_coverage(contacts, center, config) -> dict:
    settings = config["tactile_observability"]
    points = np.asarray([item.representative_point() for item in contacts], dtype=float)
    spread = float(np.linalg.norm(np.ptp(points, axis=0))) if len(points) else 0.0
    sensors = len({item.sensor_id for item in contacts})
    fingers = len({item.finger_id for item in contacts})
    singular = np.empty(0)
    condition = math.inf
    if center is not None and len(points):
        vectors = center[None, :]-points
        lengths = np.linalg.norm(vectors, axis=1)
        jacobian = vectors[lengths > 1e-12] / lengths[lengths > 1e-12, None]
        if len(jacobian):
            singular = np.linalg.svd(jacobian, compute_uv=False)
            condition = float(singular[0]/singular[-1]) if len(singular) >= 3 and singular[-1] > 0 else math.inf
    minimum_singular = float(singular[-1]) if len(singular) >= 3 else 0.0
    valid = (
        len(contacts) >= int(settings["minimum_contact_count"])
        and sensors >= int(settings["minimum_unique_sensor_count"])
        and fingers >= int(settings["minimum_unique_finger_count"])
        and spread >= float(settings["minimum_spatial_spread_m"])
        and minimum_singular >= float(settings["minimum_jacobian_singular_value"])
        and condition <= float(settings["maximum_jacobian_condition"])
    )
    return {"contact_count": len(contacts), "unique_sensor_count": sensors,
            "unique_finger_count": fingers,
            "contributing_sensor_ids": sorted({item.sensor_id for item in contacts}),
            "contributing_finger_ids": sorted({item.finger_id for item in contacts}),
            "spatial_spread_m": spread,
            "minimum_jacobian_singular_value": minimum_singular,
            "jacobian_condition": condition if math.isfinite(condition) else None,
            "observable": valid}


def _validity_reason(status: str, diagnostics: Mapping, config: Mapping) -> str:
    if status == "VALID_ESTIMATE":
        return "All required modality, spatial-coverage, and numerical-conditioning checks passed."
    if status == "INSUFFICIENT_VISION":
        return (
            "RGB silhouette is not usable: status="
            f"{diagnostics.get('vision_status', 'UNKNOWN')}, visible boundary="
            f"{diagnostics.get('visible_boundary_count', 0)}, angular coverage="
            f"{diagnostics.get('visible_angular_coverage', 0.0):.3f}."
        )
    if status == "INITIALIZATION_FAILURE":
        return "No finite center initialization could be formed from the available modality constraints."
    if status == "ILL_CONDITIONED":
        return (
            "The joint residual Jacobian did not provide a stable three-dimensional center "
            f"(condition={diagnostics.get('joint_condition_number')}, "
            f"minimum singular value={diagnostics.get('joint_minimum_singular_value')})."
        )
    settings = config["tactile_observability"]
    missing = []
    checks = (
        ("contact_count", "minimum_contact_count", "contacts"),
        ("unique_sensor_count", "minimum_unique_sensor_count", "unique sensors"),
        ("unique_finger_count", "minimum_unique_finger_count", "unique fingers"),
        ("spatial_spread_m", "minimum_spatial_spread_m", "spatial spread (m)"),
        ("minimum_jacobian_singular_value", "minimum_jacobian_singular_value",
         "tactile Jacobian minimum singular value"),
    )
    for actual_key, required_key, label in checks:
        actual, required = diagnostics.get(actual_key, 0), settings[required_key]
        if actual < required:
            missing.append(f"{label} {actual} < {required}")
    condition = diagnostics.get("jacobian_condition")
    if condition is None or condition > settings["maximum_jacobian_condition"]:
        missing.append(
            f"tactile Jacobian condition {condition} > {settings['maximum_jacobian_condition']}"
        )
    return "Insufficient tactile observability: " + "; ".join(missing or ["unknown coverage check"])


def _huber_cost(residual: np.ndarray, threshold: np.ndarray | float) -> float:
    absolute = np.abs(residual)
    return float(np.sum(np.where(absolute <= threshold, .5*residual*residual,
                                 threshold*(absolute-.5*threshold))))


def _solve(initial: np.ndarray, residual: Callable[[np.ndarray], np.ndarray], config: Mapping,
           robust_thresholds: np.ndarray | None = None):
    settings = config["solver"]
    center = np.asarray(initial, dtype=float).copy()
    threshold = (np.asarray(robust_thresholds, dtype=float) if robust_thresholds is not None
                 else float(settings["robust_huber_threshold"]))
    step_size = float(settings["finite_difference_step_m"])
    iterations = 0
    converged = False
    jacobian = np.empty((0, 3))
    for iterations in range(1, int(settings["maximum_iterations"])+1):
        values = residual(center)
        if len(values) < 3 or not np.isfinite(values).all():
            break
        columns = []
        for axis in range(3):
            shifted = center.copy(); shifted[axis] += step_size
            columns.append((residual(shifted)-values)/step_size)
        jacobian = np.column_stack(columns)
        absolute = np.abs(values)
        weights = np.ones_like(values)
        outside = absolute > threshold
        if np.ndim(threshold):
            weights[outside] = threshold[outside]/absolute[outside]
        else:
            weights[outside] = threshold/absolute[outside]
        normal = jacobian.T @ (weights[:, None]*jacobian) + float(settings["damping"])*np.eye(3)
        gradient = jacobian.T @ (weights*values)
        try:
            delta = np.linalg.solve(normal, -gradient)
        except np.linalg.LinAlgError:
            break
        maximum = float(settings["maximum_center_step_m"])
        if np.linalg.norm(delta) > maximum:
            delta *= maximum/np.linalg.norm(delta)
        old_cost = _huber_cost(values, threshold)
        scale = 1.0
        while scale > 1/128 and _huber_cost(residual(center+scale*delta), threshold) > old_cost:
            scale *= .5
        center += scale*delta
        if np.linalg.norm(scale*delta) < float(settings["convergence_step_m"]):
            converged = True
            break
    values = residual(center)
    if len(values) >= 3:
        columns = []
        for axis in range(3):
            shifted = center.copy(); shifted[axis] += step_size
            columns.append((residual(shifted)-values)/step_size)
        jacobian = np.column_stack(columns)
    singular = np.linalg.svd(jacobian, compute_uv=False) if len(jacobian) else np.empty(0)
    condition = float(singular[0]/singular[-1]) if len(singular) >= 3 and singular[-1] > 1e-12 else math.inf
    covariance = None
    if len(values) > 3 and np.isfinite(condition):
        covariance = np.linalg.pinv(jacobian.T@jacobian) * float(values@values/max(1, len(values)-3))
    return center, values, iterations, converged, singular, condition, covariance


@dataclass(frozen=True)
class FusionResult:
    method: str
    status: str
    center_world_m: tuple[float, float, float] | None
    radius_m: float
    radius_label: str
    initialization: str
    vision_residual_rms_sigma: float | None
    tactile_residual_rms_sigma: float | None
    joint_objective: float | None
    iteration_count: int
    condition_number: float | None
    minimum_singular_value: float | None
    covariance_diagonal_m2: tuple[float, float, float] | None
    validity_reason: str
    diagnostics: Mapping
    runtime_ms: float

    @property
    def valid(self) -> bool:
        return self.status in {
            "VALID_ESTIMATE", "VALID_FUSED", "VALID_RGB_FALLBACK",
            "VALID_TACTILE_FALLBACK",
        }

    def to_dict(self) -> dict:
        return {"method": self.method, "status": self.status, "valid": self.valid,
                "estimated_center_world_m": list(self.center_world_m) if self.center_world_m else None,
                "radius_m": self.radius_m, "diameter_m": 2*self.radius_m,
                "radius_label": self.radius_label, "initialization": self.initialization,
                "vision_residual_rms_sigma": self.vision_residual_rms_sigma,
                "tactile_residual_rms_sigma": self.tactile_residual_rms_sigma,
                "joint_objective": self.joint_objective, "iteration_count": self.iteration_count,
                "condition_number": self.condition_number,
                "minimum_singular_value": self.minimum_singular_value,
                "covariance_diagonal_m2": list(self.covariance_diagonal_m2) if self.covariance_diagonal_m2 else None,
                "uncertainty_standard_deviation_m": (
                    [math.sqrt(max(0.0, value)) for value in self.covariance_diagonal_m2]
                    if self.covariance_diagonal_m2 else None
                ),
                "validity_reason": self.validity_reason,
                "diagnostics": dict(self.diagnostics), "runtime_ms": self.runtime_ms}


def _failure(method, observation, status, initialization, diagnostics, config, started):
    diagnostics = {
        **diagnostics,
        "modality_weights": {key: float(value) for key, value in config["weights"].items()},
        "robust_loss": {
            "name": "Huber",
            "threshold_sigma": float(config["solver"]["robust_huber_threshold"]),
        },
    }
    return FusionResult(method, status, None, observation.known_radius_m,
                        observation.radius_label, initialization, None, None, None, 0,
                        None, None, None, _validity_reason(status, diagnostics, config),
                        diagnostics, (time.perf_counter()-started)*1000)


def _fit_method(method, observation, vision, config, initial, initialization, use_vision, tactile_mode):
    started = time.perf_counter()
    contacts = _active_constraints(observation)
    preliminary = _tactile_coverage(contacts, initial, config)
    if use_vision and vision.status != "VALID":
        return _failure(method, observation, "INSUFFICIENT_VISION", initialization,
                        {"vision_status": vision.status,
                         "visible_boundary_count": vision.visible_boundary_count,
                         "visible_angular_coverage": vision.visible_angular_coverage,
                         **preliminary}, config, started)
    if tactile_mode and not preliminary["observable"]:
        return _failure(method, observation, "INSUFFICIENT_TACTILE_COVERAGE", initialization,
                        {"vision_status": vision.status,
                         "visible_boundary_count": vision.visible_boundary_count,
                         "visible_angular_coverage": vision.visible_angular_coverage,
                         **preliminary}, config, started)
    if initial is None or not np.isfinite(initial).all():
        return _failure(method, observation, "INITIALIZATION_FAILURE", initialization,
                        {"vision_status": vision.status,
                         "visible_boundary_count": vision.visible_boundary_count,
                         "visible_angular_coverage": vision.visible_angular_coverage,
                         **preliminary}, config, started)
    vision_weight = math.sqrt(float(config["weights"]["vision"]))
    tactile_weight = math.sqrt(float(config["weights"]["tactile"]))
    patches = None
    if tactile_mode == "finite_patch":
        resolution = int(config["solver"].get("finite_patch_grid_resolution", 5))
        patches = tuple(item.patch_points_world(resolution) for item in contacts)
    def parts(center):
        vr = _vision_residual(observation, vision, center, config) if use_vision else np.empty(0)
        if tactile_mode == "finite_patch":
            values = []
            for item, patch in zip(contacts, patches):
                radial = np.linalg.norm(patch-center[None, :], axis=1)
                minimum, maximum = float(radial.min()), float(radial.max())
                distance = minimum-observation.known_radius_m if observation.known_radius_m < minimum else (
                    maximum-observation.known_radius_m if observation.known_radius_m > maximum else 0.0)
                values.append(distance/item.finite_patch_sigma_m)
            tr = np.asarray(values)
        elif tactile_mode:
            tr = np.asarray([representative_tactile_residual(
                center, item, observation.known_radius_m) for item in contacts])
        else:
            tr = np.empty(0)
        return vr, tr
    def residual(center):
        vr, tr = parts(center)
        # Group normalization makes the configured lambdas independent of the
        # number of sampled image pixels or active tactile observations.
        return np.concatenate((
            vision_weight*vr/math.sqrt(max(1, len(vr))),
            tactile_weight*tr/math.sqrt(max(1, len(tr))),
        ))
    initial_vr, initial_tr = parts(initial)
    huber = float(config["solver"]["robust_huber_threshold"])
    robust_thresholds = np.concatenate((
        np.full(len(initial_vr), huber*vision_weight/math.sqrt(max(1, len(initial_vr)))),
        np.full(len(initial_tr), huber*tactile_weight/math.sqrt(max(1, len(initial_tr)))),
    ))
    center, values, iterations, converged, singular, condition, covariance = _solve(
        initial, residual, config, robust_thresholds
    )
    vr, tr = parts(center)
    final_coverage = _tactile_coverage(contacts, center, config)
    minimum_singular = float(singular[-1]) if len(singular) >= 3 else 0.0
    valid = (np.isfinite(center).all() and len(values) >= 3 and minimum_singular > 1e-8
             and condition < 1e10 and (converged or iterations > 0))
    vision_sigma = float(config["uncertainty"]["vision_boundary_sigma_px"])
    tactile_raw = []
    if tactile_mode == "representative":
        tactile_raw = [abs(value)*item.position_sigma_m for value, item in zip(tr, contacts)]
    elif tactile_mode == "finite_patch":
        tactile_raw = [abs(value)*item.finite_patch_sigma_m for value, item in zip(tr, contacts)]
    diagnostics = {"vision_status": vision.status,
                   "visible_boundary_count": vision.visible_boundary_count,
                   "visible_angular_coverage": vision.visible_angular_coverage,
                   **final_coverage, "solver_converged": converged,
                   "vision_residual_rms_px": float(np.sqrt(np.mean(vr*vr))*vision_sigma) if len(vr) else None,
                   "tactile_residual_rms_m": float(np.sqrt(np.mean(np.square(tactile_raw)))) if tactile_raw else None,
                   "modality_weights": {"vision": float(config["weights"]["vision"]),
                                        "tactile": float(config["weights"]["tactile"])},
                   "robust_loss": {"name": "Huber", "threshold_sigma": huber},
                   "joint_condition_number": condition if math.isfinite(condition) else None,
                   "joint_minimum_singular_value": minimum_singular,
                   "objective_definition": "lambda_rgb*mean(standardized RGB residual^2) + lambda_tactile*mean(standardized tactile residual^2)",
                   "tactile_model": tactile_mode or "none"}
    status = "VALID_ESTIMATE" if valid else "ILL_CONDITIONED"
    covariance_diagonal = (
        tuple(float(x) for x in np.diag(covariance)) if covariance is not None else None
    )
    return FusionResult(
        method, status, tuple(float(x) for x in center) if valid else None,
        observation.known_radius_m, observation.radius_label, initialization,
        float(np.sqrt(np.mean(vr*vr))) if len(vr) else None,
        float(np.sqrt(np.mean(tr*tr))) if len(tr) else None,
        (float(config["weights"]["vision"])*float(np.mean(vr*vr)) if len(vr) else 0.0)
        + (float(config["weights"]["tactile"])*float(np.mean(tr*tr)) if len(tr) else 0.0)
        if len(values) else None, iterations,
        condition if math.isfinite(condition) else None, minimum_singular,
        covariance_diagonal, _validity_reason(status, diagnostics, config), diagnostics,
        (time.perf_counter()-started)*1000,
    )


def run_fusion_methods(
    observation: FusionObservation,
    rgb_baseline_result: Mapping,
    config: Mapping | None = None,
    *,
    v1_observation: FusionObservation | None = None,
    v1_config: Mapping | None = None,
) -> dict:
    """Run A-E with explicit v1/v2 observations and no evaluation truth."""
    config = config or load_fusion_config()
    v1_config = v1_config or config
    if v1_observation is None:
        # Callers that own RGB should provide a separately built v1 observation.
        # For synthetic/direct callers, the conservative compatibility mapping
        # keeps the new unreliable class out of known-background evidence.
        unreliable = observation.unreliable_segmentation_mask
        from dataclasses import replace
        v1_observation = replace(
            observation,
            unknown_occluded_mask=observation.unknown_occluded_mask | unreliable,
            unreliable_segmentation_mask=np.zeros_like(unreliable),
        )
    vision = build_vision_constraints(v1_observation, v1_config)
    sphere = rgb_baseline_result.get("sphere", rgb_baseline_result)
    rgb_center = sphere.get("estimated_center_world_m")
    rgb_initial = np.asarray(rgb_center, dtype=float) if rgb_center is not None else None
    rgb_initial_label = "existing_rgb_known_radius_baseline"
    if rgb_initial is None:
        rgb_initial = _vision_initial(v1_observation, vision)
        rgb_initial_label = "visibility_aware_rgb_boundary_cone"
    contacts = _active_constraints(v1_observation)
    tactile_initial, tactile_rank = _tactile_init(contacts)
    initial = rgb_initial if rgb_initial is not None and vision.status == "VALID" else tactile_initial
    init_label = "reliable_rgb_center" if initial is rgb_initial else "observable_tactile_algebraic_center"
    methods = {
        "rgb_only": _fit_method("rgb_only", v1_observation, vision, v1_config, rgb_initial,
                                rgb_initial_label, True, None),
        "tactile_only_representative": _fit_method(
            "tactile_only_representative", v1_observation, vision, v1_config, tactile_initial,
            f"algebraic_fixed_radius_points_rank_{tactile_rank}", False, "representative"),
        "fusion_representative": _fit_method("fusion_representative", v1_observation, vision, v1_config,
                                             initial, init_label, True, "representative"),
        "fusion_finite_patch": _fit_method("fusion_finite_patch", v1_observation, vision, v1_config,
                                           initial, init_label, True, "finite_patch"),
    }
    if "reliability_v2" in config:
        from fusion.reliability import fit_reliability_aware
        methods["reliability_aware_fusion_v2"] = fit_reliability_aware(
            observation, config
        )
    preference = ("reliability_aware_fusion_v2", "fusion_finite_patch", "fusion_representative",
                  "tactile_only_representative", "rgb_only")
    selected = next((name for name in preference if name in methods and methods[name].valid), None)
    return {
        "configuration_version": str(config["configuration_version"]),
        "input_boundary": "FusionObservation: RGB mask/calibration + named synchronized scalar tactile estimates; no ground truth",
        "known_radius_m": observation.known_radius_m,
        "known_diameter_m": 2*observation.known_radius_m,
        "radius_label": observation.radius_label,
        "object_motion_assumption": observation.object_motion_assumption,
        "weights": {key: float(value) for key, value in config["weights"].items()},
        "robust_loss": {
            "name": "Huber",
            "threshold_sigma": float(config["solver"]["robust_huber_threshold"]),
        },
        "selected_method": selected,
        "status": "VALID_ESTIMATE" if selected else "NO_VALID_ESTIMATE",
        "methods": {name: result.to_dict() for name, result in methods.items()},
    }
