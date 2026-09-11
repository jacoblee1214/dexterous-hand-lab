"""Transparent object-centric dense-SDF reconstruction baseline."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Mapping

import numpy as np

from .observation import GeneralObjectObservation, TactilePatchObservation


METHODS = {
    "rgb_only",
    "tactile_only",
    "rgb_representative_tactile",
    "rgb_finite_patch_tactile",
    "rgb_reliability_tactile",
    "rgb_perfect_contact_oracle",
}


def _neighbors_min(value: np.ndarray) -> np.ndarray:
    padded = np.pad(value, 1, constant_values=np.inf)
    return np.minimum.reduce((
        padded[:-2, 1:-1, 1:-1], padded[2:, 1:-1, 1:-1],
        padded[1:-1, :-2, 1:-1], padded[1:-1, 2:, 1:-1],
        padded[1:-1, 1:-1, :-2], padded[1:-1, 1:-1, 2:],
    ))


def _interface(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    differs = np.zeros_like(mask)
    for neighbor in (
        padded[:-2, 1:-1, 1:-1], padded[2:, 1:-1, 1:-1],
        padded[1:-1, :-2, 1:-1], padded[1:-1, 2:, 1:-1],
        padded[1:-1, 1:-1, :-2], padded[1:-1, 1:-1, 2:],
    ):
        differs |= neighbor != mask
    return differs


def _signed_manhattan_distance(occupancy: np.ndarray, spacing: float) -> np.ndarray:
    distance = np.full(occupancy.shape, np.inf, dtype=np.float32)
    distance[_interface(occupancy)] = 0.5*spacing
    for _ in range(sum(occupancy.shape)):
        updated = np.minimum(distance, _neighbors_min(distance)+spacing)
        if np.array_equal(updated, distance):
            break
        distance = updated
    return np.where(occupancy, -distance, distance).astype(np.float32)


def _grid(bounds: np.ndarray, resolution: int):
    axes = tuple(np.linspace(bounds[0, i], bounds[1, i], resolution) for i in range(3))
    values = np.meshgrid(*axes, indexing="ij")
    return axes, np.stack(values, axis=-1)


def _project(points: np.ndarray, camera_from_object: np.ndarray, intrinsic: np.ndarray):
    flat = points.reshape(-1, 3)
    homogeneous = np.column_stack((flat, np.ones(len(flat))))
    camera = (camera_from_object @ homogeneous.T).T[:, :3]
    depth = camera[:, 2]
    pixels = np.full((len(flat), 2), -1, dtype=int)
    valid = depth > 1e-8
    pixels[valid, 0] = np.rint(intrinsic[0, 0]*camera[valid, 0]/depth[valid]+intrinsic[0, 2]).astype(int)
    pixels[valid, 1] = np.rint(intrinsic[1, 1]*camera[valid, 1]/depth[valid]+intrinsic[1, 2]).astype(int)
    return pixels, valid


def _visual_hull(observation: GeneralObjectObservation, points: np.ndarray):
    flat_count = points.size//3
    allowed = np.ones(flat_count, dtype=bool)
    foreground_votes = np.zeros(flat_count, dtype=np.int16)
    unknown_votes = np.zeros(flat_count, dtype=np.int16)
    for frame in observation.rgb_observations:
        pixels, valid = _project(points, frame.camera_from_object, frame.intrinsic_matrix)
        h, w = frame.object_mask.shape
        valid &= (pixels[:, 0] >= 0) & (pixels[:, 0] < w) & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
        background = np.ones(flat_count, dtype=bool)
        foreground = np.zeros(flat_count, dtype=bool)
        unknown = np.zeros(flat_count, dtype=bool)
        indexes = np.flatnonzero(valid)
        x, y = pixels[indexes, 0], pixels[indexes, 1]
        background[indexes] = frame.known_background_mask[y, x]
        foreground[indexes] = frame.object_mask[y, x]
        unknown[indexes] = frame.hand_occluded_mask[y, x] | frame.unreliable_mask[y, x]
        allowed &= ~background
        foreground_votes += foreground
        unknown_votes += unknown
    occupancy = allowed & ((foreground_votes+unknown_votes) > 0)
    diagnostics = {
        "candidate_voxel_fraction": float(np.mean(occupancy)),
        "foreground_supported_voxel_fraction": float(np.mean(foreground_votes > 0)),
        "unknown_preserved_voxel_fraction": float(np.mean(unknown_votes > 0)),
        "known_background_is_empty_space": True,
        "hand_occluded_is_empty_space": False,
    }
    return occupancy.reshape(points.shape[:3]), diagnostics


def _latest_valid(values: tuple[TactilePatchObservation, ...]):
    selected = {}
    for value in values:
        if value.measurement_valid and (
            value.sensor_id not in selected or value.timestamp >= selected[value.sensor_id].timestamp
        ):
            selected[value.sensor_id] = value
    return tuple(selected[key] for key in sorted(selected))


def _sample_nearest(field: np.ndarray, axes, point: np.ndarray) -> float:
    index = tuple(int(np.clip(np.searchsorted(axis, point[i]), 1, len(axis)-1)) for i, axis in enumerate(axes))
    index = tuple(i if abs(axes[d][i]-point[d]) < abs(axes[d][i-1]-point[d]) else i-1
                  for d, i in enumerate(index))
    return float(field[index])


def _tactile_coverage(contacts: tuple[TactilePatchObservation, ...]) -> dict:
    points = np.asarray([item.representative_point_object_m for item in contacts], dtype=float)
    fingers = {item.finger_id for item in contacts}
    spread = 0.0
    if len(points) > 1:
        spread = float(np.max(np.linalg.norm(points[:, None]-points[None, :], axis=2)))
    coverage = min(1.0, len(contacts)/6.0)*min(1.0, len(fingers)/3.0)*min(1.0, spread/.06)
    return {
        "sensor_count": len(contacts), "finger_count": len(fingers),
        "spatial_spread_m": spread, "coverage_score": coverage,
    }


def _contact_point(field, axes, contact, mode):
    if mode == "representative":
        return np.asarray(contact.representative_point_object_m), "representative_point"
    patch = contact.patch_points_object_m
    values = np.asarray([abs(_sample_nearest(field, axes, point)) for point in patch])
    return patch[int(np.argmin(values))], "minimum_absolute_sdf_within_finite_patch"


@dataclass(frozen=True)
class DenseSDF:
    bounds_m: np.ndarray
    values_m: np.ndarray
    method: str
    status: str
    confidence: float
    diagnostics: Mapping
    runtime_ms: float

    @property
    def resolution(self) -> int:
        return int(self.values_m.shape[0])

    def axes(self):
        return tuple(np.linspace(self.bounds_m[0, i], self.bounds_m[1, i], self.values_m.shape[i])
                     for i in range(3))

    def surface_points(self, max_points: int = 2500) -> np.ndarray:
        axes = self.axes()
        result = []
        for axis in range(3):
            left = [slice(None)]*3; right = [slice(None)]*3
            left[axis] = slice(None, -1); right[axis] = slice(1, None)
            a, b = self.values_m[tuple(left)], self.values_m[tuple(right)]
            crossing = (a <= 0) != (b <= 0)
            for index in np.argwhere(crossing):
                p = np.array([axes[d][index[d]] for d in range(3)], dtype=float)
                j = index.copy(); j[axis] += 1
                q = np.array([axes[d][j[d]] for d in range(3)], dtype=float)
                va, vb = float(a[tuple(index)]), float(b[tuple(index)])
                ratio = abs(va)/(abs(va)+abs(vb)+1e-12)
                result.append(p+(q-p)*ratio)
        if not result:
            return np.empty((0, 3))
        result = np.asarray(result)
        if len(result) > max_points:
            indexes = np.linspace(0, len(result)-1, max_points, dtype=int)
            result = result[indexes]
        return result

    def to_dict(self, *, include_surface=True, max_points=1200):
        value = {
            "representation": "object-centric_dense_sdf_grid",
            "method": self.method, "status": self.status,
            "confidence": self.confidence, "bounds_m": self.bounds_m.tolist(),
            "grid_resolution": self.resolution, "runtime_ms": self.runtime_ms,
            "diagnostics": dict(self.diagnostics),
        }
        if include_surface:
            value["surface_points_object_m"] = self.surface_points(max_points).tolist()
        return value


def _reconstruct(
    observation: GeneralObjectObservation,
    method: str,
    config: Mapping,
    *,
    oracle_contact_points_object_m: np.ndarray | None = None,
) -> DenseSDF:
    """Reconstruct without accepting any object identity, radius, CAD or GT surface."""
    if method not in METHODS:
        raise ValueError(f"Unsupported general-object method: {method}")
    if method != "rgb_perfect_contact_oracle" and oracle_contact_points_object_m is not None:
        raise ValueError("Ground-truth contacts are accepted only by the explicit oracle method")
    if method == "rgb_perfect_contact_oracle" and oracle_contact_points_object_m is None:
        raise ValueError("The explicit oracle requires separately supplied evaluation-only contacts")
    started = time.perf_counter()
    axes, points = _grid(observation.reconstruction_bounds_m, observation.grid_resolution)
    spacing = float(np.mean([axis[1]-axis[0] for axis in axes]))
    contacts = _latest_valid(observation.tactile_observations)
    coverage = _tactile_coverage(contacts)
    use_rgb = method != "tactile_only"
    if use_rgb:
        occupancy, visual_diagnostics = _visual_hull(observation, points)
        if not occupancy.any() or occupancy.all():
            field = np.full(occupancy.shape, spacing*4, dtype=np.float32)
            status = "INVALID_RGB_CONSTRAINTS"
        else:
            field = _signed_manhattan_distance(occupancy, spacing)
            status = "VALID_RECONSTRUCTION"
    else:
        flat = points.reshape(-1, 3)
        if contacts:
            centers = np.asarray([item.representative_point_object_m for item in contacts])
            field = (np.min(np.linalg.norm(flat[:, None]-centers[None, :], axis=2), axis=1)
                     - float(config["tactile_only_support_radius_m"])).reshape(points.shape[:3]).astype(np.float32)
            status = "VALID_RECONSTRUCTION"
        else:
            field = np.full(points.shape[:3], spacing*4, dtype=np.float32)
            status = "INVALID_NO_TACTILE"
        visual_diagnostics = {"candidate_voxel_fraction": None,
                              "hand_occluded_is_empty_space": False}

    applied = []
    mode = None
    if method == "rgb_representative_tactile":
        mode = "representative"
    elif method in {"rgb_finite_patch_tactile", "rgb_reliability_tactile"}:
        mode = "finite_patch"
    band = float(config["contact_influence_radius_m"])
    flat_points = points.reshape(-1, 3)
    if method == "rgb_perfect_contact_oracle":
        candidates = [(np.asarray(point), "evaluation_only_exact_contact", 1.0, "oracle")
                      for point in np.asarray(oracle_contact_points_object_m)]
    elif mode:
        candidates = []
        for contact in contacts:
            point, source = _contact_point(field, axes, contact, mode)
            if method == "rgb_reliability_tactile":
                uncertainty = 1.0/(1.0+(contact.contact_region_sigma_m/band)**2)
                reliability = uncertainty*coverage["coverage_score"]
            else:
                reliability = 1.0
            candidates.append((point, source, reliability, contact.sensor_id))
    else:
        candidates = []
    for point, source, weight, sensor_id in candidates:
        before = _sample_nearest(field, axes, point)
        gaussian = np.exp(-np.sum((flat_points-point)**2, axis=1)/(2*band*band)).reshape(field.shape)
        field = field-weight*before*gaussian
        applied.append({"sensor_id": sensor_id, "source": source,
                        "weight": float(weight), "pre_constraint_sdf_m": before,
                        "point_object_m": point.tolist()})
    surface_count = len(DenseSDF(observation.reconstruction_bounds_m, field, method, status,
                                 0, {}, 0).surface_points(100000))
    if surface_count < int(config["minimum_surface_points"]):
        status = "INVALID_NO_SURFACE"
    segmentation = float(np.mean([frame.segmentation_confidence for frame in observation.rgb_observations]))
    confidence = segmentation if method == "rgb_only" else min(segmentation, .5+.5*coverage["coverage_score"])
    if method == "tactile_only":
        confidence = coverage["coverage_score"]
    diagnostics = {
        **visual_diagnostics, **coverage,
        "input_boundary": observation.input_boundary,
        "active_raw_tactile_count": len(observation.tactile_observations),
        "effective_latest_sensor_count": len(contacts),
        "tactile_constraint_mode": mode or ("oracle" if candidates else "none"),
        "applied_contact_constraints": applied,
        "unknown_rays_penalized_as_empty": False,
        "inactive_sensors_used_as_free_space": False,
        "ground_truth_input": method == "rgb_perfect_contact_oracle",
        "object_identity_input": False,
        "known_radius_input": False,
        "parameter_count": 0,
        "estimated_grid_operations": int(observation.grid_resolution**3*(
            30+len(candidates)*12)),
        "surface_point_count": surface_count,
    }
    return DenseSDF(
        observation.reconstruction_bounds_m, field.astype(np.float32), method, status,
        float(np.clip(confidence, 0, 1)), diagnostics,
        (time.perf_counter()-started)*1000.0,
    )


def reconstruct_general_object(
    observation: GeneralObjectObservation, method: str, config: Mapping
) -> DenseSDF:
    """Proposed-method entry point; its signature cannot accept evaluator truth."""
    if method == "rgb_perfect_contact_oracle":
        raise ValueError("Use the separately labeled evaluation-only oracle entry point")
    return _reconstruct(observation, method, config)


def reconstruct_perfect_contact_oracle(
    observation: GeneralObjectObservation,
    config: Mapping,
    evaluation_only_contact_points_object_m: np.ndarray,
) -> DenseSDF:
    """Explicit evaluator-only upper-bound entry point, never used by proposals."""
    return _reconstruct(
        observation, "rgb_perfect_contact_oracle", config,
        oracle_contact_points_object_m=evaluation_only_contact_points_object_m,
    )
