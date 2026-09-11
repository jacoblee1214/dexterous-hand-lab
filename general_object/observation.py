"""Strict algorithm-input boundary for general-object reconstruction.

No field in these types names an object class, analytic primitive, CAD asset,
ground-truth pose, ground-truth surface point, or ground-truth normal.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np


def _finite_array(name: str, value, shape=None, *, dtype=float, readonly=True):
    result = np.asarray(value, dtype=dtype).copy()
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if dtype is not bool and not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    if readonly:
        result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ObjectFrameInitialization:
    frame_id: str
    parent_frame: str
    timestamp: float
    world_from_object: np.ndarray
    covariance_6x6: np.ndarray
    source: str
    validity: str

    def __post_init__(self):
        if self.source not in {"declared_static_fixture", "measured_pose_tracker"}:
            raise ValueError("Object frame must come from a declared fixture or measured tracker")
        if self.validity != "VALID":
            raise ValueError("Invalid object-frame initialization cannot enter reconstruction")
        if not self.frame_id or not self.parent_frame:
            raise ValueError("Object and parent frame IDs are required")
        if not math.isfinite(float(self.timestamp)) or self.timestamp < 0:
            raise ValueError("Object-frame timestamp must be finite and non-negative")
        transform = _finite_array("world_from_object", self.world_from_object, (4, 4))
        covariance = _finite_array("covariance_6x6", self.covariance_6x6, (6, 6))
        if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-9):
            raise ValueError("Object transform must be homogeneous")
        if np.linalg.det(transform[:3, :3]) < 0.999:
            raise ValueError("Object transform rotation must be right handed")
        if np.min(np.linalg.eigvalsh(covariance)) < -1e-12:
            raise ValueError("Object-frame covariance must be positive semidefinite")
        object.__setattr__(self, "world_from_object", transform)
        object.__setattr__(self, "covariance_6x6", covariance)


@dataclass(frozen=True)
class RGBObjectObservation:
    timestamp: float
    frame_id: str
    camera_id: str
    calibration_version: str
    rgb: np.ndarray
    intrinsic_matrix: np.ndarray
    camera_from_object: np.ndarray
    object_mask: np.ndarray
    known_background_mask: np.ndarray
    hand_occluded_mask: np.ndarray
    unreliable_mask: np.ndarray
    segmentation_confidence: float
    feature_encoder: str
    feature_summary: tuple[float, ...]
    depth_prior_reference: str | None = None
    normal_prior_reference: str | None = None

    def __post_init__(self):
        rgb = _finite_array("rgb", self.rgb, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("RGB must be HxWx3")
        shape = rgb.shape[:2]
        masks = []
        for name in ("object_mask", "known_background_mask", "hand_occluded_mask", "unreliable_mask"):
            mask = _finite_array(name, getattr(self, name), shape, dtype=bool)
            masks.append(mask)
            object.__setattr__(self, name, mask)
        total = sum(mask.astype(np.uint8) for mask in masks)
        if np.any(total > 1):
            raise ValueError("RGB visibility classes must be mutually exclusive")
        if not 0 <= float(self.segmentation_confidence) <= 1:
            raise ValueError("Segmentation confidence must be in [0, 1]")
        if not self.frame_id or not self.camera_id or not self.calibration_version:
            raise ValueError("Named camera/calibration metadata are required")
        if not self.feature_encoder or not all(math.isfinite(v) for v in self.feature_summary):
            raise ValueError("A finite modular RGB feature summary is required")
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "intrinsic_matrix", _finite_array(
            "intrinsic_matrix", self.intrinsic_matrix, (3, 3)))
        object.__setattr__(self, "camera_from_object", _finite_array(
            "camera_from_object", self.camera_from_object, (4, 4)))


@dataclass(frozen=True)
class TactilePatchObservation:
    timestamp: float
    sensor_id: str
    finger_id: str
    parent_link: str
    calibration_version: str
    object_from_sensor: np.ndarray
    representative_point_object_m: tuple[float, float, float]
    patch_points_object_m: np.ndarray
    sensing_normal_object: tuple[float, float, float]
    scalar_measurement_n: float
    estimated_indentation_m: float
    contact_region_sigma_m: float
    measurement_valid: bool
    joint_state: Mapping[str, float]

    def __post_init__(self):
        if not self.sensor_id or not self.finger_id or not self.parent_link:
            raise ValueError("Named tactile sensor, finger and link are required")
        if not self.calibration_version:
            raise ValueError("Tactile calibration version is required")
        scalars = (self.timestamp, self.scalar_measurement_n,
                   self.estimated_indentation_m, self.contact_region_sigma_m)
        if any(not math.isfinite(float(v)) for v in scalars) or min(scalars) < 0:
            raise ValueError("Tactile timestamp/measurement/uncertainty must be non-negative")
        if self.contact_region_sigma_m <= 0:
            raise ValueError("Contact-region uncertainty must be positive")
        representative = _finite_array(
            "representative_point_object_m", self.representative_point_object_m, (3,))
        patch = _finite_array("patch_points_object_m", self.patch_points_object_m)
        normal = _finite_array("sensing_normal_object", self.sensing_normal_object, (3,))
        sensor_pose = _finite_array("object_from_sensor", self.object_from_sensor, (4, 4))
        if not np.allclose(sensor_pose[3], [0, 0, 0, 1], atol=1e-9):
            raise ValueError("Tactile sensor pose must be homogeneous")
        if patch.ndim != 2 or patch.shape[1] != 3 or len(patch) < 9:
            raise ValueError("Finite tactile patch needs at least nine 3D samples")
        norm = np.linalg.norm(normal)
        if not np.isclose(norm, 1.0, atol=1e-5):
            raise ValueError("Sensing normal must be unit length")
        joints = {str(k): float(v) for k, v in self.joint_state.items()}
        if any(not math.isfinite(v) for v in joints.values()):
            raise ValueError("Named joint state must be finite")
        object.__setattr__(self, "representative_point_object_m", tuple(representative))
        object.__setattr__(self, "object_from_sensor", sensor_pose)
        object.__setattr__(self, "patch_points_object_m", patch)
        object.__setattr__(self, "sensing_normal_object", tuple(normal))
        object.__setattr__(self, "joint_state", MappingProxyType(joints))


@dataclass(frozen=True)
class GeneralObjectObservation:
    timestamp: float
    time_base: str
    object_frame: ObjectFrameInitialization
    rgb_observations: tuple[RGBObjectObservation, ...]
    tactile_observations: tuple[TactilePatchObservation, ...]
    reconstruction_bounds_m: np.ndarray
    grid_resolution: int
    static_pose_assumption: str
    joint_state: Mapping[str, float]

    def __post_init__(self):
        if self.static_pose_assumption != "declared_rigid_fixture_no_motion":
            raise ValueError("Milestone 4 accepts only the declared static fixture assumption")
        if not self.time_base or not self.rgb_observations:
            raise ValueError("Time base and at least one RGB observation are required")
        if not 16 <= int(self.grid_resolution) <= 64:
            raise ValueError("Dense-grid resolution must be between 16 and 64")
        bounds = _finite_array("reconstruction_bounds_m", self.reconstruction_bounds_m, (2, 3))
        if np.any(bounds[1] <= bounds[0]):
            raise ValueError("Reconstruction bounds must be increasing")
        stamps = [self.timestamp, self.object_frame.timestamp]
        stamps += [item.timestamp for item in self.rgb_observations]
        stamps += [item.timestamp for item in self.tactile_observations]
        if max(stamps)-min(stamps) > 1e-6:
            raise ValueError("Static bundle timestamps must be synchronized")
        joints = {str(key): float(value) for key, value in self.joint_state.items()}
        if not joints or any(not math.isfinite(value) for value in joints.values()):
            raise ValueError("A finite named joint-state snapshot is required")
        object.__setattr__(self, "reconstruction_bounds_m", bounds)
        object.__setattr__(self, "rgb_observations", tuple(self.rgb_observations))
        object.__setattr__(self, "tactile_observations", tuple(self.tactile_observations))
        object.__setattr__(self, "joint_state", MappingProxyType(joints))

    @property
    def input_boundary(self) -> str:
        return (
            "calibrated RGB/features/visibility + declared static object frame + "
            "named scalar tactile patches; no CAD, shape identity, GT pose, GT surface or GT normals"
        )
