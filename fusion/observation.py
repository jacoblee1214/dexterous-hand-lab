"""Exact measured/estimated observation boundary for geometric fusion."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from types import MappingProxyType
from typing import Iterable, Mapping

import numpy as np

from reconstruction.temporal_observation import TemporalTactileObservation
from robot_data.common import CameraObservation, CommonRobotState, ProviderDescription
from sensors.sensor_kinematics import KinematicTree, load_sensor_mounts
from vision.baseline import SegmentationResult
from vision.calibration import CalibratedCamera


Vector3 = tuple[float, float, float]


@lru_cache(maxsize=8)
def _mount_registry(path: str):
    return load_sensor_mounts(path)


@lru_cache(maxsize=8)
def _kinematic_tree(path: str):
    return KinematicTree.from_urdf(path)


def _vector(name: str, value, size: int = 3) -> tuple[float, ...]:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {size}-vector")
    return tuple(float(item) for item in array)


def _readonly_mask(name: str, value, shape: tuple[int, int]) -> np.ndarray:
    result = np.asarray(value, dtype=bool).copy()
    if result.shape != shape:
        raise ValueError(f"{name} must have image shape {shape}")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class TactileContactConstraint:
    timestamp: float
    sensor_id: str
    finger_id: str
    parent_link: str
    joint_state_snapshot: Mapping[str, float]
    sensor_type: str
    sensor_center_world_m: Vector3
    surface_u_world: Vector3
    surface_v_world: Vector3
    sensing_direction_world: Vector3
    representative_contact_world_m: Vector3
    surface_width_m: float
    surface_height_m: float
    fingertip_radii_m: Vector3 | None
    scalar_measurement_n: float
    indentation_m: float
    position_sigma_m: float
    finite_patch_sigma_m: float
    activation_valid: bool = True
    response_calibration: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        timestamp = float(self.timestamp)
        scalars = (
            self.surface_width_m,
            self.surface_height_m,
            self.scalar_measurement_n,
            self.indentation_m,
            self.position_sigma_m,
            self.finite_patch_sigma_m,
        )
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Tactile constraint timestamp must be finite and non-negative")
        if any(not math.isfinite(float(value)) or float(value) < 0 for value in scalars):
            raise ValueError("Tactile constraint scalars must be finite and non-negative")
        if min(self.surface_width_m, self.surface_height_m, self.position_sigma_m,
               self.finite_patch_sigma_m) <= 0:
            raise ValueError("Patch dimensions and uncertainty scales must be positive")
        if self.sensor_type not in {"rectangular", "fingertip_surface"}:
            raise ValueError("Unsupported tactile sensor surface type")
        for name in (
            "sensor_center_world_m", "surface_u_world", "surface_v_world",
            "sensing_direction_world", "representative_contact_world_m",
        ):
            object.__setattr__(self, name, _vector(name, getattr(self, name)))
        if not self.sensor_id or not self.finger_id or not self.parent_link:
            raise ValueError("Tactile constraint identifiers are required")
        joints = {str(name): float(value) for name, value in self.joint_state_snapshot.items()}
        if not joints or any(not math.isfinite(value) for value in joints.values()):
            raise ValueError("Tactile constraint requires a finite named joint snapshot")
        object.__setattr__(self, "joint_state_snapshot", MappingProxyType(joints))
        calibration = {str(name): float(value) for name, value in (self.response_calibration or {}).items()}
        required = {"minimum_activation_force_n", "effective_sensor_area_m2",
                    "virtual_stiffness_n_per_m", "saturation_pressure_pa",
                    "contact_projection_sign"}
        if not required.issubset(calibration) or any(not math.isfinite(value) for value in calibration.values()):
            raise ValueError("Tactile response calibration is incomplete or non-finite")
        if min(calibration[name] for name in required-{"contact_projection_sign"}) <= 0:
            raise ValueError("Tactile response calibration scales must be positive")
        if calibration["contact_projection_sign"] not in {-1.0, 1.0}:
            raise ValueError("Tactile contact projection sign must be +/-1")
        object.__setattr__(self, "response_calibration", MappingProxyType(calibration))
        basis = np.column_stack((self.surface_u_world, self.surface_v_world,
                                 self.sensing_direction_world))
        if not np.allclose(basis.T @ basis, np.eye(3), atol=1e-5):
            raise ValueError("Tactile surface U/V/normal axes must be orthonormal")
        if self.sensor_type == "fingertip_surface":
            if self.fingertip_radii_m is None:
                raise ValueError("Fingertip constraint requires ellipsoid radii")
            radii = _vector("fingertip_radii_m", self.fingertip_radii_m)
            if min(radii) <= 0:
                raise ValueError("Fingertip radii must be positive")
            object.__setattr__(self, "fingertip_radii_m", radii)
        object.__setattr__(self, "timestamp", timestamp)

    def representative_point(self) -> np.ndarray:
        return np.asarray(self.representative_contact_world_m, dtype=float)

    def patch_points_world(self, resolution: int = 9) -> np.ndarray:
        """Deterministic patch samples; no MuJoCo contact is consulted."""
        if resolution < 3 or resolution % 2 == 0:
            raise ValueError("Patch resolution must be an odd integer >= 3")
        center = np.asarray(self.sensor_center_world_m, dtype=float)
        u_axis = np.asarray(self.surface_u_world, dtype=float)
        v_axis = np.asarray(self.surface_v_world, dtype=float)
        normal = np.asarray(self.sensing_direction_world, dtype=float)
        coordinates = np.linspace(-1.0, 1.0, resolution)
        points = []
        for u in coordinates:
            for v in coordinates:
                if self.sensor_type == "rectangular":
                    local_u = u * self.surface_width_m / 2.0
                    local_v = v * self.surface_height_m / 2.0
                    local_normal = 0.0
                else:
                    if u * u + v * v > 1.0 + 1e-12:
                        continue
                    radii = np.asarray(self.fingertip_radii_m, dtype=float)
                    local_u = u * radii[0]
                    local_v = v * radii[1]
                    local_normal = radii[2] * (math.sqrt(max(0.0, 1-u*u-v*v)) - 1.0)
                points.append(
                    center + local_u*u_axis + local_v*v_axis
                    + (local_normal
                       + self.response_calibration["contact_projection_sign"]*self.indentation_m)*normal
                )
        return np.asarray(points, dtype=float)


@dataclass(frozen=True)
class FusionObservation:
    timestamp: float
    camera_timestamp: float
    tactile_timestamp: float
    frame_id: str
    camera_id: str
    common_frame_id: str
    rgb_reference: str | None
    image_width: int
    image_height: int
    intrinsic_matrix: np.ndarray
    camera_cv_from_world: np.ndarray
    predicted_object_mask: np.ndarray
    known_background_mask: np.ndarray
    unknown_occluded_mask: np.ndarray
    segmentation_status: str
    segmentation_confidence: float
    segmentation_angular_coverage: float
    joint_state: Mapping[str, float]
    tactile_constraints: tuple[TactileContactConstraint, ...]
    known_radius_m: float
    radius_label: str
    calibration_version: str
    time_base: str
    object_motion_assumption: str
    unreliable_segmentation_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        stamps = (float(self.timestamp), float(self.camera_timestamp), float(self.tactile_timestamp))
        if any(not math.isfinite(value) or value < 0 for value in stamps):
            raise ValueError("Fusion timestamps must be finite and non-negative")
        if self.object_motion_assumption not in {"fixed_pose", "instantaneous_only"}:
            raise ValueError("Fusion object motion assumption must be explicit")
        if self.object_motion_assumption == "instantaneous_only" and max(stamps) - min(stamps) > 1e-9:
            raise ValueError("Moving/free objects require one instantaneous synchronized sample")
        if not self.common_frame_id:
            raise ValueError("Fusion common coordinate frame is required")
        if any(contact.timestamp > self.camera_timestamp + 1e-9 for contact in self.tactile_constraints):
            raise ValueError("Tactile observations cannot be newer than the camera observation")
        if self.object_motion_assumption == "instantaneous_only" and any(
            abs(contact.timestamp - self.tactile_timestamp) > 1e-9
            for contact in self.tactile_constraints
        ):
            raise ValueError("Moving/free objects cannot use historical tactile accumulation")
        radius = float(self.known_radius_m)
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("A positive declared sphere radius prior is required")
        if self.radius_label != "declared_known_radius_prior":
            raise ValueError("Known radius must be explicitly labeled as a prior")
        intrinsic = np.asarray(self.intrinsic_matrix, dtype=float).copy()
        extrinsic = np.asarray(self.camera_cv_from_world, dtype=float).copy()
        if intrinsic.shape != (3, 3) or extrinsic.shape != (4, 4):
            raise ValueError("Fusion calibration requires 3x3 K and 4x4 T_camera_from_world")
        if not np.isfinite(intrinsic).all() or not np.isfinite(extrinsic).all():
            raise ValueError("Fusion calibration must be finite")
        height, width = int(self.image_height), int(self.image_width)
        if height <= 0 or width <= 0:
            raise ValueError("Fusion image dimensions must be positive")
        shape = (height, width)
        object.__setattr__(self, "predicted_object_mask",
                           _readonly_mask("predicted_object_mask", self.predicted_object_mask, shape))
        object.__setattr__(self, "known_background_mask",
                           _readonly_mask("known_background_mask", self.known_background_mask, shape))
        object.__setattr__(self, "unknown_occluded_mask",
                           _readonly_mask("unknown_occluded_mask", self.unknown_occluded_mask, shape))
        unreliable = (np.zeros(shape, dtype=bool) if self.unreliable_segmentation_mask is None
                      else self.unreliable_segmentation_mask)
        object.__setattr__(self, "unreliable_segmentation_mask",
                           _readonly_mask("unreliable_segmentation_mask", unreliable, shape))
        if np.any(self.predicted_object_mask & self.known_background_mask):
            raise ValueError("Object and known-background classifications must be disjoint")
        if np.any(self.predicted_object_mask & self.unknown_occluded_mask):
            raise ValueError("Object and unknown/occluded classifications must be disjoint")
        if np.any(self.known_background_mask & self.unknown_occluded_mask):
            raise ValueError("Background and unknown/occluded classifications must be disjoint")
        if np.any(self.unreliable_segmentation_mask & self.predicted_object_mask):
            raise ValueError("Unreliable and foreground classifications must be disjoint")
        if np.any(self.unreliable_segmentation_mask & self.known_background_mask):
            raise ValueError("Unreliable and known-background classifications must be disjoint")
        if np.any(self.unreliable_segmentation_mask & self.unknown_occluded_mask):
            raise ValueError("Unreliable and unknown/occluded classifications must be disjoint")
        intrinsic.setflags(write=False)
        extrinsic.setflags(write=False)
        object.__setattr__(self, "intrinsic_matrix", intrinsic)
        object.__setattr__(self, "camera_cv_from_world", extrinsic)
        joints = {str(name): float(value) for name, value in self.joint_state.items()}
        if any(not math.isfinite(value) for value in joints.values()):
            raise ValueError("Fusion joint state must contain finite named values")
        object.__setattr__(self, "joint_state", MappingProxyType(joints))
        object.__setattr__(self, "tactile_constraints", tuple(self.tactile_constraints))
        object.__setattr__(self, "timestamp", stamps[0])
        object.__setattr__(self, "camera_timestamp", stamps[1])
        object.__setattr__(self, "tactile_timestamp", stamps[2])
        object.__setattr__(self, "known_radius_m", radius)

    @property
    def image_shape(self) -> tuple[int, int]:
        return self.predicted_object_mask.shape


def _dilate(mask: np.ndarray, pixels: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    for _ in range(max(0, int(pixels))):
        padded = np.pad(result, 1, constant_values=False)
        result = (
            padded[1:-1, 1:-1] | padded[:-2, 1:-1] | padded[2:, 1:-1]
            | padded[1:-1, :-2] | padded[1:-1, 2:]
        )
    return result


def classify_visibility(
    rgb: np.ndarray, object_mask: np.ndarray, settings
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Controlled RGB-only hand/unknown heuristic; no segmentation IDs or depth."""
    image = np.asarray(rgb, dtype=np.uint8)
    maximum = image.max(axis=2).astype(int)
    spread = maximum - image.min(axis=2).astype(int)
    margin = int(settings.get("unreliable_color_margin", 0))
    unreliable_candidate = np.zeros_like(object_mask, dtype=bool)
    if margin > 0:
        red, green, blue = [image[:, :, index].astype(int) for index in range(3)]
        unreliable_candidate = (
            (blue >= int(settings.get("segmentation_minimum_blue", 45)) - margin)
            & (blue - red >= int(settings.get("segmentation_minimum_blue_minus_red", 35)) - margin)
            & (blue - green >= int(settings.get("segmentation_minimum_blue_minus_green", 14)) - margin)
            & ~object_mask
        )
    likely_hand = (
        (maximum >= int(settings["hand_minimum_intensity"]))
        & (spread <= int(settings["hand_maximum_channel_spread"]))
        & ~object_mask & ~unreliable_candidate
    )
    unknown = _dilate(likely_hand, int(settings["unknown_dilation_pixels"])) & ~object_mask
    unreliable = unreliable_candidate & ~unknown
    background = ~(object_mask | unknown | unreliable)
    return background, unknown, unreliable


def build_fusion_observation(
    *,
    rgb: np.ndarray,
    segmentation: SegmentationResult,
    calibration: CalibratedCamera,
    camera_observation: CameraObservation,
    common_state: CommonRobotState,
    description: ProviderDescription,
    tactile_observations: Iterable[TemporalTactileObservation],
    known_radius_m: float,
    config: Mapping,
    fixed_object_pose: bool = True,
) -> FusionObservation:
    """Strip provider state into the only object visible to fusion code."""
    if camera_observation.timestamp != common_state.timestamp:
        raise ValueError("Camera and synchronized tactile state timestamps differ")
    if camera_observation.depth_available or camera_observation.depth_reference:
        raise ValueError("Fusion v1 does not accept depth")
    mounts = _mount_registry(str(description.sensor_config_path))
    tree = _kinematic_tree(str(description.kinematic_model_path))
    uncertainty = config["uncertainty"]
    constraints = []
    transforms_by_joint_state = {}
    for sample in tactile_observations:
        if sample.timestamp > common_state.timestamp + 1e-9:
            raise ValueError("Tactile observations cannot be newer than synchronized state")
        if not fixed_object_pose and abs(sample.timestamp - common_state.timestamp) > 1e-9:
            raise ValueError("Historical tactile accumulation requires a fixed object pose")
        mount = mounts[sample.sensor_id]
        state_key = tuple(sorted(sample.joint_state_snapshot.items()))
        if state_key not in transforms_by_joint_state:
            transforms_by_joint_state[state_key] = tree.forward_kinematics(sample.joint_state_snapshot)
        transform = transforms_by_joint_state[state_key][mount.parent_link] @ mount.link_to_sensor
        rotation = transform[:3, :3]
        relative = float(uncertainty["indentation_relative_sigma"]) * sample.estimated_indentation
        position_sigma = math.sqrt(
            float(uncertainty["representative_position_sigma_m"]) ** 2
            + float(uncertainty["sensor_mount_sigma_m"]) ** 2 + relative ** 2
        )
        constraints.append(TactileContactConstraint(
            timestamp=sample.timestamp,
            sensor_id=sample.sensor_id,
            finger_id=sample.finger_id,
            parent_link=sample.parent_link,
            joint_state_snapshot=sample.joint_state_snapshot,
            sensor_type=mount.sensor_type,
            sensor_center_world_m=transform[:3, 3],
            surface_u_world=rotation[:, 0],
            surface_v_world=rotation[:, 1],
            sensing_direction_world=rotation[:, 2],
            representative_contact_world_m=sample.estimated_contact_position,
            surface_width_m=mount.surface_width_m,
            surface_height_m=mount.surface_height_m,
            fingertip_radii_m=mount.fingertip_radii_xyz,
            scalar_measurement_n=sample.scalar_sensor_value,
            indentation_m=sample.estimated_indentation,
            position_sigma_m=position_sigma,
            finite_patch_sigma_m=math.sqrt(
                float(uncertainty["finite_patch_surface_sigma_m"]) ** 2
                + float(uncertainty["sensor_mount_sigma_m"]) ** 2 + relative ** 2
            ),
            activation_valid=sample.scalar_sensor_value > 0,
            response_calibration=description.calibration_config[sample.sensor_id],
        ))
    background, unknown, unreliable = classify_visibility(
        rgb, segmentation.mask, config["rgb_visibility"]
    )
    return FusionObservation(
        timestamp=common_state.timestamp,
        camera_timestamp=camera_observation.timestamp,
        tactile_timestamp=common_state.timestamp,
        frame_id=camera_observation.frame_id,
        camera_id=camera_observation.camera_id or camera_observation.frame_id,
        common_frame_id=calibration.config.parent_frame,
        rgb_reference=camera_observation.rgb_reference,
        image_width=int(camera_observation.image_width or rgb.shape[1]),
        image_height=int(camera_observation.image_height or rgb.shape[0]),
        intrinsic_matrix=calibration.intrinsic_matrix,
        camera_cv_from_world=calibration.camera_cv_from_world,
        predicted_object_mask=segmentation.mask,
        known_background_mask=background,
        unknown_occluded_mask=unknown,
        unreliable_segmentation_mask=unreliable,
        segmentation_status=segmentation.status,
        segmentation_confidence=segmentation.confidence,
        segmentation_angular_coverage=segmentation.angular_boundary_coverage,
        joint_state=common_state.joint_positions,
        tactile_constraints=tuple(constraints),
        known_radius_m=known_radius_m,
        radius_label="declared_known_radius_prior",
        calibration_version=camera_observation.calibration_version or "unversioned",
        time_base=camera_observation.time_base or "unspecified",
        object_motion_assumption="fixed_pose" if fixed_object_pose else "instantaneous_only",
    )
