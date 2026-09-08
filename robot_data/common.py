"""Versioned data and command types shared by simulation, replay, and hardware."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from types import MappingProxyType
from typing import Any, Mapping


COMMON_STATE_SCHEMA_VERSION = "1.0.0"


def _timestamp(value: float, name: str = "timestamp") -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _finite_map(values: Mapping[str, float], name: str) -> Mapping[str, float]:
    result = {str(key): float(value) for key, value in values.items()}
    if any(not math.isfinite(value) for value in result.values()):
        raise ValueError(f"{name} must contain finite named values")
    return MappingProxyType(result)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class JointInfo:
    name: str
    lower_limit_rad: float
    upper_limit_rad: float
    parent_link: str
    child_link: str

    def __post_init__(self) -> None:
        if not self.name or not self.parent_link or not self.child_link:
            raise ValueError("Joint names and links are required")
        lower, upper = float(self.lower_limit_rad), float(self.upper_limit_rad)
        if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
            raise ValueError(f"Invalid limits for joint {self.name}")
        object.__setattr__(self, "lower_limit_rad", lower)
        object.__setattr__(self, "upper_limit_rad", upper)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "lower_limit_rad": self.lower_limit_rad,
            "upper_limit_rad": self.upper_limit_rad,
            "parent_link": self.parent_link,
            "child_link": self.child_link,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "JointInfo":
        return cls(**dict(value))


@dataclass(frozen=True)
class JointStateSample:
    timestamp: float
    positions: Mapping[str, float]
    velocities: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", _timestamp(self.timestamp))
        positions = _finite_map(self.positions, "joint positions")
        velocities = _finite_map(self.velocities, "joint velocities")
        if set(positions) != set(velocities):
            raise ValueError("Joint position and velocity names must match")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "velocities", velocities)


@dataclass(frozen=True)
class TactileChannelSample:
    timestamp: float
    sensor_id: str
    scalar_value: float
    pressure: float
    active: bool
    force: float | None = None
    indentation: float | None = None

    def __post_init__(self) -> None:
        if not self.sensor_id:
            raise ValueError("sensor_id is required")
        object.__setattr__(self, "timestamp", _timestamp(self.timestamp))
        for name in ("scalar_value", "pressure"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        for name in ("force", "indentation"):
            value = getattr(self, name)
            if value is not None:
                value = float(value)
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"{name} must be finite and non-negative")
                object.__setattr__(self, name, value)
        object.__setattr__(self, "active", bool(self.active))

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "sensor_id": self.sensor_id,
            "scalar_value": self.scalar_value,
            "pressure": self.pressure,
            "active": self.active,
            "force": self.force,
            "indentation": self.indentation,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TactileChannelSample":
        return cls(**dict(value))


@dataclass(frozen=True)
class TactileStateSample:
    timestamp: float
    channels: tuple[TactileChannelSample, ...]

    def __post_init__(self) -> None:
        timestamp = _timestamp(self.timestamp)
        channels = tuple(self.channels)
        identifiers = [channel.sensor_id for channel in channels]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Tactile sensor IDs must be unique")
        if any(not math.isclose(channel.timestamp, timestamp, rel_tol=0, abs_tol=1e-12) for channel in channels):
            raise ValueError("Every tactile channel must carry the sample timestamp")
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "channels", channels)


@dataclass(frozen=True)
class CameraObservation:
    timestamp: float
    frame_id: str
    rgb_available: bool
    depth_available: bool
    camera_id: str | None = None
    image_width: int | None = None
    image_height: int | None = None
    source: str | None = None
    calibration_version: str | None = None
    intrinsic_calibration_reference: str | None = None
    rgb_reference: str | None = None
    depth_reference: str | None = None
    intrinsics: Mapping[str, Any] | None = None
    extrinsics_reference: str | None = None
    time_base: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", _timestamp(self.timestamp))
        if not self.frame_id:
            raise ValueError("camera frame_id is required")
        if bool(self.image_width is None) != bool(self.image_height is None):
            raise ValueError("Camera image width and height must be provided together")
        if self.image_width is not None:
            width, height = int(self.image_width), int(self.image_height)
            if width <= 0 or height <= 0:
                raise ValueError("Camera image dimensions must be positive")
            object.__setattr__(self, "image_width", width)
            object.__setattr__(self, "image_height", height)
        if self.rgb_reference and not self.rgb_available:
            raise ValueError("RGB reference requires rgb_available")
        if self.depth_reference and not self.depth_available:
            raise ValueError("Depth reference requires depth_available")
        if self.rgb_reference or self.depth_reference:
            if not self.time_base or not self.extrinsics_reference or not self.intrinsics:
                raise ValueError("Image references require explicit time base, intrinsics and extrinsics")
        if self.intrinsics is not None:
            values = dict(self.intrinsics)
            for key in ("width", "height", "fx", "fy", "cx", "cy"):
                if key not in values or not math.isfinite(float(values[key])):
                    raise ValueError(f"Camera intrinsics require finite {key}")
            if min(values[key] for key in ("width", "height", "fx", "fy")) <= 0:
                raise ValueError("Image dimensions and focal lengths must be positive")
            if not values.get("distortion_model"):
                raise ValueError("An explicit camera distortion model is required")
            object.__setattr__(self, "intrinsics", MappingProxyType(values))

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "timestamp": self.timestamp,
            "frame_id": self.frame_id,
            "rgb_available": bool(self.rgb_available),
            "depth_available": bool(self.depth_available),
        }
        for key in (
            "camera_id", "image_width", "image_height", "source",
            "calibration_version", "intrinsic_calibration_reference",
            "rgb_reference", "depth_reference", "intrinsics",
            "extrinsics_reference", "time_base",
        ):
            value = getattr(self, key)
            if value is not None:
                payload[key] = _plain(value)
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CameraObservation":
        return cls(**dict(value))


@dataclass(frozen=True)
class LinkPose:
    """Display-only named body pose in robot-base/world coordinates, metres."""

    name: str
    parent_link: str | None
    timestamp: float
    position_xyz: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]

    def __post_init__(self):
        if not self.name:
            raise ValueError("Link pose requires a name")
        object.__setattr__(self, "timestamp", _timestamp(self.timestamp))
        for field_name, size in (("position_xyz", 3), ("quaternion_wxyz", 4)):
            values = tuple(float(value) for value in getattr(self, field_name))
            if len(values) != size or not all(math.isfinite(value) for value in values):
                raise ValueError(f"Invalid link {field_name}")
            object.__setattr__(self, field_name, values)
        if not math.isclose(sum(value * value for value in self.quaternion_wxyz), 1.0, abs_tol=1e-6):
            raise ValueError("Link quaternion must be unit length")

    def to_dict(self):
        return {"name": self.name, "parent_link": self.parent_link, "timestamp": self.timestamp,
                "position_xyz": list(self.position_xyz), "quaternion_wxyz": list(self.quaternion_wxyz)}


@dataclass(frozen=True)
class CommonRobotState:
    """A tactile sample plus named joint state interpolated at its timestamp."""

    timestamp: float
    joint_positions: Mapping[str, float]
    joint_velocities: Mapping[str, float]
    tactile_channels: tuple[TactileChannelSample, ...]
    source: str
    camera: CameraObservation | None = None
    joint_targets: Mapping[str, float] = field(default_factory=dict)
    object_state: Mapping[str, Any] | None = None
    grasp_state: Mapping[str, Any] | None = None
    schema_version: str = COMMON_STATE_SCHEMA_VERSION
    synchronization: Mapping[str, Any] = field(default_factory=dict)
    link_poses: tuple[LinkPose, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != COMMON_STATE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported common-state schema {self.schema_version}")
        object.__setattr__(self, "timestamp", _timestamp(self.timestamp))
        if not self.source:
            raise ValueError("source is required")
        poses = tuple(self.link_poses)
        if len({pose.name for pose in poses}) != len(poses):
            raise ValueError("Duplicate link pose names")
        if any(pose.timestamp != self.timestamp for pose in poses):
            raise ValueError("Link poses must share the common-state timestamp")
        object.__setattr__(self, "link_poses", poses)
        positions = _finite_map(self.joint_positions, "joint positions")
        velocities = _finite_map(self.joint_velocities, "joint velocities")
        if set(positions) != set(velocities):
            raise ValueError("Joint position and velocity names must match")
        channels = tuple(self.tactile_channels)
        if len({item.sensor_id for item in channels}) != len(channels):
            raise ValueError("Duplicate common-state sensor IDs")
        if any(not math.isclose(item.timestamp, self.timestamp, rel_tol=0, abs_tol=1e-12) for item in channels):
            raise ValueError("Tactile channels must match the synchronized timestamp")
        object.__setattr__(self, "joint_positions", positions)
        object.__setattr__(self, "joint_velocities", velocities)
        object.__setattr__(self, "joint_targets", _finite_map(self.joint_targets, "joint targets"))
        object.__setattr__(self, "tactile_channels", channels)
        object.__setattr__(self, "synchronization", MappingProxyType(dict(self.synchronization)))
        if self.object_state is not None:
            object.__setattr__(self, "object_state", MappingProxyType(dict(self.object_state)))
        if self.grasp_state is not None:
            object.__setattr__(self, "grasp_state", MappingProxyType(dict(self.grasp_state)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "timestamp": self.timestamp,
            "source": self.source,
            "joint_positions": dict(self.joint_positions),
            "joint_velocities": dict(self.joint_velocities),
            "joint_targets": dict(self.joint_targets),
            "tactile_channels": [item.to_dict() for item in self.tactile_channels],
            "camera": self.camera.to_dict() if self.camera else None,
            "object_state": _plain(self.object_state),
            "grasp_state": _plain(self.grasp_state),
            "synchronization": _plain(self.synchronization),
            "link_poses": [pose.to_dict() for pose in self.link_poses],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CommonRobotState":
        raw = dict(value)
        raw["tactile_channels"] = tuple(
            TactileChannelSample.from_dict(item) for item in raw["tactile_channels"]
        )
        raw["link_poses"] = tuple(LinkPose(**pose) for pose in raw.get("link_poses", ()))
        if raw.get("camera") is not None:
            raw["camera"] = CameraObservation.from_dict(raw["camera"])
        return cls(**raw)


@dataclass(frozen=True)
class ProviderDescription:
    source: str
    source_display_name: str
    handedness: str
    kinematic_model_path: str
    sensor_config_path: str
    calibration_config: Mapping[str, Mapping[str, float]]
    joints: tuple[JointInfo, ...]

    def __post_init__(self) -> None:
        if not all((self.source, self.source_display_name, self.handedness, self.kinematic_model_path, self.sensor_config_path)):
            raise ValueError("Provider description fields are required")
        calibration = {
            str(sensor_id): MappingProxyType({str(key): float(value) for key, value in values.items()})
            for sensor_id, values in self.calibration_config.items()
        }
        object.__setattr__(self, "calibration_config", MappingProxyType(calibration))
        object.__setattr__(self, "joints", tuple(self.joints))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_display_name": self.source_display_name,
            "handedness": self.handedness,
            "kinematic_model_path": self.kinematic_model_path,
            "sensor_config_path": self.sensor_config_path,
            "calibration_config": _plain(self.calibration_config),
            "joints": [joint.to_dict() for joint in self.joints],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProviderDescription":
        raw = dict(value)
        raw["joints"] = tuple(JointInfo.from_dict(item) for item in raw["joints"])
        return cls(**raw)


class RobotCommandName(str, Enum):
    OPEN_HAND = "open_hand"
    GRASP = "grasp"
    RESET = "reset"
    SET_JOINT_TARGET = "set_joint_target"
    SET_GRASP_PRESET = "set_grasp_preset"
    RESET_REFERENCE_POSE = "reset_reference_pose"
    EXECUTE_GRASP_STAGE = "execute_grasp_stage"
    SAVE_GRASP_PRESET = "save_grasp_preset"
    LOAD_GRASP_PRESET = "load_grasp_preset"
    SET_EXPERIMENT_MODE = "set_experiment_mode"
    SET_TEACH_GRASP_MODE = "set_teach_grasp_mode"
    SET_POSE_VALIDATION_MODE = "set_pose_validation_mode"
    SAVE_TAUGHT_POSE = "save_taught_pose"
    LOAD_TAUGHT_POSE = "load_taught_pose"
    EXECUTE_TAUGHT_GRASP = "execute_taught_grasp"
    STOP_TAUGHT_GRASP = "stop_taught_grasp"


@dataclass(frozen=True)
class RobotCommand:
    name: RobotCommandName
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", RobotCommandName(self.name))
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
