"""Ground-truth-independent forward kinematics for mounted sensor frames.

This module intentionally consumes only the reference URDF joint graph, joint
positions, and fixed sensor mounting transforms. It has no MuJoCo contact API.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import xml.etree.ElementTree as ET

import numpy as np
import yaml


def rpy_to_matrix(rpy: np.ndarray | list[float]) -> np.ndarray:
    """Return the URDF fixed-axis roll-pitch-yaw rotation matrix."""
    roll, pitch, yaw = np.asarray(rpy, dtype=float)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def axis_angle_to_matrix(axis: np.ndarray | list[float], angle: float) -> np.ndarray:
    """Return a rotation matrix for a normalized axis and angle."""
    axis_array = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis_array)
    if norm == 0:
        raise ValueError("Rotation axis must be non-zero")
    x, y, z = axis_array / norm
    c, s, one_minus_c = np.cos(angle), np.sin(angle), 1.0 - np.cos(angle)
    return np.array(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ],
        dtype=float,
    )


def make_transform(position_xyz: np.ndarray | list[float], rotation: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=float)
    result[:3, :3] = rotation
    result[:3, 3] = np.asarray(position_xyz, dtype=float)
    return result


@dataclass(frozen=True)
class SensorMount:
    sensor_id: str
    finger_id: str
    parent_link: str
    sensor_type: str
    center_xyz: np.ndarray
    surface_u_axis: np.ndarray
    surface_v_axis: np.ndarray
    width: float
    height: float
    outward_normal: np.ndarray
    surface_thickness_m: float
    fingertip_radii_xyz: np.ndarray | None

    @property
    def link_to_sensor(self) -> np.ndarray:
        return make_transform(self.center_xyz, self.rotation_matrix)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Sensor basis in parent-link coordinates: columns are U, V, normal."""
        return np.column_stack(
            (self.surface_u_axis, self.surface_v_axis, self.outward_normal)
        )

    # Compatibility names used by FK/rendering code. The persisted schema is the
    # explicit center/U/V/normal representation above; no RPY inference is used.
    @property
    def position_xyz(self) -> np.ndarray:
        return self.center_xyz

    @property
    def local_sensing_axis(self) -> np.ndarray:
        return np.array([0.0, 0.0, 1.0])

    @property
    def surface_width_m(self) -> float:
        return self.width

    @property
    def surface_height_m(self) -> float:
        return self.height

    def surface_point_local(self, u: float, v: float) -> np.ndarray:
        """Map normalized surface coordinates to a link-local sensor point.

        Rectangular coordinates span [-1, 1] on the finite pad. Fingertips use
        the outward half of an ellipsoid, with (0, 0) at the representative
        outermost center. This keeps future fingertip contact location variable
        instead of collapsing the region to one point.
        """
        if abs(u) > 1 or abs(v) > 1:
            raise ValueError("Surface coordinates u and v must be within [-1, 1]")
        sensor_point = np.zeros(3, dtype=float)
        if self.sensor_type == "rectangular":
            sensor_point[:2] = [u * self.width / 2, v * self.height / 2]
        else:
            if u * u + v * v > 1:
                raise ValueError("Fingertip coordinates must lie within the unit disk")
            radii = self.fingertip_radii_xyz
            sensor_point = np.array(
                [u * radii[0], v * radii[1], radii[2] * (np.sqrt(1 - u * u - v * v) - 1)]
            )
        return self.center_xyz + self.rotation_matrix @ sensor_point


@dataclass(frozen=True)
class SensorPose:
    sensor_id: str
    parent_link: str
    transform: np.ndarray
    position: np.ndarray
    sensing_direction: np.ndarray


@dataclass(frozen=True)
class JointDefinition:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray


def load_sensor_mounts(path: str | Path) -> dict[str, SensorMount]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != 3:
        raise ValueError(f"Sensor config {path} must use explicit surface schema version 3")
    mounts: dict[str, SensorMount] = {}
    for item in raw["sensors"]:
        mount = SensorMount(
            sensor_id=item["sensor_id"],
            finger_id=item["finger_id"],
            parent_link=item["parent_link"],
            sensor_type=item["sensor_type"],
            center_xyz=np.asarray(item["center_xyz"], dtype=float),
            surface_u_axis=np.asarray(item["surface_u_axis"], dtype=float),
            surface_v_axis=np.asarray(item["surface_v_axis"], dtype=float),
            width=float(item["width"]),
            height=float(item["height"]),
            outward_normal=np.asarray(item["outward_normal"], dtype=float),
            surface_thickness_m=float(item["surface_thickness_m"]),
            fingertip_radii_xyz=(
                np.asarray(item["fingertip_radii_xyz"], dtype=float)
                if "fingertip_radii_xyz" in item
                else None
            ),
        )
        if mount.sensor_id in mounts:
            raise ValueError(f"Duplicate sensor_id: {mount.sensor_id}")
        vectors = (mount.center_xyz, mount.surface_u_axis, mount.surface_v_axis, mount.outward_normal)
        if any(vector.shape != (3,) for vector in vectors):
            raise ValueError(f"Sensor {mount.sensor_id} requires three-element surface vectors")
        if not np.isclose(np.linalg.norm(mount.surface_u_axis), 1.0, atol=1e-6):
            raise ValueError(f"Sensor {mount.sensor_id} U axis must be unit length")
        if not np.isclose(np.linalg.norm(mount.surface_v_axis), 1.0, atol=1e-6):
            raise ValueError(f"Sensor {mount.sensor_id} V axis must be unit length")
        if not np.isclose(np.dot(mount.surface_u_axis, mount.surface_v_axis), 0.0, atol=1e-6):
            raise ValueError(f"Sensor {mount.sensor_id} U and V axes must be orthogonal")
        computed_normal = np.cross(mount.surface_u_axis, mount.surface_v_axis)
        computed_normal /= np.linalg.norm(computed_normal)
        if not np.allclose(computed_normal, mount.outward_normal, atol=1e-6):
            raise ValueError(
                f"Sensor {mount.sensor_id} outward_normal must equal normalize(cross(U, V))"
            )
        if mount.sensor_type not in {"rectangular", "fingertip_surface"}:
            raise ValueError(f"Sensor {mount.sensor_id} has invalid sensor_type")
        if min(mount.width, mount.height, mount.surface_thickness_m) <= 0:
            raise ValueError(f"Sensor {mount.sensor_id} surface dimensions must be positive")
        if mount.sensor_type == "fingertip_surface":
            if mount.fingertip_radii_xyz is None or mount.fingertip_radii_xyz.shape != (3,):
                raise ValueError(f"Fingertip {mount.sensor_id} requires three radii")
        mounts[mount.sensor_id] = mount
    return mounts


def save_sensor_mounts(path: str | Path, mounts: Mapping[str, SensorMount]) -> None:
    """Persist explicit, manually calibrated link-local surface definitions."""
    config_path = Path(path)
    existing = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    def values(vector: np.ndarray) -> list[float]:
        return [round(float(value), 9) for value in vector]

    records = []
    for mount in mounts.values():
        record = {
            "sensor_id": mount.sensor_id,
            "finger_id": mount.finger_id,
            "parent_link": mount.parent_link,
            "sensor_type": mount.sensor_type,
            "center_xyz": values(mount.center_xyz),
            "surface_u_axis": values(mount.surface_u_axis),
            "surface_v_axis": values(mount.surface_v_axis),
            "width": round(float(mount.width), 9),
            "height": round(float(mount.height), 9),
            "outward_normal": values(mount.outward_normal),
            "surface_thickness_m": round(float(mount.surface_thickness_m), 9),
        }
        if mount.fingertip_radii_xyz is not None:
            record["fingertip_radii_xyz"] = values(mount.fingertip_radii_xyz)
        records.append(record)
    existing["schema_version"] = 3
    existing["sensors"] = records
    config_path.write_text(
        yaml.safe_dump(existing, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


class KinematicTree:
    """Small URDF FK implementation used to spatialize scalar measurements."""

    def __init__(self, base_link: str, joints: list[JointDefinition]):
        self.base_link = base_link
        self.joints = joints
        self._children: dict[str, list[JointDefinition]] = {}
        for joint in joints:
            self._children.setdefault(joint.parent, []).append(joint)

    @classmethod
    def from_urdf(cls, path: str | Path) -> "KinematicTree":
        root = ET.parse(path).getroot()
        child_links: set[str] = set()
        joints: list[JointDefinition] = []
        for element in root.findall("joint"):
            if element.attrib["type"] not in {"revolute", "continuous", "fixed"}:
                raise ValueError(f"Unsupported joint type: {element.attrib['type']}")
            origin = element.find("origin")
            axis = element.find("axis")
            parent = element.find("parent").attrib["link"]
            child = element.find("child").attrib["link"]
            child_links.add(child)
            joints.append(
                JointDefinition(
                    name=element.attrib["name"],
                    joint_type=element.attrib["type"],
                    parent=parent,
                    child=child,
                    origin_xyz=_vector(origin.attrib.get("xyz", "0 0 0")),
                    origin_rpy=_vector(origin.attrib.get("rpy", "0 0 0")),
                    axis=_vector(axis.attrib.get("xyz", "1 0 0")) if axis is not None and element.attrib["type"] != "fixed" else np.array([1.0, 0.0, 0.0]),
                )
            )
        links = {element.attrib["name"] for element in root.findall("link")}
        base_links = links - child_links
        if len(base_links) != 1:
            raise ValueError(f"Expected one URDF base link, found {sorted(base_links)}")
        return cls(base_links.pop(), joints)

    def forward_kinematics(self, joint_positions: Mapping[str, float] | None = None) -> dict[str, np.ndarray]:
        joint_positions = joint_positions or {}
        transforms = {self.base_link: np.eye(4, dtype=float)}

        def visit(parent: str) -> None:
            for joint in self._children.get(parent, []):
                origin = make_transform(joint.origin_xyz, rpy_to_matrix(joint.origin_rpy))
                angle = 0.0 if joint.joint_type == "fixed" else float(joint_positions.get(joint.name, 0.0))
                motion = make_transform([0.0, 0.0, 0.0], axis_angle_to_matrix(joint.axis, angle))
                transforms[joint.child] = transforms[parent] @ origin @ motion
                visit(joint.child)

        visit(self.base_link)
        if len(transforms) != len(self.joints) + 1:
            raise ValueError("URDF joint graph is disconnected or cyclic")
        return transforms

    def sensor_poses(
        self,
        mounts: Mapping[str, SensorMount],
        joint_positions: Mapping[str, float] | None = None,
    ) -> dict[str, SensorPose]:
        links = self.forward_kinematics(joint_positions)
        poses: dict[str, SensorPose] = {}
        for sensor_id, mount in mounts.items():
            if mount.parent_link not in links:
                raise KeyError(f"Unknown parent link for {sensor_id}: {mount.parent_link}")
            transform = links[mount.parent_link] @ mount.link_to_sensor
            direction = transform[:3, :3] @ mount.local_sensing_axis
            direction = direction / np.linalg.norm(direction)
            poses[sensor_id] = SensorPose(
                sensor_id=sensor_id,
                parent_link=mount.parent_link,
                transform=transform,
                position=transform[:3, 3].copy(),
                sensing_direction=direction,
            )
        return poses


def _vector(value: str) -> np.ndarray:
    return np.fromstring(value, sep=" ", dtype=float)
