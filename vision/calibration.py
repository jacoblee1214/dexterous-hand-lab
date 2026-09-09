"""Numerical camera calibration derived from the actual MuJoCo camera."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
import yaml


DEFAULT_CALIBRATION = Path(__file__).with_name("camera_calibration.yaml")
MUJOCO_TO_CV = np.diag([1.0, -1.0, -1.0])


@dataclass(frozen=True)
class CameraConfig:
    calibration_version: str
    camera_id: str
    camera_name: str
    frame_id: str
    parent_frame: str
    time_base: str
    width: int
    height: int
    fovy_degrees: float
    distortion_model: str
    distortion_coefficients: tuple[float, ...]
    position_world_m: tuple[float, float, float]
    xyaxes_world: tuple[float, ...]
    segmentation: Mapping[str, float]


@dataclass(frozen=True)
class CalibratedCamera:
    config: CameraConfig
    intrinsic_matrix: np.ndarray
    world_from_camera_cv: np.ndarray
    camera_cv_from_world: np.ndarray

    @property
    def intrinsics(self) -> dict:
        k = self.intrinsic_matrix
        return {
            "width": self.config.width,
            "height": self.config.height,
            "fx": float(k[0, 0]),
            "fy": float(k[1, 1]),
            "cx": float(k[0, 2]),
            "cy": float(k[1, 2]),
            "distortion_model": self.config.distortion_model,
            "distortion_coefficients": list(self.config.distortion_coefficients),
        }

    def project_world(self, points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points_world, dtype=float)
        if points.ndim == 1:
            points = points[None, :]
        homogeneous = np.column_stack((points, np.ones(len(points))))
        camera = (self.camera_cv_from_world @ homogeneous.T).T[:, :3]
        depth = camera[:, 2]
        pixels = np.full((len(points), 2), np.nan, dtype=float)
        valid = depth > 1e-9
        pixels[valid, 0] = (
            self.intrinsic_matrix[0, 0] * camera[valid, 0] / depth[valid]
            + self.intrinsic_matrix[0, 2]
        )
        pixels[valid, 1] = (
            self.intrinsic_matrix[1, 1] * camera[valid, 1] / depth[valid]
            + self.intrinsic_matrix[1, 2]
        )
        return pixels, depth

    def pixel_rays_cv(self, pixels: np.ndarray, *, normalize: bool = True) -> np.ndarray:
        pixels = np.asarray(pixels, dtype=float)
        rays = np.column_stack(
            (
                (pixels[:, 0] - self.intrinsic_matrix[0, 2]) / self.intrinsic_matrix[0, 0],
                (pixels[:, 1] - self.intrinsic_matrix[1, 2]) / self.intrinsic_matrix[1, 1],
                np.ones(len(pixels)),
            )
        )
        if normalize:
            rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        return rays

    def camera_to_world(self, points_camera_cv: np.ndarray) -> np.ndarray:
        points = np.asarray(points_camera_cv, dtype=float)
        one = points.ndim == 1
        if one:
            points = points[None, :]
        homogeneous = np.column_stack((points, np.ones(len(points))))
        result = (self.world_from_camera_cv @ homogeneous.T).T[:, :3]
        return result[0] if one else result

    def to_dict(self) -> dict:
        return {
            "calibration_version": self.config.calibration_version,
            "camera_id": self.config.camera_id,
            "camera_name": self.config.camera_name,
            "frame_id": self.config.frame_id,
            "parent_frame": self.config.parent_frame,
            "time_base": self.config.time_base,
            "fovy_degrees": self.config.fovy_degrees,
            "intrinsics": self.intrinsics,
            "T_world_from_camera_cv": self.world_from_camera_cv.tolist(),
            "T_camera_cv_from_world": self.camera_cv_from_world.tolist(),
            "coordinate_convention": {
                "mujoco_camera": "+X right, +Y up, -Z forward",
                "computer_vision_camera": "+X right, +Y down, +Z forward",
                "mujoco_to_cv_rotation": MUJOCO_TO_CV.tolist(),
            },
        }


def load_camera_config(path: str | Path = DEFAULT_CALIBRATION) -> CameraConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError("Camera calibration must use schema_version 1")
    image, distortion, pose = raw["image"], raw["distortion"], raw["mujoco_pose"]
    config = CameraConfig(
        calibration_version=str(raw["calibration_version"]),
        camera_id=str(raw["camera_id"]),
        camera_name=str(raw["camera_name"]),
        frame_id=str(raw["frame_id"]),
        parent_frame=str(raw["parent_frame"]),
        time_base=str(raw["time_base"]),
        width=int(image["width"]),
        height=int(image["height"]),
        fovy_degrees=float(image["fovy_degrees"]),
        distortion_model=str(distortion["model"]),
        distortion_coefficients=tuple(float(v) for v in distortion["coefficients"]),
        position_world_m=tuple(float(v) for v in pose["position_world_m"]),
        xyaxes_world=tuple(float(v) for v in pose["xyaxes_world"]),
        segmentation=MappingProxyType(
            {str(key): float(value) for key, value in raw["rgb_segmentation"].items()}
        ),
    )
    if (
        not config.calibration_version
        or not config.camera_id
        or not config.camera_name
        or not config.frame_id
        or config.width < 64
        or config.height < 64
        or not 1 < config.fovy_degrees < 179
        or len(config.position_world_m) != 3
        or len(config.xyaxes_world) != 6
        or not np.isfinite([*config.position_world_m, *config.xyaxes_world]).all()
    ):
        raise ValueError("Invalid research camera calibration")
    if config.distortion_model != "none" or any(config.distortion_coefficients):
        raise ValueError("MuJoCo pinhole rendering requires explicit zero distortion")
    return config


def calibration_from_mujoco(model, data, config: CameraConfig) -> CalibratedCamera:
    camera_id = model.camera(config.camera_name).id
    actual_fovy = float(model.cam_fovy[camera_id])
    if not math.isclose(actual_fovy, config.fovy_degrees, abs_tol=1e-9):
        raise ValueError("Configured and compiled MuJoCo camera fovy differ")
    fy = 0.5 * config.height / math.tan(math.radians(actual_fovy) / 2.0)
    fx = fy
    cx, cy = (config.width - 1.0) / 2.0, (config.height - 1.0) / 2.0
    intrinsic = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    rotation_world_from_mujoco = np.asarray(data.cam_xmat[camera_id]).reshape(3, 3)
    rotation_world_from_cv = rotation_world_from_mujoco @ MUJOCO_TO_CV
    world_from_cv = np.eye(4)
    world_from_cv[:3, :3] = rotation_world_from_cv
    world_from_cv[:3, 3] = data.cam_xpos[camera_id]
    camera_from_world = np.linalg.inv(world_from_cv)
    return CalibratedCamera(config, intrinsic, world_from_cv, camera_from_world)
