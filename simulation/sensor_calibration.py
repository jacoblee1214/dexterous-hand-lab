"""Manual, explicit calibration of link-local tactile sensor surfaces."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Mapping

import numpy as np

from sensors.sensor_kinematics import (
    SensorMount,
    axis_angle_to_matrix,
    save_sensor_mounts,
)


class SensorMountCalibration:
    """Mutable calibration session backed by immutable SensorMount records."""

    def __init__(
        self,
        mounts: Mapping[str, SensorMount],
        config_path: str | Path,
    ) -> None:
        if not mounts:
            raise ValueError("Calibration requires at least one sensor")
        self.mounts = dict(mounts)
        self.config_path = Path(config_path)
        self.selected_sensor_id = next(iter(self.mounts))

    @property
    def selected(self) -> SensorMount:
        return self.mounts[self.selected_sensor_id]

    @property
    def parent_links(self) -> list[str]:
        return list(dict.fromkeys(mount.parent_link for mount in self.mounts.values()))

    def select_sensor(self, sensor_id: str) -> SensorMount:
        if sensor_id not in self.mounts:
            raise KeyError(f"Unknown sensor: {sensor_id}")
        self.selected_sensor_id = sensor_id
        return self.selected

    def select_link(self, parent_link: str) -> SensorMount:
        for mount in self.mounts.values():
            if mount.parent_link == parent_link:
                return self.select_sensor(mount.sensor_id)
        raise KeyError(f"No sensor is assigned to link: {parent_link}")

    def set_center(self, center_xyz) -> SensorMount:
        return self._replace(center_xyz=np.asarray(center_xyz, dtype=float))

    def move_center(self, delta_xyz) -> SensorMount:
        return self.set_center(self.selected.center_xyz + np.asarray(delta_xyz, dtype=float))

    def rotate(self, link_axis, angle_rad: float) -> SensorMount:
        """Rotate the whole U/V/normal basis in parent-link coordinates."""
        rotation = axis_angle_to_matrix(link_axis, angle_rad)
        mount = self.selected
        return self._replace(
            surface_u_axis=rotation @ mount.surface_u_axis,
            surface_v_axis=rotation @ mount.surface_v_axis,
            outward_normal=rotation @ mount.outward_normal,
        )

    def set_size(self, width: float, height: float) -> SensorMount:
        if width <= 0 or height <= 0:
            raise ValueError("Sensor width and height must be positive")
        return self._replace(width=float(width), height=float(height))

    def flip_normal(self) -> SensorMount:
        """Flip sensing direction while preserving the same finite surface."""
        mount = self.selected
        return self._replace(
            surface_v_axis=-mount.surface_v_axis,
            outward_normal=-mount.outward_normal,
        )

    def save(self) -> None:
        save_sensor_mounts(self.config_path, self.mounts)

    def _replace(self, **changes) -> SensorMount:
        updated = replace(self.selected, **changes)
        # Reuse the production loader invariants locally before accepting edits.
        if updated.center_xyz.shape != (3,):
            raise ValueError("Center must have three coordinates")
        basis = updated.rotation_matrix
        if not np.allclose(basis.T @ basis, np.eye(3), atol=1e-6):
            raise ValueError("Surface U/V/normal basis must remain orthonormal")
        if np.linalg.det(basis) < 0.999999:
            raise ValueError("Surface basis must remain right-handed")
        self.mounts[updated.sensor_id] = updated
        return updated
