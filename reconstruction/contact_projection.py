"""Kinematics-aware projection from scalar tactile channels to 3D points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol

import numpy as np

from reconstruction.estimated_contact import EstimatedContactPoint
from reconstruction.reconstruction_input import ReconstructionInput
from sensors.sensor_kinematics import KinematicTree, SensorMount, load_sensor_mounts


def project_contact_point(
    sensor_position,
    sensor_sensing_direction,
    indentation_m: float,
    sign: float = 1.0,
) -> np.ndarray:
    """Return p_hat = p_sensor + sign * indentation * normalized(direction)."""
    if sign not in (-1.0, 1.0):
        raise ValueError("Projection sign must be +1 or -1")
    position = np.asarray(sensor_position, dtype=float)
    direction = np.asarray(sensor_sensing_direction, dtype=float)
    if position.shape != (3,) or direction.shape != (3,):
        raise ValueError("Sensor position and direction must be three-vectors")
    if not np.isfinite(position).all() or not np.isfinite(direction).all():
        raise ValueError("Sensor position and direction must be finite")
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("Sensor sensing direction must be non-zero")
    indentation = float(indentation_m)
    if not np.isfinite(indentation) or indentation < 0:
        raise ValueError("Estimated indentation must be finite and non-negative")
    return position + float(sign) * indentation * direction / norm


class ScalarCalibrationModel(Protocol):
    def estimate(self, scalar_signal: float) -> tuple[float, float]:
        """Return (estimated pressure Pa, estimated indentation m)."""


@dataclass(frozen=True)
class SyntheticScalarCalibration:
    effective_sensor_area_m2: float
    virtual_stiffness_n_per_m: float
    saturation_pressure_pa: float

    def estimate(self, scalar_signal: float) -> tuple[float, float]:
        signal = max(0.0, float(scalar_signal))
        pressure = min(
            signal / self.effective_sensor_area_m2,
            self.saturation_pressure_pa,
        )
        indentation = signal / self.virtual_stiffness_n_per_m
        return pressure, indentation


class ContactPointReconstructor:
    """Compute contacts using only joint state, fixed mounts, and scalars."""

    def __init__(
        self,
        kinematic_urdf_path: str | Path,
        calibration_model_factory: (
            Callable[[str, Mapping[str, float]], ScalarCalibrationModel] | None
        ) = None,
    ) -> None:
        self.tree = KinematicTree.from_urdf(kinematic_urdf_path)
        self._mount_cache: dict[str, Mapping[str, SensorMount]] = {}
        self._calibration_model_factory = (
            calibration_model_factory or self._synthetic_calibration_model
        )

    @staticmethod
    def _synthetic_calibration_model(
        sensor_id: str,
        parameters: Mapping[str, float],
    ) -> ScalarCalibrationModel:
        del sensor_id
        return SyntheticScalarCalibration(
            effective_sensor_area_m2=parameters["effective_sensor_area_m2"],
            virtual_stiffness_n_per_m=parameters["virtual_stiffness_n_per_m"],
            saturation_pressure_pa=parameters["saturation_pressure_pa"],
        )

    def _mounts(self, path: str) -> Mapping[str, SensorMount]:
        if path not in self._mount_cache:
            self._mount_cache[path] = load_sensor_mounts(path)
        return self._mount_cache[path]

    def reconstruct(self, frame: ReconstructionInput) -> list[EstimatedContactPoint]:
        mounts = self._mounts(frame.sensor_config_path)
        unknown_values = set(frame.sensor_values) - set(mounts)
        if unknown_values:
            raise ValueError(f"Unknown scalar sensor channels: {sorted(unknown_values)}")
        missing_calibration = set(mounts) - set(frame.calibration_config)
        if missing_calibration:
            raise ValueError(
                f"Missing reconstruction calibration: {sorted(missing_calibration)}"
            )
        poses = self.tree.sensor_poses(mounts, frame.joint_state)
        contacts = []
        for sensor_id, scalar_signal in frame.sensor_values.items():
            parameters = frame.calibration_config[sensor_id]
            if (
                scalar_signal <= 0.0
                or scalar_signal < parameters["minimum_activation_force_n"]
            ):
                continue
            mount = mounts[sensor_id]
            pose = poses[sensor_id]
            model = self._calibration_model_factory(
                sensor_id,
                parameters,
            )
            pressure, indentation = model.estimate(scalar_signal)
            if not np.isfinite(pressure) or pressure < 0:
                raise ValueError(
                    f"Calibration model returned invalid pressure for {sensor_id}"
                )
            if not np.isfinite(indentation) or indentation < 0:
                raise ValueError(
                    f"Calibration model returned invalid indentation for {sensor_id}"
                )
            estimated_position = project_contact_point(
                pose.position,
                pose.sensing_direction,
                indentation,
                parameters["contact_projection_sign"],
            )
            contacts.append(
                EstimatedContactPoint(
                    timestamp=frame.timestamp,
                    sensor_id=sensor_id,
                    finger_id=mount.finger_id,
                    parent_link=mount.parent_link,
                    sensor_position=tuple(float(value) for value in pose.position),
                    sensor_sensing_direction=tuple(
                        float(value) for value in pose.sensing_direction
                    ),
                    scalar_signal=float(scalar_signal),
                    estimated_pressure=float(pressure),
                    estimated_indentation=float(indentation),
                    estimated_contact_position=tuple(
                        float(value) for value in estimated_position
                    ),
                )
            )
        return contacts
