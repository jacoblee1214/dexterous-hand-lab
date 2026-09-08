"""MuJoCo-contact adapter exposing only calibrated scalar sensor channels.

Contact positions/normals are used privately to assign a physical contact to a
finite sensor region. They are intentionally absent from ScalarSensorReading so
downstream reconstruction code cannot consume MuJoCo ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml

from sensors.sensor_kinematics import SensorMount


@dataclass(frozen=True)
class SensorSignalConfig:
    effective_sensor_area_m2: float
    virtual_stiffness_n_per_m: float
    minimum_activation_force_n: float
    activation_release_force_n: float
    activation_hold_seconds: float
    saturation_force_n: float
    saturation_pressure_pa: float
    noise_standard_deviation_n: float
    region_tolerance_m: float
    contact_projection_sign: float


@dataclass(frozen=True)
class ScalarSensorReading:
    sensor_id: str
    parent_link: str
    sensor_type: str
    surface_area_m2: float
    normal_contact_force_n: float
    scalar_output_n: float
    pressure_pa: float
    indentation_m: float

    @property
    def active(self) -> bool:
        return self.scalar_output_n > 0.0


def _automatic_area(mount: SensorMount) -> float:
    if mount.sensor_type == "rectangular":
        return mount.width * mount.height
    return np.pi * 0.25 * mount.width * mount.height


def load_pressure_config(
    path: str | Path,
    mounts: Mapping[str, SensorMount],
) -> dict[str, SensorSignalConfig]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError("Pressure config must use schema_version 1")
    defaults = raw["defaults"]
    overrides = raw.get("sensors", {})
    unknown = set(overrides) - set(mounts)
    if unknown:
        raise ValueError(f"Pressure config contains unknown sensors: {sorted(unknown)}")
    result = {}
    for sensor_id, mount in mounts.items():
        values = {**defaults, **overrides.get(sensor_id, {})}
        area = values["effective_sensor_area_m2"]
        if area == "auto" or area is None:
            area = _automatic_area(mount)
        config = SensorSignalConfig(
            effective_sensor_area_m2=float(area),
            virtual_stiffness_n_per_m=float(values["virtual_stiffness_n_per_m"]),
            minimum_activation_force_n=float(values["minimum_activation_force_n"]),
            activation_release_force_n=float(values["activation_release_force_n"]),
            activation_hold_seconds=float(values["activation_hold_seconds"]),
            saturation_force_n=float(values["saturation_force_n"]),
            saturation_pressure_pa=float(values["saturation_pressure_pa"]),
            noise_standard_deviation_n=float(values["noise_standard_deviation_n"]),
            region_tolerance_m=float(values["region_tolerance_m"]),
            contact_projection_sign=float(values["contact_projection_sign"]),
        )
        if min(
            config.effective_sensor_area_m2,
            config.virtual_stiffness_n_per_m,
            config.saturation_force_n,
            config.saturation_pressure_pa,
            config.region_tolerance_m,
        ) <= 0:
            raise ValueError(f"Sensor {sensor_id} has a non-positive pressure parameter")
        if (
            config.minimum_activation_force_n < 0
            or config.activation_release_force_n < 0
            or config.activation_hold_seconds < 0
            or config.noise_standard_deviation_n < 0
        ):
            raise ValueError(f"Sensor {sensor_id} has a negative pressure parameter")
        if config.activation_release_force_n > config.minimum_activation_force_n:
            raise ValueError(
                f"Sensor {sensor_id} release threshold exceeds activation threshold"
            )
        if config.contact_projection_sign not in (-1.0, 1.0):
            raise ValueError(f"Sensor {sensor_id} contact projection sign must be +1 or -1")
        result[sensor_id] = config
    return result


class PressureSensorEmulator:
    """Convert MuJoCo contacts into finite-region scalar pressure readings."""

    def __init__(
        self,
        simulation,
        mounts: Mapping[str, SensorMount],
        config_path: str | Path,
        *,
        random_seed: int = 0,
    ) -> None:
        self.simulation = simulation
        self.mounts = dict(mounts)
        self.configs = load_pressure_config(config_path, self.mounts)
        self._rng = np.random.default_rng(random_seed)
        self._stable_output_force = {sensor_id: 0.0 for sensor_id in self.mounts}
        self._last_activation_time = {
            sensor_id: -np.inf for sensor_id in self.mounts
        }
        mujoco = simulation.mujoco
        self._body_ids = {
            sensor_id: mujoco.mj_name2id(
                simulation.model, mujoco.mjtObj.mjOBJ_BODY, mount.parent_link
            )
            for sensor_id, mount in self.mounts.items()
        }

    @staticmethod
    def point_is_in_region(
        mount: SensorMount,
        point_in_link: np.ndarray,
        tolerance_m: float,
    ) -> bool:
        delta = np.asarray(point_in_link, dtype=float) - mount.center_xyz
        u = float(np.dot(delta, mount.surface_u_axis))
        v = float(np.dot(delta, mount.surface_v_axis))
        n = float(np.dot(delta, mount.outward_normal))
        if mount.sensor_type == "rectangular":
            return (
                abs(u) <= mount.width / 2 + tolerance_m
                and abs(v) <= mount.height / 2 + tolerance_m
                and abs(n) <= mount.surface_thickness_m / 2 + tolerance_m
            )
        radii = mount.fingertip_radii_xyz
        # center_xyz is the outermost point of the outward half-ellipsoid.
        radial = (u / (radii[0] + tolerance_m)) ** 2 + (
            v / (radii[1] + tolerance_m)
        ) ** 2
        return radial <= 1.0 and -2 * radii[2] - tolerance_m <= n <= tolerance_m

    def _empty_reading(self, sensor_id: str) -> ScalarSensorReading:
        mount, config = self.mounts[sensor_id], self.configs[sensor_id]
        return ScalarSensorReading(
            sensor_id,
            mount.parent_link,
            mount.sensor_type,
            config.effective_sensor_area_m2,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    def _stabilize_force(
        self,
        sensor_id: str,
        measured_force_n: float,
        timestamp: float,
    ) -> float:
        """Apply per-channel hysteresis and a short dropout hold."""
        config = self.configs[sensor_id]
        measured = max(0.0, float(measured_force_n))
        previous = self._stable_output_force[sensor_id]
        if timestamp < self._last_activation_time[sensor_id]:
            previous = 0.0
            self._stable_output_force[sensor_id] = 0.0
            self._last_activation_time[sensor_id] = -np.inf
        if measured >= config.minimum_activation_force_n:
            self._last_activation_time[sensor_id] = float(timestamp)
            stable = measured
        elif previous > 0.0 and measured >= config.activation_release_force_n:
            self._last_activation_time[sensor_id] = float(timestamp)
            stable = measured
        elif previous > 0.0 and (
            timestamp - self._last_activation_time[sensor_id]
            <= config.activation_hold_seconds
        ):
            # Preserve a valid last sample during brief solver contact dropouts.
            stable = previous
        else:
            stable = 0.0
        stable = min(stable, config.saturation_force_n)
        self._stable_output_force[sensor_id] = stable
        return stable

    def read(self) -> dict[str, ScalarSensorReading]:
        """Return scalar-only channel data for the current MuJoCo state."""
        simulation = self.simulation
        mujoco, model, data = simulation.mujoco, simulation.model, simulation.data
        accumulated = {sensor_id: 0.0 for sensor_id in self.mounts}
        contact_force = np.zeros(6)
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            body1 = int(model.geom_bodyid[contact.geom1])
            body2 = int(model.geom_bodyid[contact.geom2])
            mujoco.mj_contactForce(model, data, contact_index, contact_force)
            magnitude = max(0.0, float(contact_force[0]))
            if magnitude == 0.0:
                continue
            contact_normal_world = np.asarray(contact.frame[:3], dtype=float)
            for sensor_id, mount in self.mounts.items():
                body_id = self._body_ids[sensor_id]
                if body_id not in (body1, body2):
                    continue
                rotation = data.xmat[body_id].reshape(3, 3)
                point_local = rotation.T @ (contact.pos - data.xpos[body_id])
                config = self.configs[sensor_id]
                if not self.point_is_in_region(mount, point_local, config.region_tolerance_m):
                    continue
                normal_world = rotation @ mount.outward_normal
                projected = magnitude * abs(float(np.dot(contact_normal_world, normal_world)))
                accumulated[sensor_id] += projected

        readings = {}
        for sensor_id, raw_force in accumulated.items():
            mount, config = self.mounts[sensor_id], self.configs[sensor_id]
            noisy_force = raw_force
            if config.noise_standard_deviation_n > 0:
                noisy_force += float(
                    self._rng.normal(0.0, config.noise_standard_deviation_n)
                )
            output_force = self._stabilize_force(
                sensor_id, noisy_force, float(data.time)
            )
            pressure = min(
                output_force / config.effective_sensor_area_m2,
                config.saturation_pressure_pa,
            )
            readings[sensor_id] = ScalarSensorReading(
                sensor_id=sensor_id,
                parent_link=mount.parent_link,
                sensor_type=mount.sensor_type,
                surface_area_m2=config.effective_sensor_area_m2,
                normal_contact_force_n=raw_force,
                scalar_output_n=output_force,
                pressure_pa=pressure,
                indentation_m=output_force / config.virtual_stiffness_n_per_m,
            )
        return readings
