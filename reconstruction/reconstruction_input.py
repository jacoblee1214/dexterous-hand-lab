"""Strict, ground-truth-free input boundary for tactile reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping


CALIBRATION_FIELDS = frozenset(
    {
        "effective_sensor_area_m2",
        "virtual_stiffness_n_per_m",
        "minimum_activation_force_n",
        "saturation_pressure_pa",
        "contact_projection_sign",
    }
)


def _finite_float_map(name: str, values: Mapping[str, float]) -> Mapping[str, float]:
    copied = {str(key): float(value) for key, value in values.items()}
    if any(not math.isfinite(value) for value in copied.values()):
        raise ValueError(f"{name} must contain only finite scalar values")
    return MappingProxyType(copied)


@dataclass(frozen=True)
class ReconstructionInput:
    """The complete runtime information visible to reconstruction code."""

    timestamp: float
    joint_state: Mapping[str, float]
    sensor_values: Mapping[str, float]
    sensor_config_path: str
    calibration_config: Mapping[str, Mapping[str, float]]

    def __post_init__(self) -> None:
        timestamp = float(self.timestamp)
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("timestamp must be a finite non-negative value")
        if not self.sensor_config_path:
            raise ValueError("sensor_config_path is required")
        frozen_calibration = {}
        for sensor_id, raw in self.calibration_config.items():
            unknown = set(raw) - CALIBRATION_FIELDS
            if unknown:
                raise ValueError(
                    f"Calibration for {sensor_id} contains forbidden/unknown keys: "
                    f"{sorted(unknown)}"
                )
            missing = CALIBRATION_FIELDS - set(raw)
            if missing:
                raise ValueError(
                    f"Calibration for {sensor_id} is missing keys: {sorted(missing)}"
                )
            values = _finite_float_map(f"calibration[{sensor_id}]", raw)
            if values["effective_sensor_area_m2"] <= 0:
                raise ValueError(f"{sensor_id} effective area must be positive")
            if values["virtual_stiffness_n_per_m"] <= 0:
                raise ValueError(f"{sensor_id} virtual stiffness must be positive")
            if values["minimum_activation_force_n"] < 0:
                raise ValueError(f"{sensor_id} activation threshold cannot be negative")
            if values["saturation_pressure_pa"] <= 0:
                raise ValueError(f"{sensor_id} pressure saturation must be positive")
            if values["contact_projection_sign"] not in (-1.0, 1.0):
                raise ValueError(f"{sensor_id} contact projection sign must be +1 or -1")
            frozen_calibration[str(sensor_id)] = values
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "joint_state", _finite_float_map("joint_state", self.joint_state))
        object.__setattr__(self, "sensor_values", _finite_float_map("sensor_values", self.sensor_values))
        object.__setattr__(
            self, "calibration_config", MappingProxyType(frozen_calibration)
        )
