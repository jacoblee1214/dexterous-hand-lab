"""Ground-truth-free temporal tactile observation records."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

from reconstruction.estimated_contact import EstimatedContactPoint, Vector3


@dataclass(frozen=True)
class TemporalTactileObservation:
    timestamp: float
    sensor_id: str
    finger_id: str
    parent_link: str
    joint_state_snapshot: Mapping[str, float]
    sensor_position: Vector3
    sensor_sensing_direction: Vector3
    scalar_sensor_value: float
    estimated_pressure: float
    estimated_indentation: float
    estimated_contact_position: Vector3

    def __post_init__(self) -> None:
        scalar_values = (
            self.timestamp,
            self.scalar_sensor_value,
            self.estimated_pressure,
            self.estimated_indentation,
        )
        if any(not math.isfinite(float(value)) for value in scalar_values):
            raise ValueError("Temporal observation scalars must be finite")
        if self.timestamp < 0 or min(scalar_values[1:]) < 0:
            raise ValueError("Temporal observation time and sensor values cannot be negative")
        for name, vector in (
            ("sensor_position", self.sensor_position),
            ("sensor_sensing_direction", self.sensor_sensing_direction),
            ("estimated_contact_position", self.estimated_contact_position),
        ):
            if len(vector) != 3 or any(
                not math.isfinite(float(value)) for value in vector
            ):
                raise ValueError(f"{name} must be a finite three-vector")
            object.__setattr__(self, name, tuple(float(value) for value in vector))
        if (
            math.sqrt(
                sum(value * value for value in self.sensor_sensing_direction)
            )
            == 0
        ):
            raise ValueError("sensor_sensing_direction must be non-zero")
        if not self.sensor_id or not self.finger_id or not self.parent_link:
            raise ValueError("Temporal observation identifiers are required")
        joint_state = {
            str(name): float(value)
            for name, value in self.joint_state_snapshot.items()
        }
        if any(not math.isfinite(value) for value in joint_state.values()):
            raise ValueError("joint_state_snapshot must contain finite values")
        object.__setattr__(
            self, "joint_state_snapshot", MappingProxyType(joint_state)
        )
        object.__setattr__(self, "timestamp", float(self.timestamp))
        object.__setattr__(
            self, "scalar_sensor_value", float(self.scalar_sensor_value)
        )
        object.__setattr__(self, "estimated_pressure", float(self.estimated_pressure))
        object.__setattr__(
            self, "estimated_indentation", float(self.estimated_indentation)
        )

    @classmethod
    def from_estimate(
        cls,
        estimate: EstimatedContactPoint,
        joint_state_snapshot: Mapping[str, float],
    ) -> "TemporalTactileObservation":
        return cls(
            timestamp=estimate.timestamp,
            sensor_id=estimate.sensor_id,
            finger_id=estimate.finger_id,
            parent_link=estimate.parent_link,
            joint_state_snapshot=joint_state_snapshot,
            sensor_position=estimate.sensor_position,
            sensor_sensing_direction=estimate.sensor_sensing_direction,
            scalar_sensor_value=estimate.scalar_signal,
            estimated_pressure=estimate.estimated_pressure,
            estimated_indentation=estimate.estimated_indentation,
            estimated_contact_position=estimate.estimated_contact_position,
        )


def make_temporal_observations(
    estimates: list[EstimatedContactPoint],
    joint_state_snapshot: Mapping[str, float],
) -> list[TemporalTactileObservation]:
    return [
        TemporalTactileObservation.from_estimate(estimate, joint_state_snapshot)
        for estimate in estimates
    ]
