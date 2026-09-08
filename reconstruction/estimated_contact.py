"""Ground-truth-free reconstructed tactile observation types."""

from __future__ import annotations

from dataclasses import dataclass


Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class EstimatedContactPoint:
    timestamp: float
    sensor_id: str
    finger_id: str
    parent_link: str
    sensor_position: Vector3
    sensor_sensing_direction: Vector3
    scalar_signal: float
    estimated_pressure: float
    estimated_indentation: float
    estimated_contact_position: Vector3
