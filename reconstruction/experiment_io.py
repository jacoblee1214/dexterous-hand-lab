"""Structured persistence for ground-truth-free tactile experiment data."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from reconstruction.contact_point_buffer import (
    PointBufferConfig,
    PointBufferStatistics,
)
from reconstruction.temporal_observation import TemporalTactileObservation
from reconstruction.sphere_fitting import SphereReconstructionResult


def save_reconstruction_dataset(
    path: str | Path,
    observations: tuple[TemporalTactileObservation, ...],
    config: PointBufferConfig,
    statistics: PointBufferStatistics,
    sphere_reconstruction: SphereReconstructionResult | None = None,
) -> Path:
    """Save only reconstruction-side values; simulator GT is not accepted."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "dataset_type": "temporal_tactile_point_cloud",
        "hand": "right",
        "coordinate_frame": "right_hand_base_world",
        "point_buffer_config": asdict(config),
        "statistics": asdict(statistics),
        "observations": [
            {
                "timestamp": point.timestamp,
                "sensor_id": point.sensor_id,
                "finger_id": point.finger_id,
                "parent_link": point.parent_link,
                "joint_state_snapshot": dict(point.joint_state_snapshot),
                "sensor_position": list(point.sensor_position),
                "sensor_sensing_direction": list(
                    point.sensor_sensing_direction
                ),
                "scalar_sensor_value": point.scalar_sensor_value,
                "estimated_pressure": point.estimated_pressure,
                "estimated_indentation": point.estimated_indentation,
                "estimated_contact_position": list(
                    point.estimated_contact_position
                ),
            }
            for point in observations
        ],
    }
    if sphere_reconstruction is not None:
        sphere_payload = asdict(sphere_reconstruction)
        sphere_payload["status"] = sphere_reconstruction.status.value
        payload["sphere_reconstruction"] = sphere_payload
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output
