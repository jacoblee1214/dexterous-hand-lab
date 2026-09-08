"""Bounded, deduplicated temporal tactile point-cloud buffer."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import yaml

from reconstruction.temporal_observation import TemporalTactileObservation


@dataclass(frozen=True)
class PointBufferConfig:
    """Units are Pa, seconds, count, and metres, respectively."""

    minimum_pressure_threshold: float = 0.0
    minimum_time_between_points: float = 0.02
    maximum_point_count: int = 10000
    minimum_spatial_distance_between_points: float = 0.0005

    def __post_init__(self) -> None:
        scalar_values = (
            self.minimum_pressure_threshold,
            self.minimum_time_between_points,
            self.minimum_spatial_distance_between_points,
        )
        if any(not math.isfinite(value) or value < 0 for value in scalar_values):
            raise ValueError("Point-buffer thresholds must be finite and non-negative")
        if self.maximum_point_count <= 0:
            raise ValueError("maximum_point_count must be positive")


@dataclass(frozen=True)
class PointBufferStatistics:
    raw_observation_count: int
    accepted_point_count: int
    rejected_duplicate_count: int
    rejected_pressure_count: int


def load_point_buffer_config(path: str | Path) -> PointBufferConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError("Point-buffer config must use schema_version 1")
    values = raw["point_buffer"]
    return PointBufferConfig(
        minimum_pressure_threshold=float(values["minimum_pressure_threshold"]),
        minimum_time_between_points=float(values["minimum_time_between_points"]),
        maximum_point_count=int(values["maximum_point_count"]),
        minimum_spatial_distance_between_points=float(
            values["minimum_spatial_distance_between_points"]
        ),
    )


class ContactPointBuffer:
    def __init__(
        self,
        config: PointBufferConfig | None = None,
        *,
        max_points: int | None = None,
    ) -> None:
        if config is not None and max_points is not None:
            raise ValueError("Pass either config or max_points, not both")
        self.config = config or PointBufferConfig(
            maximum_point_count=10000 if max_points is None else max_points
        )
        self._points: list[TemporalTactileObservation] = []
        self._accumulating = False
        self._raw_observation_count = 0
        self._rejected_duplicate_count = 0
        self._rejected_pressure_count = 0
        self._last_accepted_time: dict[str, float] = {}

    @property
    def accumulating(self) -> bool:
        return self._accumulating

    @property
    def points(self) -> tuple[TemporalTactileObservation, ...]:
        return tuple(self._points)

    @property
    def statistics(self) -> PointBufferStatistics:
        return PointBufferStatistics(
            raw_observation_count=self._raw_observation_count,
            accepted_point_count=len(self._points),
            rejected_duplicate_count=self._rejected_duplicate_count,
            rejected_pressure_count=self._rejected_pressure_count,
        )

    def start(self) -> None:
        self._accumulating = True

    def pause(self) -> None:
        self._accumulating = False

    def toggle(self) -> None:
        self._accumulating = not self._accumulating

    def _is_spatial_duplicate(self, observation: TemporalTactileObservation) -> bool:
        minimum_distance = self.config.minimum_spatial_distance_between_points
        if minimum_distance == 0:
            return False
        position = np.asarray(observation.estimated_contact_position)
        return any(
            point.sensor_id == observation.sensor_id
            and np.linalg.norm(
                position - np.asarray(point.estimated_contact_position)
            )
            < minimum_distance
            for point in self._points
        )

    def add(
        self, observations: list[TemporalTactileObservation]
    ) -> tuple[TemporalTactileObservation, ...]:
        if not self._accumulating:
            return ()
        accepted = []
        for observation in observations:
            if not isinstance(observation, TemporalTactileObservation):
                raise TypeError(
                    "ContactPointBuffer accepts only TemporalTactileObservation records"
                )
            self._raw_observation_count += 1
            if (
                observation.estimated_pressure
                < self.config.minimum_pressure_threshold
            ):
                self._rejected_pressure_count += 1
                continue
            last_time = self._last_accepted_time.get(observation.sensor_id)
            if last_time is not None and (
                observation.timestamp - last_time
                < self.config.minimum_time_between_points
            ):
                self._rejected_duplicate_count += 1
                continue
            if self._is_spatial_duplicate(observation):
                self._rejected_duplicate_count += 1
                continue
            self._points.append(observation)
            accepted.append(observation)
            self._last_accepted_time[observation.sensor_id] = observation.timestamp
            overflow = len(self._points) - self.config.maximum_point_count
            if overflow > 0:
                del self._points[:overflow]
        return tuple(accepted)

    def reset(self) -> None:
        self._points.clear()
        self._raw_observation_count = 0
        self._rejected_duplicate_count = 0
        self._rejected_pressure_count = 0
        self._last_accepted_time.clear()
