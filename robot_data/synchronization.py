"""Deterministic timestamp interpolation for asynchronous robot streams."""

from __future__ import annotations

from bisect import bisect_left
from collections import deque
import math

from robot_data.common import CommonRobotState, JointStateSample, TactileStateSample


class SynchronizationUnavailableError(RuntimeError):
    pass


class TimestampSynchronizer:
    def __init__(self, maximum_joint_samples: int = 2048, maximum_gap_seconds: float | None = None) -> None:
        if maximum_joint_samples < 2:
            raise ValueError("maximum_joint_samples must be at least two")
        self._joint_samples: deque[JointStateSample] = deque(maxlen=maximum_joint_samples)
        if maximum_gap_seconds is not None and (not math.isfinite(maximum_gap_seconds) or maximum_gap_seconds <= 0):
            raise ValueError("maximum_gap_seconds must be finite and positive")
        self.maximum_gap_seconds = maximum_gap_seconds

    @property
    def joint_samples(self) -> tuple[JointStateSample, ...]:
        return tuple(self._joint_samples)

    def clear(self) -> None:
        self._joint_samples.clear()

    def add_joint_state(self, sample: JointStateSample) -> None:
        if self._joint_samples and sample.timestamp < self._joint_samples[-1].timestamp:
            raise ValueError("Joint-state timestamps must be monotonic")
        if self._joint_samples and sample.timestamp == self._joint_samples[-1].timestamp:
            self._joint_samples[-1] = sample
        else:
            self._joint_samples.append(sample)

    def joint_state_at(self, timestamp: float) -> JointStateSample:
        if not self._joint_samples:
            raise SynchronizationUnavailableError("No joint-state samples are available")
        query = float(timestamp)
        if not math.isfinite(query) or query < 0:
            raise ValueError("Interpolation timestamp must be finite and non-negative")
        samples = tuple(self._joint_samples)
        times = [sample.timestamp for sample in samples]
        index = bisect_left(times, query)
        if index < len(samples) and samples[index].timestamp == query:
            sample = samples[index]
            return JointStateSample(query, sample.positions, sample.velocities)
        if index == 0 or index == len(samples):
            raise SynchronizationUnavailableError(
                f"Cannot interpolate joint state at {query}; available interval is "
                f"[{times[0]}, {times[-1]}]"
            )
        before, after = samples[index - 1], samples[index]
        if self.maximum_gap_seconds is not None and after.timestamp - before.timestamp > self.maximum_gap_seconds:
            raise SynchronizationUnavailableError("Joint sample gap exceeds maximum_gap_seconds")
        if set(before.positions) != set(after.positions):
            raise ValueError("Joint names changed between synchronization samples")
        weight = (query - before.timestamp) / (after.timestamp - before.timestamp)
        positions = {
            name: before.positions[name] + weight * (after.positions[name] - before.positions[name])
            for name in before.positions
        }
        velocities = {
            name: before.velocities[name] + weight * (after.velocities[name] - before.velocities[name])
            for name in before.velocities
        }
        return JointStateSample(query, positions, velocities)

    def synchronize(
        self,
        tactile: TactileStateSample,
        *,
        source: str,
        joint_targets=None,
        object_state=None,
        grasp_state=None,
        camera=None,
        time_base=None,
        link_poses=(),
    ) -> CommonRobotState:
        joints = self.joint_state_at(tactile.timestamp)
        times = [sample.timestamp for sample in self._joint_samples]
        index = bisect_left(times, tactile.timestamp)
        before = times[index] if times[index] == tactile.timestamp else times[index - 1]
        return CommonRobotState(
            timestamp=tactile.timestamp,
            source=source,
            joint_positions=joints.positions,
            joint_velocities=joints.velocities,
            joint_targets=joint_targets or {},
            tactile_channels=tactile.channels,
            object_state=object_state,
            grasp_state=grasp_state,
            camera=camera,
            link_poses=link_poses,
            synchronization={
                "joint_before_timestamp": before,
                "joint_after_timestamp": times[index],
                "tactile_timestamp": tactile.timestamp,
                "time_base": time_base,
            },
        )
