"""Documented extension point for a future real-hand transport.

No serial, CAN, ROS, or network protocol is assumed here. Integrators should
implement the three explicit I/O hooks once the actual hardware contract is
known, then combine their timestamped samples with TimestampSynchronizer.
"""

from __future__ import annotations

import math
import time
from collections import deque

from robot_data.common import ProviderDescription, RobotCommand
from robot_data.provider import ProviderNotImplementedError, RobotDataProvider
from robot_data.robot_description import right_hand_description
from robot_data.synchronization import TimestampSynchronizer, SynchronizationUnavailableError


class HardwareDataUnavailableError(RuntimeError):
    pass


class HardwareReadOnlyError(ProviderNotImplementedError):
    code = "READ_ONLY"


class HardwareRobotDataProvider(RobotDataProvider):
    SOURCE_DISPLAY_NAME = "REAL ROBOT"

    def __init__(self, description=None, *, source_time_now=None, time_base=None,
                 monotonic=time.monotonic, stale_after_seconds=0.5,
                 maximum_interpolation_gap_seconds=0.1):
        self._description = description or right_hand_description()
        if self._description.source != "hardware":
            raise ValueError("Hardware description must identify source hardware")
        from sensors.sensor_kinematics import load_sensor_mounts
        from sensors.registry_audit import require_approved_right_sensor_registry
        mounts = load_sensor_mounts(self._description.sensor_config_path)
        require_approved_right_sensor_registry(mounts, self._description.calibration_config)
        if not math.isfinite(stale_after_seconds) or stale_after_seconds <= 0:
            raise ValueError("stale_after_seconds must be finite and positive")
        self._source_now = source_time_now
        self.time_base = time_base
        self._monotonic = monotonic
        self.stale_after_seconds = stale_after_seconds
        self._sync = TimestampSynchronizer(maximum_gap_seconds=maximum_interpolation_gap_seconds)
        self._connection = "DISCONNECTED"
        self._detail = "Hardware transport is not configured"
        self._joint = self._tactile = self._common = None
        self._pending = deque(maxlen=128)
        self._received = {}
        self._last_tactile_timestamp = None
        self._camera = None
        self._session_id = 0

    def begin_connection(self):
        """Called by a future receive adapter after explicit clock configuration."""
        if self._source_now is None or not self.time_base:
            raise ValueError("An explicit source time base and source_time_now mapping are required")
        self._session_id += 1
        self._sync.clear()
        self._pending.clear()
        self._received.clear()
        self._joint = self._tactile = self._common = None
        self._last_tactile_timestamp = None
        self._camera = None
        self._connection, self._detail = "CONNECTING", "Waiting for synchronized measurements"

    def disconnect(self):
        self._connection, self._detail = "DISCONNECTED", "Hardware disconnected"

    def report_error(self, detail):
        self._connection, self._detail = "ERROR", str(detail)

    def accept_camera_observation(self, observation):
        """Attach calibrated image references only; no temporal/geometric fusion."""
        if self._connection not in {"CONNECTING", "STREAMING"}:
            raise HardwareDataUnavailableError("Camera input requires a configured connection")
        if observation.time_base != self.time_base:
            raise ValueError("Camera must declare the configured source time base")
        age = self._age(observation.timestamp)
        if age < -1e-6 or age > self.stale_after_seconds:
            raise ValueError("Camera source timestamp is invalid or stale")
        self._camera = observation

    def _age(self, timestamp):
        now = float(self._source_now())
        if not math.isfinite(now):
            raise ValueError("Source clock returned a non-finite time")
        return now - timestamp

    @property
    def connection_status(self):
        status, detail = self._connection, self._detail
        age = None
        if status in {"CONNECTING", "STREAMING"}:
            try:
                stamps = [sample.timestamp for sample in (self._joint, self._tactile, self._common)
                          if sample is not None]
                ages = [self._age(stamp) for stamp in stamps]
                age = max(ages) if ages else None
                watchdog = [self._monotonic() - stamp for stamp in self._received.values()]
                if any(value < -1e-6 for value in ages):
                    status, detail = "ERROR", "Source clock moved behind measurements"
                elif (age is not None and age > self.stale_after_seconds) or any(
                    value > self.stale_after_seconds for value in watchdog
                ):
                    status, detail = "STALE", "Measurement age or receive watchdog exceeded limit"
            except (ValueError, TypeError) as exc:
                status, detail = "ERROR", str(exc)
        return {"state": status, "detail": detail, "read_only": True,
                "measurement_age_seconds": age, "stale_after_seconds": self.stale_after_seconds,
                "time_base": self.time_base, "session_id": self._session_id}

    def _validate_sample(self, sample, kind):
        if self._connection not in {"CONNECTING", "STREAMING"}:
            raise HardwareDataUnavailableError("Begin a new connection before supplying measurements")
        age = self._age(sample.timestamp)
        if age < -1e-6:
            raise ValueError("Measurement timestamp is in the future; check clock mapping")
        if age > self.stale_after_seconds:
            raise ValueError("Stale source measurement rejected")
        if kind == "joint":
            if set(sample.positions) != {joint.name for joint in self.description.joints}:
                raise ValueError("Named joint registry mismatch")
            if self._joint and sample.timestamp <= self._joint.timestamp:
                raise ValueError("Joint source timestamps must strictly increase")
        else:
            if {channel.sensor_id for channel in sample.channels} != set(self.description.calibration_config):
                raise ValueError("Expected exactly the approved 18 named tactile channels")
            if self._last_tactile_timestamp is not None and sample.timestamp <= self._last_tactile_timestamp:
                raise ValueError("Tactile source timestamps must strictly increase")

    def accept_joint_state(self, sample):
        """Ingest measured, named positions/velocities in radians and radians/s."""
        try:
            self._validate_sample(sample, "joint")
            self._sync.add_joint_state(sample)
            self._joint = sample
            self._received["joint"] = self._monotonic()
            self._synchronize_pending()
        except ValueError as exc:
            self.report_error(exc)
            raise

    def accept_tactile_state(self, sample):
        """Ingest all approved calibrated scalars at their source timestamp."""
        try:
            self._validate_sample(sample, "tactile")
            if len(self._pending) == self._pending.maxlen:
                raise ValueError("Pending tactile buffer overflow; joint stream is not keeping up")
            self._pending.append(sample)
            self._tactile = sample
            self._last_tactile_timestamp = sample.timestamp
            self._received["tactile"] = self._monotonic()
            self._synchronize_pending()
        except ValueError as exc:
            self.report_error(exc)
            raise

    def _synchronize_pending(self):
        while self._pending:
            sample = self._pending[0]
            try:
                state = self._sync.synchronize(sample, source="hardware", time_base=self.time_base,
                                                camera=self._camera)
            except SynchronizationUnavailableError as exc:
                joints = self._sync.joint_samples
                if not joints or sample.timestamp > joints[-1].timestamp:
                    self._connection, self._detail = "CONNECTING", "Waiting for a bracketing encoder sample"
                    return
                raise ValueError(f"Invalid interpolation: {exc}") from exc
            if self._age(sample.timestamp) > self.stale_after_seconds:
                raise ValueError("Tactile measurement became stale while awaiting interpolation")
            self._common = state
            self._pending.popleft()
            self._connection, self._detail = "STREAMING", "Read-only synchronized measurements"

    @property
    def description(self) -> ProviderDescription:
        return self._description

    @property
    def current_timestamp(self) -> float:
        return self._common.timestamp if self._common else 0.0

    def read_joint_state(self):
        """Return a timestamped JointStateSample from the future transport."""
        raise ProviderNotImplementedError("read_joint_state")

    def read_tactile_state(self):
        """Return a timestamped TactileStateSample from the future transport."""
        raise ProviderNotImplementedError("read_tactile_state")

    def send_joint_targets(self, targets) -> None:
        """Motor output is disabled, even when receive hooks are configured."""
        del targets
        raise HardwareReadOnlyError("send_joint_targets: hardware motor control is READ_ONLY")

    def step(self) -> None:
        if self._source_now is None:
            raise ProviderNotImplementedError("poll_hardware")

    def read_common_state(self):
        if self._source_now is None:
            raise ProviderNotImplementedError("read_common_state")
        if self.connection_status["state"] != "STREAMING" or self._common is None:
            raise HardwareDataUnavailableError(self.connection_status["detail"])
        return self._common

    def send_command(self, command: RobotCommand) -> None:
        del command
        raise HardwareReadOnlyError("send_command: all hardware motor commands are READ_ONLY")
