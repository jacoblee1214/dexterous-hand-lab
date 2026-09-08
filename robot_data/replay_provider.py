"""Offline provider that emits a recorded common-state stream unchanged."""

from __future__ import annotations

from pathlib import Path

from robot_data.common import CommonRobotState, ProviderDescription, RobotCommand, RobotCommandName
from robot_data.provider import RobotDataProvider
from robot_data.recording import read_common_state_log


class ReplayRobotDataProvider(RobotDataProvider):
    @property
    def available_commands(self):
        return ("reset",)

    @property
    def poll_interval_seconds(self):
        if self.finished:
            return 0.05
        return max(0.0001, self._states[self._index + 1].timestamp - self.current_timestamp)

    @property
    def finished(self):
        return self._index == len(self._states) - 1

    @property
    def connection_status(self):
        return {"state": "STREAMING", "read_only": True,
                "detail": "Replay complete" if self.finished else "Recorded measurements",
                "replay_finished": self.finished}

    def __init__(self, path: str | Path) -> None:
        self.log_path = Path(path).resolve()
        original, self._states = read_common_state_log(self.log_path)
        self._description = ProviderDescription(
            source="replay",
            source_display_name=f"REPLAY · {original.source_display_name}",
            handedness=original.handedness,
            kinematic_model_path=original.kinematic_model_path,
            sensor_config_path=original.sensor_config_path,
            calibration_config=original.calibration_config,
            joints=original.joints,
        )
        self._index = 0
        self._paused = False

    @property
    def description(self) -> ProviderDescription:
        return self._description

    @property
    def current_timestamp(self) -> float:
        return self._states[self._index].timestamp

    def step(self) -> None:
        if not self._paused and self._index + 1 < len(self._states):
            self._index += 1

    def read_common_state(self) -> CommonRobotState:
        state = self._states[self._index]
        return CommonRobotState(
            timestamp=state.timestamp,
            source="replay",
            joint_positions=state.joint_positions,
            joint_velocities=state.joint_velocities,
            joint_targets=state.joint_targets,
            tactile_channels=state.tactile_channels,
            object_state=state.object_state,
            grasp_state=state.grasp_state,
            camera=state.camera,
            synchronization=state.synchronization,
            link_poses=state.link_poses,
        )

    def send_command(self, command: RobotCommand) -> None:
        if command.name is RobotCommandName.RESET:
            self._index = 0
            return
        # Robot actuation commands intentionally do not mutate recorded samples.
        raise ValueError(f"Command {command.name.value} is unavailable during replay")
