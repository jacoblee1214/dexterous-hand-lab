"""Robot provider contract and configuration-driven provider factory."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from robot_data.common import CommonRobotState, ProviderDescription, RobotCommand


class ProviderNotImplementedError(NotImplementedError):
    code = "NOT_IMPLEMENTED"

    def __init__(self, operation: str) -> None:
        super().__init__(f"NOT_IMPLEMENTED: {operation}")
        self.operation = operation


class RobotDataProvider(ABC):
    @property
    def connection_status(self) -> dict[str, Any]:
        return {"state": "STREAMING", "detail": "", "read_only": self.description.source != "simulation"}

    @property
    def available_commands(self) -> tuple[str, ...]:
        return ()

    @property
    def poll_interval_seconds(self) -> float | None:
        return None

    @property
    @abstractmethod
    def description(self) -> ProviderDescription:
        raise NotImplementedError

    @property
    @abstractmethod
    def current_timestamp(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def step(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def read_common_state(self) -> CommonRobotState:
        raise NotImplementedError

    @abstractmethod
    def send_command(self, command: RobotCommand) -> None:
        raise NotImplementedError

    def evaluation_payload(self, estimates, points, sphere_result) -> dict[str, Any] | None:
        del estimates, points, sphere_result
        return None

    def close(self) -> None:
        pass

    def open_debug_viewer(self):
        raise ProviderNotImplementedError("native debug viewer")

    def sync_debug_viewer(self):
        pass


def create_robot_data_provider(
    source: str,
    *,
    replay_path: str | Path | None = None,
) -> RobotDataProvider:
    normalized = source.strip().lower()
    if normalized == "simulation":
        from robot_data.mujoco_provider import MuJoCoRobotDataProvider

        return MuJoCoRobotDataProvider()
    if normalized == "hardware":
        from robot_data.hardware_provider import HardwareRobotDataProvider

        return HardwareRobotDataProvider()
    if normalized == "replay":
        if replay_path is None:
            raise ValueError("source: replay requires replay.path")
        from robot_data.replay_provider import ReplayRobotDataProvider

        return ReplayRobotDataProvider(replay_path)
    raise ValueError(f"Unknown robot data source: {source}")
