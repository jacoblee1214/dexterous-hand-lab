"""JSON Lines recording for reproducible provider-neutral replay."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from robot_data.common import CommonRobotState, ProviderDescription


LOG_SCHEMA_VERSION = "1.0.0"


class CommonStateRecorder:
    def __init__(self, path: str | Path, description: ProviderDescription) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("w", encoding="utf-8")
        self._write(
            {
                "log_schema_version": LOG_SCHEMA_VERSION,
                "record_type": "metadata",
                "provider": description.to_dict(),
            }
        )

    def _write(self, payload: dict) -> None:
        self._stream.write(json.dumps(payload, separators=(",", ":"), allow_nan=False))
        self._stream.write("\n")
        self._stream.flush()

    def record(self, state: CommonRobotState) -> None:
        self._write(
            {
                "log_schema_version": LOG_SCHEMA_VERSION,
                "record_type": "state",
                "state": state.to_dict(),
            }
        )

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()

    def __enter__(self) -> "CommonStateRecorder":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def read_common_state_log(
    path: str | Path,
) -> tuple[ProviderDescription, tuple[CommonRobotState, ...]]:
    records: Iterator[dict] = (
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    try:
        header = next(records)
    except StopIteration as exc:
        raise ValueError("Replay log is empty") from exc
    if (
        header.get("log_schema_version") != LOG_SCHEMA_VERSION
        or header.get("record_type") != "metadata"
    ):
        raise ValueError("Replay log has an invalid or unsupported metadata header")
    description = ProviderDescription.from_dict(header["provider"])
    states = []
    for record in records:
        if (
            record.get("log_schema_version") != LOG_SCHEMA_VERSION
            or record.get("record_type") != "state"
        ):
            raise ValueError("Replay log contains an invalid record")
        states.append(CommonRobotState.from_dict(record["state"]))
    if not states:
        raise ValueError("Replay log contains no common-state samples")
    if any(right.timestamp < left.timestamp for left, right in zip(states, states[1:])):
        raise ValueError("Replay state timestamps must be monotonic")
    return description, tuple(states)
