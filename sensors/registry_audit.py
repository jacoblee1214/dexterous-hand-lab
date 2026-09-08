"""Approved right-hand physical sensor registry sanity checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from sensors.sensor_kinematics import SensorMount


APPROVED_RIGHT_SENSOR_COUNT = 18
APPROVED_RIGHT_SENSOR_IDS = frozenset(
    [f"Finger{finger:02d}_{region}_Sensor" for finger in range(1, 6)
     for region in ("Link02", "Link03", "Tip")]
    + [f"Palm{pad:02d}_Sensor" for pad in range(1, 4)]
)
INTENTIONALLY_ABSENT_RIGHT_SENSORS = {
    "Finger01_Link04_Sensor": "Link04 removed by explicit user decision",
    "Finger02_Link04_Sensor": "Link04 removed by explicit user decision",
    "Finger03_Link04_Sensor": "Link04 removed by explicit user decision",
    "Finger04_Link04_Sensor": "Link04 removed by explicit user decision",
    "Finger05_Link04_Sensor": "Link04 removed by explicit user decision",
}


@dataclass(frozen=True)
class SensorRegistryAudit:
    configured_count: int
    runtime_count: int
    configured_only: tuple[str, ...]
    runtime_only: tuple[str, ...]
    intentionally_absent: Mapping[str, str]

    @property
    def approved(self) -> bool:
        return (
            self.configured_count == APPROVED_RIGHT_SENSOR_COUNT
            and self.runtime_count == APPROVED_RIGHT_SENSOR_COUNT
            and not self.configured_only
            and not self.runtime_only
        )


def audit_right_sensor_registry(
    mounts: Mapping[str, SensorMount],
    runtime_sensor_ids: Iterable[str],
) -> SensorRegistryAudit:
    configured = set(mounts)
    runtime = set(runtime_sensor_ids)
    return SensorRegistryAudit(
        configured_count=len(configured),
        runtime_count=len(runtime),
        configured_only=tuple(sorted(configured - runtime)),
        runtime_only=tuple(sorted(runtime - configured)),
        intentionally_absent=dict(INTENTIONALLY_ABSENT_RIGHT_SENSORS),
    )


def require_approved_right_sensor_registry(
    mounts: Mapping[str, SensorMount],
    runtime_sensor_ids: Iterable[str],
) -> SensorRegistryAudit:
    audit = audit_right_sensor_registry(mounts, runtime_sensor_ids)
    if not audit.approved or set(mounts) != APPROVED_RIGHT_SENSOR_IDS:
        raise RuntimeError(
            "Right sensor registry differs from the manually approved 18-channel "
            f"layout: configured={audit.configured_count}, runtime={audit.runtime_count}, "
            f"configured_only={list(audit.configured_only)}, "
            f"runtime_only={list(audit.runtime_only)}"
        )
    return audit


def format_right_sensor_registry_table(
    mounts: Mapping[str, SensorMount], runtime_sensor_ids: Iterable[str]
) -> str:
    runtime = set(runtime_sensor_ids)
    rows = [
        "Sensor ID | Finger | Parent Link | Sensor Type | State | Reason",
        "--- | --- | --- | --- | --- | ---",
    ]
    for sensor_id, mount in mounts.items():
        rows.append(
            f"{sensor_id} | {mount.finger_id} | {mount.parent_link} | "
            f"{mount.sensor_type} | {'ENABLED' if sensor_id in runtime else 'DISABLED'} | "
            f"{'-' if sensor_id in runtime else 'missing from runtime model'}"
        )
    for sensor_id, reason in INTENTIONALLY_ABSENT_RIGHT_SENSORS.items():
        finger = sensor_id.split("_")[0]
        rows.append(
            f"{sensor_id} | {finger} | Link04 | rectangular | DISABLED | {reason}"
        )
    return "\n".join(rows)
