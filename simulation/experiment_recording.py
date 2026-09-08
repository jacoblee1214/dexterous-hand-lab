"""Named, repeatable right-hand sphere experiment recording."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Mapping


EXPERIMENT_SCHEMA_VERSION = 1
_VALID_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {"count": 0, "minimum": None, "mean": None, "maximum": None, "stddev": None}
    return {
        "count": len(clean),
        "minimum": min(clean),
        "mean": statistics.fmean(clean),
        "maximum": max(clean),
        "stddev": statistics.pstdev(clean),
    }


class RepeatableSphereExperiment:
    """In-memory live trial with separate reconstruction/evaluation persistence."""

    def __init__(self, output_root: str | Path) -> None:
        self.output_root = Path(output_root)
        self.active = False
        self.name: str | None = None
        self.trial_number = 0
        self.metadata: dict[str, Any] = {}
        self.samples: list[dict[str, Any]] = []
        self.evaluation_samples: list[dict[str, Any]] = []
        self.accepted_contacts: list[dict[str, Any]] = []
        self.last_result: dict[str, Any] | None = None

    def start(self, name: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
        clean = str(name).strip()
        if not clean or not _VALID_NAME.fullmatch(clean):
            raise ValueError("Experiment name may contain only letters, numbers, '_' and '-'")
        if self.active:
            raise ValueError("Finish the active experiment trial before starting another")
        directory = self.output_root / clean
        existing = sorted(directory.glob("trial_*.reconstruction.json"))
        self.name = clean
        self.trial_number = len(existing) + 1
        self.metadata = dict(metadata)
        self.metadata.update(
            {
                "experiment_name": clean,
                "trial_number": self.trial_number,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        self.samples = []
        self.evaluation_samples = []
        self.accepted_contacts = []
        self.active = True
        self.last_result = None
        return self.status()

    def record_sample(
        self,
        common_state,
        accepted,
        grasp_diagnostics: Mapping[str, Any],
        evaluation_diagnostics: Mapping[str, Any],
    ) -> None:
        if not self.active:
            return
        accepted_payload = [
            {
                "timestamp": item.timestamp,
                "sensor_id": item.sensor_id,
                "finger_id": item.finger_id,
                "parent_link": item.parent_link,
                "joint_state_snapshot": dict(item.joint_state_snapshot),
                "sensor_position": list(item.sensor_position),
                "sensor_sensing_direction": list(item.sensor_sensing_direction),
                "scalar_sensor_value": item.scalar_sensor_value,
                "estimated_pressure": item.estimated_pressure,
                "estimated_indentation": item.estimated_indentation,
                "estimated_contact_position": list(item.estimated_contact_position),
            }
            for item in accepted
        ]
        self.accepted_contacts.extend(accepted_payload)
        self.samples.append(
            {
                "timestamp": common_state.timestamp,
                "grasp_phase": grasp_diagnostics.get("phase_name"),
                "joint_positions": dict(common_state.joint_positions),
                "joint_velocities": dict(common_state.joint_velocities),
                "joint_targets": dict(common_state.joint_targets),
                "actuator_forces_n_m": dict(grasp_diagnostics.get("actuator_forces_n_m", {})),
                "tactile_channels": [item.to_dict() for item in common_state.tactile_channels],
                "active_sensor_ids": list(grasp_diagnostics.get("active_sensor_ids", [])),
                "active_finger_count": int(grasp_diagnostics.get("active_finger_count", 0)),
                "accepted_estimated_contacts": accepted_payload,
                "settled": bool(grasp_diagnostics.get("settled", False)),
                "settling_time_seconds": grasp_diagnostics.get("settling_time_seconds"),
                "reference_timed_out": bool(grasp_diagnostics.get("reference_timed_out", False)),
            }
        )
        # Object pose/contact distances are simulator evaluation signals and stay
        # physically separate from the reconstruction-side dataset above.
        self.evaluation_samples.append(
            {"timestamp": common_state.timestamp, **dict(evaluation_diagnostics)}
        )

    def finish(
        self,
        reconstruction: Mapping[str, Any],
        evaluation: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not self.active or self.name is None:
            raise ValueError("No named experiment trial is active")
        stem = f"trial_{self.trial_number:03d}"
        directory = self.output_root / self.name
        reconstruction_path = directory / f"{stem}.reconstruction.json"
        evaluation_path = directory / f"{stem}.evaluation.json"
        reconstruction_payload = {
            "schema_version": EXPERIMENT_SCHEMA_VERSION,
            "dataset_type": "repeatable_tactile_sphere_experiment",
            "metadata": self.metadata,
            "sample_count": len(self.samples),
            "samples": self.samples,
            "accepted_estimated_contacts": self.accepted_contacts,
            "reconstruction_result": dict(reconstruction),
        }
        evaluation_payload = {
            "schema_version": EXPERIMENT_SCHEMA_VERSION,
            "dataset_type": "mujoco_sphere_experiment_evaluation",
            "experiment_name": self.name,
            "trial_number": self.trial_number,
            "samples": self.evaluation_samples,
            "evaluation_result": dict(evaluation),
        }
        _write_json(reconstruction_path, reconstruction_payload)
        _write_json(evaluation_path, evaluation_payload)
        trial_summary = self._trial_summary(reconstruction, reconstruction_path, evaluation_path)
        summary_path = directory / "summary.json"
        previous = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        trials = list(previous.get("trials", []))
        trials.append(trial_summary)
        summary = {
            "schema_version": EXPERIMENT_SCHEMA_VERSION,
            "experiment_name": self.name,
            "number_of_trials": len(trials),
            "trials": trials,
            "distribution": {
                "accepted_point_count": _distribution([item["accepted_point_count"] for item in trials]),
                "unique_sensor_count": _distribution([item["unique_sensor_count"] for item in trials]),
                "unique_finger_count": _distribution([item["unique_finger_count"] for item in trials]),
                "settling_time_seconds": _distribution(
                    [item["settling_time_seconds"] for item in trials if item["settling_time_seconds"] is not None]
                ),
                "final_max_joint_velocity_rad_s": _distribution(
                    [item["final_max_joint_velocity_rad_s"] for item in trials]
                ),
                "maximum_penetration_m": _distribution(
                    [item["maximum_penetration_m"] for item in trials]
                ),
                "maximum_object_translation_m": _distribution(
                    [item["maximum_object_translation_m"] for item in trials]
                ),
            },
        }
        _write_json(summary_path, summary)
        self.active = False
        self.last_result = {
            **trial_summary,
            "number_of_trials": len(trials),
            "summary_path": str(summary_path),
        }
        return dict(self.last_result)

    def _trial_summary(self, reconstruction, reconstruction_path, evaluation_path) -> dict[str, Any]:
        settled_samples = [item for item in self.samples if item["settled"]]
        active_ids = {sensor for item in self.samples for sensor in item["active_sensor_ids"]}
        fingers = {item["finger_id"] for item in self.accepted_contacts}
        final_velocities = self.samples[-1]["joint_velocities"] if self.samples else {}
        penetration_values = [
            float(item.get("contact_diagnostics", {}).get("maximum_penetration_m", 0.0))
            for item in self.evaluation_samples
        ]
        object_positions = [
            item.get("object_position_xyz")
            for item in self.evaluation_samples
            if item.get("object_position_xyz") is not None
            and item.get("object_in_workspace", True)
        ]
        if object_positions:
            origin = object_positions[0]
            translations = [
                math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(position, origin)))
                for position in object_positions
            ]
            final_position = [float(value) for value in object_positions[-1]]
        else:
            translations, final_position = [0.0], None
        return {
            "trial_number": self.trial_number,
            "reconstruction_path": str(reconstruction_path),
            "evaluation_path": str(evaluation_path),
            "reconstruction_status": reconstruction.get("status", "NOT_FITTED"),
            "accepted_point_count": len(self.accepted_contacts),
            "unique_sensor_count": len(active_ids),
            "unique_finger_count": len(fingers),
            "active_sensor_ids": sorted(active_ids),
            "contributing_fingers": sorted(fingers),
            "settled": bool(settled_samples),
            "settling_time_seconds": (
                settled_samples[-1]["settling_time_seconds"] if settled_samples else None
            ),
            "final_max_joint_velocity_rad_s": max(
                (abs(float(value)) for value in final_velocities.values()), default=0.0
            ),
            "maximum_penetration_m": max(penetration_values, default=0.0),
            "maximum_object_translation_m": max(translations),
            "final_object_translation_m": translations[-1],
            "final_object_position_xyz": final_position,
        }

    def status(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "name": self.name,
            "trial_number": self.trial_number if self.name else None,
            "sample_count": len(self.samples),
            "last_result": self.last_result,
        }
