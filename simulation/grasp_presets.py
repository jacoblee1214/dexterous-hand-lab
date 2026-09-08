"""Validated, persistent actuator-target presets for repeatable sphere grasps."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import shutil
from types import MappingProxyType
from typing import Mapping
import xml.etree.ElementTree as ET

import yaml


@dataclass(frozen=True)
class GraspPreset:
    name: str
    radius_m: float
    sphere_center_xyz: tuple[float, float, float]
    joint_targets: Mapping[str, float]
    is_taught: bool = False
    target_source: str = "legacy_actuator_target"
    waypoint_role: str = "final_grasp"
    actuator_targets: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType({})
    )
    measured_joint_positions: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType({})
    )
    metadata: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True)
class GraspStage:
    name: str
    joints: tuple[str, ...]


class GraspPresetLibrary:
    TAUGHT_SOURCES = ("actual_measured", "actuator_target")
    WAYPOINT_ROLES = (
        "reference_open",
        "thumb_opposition",
        "finger_approach",
        "final_grasp",
    )

    def __init__(
        self,
        path: str | Path,
        urdf_path: str | Path,
        joint_config_path: str | Path | None = None,
    ) -> None:
        self.path = Path(path)
        self.urdf_path = Path(urdf_path)
        self.joint_config_path = Path(joint_config_path) if joint_config_path else None
        self.hand = ""
        self.opposition_joint = ""
        self.secondary_opposition_joint = ""
        self.stage_duration_seconds = 0.0
        self.final_closure_duration_seconds = 0.0
        self.settling_dwell_seconds = 0.0
        self.settling_velocity_threshold_rad_s = 0.0
        self.preload_target_offset_rad = 0.0
        self.reference_position_tolerance_rad = 0.0
        self.reference_velocity_threshold_rad_s = 0.0
        self.reference_max_duration_seconds = 0.0
        self.taught_default_segment_duration_seconds = 0.0
        self.taught_default_pose_tolerance_rad = 0.0
        self.taught_max_target_velocity_rad_s = 0.0
        self.config_version = ""
        self.stages: tuple[GraspStage, ...] = ()
        self.presets: dict[str, GraspPreset] = {}
        self.joint_limits = self._read_urdf_limits()
        self.reload()

    def _read_urdf_limits(self) -> dict[str, tuple[float, float]]:
        result = {}
        for joint in ET.parse(self.urdf_path).getroot().findall("joint"):
            if joint.attrib.get("type") == "fixed":
                continue
            limit = joint.find("limit")
            result[joint.attrib["name"]] = (
                float(limit.attrib["lower"]),
                float(limit.attrib["upper"]),
            )
        return result

    def reload(self) -> None:
        raw = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != 1 or raw.get("hand") != "right":
            raise ValueError("Right grasp config must use schema_version 1 and hand: right")
        self.hand = raw["hand"]
        self.config_version = str(raw.get("config_version", "unversioned"))
        self.opposition_joint = str(raw["opposition_joint"])
        if self.opposition_joint not in self.joint_limits:
            raise ValueError(f"Unknown opposition joint: {self.opposition_joint}")
        self.secondary_opposition_joint = str(raw["secondary_opposition_joint"])
        if self.secondary_opposition_joint not in self.joint_limits:
            raise ValueError(
                f"Unknown secondary opposition joint: {self.secondary_opposition_joint}"
            )
        self.stage_duration_seconds = float(raw["stage_duration_seconds"])
        if not math.isfinite(self.stage_duration_seconds) or self.stage_duration_seconds <= 0:
            raise ValueError("stage_duration_seconds must be positive")
        compliant = raw.get("compliant_closure", {})
        self.final_closure_duration_seconds = float(
            compliant.get("duration_seconds", self.stage_duration_seconds)
        )
        self.settling_dwell_seconds = float(compliant.get("settling_dwell_seconds", 0.4))
        self.settling_velocity_threshold_rad_s = float(
            compliant.get("settling_velocity_threshold_rad_s", 0.03)
        )
        self.preload_target_offset_rad = float(
            compliant.get("preload_target_offset_rad", 0.02)
        )
        reference = raw.get("reference_settling", {})
        self.reference_position_tolerance_rad = float(
            reference.get("position_tolerance_rad", 0.02)
        )
        self.reference_velocity_threshold_rad_s = float(
            reference.get("velocity_threshold_rad_s", 0.03)
        )
        self.reference_max_duration_seconds = float(
            reference.get("maximum_duration_seconds", 5.0)
        )
        teach_in = raw.get("teach_in", {})
        self.taught_default_segment_duration_seconds = float(
            teach_in.get("default_segment_duration_seconds", 1.0)
        )
        self.taught_default_pose_tolerance_rad = float(
            teach_in.get("default_pose_tolerance_rad", 0.03)
        )
        self.taught_max_target_velocity_rad_s = float(
            teach_in.get("maximum_target_velocity_rad_s", 0.8)
        )
        if any(
            not math.isfinite(value) or value <= 0
            for value in (
                self.final_closure_duration_seconds,
                self.settling_dwell_seconds,
                self.settling_velocity_threshold_rad_s,
                self.preload_target_offset_rad,
                self.reference_position_tolerance_rad,
                self.reference_velocity_threshold_rad_s,
                self.reference_max_duration_seconds,
                self.taught_default_segment_duration_seconds,
                self.taught_default_pose_tolerance_rad,
                self.taught_max_target_velocity_rad_s,
            )
        ):
            raise ValueError("Compliant closure settings must be finite and positive")
        stages = tuple(
            GraspStage(str(item["name"]), tuple(str(name) for name in item["joints"]))
            for item in raw["stages"]
        )
        staged_joints = [name for stage in stages for name in stage.joints]
        if (
            not set(staged_joints).issubset(self.joint_limits)
            or len(staged_joints) != len(set(staged_joints))
        ):
            raise ValueError("Approach stages contain unknown or duplicate joints")
        if len(stages) != 4 or stages[-1].joints:
            raise ValueError("Grasp config requires three approach stages and an empty final stage")
        presets = {}
        for name, values in raw["presets"].items():
            targets = {str(joint): float(target) for joint, target in values["joint_targets"].items()}
            self._validate_targets(name, targets)
            radius = float(values["radius_m"])
            if not math.isfinite(radius) or radius <= 0:
                raise ValueError(f"Preset {name} has invalid radius")
            center = tuple(float(value) for value in values["sphere_center_xyz"])
            if len(center) != 3 or any(not math.isfinite(value) for value in center):
                raise ValueError(f"Preset {name} has invalid sphere_center_xyz")
            taught = values.get("taught") or {}
            is_taught = bool(taught)
            target_source = str(taught.get("target_source", "legacy_actuator_target"))
            waypoint_role = str(taught.get("waypoint_role", "final_grasp"))
            actuator_targets = {
                str(joint): float(target)
                for joint, target in taught.get("actuator_targets", targets).items()
            }
            measured_positions = {
                str(joint): float(target)
                for joint, target in taught.get("measured_joint_positions", {}).items()
            }
            if is_taught:
                if target_source not in self.TAUGHT_SOURCES:
                    raise ValueError(f"Preset {name} has unknown taught target source")
                if waypoint_role not in self.WAYPOINT_ROLES:
                    raise ValueError(f"Preset {name} has unknown waypoint role")
                self._validate_targets(f"{name} actuator targets", actuator_targets)
                self._validate_targets(f"{name} measured positions", measured_positions)
            presets[name] = GraspPreset(
                name,
                radius,
                center,
                MappingProxyType(targets),
                is_taught=is_taught,
                target_source=target_source,
                waypoint_role=waypoint_role,
                actuator_targets=MappingProxyType(actuator_targets),
                measured_joint_positions=MappingProxyType(measured_positions),
                metadata=MappingProxyType(
                    {str(key): str(value) for key, value in taught.get("metadata", {}).items()}
                ),
            )
        required = {f"sphere_{value}mm" for value in (30, 40, 50, 60)}
        if not required.issubset(presets):
            raise ValueError(f"Missing required sphere presets: {sorted(required - set(presets))}")
        self.stages = stages
        self.presets = presets

    def _validate_targets(self, preset_name: str, targets: Mapping[str, float]) -> None:
        if set(targets) != set(self.joint_limits):
            missing = sorted(set(self.joint_limits) - set(targets))
            extra = sorted(set(targets) - set(self.joint_limits))
            raise ValueError(f"Preset {preset_name} joint mismatch: missing={missing}, extra={extra}")
        for joint, target in targets.items():
            lower, upper = self.joint_limits[joint]
            if not math.isfinite(target) or not lower <= target <= upper:
                raise ValueError(
                    f"Preset {preset_name} target {joint}={target} outside URDF [{lower}, {upper}]"
                )
        # A joint sign is only a rotation about that joint's local URDF axis.
        # It is not a semantic guarantee of flexion/opposition, so user-taught
        # poses are validated against named joints and limits only.

    def preset_for_radius(self, radius_m: float) -> GraspPreset:
        canonical = f"sphere_{round(float(radius_m) * 1000)}mm"
        try:
            return self.presets[canonical]
        except KeyError as exc:
            raise KeyError(f"No canonical grasp preset for radius {radius_m} m") from exc

    def save_preset(
        self,
        name: str,
        radius_m: float,
        sphere_center_xyz: tuple[float, float, float],
        joint_targets: Mapping[str, float],
        *,
        overwrite: bool = False,
    ) -> GraspPreset:
        clean_name = name.strip()
        if not clean_name or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in clean_name):
            raise ValueError("Preset name may contain only letters, numbers, '_' and '-'")
        targets = {str(joint): float(target) for joint, target in joint_targets.items()}
        self._validate_targets(clean_name, targets)
        raw = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        if clean_name in raw.get("presets", {}) and not overwrite:
            raise ValueError(
                f"Preset '{clean_name}' already exists; choose a new explicit name"
            )
        backup_directory = self.path.parent / "grasp_backups"
        backup_directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        shutil.copy2(self.path, backup_directory / f"{self.path.name}.{stamp}.bak")
        raw.setdefault("presets", {})[clean_name] = {
            "radius_m": float(radius_m),
            "sphere_center_xyz": [float(value) for value in sphere_center_xyz],
            "joint_targets": targets,
        }
        self.path.write_text(
            yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        self.reload()
        return self.presets[clean_name]

    @staticmethod
    def _sha256(path: Path | None) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest() if path else "not-provided"

    def save_taught_pose(
        self,
        name: str,
        radius_m: float,
        sphere_center_xyz: tuple[float, float, float],
        actuator_targets: Mapping[str, float],
        measured_joint_positions: Mapping[str, float],
        *,
        target_source: str,
        waypoint_role: str,
    ) -> GraspPreset:
        clean_name = name.strip()
        if not clean_name or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for character in clean_name
        ):
            raise ValueError("Pose name may contain only letters, numbers, '_' and '-'")
        if target_source not in self.TAUGHT_SOURCES:
            raise ValueError(f"Unknown taught target source: {target_source}")
        if waypoint_role not in self.WAYPOINT_ROLES:
            raise ValueError(f"Unknown taught waypoint role: {waypoint_role}")
        actuator = {str(joint): float(value) for joint, value in actuator_targets.items()}
        measured = {
            str(joint): float(value) for joint, value in measured_joint_positions.items()
        }
        self._validate_targets(clean_name, actuator)
        self._validate_targets(clean_name, measured)
        selected = measured if target_source == "actual_measured" else actuator
        raw = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        if clean_name in raw.get("presets", {}):
            raise ValueError(
                f"Preset '{clean_name}' already exists; choose a new explicit name"
            )
        backup_directory = self.path.parent / "grasp_backups"
        backup_directory.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc)
        stamp = now.strftime("%Y%m%dT%H%M%S.%fZ")
        shutil.copy2(self.path, backup_directory / f"{self.path.name}.{stamp}.bak")
        raw.setdefault("presets", {})[clean_name] = {
            "radius_m": float(radius_m),
            "sphere_center_xyz": [float(value) for value in sphere_center_xyz],
            "joint_targets": selected,
            "taught": {
                "target_source": target_source,
                "waypoint_role": waypoint_role,
                "actuator_targets": actuator,
                "measured_joint_positions": measured,
                "metadata": {
                    "created_at_utc": now.isoformat(),
                    "hand": self.hand,
                    "grasp_config_version": self.config_version,
                    "joint_mapping_version": self._sha256(self.joint_config_path),
                    "urdf_version": self._sha256(self.urdf_path),
                },
            },
        }
        self.path.write_text(
            yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        self.reload()
        return self.presets[clean_name]


class SphereGraspController:
    """Apply configured stages through actuators; never writes hand qpos."""

    def __init__(self, simulation, object_controller, library: GraspPresetLibrary) -> None:
        self.simulation = simulation
        self.object_controller = object_controller
        self.library = library
        self.selected_preset_name = self.library.preset_for_radius(
            object_controller.sphere_radius
        ).name
        self.current_stage = 0
        self.running = False
        self._next_stage_time = 0.0
        self._final_ramp_start_time: float | None = None
        self._final_ramp_start_targets: dict[str, float] = {}
        self._settling_start_time: float | None = None
        self._final_hold_targets: dict[str, float] | None = None
        self.settled = False
        self.settling_time_seconds: float | None = None
        self.sequence_start_time: float | None = None
        self.reference_timed_out = False
        self.teach_mode = False
        self.validation_mode = "physical_sphere_grasp"
        self.selected_taught_pose_name: str | None = None
        self.execution_kind: str | None = None
        self.taught_status = "IDLE"
        self.taught_failure_reason: str | None = None
        self.taught_tolerance_rad = self.library.taught_default_pose_tolerance_rad
        self.taught_segment_duration_seconds = self.library.taught_default_segment_duration_seconds
        self.taught_max_target_velocity_rad_s = self.library.taught_max_target_velocity_rad_s
        self._taught_waypoints: list[GraspPreset] = []
        self._taught_waypoint_index = 0
        self._taught_segment_start_time = 0.0
        self._taught_segment_start_targets: dict[str, float] = {}
        self._taught_last_update_time = float(self.simulation.data.time)
        self._taught_settle_start_time: float | None = None
        self._manual_still_since: float | None = None

    @property
    def selected_preset(self) -> GraspPreset:
        return self.library.presets[self.selected_preset_name]

    def select_for_radius(self, radius_m: float) -> None:
        preset = self.library.preset_for_radius(radius_m)
        self.selected_preset_name = preset.name
        self.object_controller.set_spawn_position(preset.sphere_center_xyz)
        self.object_controller.reset()
        self.running = False
        self.current_stage = 0
        self.selected_taught_pose_name = None
        if self.teach_mode:
            self.taught_status = "TEACHING"

    def select_named(self, name: str) -> GraspPreset:
        if name not in self.library.presets:
            raise KeyError(f"Unknown grasp preset: {name}")
        preset = self.library.presets[name]
        if not math.isclose(preset.radius_m, self.object_controller.sphere_radius, abs_tol=1e-9):
            raise ValueError(
                f"Preset {name} expects {preset.radius_m * 1000:.0f} mm sphere; "
                f"current radius is {self.object_controller.sphere_radius * 1000:.0f} mm"
            )
        self.selected_preset_name = name
        self.object_controller.set_spawn_position(preset.sphere_center_xyz)
        return preset

    def start(self) -> None:
        self.execution_kind = "automatic"
        self.simulation.set_actuator_targets(self.simulation.reference_positions())
        self.object_controller.set_spawn_position(
            self.selected_preset.sphere_center_xyz
        )
        self.object_controller.prepare_reference_phase()
        self.running = True
        self.current_stage = 0
        self._final_ramp_start_time = None
        self._settling_start_time = None
        self._final_hold_targets = None
        self.settled = False
        self.settling_time_seconds = None
        self.sequence_start_time = float(self.simulation.data.time)
        self.reference_timed_out = False
        # Give the actuator-driven reset time to approach the reference pose;
        # hand qpos is never assigned directly.
        self._next_stage_time = (
            float(self.simulation.data.time) + self.library.stage_duration_seconds
        )

    def apply_next_stage(self) -> None:
        if self.current_stage >= len(self.library.stages):
            self.running = False
            return
        stage = self.library.stages[self.current_stage]
        if self.current_stage == 0:
            self.object_controller.restore_after_reference()
        targets = self.selected_preset.joint_targets
        if not stage.joints:
            self._final_ramp_start_time = float(self.simulation.data.time)
            self._final_ramp_start_targets = self.current_actuator_targets()
            self._final_hold_targets = None
        else:
            selected = {name: targets[name] for name in stage.joints}
            self.simulation.set_actuator_targets(dict(selected))
        self.current_stage += 1
        self._next_stage_time = (
            float(self.simulation.data.time) + self.library.stage_duration_seconds
        )
        if self.current_stage >= len(self.library.stages) and self._final_ramp_start_time is None:
            self.running = False

    def apply_through_stage(self, stage_number: int) -> None:
        if not 1 <= stage_number <= len(self.library.stages):
            raise ValueError("Grasp stage must be 1 through 4")
        if stage_number == 1:
            self.simulation.set_actuator_targets(self.simulation.reference_positions())
            self.object_controller.reset()
        targets = self.selected_preset.joint_targets
        names = {
            name
            for stage in self.library.stages[:stage_number]
            for name in stage.joints
        }
        if stage_number == len(self.library.stages):
            names = set(targets)
        self.simulation.set_actuator_targets({name: targets[name] for name in names})
        self.current_stage = stage_number
        self.running = False
        self.execution_kind = None
        self._final_ramp_start_time = None
        self._settling_start_time = None
        self._final_hold_targets = None
        self.settled = False
        self.settling_time_seconds = None

    def update(self) -> None:
        now = float(self.simulation.data.time)
        self._update_manual_settling(now)
        if not self.running:
            return
        if self.execution_kind == "taught":
            self._update_taught_execution(now)
            return
        if self._final_ramp_start_time is not None:
            duration = self.library.final_closure_duration_seconds
            fraction = min(1.0, max(0.0, (now - self._final_ramp_start_time) / duration))
            target = self.selected_preset.joint_targets
            eased = fraction * fraction * (3.0 - 2.0 * fraction)
            if fraction < 1.0:
                commanded = {
                    name: start + eased * (target[name] - start)
                    for name, start in self._final_ramp_start_targets.items()
                }
            else:
                if self._final_hold_targets is None:
                    commanded = {}
                    preload = self.library.preload_target_offset_rad
                    for name in self.simulation.joint_names:
                        joint_id = self.simulation.mujoco.mj_name2id(
                            self.simulation.model,
                            self.simulation.mujoco.mjtObj.mjOBJ_JOINT,
                            name,
                        )
                        actual = float(
                            self.simulation.data.qpos[
                                self.simulation.model.jnt_qposadr[joint_id]
                            ]
                        )
                        error = target[name] - actual
                        commanded[name] = actual + math.copysign(
                            min(abs(error), preload), error
                        ) if error else actual
                    self._final_hold_targets = commanded
                commanded = self._final_hold_targets
            self.simulation.set_actuator_targets(commanded)
            if fraction < 1.0:
                return
            if self._settling_start_time is None:
                self._settling_start_time = now
            maximum_speed = self.maximum_hand_joint_speed()
            if maximum_speed <= self.library.settling_velocity_threshold_rad_s:
                if now - self._settling_start_time >= self.library.settling_dwell_seconds:
                    self.settled = True
                    self.settling_time_seconds = (
                        now - self.sequence_start_time
                        if self.sequence_start_time is not None
                        else None
                    )
                    self.running = False
                    self.execution_kind = None
                    self._final_ramp_start_time = None
            else:
                self._settling_start_time = now
            return
        if now >= self._next_stage_time:
            if self.current_stage == 0 and self.sequence_start_time is not None:
                ready = (
                    self.maximum_reference_position_error()
                    <= self.library.reference_position_tolerance_rad
                    and self.maximum_hand_joint_speed()
                    <= self.library.reference_velocity_threshold_rad_s
                )
                timed_out = (
                    now - self.sequence_start_time
                    >= self.library.reference_max_duration_seconds
                )
                if not ready and not timed_out:
                    return
                self.reference_timed_out = timed_out and not ready
            self.apply_next_stage()

    def cancel(self) -> None:
        self.running = False
        self.execution_kind = None
        self._final_ramp_start_time = None
        self._final_hold_targets = None
        self.object_controller.restore_after_reference()

    def actual_joint_positions(self) -> dict[str, float]:
        positions = {}
        for joint_name in self.simulation.joint_names:
            joint_id = self.simulation.mujoco.mj_name2id(
                self.simulation.model,
                self.simulation.mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            positions[joint_name] = float(
                self.simulation.data.qpos[self.simulation.model.jnt_qposadr[joint_id]]
            )
        return positions

    def note_manual_target_change(self) -> None:
        self.cancel()
        self._manual_still_since = None
        self.taught_status = "TEACHING" if self.teach_mode else "IDLE"
        self.taught_failure_reason = None

    def _update_manual_settling(self, now: float) -> None:
        if self.running or self.maximum_hand_joint_speed() > self.library.settling_velocity_threshold_rad_s:
            self._manual_still_since = None
        elif self._manual_still_since is None:
            self._manual_still_since = now

    @property
    def manual_pose_settled(self) -> bool:
        return (
            self._manual_still_since is not None
            and float(self.simulation.data.time) - self._manual_still_since
            >= self.library.settling_dwell_seconds
        )

    @property
    def selected_taught_pose(self) -> GraspPreset | None:
        if self.selected_taught_pose_name is None:
            return None
        return self.library.presets.get(self.selected_taught_pose_name)

    def set_teach_mode(self, enabled: bool) -> None:
        self.cancel()
        self.teach_mode = bool(enabled)
        self.taught_status = "TEACHING" if self.teach_mode else "IDLE"
        self.taught_failure_reason = None

    def save_taught_pose(
        self, name: str, *, target_source: str, waypoint_role: str
    ) -> GraspPreset:
        if not self.teach_mode:
            raise ValueError("Enable Teach Grasp Pose mode before saving a taught pose")
        if target_source == "actual_measured" and not self.manual_pose_settled:
            raise ValueError(
                "Actual measured pose is not settled; wait for low velocity or save actuator targets"
            )
        preset = self.library.save_taught_pose(
            name,
            self.object_controller.sphere_radius,
            tuple(float(value) for value in self.object_controller.spawn_position),
            self.current_actuator_targets(),
            self.actual_joint_positions(),
            target_source=target_source,
            waypoint_role=waypoint_role,
        )
        self.selected_preset_name = preset.name
        self.selected_taught_pose_name = preset.name
        self.taught_status = "SAVED"
        return preset

    def load_taught_pose(self, name: str) -> GraspPreset:
        preset = self.library.presets.get(name)
        if preset is None or not preset.is_taught:
            raise ValueError(f"Unknown taught pose: {name}")
        if not math.isclose(
            preset.radius_m, self.object_controller.sphere_radius, abs_tol=1e-9
        ):
            raise ValueError(
                f"Taught pose {name} expects {preset.radius_m * 1000:.0f} mm sphere"
            )
        self.selected_preset_name = name
        self.selected_taught_pose_name = name
        self.object_controller.set_spawn_position(preset.sphere_center_xyz)
        self.taught_status = "LOADED_NOT_EXECUTED"
        return preset

    def execute_taught(
        self,
        pose_names: list[str] | tuple[str, ...],
        *,
        duration_seconds: float,
        tolerance_rad: float,
    ) -> None:
        if not self.teach_mode:
            raise ValueError("Enable Teach Grasp Pose mode before execution")
        duration = float(duration_seconds)
        tolerance = float(tolerance_rad)
        if not math.isfinite(duration) or not 0.2 <= duration <= 30.0:
            raise ValueError("Taught waypoint duration must be within [0.2, 30] seconds")
        if not math.isfinite(tolerance) or not 0.001 <= tolerance <= 0.2:
            raise ValueError("Pose tolerance must be within [0.001, 0.2] radians")
        names = [str(name).strip() for name in pose_names if str(name).strip()]
        if not names and self.selected_taught_pose_name:
            names = [self.selected_taught_pose_name]
        if not names:
            raise ValueError("Select at least one taught pose")
        waypoints = []
        for name in names:
            preset = self.library.presets.get(name)
            if preset is None or not preset.is_taught:
                raise ValueError(f"Unknown taught pose: {name}")
            if not math.isclose(
                preset.radius_m, self.object_controller.sphere_radius, abs_tol=1e-9
            ):
                raise ValueError(
                    f"Taught pose {name} expects {preset.radius_m * 1000:.0f} mm sphere"
                )
            waypoints.append(preset)
        self.cancel()
        self._taught_waypoints = waypoints
        self._taught_waypoint_index = 0
        self.taught_segment_duration_seconds = duration
        self.taught_tolerance_rad = tolerance
        self._taught_segment_start_time = float(self.simulation.data.time)
        self._taught_last_update_time = self._taught_segment_start_time
        self._taught_segment_start_targets = self.current_actuator_targets()
        self._taught_settle_start_time = None
        self.selected_taught_pose_name = waypoints[-1].name
        self.selected_preset_name = waypoints[-1].name
        self.running = True
        self.execution_kind = "taught"
        self.taught_status = "MOVING"
        self.taught_failure_reason = None
        self.sequence_start_time = self._taught_segment_start_time

    def stop_taught(self) -> None:
        actual = self.actual_joint_positions()
        self.cancel()
        self.simulation.set_actuator_targets(actual)
        self.taught_status = "STOPPED"
        self.taught_failure_reason = "Stopped by user; actuators now hold the measured pose"

    def _update_taught_execution(self, now: float) -> None:
        target = self._taught_waypoints[self._taught_waypoint_index].joint_targets
        elapsed = max(0.0, now - self._taught_segment_start_time)
        fraction = min(1.0, elapsed / self.taught_segment_duration_seconds)
        eased = fraction * fraction * (3.0 - 2.0 * fraction)
        ideal = {
            name: self._taught_segment_start_targets[name]
            + eased * (target[name] - self._taught_segment_start_targets[name])
            for name in self.simulation.joint_names
        }
        current = self.current_actuator_targets()
        dt = max(float(self.simulation.model.opt.timestep), now - self._taught_last_update_time)
        maximum_step = self.taught_max_target_velocity_rad_s * dt
        bounded = {
            name: current[name]
            + max(-maximum_step, min(maximum_step, ideal[name] - current[name]))
            for name in self.simulation.joint_names
        }
        self.simulation.set_actuator_targets(bounded)
        self._taught_last_update_time = now
        actuator_target_error = max(
            abs(bounded[name] - target[name]) for name in self.simulation.joint_names
        )
        if fraction < 1.0 or actuator_target_error > 1e-6:
            return
        if self._taught_waypoint_index + 1 < len(self._taught_waypoints):
            self._taught_waypoint_index += 1
            self._taught_segment_start_time = now
            self._taught_segment_start_targets = dict(bounded)
            self.taught_status = "MOVING"
            return
        self.taught_status = "SETTLING"
        actual = self.actual_joint_positions()
        maximum_error = max(abs(actual[name] - target[name]) for name in target)
        speed = self.maximum_hand_joint_speed()
        if maximum_error <= self.taught_tolerance_rad and speed <= self.library.settling_velocity_threshold_rad_s:
            if self._taught_settle_start_time is None:
                self._taught_settle_start_time = now
            elif now - self._taught_settle_start_time >= self.library.settling_dwell_seconds:
                self.running = False
                self.execution_kind = None
                self.settled = True
                self.taught_status = "REPRODUCED"
                self.settling_time_seconds = (
                    now - self.sequence_start_time
                    if self.sequence_start_time is not None
                    else None
                )
        else:
            self._taught_settle_start_time = None
        if elapsed >= self.taught_segment_duration_seconds + self.library.reference_max_duration_seconds:
            self.running = False
            self.execution_kind = None
            self.taught_status = "NOT_REPRODUCED"
            self.taught_failure_reason = (
                f"Maximum error {maximum_error:.6f} rad exceeds tolerance "
                f"{self.taught_tolerance_rad:.6f} rad or motion did not settle"
            )

    def current_actuator_targets(self) -> dict[str, float]:
        targets = {}
        for joint_name in self.simulation.joint_names:
            actuator_id = self.simulation.mujoco.mj_name2id(
                self.simulation.model,
                self.simulation.mujoco.mjtObj.mjOBJ_ACTUATOR,
                f"servo_{joint_name}",
            )
            targets[joint_name] = float(self.simulation.data.ctrl[actuator_id])
        return targets

    def maximum_hand_joint_speed(self) -> float:
        speeds = []
        for joint_name in self.simulation.joint_names:
            joint_id = self.simulation.mujoco.mj_name2id(
                self.simulation.model,
                self.simulation.mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            speeds.append(
                abs(float(self.simulation.data.qvel[self.simulation.model.jnt_dofadr[joint_id]]))
            )
        return max(speeds, default=0.0)

    def maximum_reference_position_error(self) -> float:
        reference = self.simulation.reference_positions()
        errors = []
        for joint_name in self.simulation.joint_names:
            joint_id = self.simulation.mujoco.mj_name2id(
                self.simulation.model,
                self.simulation.mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            actual = float(
                self.simulation.data.qpos[self.simulation.model.jnt_qposadr[joint_id]]
            )
            errors.append(abs(actual - reference[joint_name]))
        return max(errors, default=0.0)

    @property
    def phase_name(self) -> str:
        if self.execution_kind == "taught" and self._taught_waypoints:
            return f"taught_{self._taught_waypoints[self._taught_waypoint_index].waypoint_role}"
        if self.teach_mode and self.selected_taught_pose_name:
            return f"taught_{self.taught_status.lower()}"
        if self.current_stage == 0:
            return "reference_open"
        if self.current_stage <= len(self.library.stages):
            name = self.library.stages[self.current_stage - 1].name
            if self.current_stage == len(self.library.stages) and self.running:
                return "final_compliant_closure_and_settling"
            return name
        return "idle"

    def save_current_targets(self, name: str) -> GraspPreset:
        targets = self.current_actuator_targets()
        preset = self.library.save_preset(
            name,
            self.object_controller.sphere_radius,
            tuple(float(value) for value in self.object_controller.spawn_position),
            targets,
        )
        self.selected_preset_name = preset.name
        return preset

    def reload(self) -> None:
        selected = self.selected_preset_name
        self.library.reload()
        self.selected_preset_name = (
            selected
            if selected in self.library.presets
            else self.library.preset_for_radius(self.object_controller.sphere_radius).name
        )
        if self.selected_taught_pose_name not in self.library.presets:
            self.selected_taught_pose_name = None
