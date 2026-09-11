"""Serve the four-quadrant dashboard from a provider-neutral robot stream."""

from __future__ import annotations

import argparse
import asyncio
from collections import deque
from dataclasses import asdict, dataclass, replace
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import hashlib
import math
import os
from pathlib import Path
import threading
import time
import webbrowser
from urllib.parse import urlsplit, unquote

import numpy as np
import yaml

from reconstruction.common_state_adapter import reconstruction_input_from_common_state
from reconstruction.contact_point_buffer import ContactPointBuffer, load_point_buffer_config
from reconstruction.contact_projection import ContactPointReconstructor
from reconstruction.sphere_fitting import SphereReconstructionSession, load_sphere_fitting_config
from reconstruction.temporal_observation import make_temporal_observations
from robot_data.common import RobotCommand, RobotCommandName
from robot_data.provider import (
    ProviderNotImplementedError,
    RobotDataProvider,
    create_robot_data_provider,
)
from robot_data.recording import CommonStateRecorder
from robot_data.hardware_provider import HardwareDataUnavailableError
from sensors.sensor_kinematics import load_sensor_mounts
from backend.visual_assets import VisualAssets, link_transform
from backend.mujoco_offscreen import LatestFrameBuffer, MuJoCoOffscreenRenderer
from simulation.experiment_recording import RepeatableSphereExperiment
from vision.research_camera import ResearchRGBCamera
from fusion.config import DEFAULT_CONFIG as FUSION_V1_CONFIG, V2_CONFIG, load_fusion_config
from fusion.observation import build_fusion_observation
from fusion.sphere import run_fusion_methods


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("dashboard_config.yaml")
UI_DIRECTORY = PROJECT_ROOT / "ui"
RECONSTRUCTION_CONFIG = PROJECT_ROOT / "experiments" / "reconstruction_config.yaml"
GENERAL_OBJECT_DASHBOARD_CASE = (
    PROJECT_ROOT / "experiments/general_object/object_sdf_v1_20260910/dashboard_case.json"
)
STATE_SCHEMA_VERSION = "1.1.0"
CONTROL_SCHEMA_VERSION = "1.1.0"
DASHBOARD_BUILD_ID = "general-object-sdf-v1-20260910.1"


def _load_general_object_dashboard_case(path: Path = GENERAL_OBJECT_DASHBOARD_CASE) -> dict:
    if not path.is_file():
        return {"status": "UNAVAILABLE", "reason": "Run python -m general_object.experiment"}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not value.get("methods") or not value.get("input", {}).get("rgb_observations"):
        return {"status": "ERROR", "reason": "General-object dashboard artifact is incomplete"}
    return {"status": "AVAILABLE", **value}


@dataclass(frozen=True)
class DashboardConfig:
    host: str
    http_port: int
    websocket_port: int
    physics_hz: float
    visualization_hz: float
    sensor_telemetry_hz: float
    plots_hz: float
    maximum_samples_per_sensor: int
    maximum_streamed_accumulated_points: int
    source: str = "simulation"
    replay_path: str | None = None
    recording_enabled: bool = False
    recording_path: str | None = None
    debug_native_viewer: bool = False
    render_mode: str = "mujoco"
    render_backend: str = "egl"
    render_width: int = 640
    render_height: int = 480
    render_fps: float = 20.0
    render_jpeg_quality: int = 85


def load_dashboard_config(path: str | Path = DEFAULT_CONFIG) -> DashboardConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError("Dashboard config must use schema_version 1")
    rates, history = raw["rates_hz"], raw["history"]
    replay, recording = raw.get("replay", {}), raw.get("recording", {})
    rendering = raw.get("mujoco_render", {})
    config = DashboardConfig(
        host=str(raw["host"]), http_port=int(raw["http_port"]),
        websocket_port=int(raw["websocket_port"]),
        physics_hz=float(rates["physics"]), visualization_hz=float(rates["visualization"]),
        sensor_telemetry_hz=float(rates["sensor_telemetry"]), plots_hz=float(rates["plots"]),
        maximum_samples_per_sensor=int(history["maximum_samples_per_sensor"]),
        maximum_streamed_accumulated_points=int(history["maximum_streamed_accumulated_points"]),
        source=str(raw.get("source", "simulation")), replay_path=replay.get("path"),
        recording_enabled=bool(recording.get("enabled", False)), recording_path=recording.get("path"),
        debug_native_viewer=bool(raw.get("debug_native_viewer", False)),
        render_mode=str(rendering.get("mode", "mujoco")),
        render_backend=str(rendering.get("backend", "egl")),
        render_width=int(rendering.get("width", 640)),
        render_height=int(rendering.get("height", 480)),
        render_fps=float(rendering.get("fps", 20)),
        render_jpeg_quality=int(rendering.get("jpeg_quality", 85)),
    )
    rates_to_check = (config.physics_hz, config.visualization_hz, config.sensor_telemetry_hz, config.plots_hz)
    if any(not math.isfinite(value) or value <= 0 for value in rates_to_check):
        raise ValueError("Dashboard rates must be finite and positive")
    if min(config.http_port, config.websocket_port, config.maximum_samples_per_sensor,
           config.maximum_streamed_accumulated_points) <= 0:
        raise ValueError("Dashboard ports/history bounds must be positive")
    if config.source not in {"simulation", "hardware", "replay"}:
        raise ValueError("source must be simulation, hardware, or replay")
    if config.source == "replay" and not config.replay_path:
        raise ValueError("source: replay requires replay.path")
    if config.recording_enabled and not config.recording_path:
        raise ValueError("Enabled recording requires recording.path")
    if config.render_mode not in {"mujoco", "mesh"}:
        raise ValueError("mujoco_render.mode must be mujoco or mesh")
    if config.render_backend not in {"egl", "glfw", "osmesa"}:
        raise ValueError("mujoco_render.backend must be egl, glfw, or osmesa")
    if not 64 <= config.render_width <= 1920 or not 64 <= config.render_height <= 1080:
        raise ValueError("MuJoCo render resolution is outside the supported range")
    if not math.isfinite(config.render_fps) or not 1 <= config.render_fps <= 60:
        raise ValueError("MuJoCo render fps must be within [1, 60]")
    if not 20 <= config.render_jpeg_quality <= 95:
        raise ValueError("MuJoCo JPEG quality must be within [20, 95]")
    return config


def _finite(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _quaternion_wxyz(rotation: np.ndarray) -> list[float]:
    trace = float(np.trace(rotation))
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2.0
        values = [0.25 * scale, (rotation[2, 1] - rotation[1, 2]) / scale,
                  (rotation[0, 2] - rotation[2, 0]) / scale,
                  (rotation[1, 0] - rotation[0, 1]) / scale]
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = math.sqrt(1 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
            values = [(rotation[2, 1] - rotation[1, 2]) / scale, 0.25 * scale,
                      (rotation[0, 1] + rotation[1, 0]) / scale,
                      (rotation[0, 2] + rotation[2, 0]) / scale]
        elif index == 1:
            scale = math.sqrt(1 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
            values = [(rotation[0, 2] - rotation[2, 0]) / scale,
                      (rotation[0, 1] + rotation[1, 0]) / scale, 0.25 * scale,
                      (rotation[1, 2] + rotation[2, 1]) / scale]
        else:
            scale = math.sqrt(1 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
            values = [(rotation[1, 0] - rotation[0, 1]) / scale,
                      (rotation[0, 2] + rotation[2, 0]) / scale,
                      (rotation[1, 2] + rotation[2, 1]) / scale, 0.25 * scale]
    return [float(value) for value in values]


class DashboardSimulation:
    """Dashboard orchestration over any RobotDataProvider implementation."""

    def __init__(self, config: DashboardConfig | None = None,
                 provider: RobotDataProvider | None = None) -> None:
        self.config = config or load_dashboard_config()
        self.provider = provider or create_robot_data_provider(
            self.config.source, replay_path=self.config.replay_path
        )
        self.description = self.provider.description
        self.visual_assets = VisualAssets(self.description)
        self.mounts = load_sensor_mounts(self.description.sensor_config_path)
        if set(self.mounts) != set(self.description.calibration_config):
            raise ValueError("Provider sensor calibration does not match sensor mounts")
        self.reconstructor = ContactPointReconstructor(self.description.kinematic_model_path)
        self.point_buffer = ContactPointBuffer(load_point_buffer_config(RECONSTRUCTION_CONFIG))
        self.sphere_session = SphereReconstructionSession(load_sphere_fitting_config(RECONSTRUCTION_CONFIG))
        self.simulation_running = True
        self.show_ground_truth = False
        self.show_estimated_sphere = True
        self.selected_sensor_id = next(iter(self.mounts))
        self.time_series_scope = "selected_sensor"
        self.current_common_state = None
        self.current_estimates = []
        self._last_sensor_update_time = -math.inf
        self._last_history_time = -math.inf
        self.history = {key: deque(maxlen=self.config.maximum_samples_per_sensor) for key in self.mounts}
        self.last_command = None
        self._recorder = (CommonStateRecorder(self.config.recording_path, self.description)
                          if self.config.recording_enabled else None)
        self._last_recorded_timestamp = -math.inf
        self._provider_session = self.provider.connection_status.get("session_id")
        self._recording_detail = ""
        self.experiment = RepeatableSphereExperiment(PROJECT_ROOT / "experiments" / "grasp_trials")
        self._object_segment_reference: tuple[np.ndarray, np.ndarray] | None = None
        self._object_motion_segment = 0
        self._object_motion_warning = ""
        self._experiment_waiting_for_approach = False
        self.render_stream: LatestFrameBuffer | None = None
        self.render_controller: MuJoCoOffscreenRenderer | None = None
        self.research_stream: LatestFrameBuffer | None = None
        self.research_camera_calibration: dict | None = None
        self.vision_result: dict = {
            "input_boundary": "clean_rgb_only",
            "segmentation": {"status": "NO_CAMERA_OBSERVATION"},
            "sphere": {"status": "NOT_RUN", "radius_label": "provided_known_radius_prior"},
            "unknown_radius_setting": {
                "status": "SCALE_AMBIGUOUS",
                "radius_label": "not_estimated",
            },
            "projected_estimated_tactile_contacts": [],
        }
        self.vision_evaluation: dict = {"available": False}
        self.fusion_config = load_fusion_config(V2_CONFIG)
        self.v1_fusion_config = load_fusion_config(FUSION_V1_CONFIG)
        self.fusion_result: dict = {
            "configuration_version": self.fusion_config["configuration_version"],
            "status": "NOT_RUN",
            "selected_method": None,
            "known_radius_m": None,
            "known_diameter_m": None,
            "radius_label": "declared_known_radius_prior",
            "methods": {},
        }
        self.general_object_result = _load_general_object_dashboard_case()
        self._last_fusion_update_wall = -math.inf
        self._update_sensor_pipeline(force=True)

    @property
    def simulation(self):
        """Compatibility accessor for existing simulation engineering tests."""
        return getattr(self.provider, "simulation")

    @property
    def objects(self):
        return getattr(self.provider, "objects")

    @property
    def grasp(self):
        return getattr(self.provider, "grasp")

    def _general_object_state(self) -> dict:
        value = json.loads(json.dumps(self.general_object_result))
        if value.get("status") != "AVAILABLE":
            return value
        truth = value.pop("evaluation_only_gt_surface_points_object_m", None)
        oracle = value.get("methods", {}).pop("rgb_perfect_contact_oracle", None)
        value.get("input", {}).pop("forbidden_fields_absent", None)
        for result in value.get("methods", {}).values():
            result.get("diagnostics", {}).pop("ground_truth_input", None)
        value["evaluation_comparison_enabled"] = bool(self.show_ground_truth)
        if self.show_ground_truth:
            value["evaluation_surface_points_object_m"] = truth
            value["methods"]["rgb_perfect_contact_oracle"] = oracle
        return value

    def close(self) -> None:
        if self._recorder:
            self._recorder.close()
        self.provider.close()

    def _joint_states(self) -> list[dict]:
        state = self.current_common_state
        if state is None:
            return []
        grasp = dict(state.grasp_state or {})
        actuator_forces = grasp.get("actuator_forces_n_m", {})
        teach = grasp.get("teach", {})
        taught_targets = teach.get("taught_target_positions_rad", {})
        errors = teach.get("position_error_by_joint_rad", {})
        causes = teach.get("failure_cause_by_joint", {})
        mapping = grasp.get("joint_mapping", {})
        return [{"name": joint.name, "angle_rad": state.joint_positions[joint.name],
                 "velocity_rad_s": state.joint_velocities[joint.name],
                 "target_rad": state.joint_targets.get(joint.name),
                 "taught_target_rad": taught_targets.get(joint.name),
                 "position_error_rad": errors.get(joint.name),
                 "tracking_cause": causes.get(joint.name, "none"),
                 "actuator_force_n_m": actuator_forces.get(joint.name),
                 "lower_limit_rad": joint.lower_limit_rad, "upper_limit_rad": joint.upper_limit_rad,
                 "parent_link": joint.parent_link, "child_link": joint.child_link,
                 "axis_xyz": mapping.get(joint.name, {}).get("axis_xyz"),
                 "actuator_name": mapping.get(joint.name, {}).get("actuator_name")}
                for joint in self.description.joints]

    def _hand_links(self) -> list[dict]:
        if self.current_common_state is None:
            return []
        if self.current_common_state.link_poses:
            return [pose.to_dict() for pose in self.current_common_state.link_poses]
        transforms = self.reconstructor.tree.forward_kinematics(self.current_common_state.joint_positions)
        parent_by_link = {joint.child: joint.parent for joint in self.reconstructor.tree.joints}
        return [{"name": name, "parent_link": parent_by_link.get(name),
                 "timestamp": self.current_common_state.timestamp,
                 "position_xyz": [float(value) for value in transform[:3, 3]],
                 "quaternion_wxyz": _quaternion_wxyz(transform[:3, :3])}
                for name, transform in transforms.items()]

    def _sensor_states(self) -> list[dict]:
        if self.current_common_state is None:
            return []
        estimates = {item.sensor_id: item for item in self.current_estimates}
        channels = {item.sensor_id: item for item in self.current_common_state.tactile_channels}
        links = {pose["name"]: link_transform(pose) for pose in self._hand_links()}
        result = []
        for sensor_id, mount in self.mounts.items():
            channel = channels[sensor_id]
            transform = links[mount.parent_link] @ mount.link_to_sensor
            rotation, estimate = transform[:3, :3], estimates.get(sensor_id)
            result.append({
                "sensor_id": sensor_id, "timestamp": channel.timestamp,
                "finger_id": mount.finger_id, "parent_link": mount.parent_link,
                "sensor_type": mount.sensor_type, "active": channel.active,
                "scalar_sensor_value_n": channel.scalar_value,
                "normal_contact_force_n": channel.force,
                "estimated_pressure_pa": channel.pressure,
                "estimated_indentation_depth_m": channel.indentation,
                "position_world_xyz": [float(value) for value in transform[:3, 3]],
                "sensing_direction_world_xyz": [float(value) for value in rotation[:, 2]],
                "surface_u_direction_world_xyz": [float(value) for value in rotation[:, 0]],
                "surface_v_direction_world_xyz": [float(value) for value in rotation[:, 1]],
                "surface_width_m": mount.surface_width_m, "surface_height_m": mount.surface_height_m,
                "estimated_contact_position_xyz": list(estimate.estimated_contact_position) if estimate else None,
            })
        return result

    def _update_sensor_pipeline(self, *, force: bool = False) -> None:
        self._refresh_provider_session()
        if self.provider.connection_status["state"] != "STREAMING":
            return
        current_time = self.provider.current_timestamp
        if (not force and self.description.source == "simulation" and
                current_time - self._last_sensor_update_time < 1.0 / self.config.sensor_telemetry_hz):
            return
        try:
            common_state = self.provider.read_common_state()
        except (HardwareDataUnavailableError, ProviderNotImplementedError):
            return
        if not force and common_state.timestamp <= self._last_sensor_update_time:
            return
        frame = reconstruction_input_from_common_state(common_state, self.description)
        estimates = self.reconstructor.reconstruct(frame)
        grasp_state = dict(common_state.grasp_state or {})
        if (
            self.experiment.active
            and self._experiment_waiting_for_approach
            and int(grasp_state.get("stage", 0)) >= 1
        ):
            self._start_accumulation()
            self._experiment_waiting_for_approach = False
        self._guard_moving_object_accumulation(common_state)
        accepted = (self.point_buffer.add(make_temporal_observations(estimates, frame.joint_state))
                    if common_state.timestamp > self._last_sensor_update_time else [])
        self.current_common_state, self.current_estimates = common_state, estimates
        if self.experiment.active:
            grasp_diagnostics = dict(common_state.grasp_state or {})
            object_state = dict(common_state.object_state or {})
            self.experiment.record_sample(
                common_state,
                accepted,
                grasp_diagnostics,
                {
                    "object_position_xyz": object_state.get("center_xyz"),
                    "object_quaternion_wxyz": object_state.get("quaternion_wxyz"),
                    "object_in_workspace": int(grasp_diagnostics.get("stage", 0)) >= 1,
                    "experiment_mode": object_state.get("experiment_mode"),
                    "contact_diagnostics": grasp_diagnostics.get("contact_diagnostics", {}),
                },
            )
        if accepted:
            self.sphere_session.maybe_auto_fit(self.point_buffer.points)
        self._last_sensor_update_time = common_state.timestamp
        if self._recorder and common_state.timestamp > self._last_recorded_timestamp:
            self._recorder.record(common_state)
            self._last_recorded_timestamp = common_state.timestamp

    def update_fusion(self, capture) -> None:
        """Run fusion through the strict observation boundary, never evaluator truth."""
        now = time.monotonic()
        if now-self._last_fusion_update_wall < 0.2:
            return
        self._last_fusion_update_wall = now
        common = self.current_common_state
        if common is None:
            return
        object_state = dict(common.object_state or {})
        fixed_pose = object_state.get("experiment_mode") == "fixed_contact_diagnostic"
        tactile = self.point_buffer.points
        if not tactile:
            frame = reconstruction_input_from_common_state(common, self.description)
            tactile = tuple(make_temporal_observations(self.current_estimates, frame.joint_state))
        if not fixed_pose:
            tactile = tuple(item for item in tactile if abs(item.timestamp-common.timestamp) <= 1e-9)
        try:
            observation = build_fusion_observation(
                rgb=capture.rgb,
                segmentation=capture.segmentation,
                calibration=capture.calibration,
                camera_observation=capture.observation,
                common_state=common,
                description=self.description,
                tactile_observations=tactile,
                known_radius_m=float(object_state["radius_m"]),
                config=self.fusion_config,
                fixed_object_pose=fixed_pose,
            )
            v1_observation = build_fusion_observation(
                rgb=capture.rgb,
                segmentation=capture.segmentation,
                calibration=capture.calibration,
                camera_observation=capture.observation,
                common_state=common,
                description=self.description,
                tactile_observations=tactile,
                known_radius_m=float(object_state["radius_m"]),
                config=self.v1_fusion_config,
                fixed_object_pose=fixed_pose,
            )
            self.fusion_result = run_fusion_methods(
                observation, capture.vision_result, self.fusion_config,
                v1_observation=v1_observation, v1_config=self.v1_fusion_config,
            )
        except Exception as exc:
            self.fusion_result = {
                "configuration_version": self.fusion_config["configuration_version"],
                "status": "ERROR",
                "selected_method": None,
                "known_radius_m": object_state.get("radius_m"),
                "known_diameter_m": (
                    2*float(object_state["radius_m"]) if object_state.get("radius_m") else None
                ),
                "radius_label": "declared_known_radius_prior",
                "methods": {},
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _guard_moving_object_accumulation(self, common_state) -> None:
        if not self.point_buffer.accumulating:
            return
        object_state = dict(common_state.object_state or {})
        if object_state.get("experiment_mode") != "free_object_grasp":
            self._object_segment_reference = None
            return
        position = object_state.get("center_xyz")
        quaternion = object_state.get("quaternion_wxyz")
        if position is None or quaternion is None:
            self.point_buffer.pause()
            self._object_motion_warning = (
                "Accumulation paused: free-object mode has no measured pose for motion segmentation"
            )
            return
        pose = (np.asarray(position, dtype=float), np.asarray(quaternion, dtype=float))
        if self._object_segment_reference is None:
            self._object_segment_reference = pose
            return
        previous_position, previous_quaternion = self._object_segment_reference
        translation = float(np.linalg.norm(pose[0] - previous_position))
        dot = float(np.clip(abs(np.dot(pose[1], previous_quaternion)), 0.0, 1.0))
        rotation = 2.0 * math.acos(dot)
        if translation > 0.002 or rotation > math.radians(5.0):
            self.point_buffer.reset()
            self.sphere_session.reset()
            self._object_motion_segment += 1
            self._object_segment_reference = pose
            self._object_motion_warning = (
                "Free-object motion exceeded 2 mm / 5 deg; tactile accumulation was reset "
                f"to segment {self._object_motion_segment}. MuJoCo pose is used only to invalidate, not reconstruct."
            )

    def _start_accumulation(self) -> None:
        self.point_buffer.start()
        self._object_segment_reference = None
        if self.current_common_state is None:
            return
        object_state = dict(self.current_common_state.object_state or {})
        if object_state.get("experiment_mode") == "free_object_grasp":
            position = object_state.get("center_xyz")
            quaternion = object_state.get("quaternion_wxyz")
            if position is not None and quaternion is not None:
                self._object_segment_reference = (
                    np.asarray(position, dtype=float),
                    np.asarray(quaternion, dtype=float),
                )

    def _sample_history(self) -> None:
        if self.current_common_state is None or self.provider.connection_status["state"] != "STREAMING":
            return
        current_time = self.current_common_state.timestamp
        if current_time <= self._last_history_time:
            return
        for channel in self.current_common_state.tactile_channels:
            self.history[channel.sensor_id].append({
                "time_seconds": channel.timestamp, "scalar_sensor_value_n": channel.scalar_value,
                "normal_contact_force_n": channel.force, "estimated_pressure_pa": channel.pressure,
                "estimated_indentation_depth_m": channel.indentation, "active": channel.active,
            })
        self._last_history_time = current_time

    def step(self) -> None:
        if self.simulation_running:
            try:
                self.provider.step()
            except ProviderNotImplementedError:
                if self.description.source != "hardware":
                    raise
            self._update_sensor_pipeline()

    def _time_series_state(self) -> dict:
        channels = {item.sensor_id: item for item in self.current_common_state.tactile_channels} if self.current_common_state else {}
        if self.time_series_scope == "selected_sensor":
            sensor_ids = [self.selected_sensor_id]
        elif self.time_series_scope == "single_finger":
            finger = self.mounts[self.selected_sensor_id].finger_id
            sensor_ids = [key for key, mount in self.mounts.items() if mount.finger_id == finger]
        else:
            sensor_ids = [key for key, channel in channels.items() if channel.active]
        return {"scope": self.time_series_scope, "selected_sensor_id": self.selected_sensor_id,
                "series": [{"sensor_id": key, "finger_id": self.mounts[key].finger_id,
                            "samples": list(self.history[key])} for key in sensor_ids],
                "maximum_samples_per_sensor": self.config.maximum_samples_per_sensor}

    def _reconstruction_state(self) -> dict:
        result, points = self.sphere_session.result, self.point_buffer.points
        positions = np.asarray([point.estimated_contact_position for point in points], dtype=float)
        spread = float(np.linalg.norm(np.ptp(positions, axis=0))) if len(positions) else 0.0
        payload = {"status": result.status.value if result else "NOT_FITTED",
                   "fit_mode": self.sphere_session.mode, "show_estimated_sphere": self.show_estimated_sphere,
                   "number_of_input_points": len(points),
                   "number_of_unique_sensors": len({point.sensor_id for point in points}),
                   "number_of_unique_fingers": len({point.finger_id for point in points}),
                   "spatial_coverage_m": spread, "estimated_center_xyz": None,
                   "estimated_radius_m": None, "fit_residual_rmse_m": None, "condition_metric": None}
        if result:
            payload.update({"number_of_input_points": result.number_of_input_points,
                            "number_of_unique_sensors": result.number_of_unique_sensors,
                            "number_of_unique_fingers": result.number_of_unique_fingers,
                            "spatial_coverage_m": result.spatial_coverage_metrics.spatial_spread_m,
                            "estimated_center_xyz": list(result.estimated_center_xyz) if result.estimated_center_xyz is not None else None,
                            "estimated_radius_m": result.estimated_radius,
                            "fit_residual_rmse_m": result.fit_residual_rmse,
                            "condition_metric": result.condition_metric})
        return payload

    def _evaluation_state(self) -> dict:
        if not self.show_ground_truth:
            return {"enabled": False}
        payload = self.provider.evaluation_payload(self.current_estimates, self.point_buffer.points,
                                                   self.sphere_session.result)
        if payload is None:
            return {"enabled": False, "unavailable_reason": "Ground truth is unavailable for this provider"}
        payload["enabled"] = True
        payload["contact_error_rmse_m"] = _finite(payload["contact_error_rmse_m"])
        return payload

    def state(self) -> dict:
        self._refresh_provider_session()
        points = self.point_buffer.points[-self.config.maximum_streamed_accumulated_points:]
        common = self.current_common_state
        object_state = dict(common.object_state or {}) if common else {}
        grasp_state = dict(common.grasp_state or {}) if common else {}
        timestamp = common.timestamp if common else None
        connection = self.provider.connection_status
        return {
            "schema_version": STATE_SCHEMA_VERSION, "message_type": "state",
            "build": {"backend": DASHBOARD_BUILD_ID},
            "server_wall_time_seconds": time.time(),
            "source": {"kind": self.description.source, "display_name": self.description.source_display_name},
            "connection": connection,
            "available_commands": list(self.provider.available_commands),
            "snapshot": {"timestamp": timestamp, "valid": common is not None and connection["state"] == "STREAMING",
                         "synchronization": dict(common.synchronization) if common else {}},
            "simulation": {"time_seconds": timestamp,
                           "running": self.simulation_running, "handedness": self.description.handedness,
                           "object": object_state,
                           "update_rates_hz": {"physics": self.config.physics_hz,
                                               "visualization": self.config.visualization_hz,
                                               "sensor_telemetry": self.config.sensor_telemetry_hz,
                                               "plots": self.config.plots_hz}, "grasp": grasp_state},
            "experiment": self.experiment.status(),
            "object_motion_guard": {
                "segment": self._object_motion_segment,
                "warning": self._object_motion_warning,
                "tracking_interface": "unavailable_measured_tracker",
            },
            "common_state_schema_version": common.schema_version if common else "1.0.0",
            "joints": self._joint_states(), "hand": {"links": self._hand_links(),
                "manifest_url": "/robot-assets/manifest.json",
                "pose_source": "provider" if common and common.link_poses else "python_fk"},
            "debug_native_viewer": self.config.debug_native_viewer,
            "render": (self.render_stream.metadata() if self.render_stream else {
                "mode": "Mesh Digital Twin", "status": "AVAILABLE",
                "error": None, "frame_sequence": None,
                "simulation_timestamp": None, "rendering_timestamp": None,
                "source": self.description.source, "stream_url": None,
            }),
            "sensors": self._sensor_states(), "selected_sensor_id": self.selected_sensor_id,
            "current_estimated_contacts": [{"sensor_id": item.sensor_id, "finger_id": item.finger_id,
                                            "position_xyz": list(item.estimated_contact_position)}
                                           for item in self.current_estimates],
            "accumulation": {"running": self.point_buffer.accumulating,
                             "raw_observation_count": self.point_buffer.statistics.raw_observation_count,
                             "accepted_point_count": self.point_buffer.statistics.accepted_point_count,
                             "rejected_duplicate_count": self.point_buffer.statistics.rejected_duplicate_count,
                             "rejected_pressure_count": self.point_buffer.statistics.rejected_pressure_count},
            "accumulated_points": [{"timestamp": point.timestamp, "sensor_id": point.sensor_id,
                                    "finger_id": point.finger_id,
                                    "position_xyz": list(point.estimated_contact_position),
                                    "estimated_pressure_pa": point.estimated_pressure} for point in points],
            "time_series": self._time_series_state(), "sphere_reconstruction": self._reconstruction_state(),
            "evaluation": self._evaluation_state(),
            "camera": common.camera.to_dict() if common and common.camera else None,
            "research_camera_calibration": self.research_camera_calibration,
            "research_camera_debug_visible": bool(
                self.render_controller
                and self.render_controller.show_research_camera_frustum
            ),
            "research_rgb": (
                self.research_stream.metadata()
                if self.research_stream
                else {
                    "mode": "Research RGB Camera",
                    "status": "UNAVAILABLE",
                    "error": "Calibrated research RGB is unavailable for this source",
                    "stream_url": None,
                }
            ),
            "vision_only": self.vision_result,
            "vision_evaluation": self.vision_evaluation,
            "fusion": self.fusion_result,
            "general_object_reconstruction": self._general_object_state(),
            "recording": {"enabled": self._recorder is not None,
                          "path": str(self._recorder.path) if self._recorder else None,
                          "detail": self._recording_detail},
            "last_command": self.last_command,
        }

    def _reset_experiment_data(self) -> None:
        self.point_buffer.reset(); self.point_buffer.pause(); self.sphere_session.reset()
        for samples in self.history.values():
            samples.clear()
        self._last_history_time = -math.inf
        self._object_segment_reference = None
        self._object_motion_segment = 0
        self._object_motion_warning = ""

    @staticmethod
    def _file_version(path: str | Path) -> str:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]

    def _experiment_metadata(self) -> dict:
        if self.description.source != "simulation":
            raise ValueError("Named grasp experiments currently require the simulation provider")
        common = self.current_common_state
        object_state = dict(common.object_state or {}) if common else {}
        grasp_state = dict(common.grasp_state or {}) if common else {}
        model = self.simulation.model
        return {
            "sphere_radius_m": object_state.get("radius_m"),
            "grasp_preset_name": grasp_state.get("preset_name"),
            "grasp_preset_version": grasp_state.get("preset_version"),
            "sensor_calibration_version": self._file_version(self.description.sensor_config_path),
            "pressure_calibration_version": self._file_version(PROJECT_ROOT / "sensors" / "pressure_config.yaml"),
            "joint_configuration_version": self._file_version(PROJECT_ROOT / "simulation" / "joint_config_right.yaml"),
            "experiment_mode": object_state.get("experiment_mode"),
            "simulation_configuration": {
                "model_path": str(self.simulation.model_path),
                "timestep_seconds": float(model.opt.timestep),
                "gravity_m_s2": [float(value) for value in model.opt.gravity],
                "integrator": int(model.opt.integrator),
                "actuator_kp": {
                    name: float(model.actuator_gainprm[index, 0])
                    for index, name in enumerate(self.simulation.joint_names)
                },
                "actuator_force_ranges_n_m": {
                    name: [float(value) for value in model.actuator_forcerange[index]]
                    for index, name in enumerate(self.simulation.joint_names)
                },
            },
            "reconstruction_inputs": "named_joint_state + 18 scalar_tactile_channels + calibration",
            "object_pose_policy": (
                "fixed known pose" if object_state.get("experiment_mode") == "fixed_contact_diagnostic"
                else "MuJoCo GT invalidates/segments accumulation only; no pose enters reconstruction"
            ),
        }

    def _refresh_provider_session(self):
        session = self.provider.connection_status.get("session_id")
        if session == self._provider_session:
            return
        self._provider_session = session
        # A new source epoch must not mix with previous geometry or a monotonic log.
        if self.current_common_state is not None and self._recorder is not None:
            self._recorder.close()
            self._recorder = None
            self._recording_detail = "Recording stopped on hardware session change; use a new log for the new epoch"
        self._reset_experiment_data()
        self.current_common_state = None
        self.current_estimates = []
        self._last_sensor_update_time = -math.inf

    def handle_command(self, message: dict) -> dict:
        if not isinstance(message, dict):
            raise ValueError("Control message must be an object")
        if message.get("schema_version") not in {CONTROL_SCHEMA_VERSION, "1.0.0"}:
            raise ValueError(f"Control schema_version must be {CONTROL_SCHEMA_VERSION}")
        command, parameters = str(message.get("command", "")), message.get("parameters") or {}
        if not isinstance(parameters, dict):
            raise ValueError("Control parameters must be an object")
        if command == "sphere_grasp":
            command = "grasp"
        elif command == "set_sphere_radius":
            command, parameters = "set_grasp_preset", {"radius_m": parameters["radius_m"]}
        if command in {item.value for item in RobotCommandName}:
            if command not in self.provider.available_commands:
                raise ValueError(f"READ_ONLY: {command} is not enabled for {self.description.source}")
            if command == "grasp" and self.experiment.active:
                self.point_buffer.reset()
                self.point_buffer.pause()
                self.sphere_session.reset()
                self._experiment_waiting_for_approach = True
            self.provider.send_command(RobotCommand(RobotCommandName(command), parameters))
            if command in {
                "reset", "reset_reference_pose", "set_grasp_preset",
                "set_experiment_mode", "set_pose_validation_mode",
            }:
                self._reset_experiment_data()
            if command == "reset" and self.description.source == "replay":
                self._last_sensor_update_time = -math.inf
            self._update_sensor_pipeline(force=True)
        elif command in {"start_simulation", "pause_simulation"}:
            if self.description.source == "hardware":
                raise ValueError("Read-only hardware reception cannot be paused by simulation controls")
            self.simulation_running = command == "start_simulation"
        elif command == "render_camera":
            if self.render_controller is None:
                raise ValueError("MuJoCo Render camera is unavailable")
            self.render_controller.camera_command(parameters)
        elif command == "start_experiment":
            name = str(parameters.get("experiment_name", ""))
            self._reset_experiment_data()
            if "reset_reference_pose" not in self.provider.available_commands:
                raise ValueError("Named grasp experiments require controllable simulation")
            self.provider.send_command(
                RobotCommand(RobotCommandName.RESET_REFERENCE_POSE, {})
            )
            self._update_sensor_pipeline(force=True)
            self.experiment.start(name, self._experiment_metadata())
            self.point_buffer.pause()
            self._experiment_waiting_for_approach = True
        elif command == "finish_experiment":
            if self.sphere_session.result is None:
                self.sphere_session.fit(self.point_buffer.points)
            evaluation = self.provider.evaluation_payload(
                self.current_estimates, self.point_buffer.points, self.sphere_session.result
            ) or {"available": False}
            result = self.experiment.finish(self._reconstruction_state(), evaluation)
            self.point_buffer.pause()
            self._experiment_waiting_for_approach = False
            self._recording_detail = f"Saved named trial: {result['reconstruction_path']}"
        elif command == "repeat_experiment":
            if not self.experiment.name:
                raise ValueError("No previous named experiment is available to repeat")
            name = self.experiment.name
            metadata = dict(self.experiment.metadata)
            radius = float(metadata["sphere_radius_m"])
            preset_name = str(metadata["grasp_preset_name"])
            mode = str(metadata["experiment_mode"])
            self.provider.send_command(RobotCommand(RobotCommandName.SET_EXPERIMENT_MODE, {"mode": mode}))
            self.provider.send_command(RobotCommand(RobotCommandName.SET_GRASP_PRESET, {"radius_m": radius}))
            if preset_name != f"sphere_{round(radius * 1000)}mm":
                self.provider.send_command(RobotCommand(RobotCommandName.LOAD_GRASP_PRESET, {"preset_name": preset_name}))
            self.provider.send_command(RobotCommand(RobotCommandName.RESET_REFERENCE_POSE, {}))
            self._reset_experiment_data()
            self._update_sensor_pipeline(force=True)
            self.experiment.start(name, self._experiment_metadata())
            self.point_buffer.pause()
            self._experiment_waiting_for_approach = True
        elif command == "start_accumulation":
            self._experiment_waiting_for_approach = False
            self._start_accumulation()
        elif command == "pause_accumulation": self.point_buffer.pause()
        elif command == "reset_points": self.point_buffer.reset(); self.point_buffer.pause()
        elif command == "fit_sphere": self.sphere_session.fit(self.point_buffer.points)
        elif command == "reset_fit": self.sphere_session.reset()
        elif command == "show_estimated_sphere": self.show_estimated_sphere = bool(parameters["enabled"])
        elif command == "show_ground_truth": self.show_ground_truth = bool(parameters["enabled"])
        elif command == "select_sensor":
            sensor_id = str(parameters["sensor_id"])
            if sensor_id not in self.mounts: raise ValueError(f"Unknown sensor_id: {sensor_id}")
            self.selected_sensor_id = sensor_id
        elif command == "set_time_series_scope":
            scope = str(parameters["scope"])
            if scope not in {"selected_sensor", "single_finger", "active_sensors"}:
                raise ValueError(f"Unknown time-series scope: {scope}")
            self.time_series_scope = scope
        else:
            raise ValueError(f"Unknown dashboard command: {command}")
        result = getattr(self.provider, "last_command_result", None)
        if command in {"finish_experiment", "start_experiment", "repeat_experiment"}:
            result = self.experiment.status()
        self.last_command = {"command": command, "request_id": message.get("request_id"), "accepted": True,
                             "result": result}
        return self.last_command


class QuietStaticHandler(SimpleHTTPRequestHandler):
    def __init__(
        self,
        *args,
        visual_assets=None,
        render_stream=None,
        research_stream=None,
        research_camera=None,
        **kwargs,
    ):
        self.visual_assets = visual_assets
        self.render_stream = render_stream
        self.research_stream = research_stream
        self.research_camera = research_camera
        super().__init__(*args, **kwargs)

    def end_headers(self):
        # The dashboard is an engineering UI served from a long-running local
        # process. Never let an existing tab reuse an older HTML/JS build.
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("X-Dashboard-Build", DASHBOARD_BUILD_ID)
        super().end_headers()

    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        stream = {
            "/mujoco-render/stream.mjpg": self.render_stream,
            "/research-rgb/stream.mjpg": self.research_stream,
        }.get(path)
        if path in {"/mujoco-render/stream.mjpg", "/research-rgb/stream.mjpg"}:
            if stream is None:
                self.send_error(404, "Requested image stream is unavailable for this source")
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Connection", "close")
            self.end_headers()
            sequence = -1
            try:
                while True:
                    frame = stream.wait_after(sequence)
                    if frame is None:
                        if stream.error:
                            return
                        continue
                    sequence = frame.sequence
                    headers = (
                        b"--frame\r\nContent-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(frame.jpeg)}\r\n".encode()
                        + f"X-Frame-Sequence: {frame.sequence}\r\n".encode()
                        + f"X-Simulation-Timestamp: {frame.simulation_timestamp:.9f}\r\n".encode()
                        + f"X-Rendering-Timestamp: {frame.rendering_timestamp:.9f}\r\n".encode()
                        + f"X-Data-Source: {frame.source}\r\n".encode()
                        + f"X-Dashboard-Build: {DASHBOARD_BUILD_ID}\r\n\r\n".encode()
                    )
                    self.wfile.write(headers + frame.jpeg + b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return
        if path == "/research-rgb/latest.jpg":
            frame = self.research_stream.latest if self.research_stream else None
            if frame is None:
                self.send_error(503, "Research RGB frame is not ready")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(frame.jpeg)))
            self.send_header("X-Acquisition-Timestamp", f"{frame.simulation_timestamp:.9f}")
            self.end_headers()
            self.wfile.write(frame.jpeg)
            return
        if path == "/research-rgb/calibration.json":
            if self.research_camera is None:
                self.send_error(404, "Research camera calibration is unavailable")
                return
            payload = json.dumps(
                self.research_camera.calibration().to_dict(), allow_nan=False
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if path.startswith("/robot-assets/"):
            if self.visual_assets is None:
                self.send_error(404)
                return
            if path == "/robot-assets/manifest.json":
                payload = json.dumps(self.visual_assets.manifest).encode()
                mime = "application/json"
            elif path in self.visual_assets.files:
                payload = self.visual_assets.files[path].read_bytes()
                mime = "model/stl"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(payload)
            return
        super().do_GET()

    def log_message(self, format, *args): return


def start_static_server(
    config: DashboardConfig,
    visual_assets=None,
    render_stream=None,
    research_stream=None,
    research_camera=None,
) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((config.host, config.http_port),
                                 partial(QuietStaticHandler, directory=str(UI_DIRECTORY),
                                         visual_assets=visual_assets,
                                         render_stream=render_stream,
                                         research_stream=research_stream,
                                         research_camera=research_camera))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


async def run_servers(config: DashboardConfig) -> None:
    try:
        from websockets.asyncio.server import serve
    except ImportError as exc:
        raise RuntimeError("Install dashboard dependencies with: pip install -r requirements.txt") from exc
    if config.source in {"simulation", "replay"}:
        os.environ["MUJOCO_GL"] = config.render_backend
    engine, clients = DashboardSimulation(config), {}
    render_stream = None
    frame_renderer = None
    research_stream = None
    research_camera = None
    if config.render_mode == "mujoco" and engine.description.source in {"simulation", "replay"}:
        render_stream = LatestFrameBuffer()
        engine.render_stream = render_stream
        try:
            frame_renderer = MuJoCoOffscreenRenderer(
                engine.provider,
                width=config.render_width,
                height=config.render_height,
                jpeg_quality=config.render_jpeg_quality,
            )
            engine.render_controller = frame_renderer
        except Exception as exc:
            render_stream.fail(f"{type(exc).__name__}: {exc}")
    if engine.description.source in {"simulation", "replay"}:
        research_stream = LatestFrameBuffer(
            mode="Research RGB Camera", stream_url="/research-rgb/stream.mjpg"
        )
        engine.research_stream = research_stream
        try:
            research_camera = ResearchRGBCamera(engine.provider)
        except Exception as exc:
            research_stream.fail(f"{type(exc).__name__}: {exc}")
    if config.debug_native_viewer:
        if engine.description.source != "simulation":
            engine.close()
            raise ValueError("Native comparison view requires source simulation")
        engine.provider.open_debug_viewer()

    async def send_locked(connection, payload: str) -> None:
        lock = clients.get(connection)
        if lock is not None:
            async with lock: await connection.send(payload)

    async def handler(connection) -> None:
        clients[connection] = asyncio.Lock()
        try:
            await send_locked(connection, json.dumps(engine.state(), allow_nan=False))
            async for raw in connection:
                try:
                    acknowledgement = engine.handle_command(json.loads(raw))
                    response = {"schema_version": CONTROL_SCHEMA_VERSION,
                                "message_type": "command_acknowledgement", **acknowledgement}
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    response = {"schema_version": CONTROL_SCHEMA_VERSION, "message_type": "command_error",
                                "request_id": None, "error": str(exc)}
                await send_locked(connection, json.dumps(response, allow_nan=False))
        finally:
            clients.pop(connection, None)

    async def physics_loop() -> None:
        interval, deadline = 1.0 / config.physics_hz, time.monotonic()
        while True:
            if engine.provider.poll_interval_seconds is not None:
                await asyncio.sleep(engine.provider.poll_interval_seconds)
                engine.step()
                continue
            engine.step(); deadline += interval; delay = deadline - time.monotonic()
            if delay < -0.1: deadline, delay = time.monotonic(), 0.0
            await asyncio.sleep(max(0.0, delay))

    async def history_loop() -> None:
        while True: engine._sample_history(); await asyncio.sleep(1.0 / config.plots_hz)

    async def broadcast_loop() -> None:
        while True:
            if config.debug_native_viewer:
                engine.provider.sync_debug_viewer()
            if clients:
                payload, current = json.dumps(engine.state(), separators=(",", ":"), allow_nan=False), tuple(clients)
                results = await asyncio.gather(*(send_locked(client, payload) for client in current),
                                               return_exceptions=True)
                for client, result in zip(current, results):
                    if isinstance(result, Exception): clients.pop(client, None)
            await asyncio.sleep(1.0 / config.visualization_hz)

    async def render_loop() -> None:
        nonlocal frame_renderer, research_camera
        if frame_renderer is None and research_camera is None:
            while True:
                await asyncio.sleep(1.0)
        interval, deadline = 1.0 / config.render_fps, time.monotonic()
        while True:
            if frame_renderer is not None:
                try:
                    render_stream.publish(frame_renderer.render_frame(engine))
                except Exception as exc:
                    render_stream.fail(f"{type(exc).__name__}: {exc}")
                    frame_renderer.close()
                    frame_renderer = None
            if research_camera is not None:
                try:
                    capture = research_camera.capture(engine)
                    engine.research_camera_calibration = capture.calibration.to_dict()
                    research_stream.publish(capture.frame)
                    engine.vision_result = capture.vision_result
                    engine.vision_evaluation = {"available": True, **capture.evaluation}
                    engine.update_fusion(capture)
                    if engine.description.source == "simulation" and hasattr(
                        engine.provider, "publish_camera_observation"
                    ):
                        engine.provider.publish_camera_observation(capture.observation)
                    if (
                        engine.description.source == "simulation"
                        and engine.current_common_state is not None
                    ):
                        engine.current_common_state = replace(
                            engine.current_common_state, camera=capture.observation
                        )
                except Exception as exc:
                    research_stream.fail(f"{type(exc).__name__}: {exc}")
                    research_camera.close()
                    research_camera = None
            deadline += interval
            await asyncio.sleep(max(0.0, deadline - time.monotonic()))

    static_server = start_static_server(
        config,
        engine.visual_assets,
        render_stream,
        research_stream,
        research_camera,
    )
    print(f"Dashboard source: {engine.description.source_display_name}")
    print(f"Dashboard: http://{config.host}:{config.http_port}")
    print(f"WebSocket: ws://{config.host}:{config.websocket_port}")
    try:
        async with serve(handler, config.host, config.websocket_port, max_size=8 * 1024 * 1024):
            await asyncio.gather(physics_loop(), history_loop(), broadcast_loop(), render_loop())
    finally:
        if frame_renderer is not None:
            frame_renderer.close()
        if research_camera is not None:
            research_camera.close()
        engine.close(); static_server.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--host"); parser.add_argument("--http-port", type=int)
    parser.add_argument("--websocket-port", type=int)
    parser.add_argument("--source", choices=("simulation", "hardware", "replay"))
    parser.add_argument("--replay-log", type=Path); parser.add_argument("--record-log", type=Path)
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--debug-native-viewer", action="store_true", help="Open a separate native comparison view; disabled by default")
    parser.add_argument("--render-mode", choices=("mujoco", "mesh"))
    parser.add_argument("--render-backend", choices=("egl", "glfw", "osmesa"))
    args = parser.parse_args()
    config = load_dashboard_config(args.config)
    overrides = asdict(config)
    overrides.update({"host": args.host or config.host, "http_port": args.http_port or config.http_port,
                      "websocket_port": args.websocket_port or config.websocket_port,
                      "source": args.source or config.source,
                      "replay_path": str(args.replay_log) if args.replay_log else config.replay_path,
                      "recording_enabled": bool(args.record_log) or config.recording_enabled,
                      "recording_path": str(args.record_log) if args.record_log else config.recording_path,
                      "debug_native_viewer": args.debug_native_viewer or config.debug_native_viewer})
    overrides.update({"render_mode": args.render_mode or config.render_mode,
                      "render_backend": args.render_backend or config.render_backend})
    config = DashboardConfig(**overrides)
    if args.open_browser:
        threading.Timer(
            0.8,
            lambda: webbrowser.open(
                f"http://{config.host}:{config.http_port}/?build={DASHBOARD_BUILD_ID}"
            ),
        ).start()
    try:
        asyncio.run(run_servers(config))
    except KeyboardInterrupt:
        print("Dashboard stopped")
    except ProviderNotImplementedError as exc:
        raise SystemExit(f"Dashboard source {config.source!r} is not ready: {exc}") from None


if __name__ == "__main__": main()
