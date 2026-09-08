"""MuJoCo-only adapter for the provider-neutral robot data contract."""

from __future__ import annotations

from typing import Any

import numpy as np

from evaluation.contact_evaluator import collect_mujoco_ground_truth, evaluate_contact_points
from evaluation.sphere_evaluator import collect_mujoco_sphere_ground_truth, evaluate_sphere_reconstruction
from robot_data.common import (
    CameraObservation,
    JointInfo,
    JointStateSample,
    LinkPose,
    ProviderDescription,
    RobotCommand,
    RobotCommandName,
    TactileChannelSample,
    TactileStateSample,
)
from robot_data.provider import RobotDataProvider
from robot_data.synchronization import TimestampSynchronizer
from sensors.pressure_emulator import PressureSensorEmulator
from sensors.registry_audit import require_approved_right_sensor_registry
from sensors.sensor_kinematics import load_sensor_mounts
from simulation.grasp_presets import GraspPresetLibrary, SphereGraspController
from simulation.model_builder import HAND_VARIANTS, build_hand_variant
from simulation.mujoco_sim import PRESSURE_CONFIG, RIGHT_GRASP_CONFIG, HandSimulation, ObjectController


class MuJoCoRobotDataProvider(RobotDataProvider):
    @property
    def available_commands(self):
        return tuple(command.value for command in RobotCommandName)

    def __init__(self) -> None:
        variant = HAND_VARIANTS["right"]
        build_hand_variant("right")
        self.simulation = HandSimulation(variant.output)
        self.mounts = load_sensor_mounts(variant.sensors)
        require_approved_right_sensor_registry(self.mounts, self.simulation.sensor_site_names)
        self.objects = ObjectController(self.simulation, variant.sphere_position)
        self.objects.activate("sphere")
        self.objects.set_experiment_mode(ObjectController.FIXED_CONTACT_DIAGNOSTIC)
        self.pressure = PressureSensorEmulator(self.simulation, self.mounts, PRESSURE_CONFIG)
        library = GraspPresetLibrary(RIGHT_GRASP_CONFIG, variant.urdf, variant.joints)
        self.grasp = SphereGraspController(self.simulation, self.objects, library)
        self.grasp.select_for_radius(self.objects.sphere_radius)
        self._synchronizer = TimestampSynchronizer()
        self._description = self._make_description(variant)
        self._synchronizer.add_joint_state(self.read_joint_state())
        self._pose_data = self.simulation.mujoco.MjData(self.simulation.model)
        self._debug_viewer = None
        self.last_command_result: dict[str, Any] | None = None
        self._camera_observation: CameraObservation | None = None

    def publish_camera_observation(self, observation: CameraObservation) -> None:
        """Attach a rendered acquisition without changing its source timestamp."""
        if observation.source != "simulation":
            raise ValueError("Simulation provider accepts simulation camera observations only")
        if observation.timestamp > self.current_timestamp + 1e-9:
            raise ValueError("Camera acquisition timestamp cannot be in the simulation future")
        self._camera_observation = observation

    def _link_poses(self):
        simulation = self.simulation
        mj, model = simulation.mujoco, simulation.model
        # mj_step integrates coordinates after its position stage. Use a separate
        # data buffer for exact snapshot-time body poses without disturbing the
        # solver/contact data or advancing physics.
        self._pose_data.qpos[:] = simulation.data.qpos
        mj.mj_kinematics(model, self._pose_data)
        parents = {joint.child_link: joint.parent_link for joint in self.description.joints}
        names = ("Palm", *parents)
        return tuple(LinkPose(name, parents.get(name), self.current_timestamp,
                              tuple(self._pose_data.xpos[model.body(name).id]),
                              tuple(self._pose_data.xquat[model.body(name).id])) for name in names)

    def open_debug_viewer(self):
        """Opt-in separate native comparison view of this same live simulation."""
        import mujoco.viewer
        if self._debug_viewer is None:
            self._debug_viewer = mujoco.viewer.launch_passive(self.simulation.model, self.simulation.data)

    def sync_debug_viewer(self):
        if self._debug_viewer is not None and self._debug_viewer.is_running():
            self._debug_viewer.sync()

    def close(self):
        if self._debug_viewer is not None:
            self._debug_viewer.close()
            self._debug_viewer = None

    def _make_description(self, variant) -> ProviderDescription:
        joints = []
        for name in self.simulation.joint_names:
            details = self.simulation.joint_details(name)
            joints.append(
                JointInfo(
                    name=name,
                    lower_limit_rad=details["lower"],
                    upper_limit_rad=details["upper"],
                    parent_link=details["parent"],
                    child_link=details["child"],
                )
            )
        calibration = {
            sensor_id: {
                "effective_sensor_area_m2": config.effective_sensor_area_m2,
                "virtual_stiffness_n_per_m": config.virtual_stiffness_n_per_m,
                "minimum_activation_force_n": config.minimum_activation_force_n,
                "saturation_pressure_pa": config.saturation_pressure_pa,
                "contact_projection_sign": config.contact_projection_sign,
            }
            for sensor_id, config in self.pressure.configs.items()
        }
        return ProviderDescription(
            source="simulation",
            source_display_name="SIMULATION",
            handedness="right",
            kinematic_model_path=str(variant.urdf),
            sensor_config_path=str(variant.sensors),
            calibration_config=calibration,
            joints=tuple(joints),
        )

    @property
    def description(self) -> ProviderDescription:
        return self._description

    @property
    def current_timestamp(self) -> float:
        return float(self.simulation.data.time)

    def read_joint_state(self) -> JointStateSample:
        positions, velocities = {}, {}
        for name in self.simulation.joint_names:
            joint_id = self.simulation.mujoco.mj_name2id(
                self.simulation.model, self.simulation.mujoco.mjtObj.mjOBJ_JOINT, name
            )
            positions[name] = float(
                self.simulation.data.qpos[self.simulation.model.jnt_qposadr[joint_id]]
            )
            velocities[name] = float(
                self.simulation.data.qvel[self.simulation.model.jnt_dofadr[joint_id]]
            )
        return JointStateSample(self.current_timestamp, positions, velocities)

    def read_tactile_state(self) -> TactileStateSample:
        timestamp = self.current_timestamp
        readings = self.pressure.read()
        return TactileStateSample(
            timestamp,
            tuple(
                TactileChannelSample(
                    timestamp=timestamp,
                    sensor_id=sensor_id,
                    scalar_value=reading.scalar_output_n,
                    pressure=reading.pressure_pa,
                    active=reading.active,
                    force=reading.normal_contact_force_n,
                    indentation=reading.indentation_m,
                )
                for sensor_id, reading in readings.items()
            ),
        )

    def _joint_targets(self) -> dict[str, float]:
        targets = {}
        for name in self.simulation.joint_names:
            actuator_id = self.simulation.mujoco.mj_name2id(
                self.simulation.model,
                self.simulation.mujoco.mjtObj.mjOBJ_ACTUATOR,
                f"servo_{name}",
            )
            targets[name] = float(self.simulation.data.ctrl[actuator_id])
        return targets

    def _object_state(self) -> dict[str, Any]:
        _, _, geometry_id = self.objects._ids("sphere")
        body_id = self.simulation.model.geom_bodyid[geometry_id]
        return {
            "shape": "sphere",
            "present": self.objects.active_shape == "sphere",
            "center_xyz": [float(value) for value in self._pose_data.geom_xpos[geometry_id]],
            "quaternion_wxyz": [float(value) for value in self._pose_data.xquat[body_id]],
            "radius_m": self.objects.sphere_radius,
            "experiment_mode": self.objects.experiment_mode,
            "pose_source": "simulation_evaluation_pose_only",
        }

    def _actuator_forces(self) -> dict[str, float]:
        result = {}
        for name in self.simulation.joint_names:
            actuator_id = self.simulation.mujoco.mj_name2id(
                self.simulation.model,
                self.simulation.mujoco.mjtObj.mjOBJ_ACTUATOR,
                f"servo_{name}",
            )
            result[name] = float(self.simulation.data.actuator_force[actuator_id])
        return result

    def _contact_diagnostics(self) -> dict[str, Any]:
        sphere_geom = self.objects._ids("sphere")[2]
        distances = []
        normal_forces = []
        wrench = np.zeros(6, dtype=float)
        for index in range(self.simulation.data.ncon):
            contact = self.simulation.data.contact[index]
            if sphere_geom not in (contact.geom1, contact.geom2):
                continue
            distances.append(float(contact.dist))
            self.simulation.mujoco.mj_contactForce(
                self.simulation.model, self.simulation.data, index, wrench
            )
            normal_forces.append(max(0.0, float(wrench[0])))
        minimum_distance = min(distances, default=0.0)
        return {
            "sphere_contact_count": len(distances),
            "minimum_contact_distance_m": minimum_distance,
            "maximum_penetration_m": max(0.0, -minimum_distance),
            "total_normal_contact_force_n": sum(normal_forces),
        }

    def _grasp_state(self, tactile: TactileStateSample) -> dict[str, Any]:
        active = [item for item in tactile.channels if item.active]
        active_fingers = {self.mounts[item.sensor_id].finger_id for item in active}
        taught = self.grasp.selected_taught_pose
        taught_target = dict(taught.joint_targets) if taught else {}
        actual = self.grasp.actual_joint_positions()
        current_targets = self.grasp.current_actuator_targets()
        forces = self._actuator_forces()
        contact = self._contact_diagnostics()
        failure_causes = {}
        for name, target in taught_target.items():
            joint_id = self.simulation.model.joint(name).id
            actuator_id = self.simulation.model.actuator(f"servo_{name}").id
            error = target - actual[name]
            lower, upper = self.simulation.model.jnt_range[joint_id]
            force_limit = max(abs(value) for value in self.simulation.model.actuator_forcerange[actuator_id])
            if abs(error) <= self.grasp.taught_tolerance_rad:
                cause = "none"
            elif min(abs(target - lower), abs(target - upper)) <= 1e-4:
                cause = "joint_limit"
            elif abs(forces[name]) >= 0.95 * force_limit:
                cause = "actuator_force_limit"
            elif self.objects.active_shape == "sphere" and contact["sphere_contact_count"]:
                cause = "collision_obstruction"
            elif abs(float(self.simulation.data.qvel[self.simulation.model.jnt_dofadr[joint_id]])) > self.grasp.library.settling_velocity_threshold_rad_s:
                cause = "motion_not_settled"
            else:
                cause = "controller_tracking_error"
            failure_causes[name] = cause
        return {
            "preset_name": self.grasp.selected_preset_name,
            "preset_names": sorted(self.grasp.library.presets),
            "preset_version": self.grasp.library.config_version,
            "stage": self.grasp.current_stage,
            "stage_count": len(self.grasp.library.stages),
            "phase_name": self.grasp.phase_name,
            "running": self.grasp.running,
            "settled": self.grasp.settled,
            "settling_time_seconds": self.grasp.settling_time_seconds,
            "maximum_joint_velocity_rad_s": self.grasp.maximum_hand_joint_speed(),
            "maximum_reference_position_error_rad": self.grasp.maximum_reference_position_error(),
            "reference_timed_out": self.grasp.reference_timed_out,
            "actuator_forces_n_m": forces,
            "active_sensor_ids": [item.sensor_id for item in active],
            "active_finger_count": len(active_fingers),
            "normal_contact_force_by_sensor_n": {
                item.sensor_id: float(item.force or 0.0) for item in tactile.channels
            },
            "contact_diagnostics": contact,
            "teach": {
                "enabled": self.grasp.teach_mode,
                "validation_mode": self.grasp.validation_mode,
                "pose_is_settled": self.grasp.manual_pose_settled,
                "selected_pose_name": self.grasp.selected_taught_pose_name,
                "pose_names": sorted(
                    name for name, preset in self.grasp.library.presets.items()
                    if preset.is_taught
                ),
                "poses": [
                    {
                        "name": preset.name,
                        "radius_m": preset.radius_m,
                        "target_source": preset.target_source,
                        "waypoint_role": preset.waypoint_role,
                    }
                    for preset in self.grasp.library.presets.values()
                    if preset.is_taught
                ],
                "taught_target_positions_rad": taught_target,
                "saved_actuator_targets_rad": dict(taught.actuator_targets) if taught else {},
                "saved_measured_positions_rad": dict(taught.measured_joint_positions) if taught else {},
                "current_actuator_targets_rad": current_targets,
                "position_error_by_joint_rad": {
                    name: taught_target[name] - actual[name] for name in taught_target
                },
                "failure_cause_by_joint": failure_causes,
                "execution_status": self.grasp.taught_status,
                "failure_reason": self.grasp.taught_failure_reason,
                "tolerance_rad": self.grasp.taught_tolerance_rad,
                "waypoint_index": self.grasp._taught_waypoint_index,
                "waypoint_count": len(self.grasp._taught_waypoints),
                "pose_reproduction_validated": (
                    self.grasp.taught_status == "REPRODUCED"
                    and self.grasp.validation_mode == "pose_reproduction_no_object"
                ),
                "physical_grasp_evaluated_separately": (
                    self.grasp.validation_mode == "physical_sphere_grasp"
                ),
            },
            "joint_mapping": {
                name: {
                    "axis_xyz": [float(value) for value in self.simulation.model.jnt_axis[self.simulation.model.joint(name).id]],
                    "actuator_name": f"servo_{name}",
                    "actuator_force_range_n_m": [
                        float(value)
                        for value in self.simulation.model.actuator_forcerange[
                            self.simulation.model.actuator(f"servo_{name}").id
                        ]
                    ],
                }
                for name in self.simulation.joint_names
            },
        }

    def step(self) -> None:
        self.objects.maintain_inactive()
        self.simulation.step()
        self.objects.enforce_fixed_diagnostic_pose()
        self.grasp.update()
        self._synchronizer.add_joint_state(self.read_joint_state())

    def read_common_state(self):
        # MuJoCo samples are synchronous today, but they intentionally pass through
        # the same interpolation boundary required by future hardware.
        poses = self._link_poses()
        joints = self.read_joint_state()
        self._synchronizer.add_joint_state(joints)
        tactile = self.read_tactile_state()
        return self._synchronizer.synchronize(
            tactile,
            source=self.description.source,
            joint_targets=self._joint_targets(),
            object_state=self._object_state(),
            grasp_state=self._grasp_state(tactile),
            camera=self._camera_observation,
            time_base="simulation_seconds",
            link_poses=poses,
        )

    def send_joint_targets(self, targets) -> None:
        unknown = set(targets) - {joint.name for joint in self.description.joints}
        if unknown:
            raise ValueError(f"Unknown joint target names: {sorted(unknown)}")
        limits = {joint.name: joint for joint in self.description.joints}
        clean = {}
        for name, value in targets.items():
            value = float(value)
            joint = limits[name]
            if not joint.lower_limit_rad <= value <= joint.upper_limit_rad:
                raise ValueError(
                    f"Target {name}={value} outside [{joint.lower_limit_rad}, {joint.upper_limit_rad}]"
                )
            clean[name] = value
        self.simulation.set_actuator_targets(clean)

    def send_command(self, command: RobotCommand) -> None:
        parameters = command.parameters
        self.last_command_result = None
        if command.name is RobotCommandName.OPEN_HAND:
            self.grasp.cancel()
            self.send_joint_targets(self.simulation.reference_positions())
        elif command.name is RobotCommandName.GRASP:
            if self.grasp.teach_mode:
                raise ValueError(
                    "Automatic Sphere Grasp is disabled in Teach Grasp Pose mode; use Execute Taught Grasp"
                )
            self.grasp.start()
        elif command.name is RobotCommandName.RESET:
            self.grasp.cancel()
            self.send_joint_targets(self.simulation.reference_positions())
            self.objects.reset()
        elif command.name is RobotCommandName.SET_JOINT_TARGET:
            self.grasp.note_manual_target_change()
            self.send_joint_targets({str(parameters["joint_name"]): float(parameters["target_rad"])})
        elif command.name is RobotCommandName.SET_GRASP_PRESET:
            if "preset_name" in parameters:
                self.grasp.select_named(str(parameters["preset_name"]))
            elif "radius_m" in parameters:
                radius = float(parameters["radius_m"])
                self.objects.set_sphere_radius(radius)
                self.grasp.select_for_radius(radius)
            else:
                raise ValueError("set_grasp_preset requires preset_name or radius_m")
        elif command.name is RobotCommandName.RESET_REFERENCE_POSE:
            self.grasp.cancel()
            self.send_joint_targets(self.simulation.reference_positions())
            self.objects.reset()
        elif command.name is RobotCommandName.EXECUTE_GRASP_STAGE:
            if self.grasp.teach_mode:
                raise ValueError(
                    "Automatic grasp stages are disabled in Teach Grasp Pose mode"
                )
            self.grasp.apply_through_stage(int(parameters["stage_number"]))
        elif command.name is RobotCommandName.SAVE_GRASP_PRESET:
            if self.grasp.teach_mode:
                raise ValueError("Use Save Taught Pose while Teach Grasp Pose mode is enabled")
            preset = self.grasp.save_current_targets(str(parameters.get("preset_name", "")))
            self.last_command_result = {
                "preset_name": preset.name,
                "backup_directory": str(self.grasp.library.path.parent / "grasp_backups"),
            }
        elif command.name is RobotCommandName.LOAD_GRASP_PRESET:
            if self.grasp.teach_mode:
                raise ValueError("Use Load Taught Pose while Teach Grasp Pose mode is enabled")
            preset = self.grasp.select_named(str(parameters.get("preset_name", "")))
            self.send_joint_targets(dict(preset.joint_targets))
            self.last_command_result = {"preset_name": preset.name}
        elif command.name is RobotCommandName.SET_EXPERIMENT_MODE:
            self.grasp.cancel()
            self.objects.set_experiment_mode(str(parameters["mode"]))
            self.last_command_result = {"experiment_mode": self.objects.experiment_mode}
        elif command.name is RobotCommandName.SET_TEACH_GRASP_MODE:
            self.grasp.set_teach_mode(bool(parameters.get("enabled", True)))
            self.last_command_result = {"teach_grasp_pose": self.grasp.teach_mode}
        elif command.name is RobotCommandName.SET_POSE_VALIDATION_MODE:
            mode = str(parameters.get("mode", ""))
            self.grasp.cancel()
            if mode == "pose_reproduction_no_object":
                self.objects.activate(None)
            elif mode == "physical_sphere_grasp":
                self.objects.activate("sphere")
                self.objects.set_experiment_mode(ObjectController.FREE_OBJECT_GRASP)
            else:
                raise ValueError(f"Unknown pose validation mode: {mode}")
            self.grasp.validation_mode = mode
            self.grasp.taught_status = "READY"
            self.last_command_result = {"pose_validation_mode": mode}
        elif command.name is RobotCommandName.SAVE_TAUGHT_POSE:
            preset = self.grasp.save_taught_pose(
                str(parameters.get("pose_name", "")),
                target_source=str(parameters.get("target_source", "actual_measured")),
                waypoint_role=str(parameters.get("waypoint_role", "final_grasp")),
            )
            self.last_command_result = {
                "taught_pose_name": preset.name,
                "target_source": preset.target_source,
                "waypoint_role": preset.waypoint_role,
                "backup_directory": str(self.grasp.library.path.parent / "grasp_backups"),
            }
        elif command.name is RobotCommandName.LOAD_TAUGHT_POSE:
            preset = self.grasp.load_taught_pose(str(parameters.get("pose_name", "")))
            self.last_command_result = {
                "taught_pose_name": preset.name,
                "loaded_without_motion": True,
            }
        elif command.name is RobotCommandName.EXECUTE_TAUGHT_GRASP:
            names = parameters.get("pose_names") or []
            if isinstance(names, str):
                names = [item.strip() for item in names.split(",") if item.strip()]
            self.grasp.execute_taught(
                list(names),
                duration_seconds=float(parameters.get("duration_seconds", 1.0)),
                tolerance_rad=float(parameters.get("tolerance_rad", 0.03)),
            )
            self.last_command_result = {
                "taught_pose_names": [preset.name for preset in self.grasp._taught_waypoints],
                "trajectory": "bounded_actuator_targets_only",
            }
        elif command.name is RobotCommandName.STOP_TAUGHT_GRASP:
            self.grasp.stop_taught()
            self.last_command_result = {"execution_status": self.grasp.taught_status}

    def evaluation_payload(self, estimates, points, sphere_result) -> dict[str, Any]:
        truth = collect_mujoco_sphere_ground_truth(self.simulation)
        contacts = collect_mujoco_ground_truth(self.simulation, self.mounts, self.pressure.configs)
        contact_report = evaluate_contact_points(estimates, contacts)
        sphere_evaluation = (
            evaluate_sphere_reconstruction(sphere_result, points, truth)
            if sphere_result
            else None
        )
        result = {
            "ground_truth_sphere": {"center_xyz": list(truth.center_xyz), "radius_m": truth.radius},
            "ground_truth_contacts": [
                {"sensor_id": item.sensor_id, "position_xyz": list(item.position)}
                for item in contacts
            ],
            "contact_error_rmse_m": float(contact_report.metrics.euclidean.rmse_m),
            "sphere": None,
        }
        if sphere_evaluation:
            result["sphere"] = {
                "center_error_m": sphere_evaluation.center_error_m,
                "radius_error_m": sphere_evaluation.radius_error_m,
                "relative_radius_error": sphere_evaluation.relative_radius_error,
                "point_to_estimated_surface_rmse_m": sphere_evaluation.point_to_estimated_surface.rmse_m,
                "point_to_ground_truth_surface_rmse_m": sphere_evaluation.point_to_ground_truth_surface.rmse_m,
            }
        return result
