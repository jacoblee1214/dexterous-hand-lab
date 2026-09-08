"""Run hand verification and right-hand tactile contact reconstruction."""

from __future__ import annotations

import argparse
import colorsys
from dataclasses import dataclass
import math
from pathlib import Path
from queue import Empty
import time
from typing import Callable, Mapping

import numpy as np

from evaluation.contact_evaluator import (
    ContactPointError,
    GroundTruthContactPoint,
    collect_mujoco_ground_truth,
    evaluate_contact_points,
    summarize_contact_errors,
)
from evaluation.experiment_io import save_evaluation_dataset
from evaluation.sphere_evaluator import (
    SphereEvaluationResult,
    SphereGroundTruth,
    collect_mujoco_sphere_ground_truth,
    evaluate_sphere_reconstruction,
)
from reconstruction.contact_point_buffer import (
    ContactPointBuffer,
    load_point_buffer_config,
)
from reconstruction.contact_projection import ContactPointReconstructor
from reconstruction.estimated_contact import EstimatedContactPoint
from reconstruction.experiment_io import save_reconstruction_dataset
from reconstruction.reconstruction_input import ReconstructionInput
from reconstruction.temporal_observation import (
    TemporalTactileObservation,
    make_temporal_observations,
)
from reconstruction.sphere_fitting import (
    SphereReconstructionSession,
    SphereReconstructionStatus,
    load_sphere_fitting_config,
)
from sensors.registry_audit import (
    format_right_sensor_registry_table,
    require_approved_right_sensor_registry,
)
from sensors.sensor_kinematics import KinematicTree, SensorMount, load_sensor_mounts
from sensors.pressure_emulator import PressureSensorEmulator, ScalarSensorReading
from simulation.control_panel import CalibrationControlPanel, HandControlPanel
from simulation.grasp_presets import GraspPresetLibrary, SphereGraspController
from simulation.model_builder import (
    DEFAULT_OUTPUT,
    HAND_VARIANTS,
    build_hand_variant,
    build_model,
)
from simulation.sensor_calibration import SensorMountCalibration


PRESSURE_CONFIG = Path(__file__).resolve().parents[1] / "sensors" / "pressure_config.yaml"
RECONSTRUCTION_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "reconstruction_config.yaml"
)
DEFAULT_RECONSTRUCTION_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "right_sphere_run.json"
)
RIGHT_GRASP_CONFIG = Path(__file__).resolve().parent / "grasp_config_right.yaml"


@dataclass
class OverlayState:
    show_surfaces: bool = True
    show_centers: bool = True
    show_directions: bool = True
    show_ids: bool = False
    show_frames: bool = False
    show_active_sensors: bool = True
    show_estimated_contacts: bool = True
    show_accumulated_points: bool = True
    show_ground_truth_contacts: bool = False
    show_estimated_sphere: bool = True
    point_color_mode: str = "uniform"
    selected_sensor: int = 0
    selected_joint: int = 0


class HandSimulation:
    def __init__(self, model_path: str | Path = DEFAULT_OUTPUT):
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError("MuJoCo is not installed. Run: pip install -r requirements.txt") from exc
        self.mujoco = mujoco
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            build_model(self.model_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.sync_actuator_targets_to_qpos()

    def sync_actuator_targets_to_qpos(self) -> None:
        """Hold the loaded pose without injecting a startup position command."""
        for name in self.joint_names:
            joint_id = self.mujoco.mj_name2id(
                self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name
            )
            actuator_id = self.mujoco.mj_name2id(
                self.model, self.mujoco.mjtObj.mjOBJ_ACTUATOR, f"servo_{name}"
            )
            self.data.ctrl[actuator_id] = self.data.qpos[
                self.model.jnt_qposadr[joint_id]
            ]

    @property
    def joint_names(self) -> list[str]:
        return [
            self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, index)
            for index in range(self.model.njnt)
            if self.model.jnt_type[index] == self.mujoco.mjtJoint.mjJNT_HINGE
        ]

    @property
    def sensor_site_names(self) -> list[str]:
        names = []
        for index in range(self.model.nsite):
            name = self.mujoco.mj_id2name(
                self.model, self.mujoco.mjtObj.mjOBJ_SITE, index
            )
            if name and name.endswith("_Sensor"):
                names.append(name)
        return names

    def reference_positions(self) -> dict[str, float]:
        positions = {}
        for name in self.joint_names:
            joint_id = self.mujoco.mj_name2id(
                self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name
            )
            positions[name] = float(
                self.model.key_qpos[0, self.model.jnt_qposadr[joint_id]]
            )
        return positions

    def demonstration_target(self, joint_id: int) -> float:
        joint_name = self.mujoco.mj_id2name(
            self.model, self.mujoco.mjtObj.mjOBJ_JOINT, joint_id
        )
        actuator_id = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_ACTUATOR, f"servo_{joint_name}"
        )
        return float(self.model.key_ctrl[1, actuator_id])

    def set_joint_positions(self, positions: dict[str, float], *, reset_velocity: bool = True) -> None:
        if reset_velocity:
            self.data.qvel[:] = 0.0
        for name, value in positions.items():
            joint_id = self.mujoco.mj_name2id(
                self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name
            )
            if joint_id < 0:
                raise KeyError(f"Unknown joint: {name}")
            qpos_address = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_address] = value
            actuator_id = self.mujoco.mj_name2id(
                self.model, self.mujoco.mjtObj.mjOBJ_ACTUATOR, f"servo_{name}"
            )
            self.data.ctrl[actuator_id] = value
        self.mujoco.mj_forward(self.model, self.data)

    def set_actuator_targets(self, positions: dict[str, float]) -> None:
        for name, value in positions.items():
            actuator_id = self.mujoco.mj_name2id(
                self.model, self.mujoco.mjtObj.mjOBJ_ACTUATOR, f"servo_{name}"
            )
            self.data.ctrl[actuator_id] = value

    def reset_open(self) -> None:
        self.mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.sync_actuator_targets_to_qpos()
        self.mujoco.mj_forward(self.model, self.data)

    def step(self, count: int = 1) -> None:
        for _ in range(count):
            self.mujoco.mj_step(self.model, self.data)

    def joint_details(self, joint_name: str) -> dict:
        joint_id = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        child_body_id = int(self.model.jnt_bodyid[joint_id])
        parent_body_id = int(self.model.body_parentid[child_body_id])
        qpos_index = int(self.model.jnt_qposadr[joint_id])
        return {
            "joint_id": joint_id,
            "name": joint_name,
            "qpos_index": qpos_index,
            "angle": float(self.data.qpos[qpos_index]),
            "parent": self.mujoco.mj_id2name(
                self.model, self.mujoco.mjtObj.mjOBJ_BODY, parent_body_id
            ),
            "child": self.mujoco.mj_id2name(
                self.model, self.mujoco.mjtObj.mjOBJ_BODY, child_body_id
            ),
            "axis": self.model.jnt_axis[joint_id].copy(),
            "lower": float(self.model.jnt_range[joint_id, 0]),
            "upper": float(self.model.jnt_range[joint_id, 1]),
        }

    def descendants_of_joint(self, joint_name: str) -> set[str]:
        details = self.joint_details(joint_name)
        root_body = int(self.model.jnt_bodyid[details["joint_id"]])
        descendants = set()
        queue = [root_body]
        while queue:
            body_id = queue.pop()
            name = self.mujoco.mj_id2name(
                self.model, self.mujoco.mjtObj.mjOBJ_BODY, body_id
            )
            descendants.add(name)
            queue.extend(
                index
                for index in range(1, self.model.nbody)
                if int(self.model.body_parentid[index]) == body_id
            )
        return descendants

    def body_transforms(self) -> dict[str, np.ndarray]:
        return {
            self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_BODY, body_id):
            np.concatenate((self.data.xpos[body_id], self.data.xmat[body_id]))
            for body_id in range(1, self.model.nbody)
        }

    def set_object_collision_enabled(self, enabled: bool) -> None:
        geom_id = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_GEOM, "sphere_object_geom"
        )
        self.model.geom_contype[geom_id] = int(enabled)
        self.model.geom_conaffinity[geom_id] = int(enabled)


class ObjectController:
    SHAPES = ("sphere", "cylinder", "box")
    COLORS = {
        "sphere": np.array([0.22, 0.56, 0.96, 1.0]),
        "cylinder": np.array([0.96, 0.52, 0.16, 1.0]),
        "box": np.array([0.48, 0.82, 0.28, 1.0]),
    }
    SPHERE_RADII_M = (0.030, 0.040, 0.050, 0.060)
    FIXED_CONTACT_DIAGNOSTIC = "fixed_contact_diagnostic"
    FREE_OBJECT_GRASP = "free_object_grasp"
    EXPERIMENT_MODES = (FIXED_CONTACT_DIAGNOSTIC, FREE_OBJECT_GRASP)

    def __init__(self, simulation: HandSimulation, spawn_position: str):
        self.simulation = simulation
        self.spawn_position = np.fromstring(spawn_position, sep=" ", dtype=float)
        self.active_shape: str | None = None
        # Legacy engineering/object tests remain unlabelled until a caller
        # explicitly selects one of the two scientific experiment modes.
        self.experiment_mode = "unconfigured"
        self._reference_preparing = False
        self._parking = {
            shape: np.array([1.5 + 0.2 * index, 1.5, 0.2])
            for index, shape in enumerate(self.SHAPES)
        }
        self.activate(None)

    def _support_geom_id(self) -> int:
        return self.simulation.mujoco.mj_name2id(
            self.simulation.model,
            self.simulation.mujoco.mjtObj.mjOBJ_GEOM,
            "cylinder_object_geom",
        )

    def _active_body_id(self) -> int | None:
        if self.active_shape is None:
            return None
        return self.simulation.mujoco.mj_name2id(
            self.simulation.model,
            self.simulation.mujoco.mjtObj.mjOBJ_BODY,
            f"{self.active_shape}_object",
        )

    def _configure_experiment_physics(self) -> None:
        support_id = self._support_geom_id()
        enabled = (
            self.experiment_mode == self.FREE_OBJECT_GRASP
            and self.active_shape == "sphere"
        )
        cylinder_active = self.active_shape == "cylinder"
        self.simulation.model.geom_contype[support_id] = int(enabled or cylinder_active)
        self.simulation.model.geom_conaffinity[support_id] = int(enabled or cylinder_active)
        self.simulation.model.geom_rgba[support_id] = (
            [0.20, 0.25, 0.31, 1.0]
            if enabled
            else self.COLORS["cylinder"] if cylinder_active else [0.20, 0.25, 0.31, 0.0]
        )
        self.simulation.model.geom_size[support_id, :2] = (
            [0.020, 0.004] if enabled else [0.025, 0.035]
        )
        if enabled:
            self._set_pose(
                "cylinder",
                np.array(
                    [
                        self.spawn_position[0],
                        self.spawn_position[1],
                        self.spawn_position[2] - self.sphere_radius - 0.004,
                    ]
                ),
            )
        body_id = self._active_body_id()
        if body_id is not None:
            self.simulation.model.body_gravcomp[body_id] = (
                0.0 if enabled else 1.0
            )

    def set_experiment_mode(self, mode: str) -> None:
        normalized = str(mode).strip().lower()
        if normalized not in self.EXPERIMENT_MODES:
            raise ValueError(f"Unknown experiment mode: {mode}")
        self.experiment_mode = normalized
        # A fixed target behaves like an infinite-mass contact fixture. Apply
        # high damping to both labelled experiment modes so force-limited
        # closure does not chatter visibly against that fixture/support.
        hand_dofs = len(self.simulation.joint_names)
        self.simulation.model.dof_damping[:hand_dofs] = 0.8
        self.simulation.model.actuator_biasprm[:hand_dofs, 2] = -1.0
        self.simulation.model.geom_solref[:, 0] = (
            0.008 if normalized == self.FIXED_CONTACT_DIAGNOSTIC else 0.005
        )
        self._configure_experiment_physics()
        self.reset()

    def prepare_reference_phase(self) -> None:
        """Park the object while an actuator-only hand opening settles."""
        if self.active_shape is None:
            return
        _, _, geom_id = self._ids(self.active_shape)
        self.simulation.model.geom_contype[geom_id] = 0
        self.simulation.model.geom_conaffinity[geom_id] = 0
        self.simulation.model.geom_rgba[geom_id] = [0, 0, 0, 0]
        self._set_pose(self.active_shape, self._parking[self.active_shape])
        support_id = self._support_geom_id()
        self.simulation.model.geom_contype[support_id] = 0
        self.simulation.model.geom_conaffinity[support_id] = 0
        self.simulation.model.geom_rgba[support_id] = [0.20, 0.25, 0.31, 0.0]
        self._set_pose("cylinder", self._parking["cylinder"])
        self._reference_preparing = True
        self.simulation.mujoco.mj_forward(self.simulation.model, self.simulation.data)

    def restore_after_reference(self) -> None:
        if not self._reference_preparing:
            return
        shape = self.active_shape
        self._reference_preparing = False
        self.activate(shape)

    def enforce_fixed_diagnostic_pose(self) -> None:
        if (
            self.experiment_mode == self.FIXED_CONTACT_DIAGNOSTIC
            and self.active_shape is not None
            and not self._reference_preparing
        ):
            self._set_pose(self.active_shape, self.spawn_position)
            self.simulation.mujoco.mj_forward(
                self.simulation.model, self.simulation.data
            )

    def _ids(self, shape: str) -> tuple[int, int, int]:
        mujoco = self.simulation.mujoco
        joint_id = mujoco.mj_name2id(
            self.simulation.model, mujoco.mjtObj.mjOBJ_JOINT, f"{shape}_object_free"
        )
        geom_id = mujoco.mj_name2id(
            self.simulation.model, mujoco.mjtObj.mjOBJ_GEOM, f"{shape}_object_geom"
        )
        return (
            joint_id,
            int(self.simulation.model.jnt_qposadr[joint_id]),
            geom_id,
        )

    def _set_pose(self, shape: str, position: np.ndarray) -> None:
        joint_id, qpos_address, _ = self._ids(shape)
        dof_address = int(self.simulation.model.jnt_dofadr[joint_id])
        self.simulation.data.qpos[qpos_address : qpos_address + 7] = [
            *position,
            1.0,
            0.0,
            0.0,
            0.0,
        ]
        self.simulation.data.qvel[dof_address : dof_address + 6] = 0.0

    def activate(self, shape: str | None) -> None:
        if shape is not None and shape not in self.SHAPES:
            raise ValueError(f"Unknown object shape: {shape}")
        for current in self.SHAPES:
            _, _, geom_id = self._ids(current)
            self.simulation.model.geom_contype[geom_id] = 0
            self.simulation.model.geom_conaffinity[geom_id] = 0
            self.simulation.model.geom_rgba[geom_id] = [0, 0, 0, 0]
            self._set_pose(current, self._parking[current])
        self.active_shape = shape
        if shape is not None:
            _, _, geom_id = self._ids(shape)
            self.simulation.model.geom_contype[geom_id] = 1
            self.simulation.model.geom_conaffinity[geom_id] = 1
            self.simulation.model.geom_rgba[geom_id] = self.COLORS[shape]
            self._set_pose(shape, self.spawn_position)
        self._configure_experiment_physics()
        self.simulation.mujoco.mj_forward(self.simulation.model, self.simulation.data)
        print(f"Active object: {shape or 'none'}")

    def reset(self) -> None:
        if self.active_shape is not None:
            self._configure_experiment_physics()
            self._set_pose(self.active_shape, self.spawn_position)
            self.simulation.mujoco.mj_forward(
                self.simulation.model, self.simulation.data
            )

    @property
    def sphere_radius(self) -> float:
        _, _, geom_id = self._ids("sphere")
        return float(self.simulation.model.geom_size[geom_id, 0])

    def set_sphere_radius(self, radius_m: float) -> None:
        radius_m = float(radius_m)
        if not any(np.isclose(radius_m, value) for value in self.SPHERE_RADII_M):
            raise ValueError(
                f"Sphere radius must be one of {self.SPHERE_RADII_M}, got {radius_m}"
            )
        _, _, geom_id = self._ids("sphere")
        self.simulation.model.geom_size[geom_id, 0] = radius_m
        self.simulation.model.geom_rbound[geom_id] = radius_m
        self._configure_experiment_physics()
        self.reset()
        print(f"Sphere radius setting: {radius_m * 1000:.0f} mm")

    def set_spawn_position(self, position_xyz) -> None:
        position = np.asarray(position_xyz, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("Object spawn position must be a finite XYZ vector")
        self.spawn_position = position.copy()
        self._configure_experiment_physics()

    def move(self, delta_xyz: np.ndarray) -> None:
        if self.active_shape is None:
            return
        _, qpos_address, _ = self._ids(self.active_shape)
        self.simulation.data.qpos[qpos_address : qpos_address + 3] += delta_xyz
        self.simulation.mujoco.mj_forward(self.simulation.model, self.simulation.data)

    def maintain_inactive(self) -> None:
        if self._reference_preparing:
            for shape in self.SHAPES:
                self._set_pose(shape, self._parking[shape])
            return
        for shape in self.SHAPES:
            if shape != self.active_shape:
                if (
                    shape == "cylinder"
                    and self.active_shape == "sphere"
                    and self.experiment_mode == self.FREE_OBJECT_GRASP
                ):
                    self._set_pose(
                        shape,
                        np.array(
                            [
                                self.spawn_position[0],
                                self.spawn_position[1],
                                self.spawn_position[2] - self.sphere_radius - 0.004,
                            ]
                        ),
                    )
                else:
                    self._set_pose(shape, self._parking[shape])


def format_joint_details(details: dict, pose_name: str | None = None) -> str:
    axis = details["axis"]
    heading = f"\n=== Joint verification: {pose_name} ===" if pose_name else "\n=== Joint ==="
    return "\n".join(
        [
            heading,
            f"Joint name: {details['name']}",
            f"qpos index: {details['qpos_index']}",
            f"Current angle: {details['angle']:.6f} rad ({math.degrees(details['angle']):.2f} deg)",
            f"Parent link: {details['parent']}",
            f"Child link: {details['child']}",
            f"Joint axis: [{axis[0]:.6f}, {axis[1]:.6f}, {axis[2]:.6f}]",
            f"Experimental lower limit: {details['lower']:.6f} rad",
            f"Experimental upper limit: {details['upper']:.6f} rad",
        ]
    )


def format_sensor_inspection(
    simulation: HandSimulation,
    mounts: dict[str, SensorMount],
    sensor_id: str,
    reading: ScalarSensorReading | None = None,
    estimated_contact: EstimatedContactPoint | None = None,
    evaluation_error: ContactPointError | None = None,
    ground_truth_enabled: bool = False,
    reconstruction_enabled: bool = False,
) -> str:
    mount = mounts[sensor_id]
    site_id = simulation.mujoco.mj_name2id(
        simulation.model, simulation.mujoco.mjtObj.mjOBJ_SITE, sensor_id
    )
    position = simulation.data.site_xpos[site_id]
    sensor_rotation = simulation.data.site_xmat[site_id].reshape(3, 3)
    direction = sensor_rotation @ mount.local_sensing_axis
    direction /= np.linalg.norm(direction)
    if estimated_contact is not None:
        position = np.asarray(estimated_contact.sensor_position)
        direction = np.asarray(estimated_contact.sensor_sensing_direction)
    lines = [
            "\n--- Selected Sensor (live) ---",
            f"Sensor ID: {sensor_id}",
            f"Sensor Surface Type: {mount.sensor_type}",
            f"Parent Link: {mount.parent_link}",
            "Surface Dimensions: "
            f"width={mount.surface_width_m:.6f}, height={mount.surface_height_m:.6f}, "
            f"thickness={mount.surface_thickness_m:.6f} m",
            "Sensor Position in Base/World Coordinate Frame: "
            f"x={position[0]: .6f}, y={position[1]: .6f}, z={position[2]: .6f}",
            "Sensor Local Sensing Axis: "
            f"sx={mount.local_sensing_axis[0]: .6f}, sy={mount.local_sensing_axis[1]: .6f}, sz={mount.local_sensing_axis[2]: .6f}",
            "Sensor Sensing Direction in Base/World Coordinate Frame: "
            f"nx={direction[0]: .6f}, ny={direction[1]: .6f}, nz={direction[2]: .6f}",
            "Link-to-Sensor Translation: "
            f"x={mount.position_xyz[0]: .6f}, y={mount.position_xyz[1]: .6f}, z={mount.position_xyz[2]: .6f}",
            f"Surface U Axis (link): {mount.surface_u_axis.round(6).tolist()}",
            f"Surface V Axis (link): {mount.surface_v_axis.round(6).tolist()}",
            f"Outward Normal (link): {mount.outward_normal.round(6).tolist()}",
        ]
    if reading is not None:
        lines.append(f"Sensor Surface Area: {reading.surface_area_m2:.9f} m^2")
        if not reconstruction_enabled:
            lines.append(
                f"Normal Contact Force: {reading.normal_contact_force_n:.6f} N"
            )
        lines.append(f"Simulated Scalar Output: {reading.scalar_output_n:.6f} N")
        if not reconstruction_enabled:
            lines.extend(
                [
                    f"Estimated Pressure: {reading.pressure_pa:.3f} Pa",
                    f"Estimated Indentation: {reading.indentation_m:.9f} m",
                ]
            )
    if reconstruction_enabled:
        if estimated_contact is not None:
            lines.extend(
                [
                    f"Estimated Pressure: {estimated_contact.estimated_pressure:.3f} Pa",
                    "Estimated Indentation Depth: "
                    f"{estimated_contact.estimated_indentation:.9f} m",
                    "Estimated Three-Dimensional Contact Position: "
                    f"{[round(value, 6) for value in estimated_contact.estimated_contact_position]}",
                ]
            )
        else:
            lines.extend(
                [
                    "Estimated Pressure: inactive",
                    "Estimated Indentation Depth: inactive",
                    "Estimated Three-Dimensional Contact Position: inactive",
                ]
            )
    if ground_truth_enabled:
        if evaluation_error is None:
            lines.extend(
                [
                    "Ground-Truth Contact Position: none assigned",
                    "Euclidean Contact Error: unavailable",
                    "Normal-Direction Error: unavailable",
                    "Tangential-Plane Error: unavailable",
                ]
            )
        else:
            lines.extend(
                [
                    "Ground-Truth Contact Position: "
                    f"{[round(value, 6) for value in evaluation_error.ground_truth_position]}",
                    f"Euclidean Contact Error: {evaluation_error.euclidean_error_m:.6f} m",
                    "Normal-Direction Error: "
                    f"{evaluation_error.normal_direction_error_m:.6f} m",
                    "Tangential-Plane Error: "
                    f"{evaluation_error.tangential_plane_error_m:.6f} m",
                ]
            )
    return "\n".join(lines)


def update_sensor_overlays(
    simulation: HandSimulation,
    scene,
    mounts: dict[str, SensorMount],
    state: OverlayState,
    *,
    reset_scene: bool = True,
    live_mount_transforms: bool = False,
    readings: Mapping[str, ScalarSensorReading] | None = None,
) -> None:
    mujoco = simulation.mujoco
    if reset_scene:
        scene.ngeom = 0

    def connector(kind, width: float, start, end, color) -> None:
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_connector(geom, kind, width, start, end)
        geom.rgba[:] = color
        geom.emission = 1.0
        scene.ngeom += 1

    for sensor_id, mount in mounts.items():
        if live_mount_transforms:
            body_id = mujoco.mj_name2id(
                simulation.model, mujoco.mjtObj.mjOBJ_BODY, mount.parent_link
            )
            body_rotation = simulation.data.xmat[body_id].reshape(3, 3)
            position = simulation.data.xpos[body_id] + body_rotation @ mount.center_xyz
            rotation = body_rotation @ mount.rotation_matrix
        else:
            site_id = mujoco.mj_name2id(
                simulation.model, mujoco.mjtObj.mjOBJ_SITE, sensor_id
            )
            position = simulation.data.site_xpos[site_id]
            rotation = simulation.data.site_xmat[site_id].reshape(3, 3)
        direction = rotation @ mount.local_sensing_axis
        # The configured position is the representative sensing-surface point.
        # Offset only the debug rendering (not FK) to avoid mesh z-fighting.
        visual_surface_position = position + direction * 0.0004
        sensor_is_active = (
            readings is not None
            and sensor_id in readings
            and readings[sensor_id].active
        )
        show_active_surface = state.show_active_sensors and sensor_is_active
        if (state.show_surfaces or show_active_surface) and scene.ngeom < scene.maxgeom:
            geom = scene.geoms[scene.ngeom]
            if mount.sensor_type == "rectangular":
                size = np.array(
                    [
                        mount.surface_width_m / 2,
                        mount.surface_height_m / 2,
                        mount.surface_thickness_m / 2,
                    ]
                )
                geom_type = mujoco.mjtGeom.mjGEOM_BOX
            else:
                size = mount.fingertip_radii_xyz
                geom_type = mujoco.mjtGeom.mjGEOM_ELLIPSOID
            geom_position = visual_surface_position
            if mount.sensor_type == "fingertip_surface":
                geom_position = position - direction * size[2] + direction * 0.0004
            is_selected = sensor_id == list(mounts)[state.selected_sensor]
            if show_active_surface:
                surface_color = np.array([1.0, 0.06, 0.01, 1.0])
            elif live_mount_transforms and is_selected:
                surface_color = np.array([0.95, 0.15, 0.75, 0.58])
            else:
                surface_color = np.array([0.05, 0.90, 0.95, 0.72])
            mujoco.mjv_initGeom(
                geom,
                geom_type,
                size,
                geom_position,
                rotation.reshape(-1),
                surface_color,
            )
            geom.emission = 0.85
            scene.ngeom += 1
            if mount.sensor_type == "rectangular" and scene.ngeom < scene.maxgeom:
                boundary = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(
                    boundary,
                    mujoco.mjtGeom.mjGEOM_LINEBOX,
                    size * np.array([1.03, 1.03, 1.8]),
                    visual_surface_position,
                    rotation.reshape(-1),
                    (
                        np.array([1.0, 0.06, 0.01, 1.0])
                        if show_active_surface
                        else np.array([0.0, 1.0, 1.0, 1.0])
                    ),
                )
                boundary.emission = 1.0
                scene.ngeom += 1
        if state.show_centers and scene.ngeom < scene.maxgeom:
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.0020, 0.0020, 0.0020]),
                position,
                np.eye(3).reshape(-1),
                np.array([0.10, 0.45, 1.0, 1.0]),
            )
            geom.emission = 1.0
            scene.ngeom += 1
        if state.show_directions:
            arrow_origin = position + direction * 0.0004
            connector(
                mujoco.mjtGeom.mjGEOM_ARROW,
                0.0018,
                arrow_origin,
                arrow_origin + 0.020 * direction,
                [1.0, 0.78, 0.02, 1.0],
            )
        if state.show_frames:
            for axis, color in zip(
                rotation.T,
                ([1.0, 0.1, 0.1, 1.0], [0.1, 1.0, 0.1, 1.0], [0.1, 0.4, 1.0, 1.0]),
            ):
                connector(
                    mujoco.mjtGeom.mjGEOM_ARROW,
                    0.00055,
                    position,
                    position + 0.010 * axis,
                    color,
                )
        if state.show_ids and scene.ngeom < scene.maxgeom:
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_LABEL,
                np.zeros(3),
                position + np.array([0.0, 0.0, 0.005]),
                np.eye(3).reshape(-1),
                np.array([1.0, 1.0, 1.0, 1.0]),
            )
            geom.label = sensor_id
            scene.ngeom += 1


def make_reconstruction_input(
    simulation: HandSimulation,
    pressure: PressureSensorEmulator,
    sensor_config_path: Path,
    readings: Mapping[str, ScalarSensorReading],
) -> ReconstructionInput:
    """Build the only data object crossing into reconstruction code."""
    joint_state = {}
    for name in simulation.joint_names:
        joint_id = simulation.mujoco.mj_name2id(
            simulation.model, simulation.mujoco.mjtObj.mjOBJ_JOINT, name
        )
        joint_state[name] = float(
            simulation.data.qpos[simulation.model.jnt_qposadr[joint_id]]
        )
    calibration = {
        sensor_id: {
            "effective_sensor_area_m2": config.effective_sensor_area_m2,
            "virtual_stiffness_n_per_m": config.virtual_stiffness_n_per_m,
            "minimum_activation_force_n": config.minimum_activation_force_n,
            "saturation_pressure_pa": config.saturation_pressure_pa,
            "contact_projection_sign": config.contact_projection_sign,
        }
        for sensor_id, config in pressure.configs.items()
    }
    return ReconstructionInput(
        timestamp=float(simulation.data.time),
        joint_state=joint_state,
        sensor_values={
            sensor_id: reading.scalar_output_n
            for sensor_id, reading in readings.items()
        },
        sensor_config_path=str(sensor_config_path),
        calibration_config=calibration,
    )


def update_reconstruction_overlays(
    simulation: HandSimulation,
    scene,
    state: OverlayState,
    current_estimates: list[EstimatedContactPoint],
    accumulated_estimates: tuple[TemporalTactileObservation, ...],
    ground_truth: list[GroundTruthContactPoint],
) -> None:
    """Append clearly separated estimated/accumulated/evaluation point markers."""
    mujoco = simulation.mujoco

    finger_colors = {
        "Finger01": [1.0, 0.35, 0.25, 0.85],
        "Finger02": [0.20, 0.80, 1.0, 0.85],
        "Finger03": [0.35, 1.0, 0.30, 0.85],
        "Finger04": [1.0, 0.75, 0.15, 0.85],
        "Finger05": [0.90, 0.25, 1.0, 0.85],
        "Palm": [1.0, 1.0, 1.0, 0.85],
    }
    timestamps = [point.timestamp for point in accumulated_estimates]
    earliest = min(timestamps, default=0.0)
    latest = max(timestamps, default=0.0)
    sensor_ids = sorted({point.sensor_id for point in accumulated_estimates})

    def history_color(observation: TemporalTactileObservation):
        if state.point_color_mode == "finger":
            return finger_colors.get(observation.finger_id, [0.8, 0.8, 0.8, 0.85])
        if state.point_color_mode == "sensor":
            hue = sensor_ids.index(observation.sensor_id) / max(1, len(sensor_ids))
            red, green, blue = colorsys.hsv_to_rgb(hue, 0.75, 1.0)
            return [red, green, blue, 0.85]
        if state.point_color_mode == "time":
            span = latest - earliest
            fraction = 0.5 if span == 0 else (observation.timestamp - earliest) / span
            return [fraction, 0.25, 1.0 - fraction, 0.85]
        return [0.72, 0.12, 1.0, 0.80]

    def point(position, radius: float, color) -> None:
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.full(3, radius),
            np.asarray(position, dtype=float),
            np.eye(3).reshape(-1),
            np.asarray(color, dtype=float),
        )
        geom.emission = 1.0
        scene.ngeom += 1

    if state.show_accumulated_points:
        reserved = 0
        if state.show_estimated_contacts:
            reserved += len(current_estimates)
        if state.show_ground_truth_contacts:
            reserved += len(ground_truth)
        available = max(0, scene.maxgeom - scene.ngeom - reserved)
        if available:
            for estimate in accumulated_estimates[-available:]:
                point(
                    estimate.estimated_contact_position,
                    0.0018,
                    history_color(estimate),
                )
    if state.show_estimated_contacts:
        for estimate in current_estimates:
            point(
                estimate.estimated_contact_position,
                0.0034,
                [1.0, 0.92, 0.02, 1.0],
            )
    if state.show_ground_truth_contacts:
        for truth in ground_truth:
            point(truth.position, 0.0028, [0.10, 1.0, 0.25, 1.0])


def update_sphere_reconstruction_overlays(
    simulation: HandSimulation,
    scene,
    state: OverlayState,
    session: SphereReconstructionSession,
    ground_truth: SphereGroundTruth,
) -> None:
    """Draw valid estimate in blue and opt-in evaluation GT in green."""

    def sphere(center, radius: float, color) -> None:
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        simulation.mujoco.mjv_initGeom(
            geom,
            simulation.mujoco.mjtGeom.mjGEOM_SPHERE,
            np.full(3, radius),
            np.asarray(center, dtype=float),
            np.eye(3).reshape(-1),
            np.asarray(color, dtype=float),
        )
        scene.ngeom += 1

    result = session.result
    if (
        state.show_estimated_sphere
        and result is not None
        and result.status is SphereReconstructionStatus.VALID_RECONSTRUCTION
        and result.estimated_center_xyz is not None
        and result.estimated_radius is not None
    ):
        sphere(result.estimated_center_xyz, result.estimated_radius, [0.05, 0.75, 1.0, 0.24])
    if state.show_ground_truth_contacts:
        sphere(ground_truth.center_xyz, ground_truth.radius, [0.15, 1.0, 0.25, 0.16])


def print_sphere_reconstruction_result(result) -> None:
    condition = (
        f"{result.condition_metric:.3g}"
        if result.condition_metric is not None
        else "undefined"
    )
    print(
        "Sphere reconstruction: "
        f"status={result.status.value}, points={result.number_of_input_points}, "
        f"sensors={result.number_of_unique_sensors}, fingers={result.number_of_unique_fingers}, "
        f"condition={condition}"
    )
    if result.estimated_center_xyz is not None:
        print(
            f"Estimated center={np.round(result.estimated_center_xyz, 6).tolist()} m, "
            f"radius={result.estimated_radius:.6f} m, "
            f"fit RMSE={result.fit_residual_rmse:.6f} m"
        )


def configure_viewer(
    viewer,
    simulation: HandSimulation,
    state: OverlayState,
    *,
    free_camera: bool = False,
    hand: str = "right",
) -> None:
    if free_camera:
        reset_calibration_camera(viewer.cam, hand)
    else:
        viewer.cam.fixedcamid = simulation.mujoco.mj_name2id(
            simulation.model, simulation.mujoco.mjtObj.mjOBJ_CAMERA, "overview"
        )
        viewer.cam.type = simulation.mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.opt.sitegroup[3] = False  # Replaced by bright overlay spheres.
    viewer.opt.sitegroup[4] = False  # Replaced by unmistakable arrow overlays.
    viewer.opt.geomgroup[5] = False  # Collision hulls stay physical but invisible.


def reset_calibration_camera(camera, hand: str) -> None:
    """Set a useful free-camera pose without framing the parked object bodies."""
    camera.type = 0  # mujoco.mjtCamera.mjCAMERA_FREE
    camera.fixedcamid = -1
    camera.lookat[:] = [0.0, 0.0, 0.095]
    camera.distance = 0.34
    camera.azimuth = 135.0 if hand == "right" else -135.0
    camera.elevation = -25.0


def apply_free_camera_command(camera, hand: str, action: str) -> None:
    """Control a free viewer camera from either calibration or experiment UI."""
    if action == "orbit_left":
        camera.azimuth -= 10.0
    elif action == "orbit_right":
        camera.azimuth += 10.0
    elif action == "orbit_up":
        camera.elevation = min(89.0, camera.elevation + 10.0)
    elif action == "orbit_down":
        camera.elevation = max(-89.0, camera.elevation - 10.0)
    elif action == "zoom_in":
        camera.distance = max(0.04, camera.distance * 0.82)
    elif action == "zoom_out":
        camera.distance = min(2.0, camera.distance * 1.22)
    elif action == "reset":
        reset_calibration_camera(camera, hand)
    else:
        raise ValueError(f"Unknown free-camera action: {action}")


def apply_calibration_camera_command(
    camera,
    simulation: HandSimulation,
    calibration: SensorMountCalibration,
    hand: str,
    command: tuple,
) -> None:
    """Apply camera-only UI commands; never mutate sensor calibration values."""
    if command[0] != "camera":
        raise ValueError(f"Not a camera command: {command}")
    action = command[1]
    if action == "focus_selected":
        mount = calibration.selected
        body_id = simulation.mujoco.mj_name2id(
            simulation.model,
            simulation.mujoco.mjtObj.mjOBJ_BODY,
            mount.parent_link,
        )
        rotation = simulation.data.xmat[body_id].reshape(3, 3)
        camera.lookat[:] = (
            simulation.data.xpos[body_id] + rotation @ mount.center_xyz
        )
        camera.distance = min(camera.distance, 0.16)
    else:
        apply_free_camera_command(camera, hand, action)


def make_key_callback(
    state: OverlayState,
    sensor_names: list[str],
    simulation: HandSimulation,
    mounts: dict[str, SensorMount],
    finger_groups: dict[str, list[str]] | None = None,
    object_controller: ObjectController | None = None,
    point_buffer: ContactPointBuffer | None = None,
    sphere_only: bool = False,
    save_experiment_callback: Callable[[], None] | None = None,
    reset_evaluation_callback: Callable[[], None] | None = None,
    sphere_session: SphereReconstructionSession | None = None,
    grasp_controller: SphereGraspController | None = None,
):
    finger_groups = finger_groups or {}
    finger_names = list(finger_groups)

    def callback(key: int) -> None:
        if key in (ord("V"), ord("v")):
            state.show_surfaces = not state.show_surfaces
            print(f"Show Pressure Sensor Surfaces: {state.show_surfaces}")
        elif key in (ord("X"), ord("x")):
            state.show_active_sensors = not state.show_active_sensors
            print(f"Show Active Pressure Sensors: {state.show_active_sensors}")
        elif key in (ord("E"), ord("e")):
            state.show_estimated_contacts = not state.show_estimated_contacts
            print(f"Show Estimated Contact Points: {state.show_estimated_contacts}")
        elif key in (ord("M"), ord("m")):
            state.show_accumulated_points = not state.show_accumulated_points
            print(f"Show Accumulated Contact Points: {state.show_accumulated_points}")
        elif point_buffer is not None and key in (ord("T"), ord("t")):
            modes = ("uniform", "finger", "sensor", "time")
            state.point_color_mode = modes[
                (modes.index(state.point_color_mode) + 1) % len(modes)
            ]
            print(f"Accumulated-point color mode: {state.point_color_mode}")
        elif key in (ord("P"), ord("p")):
            state.show_centers = not state.show_centers
            print(f"Show Sensor Centers: {state.show_centers}")
        elif key in (ord("D"), ord("d")):
            state.show_directions = not state.show_directions
            print(f"Show Sensor Sensing Directions: {state.show_directions}")
        elif key in (ord("I"), ord("i")):
            state.show_ids = not state.show_ids
            print(f"Show Sensor IDs: {state.show_ids}")
        elif key in (ord("F"), ord("f")):
            state.show_frames = not state.show_frames
            print(f"Show Sensor Coordinate Frames: {state.show_frames}")
        elif key == ord("["):
            state.selected_sensor = (state.selected_sensor - 1) % len(sensor_names)
            print(
                format_sensor_inspection(
                    simulation,
                    mounts,
                    sensor_names[state.selected_sensor],
                    reconstruction_enabled=point_buffer is not None,
                )
            )
        elif key == ord("]"):
            state.selected_sensor = (state.selected_sensor + 1) % len(sensor_names)
            print(
                format_sensor_inspection(
                    simulation,
                    mounts,
                    sensor_names[state.selected_sensor],
                    reconstruction_enabled=point_buffer is not None,
                )
            )
        elif key in (ord("G"), ord("g")):
            if point_buffer is None:
                print("Ground-truth contact visualization is available only in reconstruction experiment mode")
            else:
                state.show_ground_truth_contacts = not state.show_ground_truth_contacts
                print(
                    "Show Ground-Truth Contacts + Sphere (evaluation only): "
                    f"{state.show_ground_truth_contacts}"
                )
        elif sphere_session is not None and key in (ord("U"), ord("u")):
            print_sphere_reconstruction_result(sphere_session.fit(point_buffer.points))
        elif sphere_session is not None and key in (ord("Q"), ord("q")):
            sphere_session.reset()
            print("Sphere reconstruction reset (tactile points retained)")
        elif sphere_session is not None and key in (ord("W"), ord("w")):
            state.show_estimated_sphere = not state.show_estimated_sphere
            print(f"Show Estimated Sphere: {state.show_estimated_sphere}")
        elif sphere_session is not None and key == ord("0"):
            sphere_session.toggle_auto_fit()
            print(f"Sphere fitting mode: {sphere_session.mode}")
        elif grasp_controller is not None and key == ord(" "):
            grasp_controller.start()
            print(
                f"Sphere Grasp started: {grasp_controller.selected_preset_name} "
                "(staged actuator targets)"
            )
        elif point_buffer is not None and key in (ord("A"), ord("a")):
            point_buffer.toggle()
            print(
                "Contact-point accumulation: "
                f"{'RUNNING' if point_buffer.accumulating else 'PAUSED'}"
            )
        elif point_buffer is not None and key in (ord("Z"), ord("z")):
            point_buffer.reset()
            if sphere_session is not None:
                sphere_session.reset()
            if reset_evaluation_callback is not None:
                reset_evaluation_callback()
            point_buffer.pause()
            print("Accumulated contact points reset and PAUSED; press A to resume")
        elif save_experiment_callback is not None and key in (ord("L"), ord("l")):
            save_experiment_callback()
        elif key in (ord("O"), ord("o")):
            simulation.set_actuator_targets(simulation.reference_positions())
            print("Hand target: OPEN")
        elif key in (ord("C"), ord("c")):
            targets = {
                name: simulation.demonstration_target(
                    simulation.mujoco.mj_name2id(
                        simulation.model,
                        simulation.mujoco.mjtObj.mjOBJ_JOINT,
                        name,
                    )
                )
                for name in simulation.joint_names
            }
            simulation.set_actuator_targets(targets)
            print("Hand target: CLOSE")
        elif key in (ord("H"), ord("h")):
            # Interactive control must flow through actuators. Direct qpos
            # assignment is reserved for deterministic FK/joint tests.
            simulation.set_actuator_targets(simulation.reference_positions())
            if grasp_controller is not None:
                grasp_controller.cancel()
            if object_controller:
                object_controller.reset()
            print("Hand actuator targets reset to reference")
        elif ord("1") <= key <= ord("5") and finger_names:
            finger_index = key - ord("1")
            if finger_index < len(finger_names):
                finger = finger_names[finger_index]
                targets = {}
                first_name = finger_groups[finger][0]
                first_actuator = simulation.mujoco.mj_name2id(
                    simulation.model,
                    simulation.mujoco.mjtObj.mjOBJ_ACTUATOR,
                    f"servo_{first_name}",
                )
                first_joint_id = simulation.mujoco.mj_name2id(
                    simulation.model,
                    simulation.mujoco.mjtObj.mjOBJ_JOINT,
                    first_name,
                )
                reference = simulation.reference_positions()
                is_open = np.isclose(
                    simulation.data.ctrl[first_actuator], reference[first_name], atol=0.05
                )
                for name in finger_groups[finger]:
                    joint_id = simulation.mujoco.mj_name2id(
                        simulation.model,
                        simulation.mujoco.mjtObj.mjOBJ_JOINT,
                        name,
                    )
                    targets[name] = (
                        simulation.demonstration_target(joint_id)
                        if is_open
                        else reference[name]
                    )
                simulation.set_actuator_targets(targets)
                print(f"{finger} target: {'FLEX' if is_open else 'OPEN'}")
        elif key in (ord("J"), ord("j"), ord("K"), ord("k")):
            direction = -1 if key in (ord("J"), ord("j")) else 1
            state.selected_joint = (
                state.selected_joint + direction
            ) % len(simulation.joint_names)
            print(
                format_joint_details(
                    simulation.joint_details(
                        simulation.joint_names[state.selected_joint]
                    )
                )
            )
        elif key in (ord(","), ord(".")):
            name = simulation.joint_names[state.selected_joint]
            details = simulation.joint_details(name)
            actuator_id = simulation.mujoco.mj_name2id(
                simulation.model,
                simulation.mujoco.mjtObj.mjOBJ_ACTUATOR,
                f"servo_{name}",
            )
            delta = -0.05 if key == ord(",") else 0.05
            simulation.data.ctrl[actuator_id] = np.clip(
                simulation.data.ctrl[actuator_id] + delta,
                details["lower"],
                details["upper"],
            )
            print(
                f"Joint target {name}: {simulation.data.ctrl[actuator_id]:.4f} rad; "
                f"range [{details['lower']:.4f}, {details['upper']:.4f}]"
            )
        elif object_controller and key in (ord("S"), ord("s")):
            object_controller.activate("sphere")
        elif object_controller and key in (ord("Y"), ord("y")):
            if sphere_only:
                print("Cylinder is disabled in the right-hand sphere reconstruction experiment")
            else:
                object_controller.activate("cylinder")
        elif object_controller and key in (ord("B"), ord("b")):
            if sphere_only:
                print("Box is disabled in the right-hand sphere reconstruction experiment")
            else:
                object_controller.activate("box")
        elif object_controller and key in (ord("N"), ord("n")):
            if sphere_only:
                print("Object removal is disabled in the right-hand sphere reconstruction experiment")
            else:
                object_controller.activate(None)
        elif object_controller and key in (ord("R"), ord("r")):
            object_controller.reset()
            print("Object position reset")
        elif (
            object_controller
            and sphere_only
            and key in (ord("6"), ord("7"), ord("8"), ord("9"))
        ):
            object_controller.set_sphere_radius(
                ObjectController.SPHERE_RADII_M[key - ord("6")]
            )
            if grasp_controller is not None:
                grasp_controller.select_for_radius(object_controller.sphere_radius)
            if sphere_session is not None:
                sphere_session.reset()
            if point_buffer is not None:
                point_buffer.reset()
                point_buffer.pause()
            if reset_evaluation_callback is not None:
                reset_evaluation_callback()
            print("Sphere size changed; tactile points/reconstruction cleared and PAUSED")
        elif object_controller and key in (262, 263, 264, 265, 266, 267):
            movements = {
                262: np.array([0.005, 0.0, 0.0]),
                263: np.array([-0.005, 0.0, 0.0]),
                264: np.array([0.0, -0.005, 0.0]),
                265: np.array([0.0, 0.005, 0.0]),
                266: np.array([0.0, 0.0, 0.005]),
                267: np.array([0.0, 0.0, -0.005]),
            }
            object_controller.move(movements[key])

    return callback


def apply_hand_control_command(
    simulation: HandSimulation,
    finger_groups: dict[str, list[str]],
    command: tuple,
) -> None:
    """Apply a GUI command only through MuJoCo position actuator targets."""
    kind = command[0]
    reference = simulation.reference_positions()
    if kind == "hand":
        action = command[1]
        if action in {"open", "reset"}:
            simulation.set_actuator_targets(reference)
        elif action == "close":
            simulation.set_actuator_targets(
                {
                    name: simulation.demonstration_target(
                        simulation.mujoco.mj_name2id(
                            simulation.model,
                            simulation.mujoco.mjtObj.mjOBJ_JOINT,
                            name,
                        )
                    )
                    for name in simulation.joint_names
                }
            )
        else:
            raise ValueError(f"Unknown hand action: {action}")
    elif kind == "finger":
        finger, fraction = command[1], float(np.clip(command[2], 0.0, 1.0))
        targets = {}
        for name in finger_groups[finger]:
            joint_id = simulation.mujoco.mj_name2id(
                simulation.model, simulation.mujoco.mjtObj.mjOBJ_JOINT, name
            )
            target = simulation.demonstration_target(joint_id)
            targets[name] = reference[name] + fraction * (target - reference[name])
        simulation.set_actuator_targets(targets)
    elif kind == "joint":
        name, requested = command[1], float(command[2])
        details = simulation.joint_details(name)
        simulation.set_actuator_targets(
            {name: float(np.clip(requested, details["lower"], details["upper"]))}
        )
    else:
        raise ValueError(f"Unknown actuator control command: {command}")


def make_hand_control_panel(
    simulation: HandSimulation,
    groups: dict[str, list[str]],
    reconstruction_controls: bool = False,
    mounts: Mapping[str, SensorMount] | None = None,
) -> HandControlPanel:
    details = []
    for name in simulation.joint_names:
        item = simulation.joint_details(name)
        actuator_id = simulation.mujoco.mj_name2id(
            simulation.model,
            simulation.mujoco.mjtObj.mjOBJ_ACTUATOR,
            f"servo_{name}",
        )
        item["target"] = float(simulation.data.ctrl[actuator_id])
        details.append(item)
    sensor_details = [
        {
            "sensor_id": sensor_id,
            "finger_id": mount.finger_id,
            "parent_link": mount.parent_link,
        }
        for sensor_id, mount in (mounts or {}).items()
    ]
    return HandControlPanel(
        details, list(groups), reconstruction_controls, sensor_details
    )


def service_hand_control_panel(
    panel: HandControlPanel,
    simulation: HandSimulation,
    groups: dict[str, list[str]],
    auxiliary_command: Callable[[tuple], None] | None = None,
    reconstruction_state: OverlayState | None = None,
    point_buffer: ContactPointBuffer | None = None,
    sensor_readings: Mapping[str, ScalarSensorReading] | None = None,
    sphere_session: SphereReconstructionSession | None = None,
    object_controller: ObjectController | None = None,
    mounts: Mapping[str, SensorMount] | None = None,
    current_estimates: list[EstimatedContactPoint] | None = None,
    grasp_controller: SphereGraspController | None = None,
) -> None:
    try:
        while True:
            command = panel.commands.get_nowait()
            if command[0] in {"hand", "finger", "joint"}:
                apply_hand_control_command(simulation, groups, command)
            elif auxiliary_command is not None:
                auxiliary_command(command)
            else:
                raise ValueError(f"Unsupported control-panel command: {command}")
    except Empty:
        pass
    published = {
        "joint_angles": {
            name: simulation.joint_details(name)["angle"]
            for name in simulation.joint_names
        },
        "joint_targets": {
            name: float(
                simulation.data.ctrl[
                    simulation.mujoco.mj_name2id(
                        simulation.model,
                        simulation.mujoco.mjtObj.mjOBJ_ACTUATOR,
                        f"servo_{name}",
                    )
                ]
            )
            for name in simulation.joint_names
        },
    }
    if reconstruction_state is not None and point_buffer is not None:
        active_sensor_ids = [
            sensor_id
            for sensor_id, reading in (sensor_readings or {}).items()
            if reading.active
        ]
        active_fingers = {
            mounts[sensor_id].finger_id
            for sensor_id in active_sensor_ids
            if mounts is not None
        }
        contributing_sensors = {point.sensor_id for point in point_buffer.points}
        contributing_fingers = {point.finger_id for point in point_buffer.points}
        positions = np.asarray(
            [point.estimated_contact_position for point in point_buffer.points],
            dtype=float,
        )
        spatial_spread = (
            float(np.linalg.norm(np.ptp(positions, axis=0)))
            if len(positions)
            else 0.0
        )
        estimate_by_sensor = {
            estimate.sensor_id: estimate for estimate in (current_estimates or [])
        }
        telemetry = []
        if mounts is not None:
            for sensor_id, mount in mounts.items():
                reading = (sensor_readings or {}).get(sensor_id)
                estimate = estimate_by_sensor.get(sensor_id)
                site_id = simulation.mujoco.mj_name2id(
                    simulation.model,
                    simulation.mujoco.mjtObj.mjOBJ_SITE,
                    sensor_id,
                )
                rotation = simulation.data.site_xmat[site_id].reshape(3, 3)
                direction = rotation @ mount.local_sensing_axis
                direction /= np.linalg.norm(direction)
                telemetry.append(
                    {
                        "sensor_id": sensor_id,
                        "finger_id": mount.finger_id,
                        "parent_link": mount.parent_link,
                        "active": bool(reading and reading.active),
                        "scalar_value_n": reading.scalar_output_n if reading else 0.0,
                        "normal_force_n": reading.normal_contact_force_n if reading else 0.0,
                        "pressure_pa": reading.pressure_pa if reading else 0.0,
                        "indentation_m": reading.indentation_m if reading else 0.0,
                        "position_xyz": tuple(float(v) for v in simulation.data.site_xpos[site_id]),
                        "sensing_direction_xyz": tuple(float(v) for v in direction),
                        "estimated_contact_xyz": (
                            estimate.estimated_contact_position if estimate else None
                        ),
                    }
                )
        published["reconstruction"] = {
            "surfaces": "ON" if reconstruction_state.show_surfaces else "OFF",
            "active": "ON" if reconstruction_state.show_active_sensors else "OFF",
            "current": "ON" if reconstruction_state.show_estimated_contacts else "OFF",
            "history": "ON" if reconstruction_state.show_accumulated_points else "OFF",
            "color_mode": reconstruction_state.point_color_mode,
            "ground_truth": (
                "ON" if reconstruction_state.show_ground_truth_contacts else "OFF"
            ),
            "accumulation": "RUNNING" if point_buffer.accumulating else "PAUSED",
            "point_count": len(point_buffer.points),
            "raw_count": point_buffer.statistics.raw_observation_count,
            "duplicate_count": point_buffer.statistics.rejected_duplicate_count,
            "active_sensor_ids": active_sensor_ids,
            "physical_sensor_count": len(sensor_readings or {}),
            "currently_active_count": len(active_sensor_ids),
            "unique_active_sensor_count": len(set(active_sensor_ids)),
            "active_finger_count": len(active_fingers),
            "unique_contributing_sensor_count": len(contributing_sensors),
            "unique_contributing_finger_count": len(contributing_fingers),
            "spatial_spread_m": spatial_spread,
            "sensor_telemetry": telemetry,
            "selected_sensor_id": (
                list(mounts)[reconstruction_state.selected_sensor]
                if mounts is not None
                else None
            ),
            "estimated_sphere": (
                "ON" if reconstruction_state.show_estimated_sphere else "OFF"
            ),
            "fit_mode": sphere_session.mode if sphere_session else "-",
            "fit_status": (
                sphere_session.result.status.value
                if sphere_session and sphere_session.result
                else "NOT_FITTED"
            ),
            "sphere_radius_mm": (
                object_controller.sphere_radius * 1000.0
                if object_controller is not None
                else 0.0
            ),
            "grasp_preset": (
                grasp_controller.selected_preset_name if grasp_controller else "-"
            ),
            "grasp_stage": grasp_controller.current_stage if grasp_controller else 0,
            "grasp_running": bool(grasp_controller and grasp_controller.running),
        }
    panel.publish(published)


def run_viewer(
    simulation: HandSimulation,
    mounts: dict[str, SensorMount],
    selected_sensor: str | None = None,
    hand: str = "right",
    object_controller: ObjectController | None = None,
    collision_monitor: bool = False,
    reconstruction_experiment: bool = False,
    reconstruction_output: Path = DEFAULT_RECONSTRUCTION_OUTPUT,
) -> None:
    import mujoco.viewer

    if reconstruction_experiment:
        if hand != "right":
            raise ValueError("Contact reconstruction experiment currently supports only --hand right")
        if object_controller is None or object_controller.active_shape != "sphere":
            raise ValueError("Contact reconstruction experiment requires one active sphere")
        audit = require_approved_right_sensor_registry(
            mounts, simulation.sensor_site_names
        )
        print(format_right_sensor_registry_table(mounts, simulation.sensor_site_names))
        print(
            "Right sensor registry approved: "
            f"configured={audit.configured_count}, runtime={audit.runtime_count}; "
            "five Link04 channels intentionally absent by manual approval."
        )

    sensor_names = list(mounts)
    state = OverlayState(
        selected_sensor=sensor_names.index(selected_sensor) if selected_sensor else 0
    )
    groups = finger_joint_groups(hand, mounts)
    point_buffer = (
        ContactPointBuffer(load_point_buffer_config(RECONSTRUCTION_CONFIG))
        if reconstruction_experiment
        else ContactPointBuffer(max_points=10000)
    )
    evaluation_errors: list[ContactPointError] = []
    sphere_session = (
        SphereReconstructionSession(load_sphere_fitting_config(RECONSTRUCTION_CONFIG))
        if reconstruction_experiment
        else None
    )
    grasp_controller = None
    if reconstruction_experiment:
        grasp_library = GraspPresetLibrary(
            RIGHT_GRASP_CONFIG, HAND_VARIANTS[hand].urdf
        )
        grasp_controller = SphereGraspController(
            simulation, object_controller, grasp_library
        )
        grasp_controller.select_for_radius(object_controller.sphere_radius)

    def reset_evaluation() -> None:
        evaluation_errors.clear()

    def save_experiment() -> None:
        reconstruction_path = save_reconstruction_dataset(
            reconstruction_output,
            point_buffer.points,
            point_buffer.config,
            point_buffer.statistics,
            sphere_session.result if sphere_session else None,
        )
        evaluation_path = reconstruction_path.with_name(
            f"{reconstruction_path.stem}.evaluation.json"
        )
        sphere_truth = collect_mujoco_sphere_ground_truth(simulation)
        sphere_evaluation = (
            evaluate_sphere_reconstruction(
                sphere_session.result, point_buffer.points, sphere_truth
            )
            if sphere_session and sphere_session.result
            else None
        )
        save_evaluation_dataset(
            evaluation_path,
            evaluation_errors,
            sphere_truth,
            sphere_evaluation,
        )
        print(f"Saved reconstruction dataset: {reconstruction_path}")
        print(f"Saved separate evaluation dataset: {evaluation_path}")

    callback = make_key_callback(
        state,
        sensor_names,
        simulation,
        mounts,
        groups,
        object_controller,
        point_buffer if reconstruction_experiment else None,
        reconstruction_experiment,
        save_experiment if reconstruction_experiment else None,
        reset_evaluation if reconstruction_experiment else None,
        sphere_session,
        grasp_controller,
    )
    control_panel = make_hand_control_panel(
        simulation,
        groups,
        reconstruction_controls=reconstruction_experiment,
        mounts=mounts if reconstruction_experiment else None,
    )
    pressure = PressureSensorEmulator(simulation, mounts, PRESSURE_CONFIG)
    reconstructor = (
        ContactPointReconstructor(HAND_VARIANTS[hand].urdf)
        if reconstruction_experiment
        else None
    )
    readings = pressure.read()
    control_panel.start()
    print("Sensor: V surfaces | P centers | D directions | I IDs | F frames | [ ] select")
    print("Hand: O open | C close | H reset | 1-5 finger flex | J/K select joint | ,/. adjust")
    print("A separate control panel provides Open/Close/Reset, five finger sliders, and all 20 joint sliders.")
    if reconstruction_experiment:
        print("Reconstruction: X active surfaces | E current estimates | M accumulated points")
        print("Reconstruction: G evaluation-only GT | A start/pause | Z clear+pause | L save")
        print("Sphere fit: U fit | Q reset fit | W estimated sphere | 0 manual/auto")
        print("Sphere radius: 6/7/8/9 = 30/40/50/60 mm (changing it clears points)")
        print("Grasp: Space = configured staged Sphere Grasp; panel has stages/save/load")
        print("Point coloring: T cycles uniform/finger/sensor/time")
        print("Colors: cyan sensor | red active | yellow current estimate | violet history | green ground truth")
        print("Object: one sphere only | S/R spawn-reset | arrows/PgUp/PgDn move")
        print("View: left-drag orbit | right-drag pan | mouse wheel zoom (free camera)")
        print("Sensor surfaces start ON; V toggles them. X controls the red active-only layer independently.")
        print(f"Accumulation starts PAUSED. Output: {reconstruction_output}")
    else:
        print("Object: S sphere | Y cylinder | B box | N none | R reset | arrows/PgUp/PgDn move")
    last_inspection = 0.0
    current_estimates: list[EstimatedContactPoint] = []
    ground_truth: list[GroundTruthContactPoint] = []
    ground_truth_sphere = (
        collect_mujoco_sphere_ground_truth(simulation)
        if reconstruction_experiment
        else None
    )
    try:
        with mujoco.viewer.launch_passive(
            simulation.model, simulation.data, key_callback=callback
        ) as viewer:
            configure_viewer(
                viewer,
                simulation,
                state,
                free_camera=reconstruction_experiment,
                hand=hand,
            )

            def handle_auxiliary_panel_command(command: tuple) -> None:
                if command[0] == "viewer_key":
                    callback(int(command[1]))
                elif command[0] == "camera" and reconstruction_experiment:
                    apply_free_camera_command(viewer.cam, hand, command[1])
                elif command[0] == "select_sensor" and reconstruction_experiment:
                    state.selected_sensor = sensor_names.index(command[1])
                elif grasp_controller is not None and command[0] in {
                    "sphere_grasp",
                    "grasp_stage",
                    "save_grasp",
                    "load_grasp",
                    "reload_grasps",
                }:
                    try:
                        if command[0] == "sphere_grasp":
                            grasp_controller.start()
                            print(
                                f"Sphere Grasp started: {grasp_controller.selected_preset_name}"
                            )
                        elif command[0] == "grasp_stage":
                            grasp_controller.apply_through_stage(int(command[1]))
                            print(f"Applied grasp through stage {int(command[1])}")
                        elif command[0] == "save_grasp":
                            preset = grasp_controller.save_current_targets(str(command[1]))
                            print(f"Saved grasp preset {preset.name}: {RIGHT_GRASP_CONFIG}")
                        elif command[0] == "load_grasp":
                            preset = grasp_controller.select_named(str(command[1]))
                            print(f"Loaded grasp preset {preset.name}; press Sphere Grasp")
                        else:
                            grasp_controller.reload()
                            print(f"Reloaded grasp presets: {RIGHT_GRASP_CONFIG}")
                    except (KeyError, ValueError) as exc:
                        print(f"Grasp command rejected: {exc}")
                else:
                    raise ValueError(f"Unsupported viewer-panel command: {command}")

            while viewer.is_running():
                service_hand_control_panel(
                    control_panel,
                    simulation,
                    groups,
                    handle_auxiliary_panel_command,
                    state if reconstruction_experiment else None,
                    point_buffer if reconstruction_experiment else None,
                    readings if reconstruction_experiment else None,
                    sphere_session,
                    object_controller,
                    mounts if reconstruction_experiment else None,
                    current_estimates if reconstruction_experiment else None,
                    grasp_controller,
                )
                if object_controller:
                    object_controller.maintain_inactive()
                simulation.step()
                if reconstruction_experiment and object_controller is not None:
                    object_controller.enforce_fixed_diagnostic_pose()
                if grasp_controller is not None:
                    grasp_controller.update()
                readings = pressure.read()
                current_error_by_sensor: dict[str, ContactPointError] = {}
                if reconstructor is not None:
                    frame = make_reconstruction_input(
                        simulation,
                        pressure,
                        HAND_VARIANTS[hand].sensors,
                        readings,
                    )
                    current_estimates = reconstructor.reconstruct(frame)
                    ground_truth = collect_mujoco_ground_truth(
                        simulation, mounts, pressure.configs
                    )
                    report = evaluate_contact_points(current_estimates, ground_truth)
                    current_error_by_sensor = {
                        error.sensor_id: error for error in report.errors
                    }
                    temporal_observations = make_temporal_observations(
                        current_estimates, frame.joint_state
                    )
                    accepted = point_buffer.add(temporal_observations)
                    if accepted and sphere_session is not None:
                        auto_result = sphere_session.maybe_auto_fit(point_buffer.points)
                        if auto_result is not None:
                            print_sphere_reconstruction_result(auto_result)
                    accepted_sensor_ids = {
                        observation.sensor_id for observation in accepted
                    }
                    evaluation_errors.extend(
                        error
                        for error in report.errors
                        if error.sensor_id in accepted_sensor_ids
                    )
                    maximum_errors = point_buffer.config.maximum_point_count
                    if len(evaluation_errors) > maximum_errors:
                        del evaluation_errors[:-maximum_errors]
                update_sensor_overlays(
                    simulation, viewer.user_scn, mounts, state, readings=readings
                )
                if reconstructor is not None:
                    update_reconstruction_overlays(
                        simulation,
                        viewer.user_scn,
                        state,
                        current_estimates,
                        point_buffer.points,
                        ground_truth,
                    )
                    ground_truth_sphere = collect_mujoco_sphere_ground_truth(simulation)
                    update_sphere_reconstruction_overlays(
                        simulation,
                        viewer.user_scn,
                        state,
                        sphere_session,
                        ground_truth_sphere,
                    )
                viewer.sync()
                if time.monotonic() - last_inspection >= 1.0:
                    selected_id = sensor_names[state.selected_sensor]
                    current_by_sensor = {
                        estimate.sensor_id: estimate
                        for estimate in current_estimates
                    }
                    print(
                        format_sensor_inspection(
                            simulation,
                            mounts,
                            selected_id,
                            readings[selected_id],
                            current_by_sensor.get(selected_id),
                            current_error_by_sensor.get(selected_id),
                            state.show_ground_truth_contacts,
                            reconstructor is not None,
                        )
                    )
                    if reconstructor is not None:
                        metrics = summarize_contact_errors(evaluation_errors)
                        if metrics.count:
                            for label, values in (
                                ("Euclidean", metrics.euclidean),
                                ("Normal", metrics.normal_direction),
                                ("Tangential", metrics.tangential_plane),
                            ):
                                print(
                                    f"{label} error: n={values.count}, "
                                    f"mean={values.mean_m:.6f} m, "
                                    f"median={values.median_m:.6f} m, "
                                    f"RMSE={values.rmse_m:.6f} m, "
                                    f"p95={values.percentile_95_m:.6f} m"
                                )
                        else:
                            print("Contact-point evaluation: no assigned contact pairs yet")
                        statistics = point_buffer.statistics
                        print(
                            "Point buffer: "
                            f"raw={statistics.raw_observation_count}, "
                            f"accepted={statistics.accepted_point_count}, "
                            f"duplicates={statistics.rejected_duplicate_count}, "
                            f"pressure_rejected={statistics.rejected_pressure_count}"
                        )
                        if sphere_session.result is None:
                            print(
                                f"Sphere reconstruction: NOT_FITTED ({sphere_session.mode}); "
                                "press U to fit"
                            )
                        else:
                            print_sphere_reconstruction_result(sphere_session.result)
                            sphere_evaluation = evaluate_sphere_reconstruction(
                                sphere_session.result,
                                point_buffer.points,
                                ground_truth_sphere,
                            )
                            if sphere_evaluation is not None:
                                print(
                                    "Evaluation-only sphere error: "
                                    f"center={sphere_evaluation.center_error_m:.6f} m, "
                                    f"radius={sphere_evaluation.radius_error_m:.6f} m, "
                                    f"relative_radius={sphere_evaluation.relative_radius_error:.4f}"
                                )
                    if collision_monitor and object_controller and object_controller.active_shape:
                        shape = object_controller.active_shape
                        _, qpos_address, geom_id = object_controller._ids(shape)
                        distances = [
                            simulation.data.contact[index].dist
                            for index in range(simulation.data.ncon)
                            if geom_id
                            in (
                                simulation.data.contact[index].geom1,
                                simulation.data.contact[index].geom2,
                            )
                        ]
                        position = simulation.data.qpos[qpos_address : qpos_address + 3]
                        print(
                            f"Collision monitor: {shape} position={position.round(4).tolist()} "
                            f"contacts={len(distances)} min_distance="
                            f"{min(distances) if distances else 'none'}"
                        )
                    last_inspection = time.monotonic()
                time.sleep(max(0.0, simulation.model.opt.timestep * 0.8))
    finally:
        control_panel.close()


def verify_single_joint_numerically(
    simulation: HandSimulation,
    joint_name: str,
) -> tuple[set[str], set[str]]:
    reference = simulation.reference_positions()
    simulation.set_joint_positions(reference)
    baseline = simulation.body_transforms()
    details = simulation.joint_details(joint_name)
    test_value = details["upper"]
    if np.isclose(test_value, reference[joint_name]):
        test_value = details["lower"]
    pose = dict(reference)
    pose[joint_name] = test_value
    simulation.set_joint_positions(pose)
    changed = {
        name
        for name, transform in simulation.body_transforms().items()
        if not np.allclose(transform, baseline[name], atol=1e-9)
    }
    descendants = simulation.descendants_of_joint(joint_name)
    return changed, descendants


def run_joint_test(
    simulation: HandSimulation,
    mounts: dict[str, SensorMount],
    joint_filter: str | None,
    hold_seconds: float,
    headless: bool,
) -> None:
    names = [joint_filter] if joint_filter else simulation.joint_names
    unknown = set(names) - set(simulation.joint_names)
    if unknown:
        raise ValueError(f"Unknown joint(s): {sorted(unknown)}")
    reference = simulation.reference_positions()

    for name in names:
        changed, descendants = verify_single_joint_numerically(simulation, name)
        unexpected = changed - descendants
        missing = descendants - changed
        print(
            f"Downstream audit {name}: changed={sorted(changed)}; "
            f"unexpected={sorted(unexpected)}; unchanged descendants={sorted(missing)}"
        )
        if unexpected or missing:
            raise RuntimeError(f"Downstream-chain verification failed for {name}")
    if headless:
        for name in names:
            details = simulation.joint_details(name)
            values = [
                ("minimum", details["lower"]),
                ("reference", reference[name]),
                ("midpoint", 0.5 * (details["lower"] + details["upper"])),
                ("maximum", details["upper"]),
            ]
            for pose_name, value in values:
                pose = dict(reference)
                pose[name] = value
                simulation.set_joint_positions(pose)
                print(format_joint_details(simulation.joint_details(name), pose_name))
        return

    import mujoco.viewer

    state = OverlayState(show_ids=True)
    sensor_names = list(mounts)
    callback = make_key_callback(state, sensor_names, simulation, mounts)
    simulation.set_object_collision_enabled(False)
    print("Joint test uses exact qpos poses; objects are disabled for unobstructed FK verification.")
    with mujoco.viewer.launch_passive(
        simulation.model, simulation.data, key_callback=callback
    ) as viewer:
        configure_viewer(viewer, simulation, state)
        for name in names:
            details = simulation.joint_details(name)
            values = [
                ("minimum", details["lower"]),
                ("reference", reference[name]),
                ("midpoint", 0.5 * (details["lower"] + details["upper"])),
                ("maximum", details["upper"]),
            ]
            for pose_name, value in values:
                if not viewer.is_running():
                    return
                pose = dict(reference)
                pose[name] = value
                simulation.set_joint_positions(pose)
                print(format_joint_details(simulation.joint_details(name), pose_name))
                print(format_sensor_inspection(simulation, mounts, sensor_names[state.selected_sensor]))
                deadline = time.monotonic() + hold_seconds
                while viewer.is_running() and time.monotonic() < deadline:
                    update_sensor_overlays(simulation, viewer.user_scn, mounts, state)
                    viewer.sync()
                    time.sleep(0.02)


def finger_joint_groups(hand: str, mounts: dict[str, SensorMount]) -> dict[str, list[str]]:
    tree = KinematicTree.from_urdf(HAND_VARIANTS[hand].urdf)
    parent_joint = {joint.child: joint for joint in tree.joints}
    groups: dict[str, set[str]] = {}
    for mount in mounts.values():
        link = mount.parent_link
        while link in parent_joint:
            joint = parent_joint[link]
            if joint.joint_type != "fixed":
                groups.setdefault(mount.finger_id, set()).add(joint.name)
            link = joint.parent
    order = {joint.name: index for index, joint in enumerate(tree.joints)}
    return {
        finger: sorted(joints, key=order.get) for finger, joints in sorted(groups.items())
    }


def run_sensor_verification(
    simulation: HandSimulation,
    mounts: dict[str, SensorMount],
    hand: str,
    object_controller: ObjectController,
) -> None:
    import mujoco.viewer

    groups = finger_joint_groups(hand, mounts)
    sensor_names = list(mounts)
    state = OverlayState(show_ids=True, show_frames=True)
    callback = make_key_callback(state, sensor_names, simulation, mounts, groups)
    control_panel = make_hand_control_panel(simulation, groups)
    control_panel.start()
    object_controller.activate(None)
    print("Sensor mount verification: no object is active. Use the actuator control panel or keyboard.")
    try:
        with mujoco.viewer.launch_passive(
            simulation.model, simulation.data, key_callback=callback
        ) as viewer:
            configure_viewer(viewer, simulation, state)
            while viewer.is_running():
                service_hand_control_panel(control_panel, simulation, groups)
                object_controller.maintain_inactive()
                simulation.step()
                update_sensor_overlays(simulation, viewer.user_scn, mounts, state)
                viewer.sync()
                time.sleep(0.02)
    finally:
        control_panel.close()


def calibration_snapshot(calibration: SensorMountCalibration) -> dict:
    mount = calibration.selected
    return {
        "sensor_id": mount.sensor_id,
        "parent_link": mount.parent_link,
        "sensor_type": mount.sensor_type,
        "center_xyz": mount.center_xyz.round(6).tolist(),
        "surface_u_axis": mount.surface_u_axis.round(6).tolist(),
        "surface_v_axis": mount.surface_v_axis.round(6).tolist(),
        "outward_normal": mount.outward_normal.round(6).tolist(),
        "width": mount.width,
        "height": mount.height,
    }


def apply_calibration_command(
    calibration: SensorMountCalibration,
    state: OverlayState,
    command: tuple,
) -> None:
    kind = command[0]
    if kind == "select_sensor":
        calibration.select_sensor(command[1])
    elif kind == "select_link":
        calibration.select_link(command[1])
    elif kind == "move_center":
        delta = np.zeros(3)
        delta[int(command[1])] = float(command[2])
        calibration.move_center(delta)
    elif kind == "rotate":
        axis = np.eye(3)[int(command[1])]
        calibration.rotate(axis, math.radians(float(command[2])))
    elif kind == "resize":
        dimension, delta = command[1], float(command[2])
        mount = calibration.selected
        width = max(0.0005, mount.width + delta) if dimension == "width" else mount.width
        height = max(0.0005, mount.height + delta) if dimension == "height" else mount.height
        calibration.set_size(width, height)
    elif kind == "flip_normal":
        calibration.flip_normal()
    elif kind == "save":
        calibration.save()
        print(f"Saved explicit sensor mounts: {calibration.config_path}")
    else:
        raise ValueError(f"Unknown calibration command: {command}")
    state.selected_sensor = list(calibration.mounts).index(
        calibration.selected_sensor_id
    )
    print(calibration_snapshot(calibration))


def run_sensor_mount_calibration(
    simulation: HandSimulation,
    mounts: dict[str, SensorMount],
    config_path: Path,
    object_controller: ObjectController,
    hand: str,
) -> None:
    import mujoco.viewer

    object_controller.activate(None)
    calibration = SensorMountCalibration(mounts, config_path)
    state = OverlayState(show_ids=True, show_frames=True)
    sensor_names = list(calibration.mounts)
    panel = CalibrationControlPanel(sensor_names, calibration.parent_links)
    panel.start()

    def key_callback(key: int) -> None:
        commands = {
            ord("A"): ("move_center", 0, -0.0005),
            ord("a"): ("move_center", 0, -0.0005),
            ord("D"): ("move_center", 0, 0.0005),
            ord("d"): ("move_center", 0, 0.0005),
            ord("W"): ("move_center", 1, 0.0005),
            ord("w"): ("move_center", 1, 0.0005),
            ord("S"): ("move_center", 1, -0.0005),
            ord("s"): ("move_center", 1, -0.0005),
            ord("Q"): ("move_center", 2, 0.0005),
            ord("q"): ("move_center", 2, 0.0005),
            ord("E"): ("move_center", 2, -0.0005),
            ord("e"): ("move_center", 2, -0.0005),
            ord("1"): ("rotate", 0, -2.0),
            ord("2"): ("rotate", 0, 2.0),
            ord("3"): ("rotate", 1, -2.0),
            ord("4"): ("rotate", 1, 2.0),
            ord("5"): ("rotate", 2, -2.0),
            ord("6"): ("rotate", 2, 2.0),
            ord(","): ("resize", "width", -0.0005),
            ord("."): ("resize", "width", 0.0005),
            ord(";"): ("resize", "height", -0.0005),
            ord("'"): ("resize", "height", 0.0005),
            ord("L"): ("flip_normal",),
            ord("l"): ("flip_normal",),
            257: ("save",),  # GLFW Enter
        }
        if key == ord("["):
            index = (state.selected_sensor - 1) % len(sensor_names)
            apply_calibration_command(
                calibration, state, ("select_sensor", sensor_names[index])
            )
        elif key == ord("]"):
            index = (state.selected_sensor + 1) % len(sensor_names)
            apply_calibration_command(
                calibration, state, ("select_sensor", sensor_names[index])
            )
        elif key in commands:
            apply_calibration_command(calibration, state, commands[key])

    print("Sensor mount calibration: NO OBJECT. Selected surface is magenta.")
    print("Panel: select link/sensor, move center, rotate, resize, Flip Normal, Save.")
    print("View: left-drag orbit | right-drag pan | mouse wheel zoom; camera buttons are in the panel.")
    print("Keyboard: [ ] select | A/D X | W/S Y | Q/E Z | 1-6 rotate | ,/. width | ;/' height | L flip | Enter save")
    try:
        with mujoco.viewer.launch_passive(
            simulation.model, simulation.data, key_callback=key_callback
        ) as viewer:
            configure_viewer(viewer, simulation, state)
            # Calibration needs unrestricted inspection of every side of each
            # mesh. Switching away from the fixed overview camera enables the
            # viewer's native mouse orbit/pan/zoom behavior.
            reset_calibration_camera(viewer.cam, hand)
            while viewer.is_running():
                try:
                    while True:
                        command = panel.commands.get_nowait()
                        if command[0] == "camera":
                            apply_calibration_camera_command(
                                viewer.cam, simulation, calibration, hand, command
                            )
                        else:
                            apply_calibration_command(calibration, state, command)
                except Empty:
                    pass
                panel.publish(calibration_snapshot(calibration))
                object_controller.maintain_inactive()
                simulation.step()
                update_sensor_overlays(
                    simulation,
                    viewer.user_scn,
                    calibration.mounts,
                    state,
                    live_mount_transforms=True,
                )
                viewer.sync()
                time.sleep(0.02)
    finally:
        panel.close()


def run_idle_diagnostics(
    simulation: HandSimulation,
    object_controller: ObjectController,
    duration_seconds: float = 5.0,
) -> bool:
    """Run reproducible gravity-on/off idle audits with no external object."""
    if duration_seconds < 3.0:
        raise ValueError("Idle diagnostics must run for at least 3 seconds")
    mujoco, model, data = simulation.mujoco, simulation.model, simulation.data
    object_controller.activate(None)
    original_gravity = model.opt.gravity.copy()
    all_passed = True

    for gravity_enabled in (True, False):
        simulation.reset_open()
        object_controller.activate(None)
        model.opt.gravity[:] = original_gravity if gravity_enabled else 0.0
        simulation.sync_actuator_targets_to_qpos()
        mujoco.mj_forward(model, data)
        initial_mismatch = 0.0
        statistics = {}
        for name in simulation.joint_names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            actuator_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"servo_{name}"
            )
            qpos_address = int(model.jnt_qposadr[joint_id])
            initial_mismatch = max(
                initial_mismatch, abs(float(data.ctrl[actuator_id] - data.qpos[qpos_address]))
            )
            statistics[name] = {
                "joint_id": joint_id,
                "actuator_id": actuator_id,
                "qpos": qpos_address,
                "dof": int(model.jnt_dofadr[joint_id]),
                "body": int(model.jnt_bodyid[joint_id]),
                "max_qvel": 0.0,
                "max_error": 0.0,
                "max_force": 0.0,
                "contact": False,
            }

        max_contacts = max_self_contacts = 0
        maximum_contact_force = 0.0
        steps = int(np.ceil(duration_seconds / model.opt.timestep))
        report_every = max(1, int(round(1.0 / model.opt.timestep)))
        print(
            f"\n=== IDLE DIAGNOSTICS: gravity={'ON' if gravity_enabled else 'OFF'} "
            f"duration={duration_seconds:.1f}s ==="
        )
        print(f"Initial max |ctrl-qpos|: {initial_mismatch:.3e} rad")
        for step_index in range(steps):
            object_controller.maintain_inactive()
            mujoco.mj_step(model, data)
            contact_bodies: set[int] = set()
            self_contacts = 0
            force = np.zeros(6)
            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                bodies = {
                    int(model.geom_bodyid[contact.geom1]),
                    int(model.geom_bodyid[contact.geom2]),
                }
                contact_bodies.update(bodies)
                names = [
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "world"
                    for body_id in bodies
                ]
                if 0 not in bodies and all(not name.endswith("_object") for name in names):
                    self_contacts += 1
                mujoco.mj_contactForce(model, data, contact_index, force)
                maximum_contact_force = max(
                    maximum_contact_force, float(np.linalg.norm(force[:3]))
                )
            max_contacts = max(max_contacts, data.ncon)
            max_self_contacts = max(max_self_contacts, self_contacts)
            for values in statistics.values():
                qpos_address, dof, actuator_id = (
                    values["qpos"], values["dof"], values["actuator_id"]
                )
                values["max_qvel"] = max(values["max_qvel"], abs(float(data.qvel[dof])))
                values["max_error"] = max(
                    values["max_error"],
                    abs(float(data.ctrl[actuator_id] - data.qpos[qpos_address])),
                )
                values["max_force"] = max(
                    values["max_force"], abs(float(data.actuator_force[actuator_id]))
                )
                values["contact"] |= values["body"] in contact_bodies
            if (step_index + 1) % report_every == 0 or step_index + 1 == steps:
                print(
                    f"t={data.time:.3f}s contacts={data.ncon} "
                    f"self_contacts={self_contacts} max_contact_force={maximum_contact_force:.6f}N"
                )

        print("Joint | qpos | qvel | ctrl | actuator_force | qfrc_act | qfrc_constraint | qfrc_bias | error")
        for name, values in statistics.items():
            qpos_address, dof, actuator_id = (
                values["qpos"], values["dof"], values["actuator_id"]
            )
            print(
                f"{name} | {data.qpos[qpos_address]:+.6f} | {data.qvel[dof]:+.6e} | "
                f"{data.ctrl[actuator_id]:+.6f} | {data.actuator_force[actuator_id]:+.6f} | "
                f"{data.qfrc_actuator[dof]:+.6f} | {data.qfrc_constraint[dof]:+.6f} | "
                f"{data.qfrc_bias[dof]:+.6f} | "
                f"{data.ctrl[actuator_id] - data.qpos[qpos_address]:+.6f}"
            )
        print("\nJoint | max abs qvel | max position error | max actuator force | contact involvement")
        for name, values in statistics.items():
            print(
                f"{name} | {values['max_qvel']:.6f} | {values['max_error']:.6f} | "
                f"{values['max_force']:.6f} | {'yes' if values['contact'] else 'no'}"
            )
        final_max_velocity = max(
            abs(float(data.qvel[values["dof"]])) for values in statistics.values()
        )
        scenario_passed = (
            initial_mismatch <= 1e-12
            and max_self_contacts == 0
            and final_max_velocity < 1e-3
        )
        print(
            f"Automated idle result: {'PASS' if scenario_passed else 'FAIL'}; "
            f"max_contacts={max_contacts}, max_self_contacts={max_self_contacts}, "
            f"final_max_abs_qvel={final_max_velocity:.3e} rad/s"
        )
        all_passed &= scenario_passed

    model.opt.gravity[:] = original_gravity
    simulation.reset_open()
    print("Visual no-twitch acceptance remains MANUAL and is not claimed by this diagnostic.")
    return all_passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", choices=sorted(HAND_VARIANTS), default="right")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--headless-check", action="store_true")
    parser.add_argument("--joint-test", action="store_true")
    parser.add_argument("--joint", help="Test one named joint instead of all 20")
    parser.add_argument("--pose-hold-seconds", type=float, default=2.0)
    parser.add_argument("--sensor-verification", action="store_true")
    parser.add_argument("--sensor-mount-calibration", action="store_true")
    parser.add_argument("--collision-verification", action="store_true")
    parser.add_argument(
        "--reconstruction-experiment",
        action="store_true",
        help="Run the right-hand, one-sphere scalar-to-3D contact experiment",
    )
    parser.add_argument(
        "--reconstruction-output",
        type=Path,
        default=DEFAULT_RECONSTRUCTION_OUTPUT,
        help="Ground-truth-free JSON path written by L in reconstruction mode",
    )
    parser.add_argument("--idle-diagnostics", action="store_true")
    parser.add_argument("--idle-seconds", type=float, default=5.0)
    parser.add_argument(
        "--object",
        choices=("none", "sphere", "cylinder", "box"),
        default="none",
    )
    parser.add_argument("--sensor-id", help="Initially selected sensor ID")
    args = parser.parse_args()

    variant = HAND_VARIANTS[args.hand]
    model_path = args.model or variant.output
    if args.model:
        build_model(
            args.model,
            variant.urdf,
            variant.sensors,
            variant.meshes,
            variant.sphere_position,
            variant.joints,
            variant.camera_position,
            variant.camera_xyaxes,
        )
    else:
        build_hand_variant(args.hand)
    simulation = HandSimulation(model_path)
    mounts = load_sensor_mounts(variant.sensors)
    objects = ObjectController(simulation, variant.sphere_position)
    if args.sensor_id and args.sensor_id not in mounts:
        raise ValueError(f"Unknown sensor ID: {args.sensor_id}")

    if args.joint_test:
        objects.activate(None)
        run_joint_test(
            simulation,
            mounts,
            args.joint,
            args.pose_hold_seconds,
            args.headless_check,
        )
        return
    if args.idle_diagnostics:
        if not run_idle_diagnostics(simulation, objects, args.idle_seconds):
            raise RuntimeError("Automated idle stability acceptance failed")
        return
    if args.sensor_verification:
        run_sensor_verification(simulation, mounts, args.hand, objects)
        return
    if args.sensor_mount_calibration:
        run_sensor_mount_calibration(
            simulation, mounts, variant.sensors, objects, args.hand
        )
        return
    if args.collision_verification:
        objects.activate("sphere")
        objects.set_experiment_mode(ObjectController.FIXED_CONTACT_DIAGNOSTIC)
        run_viewer(
            simulation,
            mounts,
            args.sensor_id,
            args.hand,
            objects,
            collision_monitor=True,
        )
        return
    if args.reconstruction_experiment:
        if args.hand != "right":
            raise ValueError("--reconstruction-experiment currently supports only --hand right")
        if args.object not in ("none", "sphere"):
            raise ValueError(
                "--reconstruction-experiment permits only its automatically spawned sphere"
            )
        objects.activate("sphere")
        run_viewer(
            simulation,
            mounts,
            args.sensor_id,
            args.hand,
            objects,
            reconstruction_experiment=True,
            reconstruction_output=args.reconstruction_output,
        )
        return
    if args.headless_check:
        targets = simulation.reference_positions()
        for joint_id, name in enumerate(simulation.joint_names):
            targets[name] += 0.2 * (
                simulation.demonstration_target(joint_id) - targets[name]
            )
        simulation.set_joint_positions(targets)
        simulation.step(10)
        print(
            f"MuJoCo model OK: {simulation.model.nbody} bodies, "
            f"{len(simulation.joint_names)} movable joints, "
            f"{len(simulation.sensor_site_names)} sensor frames"
        )
        return
    objects.activate(None if args.object == "none" else args.object)
    run_viewer(
        simulation,
        mounts,
        args.sensor_id,
        args.hand,
        objects,
    )


if __name__ == "__main__":
    main()
