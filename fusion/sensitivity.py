"""Controlled sim-to-real sensitivity study for geometric fusion v1.

All perturbation ranges are explicitly exploratory until measured hardware
specifications are supplied. Simulator truth is used only after each solve.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from backend.dashboard_server import DashboardSimulation
from fusion.config import load_fusion_config
from fusion.observation import FusionObservation, TactileContactConstraint, build_fusion_observation
from fusion.sphere import run_fusion_methods
from robot_data.common import RobotCommand, RobotCommandName
from robot_data.mujoco_provider import MuJoCoRobotDataProvider
from sensors.sensor_kinematics import KinematicTree, load_sensor_mounts
from simulation.mujoco_sim import ObjectController
from vision.research_camera import ResearchRGBCamera


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments/fusion/sensitivity_v1_20260909"
METHODS = (
    "rgb_only", "tactile_only_representative",
    "fusion_representative", "fusion_finite_patch",
)


def _command(provider, name, **parameters):
    provider.send_command(RobotCommand(name, parameters))


def _step(engine, count):
    for _ in range(count):
        engine.step()


def _seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big")


def _unit(label: str) -> np.ndarray:
    rng = np.random.default_rng(_seed(label))
    value = rng.normal(size=3)
    return value / np.linalg.norm(value)


def _rotation(axis, angle_rad):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle_rad) * skew + (1 - math.cos(angle_rad)) * (skew @ skew)


def _rigid_contact(item, *, translation=np.zeros(3), rotation=np.eye(3), indentation=None):
    center = np.asarray(item.sensor_center_world_m) + translation
    u = rotation @ np.asarray(item.surface_u_world)
    v = rotation @ np.asarray(item.surface_v_world)
    normal = rotation @ np.asarray(item.sensing_direction_world)
    if indentation is None:
        representative = center + rotation @ (
            np.asarray(item.representative_contact_world_m) - np.asarray(item.sensor_center_world_m)
        )
        indentation = item.indentation_m
    else:
        representative = center + (
            item.response_calibration["contact_projection_sign"] * indentation * normal
        )
    return replace(
        item,
        sensor_center_world_m=tuple(center),
        surface_u_world=tuple(u),
        surface_v_world=tuple(v),
        sensing_direction_world=tuple(normal),
        representative_contact_world_m=tuple(representative),
        indentation_m=float(indentation),
    )


def _mount_translation(observation, level):
    return replace(observation, tactile_constraints=tuple(
        _rigid_contact(item, translation=_unit("mount-t-" + item.sensor_id) * level)
        for item in observation.tactile_constraints
    ))


def _mount_orientation(observation, level_deg):
    return replace(observation, tactile_constraints=tuple(
        _rigid_contact(item, rotation=_rotation(_unit("mount-r-" + item.sensor_id), math.radians(level_deg)))
        for item in observation.tactile_constraints
    ))


def _recompute_from_joint_state(item, joint_state, tree, mounts):
    mount = mounts[item.sensor_id]
    transform = tree.forward_kinematics(joint_state)[mount.parent_link] @ mount.link_to_sensor
    rotation = transform[:3, :3]
    center = transform[:3, 3]
    representative = center + (
        item.response_calibration["contact_projection_sign"] * item.indentation_m * rotation[:, 2]
    )
    return replace(
        item, joint_state_snapshot=joint_state,
        sensor_center_world_m=tuple(center), surface_u_world=tuple(rotation[:, 0]),
        surface_v_world=tuple(rotation[:, 1]), sensing_direction_world=tuple(rotation[:, 2]),
        representative_contact_world_m=tuple(representative),
    )


def _joint_perturbation(observation, level_deg, tree, mounts, *, noise):
    changed = []
    for index, item in enumerate(observation.tactile_constraints):
        state = dict(item.joint_state_snapshot)
        rng = np.random.default_rng(_seed(f"joint-{noise}-{index}-{item.sensor_id}"))
        for name in sorted(state):
            if noise:
                delta = rng.normal(0, math.radians(level_deg))
            else:
                sign = -1 if _seed("joint-bias-" + name) % 2 else 1
                delta = sign * math.radians(level_deg)
            state[name] += delta
        changed.append(_recompute_from_joint_state(item, state, tree, mounts))
    return replace(observation, tactile_constraints=tuple(changed))


def _scalar_calibration(observation, relative_level):
    changed = []
    for item in observation.tactile_constraints:
        sign = -1 if _seed("scalar-" + item.sensor_id) % 2 else 1
        indentation = max(0.0, item.indentation_m * (1 + sign * relative_level))
        changed.append(_rigid_contact(item, indentation=indentation))
    return replace(observation, tactile_constraints=tuple(changed))


def _dropout(observation, fraction):
    sensor_ids = sorted({item.sensor_id for item in observation.tactile_constraints}, key=lambda x: _seed("drop-" + x))
    drop_count = int(round(len(sensor_ids) * fraction))
    dropped = set(sensor_ids[:drop_count])
    return replace(observation, tactile_constraints=tuple(
        item for item in observation.tactile_constraints if item.sensor_id not in dropped
    ))


def _intrinsics(observation, relative_level):
    intrinsic = observation.intrinsic_matrix.copy()
    intrinsic[0, 0] *= 1 + relative_level
    intrinsic[1, 1] *= 1 + relative_level
    return replace(observation, intrinsic_matrix=intrinsic)


def _extrinsics(observation, level):
    translation_m, rotation_deg = level
    delta = np.eye(4)
    delta[:3, :3] = _rotation(_unit("camera-extrinsic-axis"), math.radians(rotation_deg))
    delta[:3, 3] = _unit("camera-extrinsic-translation") * translation_m
    return replace(observation, camera_cv_from_world=delta @ observation.camera_cv_from_world)


def _mask_degradation(observation, fraction):
    mask = observation.predicted_object_mask.copy()
    rows, columns = np.nonzero(mask)
    if len(columns) and fraction:
        cutoff = np.quantile(columns, 1 - fraction)
        removed = mask & (np.indices(mask.shape)[1] >= cutoff)
        mask[removed] = False
        background = observation.known_background_mask.copy()
        background[removed] = True
    else:
        background = observation.known_background_mask.copy()
    return replace(observation, predicted_object_mask=mask, known_background_mask=background)


def _timestamp_mismatch(observation, offset_s, tree, mounts):
    contacts = observation.tactile_constraints
    if not contacts or not offset_s:
        return observation
    times = np.asarray([item.timestamp for item in contacts])
    changed = []
    for item in contacts:
        source = contacts[int(np.argmin(np.abs(times - (item.timestamp + offset_s))))]
        changed.append(_recompute_from_joint_state(item, source.joint_state_snapshot, tree, mounts))
    return replace(observation, tactile_constraints=tuple(changed))


def _stats(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    return {"count": int(len(values)), "mean": float(values.mean()) if len(values) else None,
            "maximum": float(values.max()) if len(values) else None}


def run_sensitivity(output: str | Path = DEFAULT_OUTPUT, *, trials: int = 3) -> dict:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Sensitivity output already exists: {output}")
    output.mkdir(parents=True)
    config = load_fusion_config()
    provider = MuJoCoRobotDataProvider()
    engine = DashboardSimulation(provider=provider)
    camera = ResearchRGBCamera(provider)
    tree = KinematicTree.from_urdf(provider.description.kinematic_model_path)
    mounts = load_sensor_mounts(provider.description.sensor_config_path)
    observations = []
    try:
        _command(provider, RobotCommandName.SET_GRASP_PRESET, radius_m=.04)
        _command(provider, RobotCommandName.SET_EXPERIMENT_MODE,
                 mode=ObjectController.FIXED_CONTACT_DIAGNOSTIC)
        for trial in range(1, trials + 1):
            _command(provider, RobotCommandName.RESET_REFERENCE_POSE)
            _step(engine, 500)
            engine.point_buffer.reset(); engine.point_buffer.start()
            _command(provider, RobotCommandName.EXECUTE_GRASP_STAGE, stage_number=2)
            _step(engine, 2700)
            _command(provider, RobotCommandName.EXECUTE_GRASP_STAGE, stage_number=4)
            _step(engine, 4500)
            capture = camera.capture(engine)
            common = engine.current_common_state
            observation = build_fusion_observation(
                rgb=capture.rgb, segmentation=capture.segmentation,
                calibration=capture.calibration, camera_observation=capture.observation,
                common_state=common, description=provider.description,
                tactile_observations=tuple(engine.point_buffer.points), known_radius_m=.04,
                config=config, fixed_object_pose=True,
            )
            observations.append((trial, observation, np.asarray(dict(common.object_state)["center_xyz"])))
    finally:
        camera.close(); engine.close()

    families = {
        "sensor_mount_translation_m": ([0, .001, .002, .005], lambda o, x: _mount_translation(o, x)),
        "sensor_mount_orientation_deg": ([0, 1, 3, 5], lambda o, x: _mount_orientation(o, x)),
        "joint_encoder_bias_deg": ([0, .25, .5, 1], lambda o, x: _joint_perturbation(o, x, tree, mounts, noise=False)),
        "joint_encoder_noise_std_deg": ([0, .25, .5, 1], lambda o, x: _joint_perturbation(o, x, tree, mounts, noise=True)),
        "scalar_calibration_relative_error": ([0, .05, .10, .20], lambda o, x: _scalar_calibration(o, x)),
        "tactile_channel_dropout_fraction": ([0, .25, .50, .75], lambda o, x: _dropout(o, x)),
        "camera_focal_length_relative_error": ([0, .01, .02, .05], lambda o, x: _intrinsics(o, x)),
        "camera_extrinsic_translation_m_rotation_deg": (
            [(0, 0), (.001, .5), (.002, 1), (.005, 2)], lambda o, x: _extrinsics(o, x)
        ),
        "rgb_mask_removed_fraction": ([0, .10, .25, .40], lambda o, x: _mask_degradation(o, x)),
        "encoder_tactile_timestamp_mismatch_s": ([0, .010, .025, .050], lambda o, x: _timestamp_mismatch(o, x, tree, mounts)),
    }
    run_rows = []
    for family, (levels, perturb) in families.items():
        for level_index, level in enumerate(levels):
            for trial, observation, truth in observations:
                changed = perturb(observation, level)
                # No baseline center is carried across calibration perturbations.
                result = run_fusion_methods(changed, {"sphere": {"estimated_center_world_m": None}}, config)
                for method in METHODS:
                    method_result = result["methods"][method]
                    center = method_result["estimated_center_world_m"]
                    run_rows.append({
                        "family": family, "level_index": level_index,
                        "level": list(level) if isinstance(level, tuple) else level,
                        "trial": trial, "method": method,
                        "success": bool(method_result["valid"]),
                        "failure_status": None if method_result["valid"] else method_result["status"],
                        "center_error_m": float(np.linalg.norm(np.asarray(center) - truth)) if center else None,
                        "condition_number": method_result["condition_number"],
                        "validity_reason": method_result["validity_reason"],
                    })
    summary = {}
    for family, (levels, _) in families.items():
        summary[family] = []
        for level_index, level in enumerate(levels):
            block = {"level_index": level_index,
                     "level": list(level) if isinstance(level, tuple) else level,
                     "methods": {}}
            for method in METHODS:
                rows = [row for row in run_rows if row["family"] == family
                        and row["level_index"] == level_index and row["method"] == method]
                valid = [row for row in rows if row["success"]]
                block["methods"][method] = {
                    "attempt_count": len(rows), "success_count": len(valid),
                    "failure_rate": (len(rows) - len(valid)) / len(rows),
                    "center_error_m": _stats(row["center_error_m"] for row in valid),
                    "condition_number": _stats(row["condition_number"] for row in valid),
                    "failure_statuses": sorted({row["failure_status"] for row in rows if row["failure_status"]}),
                }
            summary[family].append(block)
    metadata = {
        "analysis_version": "sim-to-real-sensitivity-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "fresh deterministic simulation repetitions of the committed strong_occlusion protocol",
        "trials": trials, "sphere_radius_m": .04,
        "range_status": "EXPLORATORY_NOT_HARDWARE_JUSTIFIED",
        "range_reason": "No real encoder, sensor, camera, or timestamp accuracy specification has been supplied.",
        "baseline_policy": "Level zero is unperturbed; fusion/config.yaml and modality weights are unchanged.",
        "algorithm_input_boundary": "Perturbed RGB/calibration/named joint/scalar tactile inputs only; no simulator truth.",
        "evaluation_boundary": "MuJoCo sphere center is read only after solving to compute center error.",
        "timestamp_model": "Offset selects the nearest recorded named joint snapshot at tactile_time + offset and recomputes sensor geometry.",
        "camera_extrinsic_level_format": "[translation_m, rotation_deg]",
        "input_sha256": {
            str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                PROJECT_ROOT / "fusion/config.yaml",
                PROJECT_ROOT / "vision/camera_calibration.yaml",
                PROJECT_ROOT / "sensors/sensor_config.yaml",
                PROJECT_ROOT / "sensors/pressure_config.yaml",
                PROJECT_ROOT / "simulation/grasp_config_right.yaml",
            )
        },
    }
    (output / "experiment_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "run_diagnostics.json").write_text(json.dumps(run_rows, indent=2, allow_nan=False), encoding="utf-8")
    (output / "sensitivity_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return {"output": str(output), "run_count": len(run_rows), "summary": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    result = run_sensitivity(args.output, trials=args.trials)
    print(json.dumps({"output": result["output"], "run_count": result["run_count"]}, indent=2))


if __name__ == "__main__":
    main()
