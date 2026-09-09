"""Run the fixed-pose 40 mm geometric RGB/tactile fusion v1 matrix."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from backend.dashboard_server import DashboardSimulation
from fusion.config import DEFAULT_CONFIG, load_fusion_config
from fusion.evaluation import evaluate_methods
from fusion.observation import build_fusion_observation
from fusion.sphere import run_fusion_methods
from robot_data.common import RobotCommand, RobotCommandName
from robot_data.mujoco_provider import MuJoCoRobotDataProvider
from robot_data.recording import CommonStateRecorder
from simulation.mujoco_sim import ObjectController
from vision.baseline import fit_known_radius_sphere, segment_blue_sphere
from vision.calibration import DEFAULT_CALIBRATION
from vision.research_camera import ResearchRGBCamera


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "fusion" / "geometric_v1_20260909"


def _json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def _command(provider, name, **parameters):
    provider.send_command(RobotCommand(name, parameters))


def _step(engine, count):
    for _ in range(count):
        engine.step()


def _degraded_rgb(rgb: np.ndarray, mask: np.ndarray, trial: int) -> np.ndarray:
    """Deterministically remove 35--45% of the predicted blue region."""
    changed = rgb.copy()
    rows, columns = np.nonzero(mask)
    if len(columns):
        fraction = .55 + .05*((trial-1) % 3)
        cutoff = int(np.quantile(columns, fraction))
        removed = mask & (np.indices(mask.shape)[1] >= cutoff)
        changed[removed] = [18, 24, 30]
    return changed


def _select_constraints(points, mode):
    points = tuple(points)
    if mode == "none":
        return ()
    if mode == "all":
        return points
    if mode == "sparse":
        chosen, sensors = [], set()
        for item in points:
            if item.sensor_id not in sensors:
                chosen.append(item); sensors.add(item.sensor_id)
            if len(chosen) == 3:
                break
        return tuple(chosen)
    if mode == "single_finger":
        if not points:
            return ()
        finger = points[0].finger_id
        return tuple(item for item in points if item.finger_id == finger)
    raise ValueError(mode)


def _statistics(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "rmse": None, "p95": None}
    return {"count": len(values), "mean": float(values.mean()),
            "median": float(np.median(values)), "rmse": float(np.sqrt(np.mean(values*values))),
            "p95": float(np.percentile(values, 95))}


def run_experiment(output: str | Path = DEFAULT_OUTPUT, *, trials: int = 3) -> dict:
    root = Path(output)
    if root.exists():
        raise FileExistsError(f"Experiment directory already exists: {root}")
    inputs, algorithm, evaluation = root/"inputs", root/"algorithm", root/"evaluation"
    for directory in (inputs, algorithm, evaluation):
        directory.mkdir(parents=True, exist_ok=False)
    config = load_fusion_config()
    provider = MuJoCoRobotDataProvider()
    engine = DashboardSimulation(provider=provider)
    camera = ResearchRGBCamera(provider)
    rows = []
    try:
        _command(provider, RobotCommandName.SET_GRASP_PRESET, radius_m=.04)
        _command(provider, RobotCommandName.SET_EXPERIMENT_MODE,
                 mode=ObjectController.FIXED_CONTACT_DIAGNOSTIC)
        camera.render_clean_rgb(engine.current_common_state)
        _json(root/"camera_calibration.json", camera.calibration().to_dict())
        _json(root/"experiment_config.json", {
            "experiment": "geometric_visuo_tactile_fusion_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "sphere_radius_m": .04, "sphere_diameter_m": .08,
            "radius_role": "declared_known_radius_prior_not_estimated",
            "object_motion": "fixed_contact_diagnostic; no moving-object samples mixed",
            "trial_count": trials,
            "conditions_per_trial": ["clear_no_touch", "partial_occlusion",
                "strong_occlusion", "controlled_mask_degradation", "sparse_touch",
                "single_finger_insufficient_distribution"],
            "algorithm_inputs": ["clean/calibrated RGB", "predicted object mask",
                "known background/unknown visibility classes", "named synchronized joints",
                "18-channel scalar tactile measurements", "sensor/calibration transforms",
                "declared radius prior"],
            "excluded_algorithm_inputs": ["MuJoCo object center/pose", "MuJoCo contact position",
                "MuJoCo contact normal", "MuJoCo segmentation", "depth", "object mesh"],
            "fusion_config_sha256": hashlib.sha256(DEFAULT_CONFIG.read_bytes()).hexdigest(),
            "camera_config_sha256": hashlib.sha256(DEFAULT_CALIBRATION.read_bytes()).hexdigest(),
            "method_matrix": {"A": "rgb_only", "B": "tactile_only_representative",
                "C": "fusion_representative", "D": "fusion_finite_patch"},
        })
        with CommonStateRecorder(root/"observations.jsonl", provider.description) as recorder:
            for trial in range(1, trials+1):
                _command(provider, RobotCommandName.RESET_REFERENCE_POSE)
                _step(engine, 500)
                engine.point_buffer.reset(); engine.point_buffer.start()
                clear = camera.capture(engine)
                clear_common = engine.current_common_state
                _command(provider, RobotCommandName.EXECUTE_GRASP_STAGE, stage_number=2)
                _step(engine, 2700)
                partial_points = tuple(engine.point_buffer.points)
                partial = camera.capture(engine)
                partial_common = engine.current_common_state
                _command(provider, RobotCommandName.EXECUTE_GRASP_STAGE, stage_number=4)
                _step(engine, 4500)
                all_points = tuple(engine.point_buffer.points)
                strong = camera.capture(engine)
                strong_common = engine.current_common_state
                degraded_rgb = _degraded_rgb(strong.rgb, strong.segmentation_mask, trial)
                degraded_segmentation = segment_blue_sphere(degraded_rgb, camera.config.segmentation)
                degraded_vision = {**strong.vision_result,
                    "segmentation": degraded_segmentation.diagnostics(),
                    "sphere": fit_known_radius_sphere(degraded_segmentation, strong.calibration, .04)}
                cases = [
                    ("clear_no_touch", clear, clear_common, clear.rgb, clear.segmentation,
                     clear.vision_result, (), "none"),
                    ("partial_occlusion", partial, partial_common, partial.rgb,
                     partial.segmentation, partial.vision_result, partial_points, "all"),
                    ("strong_occlusion", strong, strong_common, strong.rgb,
                     strong.segmentation, strong.vision_result, all_points, "all"),
                    ("controlled_mask_degradation", strong, strong_common, degraded_rgb,
                     degraded_segmentation, degraded_vision, all_points, "all"),
                    ("sparse_touch", strong, strong_common, strong.rgb, strong.segmentation,
                     strong.vision_result, all_points, "sparse"),
                    ("single_finger_insufficient_distribution", strong, strong_common,
                     strong.rgb, strong.segmentation, strong.vision_result, all_points,
                     "single_finger"),
                ]
                for condition, capture, common, rgb, segmentation, vision_result, available, tactile_mode in cases:
                    name = f"trial_{trial:02d}_{condition}"
                    in_dir, alg_dir, eval_dir = inputs/name, algorithm/name, evaluation/name
                    in_dir.mkdir(); alg_dir.mkdir(); eval_dir.mkdir()
                    Image.fromarray(rgb).save(in_dir/"rgb.png")
                    Image.fromarray(segmentation.mask.astype(np.uint8)*255).save(in_dir/"predicted_mask.png")
                    tactile = _select_constraints(available, tactile_mode)
                    observation = build_fusion_observation(
                        rgb=rgb, segmentation=segmentation, calibration=capture.calibration,
                        camera_observation=capture.observation, common_state=common,
                        description=provider.description, tactile_observations=tactile,
                        known_radius_m=.04, config=config, fixed_object_pose=True,
                    )
                    result = run_fusion_methods(observation, vision_result, config)
                    _json(alg_dir/"result.json", result)
                    _json(in_dir/"observation_summary.json", {
                        "camera_timestamp": observation.camera_timestamp,
                        "tactile_timestamps": sorted({item.timestamp for item in observation.tactile_constraints}),
                        "time_base": observation.time_base,
                        "object_motion_assumption": observation.object_motion_assumption,
                        "joint_names": sorted(observation.joint_state),
                        "tactile_contact_count": len(observation.tactile_constraints),
                        "unique_sensor_ids": sorted({item.sensor_id for item in observation.tactile_constraints}),
                        "unique_finger_ids": sorted({item.finger_id for item in observation.tactile_constraints}),
                        "predicted_mask_pixels": int(observation.predicted_object_mask.sum()),
                        "unknown_occluded_pixels": int(observation.unknown_occluded_mask.sum()),
                        "known_radius_m": observation.known_radius_m,
                    })
                    true_center = dict(common.object_state)["center_xyz"]
                    metrics = evaluate_methods(result,
                        ground_truth_center_world_m=true_center,
                        ground_truth_visible_mask=capture.ground_truth_mask,
                        calibration=capture.calibration)
                    metrics.update({"trial": trial, "condition": condition,
                                    "ground_truth_source": "MuJoCo evaluation path only",
                                    "segmentation_iou": (int(np.logical_and(segmentation.mask, capture.ground_truth_mask).sum()) /
                                        max(1, int(np.logical_or(segmentation.mask, capture.ground_truth_mask).sum()))),
                                    "measured_visible_fraction": capture.evaluation.get("visible_fraction")})
                    _json(eval_dir/"metrics.json", metrics)
                    recorded_observation = replace(capture.observation,
                        rgb_reference=str((in_dir/"rgb.png").relative_to(root)),
                        intrinsic_calibration_reference="camera_calibration.json#intrinsics",
                        extrinsics_reference="camera_calibration.json#T_world_from_camera_cv")
                    recorder.record(replace(common, camera=recorded_observation))
                    rows.append({"trial": trial, "condition": condition,
                                 "algorithm": result, "evaluation": metrics})
        method_names = list(rows[0]["algorithm"]["methods"])
        method_summary = {}
        for method in method_names:
            attempted = len(rows)
            valid = [row for row in rows if row["evaluation"]["methods"][method]["success"]]
            center_errors = [row["evaluation"]["methods"][method]["center_error_m"] for row in valid]
            ious = [row["evaluation"]["methods"][method]["visible_silhouette_iou"] for row in valid]
            runtimes = [row["algorithm"]["methods"][method]["runtime_ms"] for row in rows]
            tactile_residuals = [row["algorithm"]["methods"][method]["tactile_residual_rms_sigma"]
                                 for row in valid if row["algorithm"]["methods"][method]["tactile_residual_rms_sigma"] is not None]
            method_summary[method] = {
                "all_attempts": {"count": attempted, "success_count": len(valid),
                    "failure_count": attempted-len(valid), "failure_rate": (attempted-len(valid))/attempted,
                    "center_error_aggregate": None,
                    "reason": "failed attempts have no center; see valid_only and failure_rate"},
                "valid_only": {"center_error_m": _statistics(center_errors),
                    "visible_silhouette_iou": _statistics(ious),
                    "tactile_residual_rms_sigma": _statistics(tactile_residuals)},
                "runtime_ms_all_attempts": _statistics(runtimes),
                "failure_status_counts": {status: sum(
                    row["algorithm"]["methods"][method]["status"] == status for row in rows)
                    for status in sorted({row["algorithm"]["methods"][method]["status"] for row in rows})},
            }
        comparisons = {}
        rgb_name = "rgb_only"
        for method in ("fusion_representative", "fusion_finite_patch"):
            paired = [(row["evaluation"]["methods"][rgb_name]["center_error_m"],
                       row["evaluation"]["methods"][method]["center_error_m"])
                      for row in rows if row["evaluation"]["methods"][rgb_name]["success"]
                      and row["evaluation"]["methods"][method]["success"]]
            deltas = [fusion-rgb for rgb, fusion in paired]
            mean_delta = float(np.mean(deltas)) if deltas else None
            comparisons[method] = {"paired_valid_count": len(paired),
                "fusion_minus_rgb_center_error_m": _statistics(deltas),
                "conclusion": ("improved" if mean_delta is not None and mean_delta < -1e-6 else
                    "degraded" if mean_delta is not None and mean_delta > 1e-6 else
                    "no material difference" if mean_delta is not None else "insufficient paired valid estimates")}
        _json(root/"algorithm_summary.json", {"attempt_count": len(rows),
            "methods": {method: {"statuses": [row["algorithm"]["methods"][method]["status"] for row in rows]}
                        for method in method_names}})
        _json(root/"evaluation_summary.json", {"boundary": "evaluation_only",
            "attempt_count": len(rows), "method_summary": method_summary,
            "paired_comparison_to_rgb": comparisons,
            "radius_error_m": None,
            "radius_error_reason": "known 40 mm radius was supplied, not estimated"})
        return {"output": str(root), "attempt_count": len(rows),
                "paired_comparison_to_rgb": comparisons}
    finally:
        camera.close(); engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(run_experiment(args.output, trials=args.trials), indent=2))


if __name__ == "__main__":
    main()
