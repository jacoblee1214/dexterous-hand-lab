"""Run the preregistered reliability-aware fusion v2 sphere experiment."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from backend.dashboard_server import DashboardSimulation
from fusion.config import DEFAULT_CONFIG, V2_CONFIG, load_fusion_config
from fusion.evaluation import evaluate_methods
from fusion.experiment import _degraded_rgb, _select_constraints
from fusion.observation import build_fusion_observation
from fusion.reliability import run_v2_ablations
from fusion.sensitivity import _dropout, _extrinsics, _mount_orientation, _mount_translation
from fusion.sphere import run_fusion_methods
from robot_data.common import RobotCommand, RobotCommandName
from robot_data.mujoco_provider import MuJoCoRobotDataProvider
from simulation.mujoco_sim import ObjectController
from vision.baseline import segment_blue_sphere
from vision.calibration import DEFAULT_CALIBRATION
from vision.research_camera import ResearchRGBCamera


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments/fusion/reliability_v2_20260909"
PROTOCOL = PROJECT_ROOT / "experiments/fusion/reliability_v2_protocol.yaml"
METHODS = (
    "rgb_only", "tactile_only_representative", "fusion_representative",
    "fusion_finite_patch", "reliability_aware_fusion_v2",
)


def _command(provider, name, **parameters):
    provider.send_command(RobotCommand(name, parameters))


def _step(engine, count):
    for _ in range(count):
        engine.step()


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _stats(values):
    values = np.asarray([value for value in values if value is not None], dtype=float)
    return {
        "count": int(len(values)), "mean": float(values.mean()) if len(values) else None,
        "median": float(np.median(values)) if len(values) else None,
        "minimum": float(values.min()) if len(values) else None,
        "maximum": float(values.max()) if len(values) else None,
        "p95": float(np.percentile(values, 95)) if len(values) else None,
    }


def _observation_manifest(observation, v1_observation, rgb_path, mask_path):
    return {
        "algorithm_input_only": True,
        "rgb_reference": str(rgb_path), "predicted_mask_reference": str(mask_path),
        "timestamp": observation.timestamp, "camera_timestamp": observation.camera_timestamp,
        "tactile_timestamp": observation.tactile_timestamp, "time_base": observation.time_base,
        "camera_id": observation.camera_id, "frame_id": observation.frame_id,
        "common_frame_id": observation.common_frame_id,
        "calibration_version": observation.calibration_version,
        "intrinsic_matrix": observation.intrinsic_matrix.tolist(),
        "T_camera_cv_from_world": observation.camera_cv_from_world.tolist(),
        "visibility_pixel_counts": {
            "foreground": int(observation.predicted_object_mask.sum()),
            "background": int(observation.known_background_mask.sum()),
            "unknown_occluded": int(observation.unknown_occluded_mask.sum()),
            "unreliable_segmentation": int(observation.unreliable_segmentation_mask.sum()),
        },
        "v1_visibility_pixel_counts": {
            "foreground": int(v1_observation.predicted_object_mask.sum()),
            "background": int(v1_observation.known_background_mask.sum()),
            "unknown_occluded": int(v1_observation.unknown_occluded_mask.sum()),
            "unreliable_segmentation": int(
                v1_observation.unreliable_segmentation_mask.sum()
            ),
        },
        "segmentation_status": observation.segmentation_status,
        "segmentation_confidence": observation.segmentation_confidence,
        "segmentation_angular_coverage": observation.segmentation_angular_coverage,
        "joint_state": dict(observation.joint_state),
        "known_radius_m": observation.known_radius_m,
        "radius_label": observation.radius_label,
        "object_motion_assumption": observation.object_motion_assumption,
        "tactile_constraints": [
            {
                "timestamp": item.timestamp, "sensor_id": item.sensor_id,
                "finger_id": item.finger_id, "parent_link": item.parent_link,
                "joint_state_snapshot": dict(item.joint_state_snapshot),
                "sensor_type": item.sensor_type,
                "sensor_center_world_m": item.sensor_center_world_m,
                "surface_u_world": item.surface_u_world,
                "surface_v_world": item.surface_v_world,
                "sensing_direction_world": item.sensing_direction_world,
                "representative_contact_world_m": item.representative_contact_world_m,
                "surface_width_m": item.surface_width_m,
                "surface_height_m": item.surface_height_m,
                "fingertip_radii_m": item.fingertip_radii_m,
                "scalar_measurement_n": item.scalar_measurement_n,
                "indentation_m": item.indentation_m,
                "position_sigma_m": item.position_sigma_m,
                "finite_patch_sigma_m": item.finite_patch_sigma_m,
                "activation_valid": item.activation_valid,
                "response_calibration": dict(item.response_calibration),
            }
            for item in observation.tactile_constraints
        ],
    }


def _summarize(rows, methods):
    result = {}
    for method in methods:
        attempted = len(rows)
        valid = [row for row in rows if row["evaluation"]["methods"][method]["success"]]
        result[method] = {
            "all_attempts": {
                "attempt_count": attempted, "success_count": len(valid),
                "failure_count": attempted - len(valid),
                "success_rate": len(valid) / attempted,
                "status_counts": dict(Counter(
                    row["algorithm"]["methods"][method]["status"] for row in rows
                )),
            },
            "valid_center_error_m": _stats(
                row["evaluation"]["methods"][method]["center_error_m"] for row in valid
            ),
            "runtime_ms_all_attempts": _stats(
                row["algorithm"]["methods"][method]["runtime_ms"] for row in rows
            ),
        }
    return result


def generate_diagnostic_report(output: str | Path = DEFAULT_OUTPUT) -> dict:
    """Aggregate saved outputs without rerunning or exposing truth to algorithms."""
    root = Path(output)
    rows = []
    for result_path in sorted((root / "algorithm").glob("*/result.json")):
        case_id = result_path.parent.name
        rows.append({
            "case_id": case_id,
            "condition": case_id.split("_", 2)[2],
            "algorithm": json.loads(result_path.read_text(encoding="utf-8")),
            "evaluation": json.loads(
                (root / "evaluation" / case_id / "metrics.json").read_text(
                    encoding="utf-8"
                )
            ),
        })

    def diagnostic_stats(method: str, rows_subset) -> dict:
        results = [row["algorithm"]["methods"][method] for row in rows_subset]
        diagnostics = [result.get("diagnostics", {}) for result in results]
        return {
            "vision_residual_rms_sigma": _stats(
                result.get("vision_residual_rms_sigma") for result in results
            ),
            "silhouette_residual_rms_px": _stats(
                item.get("vision_residual_rms_px") for item in diagnostics
            ),
            "tactile_residual_rms_sigma": _stats(
                result.get("tactile_residual_rms_sigma") for result in results
            ),
            "tactile_residual_rms_m": _stats(
                item.get("tactile_residual_rms_m") for item in diagnostics
            ),
            "optimizer_condition_number": _stats(
                result.get("condition_number") for result in results
            ),
            "optimizer_minimum_singular_value": _stats(
                result.get("minimum_singular_value") for result in results
            ),
            "contact_count": _stats(item.get("contact_count") for item in diagnostics),
            "unique_sensor_count": _stats(
                item.get("unique_sensor_count") for item in diagnostics
            ),
            "unique_finger_count": _stats(
                item.get("unique_finger_count") for item in diagnostics
            ),
            "spatial_spread_m": _stats(
                item.get("spatial_spread_m") for item in diagnostics
            ),
            "tactile_coverage_condition": _stats(
                item.get("jacobian_condition") for item in diagnostics
            ),
            "runtime_ms": _stats(result.get("runtime_ms") for result in results),
        }

    per_condition = {}
    for condition in sorted({row["condition"] for row in rows}):
        subset = [row for row in rows if row["condition"] == condition]
        per_condition[condition] = {
            method: diagnostic_stats(method, subset) for method in METHODS
        }
    e_diagnostics = [
        row["algorithm"]["methods"]["reliability_aware_fusion_v2"].get(
            "diagnostics", {}
        )
        for row in rows
    ]
    report = {
        "report_version": "reliability-aware-fusion-v2-diagnostics-v1",
        "source": str(root.relative_to(PROJECT_ROOT)),
        "attempt_count": len(rows),
        "truth_boundary": (
            "algorithm/*.json is read independently; center errors are joined from "
            "evaluation/*.json only for this report"
        ),
        "all_attempt_diagnostics": {
            method: diagnostic_stats(method, rows) for method in METHODS
        },
        "v2_reliability_diagnostics": {
            "raw_tactile_observation_count": _stats(
                item.get("raw_tactile_observation_count") for item in e_diagnostics
            ),
            "effective_tactile_observation_count": _stats(
                item.get("effective_tactile_observation_count") for item in e_diagnostics
            ),
            "unreliable_segmentation_pixel_count": _stats(
                item.get("unreliable_segmentation_pixel_count") for item in e_diagnostics
            ),
            "cross_modal_residual_rms_sigma": _stats(
                item.get("tactile_reliability", {}).get(
                    "cross_modal_residual_rms_sigma"
                ) for item in e_diagnostics
            ),
            "tactile_information_fraction_in_weak_rgb_direction": _stats(
                item.get("complementary_information", {}).get(
                    "tactile_fraction_in_weak_rgb_direction"
                ) for item in e_diagnostics
            ),
        },
        "per_condition": per_condition,
        "calibration_sensitivity_note": (
            "Camera and sensor-mount perturbations share the already-invalid strong-"
            "occlusion baseline for v2, so incremental v2 accuracy sensitivity is not "
            "identifiable; the failure floor is preserved rather than omitted."
        ),
        "local_uncertainty_warning": (
            "Jacobian covariance/condition diagnostics are local model quantities, not "
            "complete real-world uncertainty or guaranteed accuracy."
        ),
    }
    _json(root / "diagnostic_summary.json", report)
    return report


def run_experiment(output: str | Path = DEFAULT_OUTPUT, *, trials: int = 3) -> dict:
    root = Path(output)
    if root.exists():
        raise FileExistsError(f"Experiment output already exists: {root}")
    for name in ("inputs", "algorithm", "evaluation"):
        (root / name).mkdir(parents=True, exist_ok=name == "inputs")
    config = load_fusion_config(V2_CONFIG)
    v1_config = load_fusion_config(DEFAULT_CONFIG)
    provider = MuJoCoRobotDataProvider()
    engine = DashboardSimulation(provider=provider)
    camera = ResearchRGBCamera(provider)
    rows = []
    try:
        _command(provider, RobotCommandName.SET_GRASP_PRESET, radius_m=.04)
        _command(provider, RobotCommandName.SET_EXPERIMENT_MODE,
                 mode=ObjectController.FIXED_CONTACT_DIAGNOSTIC)
        camera.render_clean_rgb(engine.current_common_state)
        _json(root / "camera_calibration.json", camera.calibration().to_dict())
        for trial in range(1, trials + 1):
            _command(provider, RobotCommandName.RESET_REFERENCE_POSE)
            _step(engine, 500)
            engine.point_buffer.reset(); engine.point_buffer.start()
            clear = camera.capture(engine); clear_common = engine.current_common_state
            _command(provider, RobotCommandName.EXECUTE_GRASP_STAGE, stage_number=2)
            _step(engine, 2700)
            partial_points = tuple(engine.point_buffer.points)
            partial = camera.capture(engine); partial_common = engine.current_common_state
            _command(provider, RobotCommandName.EXECUTE_GRASP_STAGE, stage_number=4)
            _step(engine, 4500)
            strong_points = tuple(engine.point_buffer.points)
            strong = camera.capture(engine); strong_common = engine.current_common_state
            degraded_rgb = _degraded_rgb(strong.rgb, strong.segmentation_mask, trial)
            degraded_segmentation = segment_blue_sphere(
                degraded_rgb, camera.config.segmentation
            )
            definitions = [
                ("clear_rgb_no_touch", clear, clear_common, clear.rgb, clear.segmentation, ()),
                ("partial_actual_hand_occlusion", partial, partial_common, partial.rgb,
                 partial.segmentation, partial_points),
                ("strong_actual_hand_occlusion", strong, strong_common, strong.rgb,
                 strong.segmentation, strong_points),
                ("controlled_artificial_mask_degradation", strong, strong_common,
                 degraded_rgb, degraded_segmentation, strong_points),
                ("spatially_diverse_tactile", strong, strong_common, strong.rgb,
                 strong.segmentation, strong_points),
                ("clustered_single_finger_tactile", strong, strong_common, strong.rgb,
                 strong.segmentation, _select_constraints(strong_points, "single_finger")),
                ("tactile_channel_dropout_50_percent", strong, strong_common, strong.rgb,
                 strong.segmentation, strong_points),
                ("camera_calibration_exploratory_2mm_1deg", strong, strong_common,
                 strong.rgb, strong.segmentation, strong_points),
                ("sensor_mount_exploratory_2mm_3deg", strong, strong_common,
                 strong.rgb, strong.segmentation, strong_points),
            ]
            for condition, capture, common, rgb, segmentation, tactile in definitions:
                case_id = f"trial_{trial:02d}_{condition}"
                input_dir = root / "inputs" / case_id
                algorithm_dir = root / "algorithm" / case_id
                evaluation_dir = root / "evaluation" / case_id
                input_dir.mkdir(); algorithm_dir.mkdir(); evaluation_dir.mkdir()
                Image.fromarray(rgb).save(input_dir / "rgb.png")
                Image.fromarray(segmentation.mask.astype(np.uint8) * 255).save(
                    input_dir / "predicted_mask.png"
                )
                observation = build_fusion_observation(
                    rgb=rgb, segmentation=segmentation, calibration=capture.calibration,
                    camera_observation=capture.observation, common_state=common,
                    description=provider.description, tactile_observations=tactile,
                    known_radius_m=.04, config=config, fixed_object_pose=True,
                )
                v1_observation = build_fusion_observation(
                    rgb=rgb, segmentation=segmentation, calibration=capture.calibration,
                    camera_observation=capture.observation, common_state=common,
                    description=provider.description, tactile_observations=tactile,
                    known_radius_m=.04, config=v1_config, fixed_object_pose=True,
                )
                if condition == "tactile_channel_dropout_50_percent":
                    observation = _dropout(observation, .50)
                    v1_observation = _dropout(v1_observation, .50)
                elif condition == "camera_calibration_exploratory_2mm_1deg":
                    observation = _extrinsics(observation, (.002, 1.0))
                    v1_observation = _extrinsics(v1_observation, (.002, 1.0))
                elif condition == "sensor_mount_exploratory_2mm_3deg":
                    observation = _mount_orientation(
                        _mount_translation(observation, .002), 3.0
                    )
                    v1_observation = _mount_orientation(
                        _mount_translation(v1_observation, .002), 3.0
                    )
                result = run_fusion_methods(
                    observation, {"sphere": {"estimated_center_world_m": None}}, config,
                    v1_observation=v1_observation, v1_config=v1_config,
                )
                ablations = run_v2_ablations(observation, config)
                _json(algorithm_dir / "result.json", result)
                _json(algorithm_dir / "v2_ablations.json", ablations)
                _json(input_dir / "observation_manifest.json", _observation_manifest(
                    observation, v1_observation,
                    Path("inputs") / case_id / "rgb.png",
                    Path("inputs") / case_id / "predicted_mask.png",
                ))
                truth = dict(common.object_state)["center_xyz"]
                metrics = evaluate_methods(
                    result, ground_truth_center_world_m=truth,
                    ground_truth_visible_mask=capture.ground_truth_mask,
                    calibration=capture.calibration,
                )
                ablation_metrics = evaluate_methods(
                    {"methods": ablations}, ground_truth_center_world_m=truth,
                    ground_truth_visible_mask=capture.ground_truth_mask,
                    calibration=capture.calibration,
                )
                metrics.update({
                    "case_id": case_id, "trial": trial, "condition": condition,
                    "segmentation_iou": int(np.logical_and(
                        segmentation.mask, capture.ground_truth_mask
                    ).sum()) / max(1, int(np.logical_or(
                        segmentation.mask, capture.ground_truth_mask
                    ).sum())),
                    "visible_fraction": capture.evaluation.get("visible_fraction"),
                    "ground_truth_boundary": "evaluation_only_after_all_algorithm_outputs",
                    "v2_ablations": ablation_metrics["methods"],
                })
                _json(evaluation_dir / "metrics.json", metrics)
                rows.append({"case_id": case_id, "condition": condition,
                             "algorithm": result, "evaluation": metrics,
                             "ablations": ablations})
    finally:
        camera.close(); engine.close()

    common = [row for row in rows if all(
        row["evaluation"]["methods"][method]["success"] for method in METHODS
    )]
    conditions = {}
    for condition in sorted({row["condition"] for row in rows}):
        subset = [row for row in rows if row["condition"] == condition]
        conditions[condition] = {
            "attempt_count": len(subset),
            "segmentation_iou": _stats(row["evaluation"]["segmentation_iou"] for row in subset),
            "methods": _summarize(subset, METHODS),
        }
    ablation_names = list(rows[0]["ablations"])
    ablation_summary = {}
    for name in ablation_names:
        success = [row for row in rows if row["evaluation"]["v2_ablations"][name]["success"]]
        ablation_summary[name] = {
            "attempt_count": len(rows), "success_count": len(success),
            "success_rate": len(success) / len(rows),
            "center_error_m": _stats(
                row["evaluation"]["v2_ablations"][name]["center_error_m"] for row in success
            ),
            "status_counts": dict(Counter(row["ablations"][name]["status"] for row in rows)),
        }
    summary = {
        "report_version": "reliability-aware-fusion-v2-results-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(PROTOCOL.relative_to(PROJECT_ROOT)),
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "configuration": str(V2_CONFIG.relative_to(PROJECT_ROOT)),
        "configuration_sha256": hashlib.sha256(V2_CONFIG.read_bytes()).hexdigest(),
        "v1_configuration": str(DEFAULT_CONFIG.relative_to(PROJECT_ROOT)),
        "v1_configuration_sha256": hashlib.sha256(
            DEFAULT_CONFIG.read_bytes()
        ).hexdigest(),
        "attempt_count": len(rows), "methods": _summarize(rows, METHODS),
        "common_valid_all_methods": {
            "count": len(common), "case_ids": [row["case_id"] for row in common],
            "center_error_m": {
                method: _stats(row["evaluation"]["methods"][method]["center_error_m"]
                               for row in common)
                for method in METHODS
            },
        },
        "paired_delta_v2_minus_baseline_m": {
            baseline: _stats(
                row["evaluation"]["methods"]["reliability_aware_fusion_v2"]["center_error_m"]
                - row["evaluation"]["methods"][baseline]["center_error_m"]
                for row in rows
                if row["evaluation"]["methods"][baseline]["success"]
                and row["evaluation"]["methods"]["reliability_aware_fusion_v2"]["success"]
            )
            for baseline in ("rgb_only", "fusion_representative", "fusion_finite_patch")
        },
        "v2_decision_counts": dict(Counter(
            row["algorithm"]["methods"]["reliability_aware_fusion_v2"]["status"]
            for row in rows
        )),
        "per_condition": conditions,
        "ablations": ablation_summary,
        "radius_error_m": None,
        "radius_error_reason": "40 mm radius is supplied as a prior; only center is estimated",
        "truth_isolation": "MuJoCo truth is present only in evaluation/*.json",
    }
    _json(root / "dataset_manifest.json", {
        "dataset_version": "reliability-aware-fusion-v2-sphere-simulation-v1",
        "case_count": len(rows), "case_ids": [row["case_id"] for row in rows],
        "input_manifest": "inputs/<case_id>/observation_manifest.json",
        "algorithm_output": "algorithm/<case_id>/result.json",
        "ablation_output": "algorithm/<case_id>/v2_ablations.json",
        "evaluation_output": "evaluation/<case_id>/metrics.json",
        "raw_image_policy": "PNG inputs are local artifact-managed and Git-ignored",
        "protocol_sha256": summary["protocol_sha256"],
        "configuration_sha256": summary["configuration_sha256"],
        "v1_configuration_sha256": summary["v1_configuration_sha256"],
    })
    _json(root / "evaluation_summary.json", summary)
    generate_diagnostic_report(root)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        result = generate_diagnostic_report(args.output)
        print(json.dumps({"output": str(args.output),
                          "attempt_count": result["attempt_count"]}, indent=2))
        return
    result = run_experiment(args.output, trials=args.trials)
    print(json.dumps({"output": str(args.output), "attempt_count": result["attempt_count"],
                      "v2_decisions": result["v2_decision_counts"]}, indent=2))


if __name__ == "__main__":
    main()
