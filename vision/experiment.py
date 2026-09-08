"""Run the reproducible 40 mm calibrated RGB sphere baseline experiment."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from robot_data.common import RobotCommand, RobotCommandName
from robot_data.mujoco_provider import MuJoCoRobotDataProvider
from robot_data.recording import CommonStateRecorder
from simulation.mujoco_sim import ObjectController, RIGHT_GRASP_CONFIG
from vision.calibration import DEFAULT_CALIBRATION
from vision.research_camera import ResearchRGBCamera


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "vision_baseline" / "rgb_milestone1_20260908"


def _json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )


def _command(provider, name: RobotCommandName, **parameters) -> None:
    provider.send_command(RobotCommand(name, parameters))


def _settle(provider, maximum_steps: int = 3000) -> None:
    for _ in range(maximum_steps):
        provider.step()
        if provider.grasp.maximum_hand_joint_speed() < 0.02:
            # Continue for the configured dwell instead of accepting one quiet sample.
            quiet_steps = round(
                provider.grasp.library.settling_dwell_seconds
                / provider.simulation.model.opt.timestep
            )
            for _ in range(quiet_steps):
                provider.step()
            if provider.grasp.maximum_hand_joint_speed() < 0.02:
                return


def run_experiment(output: str | Path = DEFAULT_OUTPUT) -> dict:
    root = Path(output)
    if root.exists():
        raise FileExistsError(f"Experiment directory already exists: {root}")
    inputs, results, evaluation = root / "inputs", root / "algorithm", root / "evaluation"
    for directory in (inputs, results, evaluation):
        directory.mkdir(parents=True, exist_ok=False)

    provider = MuJoCoRobotDataProvider()
    camera = ResearchRGBCamera(provider)
    from backend.dashboard_server import DashboardSimulation

    engine = DashboardSimulation(provider=provider)
    try:
        _command(provider, RobotCommandName.SET_GRASP_PRESET, radius_m=0.04)
        _command(
            provider,
            RobotCommandName.SET_EXPERIMENT_MODE,
            mode=ObjectController.FIXED_CONTACT_DIAGNOSTIC,
        )
        _command(provider, RobotCommandName.RESET_REFERENCE_POSE)
        _settle(provider)
        engine._update_sensor_pipeline(force=True)

        # Populate the dedicated camera data from the authoritative simulation
        # before serializing its pose.  A newly allocated mjData otherwise still
        # contains MuJoCo's zero/default derived camera fields.
        camera.render_clean_rgb(engine.current_common_state)
        calibration = camera.calibration()
        _json(root / "camera_calibration.json", calibration.to_dict())
        configuration = {
            "experiment": "calibrated_rgb_vision_only_sphere_baseline",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "sphere_radius_m": 0.04,
            "radius_role": "declared_known_radius_prior_not_estimated",
            "unknown_radius_setting": "reported_scale_ambiguous",
            "camera_calibration_version": camera.config.calibration_version,
            "camera_config_sha256": hashlib.sha256(DEFAULT_CALIBRATION.read_bytes()).hexdigest(),
            "grasp_config_sha256": hashlib.sha256(RIGHT_GRASP_CONFIG.read_bytes()).hexdigest(),
            "grasp_configuration": (
                "sphere_40mm existing automatic preset; development configuration only; "
                "no user-approved taught pose was present"
            ),
            "algorithm_input": ["clean RGB", "camera calibration", "declared radius prior"],
            "excluded_algorithm_inputs": [
                "MuJoCo segmentation IDs",
                "depth",
                "ground-truth object center",
                "ground-truth object geometry",
                "tactile measurements",
            ],
        }
        _json(root / "experiment_config.json", configuration)

        conditions = []

        def capture_condition(name: str, declared_condition: str) -> None:
            engine._update_sensor_pipeline(force=True)
            capture = camera.capture(engine)
            trial_input = inputs / name
            trial_algorithm = results / name
            trial_evaluation = evaluation / name
            for directory in (trial_input, trial_algorithm, trial_evaluation):
                directory.mkdir()
            (trial_input / "rgb.jpg").write_bytes(capture.frame.jpeg)
            Image.fromarray(capture.segmentation_mask.astype(np.uint8) * 255).save(
                trial_algorithm / "predicted_mask.png"
            )
            Image.fromarray(capture.ground_truth_mask.astype(np.uint8) * 255).save(
                trial_evaluation / "ground_truth_mask.png"
            )
            observation = replace(
                capture.observation,
                rgb_reference=str((trial_input / "rgb.jpg").relative_to(root)),
                intrinsic_calibration_reference="camera_calibration.json#intrinsics",
                extrinsics_reference="camera_calibration.json#T_world_from_camera_cv",
            )
            provider.publish_camera_observation(observation)
            common = provider.read_common_state()
            _json(trial_algorithm / "result.json", capture.vision_result)
            _json(
                trial_evaluation / "metrics.json",
                {
                    **capture.evaluation,
                    "declared_condition": declared_condition,
                    "ground_truth_source": "MuJoCo evaluation-only segmentation and object pose",
                },
            )
            recorder.record(common)
            conditions.append(
                {
                    "name": name,
                    "declared_condition": declared_condition,
                    "algorithm": capture.vision_result,
                    "evaluation": capture.evaluation,
                }
            )

        with CommonStateRecorder(root / "observations.jsonl", provider.description) as recorder:
            capture_condition("center_reference", "centered sphere, reference hand")

            provider.objects.set_spawn_position(np.array([0.005, -0.070, 0.115]))
            provider.objects.reset()
            _settle(provider, 300)
            capture_condition("offset_reference", "controlled +20 mm world-X object position")

            provider.objects.set_spawn_position(
                provider.grasp.library.presets["sphere_40mm"].sphere_center_xyz
            )
            provider.objects.reset()
            _command(provider, RobotCommandName.RESET_REFERENCE_POSE)
            _settle(provider)
            _command(provider, RobotCommandName.EXECUTE_GRASP_STAGE, stage_number=2)
            _settle(provider)
            capture_condition(
                "development_approach",
                "partially occluded by thumb/index/middle development approach",
            )

            _command(provider, RobotCommandName.RESET_REFERENCE_POSE)
            _settle(provider)
            _command(provider, RobotCommandName.GRASP)
            for _ in range(6000):
                provider.step()
                if not provider.grasp.running:
                    break
            capture_condition(
                "development_settled_grasp",
                "occluded by existing automatic development grasp; not user-approved",
            )

        algorithm_rows = [
            {
                "condition": item["name"],
                "segmentation_status": item["algorithm"]["segmentation"]["status"],
                "sphere_status": item["algorithm"]["sphere"]["status"],
                "radius_label": item["algorithm"]["sphere"]["radius_label"],
                "silhouette_residual_pixels": item["algorithm"]["sphere"].get(
                    "silhouette_residual_pixels"
                ),
                "processing_time_ms": item["algorithm"]["segmentation"].get(
                    "processing_time_ms", 0
                )
                + item["algorithm"]["sphere"].get("processing_time_ms", 0),
                "unknown_radius_status": item["algorithm"]["unknown_radius_setting"]["status"],
            }
            for item in conditions
        ]
        _json(
            root / "algorithm_summary.json",
            {
                "number_of_conditions": len(algorithm_rows),
                "failure_count": sum(
                    row["sphere_status"] != "KNOWN_RADIUS_CENTER_ESTIMATED"
                    for row in algorithm_rows
                ),
                "failure_rate": sum(
                    row["sphere_status"] != "KNOWN_RADIUS_CENTER_ESTIMATED"
                    for row in algorithm_rows
                )
                / len(algorithm_rows),
                "conditions": algorithm_rows,
            },
        )
        evaluation_rows = [
            {
                "condition": item["name"],
                "segmentation_iou": item["evaluation"]["segmentation_iou"],
                "sphere_center_error_m": item["evaluation"]["sphere_center_error_m"],
                "radius_error_m": None,
                "visible_fraction": item["evaluation"].get("visible_fraction"),
                "occlusion_condition": item["evaluation"].get("occlusion_condition"),
            }
            for item in conditions
        ]
        _json(
            root / "evaluation_summary.json",
            {
                "boundary": "ground_truth_evaluation_only",
                "known_radius_and_unknown_radius_metrics_are_not_mixed": True,
                "conditions": evaluation_rows,
                "segmentation_iou": {
                    "minimum": min(row["segmentation_iou"] for row in evaluation_rows),
                    "maximum": max(row["segmentation_iou"] for row in evaluation_rows),
                    "mean": float(np.mean([row["segmentation_iou"] for row in evaluation_rows])),
                },
                "sphere_center_error_m": {
                    "values": [row["sphere_center_error_m"] for row in evaluation_rows],
                    "mean": float(np.mean([
                        row["sphere_center_error_m"]
                        for row in evaluation_rows
                        if row["sphere_center_error_m"] is not None
                    ])),
                },
                "radius_error_m": None,
                "radius_error_reason": "known radius was supplied as a prior and was not estimated",
            },
        )
        return {
            "output": str(root),
            "number_of_conditions": len(conditions),
            "algorithm_summary": str(root / "algorithm_summary.json"),
            "evaluation_summary": str(root / "evaluation_summary.json"),
        }
    finally:
        camera.close()
        engine.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run_experiment(args.output), indent=2))


if __name__ == "__main__":
    main()
