"""Persistence for evaluation-only ground-truth comparison data."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from evaluation.contact_evaluator import ContactPointError, summarize_contact_errors
from evaluation.sphere_evaluator import SphereEvaluationResult, SphereGroundTruth


def save_evaluation_dataset(
    path: str | Path,
    errors: list[ContactPointError] | tuple[ContactPointError, ...],
    sphere_ground_truth: SphereGroundTruth | None = None,
    sphere_evaluation: SphereEvaluationResult | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics = summarize_contact_errors(errors)
    metrics_payload = asdict(metrics)
    for values in metrics_payload.values():
        if values["count"] == 0:
            for name in ("mean_m", "median_m", "rmse_m", "percentile_95_m"):
                values[name] = None
    payload = {
        "schema_version": 1,
        "dataset_type": "mujoco_contact_evaluation",
        "metrics": metrics_payload,
        "errors": [asdict(error) for error in errors],
    }
    if sphere_ground_truth is not None:
        payload["sphere_ground_truth"] = asdict(sphere_ground_truth)
    if sphere_evaluation is not None:
        sphere_payload = asdict(sphere_evaluation)
        for key in ("point_to_estimated_surface", "point_to_ground_truth_surface"):
            values = sphere_payload[key]
            if values["count"] == 0:
                for name in ("mean_m", "median_m", "rmse_m", "percentile_95_m"):
                    values[name] = None
        payload["sphere_evaluation"] = sphere_payload
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output
