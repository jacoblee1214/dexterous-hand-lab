"""Evaluation-only metrics. Simulator truth enters only through this module."""

from __future__ import annotations

import numpy as np

from vision.baseline import sphere_silhouette_mask


def evaluate_methods(
    algorithm_result: dict,
    *,
    ground_truth_center_world_m,
    ground_truth_visible_mask: np.ndarray,
    calibration,
) -> dict:
    truth = np.asarray(ground_truth_center_world_m, dtype=float)
    rows = {}
    for name, result in algorithm_result["methods"].items():
        center = result.get("estimated_center_world_m")
        if center is None:
            rows[name] = {
                "success": False,
                "failure_status": result["status"],
                "center_error_m": None,
                "visible_silhouette_iou": None,
                "silhouette_iou_interpretation": "not available for a failed estimate",
            }
            continue
        center = np.asarray(center, dtype=float)
        center_cv = (calibration.camera_cv_from_world @ np.r_[center, 1.0])[:3]
        predicted = sphere_silhouette_mask(calibration, center_cv, result["radius_m"])
        # Unknown/hand-occluded pixels are excluded from the algorithm residual,
        # but evaluation reports overlap against the actually visible GT pixels.
        intersection = int(np.logical_and(predicted, ground_truth_visible_mask).sum())
        union = int(np.logical_or(predicted, ground_truth_visible_mask).sum())
        rows[name] = {
            "success": True,
            "failure_status": None,
            "center_error_m": float(np.linalg.norm(center-truth)),
            "visible_silhouette_iou": intersection/union if union else 1.0,
            "silhouette_iou_interpretation": (
                "estimated full sphere silhouette versus MuJoCo-visible sphere pixels; "
                "occluding hand therefore lowers this diagnostic"
            ),
        }
    return {
        "boundary": "evaluation_only; MuJoCo object pose and segmentation never enter fusion",
        "methods": rows,
        "radius_error_m": None,
        "radius_error_reason": "radius is a declared prior, not an estimated parameter",
    }
