"""Evaluation-only general-surface metrics with explicit region partitions."""

from __future__ import annotations

import numpy as np

from .sdf import DenseSDF


def _nearest(source: np.ndarray, target: np.ndarray, chunk=512) -> np.ndarray:
    if not len(source) or not len(target):
        return np.full(len(source), np.inf)
    result = []
    for start in range(0, len(source), chunk):
        values = source[start:start+chunk]
        squared = np.sum((values[:, None]-target[None, :])**2, axis=2)
        result.append(np.sqrt(np.min(squared, axis=1)))
    return np.concatenate(result)


def _region_metrics(distances: np.ndarray, thresholds_m: tuple[float, ...]):
    finite = distances[np.isfinite(distances)]
    if not len(finite):
        return {"count": 0, "mean_point_to_surface_m": None,
                "median_point_to_surface_m": None,
                "completeness": {str(v): 0.0 for v in thresholds_m}}
    return {
        "count": int(len(finite)),
        "mean_point_to_surface_m": float(np.mean(finite)),
        "median_point_to_surface_m": float(np.median(finite)),
        "completeness": {str(v): float(np.mean(finite <= v)) for v in thresholds_m},
    }


def evaluate_surface(
    result: DenseSDF,
    evaluation_truth: dict,
    *,
    thresholds_m=(.002, .005),
) -> dict:
    """Truth enters only here, after the algorithm result is already complete."""
    predicted = result.surface_points(3000)
    gt = np.asarray(evaluation_truth["surface_points_object_m"], dtype=float)
    labels = np.asarray(evaluation_truth["surface_region_labels"])
    gt_to_prediction = _nearest(gt, predicted)
    prediction_to_gt = _nearest(predicted, gt)
    regions = {}
    for name in ("visible_to_rgb", "occluded_by_hand", "other_unobserved"):
        regions[name] = _region_metrics(gt_to_prediction[labels == name], thresholds_m)
    contacts = np.asarray(evaluation_truth["oracle_contact_points_object_m"], dtype=float)
    if len(contacts):
        near_contact = _nearest(gt, contacts) <= .012
        contact_region = _region_metrics(gt_to_prediction[near_contact], thresholds_m)
    else:
        contact_region = _region_metrics(np.empty(0), thresholds_m)
    precision = {}; recall = {}; f_score = {}
    for threshold in thresholds_m:
        p = float(np.mean(prediction_to_gt <= threshold)) if len(prediction_to_gt) else 0.0
        r = float(np.mean(gt_to_prediction <= threshold)) if len(gt_to_prediction) else 0.0
        precision[str(threshold)], recall[str(threshold)] = p, r
        f_score[str(threshold)] = float(2*p*r/(p+r)) if p+r else 0.0
    chamfer = None
    if len(prediction_to_gt) and len(gt_to_prediction):
        chamfer = float(np.mean(prediction_to_gt)+np.mean(gt_to_prediction))
    return {
        "boundary": "evaluation_only_truth_consumed_after_reconstruction",
        "method": result.method, "algorithm_status": result.status,
        "whole_surface_chamfer_l1_m": chamfer,
        "whole_surface_point_to_surface_m": float(np.mean(gt_to_prediction)) if len(gt_to_prediction) else None,
        "precision": precision, "recall_completeness": recall, "f_score": f_score,
        "regions": regions, "contact_region": contact_region,
        "surface_counts": {"predicted": int(len(predicted)), "ground_truth": int(len(gt))},
        "ground_truth_was_algorithm_input": result.diagnostics["ground_truth_input"],
    }
