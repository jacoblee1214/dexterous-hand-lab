"""Evaluation-only access to simulator ground truth."""

from evaluation.contact_evaluator import (
    ContactPointError,
    ErrorStatistics,
    EvaluationMetrics,
    EvaluationReport,
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

__all__ = [
    "ContactPointError",
    "ErrorStatistics",
    "EvaluationMetrics",
    "EvaluationReport",
    "GroundTruthContactPoint",
    "collect_mujoco_ground_truth",
    "evaluate_contact_points",
    "summarize_contact_errors",
    "save_evaluation_dataset",
    "SphereEvaluationResult",
    "SphereGroundTruth",
    "collect_mujoco_sphere_ground_truth",
    "evaluate_sphere_reconstruction",
]
