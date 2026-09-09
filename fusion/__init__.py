"""Ground-truth-free geometric RGB and scalar-tactile fusion baselines."""

from fusion.observation import (
    FusionObservation,
    TactileContactConstraint,
    build_fusion_observation,
)
from fusion.sphere import FusionResult, run_fusion_methods

__all__ = [
    "FusionObservation",
    "FusionResult",
    "TactileContactConstraint",
    "build_fusion_observation",
    "run_fusion_methods",
]
