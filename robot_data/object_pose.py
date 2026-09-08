"""Explicit extension point for measured object-pose tracking.

Reconstruction never consumes this interface.  It exists so a future calibrated
camera/marker tracker can segment or align moving-object trials without treating
MuJoCo ground truth as a measured observation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ObjectPoseObservation:
    timestamp: float
    frame_id: str
    position_xyz: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    calibration_reference: str

    def __post_init__(self) -> None:
        values = (*self.position_xyz, *self.quaternion_wxyz, self.timestamp)
        if not self.frame_id or not self.calibration_reference:
            raise ValueError("Object pose requires frame and calibration references")
        if len(self.position_xyz) != 3 or len(self.quaternion_wxyz) != 4:
            raise ValueError("Object pose requires XYZ and WXYZ quaternion")
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Object pose values must be finite")


class ObjectPoseTracker(ABC):
    @abstractmethod
    def latest(self) -> ObjectPoseObservation | None:
        """Return a measured pose, or None when no synchronized pose is available."""


class UnavailableObjectPoseTracker(ObjectPoseTracker):
    def latest(self) -> None:
        return None
