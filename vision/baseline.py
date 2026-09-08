"""Transparent color segmentation and calibrated monocular sphere baselines."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

from vision.calibration import CalibratedCamera


@dataclass(frozen=True)
class SegmentationResult:
    mask: np.ndarray
    status: str
    confidence: float
    pixel_count: int
    boundary_pixels_uv: np.ndarray
    angular_boundary_coverage: float
    processing_time_ms: float

    def diagnostics(self) -> dict:
        return {
            "status": self.status,
            "confidence": self.confidence,
            "pixel_count": self.pixel_count,
            "boundary_pixel_count": len(self.boundary_pixels_uv),
            "angular_boundary_coverage": self.angular_boundary_coverage,
            "processing_time_ms": self.processing_time_ms,
            "assumption": "controlled blue-sphere color threshold; not general object segmentation",
        }


def _boundary(mask: np.ndarray) -> np.ndarray:
    # Use pixel-face midpoints, not the centers of the last foreground pixels.
    # The latter systematically shrinks a rasterized silhouette and biases
    # monocular depth away from the camera.
    padded = np.pad(mask, 1, constant_values=False)
    center = padded[1:-1, 1:-1]
    edges = []
    for neighbor, du, dv in (
        (padded[1:-1, 2:], 0.5, 0.0),
        (padded[1:-1, :-2], -0.5, 0.0),
        (padded[2:, 1:-1], 0.0, 0.5),
        (padded[:-2, 1:-1], 0.0, -0.5),
    ):
        rows, columns = np.nonzero(center & ~neighbor)
        if len(rows):
            edges.append(np.column_stack((columns + du, rows + dv)))
    return np.concatenate(edges).astype(float) if edges else np.empty((0, 2))


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Return the largest four-connected threshold region without extra dependencies."""
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    largest: list[tuple[int, int]] = []
    for row, column in zip(*np.nonzero(mask)):
        if visited[row, column]:
            continue
        visited[row, column] = True
        stack = [(int(row), int(column))]
        component = []
        while stack:
            current_row, current_column = stack.pop()
            component.append((current_row, current_column))
            for next_row, next_column in (
                (current_row - 1, current_column),
                (current_row + 1, current_column),
                (current_row, current_column - 1),
                (current_row, current_column + 1),
            ):
                if (
                    0 <= next_row < height
                    and 0 <= next_column < width
                    and mask[next_row, next_column]
                    and not visited[next_row, next_column]
                ):
                    visited[next_row, next_column] = True
                    stack.append((next_row, next_column))
        if len(component) > len(largest):
            largest = component
    result = np.zeros_like(mask, dtype=bool)
    if largest:
        rows, columns = zip(*largest)
        result[rows, columns] = True
    return result


def segment_blue_sphere(rgb: np.ndarray, settings) -> SegmentationResult:
    started = time.perf_counter()
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("RGB input must be an HxWx3 uint8 image")
    red, green, blue = [image[:, :, index].astype(np.int16) for index in range(3)]
    blue_red = blue - red
    blue_green = blue - green
    threshold_mask = (
        (blue >= settings["minimum_blue"])
        & (blue_red >= settings["minimum_blue_minus_red"])
        & (blue_green >= settings["minimum_blue_minus_green"])
    )
    mask = _largest_component(threshold_mask)
    boundary = _boundary(mask)
    pixel_count = int(mask.sum())
    minimum = int(settings["minimum_component_pixels"])
    if pixel_count:
        margin = np.minimum(
            blue_red[mask] - settings["minimum_blue_minus_red"],
            blue_green[mask] - settings["minimum_blue_minus_green"],
        )
        confidence = float(np.clip(np.median(margin) / 80.0, 0.0, 1.0))
    else:
        confidence = 0.0
    coverage = 0.0
    if len(boundary) >= 8:
        center = boundary.mean(axis=0)
        angles = np.mod(np.arctan2(boundary[:, 1] - center[1], boundary[:, 0] - center[0]), 2 * np.pi)
        coverage = float(len(np.unique(np.floor(angles / (2 * np.pi) * 36).astype(int))) / 36.0)
    status = (
        "SEGMENTATION_OK"
        if pixel_count >= minimum and len(boundary) >= 20 and confidence >= 0.1
        else "SEGMENTATION_UNRELIABLE"
    )
    return SegmentationResult(
        mask=mask,
        status=status,
        confidence=confidence,
        pixel_count=pixel_count,
        boundary_pixels_uv=boundary,
        angular_boundary_coverage=coverage,
        processing_time_ms=(time.perf_counter() - started) * 1000.0,
    )


def sphere_silhouette_mask(
    calibration: CalibratedCamera, center_camera_cv: np.ndarray, radius_m: float
) -> np.ndarray:
    center = np.asarray(center_camera_cv, dtype=float)
    radius = float(radius_m)
    height, width = calibration.config.height, calibration.config.width
    rows, columns = np.indices((height, width), dtype=float)
    x = (columns - calibration.intrinsic_matrix[0, 2]) / calibration.intrinsic_matrix[0, 0]
    y = (rows - calibration.intrinsic_matrix[1, 2]) / calibration.intrinsic_matrix[1, 1]
    dot = x * center[0] + y * center[1] + center[2]
    norm2 = x * x + y * y + 1.0
    discriminant = dot * dot - norm2 * (float(center @ center) - radius * radius)
    return (dot > 0) & (discriminant >= 0)


def _initial_center(boundary_uv: np.ndarray, calibration: CalibratedCamera, radius: float) -> np.ndarray:
    minimum, maximum = boundary_uv.min(axis=0), boundary_uv.max(axis=0)
    center_pixel = (minimum + maximum) / 2.0
    radius_pixels = max(1.0, float(np.mean((maximum - minimum) / 2.0)))
    focal = math.sqrt(
        calibration.intrinsic_matrix[0, 0] * calibration.intrinsic_matrix[1, 1]
    )
    angular_radius = math.atan(radius_pixels / focal)
    distance = radius / max(1e-6, math.sin(angular_radius))
    direction = calibration.pixel_rays_cv(center_pixel[None, :])[0]
    return direction * distance


def fit_known_radius_sphere(
    segmentation: SegmentationResult,
    calibration: CalibratedCamera,
    radius_m: float,
) -> dict:
    started = time.perf_counter()
    radius = float(radius_m)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("Known radius prior must be finite and positive")
    if segmentation.status != "SEGMENTATION_OK":
        return {
            "status": "SEGMENTATION_FAILURE",
            "identifiability": "metric center unavailable because the RGB mask is unreliable",
            "radius_m": radius,
            "radius_label": "provided_known_radius_prior",
            "estimated_center_camera_cv_m": None,
            "estimated_center_world_m": None,
            "silhouette_residual_pixels": None,
            "processing_time_ms": (time.perf_counter() - started) * 1000.0,
        }
    boundary = segmentation.boundary_pixels_uv
    if len(boundary) > 2000:
        boundary = boundary[np.linspace(0, len(boundary) - 1, 2000).astype(int)]
    rays = calibration.pixel_rays_cv(boundary)
    center = _initial_center(boundary, calibration, radius)
    damping = 1e-8
    for _ in range(60):
        projection = rays @ center
        perpendicular = center[None, :] - projection[:, None] * rays
        distances = np.linalg.norm(perpendicular, axis=1)
        valid = distances > 1e-9
        residual = distances[valid] - radius
        jacobian = perpendicular[valid] / distances[valid, None]
        scale = max(1e-5, float(np.median(np.abs(residual))) * 2.5)
        weights = np.minimum(1.0, scale / np.maximum(np.abs(residual), 1e-12))
        normal = jacobian.T @ (weights[:, None] * jacobian) + damping * np.eye(3)
        gradient = jacobian.T @ (weights * residual)
        delta = np.linalg.solve(normal, -gradient)
        center += delta
        if center[2] <= radius:
            center[2] = radius * 1.01
        if np.linalg.norm(delta) < 1e-10:
            break
    projection = rays @ center
    perpendicular_distance = np.sqrt(np.maximum(0.0, float(center @ center) - projection * projection))
    residual_m = perpendicular_distance - radius
    residual_pixels = float(
        np.sqrt(np.mean(residual_m * residual_m))
        * math.sqrt(calibration.intrinsic_matrix[0, 0] * calibration.intrinsic_matrix[1, 1])
        / center[2]
    )
    world = calibration.camera_to_world(center)
    condition = float(np.linalg.cond(normal))
    reliable = (
        np.isfinite(center).all()
        and center[2] > radius
        and residual_pixels < 4.0
        and condition < 1e10
    )
    return {
        "status": "KNOWN_RADIUS_CENTER_ESTIMATED" if reliable else "FIT_UNRELIABLE",
        "identifiability": "metric center identifiable from calibrated silhouette plus declared radius",
        "radius_m": radius,
        "radius_label": "provided_known_radius_prior",
        "estimated_center_camera_cv_m": center.tolist() if reliable else None,
        "estimated_center_world_m": world.tolist() if reliable else None,
        "silhouette_residual_pixels": residual_pixels,
        "normal_matrix_condition": condition,
        "processing_time_ms": (time.perf_counter() - started) * 1000.0,
    }


def describe_unknown_radius(segmentation: SegmentationResult, calibration: CalibratedCamera) -> dict:
    boundary = segmentation.boundary_pixels_uv
    direction = None
    angular_radius = None
    if len(boundary):
        minimum, maximum = boundary.min(axis=0), boundary.max(axis=0)
        pixel = (minimum + maximum) / 2.0
        direction = calibration.pixel_rays_cv(pixel[None, :])[0].tolist()
        radius_pixels = float(np.mean((maximum - minimum) / 2.0))
        focal = math.sqrt(
            calibration.intrinsic_matrix[0, 0] * calibration.intrinsic_matrix[1, 1]
        )
        angular_radius = math.atan(radius_pixels / focal)
    return {
        "status": "SCALE_AMBIGUOUS",
        "identifiability": (
            "single calibrated RGB silhouette constrains viewing direction and angular size, "
            "but absolute center depth and radius share an unknown metric scale"
        ),
        "radius_m": None,
        "radius_label": "not_estimated",
        "center_depth_m": None,
        "approximate_viewing_direction_camera_cv": direction,
        "approximate_angular_radius_rad": angular_radius,
    }
