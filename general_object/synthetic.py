"""Controlled multi-geometry simulator used only to generate observations/truth.

Analytic shape functions live in this module and are never imported by the
reconstructor or dashboard backend.  The returned algorithm observation and
evaluation truth are separate values and are serialized to separate paths.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from vision.calibration import calibration_from_declared_pose, load_camera_config

from .features import AnalyticColorEdgeEncoder
from .observation import (
    GeneralObjectObservation,
    ObjectFrameInitialization,
    RGBObjectObservation,
    TactilePatchObservation,
)
from .sdf import DenseSDF


@dataclass(frozen=True)
class ShapeSpec:
    instance_id: str
    geometry: str
    parameters: dict
    split: str


SHAPES = (
    ShapeSpec("dev_sphere_01", "sphere", {"radius_m": .038}, "development"),
    ShapeSpec("dev_cylinder_01", "cylinder", {"radius_m": .030, "half_height_m": .045}, "development"),
    ShapeSpec("val_rounded_box_01", "rounded_box", {"half_extents_m": [.040, .030, .033], "round_m": .008}, "validation"),
    ShapeSpec("test_cuboid_01", "cuboid", {"half_extents_m": [.044, .027, .035]}, "test"),
    ShapeSpec("test_asymmetric_l_01", "asymmetric_l", {
        "box_a_half_m": [.040, .018, .034], "box_a_center_m": [0, -.014, 0],
        "box_b_half_m": [.018, .032, .024], "box_b_center_m": [-.022, .012, .010],
    }, "test"),
)


def _box_sdf(points, half_extents, center=(0, 0, 0)):
    q = np.abs(points-np.asarray(center))-np.asarray(half_extents)
    return np.linalg.norm(np.maximum(q, 0), axis=-1)+np.minimum(np.max(q, axis=-1), 0)


def shape_sdf(points: np.ndarray, spec: ShapeSpec) -> np.ndarray:
    """Evaluation/generation-only analytic SDF; never an algorithm input."""
    p = np.asarray(points, dtype=float)
    if spec.geometry == "sphere":
        return np.linalg.norm(p, axis=-1)-spec.parameters["radius_m"]
    if spec.geometry == "cylinder":
        d = np.stack((np.linalg.norm(p[..., :2], axis=-1)-spec.parameters["radius_m"],
                      np.abs(p[..., 2])-spec.parameters["half_height_m"]), axis=-1)
        return np.linalg.norm(np.maximum(d, 0), axis=-1)+np.minimum(np.max(d, axis=-1), 0)
    if spec.geometry == "cuboid":
        return _box_sdf(p, spec.parameters["half_extents_m"])
    if spec.geometry == "rounded_box":
        return _box_sdf(p, np.asarray(spec.parameters["half_extents_m"])-spec.parameters["round_m"])-spec.parameters["round_m"]
    if spec.geometry == "asymmetric_l":
        a = _box_sdf(p, spec.parameters["box_a_half_m"], spec.parameters["box_a_center_m"])
        b = _box_sdf(p, spec.parameters["box_b_half_m"], spec.parameters["box_b_center_m"])
        return np.minimum(a, b)
    raise ValueError(spec.geometry)


def _rotation(yaw_degrees: float, pitch_degrees: float):
    yaw, pitch = map(math.radians, (yaw_degrees, pitch_degrees))
    rz = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                   [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)],
                   [0, math.sin(pitch), math.cos(pitch)]])
    return rz@rx


def _surface(spec: ShapeSpec, bounds, resolution=58, max_points=7000):
    axes = tuple(np.linspace(bounds[0][i], bounds[1][i], resolution) for i in range(3))
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    values = shape_sdf(grid, spec).astype(np.float32)
    dummy = DenseSDF(np.asarray(bounds), values, "evaluation_truth", "VALID", 1, {}, 0)
    return dummy.surface_points(max_points)


def _dilate(mask, iterations=1):
    result = mask.copy()
    for _ in range(iterations):
        padded = np.pad(result, 1)
        result = np.logical_or.reduce((
            padded[1:-1, 1:-1], padded[:-2, 1:-1], padded[2:, 1:-1],
            padded[1:-1, :-2], padded[1:-1, 2:],
            padded[:-2, :-2], padded[:-2, 2:], padded[2:, :-2], padded[2:, 2:],
        ))
    return result


def _render(points, camera_from_object, intrinsic, size, occlusion_fraction):
    h, w = size
    homogeneous = np.column_stack((points, np.ones(len(points))))
    camera = (camera_from_object@homogeneous.T).T[:, :3]
    depth = camera[:, 2]
    u = np.rint(intrinsic[0, 0]*camera[:, 0]/depth+intrinsic[0, 2]).astype(int)
    v = np.rint(intrinsic[1, 1]*camera[:, 1]/depth+intrinsic[1, 2]).astype(int)
    in_frame = (depth > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    zbuffer = np.full((h, w), np.inf)
    np.minimum.at(zbuffer, (v[in_frame], u[in_frame]), depth[in_frame])
    full_mask = np.isfinite(zbuffer)
    full_mask = _dilate(full_mask, 2)
    # Dense surface splats may leave scanline holes; close only bounded gaps.
    for row in range(h):
        columns = np.flatnonzero(full_mask[row])
        if len(columns) > 1:
            full_mask[row, columns[0]:columns[-1]+1] = True
    ys, xs = np.nonzero(full_mask)
    cx, cy = (float(xs.mean()), float(ys.mean())) if len(xs) else (w/2, h/2)
    width = max(4.0, float(xs.max()-xs.min()+1) if len(xs) else w*.2)
    height = max(4.0, float(ys.max()-ys.min()+1) if len(ys) else h*.2)
    yy, xx = np.mgrid[:h, :w]
    hand = (((xx-(cx+width*.12)) / max(2, width*occlusion_fraction*.65))**2
            + ((yy-(cy+height*.05))/max(2, height*.62))**2 <= 1)
    hand &= _dilate(full_mask, 5)
    visible_object = full_mask & ~hand
    boundary = _dilate(visible_object, 1) & ~visible_object & ~hand
    object_mask = visible_object.copy()
    unreliable = boundary.copy()
    background = ~(object_mask | hand | unreliable)
    rgb = np.empty((h, w, 3), dtype=np.uint8); rgb[:] = [22, 28, 34]
    rgb[object_mask] = [32, 82, 205]
    rgb[unreliable] = [50, 73, 120]
    rgb[hand] = [132, 118, 106]
    # Classify only physically front-most GT surface points into three evaluator regions.
    nearest = np.full(len(points), np.inf)
    nearest[in_frame] = zbuffer[v[in_frame], u[in_frame]]
    front = in_frame & (depth <= nearest+.004)
    point_hand = np.zeros(len(points), dtype=bool); point_visible = np.zeros(len(points), dtype=bool)
    point_hand[in_frame] = hand[v[in_frame], u[in_frame]]
    point_visible[in_frame] = object_mask[v[in_frame], u[in_frame]]
    labels = np.full(len(points), "other_unobserved", dtype="U24")
    labels[front & point_visible] = "visible_to_rgb"
    labels[front & point_hand] = "occluded_by_hand"
    return rgb, object_mask, background, hand, unreliable, labels


def _basis(normal):
    reference = np.array([0., 0., 1.]) if abs(normal[2]) < .8 else np.array([1., 0., 0.])
    u = np.cross(normal, reference); u /= np.linalg.norm(u)
    v = np.cross(normal, u); v /= np.linalg.norm(v)
    return u, v


def _contacts(points, labels, count, rng, timestamp):
    candidates = points[labels == "occluded_by_hand"]
    if len(candidates) < count:
        candidates = points[labels != "visible_to_rgb"]
    if count == 0 or not len(candidates):
        return (), np.empty((0, 3))
    order = np.argsort(np.arctan2(candidates[:, 1], candidates[:, 0]))
    indexes = np.linspace(0, len(order)-1, count, dtype=int)
    exact = candidates[order[indexes]]
    fingers = ("thumb", "index", "middle", "ring")
    observations = []
    for index, point in enumerate(exact):
        outward = point/(np.linalg.norm(point)+1e-12)
        sensing = -outward
        u, v = _basis(sensing)
        tangent_bias = rng.normal(0, .0016)*u+rng.normal(0, .0016)*v
        representative = point+tangent_bias+rng.normal(0, .0006)*sensing
        coordinates = np.linspace(-.0045, .0045, 5)
        patch = np.asarray([representative+a*u+b*v for a in coordinates for b in coordinates])
        object_from_sensor = np.eye(4)
        object_from_sensor[:3, :3] = np.column_stack((u, v, sensing))
        object_from_sensor[:3, 3] = representative
        observations.append(TactilePatchObservation(
            timestamp=timestamp, sensor_id=f"sim_sensor_{index+1:02d}",
            finger_id=fingers[index % len(fingers)], parent_link=f"sim_{fingers[index % len(fingers)]}_link",
            calibration_version="simulation-scalar-patch-v1",
            object_from_sensor=object_from_sensor,
            representative_point_object_m=tuple(representative), patch_points_object_m=patch,
            sensing_normal_object=tuple(sensing), scalar_measurement_n=float(.7+.12*index),
            estimated_indentation_m=float(.0008+.00008*index),
            contact_region_sigma_m=.0025, measurement_valid=True,
            joint_state={f"{fingers[index % len(fingers)]}_joint": .4+.03*index},
        ))
    return tuple(observations), exact


def generate_case(
    spec: ShapeSpec,
    *,
    seed: int,
    yaw_degrees: float,
    pitch_degrees: float,
    occlusion_fraction: float,
    active_sensor_count: int,
    grid_resolution: int,
    image_downsample: int = 4,
    feature_encoder=None,
):
    rng = np.random.default_rng(seed)
    bounds = np.array([[-.065, -.065, -.065], [.065, .065, .065]])
    surface = _surface(spec, bounds)
    camera = calibration_from_declared_pose(load_camera_config())
    rotation = _rotation(yaw_degrees, pitch_degrees)
    world_from_object = np.eye(4); world_from_object[:3, :3] = rotation
    world_from_object[:3, 3] = [0, -.045, .105]
    camera_from_object = camera.camera_cv_from_world@world_from_object
    scale = 1.0/image_downsample
    intrinsic = camera.intrinsic_matrix.copy()
    intrinsic[0, 0] *= scale; intrinsic[1, 1] *= scale
    intrinsic[0, 2] = (intrinsic[0, 2]+.5)*scale-.5
    intrinsic[1, 2] = (intrinsic[1, 2]+.5)*scale-.5
    size = (camera.config.height//image_downsample, camera.config.width//image_downsample)
    rgb, foreground, background, hand, unreliable, labels = _render(
        surface, camera_from_object, intrinsic, size, occlusion_fraction)
    encoding = (feature_encoder or AnalyticColorEdgeEncoder()).encode(rgb)
    timestamp = 1.0
    tactile, oracle = _contacts(surface, labels, active_sensor_count, rng, timestamp)
    object_frame = ObjectFrameInitialization(
        "declared_static_object", "world", timestamp, world_from_object,
        np.diag([1e-6]*3+[math.radians(.25)**2]*3),
        "declared_static_fixture", "VALID",
    )
    rgb_observation = RGBObjectObservation(
        timestamp, camera.config.frame_id, camera.config.camera_id,
        camera.config.calibration_version, rgb, intrinsic, camera_from_object,
        foreground, background, hand, unreliable, .92,
        encoding.name, encoding.summary,
    )
    observation = GeneralObjectObservation(
        timestamp, camera.config.time_base, object_frame, (rgb_observation,), tactile,
        bounds, grid_resolution, "declared_rigid_fixture_no_motion",
        {f"joint_{index+1:02d}": float(.05*np.sin(index+seed)) for index in range(20)},
    )
    truth = {
        "boundary": "evaluation_only_never_passed_to_proposed_methods",
        "shape_instance_id": spec.instance_id, "geometry": spec.geometry,
        "shape_parameters": spec.parameters, "split": spec.split,
        "ground_truth_world_from_object": world_from_object.tolist(),
        "surface_points_object_m": surface.tolist(),
        "surface_region_labels": labels.tolist(),
        "oracle_contact_points_object_m": oracle.tolist(),
        "generation": {"seed": seed, "yaw_degrees": yaw_degrees,
                       "pitch_degrees": pitch_degrees,
                       "occlusion_fraction": occlusion_fraction,
                       "active_sensor_count": active_sensor_count,
                       "image_downsample": image_downsample},
        "rgb_feature_efficiency": {
            "encoder": encoding.name, "parameter_count": encoding.parameter_count,
            "estimated_flops": encoding.estimated_flops, "runtime_ms": encoding.runtime_ms,
            "weight_provenance": encoding.weight_provenance,
        },
    }
    return observation, truth
