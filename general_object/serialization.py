"""Portable serialization for the GT-free general-object input contract."""

from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
from PIL import Image

from .observation import (
    GeneralObjectObservation,
    ObjectFrameInitialization,
    RGBObjectObservation,
    TactilePatchObservation,
)


def _pack_mask(mask: np.ndarray) -> dict:
    return {
        "shape": list(mask.shape),
        "packed_base64": base64.b64encode(np.packbits(mask.reshape(-1)).tobytes()).decode("ascii"),
    }


def _unpack_mask(value: dict) -> np.ndarray:
    shape = tuple(int(v) for v in value["shape"])
    raw = np.frombuffer(base64.b64decode(value["packed_base64"]), dtype=np.uint8)
    return np.unpackbits(raw, count=int(np.prod(shape))).reshape(shape).astype(bool)


def _pack_rgb(rgb: np.ndarray) -> str:
    stream = BytesIO()
    Image.fromarray(rgb).save(stream, format="PNG", optimize=True)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _unpack_rgb(value: str) -> np.ndarray:
    return np.asarray(Image.open(BytesIO(base64.b64decode(value))).convert("RGB"))


def observation_to_dict(value: GeneralObjectObservation) -> dict:
    frame = value.object_frame
    return {
        "schema_version": 1,
        "algorithm_input_boundary": value.input_boundary,
        "timestamp": value.timestamp,
        "time_base": value.time_base,
        "static_pose_assumption": value.static_pose_assumption,
        "object_frame": {
            "frame_id": frame.frame_id, "parent_frame": frame.parent_frame,
            "timestamp": frame.timestamp, "world_from_object": frame.world_from_object.tolist(),
            "covariance_6x6": frame.covariance_6x6.tolist(), "source": frame.source,
            "validity": frame.validity,
            "transform_direction": f"T_{frame.parent_frame}_from_{frame.frame_id}",
        },
        "reconstruction_bounds_m": value.reconstruction_bounds_m.tolist(),
        "grid_resolution": value.grid_resolution,
        "joint_state": dict(value.joint_state),
        "rgb_observations": [{
            "timestamp": item.timestamp, "frame_id": item.frame_id,
            "camera_id": item.camera_id, "calibration_version": item.calibration_version,
            "rgb_png_base64": _pack_rgb(item.rgb),
            "intrinsic_matrix": item.intrinsic_matrix.tolist(),
            "camera_from_object": item.camera_from_object.tolist(),
            "transform_direction": f"T_{item.frame_id}_from_{frame.frame_id}",
            "object_mask": _pack_mask(item.object_mask),
            "known_background_mask": _pack_mask(item.known_background_mask),
            "hand_occluded_mask": _pack_mask(item.hand_occluded_mask),
            "unreliable_mask": _pack_mask(item.unreliable_mask),
            "segmentation_confidence": item.segmentation_confidence,
            "feature_encoder": item.feature_encoder,
            "feature_summary": list(item.feature_summary),
            "depth_prior_reference": item.depth_prior_reference,
            "normal_prior_reference": item.normal_prior_reference,
            "depth_is_measured": False,
        } for item in value.rgb_observations],
        "tactile_observations": [{
            "timestamp": item.timestamp, "sensor_id": item.sensor_id,
            "finger_id": item.finger_id, "parent_link": item.parent_link,
            "calibration_version": item.calibration_version,
            "object_from_sensor": item.object_from_sensor.tolist(),
            "transform_direction": f"T_{frame.frame_id}_from_{item.sensor_id}",
            "representative_point_object_m": list(item.representative_point_object_m),
            "patch_points_object_m": item.patch_points_object_m.tolist(),
            "sensing_normal_object": list(item.sensing_normal_object),
            "scalar_measurement_n": item.scalar_measurement_n,
            "estimated_indentation_m": item.estimated_indentation_m,
            "contact_region_sigma_m": item.contact_region_sigma_m,
            "measurement_valid": item.measurement_valid,
            "joint_state": dict(item.joint_state),
            "scalar_is_3d_force_vector": False,
            "representative_point_is_exact_contact": False,
        } for item in value.tactile_observations],
        "forbidden_fields_absent": [
            "object_identity", "known_radius", "CAD_mesh", "ground_truth_pose",
            "ground_truth_surface", "ground_truth_contact_normal",
        ],
    }


def observation_from_dict(raw: dict) -> GeneralObjectObservation:
    if raw.get("schema_version") != 1:
        raise ValueError("Unsupported general-object observation schema")
    frame_raw = raw["object_frame"]
    frame = ObjectFrameInitialization(
        frame_raw["frame_id"], frame_raw["parent_frame"], frame_raw["timestamp"],
        frame_raw["world_from_object"], frame_raw["covariance_6x6"],
        frame_raw["source"], frame_raw["validity"],
    )
    rgb = tuple(RGBObjectObservation(
        item["timestamp"], item["frame_id"], item["camera_id"], item["calibration_version"],
        _unpack_rgb(item["rgb_png_base64"]), item["intrinsic_matrix"], item["camera_from_object"],
        _unpack_mask(item["object_mask"]), _unpack_mask(item["known_background_mask"]),
        _unpack_mask(item["hand_occluded_mask"]), _unpack_mask(item["unreliable_mask"]),
        item["segmentation_confidence"], item["feature_encoder"], tuple(item["feature_summary"]),
        item.get("depth_prior_reference"), item.get("normal_prior_reference"),
    ) for item in raw["rgb_observations"])
    tactile = tuple(TactilePatchObservation(
        timestamp=item["timestamp"], sensor_id=item["sensor_id"], finger_id=item["finger_id"],
        parent_link=item["parent_link"], calibration_version=item["calibration_version"],
        object_from_sensor=item["object_from_sensor"],
        representative_point_object_m=tuple(item["representative_point_object_m"]),
        patch_points_object_m=item["patch_points_object_m"],
        sensing_normal_object=tuple(item["sensing_normal_object"]),
        scalar_measurement_n=item["scalar_measurement_n"],
        estimated_indentation_m=item["estimated_indentation_m"],
        contact_region_sigma_m=item["contact_region_sigma_m"],
        measurement_valid=item["measurement_valid"], joint_state=item["joint_state"],
    ) for item in raw["tactile_observations"])
    return GeneralObjectObservation(
        raw["timestamp"], raw["time_base"], frame, rgb, tactile,
        raw["reconstruction_bounds_m"], raw["grid_resolution"], raw["static_pose_assumption"],
        raw["joint_state"],
    )
