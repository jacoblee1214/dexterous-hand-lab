"""Materialization and batching for the neural SDF benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .dataset import DEFAULT_MANIFEST, load_sample


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "experiments/neural_sdf/generated_cache/dataset_v1.pt"
DEFAULT_FEATURE_CACHE = ROOT / "experiments/neural_sdf/generated_cache/dataset_v1_resnet_features.pt"


def _coverage(tactile):
    if not tactile:
        return 0.0
    points = np.asarray([item.representative_point_object_m for item in tactile])
    fingers = len({item.finger_id for item in tactile})
    spread = (float(np.max(np.linalg.norm(points[:, None]-points[None, :], axis=-1)))
              if len(points) > 1 else 0.0)
    return min(len(tactile)/6, 1)*min(fingers/3, 1)*min(spread/.06, 1)


def _model_input(observation, maximum_contacts=8, patch_samples=25):
    if len(observation.rgb_observations) != 1:
        raise ValueError("Neural SDF v1 requires exactly one synchronized RGB frame")
    frame = observation.rgb_observations[0]
    tactile = tuple(item for item in observation.tactile_observations if item.measurement_valid)
    if len(tactile) > maximum_contacts:
        raise ValueError("Tactile input exceeds frozen maximum")
    rgb = torch.from_numpy(np.asarray(frame.rgb).copy()).permute(2, 0, 1)
    visibility = torch.from_numpy(np.stack((
        frame.object_mask, frame.known_background_mask,
        frame.hand_occluded_mask, frame.unreliable_mask,
    )).astype(np.float32))
    values = {
        "position": torch.zeros(maximum_contacts, 3),
        "normal": torch.zeros(maximum_contacts, 3),
        "scalar_n": torch.zeros(maximum_contacts, 1),
        "indentation_m": torch.zeros(maximum_contacts, 1),
        "sigma_m": torch.ones(maximum_contacts, 1)*.01,
        "patch": torch.zeros(maximum_contacts, patch_samples, 3),
        "valid": torch.zeros(maximum_contacts, dtype=torch.bool),
        "sensor_index": torch.zeros(maximum_contacts, dtype=torch.long),
        "finger_index": torch.zeros(maximum_contacts, dtype=torch.long),
    }
    finger_names = ("thumb", "index", "middle", "ring")
    for index, item in enumerate(tactile):
        patch = np.asarray(item.patch_points_object_m, dtype=np.float32)
        if len(patch) != patch_samples:
            raise ValueError("Frozen finite tactile patch sample count changed")
        values["position"][index] = torch.tensor(item.representative_point_object_m)
        values["normal"][index] = torch.tensor(item.sensing_normal_object)
        values["scalar_n"][index, 0] = item.scalar_measurement_n
        values["indentation_m"][index, 0] = item.estimated_indentation_m
        values["sigma_m"][index, 0] = item.contact_region_sigma_m
        values["patch"][index] = torch.from_numpy(patch)
        values["valid"][index] = True
        # Stable named-channel mapping: never encode arrival/list order.
        sensor_number = int(item.sensor_id.rsplit("_", 1)[-1])
        if not 1 <= sensor_number <= maximum_contacts:
            raise ValueError(f"Unknown tactile sensor mapping: {item.sensor_id}")
        values["sensor_index"][index] = sensor_number-1
        values["finger_index"][index] = finger_names.index(item.finger_id)
    return {
        "rgb": rgb,
        "visibility_masks": visibility,
        "intrinsic": torch.tensor(np.asarray(frame.intrinsic_matrix), dtype=torch.float32),
        "camera_from_object": torch.tensor(np.asarray(frame.camera_from_object), dtype=torch.float32),
        "joint_state": torch.tensor(list(observation.joint_state.values()), dtype=torch.float32),
        "tactile": values,
        "coverage": torch.tensor([_coverage(tactile)], dtype=torch.float32),
        "timestamp": torch.tensor([observation.timestamp], dtype=torch.float64),
    }


def materialize(manifest_path=DEFAULT_MANIFEST, cache_path=DEFAULT_CACHE, *, force=False):
    manifest_path, cache_path = Path(manifest_path), Path(cache_path)
    if cache_path.is_file() and not force:
        records = torch.load(cache_path, map_location="cpu", weights_only=False)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        samples = {item["sample_id"]: item for item in manifest["samples"]}
        for record in records:
            sample = samples[record["audit"]["sample_id"]]
            record["audit"].update({
                "contact_pattern": sample["contact_pattern"],
                "declared_active_sensor_count": sample["active_sensor_count"],
                "occlusion_fraction": sample["occlusion_fraction"],
            })
        return records
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    for index, sample_record in enumerate(manifest["samples"]):
        sample = load_sample(manifest, sample_record)
        truth = sample.training_or_evaluation_truth
        labels = {name: code for code, name in enumerate(
            ("visible_to_rgb", "occluded_by_hand", "other_unobserved"))}
        records.append({
            "model_input": _model_input(sample.model_input),
            "supervision": {
                "surface_points": torch.tensor(truth["surface_points_object_m"], dtype=torch.float32),
                "surface_labels": torch.tensor(
                    [labels[value] for value in truth["surface_region_labels"]], dtype=torch.uint8),
                "oracle_contact_points": torch.tensor(
                    truth["oracle_contact_points_object_m"], dtype=torch.float32),
                "family": sample.audit_metadata["family"],
                "parameters": sample.audit_metadata["parameters"],
            },
            "audit": sample.audit_metadata,
        })
        if (index+1) % 48 == 0:
            print(f"materialized {index+1}/{len(manifest['samples'])}", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(records, cache_path)
    return records


def collate_model_inputs(records, device="cpu"):
    values = {}
    for key in ("visibility_masks", "intrinsic", "camera_from_object", "joint_state", "coverage"):
        values[key] = torch.stack([item["model_input"][key] for item in records]).to(device)
    values["tactile"] = {
        key: torch.stack([item["model_input"]["tactile"][key] for item in records]).to(device)
        for key in records[0]["model_input"]["tactile"]
    }
    if "rgb_feature_map" in records[0]["model_input"]:
        values["rgb_feature_map"] = torch.stack(
            [item["model_input"]["rgb_feature_map"] for item in records]).to(device).float()
    return values


def attach_rgb_features(records, encoder, *, batch_size=16, device="cpu"):
    encoder = encoder.to(device).eval()
    for start in range(0, len(records), batch_size):
        subset = records[start:start+batch_size]
        rgb = torch.stack([item["model_input"]["rgb"] for item in subset]).to(device)
        with torch.no_grad():
            feature = encoder(rgb).cpu().to(torch.float16)
        for item, value in zip(subset, feature):
            item["model_input"]["rgb_feature_map"] = value
    return records


def materialize_with_rgb_features(encoder, *, device="cpu", force_data=False,
                                  feature_cache_path=DEFAULT_FEATURE_CACHE):
    """Load immutable samples plus deterministic frozen visual features.

    The feature cache is derived solely from the allowed RGB observation and
    the SHA-verified frozen encoder.  It contains no supervision or geometry
    beyond the separately cached benchmark records.
    """
    records = materialize(force=force_data)
    feature_cache_path = Path(feature_cache_path)
    if feature_cache_path.is_file() and not force_data:
        cached = torch.load(feature_cache_path, map_location="cpu", weights_only=False)
        expected_ids = [item["audit"]["sample_id"] for item in records]
        if cached.get("sample_ids") != expected_ids:
            raise ValueError("Frozen RGB feature cache does not match the dataset manifest")
        for record, feature in zip(records, cached["features"]):
            record["model_input"]["rgb_feature_map"] = feature
        return records
    attach_rgb_features(records, encoder, device=device)
    feature_cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "sample_ids": [item["audit"]["sample_id"] for item in records],
        "features": torch.stack(
            [item["model_input"]["rgb_feature_map"] for item in records]),
    }, feature_cache_path)
    return records


def split_records(records, split):
    return [item for item in records if item["audit"]["split"] == split]
