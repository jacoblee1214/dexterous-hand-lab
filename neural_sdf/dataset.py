"""Deterministic procedural dataset with a strict-the-GT boundary.

Manifest metadata selects a simulation generator.  ``model_input`` never
contains the instance/family/parameter metadata used to create supervision.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

from general_object.observation import GeneralObjectObservation
from general_object.synthetic import ShapeSpec, generate_case


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "experiments/neural_sdf/protocol_v1.yaml"
DEFAULT_MANIFEST = ROOT / "experiments/neural_sdf/dataset_manifest_v1.json"


@dataclass(frozen=True)
class NeuralDatasetSample:
    model_input: GeneralObjectObservation
    training_or_evaluation_truth: dict
    audit_metadata: dict


class _NoFeatureEncoder:
    def encode(self, rgb):
        from general_object.features import FeatureEncoding
        return FeatureEncoding("deferred-frozen-resnet18", np.empty((0,), np.float32),
                               (0.0,), 0, 0, 0.0,
                               "computed by neural_sdf FrozenResNet18GridEncoder")


def _uniform(rng, values):
    return float(rng.uniform(float(values[0]), float(values[1])))


def _instance_parameters(family: str, split: str, rng, ranges):
    r = ranges[split]
    if family == "sphere":
        return {"radius_m": _uniform(rng, r["sphere_radius"])}
    if family == "cylinder":
        return {"radius_m": _uniform(rng, r["cylinder_radius"]),
                "half_height_m": _uniform(rng, r["cylinder_half_height"])}
    if family in {"cuboid", "rounded_box"}:
        lo, hi = r["box_half_extent"]
        half = [float(rng.uniform(lo, hi)) for _ in range(3)]
        # Reject nearly isotropic boxes so aspect ratio remains informative.
        half[int(rng.integers(0, 3))] = float(rng.uniform(max(lo, hi*.78), hi))
        value = {"half_extents_m": half}
        if family == "rounded_box":
            value["round_m"] = min(_uniform(rng, r["rounded_box_round"]), min(half)*.45)
        return value
    if family == "asymmetric_l":
        scale = float(rng.uniform(.82, 1.13))
        skew = float(rng.uniform(-.004, .004))
        return {
            "box_a_half_m": [0.038*scale, 0.017*scale, 0.032*scale],
            "box_a_center_m": [0.0, -0.014*scale, 0.0],
            "box_b_half_m": [0.017*scale, 0.029*scale, 0.023*scale],
            "box_b_center_m": [-0.021*scale, 0.012*scale+skew, 0.009*scale],
        }
    raise ValueError(family)


def build_manifest(protocol_path=DEFAULT_PROTOCOL) -> dict:
    path = Path(protocol_path)
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = int(protocol["random_seed_base"])
    counts = protocol["dataset"]
    seen = tuple(protocol["families"]["seen_during_training"])
    instances, samples = [], []
    split_counts = {
        "train": (int(counts["train_instances_per_seen_family"]),
                  int(counts["samples_per_train_instance"]), seen),
        "validation": (int(counts["validation_instances_per_seen_family"]),
                       int(counts["samples_per_validation_instance"]), seen),
        "test": (int(counts["test_instances_per_seen_family"]),
                 int(counts["samples_per_test_instance"]),
                 seen+tuple(protocol["families"]["family_holdout"])),
    }
    instance_index = sample_index = 0
    for split, (per_family, samples_per_instance, families) in split_counts.items():
        for family in families:
            family_count = (int(counts["test_asymmetric_l_instances"])
                            if split == "test" and family == "asymmetric_l" else per_family)
            for family_index in range(family_count):
                instance_index += 1
                seed = base+instance_index
                rng = np.random.default_rng(seed)
                instance_id = f"instance_{instance_index:04d}"
                instances.append({
                    "instance_id": instance_id, "split": split, "family": family,
                    "parameters": _instance_parameters(family, split, rng,
                                                       protocol["parameter_ranges_m"]),
                    "seed": seed,
                })
                for local_index in range(samples_per_instance):
                    sample_index += 1
                    sample_seed = base+100000+sample_index
                    srng = np.random.default_rng(sample_seed)
                    conditions = protocol["conditions"]
                    sensor_counts = conditions["active_sensor_counts"]
                    # Cycle counts before jittering order so every split includes no contact.
                    count = int(sensor_counts[(family_index+local_index) % len(sensor_counts)])
                    samples.append({
                        "sample_id": f"sample_{sample_index:05d}",
                        "instance_id": instance_id, "split": split, "seed": sample_seed,
                        "yaw_degrees": _uniform(srng, conditions["yaw_degrees_range"]),
                        "pitch_degrees": _uniform(srng, conditions["pitch_degrees_range"]),
                        "occlusion_fraction": _uniform(srng, conditions["occlusion_fraction_range"]),
                        "active_sensor_count": count,
                        "contact_pattern": conditions["contact_patterns"][sample_index % 2],
                        "lighting_scale": _uniform(srng, conditions["lighting_scale_range"]),
                        "object_rgb": [int(srng.integers(a, b+1)) for a, b in zip(*conditions["object_rgb_ranges"])],
                        "background_rgb": [int(srng.integers(a, b+1)) for a, b in zip(*conditions["background_rgb_ranges"])],
                    })
    return {
        "schema_version": 1, "protocol_version": protocol["protocol_version"],
        "protocol_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "generator": "neural_sdf.dataset.build_manifest/v1",
        "model_input_metadata_policy": protocol["model_input_excludes"],
        "instances": instances, "samples": samples,
        "counts": {
            "instances": len(instances), "samples": len(samples),
            "by_split": {split: sum(s["split"] == split for s in samples)
                         for split in ("train", "validation", "test")},
        },
    }


def write_manifest(manifest: dict, path=DEFAULT_MANIFEST):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, sort_keys=True)+"\n"
    path.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode()).hexdigest()
    path.with_suffix(path.suffix+".sha256").write_text(digest+"  "+path.name+"\n", encoding="utf-8")
    return digest


def load_sample(manifest: dict, sample_record: dict, protocol_path=DEFAULT_PROTOCOL) -> NeuralDatasetSample:
    protocol = yaml.safe_load(Path(protocol_path).read_text(encoding="utf-8"))
    instance = next(item for item in manifest["instances"]
                    if item["instance_id"] == sample_record["instance_id"])
    spec = ShapeSpec(instance["instance_id"], instance["family"],
                     instance["parameters"], sample_record["split"])
    kwargs = {key: sample_record[key] for key in (
        "seed", "yaw_degrees", "pitch_degrees", "occlusion_fraction",
        "active_sensor_count", "contact_pattern", "object_rgb",
        "background_rgb", "lighting_scale")}
    observation, truth = generate_case(
        spec, grid_resolution=24,
        image_downsample=int(protocol["dataset"]["image_downsample"]),
        surface_resolution=int(protocol["dataset"]["surface_resolution"]),
        feature_encoder=_NoFeatureEncoder(), **kwargs)
    # Truth and metadata are returned on separately named paths.  Training code
    # must pass only observation-derived tensors to model.forward().
    return NeuralDatasetSample(
        model_input=observation,
        training_or_evaluation_truth=truth,
        audit_metadata={
            "sample_id": sample_record["sample_id"], "instance_id": instance["instance_id"],
            "split": sample_record["split"], "family": instance["family"],
            "parameters": instance["parameters"],
        },
    )


def iter_split(manifest: dict, split: str) -> Iterable[NeuralDatasetSample]:
    for item in manifest["samples"]:
        if item["split"] == split:
            yield load_sample(manifest, item)


if __name__ == "__main__":
    value = build_manifest()
    print(write_manifest(value))
    print(json.dumps(value["counts"], indent=2))
