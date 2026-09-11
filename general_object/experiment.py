"""Reproducible controlled general-object dataset and evaluation runner."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import statistics
import zlib

import numpy as np
import yaml

from .evaluation import evaluate_surface
from .features import AnalyticColorEdgeEncoder, LocalResNet18Encoder
from .serialization import observation_to_dict, observation_from_dict
from .sdf import METHODS, reconstruct_general_object, reconstruct_perfect_contact_oracle
from .synthetic import SHAPES, generate_case


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT_ROOT / "experiments/general_object/protocol_v1.yaml"
DEFAULT_CONFIG = Path(__file__).with_name("config_v1.yaml")
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments/general_object/object_sdf_v1_20260910"
PRIMARY_METHODS = (
    "rgb_only", "tactile_only", "rgb_representative_tactile",
    "rgb_finite_patch_tactile", "rgb_reliability_tactile",
    "rgb_perfect_contact_oracle",
)


def _json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n", encoding="utf-8")


def _sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pack_array(value: np.ndarray):
    array = np.asarray(value)
    return {
        "dtype": str(array.dtype), "shape": list(array.shape),
        "zlib_base64": base64.b64encode(zlib.compress(array.tobytes(), 9)).decode("ascii"),
    }


def _unpack_array(value):
    raw = zlib.decompress(base64.b64decode(value["zlib_base64"]))
    return np.frombuffer(raw, dtype=value["dtype"]).reshape(value["shape"])


def _truth_to_dict(truth):
    labels = np.asarray(truth["surface_region_labels"])
    label_names = ["visible_to_rgb", "occluded_by_hand", "other_unobserved"]
    codes = np.asarray([label_names.index(value) for value in labels], dtype=np.uint8)
    return {
        "schema_version": 1,
        "boundary": "evaluation_only_never_loaded_by_reconstructor",
        "object_instance_id": truth["object_instance_id"],
        "geometry": truth["geometry"], "geometry_parameters": truth["geometry_parameters"],
        "split": truth["split"], "ground_truth_world_from_object": truth["ground_truth_world_from_object"],
        "surface_points_object_m": _pack_array(np.asarray(truth["surface_points_object_m"], dtype=np.float32)),
        "surface_region_label_names": label_names,
        "surface_region_codes": _pack_array(codes),
        "oracle_contact_points_object_m": _pack_array(
            np.asarray(truth["oracle_contact_points_object_m"], dtype=np.float32)),
        "rgb_feature_compute": truth["rgb_feature_compute"],
    }


def _truth_from_dict(value):
    names = value["surface_region_label_names"]
    codes = _unpack_array(value["surface_region_codes"])
    return {
        "surface_points_object_m": _unpack_array(value["surface_points_object_m"]),
        "surface_region_labels": np.asarray([names[int(code)] for code in codes]),
        "oracle_contact_points_object_m": _unpack_array(value["oracle_contact_points_object_m"]),
    }


def _clean_truth(spec, observation, generated):
    return {
        "object_instance_id": spec.instance_id, "geometry": spec.geometry,
        "geometry_parameters": spec.parameters, "split": spec.split,
        "ground_truth_world_from_object": observation.object_frame.world_from_object.tolist(),
        "surface_points_object_m": generated["surface_points_object_m"],
        "surface_region_labels": generated["surface_region_labels"],
        "oracle_contact_points_object_m": generated["oracle_contact_points_object_m"],
        "rgb_feature_compute": generated["rgb_feature_efficiency"],
    }


def _aggregate(records, protocol):
    threshold = str(float(protocol["evaluation"]["distance_thresholds_m"][1]))
    summary = {"schema_version": 1, "protocol_version": protocol["protocol_version"],
               "attempt_count": len({item["case_id"] for item in records}), "methods": {},
               "held_out_test": {}, "sensor_sparsity": {}, "by_geometry": {},
               "spatial_coverage": {}}

    def aggregate(subset):
        valid = [item for item in subset if item["algorithm_status"] == "VALID_RECONSTRUCTION"]
        def mean(path):
            values = []
            for item in valid:
                value = item
                for key in path: value = value[key]
                if value is not None: values.append(value)
            return float(np.mean(values)) if values else None
        return valid, mean

    for method in PRIMARY_METHODS:
        subset = [item for item in records if item["method"] == method]
        valid, mean = aggregate(subset)
        summary["methods"][method] = {
            "valid": len(valid), "attempted": len(subset), "success_rate": len(valid)/max(1, len(subset)),
            "mean_chamfer_l1_m": mean(["whole_surface_chamfer_l1_m"]),
            "mean_visible_error_m": mean(["regions", "visible_to_rgb", "mean_point_to_surface_m"]),
            "mean_occluded_error_m": mean(["regions", "occluded_by_hand", "mean_point_to_surface_m"]),
            "mean_other_unobserved_error_m": mean(["regions", "other_unobserved", "mean_point_to_surface_m"]),
            "mean_contact_region_error_m": mean(["contact_region", "mean_point_to_surface_m"]),
            "mean_f_score_5mm": mean(["f_score", threshold]),
            "mean_runtime_ms": float(np.mean([item["runtime_ms"] for item in subset])),
        }
    held = [item for item in records if item["split"] == "test"]
    for method in PRIMARY_METHODS:
        subset = [item for item in held if item["method"] == method]
        valid, mean = aggregate(subset)
        summary["held_out_test"][method] = {
            "valid": len(valid), "attempted": len(subset),
            "mean_occluded_error_m": mean(["regions", "occluded_by_hand", "mean_point_to_surface_m"]),
            "mean_visible_error_m": mean(["regions", "visible_to_rgb", "mean_point_to_surface_m"]),
        }
    for geometry in sorted({item["geometry"] for item in records}):
        summary["by_geometry"][geometry] = {}
        for method in PRIMARY_METHODS:
            subset = [item for item in records if item["geometry"] == geometry and item["method"] == method]
            valid, mean = aggregate(subset)
            summary["by_geometry"][geometry][method] = {
                "valid": len(valid),
                "mean_occluded_error_m": mean(["regions", "occluded_by_hand", "mean_point_to_surface_m"]),
            }
    for count in protocol["active_sensor_counts"]:
        summary["sensor_sparsity"][str(count)] = {}
        for method in PRIMARY_METHODS[:-1]:
            subset = [item for item in records if item["method"] == method and item["active_sensor_count"] == count]
            valid, mean = aggregate(subset)
            summary["sensor_sparsity"][str(count)][method] = {
                "valid": len(valid), "attempted": len(subset),
                "dropout_fraction_relative_to_8": 1.0-float(count)/8.0,
                "contributing_fingers": min(int(count), 4),
                "mean_occluded_error_m": mean(["regions", "occluded_by_hand", "mean_point_to_surface_m"]),
            }
    for label, lower, upper in (("low", 0.0, .34), ("medium", .34, .67), ("high", .67, 1.01)):
        summary["spatial_coverage"][label] = {}
        for method in PRIMARY_METHODS[:-1]:
            subset = [item for item in records if item["method"] == method and
                      lower <= float(item["diagnostics"].get("coverage_score", 0)) < upper]
            valid, mean = aggregate(subset)
            summary["spatial_coverage"][label][method] = {
                "valid": len(valid), "attempted": len(subset),
                "mean_occluded_error_m": mean(["regions", "occluded_by_hand", "mean_point_to_surface_m"]),
            }
    rgb = summary["held_out_test"]["rgb_only"]["mean_occluded_error_m"]
    rgb_by_case = {item["case_id"]: item for item in held if item["method"] == "rgb_only"}
    summary["central_result"] = {}
    for method in ("rgb_representative_tactile", "rgb_finite_patch_tactile", "rgb_reliability_tactile"):
        value = summary["held_out_test"][method]["mean_occluded_error_m"]
        paired = []
        for item in held:
            if item["method"] != method: continue
            rgb_item = rgb_by_case[item["case_id"]]
            a = rgb_item["regions"]["occluded_by_hand"]["mean_point_to_surface_m"]
            b = item["regions"]["occluded_by_hand"]["mean_point_to_surface_m"]
            if a is not None and b is not None: paired.append(a-b)
        summary["central_result"][method] = {
            "paired_mean_occluded_improvement_m_vs_rgb": None if value is None or rgb is None else rgb-value,
            "improved": bool(value is not None and rgb is not None and value < rgb),
            "paired_case_count": len(paired), "cases_improved": sum(v > 1e-12 for v in paired),
            "cases_worsened": sum(v < -1e-12 for v in paired),
            "cases_tied": sum(abs(v) <= 1e-12 for v in paired),
        }
    return summary


def run(protocol_path=DEFAULT_PROTOCOL, config_path=DEFAULT_CONFIG, output=DEFAULT_OUTPUT,
        feature_weights=None):
    protocol_path, config_path, output = map(Path, (protocol_path, config_path, output))
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    solver_config = {**config["solver"], **config["reliability"]}
    if config["features"]["primary_encoder"].startswith("torchvision-resnet18"):
        if feature_weights is None:
            raise ValueError("Primary pretrained RGB encoder requires --feature-weights")
        feature_weights = Path(feature_weights)
        if _sha256(feature_weights) != config["features"]["weight_sha256"]:
            raise ValueError("Pretrained RGB feature weight SHA-256 does not match frozen config")
        feature_encoder = LocalResNet18Encoder(
            feature_weights, expected_sha256=config["features"]["weight_sha256"])
    else:
        feature_encoder = AnalyticColorEdgeEncoder()
    output.mkdir(parents=True, exist_ok=True)
    case_manifest, records, feature_reports = [], [], []
    case_index = 0
    shape_by_id = {item.instance_id: item for item in SHAPES}
    for split, instance_ids in protocol["objects"].items():
        for instance_id in instance_ids:
            spec = shape_by_id[instance_id]
            if spec.split != split: raise ValueError("Protocol and generator split disagree")
            for pose_index, pose in enumerate(protocol["poses"]):
                for sensor_count in protocol["active_sensor_counts"]:
                    case_index += 1; case_id = f"case_{case_index:03d}"
                    seed = int(protocol["random_seed_base"])+case_index
                    observation, generated = generate_case(
                        spec, seed=seed, active_sensor_count=int(sensor_count),
                        grid_resolution=int(config["representation"]["grid_resolution"]), **pose,
                        image_downsample=int(protocol["camera"]["downsample_factor"]),
                        feature_encoder=feature_encoder,
                    )
                    feature_reports.append(generated["rgb_feature_efficiency"])
                    truth = _clean_truth(spec, observation, generated)
                    input_path = output/"algorithm_inputs"/case_id/"observation.json"
                    truth_path = output/"evaluation_truth"/case_id/"ground_truth.json"
                    _json(input_path, observation_to_dict(observation))
                    _json(truth_path, _truth_to_dict(truth))
                    # This reload is the programmatic isolation boundary: methods receive only input JSON.
                    isolated_observation = observation_from_dict(json.loads(input_path.read_text()))
                    isolated_truth = _truth_from_dict(json.loads(truth_path.read_text()))
                    case_entry = {
                        "case_id": case_id, "split": split, "object_instance_id": instance_id,
                        "geometry": spec.geometry, "pose_index": pose_index,
                        "active_sensor_count": int(sensor_count), "seed": seed,
                        "algorithm_input": str(input_path.relative_to(output)),
                        "evaluation_truth": str(truth_path.relative_to(output)),
                    }
                    case_manifest.append(case_entry)
                    dashboard_payload = None
                    for method in PRIMARY_METHODS:
                        if method == "rgb_perfect_contact_oracle":
                            result = reconstruct_perfect_contact_oracle(
                                isolated_observation, solver_config,
                                isolated_truth["oracle_contact_points_object_m"])
                        else:
                            result = reconstruct_general_object(
                                isolated_observation, method, solver_config)
                        metrics = evaluate_surface(result, isolated_truth,
                            thresholds_m=tuple(protocol["evaluation"]["distance_thresholds_m"]))
                        record = {**metrics, "case_id": case_id, "split": split,
                                  "object_instance_id": instance_id, "geometry": spec.geometry,
                                  "active_sensor_count": int(sensor_count), "runtime_ms": result.runtime_ms,
                                  "confidence": result.confidence, "diagnostics": dict(result.diagnostics)}
                        records.append(record)
                        _json(output/"results"/case_id/f"{method}.json", {
                            "algorithm": result.to_dict(include_surface=False), "evaluation": metrics,
                        })
                        if instance_id == "test_asymmetric_l_01" and pose_index == 1 and sensor_count == 4:
                            dashboard_payload = dashboard_payload or {
                                "case_id": case_id, "input": observation_to_dict(isolated_observation),
                                "evaluation_only_gt_enabled_by_default": False, "methods": {},
                            }
                            dashboard_payload["methods"][method] = result.to_dict(include_surface=True, max_points=450)
                            if method == "rgb_reliability_tactile":
                                dashboard_payload["evaluation_only_gt_surface_points_object_m"] = (
                                    isolated_truth["surface_points_object_m"][::max(1, len(isolated_truth["surface_points_object_m"])//450)].tolist())
                    if dashboard_payload:
                        _json(output/"dashboard_case.json", dashboard_payload)
    if case_index != int(protocol["attempt_count"]):
        raise ValueError("Frozen attempt_count does not match generated cases")
    dataset_manifest = {
        "schema_version": 1, "protocol_version": protocol["protocol_version"],
        "protocol_sha256": _sha256(protocol_path), "config_sha256": _sha256(config_path),
        "rgb_feature_encoder": config["features"]["primary_encoder"],
        "rgb_feature_weight_sha256": config["features"].get("weight_sha256"),
        "algorithm_input_and_evaluation_isolation": (
            "separate paths; reconstructor reloads only algorithm_inputs"),
        "training_performed": False, "split_policy": protocol["objects"], "cases": case_manifest,
    }
    _json(output/"dataset_manifest.json", dataset_manifest)
    _json(output/"evaluation_records.json", records)
    summary = _aggregate(records, protocol)
    _json(output/"evaluation_summary.json", summary)
    runtimes = [record["runtime_ms"] for record in records]
    _json(output/"compute_report.json", {
        "total_parameter_count": int(feature_reports[0]["parameter_count"]),
        "trainable_parameter_count": 0,
        "learned_sdf_decoder": False, "frozen_pretrained_rgb_encoder": True,
        "primary_encoder": config["features"]["primary_encoder"],
        "primary_encoder_parameter_count": int(feature_reports[0]["parameter_count"]),
        "primary_encoder_estimated_flops_per_224px_frame": int(
            feature_reports[0]["estimated_flops"]),
        "primary_encoder_weight_sha256": config["features"].get("weight_sha256"),
        "fixed_feature_ablation": config["features"]["fixed_weight_ablation"],
        "feature_runtime_ms": {
            "mean": statistics.mean(item["runtime_ms"] for item in feature_reports),
            "maximum": max(item["runtime_ms"] for item in feature_reports),
        },
        "dense_sdf_float32_memory_bytes": int(config["representation"]["grid_resolution"]**3*4),
        "runtime_ms": {"mean": statistics.mean(runtimes), "median": statistics.median(runtimes),
                       "minimum": min(runtimes), "maximum": max(runtimes)},
        "gpu_memory_bytes": 0, "execution_device": "CPU/NumPy",
        "flops_note": "No neural decoder. Per-result deterministic grid-operation estimates are recorded in diagnostics.",
    })
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--feature-weights")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        protocol = yaml.safe_load(Path(args.protocol).read_text(encoding="utf-8"))
        records = json.loads((Path(args.output)/"evaluation_records.json").read_text(encoding="utf-8"))
        summary = _aggregate(records, protocol)
        _json(Path(args.output)/"evaluation_summary.json", summary)
    else:
        summary = run(args.protocol, args.config, args.output, args.feature_weights)
    print(json.dumps(summary["central_result"], indent=2))


if __name__ == "__main__":
    main()
