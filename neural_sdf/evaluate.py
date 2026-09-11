"""Held-out evaluation, tactile controls, robustness, and dashboard artifact."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import resource
import statistics
import time

import numpy as np
import torch
import yaml

from general_object.evaluation import evaluate_surface
from general_object.sdf import DenseSDF
from general_object.serialization import observation_to_dict
from .data import collate_model_inputs, materialize_with_rgb_features, split_records
from .dataset import DEFAULT_MANIFEST, load_sample
from .model import FrozenResNet18GridEncoder, MODES, NeuralSDFModel


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "neural_sdf/config_v1.yaml"
DEFAULT_OUTPUT = ROOT / "experiments/neural_sdf/learned_v1_20260911"


def load_model(mode, config, output, device="cpu"):
    model_config = config["model"]
    model = NeuralSDFModel(
        mode, rgb_channels=int(config["rgb_encoder"]["output_channels"]),
        tactile_hidden=int(model_config["tactile_hidden"]),
        decoder_hidden=int(model_config["decoder_hidden"]),
        fourier_frequencies=int(model_config["query_fourier_frequencies"])).to(device)
    checkpoint = torch.load(Path(output)/"checkpoints"/mode/"best.pt",
                            map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"]); model.eval()
    return model, checkpoint


def _clone_input(value):
    return {key: ({name: tensor.clone() for name, tensor in item.items()}
                  if isinstance(item, dict) else item.clone())
            for key, item in value.items()}


def _concatenate_inputs(items):
    return {
        key: ({name: torch.cat([item[key][name] for item in items], dim=0)
               for name in items[0][key]}
              if isinstance(items[0][key], dict)
              else torch.cat([item[key] for item in items], dim=0))
        for key in items[0]
    }


def controlled_input(base, control="normal", donor=None):
    """Apply test-only controls without reading evaluator geometry."""
    value = _clone_input(base)
    tactile = value["tactile"]
    if control == "normal":
        return value
    if control == "zero_tactile":
        tactile["valid"][:] = False; value["coverage"][:] = 0
    elif control == "shuffled_tactile":
        if donor is None: raise ValueError("shuffled_tactile requires a donor")
        value["tactile"] = {key: item.clone() for key, item in donor["tactile"].items()}
        value["coverage"] = donor["coverage"].clone()
    elif control == "position_shuffled":
        offset = value["tactile"]["position"].new_tensor([.019, -.013, .011])
        tactile["position"] = (tactile["position"]+offset).clamp(-.064, .064)
        tactile["patch"] = (tactile["patch"]+offset).clamp(-.064, .064)
    elif control == "scalar_shuffled":
        if donor is None: raise ValueError("scalar_shuffled requires a donor")
        tactile["scalar_n"] = donor["tactile"]["scalar_n"].clone()
        tactile["indentation_m"] = donor["tactile"]["indentation_m"].clone()
    elif control == "sensor_mount_error":
        offset = value["tactile"]["position"].new_tensor([.003, -.002, .001])
        tactile["position"] += offset; tactile["patch"] += offset
    elif control == "joint_bias_proxy":
        # Static synthetic FK is already materialized as sensor geometry.  This
        # perturbation applies its declared 2 mm pose consequence and records
        # the joint joint-vector bias separately.
        tactile["position"][..., 2] += .002; tactile["patch"][..., 2] += .002
        value["joint_state"] += .02
    elif control in {"tactile_dropout_25pct", "tactile_dropout_50pct",
                     "tactile_dropout_75pct"}:
        fraction = int(control.split("_")[-1].removesuffix("pct"))/100
        # Fixed named-channel subsets make repeated runs exactly comparable.
        sensor_index = tactile["sensor_index"]
        keep = ((sensor_index*37+11) % 100) >= int(fraction*100)
        tactile["valid"] &= keep
        value["coverage"] *= (1-fraction)
    elif control == "incorrect_contact_geometry":
        offset = value["tactile"]["position"].new_tensor([.02, .0, -.015])
        tactile["position"] = (tactile["position"]+offset).clamp(-.064, .064)
        tactile["patch"] = (tactile["patch"]+offset).clamp(-.064, .064)
    elif control == "camera_extrinsic_error":
        angle = np.deg2rad(2.0); c, s = np.cos(angle), np.sin(angle)
        rotation = value["camera_from_object"].new_tensor(
            [[c, -s, 0, .004], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        value["camera_from_object"] = torch.bmm(rotation[None].expand(
            len(value["camera_from_object"]), -1, -1), value["camera_from_object"])
    elif control == "segmentation_degradation":
        masks = value["visibility_masks"]
        object_mask = masks[:, 0:1]
        eroded = -torch.nn.functional.max_pool2d(-object_mask, 5, stride=1, padding=2)
        removed = (object_mask-eroded).clamp(0, 1)
        masks[:, 0:1] = eroded; masks[:, 3:4] = torch.maximum(masks[:, 3:4], removed)
    else:
        raise ValueError(control)
    return value


@torch.no_grad()
def reconstruct(model, model_input, resolution, *, method, device="cpu"):
    axis = torch.linspace(-.065, .065, resolution, device=device)
    query = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(-1, 3)
    values = []; started = time.perf_counter()
    for start in range(0, len(query), 8192):
        values.append(model(query[start:start+8192][None], model_input).cpu())
    field = torch.cat(values, dim=1).reshape(resolution, resolution, resolution).numpy()
    runtime = (time.perf_counter()-started)*1000
    result = DenseSDF(np.array([[-.065]*3, [.065]*3]), field.astype(np.float32),
                      method, "VALID_RECONSTRUCTION", 0.0,
                      {"ground_truth_input": False}, runtime)
    if len(result.surface_points()) < 20:
        result = DenseSDF(result.bounds_m, result.values_m, method, "INVALID_NO_SURFACE",
                          0.0, result.diagnostics, runtime)
    return result


@torch.no_grad()
def reconstruct_batch(model, model_input, resolution, *, method, device="cpu"):
    """Extract several independent fields in one vectorized forward pass."""
    batch_size = int(model_input["intrinsic"].shape[0])
    axis = torch.linspace(-.065, .065, resolution, device=device)
    query = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(-1, 3)
    values = []; started = time.perf_counter()
    # A smaller chunk controls peak memory while batching amortizes CPU model overhead.
    for start in range(0, len(query), 4096):
        expanded = query[start:start+4096][None].expand(batch_size, -1, -1)
        values.append(model(expanded, model_input).cpu())
    elapsed_ms = (time.perf_counter()-started)*1000
    fields = torch.cat(values, dim=1).reshape(
        batch_size, resolution, resolution, resolution).numpy()
    results = []
    for field in fields:
        result = DenseSDF(np.array([[-.065]*3, [.065]*3]), field.astype(np.float32),
                          method, "VALID_RECONSTRUCTION", 0.0,
                          {"ground_truth_input": False}, elapsed_ms/batch_size)
        if len(result.surface_points()) < 20:
            result = DenseSDF(
                result.bounds_m, result.values_m, method, "INVALID_NO_SURFACE",
                0.0, result.diagnostics, elapsed_ms/batch_size)
        results.append(result)
    return results


def _truth(record):
    names = np.asarray(("visible_to_rgb", "occluded_by_hand", "other_unobserved"))
    return {
        "surface_points_object_m": record["supervision"]["surface_points"].numpy(),
        "surface_region_labels": names[record["supervision"]["surface_labels"].numpy()],
        "oracle_contact_points_object_m": record["supervision"]["oracle_contact_points"].numpy(),
    }


def _metric_summary(records, key_path, seed=51):
    values = []
    for item in records:
        value = item
        for key in key_path: value = value[key]
        if value is not None: values.append(float(value))
    if not values:
        return {"count": 0, "mean": None, "median": None, "std": None, "bootstrap_95_ci": None}
    rng = np.random.default_rng(seed); array = np.asarray(values)
    means = [np.mean(rng.choice(array, len(array), replace=True)) for _ in range(500)]
    return {"count": len(values), "mean": float(np.mean(array)),
            "median": float(np.median(array)), "std": float(np.std(array)),
            "bootstrap_95_ci": [float(np.percentile(means, 2.5)),
                                float(np.percentile(means, 97.5))]}


def aggregate(records):
    result = {}
    paths = {
        "whole_surface_chamfer_l1_m": ("whole_surface_chamfer_l1_m",),
        "visible_error_m": ("regions", "visible_to_rgb", "mean_point_to_surface_m"),
        "hand_occluded_error_m": ("regions", "occluded_by_hand", "mean_point_to_surface_m"),
        "other_unobserved_error_m": ("regions", "other_unobserved", "mean_point_to_surface_m"),
        "contact_region_error_m": ("contact_region", "mean_point_to_surface_m"),
        "f_score_5mm": ("f_score", "0.005"),
        "surface_completeness_5mm": ("recall_completeness", "0.005"),
        "surface_precision_5mm": ("precision", "0.005"),
    }
    for mode in sorted({item["mode"] for item in records}):
        mode_records = [item for item in records if item["mode"] == mode]
        result[mode] = {}
        for subset_name, subset in (
            ("all", mode_records),
            ("contact_present", [x for x in mode_records if x["active_sensor_count"] > 0]),
            ("no_contact", [x for x in mode_records if x["active_sensor_count"] == 0]),
        ):
            valid = [item for item in subset if item["algorithm_status"] == "VALID_RECONSTRUCTION"]
            result[mode][subset_name] = {
                "valid": len(valid), "attempted": len(subset),
                **{name: _metric_summary(valid, path) for name, path in paths.items()},
            }
    return result


def paired_comparison(records, reference_mode, candidate_mode, key_path, seed=51001):
    """Paired candidate-minus-reference statistics on identical case IDs."""
    grouped = {}
    for item in records:
        if item["mode"] in {reference_mode, candidate_mode}:
            grouped.setdefault(item["sample_id"], {})[item["mode"]] = item
    differences = []
    for modes in grouped.values():
        if set(modes) != {reference_mode, candidate_mode}:
            continue
        values = []
        for mode in (reference_mode, candidate_mode):
            value = modes[mode]
            for key in key_path:
                value = value[key]
            values.append(float(value))
        differences.append(values[1]-values[0])
    array = np.asarray(differences)
    rng = np.random.default_rng(seed)
    means = [np.mean(rng.choice(array, len(array), replace=True)) for _ in range(500)]
    return {
        "reference": reference_mode, "candidate": candidate_mode,
        "difference_definition": "candidate_minus_reference",
        "paired_case_count": len(array), "mean_difference": float(np.mean(array)),
        "median_difference": float(np.median(array)),
        "improved_case_fraction": float(np.mean(array < 0)),
        "bootstrap_95_ci": [float(np.percentile(means, 2.5)),
                             float(np.percentile(means, 97.5))],
    }


def evaluate_modes(models, records, config, *, resolution, device="cpu"):
    output = []
    thresholds = tuple(config["evaluation"]["distance_thresholds_m"])
    for mode, model in models.items():
        for start in range(0, len(records), 8):
            subset = records[start:start+8]
            model_input = collate_model_inputs(subset, device)
            results = reconstruct_batch(
                model, model_input, resolution, method=mode, device=device)
            for local_index, (record, result) in enumerate(zip(subset, results)):
                metrics = evaluate_surface(result, _truth(record), thresholds_m=thresholds)
                valid = model_input["tactile"]["valid"][local_index]
                output.append({
                    **metrics, "mode": mode, "sample_id": record["audit"]["sample_id"],
                    "instance_id": record["audit"]["instance_id"],
                    "family": record["audit"]["family"],
                    "active_sensor_count": int(valid.sum()),
                    "finger_count": len(set(model_input["tactile"]["finger_index"][local_index]
                                            [valid].tolist())),
                    "coverage": float(model_input["coverage"][local_index, 0]),
                    "contact_pattern": record["audit"]["contact_pattern"],
                    "runtime_ms": result.runtime_ms, "extraction_resolution": resolution,
                })
            completed = min(start+len(subset), len(records))
            if completed % 24 == 0 or completed == len(records):
                print(f"evaluated {mode} {completed}/{len(records)} @ {resolution}", flush=True)
    return output


def evaluate_controls(model, records, config, *, controls, resolution=24, device="cpu"):
    output = []; thresholds = tuple(config["evaluation"]["distance_thresholds_m"])
    donors = [item for item in records if item["model_input"]["tactile"]["valid"].any()]
    for control in controls:
        for start in range(0, len(records), 8):
            subset = records[start:start+8]
            bases, altered = [], []
            for local_index, record in enumerate(subset):
                index = start+local_index
                base = collate_model_inputs([record], device)
                donor_record = donors[(index+17) % len(donors)]
                donor = collate_model_inputs([donor_record], device)
                bases.append(base); altered.append(controlled_input(base, control, donor))
            results = reconstruct_batch(
                model, _concatenate_inputs(altered), resolution,
                method=f"{model.mode}:{control}", device=device)
            for record, base, result in zip(subset, bases, results):
                metrics = evaluate_surface(result, _truth(record), thresholds_m=thresholds)
                output.append({
                    **metrics, "mode": control, "sample_id": record["audit"]["sample_id"],
                    "family": record["audit"]["family"],
                    "active_sensor_count": int(base["tactile"]["valid"].sum()),
                    "runtime_ms": result.runtime_ms, "extraction_resolution": resolution,
                })
        print(f"evaluated control {control} @ {resolution}", flush=True)
    return output


def _by_family(records):
    result = {}
    for family in sorted({item["family"] for item in records}):
        subset = [item for item in records if item["family"] == family]
        result[family] = aggregate(subset)
    return result


def _sparsity(records):
    sensor_counts = {}
    for count in sorted({item["active_sensor_count"] for item in records}):
        subset = [item for item in records if item["active_sensor_count"] == count]
        sensor_counts[str(count)] = aggregate(subset)
    finger_counts = {}
    for count in sorted({item["finger_count"] for item in records}):
        finger_counts[str(count)] = aggregate(
            [item for item in records if item["finger_count"] == count])
    coverage_bins = {}
    for name, lower, upper in (("zero", -.01, .000001), ("low", .000001, .20),
                               ("medium", .20, .55), ("high", .55, 1.01)):
        subset = [item for item in records if lower <= item["coverage"] < upper]
        coverage_bins[name] = aggregate(subset) if subset else {}
    patterns = {}
    for pattern in sorted({item["contact_pattern"] for item in records}):
        patterns[pattern] = aggregate(
            [item for item in records if item["contact_pattern"] == pattern])
    return {"by_active_sensor_count": sensor_counts,
            "by_finger_count": finger_counts,
            "by_spatial_coverage": coverage_bins,
            "by_contact_pattern": patterns}


def _checkpoint_bytes(output, modes):
    return {mode: (Path(output)/"checkpoints"/mode/"best.pt").stat().st_size for mode in modes}


def _load_records(path):
    return json.loads(path.read_text()) if path.is_file() else []


def _write_records(path, records):
    path.write_text(json.dumps(records, indent=2)+"\n")


def run(config_path, output, weights_path, *, device="cpu"):
    config = yaml.safe_load(Path(config_path).read_text())
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    encoder = FrozenResNet18GridEncoder(weights_path, config["rgb_encoder"]["weight_sha256"])
    records = materialize_with_rgb_features(encoder, device=device)
    test = split_records(records, "test")
    validation = split_records(records, "validation")
    models = {mode: load_model(mode, config, output, device)[0] for mode in MODES}
    selected_resolution = int(config["evaluation"]["selected_resolution"])
    primary_path = output/"evaluation_records.json"
    primary = _load_records(primary_path)
    for mode in MODES:
        complete = sum(item["mode"] == mode for item in primary)
        if complete == len(test):
            print(f"resuming: {mode} primary already complete", flush=True)
            continue
        primary = [item for item in primary if item["mode"] != mode]
        primary.extend(evaluate_modes(
            {mode: models[mode]}, test, config,
            resolution=selected_resolution, device=device))
        _write_records(primary_path, primary)
    primary_summary = aggregate(primary)

    resolution_records = validation[::4]
    resolution_study = {}
    for resolution in config["evaluation"]["extraction_resolutions"]:
        resolution_path = output/f"resolution_validation_{int(resolution)}_records.json"
        values = _load_records(resolution_path)
        for mode in ("learned_rgb_only", "learned_rgb_tactile_reliability"):
            if sum(item["mode"] == mode for item in values) == len(resolution_records):
                continue
            values = [item for item in values if item["mode"] != mode]
            values.extend(evaluate_modes(
                {mode: models[mode]}, resolution_records, config,
                resolution=int(resolution), device=device))
            _write_records(resolution_path, values)
        resolution_study[str(resolution)] = aggregate(values)

    negative_names = (
        "normal", "zero_tactile", "shuffled_tactile",
        "position_shuffled", "scalar_shuffled",
    )
    negative_path = output/"negative_control_records.json"
    negative = _load_records(negative_path)
    for control in negative_names:
        if sum(item["mode"] == control for item in negative) == len(test):
            continue
        negative = [item for item in negative if item["mode"] != control]
        negative.extend(evaluate_controls(
            models["learned_rgb_tactile_reliability"], test, config,
            controls=(control,), resolution=24, device=device))
        _write_records(negative_path, negative)
    robustness_names = (
        "camera_extrinsic_error", "sensor_mount_error", "joint_bias_proxy",
        "tactile_dropout_25pct", "tactile_dropout_50pct",
        "tactile_dropout_75pct", "incorrect_contact_geometry",
        "segmentation_degradation",
    )
    robustness = {}
    for mode in ("learned_rgb_only", "learned_rgb_tactile_reliability"):
        robustness_path = output/f"robustness_{mode}_records.json"
        values = _load_records(robustness_path)
        for control in ("normal",)+robustness_names:
            if sum(item["mode"] == control for item in values) == len(test):
                continue
            values = [item for item in values if item["mode"] != control]
            values.extend(evaluate_controls(
                models[mode], test, config, controls=(control,),
                resolution=24, device=device))
            _write_records(robustness_path, values)
        robustness[mode] = aggregate(values)

    result_payload = {
        "schema_version": 1, "configuration_version": config["configuration_version"],
        "preliminary_single_training_seed": True,
        "checkpoint_selected_on_validation_only": True,
        "selected_extraction_resolution": selected_resolution,
        "primary_test": primary_summary, "by_held_out_family": _by_family(primary),
        "paired_primary_hand_occluded_vs_rgb_only": {
            mode: {
                subset: paired_comparison(
                    ([item for item in primary if item["active_sensor_count"] > 0]
                     if subset == "contact_present" else
                     [item for item in primary if item["active_sensor_count"] == 0]
                     if subset == "no_contact" else primary),
                    "learned_rgb_only", mode,
                    ("regions", "occluded_by_hand", "mean_point_to_surface_m"))
                for subset in ("all", "contact_present", "no_contact")
            }
            for mode in MODES if mode != "learned_rgb_only"
        },
        "paired_primary_l4_minus_l0": {
            "all": paired_comparison(
                primary, "learned_rgb_only", "learned_rgb_tactile_reliability",
                ("regions", "occluded_by_hand", "mean_point_to_surface_m")),
            "contact_present": paired_comparison(
                [item for item in primary if item["active_sensor_count"] > 0],
                "learned_rgb_only", "learned_rgb_tactile_reliability",
                ("regions", "occluded_by_hand", "mean_point_to_surface_m")),
            "no_contact": paired_comparison(
                [item for item in primary if item["active_sensor_count"] == 0],
                "learned_rgb_only", "learned_rgb_tactile_reliability",
                ("regions", "occluded_by_hand", "mean_point_to_surface_m")),
        },
        "sensor_sparsity": _sparsity(primary),
        "negative_controls_24": aggregate(negative),
        "paired_negative_controls_vs_normal_24": {
            control: paired_comparison(
                negative, "normal", control,
                ("regions", "occluded_by_hand", "mean_point_to_surface_m"))
            for control in negative_names if control != "normal"
        },
        "resolution_validation_subset": resolution_study,
        "robustness_24": robustness,
    }
    _write_records(output/"evaluation_records.json", primary)
    _write_records(output/"negative_control_records.json", negative)
    (output/"evaluation_summary.json").write_text(json.dumps(result_payload, indent=2)+"\n")
    runtimes = [item["runtime_ms"] for item in primary]
    compute = {
        "execution_device": device, "cuda_available": torch.cuda.is_available(),
        "frozen_rgb_parameters": encoder.parameter_count,
        "frozen_rgb_estimated_flops_per_frame": encoder.estimated_flops_per_frame,
        "trainable_parameters_by_model": {mode: model.trainable_parameter_count
                                          for mode, model in models.items()},
        "checkpoint_bytes": _checkpoint_bytes(output, MODES),
        "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
        "gpu_peak_memory_bytes": (torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0),
        "inference_latency_ms": {"mean": statistics.mean(runtimes),
                                 "median": statistics.median(runtimes),
                                 "maximum": max(runtimes)},
        "sdf_query_throughput_per_second": float(
            selected_resolution**3/(statistics.mean(runtimes)/1000)),
        "selected_query_count": selected_resolution**3,
    }
    (output/"compute_report.json").write_text(json.dumps(compute, indent=2)+"\n")

    # One held-out asymmetric sample drives a prediction-only dashboard replay.
    dashboard_record = next(item for item in test if item["audit"]["family"] == "asymmetric_l"
                            and item["model_input"]["tactile"]["valid"].sum() >= 4)
    manifest = json.loads(Path(DEFAULT_MANIFEST).read_text())
    sample_raw = next(item for item in manifest["samples"]
                      if item["sample_id"] == dashboard_record["audit"]["sample_id"])
    original = load_sample(manifest, sample_raw).model_input
    dashboard = {
        "case_id": dashboard_record["audit"]["sample_id"],
        "checkpoint_selection": "validation_only", "preliminary_single_seed": True,
        "input": observation_to_dict(original), "methods": {},
        "evaluation_only_gt_enabled_by_default": False,
        "evaluation_only_gt_surface_points_object_m": _truth(dashboard_record)["surface_points_object_m"][::12].tolist(),
    }
    batch = collate_model_inputs([dashboard_record], device)
    for mode, model in models.items():
        reconstructed = reconstruct(model, batch, selected_resolution, method=mode, device=device)
        dashboard["methods"][mode] = reconstructed.to_dict(include_surface=True, max_points=700)
        dashboard["methods"][mode]["checkpoint"] = str(
            (output/"checkpoints"/mode/"best.pt").relative_to(ROOT))
        dashboard["methods"][mode]["active_sensor_count"] = int(
            batch["tactile"]["valid"].sum())
        dashboard["methods"][mode]["observable_reliability"] = float(batch["coverage"][0, 0])
    (output/"dashboard_case.json").write_text(json.dumps(dashboard, indent=2)+"\n")
    return result_payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--feature-weights", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run(args.config, args.output, args.feature_weights, device=args.device)


if __name__ == "__main__":
    main()
