import hashlib
import json
from pathlib import Path

from backend.dashboard_server import (
    DASHBOARD_BUILD_ID,
    _load_neural_sdf_dashboard_case,
)
from general_object.serialization import observation_to_dict
from neural_sdf.dataset import load_sample


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "experiments/neural_sdf/dataset_manifest_v1.json"


def test_neural_dataset_manifest_is_immutable_and_instance_disjoint():
    manifest = json.loads(MANIFEST_PATH.read_text())
    recorded = MANIFEST_PATH.with_suffix(".json.sha256").read_text().split()[0]
    assert hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest() == recorded
    assert manifest["counts"] == {
        "instances": 84,
        "samples": 432,
        "by_split": {"test": 96, "train": 288, "validation": 48},
    }
    by_split = {
        split: {item["instance_id"] for item in manifest["instances"]
                if item["split"] == split}
        for split in ("train", "validation", "test")
    }
    assert not (by_split["train"] & by_split["validation"])
    assert not (by_split["train"] & by_split["test"])
    assert not (by_split["validation"] & by_split["test"])
    l_shapes = [item for item in manifest["instances"]
                if item["family"] == "asymmetric_l"]
    assert len(l_shapes) == 8
    assert {item["split"] for item in l_shapes} == {"test"}


def test_serialized_model_input_has_no_truth_identity_or_shape_parameters():
    manifest = json.loads(MANIFEST_PATH.read_text())
    sample_record = next(item for item in manifest["samples"]
                         if item["split"] == "test" and item["active_sensor_count"] == 4)
    sample = load_sample(manifest, sample_record)
    raw = observation_to_dict(sample.model_input)
    forbidden = {
        "family", "geometry_family", "object_instance_id", "shape_parameters",
        "radius", "radius_m", "ground_truth_surface", "oracle_contact_points",
    }
    assert not (forbidden & set(raw))
    assert sample.model_input.object_frame.source == "declared_static_fixture"
    assert sample.model_input.object_frame.validity == "VALID"
    assert len(sample.model_input.tactile_observations) == 4
    assert all(item.sensor_id.startswith("sim_sensor_")
               for item in sample.model_input.tactile_observations)


def test_neural_dashboard_mode_and_gt_boundary_are_integrated():
    html = (ROOT / "ui/index.html").read_text()
    script = (ROOT / "ui/app.js").read_text()
    assert DASHBOARD_BUILD_ID == "learned-neural-sdf-v1-20260911.1"
    assert 'option value="neural_sdf"' in html
    assert 'id="neural-sdf-method"' in html
    assert "neural_sdf_reconstruction" in script
    artifact = _load_neural_sdf_dashboard_case()
    assert artifact["status"] in {"AVAILABLE", "UNAVAILABLE"}
    if artifact["status"] == "AVAILABLE":
        assert set(artifact["methods"]) == {
            "learned_rgb_only", "learned_rgb_tactile_global",
            "learned_rgb_tactile_representative",
            "learned_rgb_tactile_finite_patch",
            "learned_rgb_tactile_reliability",
        }
        assert artifact["evaluation_only_gt_enabled_by_default"] is False


def test_saved_held_out_evaluation_is_paired_and_reports_no_gt_input():
    output = ROOT / "experiments/neural_sdf/learned_v1_20260911"
    records = json.loads((output / "evaluation_records.json").read_text())
    modes = {item["mode"] for item in records}
    assert len(records) == 96 * 5
    assert len(modes) == 5
    sample_sets = {
        mode: {item["sample_id"] for item in records if item["mode"] == mode}
        for mode in modes
    }
    assert len({frozenset(values) for values in sample_sets.values()}) == 1
    assert all(item["ground_truth_was_algorithm_input"] is False for item in records)
    summary = json.loads((output / "evaluation_summary.json").read_text())
    paired = summary["paired_primary_hand_occluded_vs_rgb_only"]
    assert paired["learned_rgb_tactile_reliability"]["all"]["paired_case_count"] == 96
    assert set(summary["negative_controls_24"]) == {
        "normal", "zero_tactile", "shuffled_tactile",
        "position_shuffled", "scalar_shuffled",
    }


def test_rgb_only_saved_control_is_invariant_to_every_tactile_perturbation():
    path = (ROOT / "experiments/neural_sdf/learned_v1_20260911"
            / "robustness_learned_rgb_only_records.json")
    records = json.loads(path.read_text())
    by_condition = {}
    for item in records:
        by_condition.setdefault(item["mode"], {})[item["sample_id"]] = item
    reference = by_condition["normal"]
    tactile_only_controls = {
        "sensor_mount_error", "joint_bias_proxy", "tactile_dropout_25pct",
        "tactile_dropout_50pct", "tactile_dropout_75pct",
        "incorrect_contact_geometry",
    }
    for control in tactile_only_controls:
        for sample_id, normal in reference.items():
            changed = by_condition[control][sample_id]
            assert changed["whole_surface_chamfer_l1_m"] == normal["whole_surface_chamfer_l1_m"]
            assert (changed["regions"]["occluded_by_hand"]["mean_point_to_surface_m"] ==
                    normal["regions"]["occluded_by_hand"]["mean_point_to_surface_m"])
