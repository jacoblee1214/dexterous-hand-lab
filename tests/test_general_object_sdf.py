from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from backend.dashboard_server import _load_general_object_dashboard_case
from general_object.evaluation import evaluate_surface
from general_object.observation import ObjectFrameInitialization
from general_object.serialization import observation_from_dict, observation_to_dict
from general_object.sdf import reconstruct_general_object, reconstruct_perfect_contact_oracle
from general_object.synthetic import SHAPES, generate_case, shape_sdf
from vision.calibration import calibration_from_declared_pose, load_camera_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT/"general_object/config_v1.yaml").read_text())
SOLVER = CONFIG["solver"] | CONFIG["reliability"]


@pytest.fixture(scope="module")
def asymmetric_case():
    return generate_case(
        SHAPES[-1], seed=99, yaw_degrees=27, pitch_degrees=-12,
        occlusion_fraction=.48, active_sensor_count=4, grid_resolution=24,
    )


def test_declared_camera_pose_matches_compiled_baseline_calibration():
    actual = calibration_from_declared_pose(load_camera_config())
    stored = json.loads((ROOT/"experiments/fusion/reliability_v2_20260909/camera_calibration.json").read_text())
    intrinsics = stored["intrinsics"]
    expected = [[intrinsics["fx"], 0, intrinsics["cx"]],
                [0, intrinsics["fy"], intrinsics["cy"]], [0, 0, 1]]
    assert np.allclose(actual.intrinsic_matrix, expected)
    assert np.allclose(actual.camera_cv_from_world, stored["T_camera_cv_from_world"])


def test_algorithm_input_round_trip_has_no_shape_or_truth_payload(asymmetric_case):
    observation, _ = asymmetric_case
    raw = observation_to_dict(observation)
    assert set(raw["forbidden_fields_absent"]) == {
        "object_identity", "known_radius", "CAD_mesh", "ground_truth_pose",
        "ground_truth_surface", "ground_truth_contact_normal",
    }
    assert not ({"geometry", "shape_parameters", "ground_truth_world_from_object"} & set(raw))
    restored = observation_from_dict(json.loads(json.dumps(raw)))
    assert restored.input_boundary == observation.input_boundary
    assert len(restored.joint_state) == 20
    assert len(restored.tactile_observations) == 4
    assert all(item.object_from_sensor.shape == (4, 4)
               for item in restored.tactile_observations)
    assert all(np.allclose(item.object_from_sensor[3], [0, 0, 0, 1])
               for item in restored.tactile_observations)


def test_object_frame_rejects_truth_or_unknown_pose_sources():
    with pytest.raises(ValueError, match="declared fixture or measured tracker"):
        ObjectFrameInitialization("object", "world", 1, np.eye(4), np.eye(6), "mujoco_ground_truth", "VALID")


def test_controlled_set_contains_five_distinct_geometries():
    assert {item.geometry for item in SHAPES} == {
        "sphere", "cylinder", "cuboid", "rounded_box", "asymmetric_l"}
    query = np.array([[.04, .04, .04], [.0, .0, .0], [-.04, .02, .01]])
    signatures = {tuple(np.round(shape_sdf(query, item), 5)) for item in SHAPES}
    assert len(signatures) == 5


def test_rgb_unknown_rays_are_not_empty_and_proposed_method_has_no_truth(asymmetric_case):
    observation, _ = asymmetric_case
    result = reconstruct_general_object(observation, "rgb_only", SOLVER)
    assert result.status == "VALID_RECONSTRUCTION"
    assert result.diagnostics["hand_occluded_is_empty_space"] is False
    assert result.diagnostics["unknown_rays_penalized_as_empty"] is False
    assert result.diagnostics["ground_truth_input"] is False
    assert result.diagnostics["known_radius_input"] is False
    assert result.diagnostics["object_identity_input"] is False


def test_scalar_patch_constraints_do_not_claim_exact_contact(asymmetric_case):
    observation, _ = asymmetric_case
    finite = reconstruct_general_object(observation, "rgb_finite_patch_tactile", SOLVER)
    assert finite.diagnostics["tactile_constraint_mode"] == "finite_patch"
    assert all(item["source"] == "minimum_absolute_sdf_within_finite_patch"
               for item in finite.diagnostics["applied_contact_constraints"])


def test_latest_measurement_per_named_sensor_is_effective(asymmetric_case):
    observation, _ = asymmetric_case
    duplicate = replace(observation.tactile_observations[0], scalar_measurement_n=2.0)
    changed = replace(observation, tactile_observations=observation.tactile_observations+(duplicate,))
    result = reconstruct_general_object(changed, "rgb_representative_tactile", SOLVER)
    assert result.diagnostics["active_raw_tactile_count"] == 5
    assert result.diagnostics["effective_latest_sensor_count"] == 4


def test_oracle_is_an_explicit_separate_method(asymmetric_case):
    observation, truth = asymmetric_case
    oracle = np.asarray(truth["oracle_contact_points_object_m"])
    with pytest.raises(TypeError):
        reconstruct_general_object(observation, "rgb_only", SOLVER, oracle)
    with pytest.raises(ValueError, match="separately labeled"):
        reconstruct_general_object(observation, "rgb_perfect_contact_oracle", SOLVER)
    result = reconstruct_perfect_contact_oracle(observation, SOLVER, oracle)
    assert result.diagnostics["ground_truth_input"] is True


def test_evaluation_partitions_visible_occluded_and_other_regions(asymmetric_case):
    observation, truth = asymmetric_case
    result = reconstruct_general_object(observation, "rgb_reliability_tactile", SOLVER)
    metrics = evaluate_surface(result, truth)
    regions = metrics["regions"]
    assert set(regions) == {"visible_to_rgb", "occluded_by_hand", "other_unobserved"}
    assert sum(item["count"] for item in regions.values()) == len(truth["surface_points_object_m"])
    assert metrics["ground_truth_was_algorithm_input"] is False


def test_frozen_dataset_has_disjoint_splits_and_paired_held_out_results():
    root = ROOT/"experiments/general_object/object_sdf_v1_20260910"
    manifest = json.loads((root/"dataset_manifest.json").read_text())
    split = manifest["split_policy"]
    development, validation, test = map(set, (
        split["development"], split["validation"], split["test"]))
    assert not (development & validation or development & test or validation & test)
    assert len(manifest["cases"]) == 40
    zero_touch = json.loads((root/"algorithm_inputs/case_001/observation.json").read_text())
    assert len(zero_touch["joint_state"]) == 20
    assert zero_touch["tactile_observations"] == []
    touch = json.loads((root/"algorithm_inputs/case_004/observation.json").read_text())
    assert len(touch["tactile_observations"]) == 8
    assert all(np.asarray(item["object_from_sensor"]).shape == (4, 4)
               for item in touch["tactile_observations"])
    summary = json.loads((root/"evaluation_summary.json").read_text())
    assert summary["central_result"]["rgb_finite_patch_tactile"]["paired_case_count"] == 16
    assert summary["methods"]["tactile_only"]["valid"] == 30


def test_dashboard_artifact_and_controls_are_integrated():
    artifact = _load_general_object_dashboard_case()
    assert artifact["status"] == "AVAILABLE"
    assert "rgb_reliability_tactile" in artifact["methods"]
    html = (ROOT/"ui/index.html").read_text()
    script = (ROOT/"ui/app.js").read_text()
    assert 'id="reconstruction-family"' in html
    assert 'id="general-object-method"' in html
    assert "drawGeneral" in script
