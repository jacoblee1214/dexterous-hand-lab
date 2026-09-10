from dataclasses import fields, replace
import inspect

import numpy as np

from fusion.config import DEFAULT_CONFIG, V2_CONFIG, load_fusion_config
from fusion.observation import FusionObservation, classify_visibility
from fusion.reliability import (
    effective_tactile_constraints,
    fit_reliability_aware,
    run_v2_ablations,
)
from fusion.sphere import run_fusion_methods
from tests.test_geometric_fusion import observation


CONFIG = load_fusion_config(V2_CONFIG)


def test_v2_full_method_is_deterministic_and_known_radius_is_not_estimated():
    value = observation()
    first = fit_reliability_aware(value, CONFIG).to_dict()
    second = fit_reliability_aware(value, CONFIG).to_dict()
    assert first["status"] == "VALID_FUSED"
    assert first["estimated_center_world_m"] == second["estimated_center_world_m"]
    assert first["radius_m"] == .04 and first["diameter_m"] == .08
    assert first["radius_label"] == "declared_known_radius_prior"


def test_held_contact_duplicates_collapse_to_latest_named_sensor_measurement():
    value = observation()
    older = tuple(replace(item, timestamp=.5) for item in value.tactile_constraints)
    repeated = replace(value, tactile_constraints=older + value.tactile_constraints)
    effective = effective_tactile_constraints(repeated)
    assert len(effective) == len(value.tactile_constraints)
    assert all(item.timestamp == 1.0 for item in effective)
    result = fit_reliability_aware(repeated, CONFIG).to_dict()
    assert result["diagnostics"]["raw_tactile_observation_count"] == 12
    assert result["diagnostics"]["effective_tactile_observation_count"] == 6


def test_insufficient_touch_preserves_reliable_rgb_with_explicit_fallback():
    result = fit_reliability_aware(observation(contacts=False), CONFIG).to_dict()
    assert result["status"] == "VALID_RGB_FALLBACK"
    assert result["valid"] and not result["diagnostics"]["genuinely_fused"]
    assert "insufficient" in result["validity_reason"].lower()


def test_contradictory_touch_does_not_force_fusion_when_rgb_is_reliable():
    value = observation()
    shift = np.array([.08, 0, 0])
    contacts = tuple(replace(
        item,
        sensor_center_world_m=tuple(np.asarray(item.sensor_center_world_m) + shift),
        representative_contact_world_m=tuple(
            np.asarray(item.representative_contact_world_m) + shift
        ),
    ) for item in value.tactile_constraints)
    result = fit_reliability_aware(
        replace(value, tactile_constraints=contacts), CONFIG
    ).to_dict()
    assert result["status"] == "VALID_RGB_FALLBACK"
    assert result["diagnostics"]["tactile_reliability"]["consistent_with_rgb"] is False
    assert "contradicts" in result["validity_reason"]


def test_invalid_rgb_can_use_observable_tactile_without_rgb_rejection():
    value = observation()
    empty = np.zeros_like(value.predicted_object_mask)
    # Move each patch radially outside the exact-radius zero-residual plateau.
    # This gives the finite-patch tactile model a numerically observable local
    # solution without pretending the patch center is the contact point.
    object_center = np.array([0.0, 0.0, 0.5])
    contacts = []
    for item in value.tactile_constraints:
        point = np.asarray(item.sensor_center_world_m)
        point = point + 0.004 * (point-object_center) / np.linalg.norm(point-object_center)
        contacts.append(replace(
            item,
            sensor_center_world_m=tuple(point),
            representative_contact_world_m=tuple(point),
        ))
    changed = replace(
        value, predicted_object_mask=empty, known_background_mask=~empty,
        segmentation_status="SEGMENTATION_UNRELIABLE",
        tactile_constraints=tuple(contacts),
    )
    result = fit_reliability_aware(changed, CONFIG).to_dict()
    assert result["status"] == "VALID_TACTILE_FALLBACK"
    assert result["valid"] and not result["diagnostics"]["genuinely_fused"]


def test_visibility_has_four_disjoint_classes_and_unreliable_is_not_background():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[2:4, 2:4] = [5, 20, 44]  # just below the configured blue threshold
    foreground = np.zeros((8, 8), dtype=bool)
    background, unknown, unreliable = classify_visibility(
        rgb, foreground, CONFIG["rgb_visibility"]
    )
    assert unreliable[2:4, 2:4].all()
    assert not background[unreliable].any()
    assert not unknown[unreliable].any()


def test_uncertainty_cross_modal_and_patch_ablations_are_explicit():
    result = run_v2_ablations(observation(), CONFIG)
    assert set(result) == {
        "full_v2", "no_cross_modal_gate",
        "representative_instead_of_finite_patch", "no_uncertainty_normalization",
    }
    assert result["full_v2"]["diagnostics"]["uncertainty_normalization_enabled"]
    assert not result["no_cross_modal_gate"]["diagnostics"]["cross_modal_gate_enabled"]
    assert result["representative_instead_of_finite_patch"]["diagnostics"]["tactile_model"] == "representative"
    assert not result["no_uncertainty_normalization"]["diagnostics"]["uncertainty_normalization_enabled"]


def test_calibration_perturbation_changes_geometry_but_not_input_boundary():
    value = observation()
    transform = value.camera_cv_from_world.copy(); transform[0, 3] = .01
    changed = replace(value, camera_cv_from_world=transform)
    first = fit_reliability_aware(value, CONFIG).to_dict()
    second = fit_reliability_aware(changed, CONFIG).to_dict()
    assert first["estimated_center_world_m"] != second["estimated_center_world_m"]
    assert second["diagnostics"]["exact_inputs"].startswith("clean RGB")


def test_replay_timestamp_change_preserves_v2_geometry_and_status():
    value = observation()
    replay = replace(
        value, timestamp=2.0, camera_timestamp=2.0, tactile_timestamp=2.0,
        tactile_constraints=tuple(replace(item, timestamp=2.0)
                                  for item in value.tactile_constraints),
    )
    live = fit_reliability_aware(value, CONFIG).to_dict()
    restored = fit_reliability_aware(replay, CONFIG).to_dict()
    assert live["status"] == restored["status"]
    np.testing.assert_allclose(
        live["estimated_center_world_m"], restored["estimated_center_world_m"], atol=1e-12
    )


def test_algorithm_has_no_ground_truth_argument_or_simulator_import():
    names = {field.name for field in fields(FusionObservation)}
    assert not any("ground_truth" in name or "mujoco" in name for name in names)
    source = inspect.getsource(inspect.getmodule(fit_reliability_aware)).lower()
    assert "import mujoco" not in source
    assert "ground_truth" not in source


def test_run_fusion_methods_adds_e_only_for_v2_configuration():
    v2 = run_fusion_methods(observation(), {"sphere": {}}, CONFIG)
    v1 = run_fusion_methods(observation(), {"sphere": {}})
    assert set(v2["methods"]) == set(v1["methods"]) | {"reliability_aware_fusion_v2"}


def test_explicit_v1_observation_preserves_a_through_d_exactly():
    value = observation()
    v1_config = load_fusion_config(DEFAULT_CONFIG)
    baseline = run_fusion_methods(value, {"sphere": {}}, v1_config)
    v2 = run_fusion_methods(
        replace(
            value,
            unreliable_segmentation_mask=np.zeros_like(value.predicted_object_mask),
        ),
        {"sphere": {}}, CONFIG,
        v1_observation=value, v1_config=v1_config,
    )
    for method in baseline["methods"]:
        expected = {
            key: item for key, item in baseline["methods"][method].items()
            if key != "runtime_ms"
        }
        actual = {
            key: item for key, item in v2["methods"][method].items()
            if key != "runtime_ms"
        }
        assert actual == expected
