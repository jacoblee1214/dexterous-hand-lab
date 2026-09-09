"""Boundary, residual, observability, and determinism tests for fusion v1."""

from dataclasses import fields, replace
import json

import numpy as np
import pytest

from fusion.config import load_fusion_config
from fusion.observation import FusionObservation, TactileContactConstraint
from fusion.sphere import (
    finite_patch_tactile_residual,
    representative_tactile_residual,
    run_fusion_methods,
)


def constraint(point, index, *, timestamp=1.0, width=.006):
    return TactileContactConstraint(
        timestamp=timestamp,
        sensor_id=f"sensor_{index}",
        finger_id=f"finger_{index % 3}",
        parent_link=f"link_{index}",
        joint_state_snapshot={"joint": .1},
        sensor_type="rectangular",
        sensor_center_world_m=point,
        surface_u_world=(1, 0, 0),
        surface_v_world=(0, 1, 0),
        sensing_direction_world=(0, 0, 1),
        representative_contact_world_m=point,
        surface_width_m=width,
        surface_height_m=width,
        fingertip_radii_m=None,
        scalar_measurement_n=1.0,
        indentation_m=0.0,
        position_sigma_m=.001,
        finite_patch_sigma_m=.001,
        response_calibration={"minimum_activation_force_n": .01,
            "effective_sensor_area_m2": 1e-4, "virtual_stiffness_n_per_m": 1000,
            "saturation_pressure_pa": 1e6, "contact_projection_sign": 1},
    )


def observation(*, contacts=True):
    width, height, focal = 100, 80, 160.0
    K = np.array([[focal, 0, (width-1)/2], [0, focal, (height-1)/2], [0, 0, 1.]])
    center, radius = np.array([0., 0., .5]), .04
    rows, columns = np.indices((height, width), dtype=float)
    rays = np.stack(((columns-K[0,2])/focal, (rows-K[1,2])/focal,
                     np.ones_like(rows)), axis=-1)
    dot = rays @ center
    mask = (dot > 0) & (dot*dot-np.sum(rays*rays, axis=2)*(center@center-radius*radius) >= 0)
    directions = np.asarray([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1],
    ], dtype=float)
    tactile = tuple(constraint(center+radius*direction, i) for i, direction in enumerate(directions)) if contacts else ()
    return FusionObservation(
        timestamp=1.0, camera_timestamp=1.0, tactile_timestamp=1.0,
        frame_id="camera", camera_id="camera", common_frame_id="world",
        rgb_reference="frame.jpg",
        image_width=width, image_height=height,
        intrinsic_matrix=K, camera_cv_from_world=np.eye(4),
        predicted_object_mask=mask, known_background_mask=~mask,
        unknown_occluded_mask=np.zeros_like(mask), segmentation_status="SEGMENTATION_OK",
        segmentation_confidence=1.0, segmentation_angular_coverage=1.0,
        joint_state={"joint": .1}, tactile_constraints=tactile,
        known_radius_m=radius, radius_label="declared_known_radius_prior",
        calibration_version="test", time_base="simulation_time",
        object_motion_assumption="fixed_pose",
    )


def test_configuration_names_40_mm_as_radius_and_80_mm_as_diameter():
    config = load_fusion_config()
    assert config["sphere_experiment"]["radius_m"] == pytest.approx(.04)
    assert config["sphere_experiment"]["diameter_m"] == pytest.approx(.08)
    assert config["sphere_experiment"]["radius_role"] == "declared_known_radius_prior"


def test_fusion_observation_boundary_has_no_ground_truth_or_simulator_geometry():
    names = {field.name for field in fields(FusionObservation)}
    forbidden = {"ground_truth_center", "true_contact", "contact_normal", "object_pose", "object_mesh"}
    assert names.isdisjoint(forbidden)
    payload = {"fields": sorted(names), "radius_label": observation().radius_label}
    serialized = json.dumps(payload).lower()
    assert "ground_truth" not in serialized and "mujoco" not in serialized


def test_unknown_pixels_are_not_known_background_or_negative_evidence():
    value = observation()
    unknown = value.unknown_occluded_mask.copy()
    unknown[5:15, 5:15] = True
    background = value.known_background_mask.copy()
    background[unknown] = False
    changed = replace(value, unknown_occluded_mask=unknown, known_background_mask=background)
    assert not np.any(changed.unknown_occluded_mask & changed.known_background_mask)
    assert not np.any(changed.unknown_occluded_mask & changed.predicted_object_mask)


def test_representative_and_finite_patch_are_distinct_measurement_models():
    item = constraint((.043, 0, 0), 0, width=.010)
    center = np.zeros(3)
    assert representative_tactile_residual(center, item, .04) == pytest.approx(3.0)
    assert finite_patch_tactile_residual(center, item, .04) == pytest.approx(0.0)


def test_tactile_residual_is_normalized_by_declared_position_uncertainty():
    center = np.zeros(3)
    precise = constraint((.05, 0, 0), 0)
    uncertain = replace(precise, position_sigma_m=.01)
    assert representative_tactile_residual(center, precise, .04) == pytest.approx(10.0)
    assert representative_tactile_residual(center, uncertain, .04) == pytest.approx(1.0)


def test_synthetic_fixed_radius_center_is_recovered_without_truth_argument():
    value = observation()
    rgb = {"sphere": {"estimated_center_world_m": [0.0, 0.0, .5]}}
    result = run_fusion_methods(value, rgb)
    for name in ("rgb_only", "tactile_only_representative", "fusion_representative"):
        method = result["methods"][name]
        assert method["status"] == "VALID_ESTIMATE", (name, method)
        np.testing.assert_allclose(method["estimated_center_world_m"], [0, 0, .5], atol=4e-3)
    assert result["known_radius_m"] == .04 and result["known_diameter_m"] == .08


def test_degenerate_or_sparse_tactile_data_fails_explicitly():
    value = observation(contacts=False)
    sparse = tuple(constraint((x, 0, .5), i) for i, x in enumerate((-.04, 0, .04)))
    value = replace(value, tactile_constraints=sparse)
    result = run_fusion_methods(value, {"sphere": {"estimated_center_world_m": [0, 0, .5]}})
    assert result["methods"]["tactile_only_representative"]["status"] == "INSUFFICIENT_TACTILE_COVERAGE"
    assert result["methods"]["fusion_representative"]["status"] == "INSUFFICIENT_TACTILE_COVERAGE"


def test_solver_is_deterministic_for_identical_observation():
    value = observation()
    rgb = {"sphere": {"estimated_center_world_m": [.002, -.001, .502]}}
    first, second = run_fusion_methods(value, rgb), run_fusion_methods(value, rgb)
    for name in first["methods"]:
        left, right = first["methods"][name], second["methods"][name]
        assert left["status"] == right["status"]
        assert left["estimated_center_world_m"] == right["estimated_center_world_m"]
        assert left["joint_objective"] == right["joint_objective"]


def test_world_camera_coordinate_change_translates_estimate_consistently():
    value = observation()
    translation = np.array([.1, -.2, .3])
    moved_contacts = tuple(replace(item,
        sensor_center_world_m=tuple(np.asarray(item.sensor_center_world_m)+translation),
        representative_contact_world_m=tuple(np.asarray(item.representative_contact_world_m)+translation))
        for item in value.tactile_constraints)
    camera_from_world = np.eye(4); camera_from_world[:3, 3] = -translation
    moved = replace(value, camera_cv_from_world=camera_from_world,
                    tactile_constraints=moved_contacts)
    base = run_fusion_methods(value, {"sphere": {"estimated_center_world_m": [0, 0, .5]}})
    shifted = run_fusion_methods(moved, {"sphere": {
        "estimated_center_world_m": (np.array([0, 0, .5])+translation).tolist()}})
    for name in ("rgb_only", "tactile_only_representative", "fusion_representative"):
        np.testing.assert_allclose(
            shifted["methods"][name]["estimated_center_world_m"],
            np.asarray(base["methods"][name]["estimated_center_world_m"])+translation,
            atol=1e-8,
        )


def test_group_normalization_is_invariant_to_duplicate_tactile_samples():
    value = observation()
    rgb = {"sphere": {"estimated_center_world_m": [.002, 0, .502]}}
    doubled = replace(value, tactile_constraints=value.tactile_constraints*2)
    first, second = run_fusion_methods(value, rgb), run_fusion_methods(doubled, rgb)
    np.testing.assert_allclose(
        first["methods"]["fusion_representative"]["estimated_center_world_m"],
        second["methods"]["fusion_representative"]["estimated_center_world_m"], atol=1e-10)


def test_replay_metadata_and_later_timestamps_do_not_change_geometry_result():
    value = observation()
    replay = replace(value, timestamp=2.0, camera_timestamp=2.0, tactile_timestamp=2.0,
                     rgb_reference="archive/frames/rgb.png",
                     tactile_constraints=tuple(replace(item, timestamp=2.0)
                                               for item in value.tactile_constraints))
    rgb = {"sphere": {"estimated_center_world_m": [0, 0, .5]}}
    live, restored = run_fusion_methods(value, rgb), run_fusion_methods(replay, rgb)
    for name in live["methods"]:
        assert live["methods"][name]["status"] == restored["methods"][name]["status"]
        assert live["methods"][name]["estimated_center_world_m"] == restored["methods"][name]["estimated_center_world_m"]


def test_historical_tactile_is_allowed_only_under_fixed_pose_assumption():
    value = observation()
    old = replace(value.tactile_constraints[0], timestamp=.5)
    fixed = replace(value, tactile_constraints=(old,)+value.tactile_constraints[1:])
    assert fixed.tactile_constraints[0].timestamp == .5
    with pytest.raises(ValueError, match="instantaneous"):
        replace(fixed, object_motion_assumption="instantaneous_only", tactile_timestamp=.5)
