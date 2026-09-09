import numpy as np
import json
from pathlib import Path

from fusion.sensitivity import _intrinsics, _mask_degradation, _mount_translation
from tests.test_geometric_fusion import observation


def test_calibration_and_mount_perturbations_do_not_mutate_baseline():
    baseline = observation()
    original_k = baseline.intrinsic_matrix.copy()
    original_center = baseline.tactile_constraints[0].sensor_center_world_m
    changed_k = _intrinsics(baseline, .02)
    changed_mount = _mount_translation(baseline, .001)
    assert np.array_equal(baseline.intrinsic_matrix, original_k)
    assert changed_k.intrinsic_matrix[0, 0] == original_k[0, 0] * 1.02
    assert baseline.tactile_constraints[0].sensor_center_world_m == original_center
    assert changed_mount.tactile_constraints[0].sensor_center_world_m != original_center


def test_mask_degradation_preserves_visibility_partition():
    baseline = observation()
    changed = _mask_degradation(baseline, .25)
    assert changed.predicted_object_mask.sum() <= baseline.predicted_object_mask.sum()
    assert not np.logical_and(changed.predicted_object_mask, changed.known_background_mask).any()


def test_committed_sensitivity_report_has_all_required_families_and_level_zero():
    path = Path(__file__).parents[1] / "experiments/fusion/sensitivity_v1_20260909/sensitivity_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "sensor_mount_translation_m", "sensor_mount_orientation_deg",
        "joint_encoder_bias_deg", "joint_encoder_noise_std_deg",
        "scalar_calibration_relative_error", "tactile_channel_dropout_fraction",
        "camera_focal_length_relative_error",
        "camera_extrinsic_translation_m_rotation_deg", "rgb_mask_removed_fraction",
        "encoder_tactile_timestamp_mismatch_s",
    }
    assert set(summary) == required
    for levels in summary.values():
        assert len(levels) == 4 and levels[0]["level_index"] == 0
        assert set(levels[0]["methods"]) == {
            "rgb_only", "tactile_only_representative",
            "fusion_representative", "fusion_finite_patch",
        }
