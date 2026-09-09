from pathlib import Path

from fusion.reanalysis import METHODS, analyze


def test_paired_reanalysis_uses_matching_case_ids_and_all_attempt_success_rates():
    result = analyze()
    assert result["attempt_count"] == 18
    assert result["common_valid_all_methods"]["count"] == 6
    assert len(set(result["common_valid_all_methods"]["case_ids"])) == 6
    assert result["success_over_all_attempts"]["rgb_only"]["success_count"] == 18
    for method in METHODS[1:]:
        assert result["success_over_all_attempts"][method]["success_count"] == 6


def test_strong_occlusion_degradation_is_preserved():
    result = analyze()
    strong = result["paired_by_condition"]["strong_occlusion"]
    assert strong["common_valid_count"] == 3
    assert strong["delta_to_rgb_m"]["fusion_representative"]["minimum"] > 0
    assert strong["delta_to_rgb_m"]["fusion_finite_patch"]["minimum"] > 0
