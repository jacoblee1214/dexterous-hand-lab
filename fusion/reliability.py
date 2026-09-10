"""Transparent reliability-aware known-radius geometric fusion v2.

This module receives only FusionObservation. It has no simulator imports and no
evaluation/object-truth argument.
"""

from __future__ import annotations

import copy
from dataclasses import replace
import math
import time
from typing import Mapping

import numpy as np

from fusion.observation import FusionObservation, TactileContactConstraint
from fusion.sphere import (
    FusionResult,
    _fit_method,
    _tactile_coverage,
    _tactile_init,
    _vision_initial,
    _vision_residual,
    build_vision_constraints,
    finite_patch_tactile_residual,
    representative_tactile_residual,
)


METHOD_NAME = "reliability_aware_fusion_v2"


def effective_tactile_constraints(
    observation: FusionObservation,
) -> tuple[TactileContactConstraint, ...]:
    """Use one latest valid measurement per channel, not held-state duplicates."""
    latest: dict[str, TactileContactConstraint] = {}
    for item in observation.tactile_constraints:
        if not item.activation_valid:
            continue
        previous = latest.get(item.sensor_id)
        if previous is None or item.timestamp >= previous.timestamp:
            latest[item.sensor_id] = item
    return tuple(latest[key] for key in sorted(latest))


def _v2_visibility(observation: FusionObservation) -> FusionObservation:
    unreliable = observation.unreliable_segmentation_mask
    return replace(
        observation,
        unknown_occluded_mask=observation.unknown_occluded_mask | unreliable,
        unreliable_segmentation_mask=np.zeros_like(unreliable),
    )


def _jacobian(residual, center, step: float) -> np.ndarray:
    center = np.asarray(center, dtype=float)
    baseline = residual(center)
    if not len(baseline):
        return np.empty((0, 3))
    columns = []
    for axis in range(3):
        shifted = center.copy(); shifted[axis] += step
        columns.append((residual(shifted) - baseline) / step)
    return np.column_stack(columns)


def _information_diagnostics(observation, vision, contacts, center, config, tactile_mode):
    step = float(config["solver"]["finite_difference_step_m"])
    def vision_group(value):
        residual = _vision_residual(observation, vision, value, config)
        return residual / math.sqrt(max(1, len(residual)))

    def tactile_group(value):
        if tactile_mode == "representative":
            residual = np.asarray([
                representative_tactile_residual(value, item, observation.known_radius_m)
                for item in contacts
            ])
        else:
            residual = np.asarray([
                finite_patch_tactile_residual(value, item, observation.known_radius_m)
                for item in contacts
            ])
        return residual / math.sqrt(max(1, len(residual)))

    j_rgb = _jacobian(vision_group, center, step)
    j_tactile = _jacobian(tactile_group, center, step)
    i_rgb = j_rgb.T @ j_rgb if len(j_rgb) else np.zeros((3, 3))
    i_tactile = j_tactile.T @ j_tactile if len(j_tactile) else np.zeros((3, 3))
    eigenvalues, eigenvectors = np.linalg.eigh(i_rgb)
    weak_direction = eigenvectors[:, 0]
    rgb_weak = float(weak_direction @ i_rgb @ weak_direction)
    tactile_weak = float(weak_direction @ i_tactile @ weak_direction)
    tactile_fraction = tactile_weak / max(1e-18, rgb_weak + tactile_weak)
    combined = i_rgb + i_tactile
    combined_eigenvalues = np.linalg.eigvalsh(combined)
    return {
        "rgb_information_eigenvalues": [float(max(0.0, value)) for value in eigenvalues],
        "tactile_information_eigenvalues": [
            float(max(0.0, value)) for value in np.linalg.eigvalsh(i_tactile)
        ],
        "combined_information_eigenvalues": [
            float(max(0.0, value)) for value in combined_eigenvalues
        ],
        "weak_rgb_direction_world": [float(value) for value in weak_direction],
        "rgb_information_in_weak_direction": rgb_weak,
        "tactile_information_in_weak_rgb_direction": tactile_weak,
        "tactile_information_fraction_in_weak_rgb_direction": tactile_fraction,
        "interpretation": (
            "Local standardized-residual information only; this is not complete "
            "real-world uncertainty or an accuracy guarantee."
        ),
    }


def _fallback(
    source: FusionResult,
    *,
    status: str,
    reason: str,
    diagnostics: Mapping,
    started: float,
    method_name: str,
) -> FusionResult:
    merged = {**dict(source.diagnostics), **dict(diagnostics),
              "fusion_decision": status, "genuinely_fused": False}
    return replace(
        source,
        method=method_name,
        status=status,
        validity_reason=reason,
        diagnostics=merged,
        runtime_ms=(time.perf_counter() - started) * 1000,
    )


def fit_reliability_aware(
    observation: FusionObservation,
    config: Mapping,
    *,
    method_name: str = METHOD_NAME,
    use_cross_modal_gate: bool = True,
    use_finite_patch: bool = True,
    use_uncertainty_normalization: bool = True,
) -> FusionResult:
    """Fit v2 or return an explicitly labeled modality fallback/invalid state."""
    started = time.perf_counter()
    settings = config["reliability_v2"]
    effective = effective_tactile_constraints(observation)
    working = replace(_v2_visibility(observation), tactile_constraints=effective)
    working_config = copy.deepcopy(dict(config))
    working_config["weights"] = dict(config["weights"])
    if not use_uncertainty_normalization:
        # Deliberately unit-mismatched diagnostic ablation; never the primary method.
        working_config["uncertainty"] = dict(config["uncertainty"])
        working_config["uncertainty"]["vision_boundary_sigma_px"] = 1.0
        working = replace(working, tactile_constraints=tuple(
            replace(item, position_sigma_m=1.0, finite_patch_sigma_m=1.0)
            for item in effective
        ))
        effective = working.tactile_constraints

    vision = build_vision_constraints(working, working_config)
    rgb_initial = _vision_initial(working, vision)
    rgb = _fit_method(
        "v2_internal_rgb", working, vision, working_config, rgb_initial,
        "visibility_aware_rgb_boundary_cone", True, None,
    )
    tactile_initial, tactile_rank = _tactile_init(effective)
    tactile_coverage = _tactile_coverage(effective, tactile_initial, working_config)
    tactile_observable = bool(tactile_coverage["observable"])
    rgb_rms = (
        float(rgb.vision_residual_rms_sigma)
        if rgb.vision_residual_rms_sigma is not None else math.inf
    )
    rgb_condition = float(rgb.condition_number) if rgb.condition_number is not None else math.inf
    rgb_reliable = bool(
        rgb.valid
        and rgb_rms <= float(settings["rgb_expected_residual_rms_sigma_max"])
        and rgb_condition <= float(settings["maximum_rgb_jacobian_condition"])
    )
    model = "finite_patch" if use_finite_patch else "representative"
    base_diagnostics = {
        "algorithm_version": "reliability-aware-geometric-fusion-v2",
        "exact_inputs": (
            "clean RGB predicted/visibility masks + K/T/timestamps + named synchronized "
            "joints + named scalar tactile calibration/finite regions + declared radius prior"
        ),
        "raw_tactile_observation_count": len(observation.tactile_constraints),
        "effective_tactile_observation_count": len(effective),
        "held_contact_policy": settings["held_contact_policy"],
        "effective_sensor_ids": [item.sensor_id for item in effective],
        "effective_finger_ids": sorted({item.finger_id for item in effective}),
        "unreliable_segmentation_pixel_count": int(
            observation.unreliable_segmentation_mask.sum()
        ),
        "tactile_observability": tactile_coverage,
        "tactile_algebraic_initialization_rank": tactile_rank,
        "rgb_reliability": {
            "reliable": rgb_reliable,
            "vision_residual_rms_sigma": rgb_rms if math.isfinite(rgb_rms) else None,
            "expected_rms_sigma_max": float(settings["rgb_expected_residual_rms_sigma_max"]),
            "jacobian_condition": rgb.condition_number,
        },
        "tactile_reliability": {
            "observable": tactile_observable,
            "cross_modal_residual_rms_sigma": None,
            "consistent_with_rgb": None,
            "consistency_gate_sigma": float(settings["cross_modal_consistency_huber_sigma"]),
        },
        "uncertainty_normalization_enabled": use_uncertainty_normalization,
        "cross_modal_gate_enabled": use_cross_modal_gate,
        "tactile_model": model,
    }
    if not rgb.valid and not tactile_observable:
        diagnostics = {**base_diagnostics, "fusion_decision": "INVALID_NO_RELIABLE_MODALITY",
                       "genuinely_fused": False}
        return FusionResult(
            method_name, "INVALID_NO_RELIABLE_MODALITY", None,
            observation.known_radius_m, observation.radius_label,
            "no_valid_initialization", None, None, None, 0, None, None, None,
            "Neither RGB visibility nor tactile geometric coverage is sufficient.",
            diagnostics, (time.perf_counter() - started) * 1000,
        )
    if not rgb.valid:
        tactile = _fit_method(
            method_name, working, vision, working_config, tactile_initial,
            f"algebraic_fixed_radius_points_rank_{tactile_rank}", False, model,
        )
        if tactile.valid:
            return replace(
                tactile, status="VALID_TACTILE_FALLBACK",
                validity_reason="RGB is invalid; estimate uses observable tactile geometry only.",
                diagnostics={**dict(tactile.diagnostics), **base_diagnostics,
                             "fusion_decision": "VALID_TACTILE_FALLBACK",
                             "genuinely_fused": False},
                runtime_ms=(time.perf_counter() - started) * 1000,
            )
        return replace(
            tactile, method=method_name, status="INVALID_NO_RELIABLE_MODALITY",
            validity_reason="RGB is invalid and tactile-only optimization is not reliable.",
            diagnostics={**dict(tactile.diagnostics), **base_diagnostics,
                         "fusion_decision": "INVALID_NO_RELIABLE_MODALITY",
                         "genuinely_fused": False},
        )
    if not tactile_observable:
        if not rgb_reliable:
            return replace(
                rgb, method=method_name, status="INVALID_NO_RELIABLE_MODALITY",
                validity_reason=(
                    "Tactile coverage is insufficient and the RGB residual/conditioning "
                    "does not pass the declared reliability checks."
                ),
                diagnostics={**dict(rgb.diagnostics), **base_diagnostics,
                             "fusion_decision": "INVALID_NO_RELIABLE_MODALITY",
                             "genuinely_fused": False},
                runtime_ms=(time.perf_counter() - started) * 1000,
            )
        return _fallback(
            rgb, status="VALID_RGB_FALLBACK",
            reason="Tactile coverage is insufficient; the reliable RGB estimate is preserved.",
            diagnostics=base_diagnostics, started=started, method_name=method_name,
        )

    center = np.asarray(rgb.center_world_m)
    cross_values = np.asarray([
        (finite_patch_tactile_residual(center, item, working.known_radius_m)
         if use_finite_patch else
         representative_tactile_residual(center, item, working.known_radius_m))
        for item in effective
    ])
    cross_rms = float(np.sqrt(np.mean(cross_values * cross_values)))
    information = _information_diagnostics(
        working, vision, effective, center, working_config, model
    )
    complementarity = information["tactile_information_fraction_in_weak_rgb_direction"]
    consistent = cross_rms <= float(settings["cross_modal_consistency_huber_sigma"])
    complementary = complementarity >= float(
        settings["minimum_tactile_information_fraction_in_weak_rgb_direction"]
    )
    reliability = {
        **base_diagnostics,
        **information,
        "rgb_reliability": {
            "reliable": rgb_reliable, "vision_residual_rms_sigma": rgb_rms,
            "expected_rms_sigma_max": float(settings["rgb_expected_residual_rms_sigma_max"]),
            "jacobian_condition": rgb.condition_number,
        },
        "tactile_reliability": {
            "observable": tactile_observable,
            "cross_modal_residual_rms_sigma": cross_rms,
            "consistent_with_rgb": consistent,
            "consistency_gate_sigma": float(settings["cross_modal_consistency_huber_sigma"]),
        },
        "complementary_information": {
            "passes": complementary,
            "tactile_fraction_in_weak_rgb_direction": complementarity,
            "minimum_fraction": float(
                settings["minimum_tactile_information_fraction_in_weak_rgb_direction"]
            ),
        },
    }
    if not complementary:
        return _fallback(
            rgb, status="VALID_RGB_FALLBACK",
            reason="Tactile geometry is observable but adds too little information in the weak RGB direction.",
            diagnostics=reliability, started=started, method_name=method_name,
        )
    if use_cross_modal_gate and not consistent and rgb_reliable:
        return _fallback(
            rgb, status="VALID_RGB_FALLBACK",
            reason=("Tactile geometry contradicts a locally reliable RGB estimate beyond the "
                    "declared uncertainty; RGB is preserved without claiming fusion."),
            diagnostics=reliability, started=started, method_name=method_name,
        )

    # Reliability derives only from standardized residuals and local information.
    # Disagreement never suppresses tactile data when RGB itself is uncertain.
    rgb_weight = 1.0 / (1.0 + max(0.0, rgb_rms * rgb_rms - 1.0))
    tactile_weight = 1.0
    working_config["weights"] = {"vision": rgb_weight, "tactile": tactile_weight}
    fused = _fit_method(
        method_name, working, vision, working_config, center,
        "reliable_v2_rgb_center", True, model,
    )
    reliability["applied_modality_weights"] = {
        "vision": rgb_weight, "tactile": tactile_weight,
        "source": "standardized residual reliability; no evaluation truth",
    }
    if fused.valid:
        return replace(
            fused, status="VALID_FUSED",
            validity_reason=("RGB and effective tactile geometry are sufficiently observable and "
                             "complementary; reliability-aware fusion was applied."),
            diagnostics={**dict(fused.diagnostics), **reliability,
                         "fusion_decision": "VALID_FUSED", "genuinely_fused": True},
            runtime_ms=(time.perf_counter() - started) * 1000,
        )
    if rgb_reliable:
        return _fallback(
            rgb, status="VALID_RGB_FALLBACK",
            reason="The fused optimization failed numerical checks; reliable RGB is preserved.",
            diagnostics=reliability, started=started, method_name=method_name,
        )
    return replace(
        fused, method=method_name, status="INVALID_UNRELIABLE_FUSION",
        validity_reason="Fused optimization failed and RGB is not reliable enough for fallback.",
        diagnostics={**dict(fused.diagnostics), **reliability,
                     "fusion_decision": "INVALID_UNRELIABLE_FUSION",
                     "genuinely_fused": False},
        runtime_ms=(time.perf_counter() - started) * 1000,
    )


def run_v2_ablations(observation: FusionObservation, config: Mapping) -> dict:
    definitions = {
        "full_v2": {},
        "no_cross_modal_gate": {"use_cross_modal_gate": False},
        "representative_instead_of_finite_patch": {"use_finite_patch": False},
        "no_uncertainty_normalization": {"use_uncertainty_normalization": False},
    }
    return {
        name: fit_reliability_aware(
            observation, config, method_name=f"v2_ablation_{name}", **options
        ).to_dict()
        for name, options in definitions.items()
    }
