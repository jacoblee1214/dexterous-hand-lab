"""Paired, case-ID-preserving analysis of the committed 18-case fusion experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = PROJECT_ROOT / "experiments/fusion/geometric_v1_20260909"
METHODS = (
    "rgb_only",
    "tactile_only_representative",
    "fusion_representative",
    "fusion_finite_patch",
)


def _stats(values):
    values = [float(value) for value in values if value is not None]
    return {
        "count": len(values),
        "mean": mean(values) if values else None,
        "median": median(values) if values else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def _load_cases(root: Path) -> list[dict]:
    cases = []
    for metric_path in sorted((root / "evaluation").glob("*/metrics.json")):
        case_id = metric_path.parent.name
        algorithm_path = root / "algorithm" / case_id / "result.json"
        if not algorithm_path.is_file():
            raise FileNotFoundError(f"Missing matching algorithm result: {algorithm_path}")
        cases.append({
            "case_id": case_id,
            "evaluation": json.loads(metric_path.read_text(encoding="utf-8")),
            "algorithm": json.loads(algorithm_path.read_text(encoding="utf-8")),
        })
    if len(cases) != 18:
        raise ValueError(f"Expected 18 matching case IDs, found {len(cases)}")
    return cases


def analyze(root: str | Path = DEFAULT_EXPERIMENT) -> dict:
    root = Path(root)
    cases = _load_cases(root)
    successes = {
        method: [case for case in cases if case["evaluation"]["methods"][method]["success"]]
        for method in METHODS
    }
    common = [
        case for case in cases
        if all(case["evaluation"]["methods"][method]["success"] for method in METHODS)
    ]
    common_rows = []
    for case in common:
        errors = {
            method: case["evaluation"]["methods"][method]["center_error_m"]
            for method in METHODS
        }
        common_rows.append({
            "case_id": case["case_id"],
            "condition": case["evaluation"]["condition"],
            "center_error_m": errors,
            "delta_to_rgb_m": {
                method: errors[method] - errors["rgb_only"] for method in METHODS[1:]
            },
        })

    condition_pairs = {}
    for condition in sorted({case["evaluation"]["condition"] for case in common}):
        subset = [row for row in common_rows if row["condition"] == condition]
        condition_pairs[condition] = {
            "common_valid_count": len(subset),
            "center_error_m": {
                method: _stats(row["center_error_m"][method] for row in subset)
                for method in METHODS
            },
            "delta_to_rgb_m": {
                method: _stats(row["delta_to_rgb_m"][method] for row in subset)
                for method in METHODS[1:]
            },
        }

    diagnosis = {}
    for condition in ("strong_occlusion", "controlled_mask_degradation"):
        subset = [case for case in common if case["evaluation"]["condition"] == condition]
        diagnosis[condition] = {
            "case_count": len(subset),
            "segmentation_iou": _stats(case["evaluation"]["segmentation_iou"] for case in subset),
            "measured_visible_fraction": _stats(
                case["evaluation"]["measured_visible_fraction"] for case in subset
            ),
            "methods": {
                method: {
                    "center_error_m": _stats(
                        case["evaluation"]["methods"][method]["center_error_m"] for case in subset
                    ),
                    "rgb_residual_rms_sigma": _stats(
                        case["algorithm"]["methods"][method]["vision_residual_rms_sigma"]
                        for case in subset
                    ),
                    "tactile_residual_rms_sigma": _stats(
                        case["algorithm"]["methods"][method]["tactile_residual_rms_sigma"]
                        for case in subset
                    ),
                    "condition_number": _stats(
                        case["algorithm"]["methods"][method]["condition_number"] for case in subset
                    ),
                    "initializations": sorted({
                        case["algorithm"]["methods"][method]["initialization"] for case in subset
                    }),
                }
                for method in METHODS
            },
            "tactile_coverage": {
                key: _stats(
                    case["algorithm"]["methods"]["fusion_representative"]["diagnostics"].get(key)
                    for case in subset
                )
                for key in (
                    "contact_count", "unique_sensor_count", "unique_finger_count",
                    "spatial_spread_m", "jacobian_condition",
                )
            },
        }

    return {
        "analysis_version": "paired-explainability-v1",
        "source_experiment": str(root),
        "matching_key": "trial_XX_condition directory/case ID",
        "attempt_count": len(cases),
        "success_over_all_attempts": {
            method: {
                "attempt_count": len(cases),
                "success_count": len(successes[method]),
                "failure_count": len(cases) - len(successes[method]),
                "success_rate": len(successes[method]) / len(cases),
            }
            for method in METHODS
        },
        "common_valid_all_methods": {
            "count": len(common),
            "case_ids": [case["case_id"] for case in common],
            "center_error_m": {
                method: _stats(
                    case["evaluation"]["methods"][method]["center_error_m"] for case in common
                )
                for method in METHODS
            },
            "rows": common_rows,
        },
        "paired_by_condition": condition_pairs,
        "failure_diagnosis_evidence": diagnosis,
        "interpretation": {
            "supported": [
                "All four methods are jointly valid only in the three strong-occlusion and three controlled-mask-degradation cases; common-valid comparisons therefore use exactly those six IDs.",
                "In ordinary strong occlusion, both fusion methods degrade center error versus RGB-only on every paired trial; finite-patch fusion degrades less than representative-point fusion.",
                "Finite-patch fusion has a lower tactile residual than representative fusion in strong occlusion, supporting within-patch contact-location ambiguity as one contributor, but not proving it is the only cause.",
                "The controlled mask degradation lowers RGB mask quality and makes RGB-only much worse; both fusion modes then improve the paired estimate.",
                "The strong-occlusion runs use the same sensor/finger coverage and deterministic reliable-RGB initialization, while their reported condition numbers remain finite; gross rank loss and initialization failure are not supported as the primary cause in those cases.",
            ],
            "unresolved_hypotheses": [
                "The committed 18-case logs do not independently identify sensor-mount, joint-encoder, pressure-response, or camera-calibration error; sensitivity experiments bound these effects but do not validate hardware calibration.",
                "Representative contact error cannot be measured directly without evaluation-only contact truth, which is intentionally excluded from algorithm input.",
                "Equal standardized modality weights may still be suboptimal under occlusion, but changing them from held-out ground-truth error would be evaluation leakage and was not performed.",
                "The experiment does not isolate optimizer basin effects beyond recording the successful initialization labels and finite conditioning.",
            ],
        },
        "warning": "Valid-only means from different method subsets are not used as a paired comparison.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.experiment)
    output = args.output or args.experiment / "paired_reanalysis.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "common_valid": result["common_valid_all_methods"]["count"]}, indent=2))


if __name__ == "__main__":
    main()
