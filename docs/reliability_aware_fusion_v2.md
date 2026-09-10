# Research Milestone 3: Reliability-aware geometric fusion v2

## Scope, provenance, and frozen protocol

This milestone remains a known-radius sphere experiment. The radius is supplied
as a prior (`0.040 m`, diameter `0.080 m`) and only the center is estimated. It
does not demonstrate general-object reconstruction or hardware performance.

Work started on branch `research/reliability-aware-fusion-v2` from commit
`1ccb18fbfb572f9b61e915c244b47f70d0b91e8d`. The pre-change regression result
was 165 passed and 3 opt-in browser tests skipped; the three browser tests also
passed separately. The baseline environment was Python 3.12 with MuJoCo 3.3.7,
NumPy 2.3.2, PyYAML 6.0.2, Pillow 11.3.0, websockets 15.0.1, pytest 9.0.2, and
Playwright 1.55.0.

Baseline SHA-256 values were:

| File | SHA-256 |
|---|---|
| `fusion/config.yaml` | `2db4d56ca2127885c5db1b166bc421dfdd82efb105fcb329a1b664ac15591de7` |
| `vision/camera_calibration.yaml` | `aa7b7035696672bab51cfadf490ed8588fc0c3f264b3da6eff66cd6799dd50c7` |
| `sensors/sensor_config.yaml` | `6de2cb3ce5a21e95e1d7c54bf7d15b5ccaad855936c706f213e34f6a0a32cf56` |
| `sensors/pressure_config.yaml` | `e2b5b87bf635d62b3e5879015efb23aba760f637a3376ca80ac32df459533b35` |
| `simulation/grasp_config_right.yaml` | `5cef97f918500167d3ebd51a06b3bd27d701588fca9572da197df474104ab188` |
| `requirements.txt` | `2c371a3fdd1b3e2a70023f7dc8bb555266d56fbc0430310ae10666a9b67e0866` |

The method thresholds and 27-case evaluation matrix were frozen before the run
in `experiments/fusion/reliability_v2_protocol.yaml`. Its SHA-256 is
`09f32f643e6cafc939ccab5beca2d115954c5b917eb1c37706828fe3a2f54d81`.
The evaluated v2 configuration SHA-256 is
`5384e85d0eeb854fd463d3c82f59570d0b26eeea20a1a489fefde75a3ce3da03`.
No threshold was changed after viewing evaluation truth.

## Exact algorithm

The input boundary is `FusionObservation`; it contains clean predicted RGB
masks, the four visibility classes, calibrated rays/transforms and source
timestamps, named synchronized joint states, named scalar tactile measurements,
finite calibrated sensing regions, response/indentation estimates and declared
uncertainties. It contains no MuJoCo contact, object pose, hidden mesh, or
ground-truth mask. Truth is passed only to `fusion/evaluation.py` after an
algorithm result has been serialized.

The four visibility classes are mutually exclusive: predicted foreground,
known background, hand-occluded/unknown, and an exploratory near-threshold
unreliable color band. V2 moves unreliable pixels into unknown evidence; it
never treats them as observed free space. A–D retain their original v1
classification semantics for a fair comparison.

V2 keeps the latest valid observation for each named sensor. This prevents
repeated held state from being counted as independent measurements. Each finite
patch contributes the standardized interval residual already defined by v1:

```text
e_patch_i(c) = distance(r, [min_q ||q-c||, max_q ||q-c||]) / sigma_patch_i
```

where `q` ranges over deterministic samples of the calibrated patch and `r` is
the fixed radius. RGB and tactile residual groups are each divided by the square
root of their sample count. Huber IRLS uses threshold `2.5 sigma`.

The decision sequence is deterministic:

1. Fit a visibility-aware RGB center and test its standardized RMS (`<= 1.0`)
   and Jacobian condition (`<= 1000`).
2. Test tactile coverage: at least four contacts, three sensors, two fingers,
   15 mm spread, minimum representative-point Jacobian singular value `0.08`,
   and condition at most `1000`.
3. At the RGB center, calculate finite-patch cross-modal RMS. The consistency
   threshold is the same Huber scale, `2.5 sigma`.
4. Compute local standardized-residual information matrices `J'J`. Touch is
   complementary when it supplies at least 10% of information in the weakest
   RGB direction.
5. If RGB is reliable and touch is missing, contradictory, or uninformative,
   preserve RGB and report `VALID_RGB_FALLBACK`. If RGB is invalid but finite-
   patch tactile optimization is observable and numerically valid, report
   `VALID_TACTILE_FALLBACK`. If neither is reliable, report
   `INVALID_NO_RELIABLE_MODALITY`.
6. Otherwise solve jointly with tactile weight 1 and RGB weight
   `1 / (1 + max(0, RMS_RGB^2 - 1))`, then report `VALID_FUSED` only after
   numerical checks pass.

The local information eigenvalues, optimizer covariance, and condition numbers
are diagnostics of this local model only. They are not complete real-world
uncertainty or accuracy guarantees. All uncertainty values in v2 are explicitly
labeled exploratory until measured hardware calibration exists.

## Frozen evaluation result

The run contains 27 attempts: three trials across clear RGB/no touch, partial
actual hand occlusion, strong actual hand occlusion, artificial mask damage,
diverse touch, single-finger touch, 50% tactile dropout, exploratory camera
perturbation (2 mm/1 degree), and exploratory sensor-mount perturbation
(2 mm/3 degrees). The observations for A–E are identical within each case.

| Method | Valid / 27 | Success | Valid-only mean / median center error |
|---|---:|---:|---:|
| A RGB-only | 9 | 33.3% | 56.57 / 3.02 mm |
| B tactile representative | 15 | 55.6% | 16.28 / 16.17 mm |
| C v1 representative fusion | 3 | 11.1% | 41.73 / 41.67 mm |
| D v1 finite-patch fusion | 3 | 11.1% | 19.81 / 18.85 mm |
| E reliability-aware v2 | 9 | 33.3% | 9.74 / 3.47 mm |

Those valid-only rows are not a paired comparison. All five methods were valid
on only three common cases, all from the artificial-mask condition. On that
subset, mean errors were A 164.00 mm, B 16.25 mm, C 41.73 mm, D 19.81 mm, and E
23.23 mm. Thus E improved A by 140.77 mm and C by 18.50 mm, but was worse than D
by 3.42 mm. E's explicit RGB fallbacks gave it three times the coverage of C/D,
but the only all-method accuracy comparison is an artificial-mask stress test;
this does not establish better genuine-occlusion reconstruction.

E decisions were 3 genuine fusions, 6 RGB fallbacks, 0 tactile fallbacks, and 18
invalid results. Clear/no-touch and partial-occlusion cases used RGB fallback
(mean 2.52 and 3.47 mm respectively). A's corresponding v1-classified RGB means
were 2.69 and 3.02 mm, so E slightly improved clear but slightly worsened partial
occlusion. In actual strong occlusion, E returned
invalid: reliable silhouette angular coverage fell to 0.194, below the frozen
0.20 gate, while four sensors across three fingers passed the coarse coverage
gate but the finite-patch objective had zero local minimum singular value. It
therefore did not present a locally degenerate tactile solution as valid.

The artificial-mask cases produced `VALID_FUSED`, but RGB–touch disagreement was
very large (36.08–55.33 sigma; mean 44.52) and RGB RMS was just beyond its reliability limit.
Following the preregistered rule, uncertain RGB did not veto tactile. The result
still underperformed v1 finite-patch fusion. This is retained as a failure, not
used to retune the gate.

### Ablations

| Ablation | Valid / 27 | Valid-only mean error | Interpretation |
|---|---:|---:|---|
| Full v2 | 9 | 9.74 mm | Frozen primary method |
| No consistency gate | 9 | 9.74 mm | Gate never determined a decision in this matrix |
| Representative instead of patch | 21 | 12.92 mm | More coverage, but assumes an exact within-pad point |
| No uncertainty normalization | 9 | 54.69 mm | Same coverage and much worse error; units are intentionally mismatched |

The representative ablation's 77.8% success confirms that finite-patch local
degeneracy, not merely a lack of active sensor IDs, is the main v2 coverage
limitation in this run. It is not evidence that the representative assumption is
physically correct.

## Diagnosis: supported findings and open hypotheses

Supported by paired v1 logs and the frozen v2 run:

- V1's repeated historical contacts came from only four unique sensors and
  three fingers in strong occlusion. Counting roughly 137 records improved the
  numerical appearance without creating 137 independent spatial constraints.
- The representative tactile-only estimate has a stable roughly 16.25 mm bias.
  V1 C and D pulled the accurate strong-occlusion RGB result away from truth;
  mean degradation was 3.88 mm for C and 1.71 mm for D.
- Good local conditioning did not imply accuracy: the degraded v1 estimates had
  low condition numbers while remaining biased.
- Finite-patch handling reduced v1's strong-occlusion degradation relative to
  representative points, but its feasible interval creates locally flat regions
  after latest-per-sensor deduplication.
- V2 uncertainty normalization mattered in the artificial-mask cases; removing
  it increased the full-run valid-only mean from 9.74 to 54.69 mm.
- Camera/sensor perturbation cases could not measure incremental v2 error because
  the unperturbed strong-occlusion case was already invalid. Existing v1
  sensitivity results remain the applicable exploratory curves.

Still unresolved without measured calibration or more varied observations:

- how much of the tactile bias comes from sensor mounting, indentation response,
  joint state, or the representative/patch geometry itself;
- whether a continuous probabilistic patch likelihood can retain physical
  honesty without the sampled interval's local plateau;
- whether the near-threshold RGB unreliable band transfers outside the controlled
  blue-background setup;
- whether additional independently located physical sensors would constrain the
  occluded direction enough to produce a valid finite-patch estimate;
- whether the artificial-mask RGB/tactile conflict is dominated by mask geometry,
  calibration, or initialization basin.

An initial run accidentally projected v2 unreliable pixels into A–D and was
invalidated before final reporting. It remains locally under the exact ignored
directory `reliability_v2_20260909_invalid_v1_visibility_projection`; no metric
from it appears above. The corrected run builds A–D from `fusion/config.yaml` and
E from `fusion/config_v2.yaml`, using the same RGB, mask, calibration, joint and
tactile source observations.

Exact per-case inputs, outputs, residuals, contributors, conditioning, runtime,
and evaluation-only truth are under
`experiments/fusion/reliability_v2_20260909/`; aggregated residual, coverage,
conditioning, contributor, and runtime distributions are in its
`diagnostic_summary.json`. PNG inputs are retained locally
and Git-ignored; JSON manifests and results are versioned. The result is a more
conservative and explainable decision system. It improves labeled fallback
coverage over v1 fusion in this dataset, but does **not** prove better finite-
patch accuracy or better reconstruction under genuine strong occlusion.

## Verification and browser evidence

The final EGL regression run reports 177 passed and 3 opt-in browser tests
skipped. The separate real Chromium/HTTP/WebSocket suite reports 3 passed. Its
full-page evidence is retained locally at
`experiments/fusion/reliability_v2_20260909/browser/reliability-v2-dashboard.png`;
it shows the actual MuJoCo STL hand, live active sensor/telemetry/plots, and E's
reliability/fallback diagnostics in the unchanged four-quadrant dashboard. This
is automated browser evidence, not a claim of human visual or collision
acceptance.
