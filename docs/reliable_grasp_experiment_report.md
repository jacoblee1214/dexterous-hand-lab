# Reliable right-hand grasp experiment report

Date: 2026-09-08

This report is an automated MuJoCo result. It is not a substitute for the
required user visual acceptance of thumb opposition, natural closure, and lack
of visible oscillation.

## Automated verification at the time of this earlier experiment

- full suite: `133 passed, 3 skipped`;
- opt-in Chromium/HTTP/WebSocket dashboard suite: `3 passed`;
- live backend/frontend build: `reliable-grasp-20260908.1`;
- live renderer endpoint: `/mujoco-render/stream.mjpg` returned HTTP 200 with
  `multipart/x-mixed-replace` JPEG frames;
- browser evidence: `experiments/reliable_grasp_dashboard_20260908.png`.

This build was subsequently superseded by `teach-grasp-20260908.1`; see
`docs/user_taught_grasp_report.md` for the current dashboard verification.

The three skipped tests are the browser checks when the normal suite is run
without `RUN_BROWSER_TESTS=1`; those same checks were also run explicitly and
passed.

## Fixed-object contact diagnostic — 40 mm sphere

Three trials used `sphere_40mm`, `right-sphere-grasp-v3`, the approved 18 scalar
channels, identical spawn state, and actuator-only staged control. The object
was explicitly held at its known pose, so these results are contact diagnostic
results and not a free-grasp claim.

- completed/settled trials: 3/3;
- active/contributing channels in every trial:
  `Finger04_Link02_Sensor`, `Finger04_Link03_Sensor`,
  `Finger05_Link03_Sensor`;
- contributing fingers: Finger04 and Finger05;
- accepted points: 45–47, mean 45.67;
- settling time including reference opening: 4.858–8.718 s;
- final maximum absolute joint velocity: 0.00765–0.02240 rad/s;
- maximum transient sphere-contact penetration: 1.383–1.392 mm;
- reconstruction status: `VALID_RECONSTRUCTION` in 3/3 trials under the current
  configured fit/coverage criteria.

The exact trial distribution is in
`experiments/grasp_trials/right_40mm_fixed_final_v2_20260908/summary.json`.
Reconstruction and evaluation are stored in separate files for every trial.

## Supported free-object grasp — 40 mm sphere

Three separate trials enabled gravity, a visible physical support, the sphere
free joint, and all MuJoCo contacts. The object was parked only while the hand
returned to reference, then restored before thumb approach. It was not locked
during approach, closure, settling, or hold.

- completed/settled trials: 3/3;
- active channels in every trial:
  `Finger03_Tip_Sensor`, `Finger04_Link02_Sensor`,
  `Finger04_Link03_Sensor`, `Finger05_Link03_Sensor`;
- contributing fingers: Finger03, Finger04, and Finger05;
- accepted points remaining after motion segmentation: 19–22;
- final maximum absolute joint velocity: 0.00650–0.00669 rad/s;
- maximum transient sphere-contact penetration: 0.543–0.548 mm;
- object translation from restored spawn pose: 8.288–8.314 mm;
- motion-reset segments: 4–5 per trial;
- reconstruction status: one `POOR_SPATIAL_COVERAGE`, two
  `INSUFFICIENT_DATA`.

The free grasp therefore did not freely penetrate the sphere in the automated
contact metric and did settle on the support, but it is **not** reported as a
successful free-object reconstruction. Object motion correctly invalidated and
segmented the stationary-world accumulation assumption. The exact distribution
is in `experiments/grasp_trials/right_40mm_free_final_v2_20260908/summary.json`.

## Manual acceptance still required

Launch with:

```bash
python -m backend.dashboard_server --open-browser
```

In `Individual Joint Targets & Grasp Presets`, first select the fixed diagnostic
and execute reference/stages 1–4 while checking the rendered thumb and hand.
Then select the supported free-object mode and repeat. The user must confirm:

1. negative `joint_00`/`joint_01` produce intended thumb opposition;
2. `joint_02`/`joint_03` curl inward rather than hyperextend;
3. closure looks natural and the final hold has no visible oscillation;
4. the free sphere approaches, remains contact-constrained, can be held, and is
   released when `Open Hand` is pressed.

Until that observation is made, manual acceptance remains `PENDING`.
