# Right-hand sphere experiments

`reconstruction_config.yaml` configures pressure/time/spatial acceptance, the
bounded point count, sphere observability, robust loss, and auto-fit throttling.
In the reconstruction viewer, press `U` to fit and `L` to write a
ground-truth-free `right_sphere_run.json` plus a separate
`right_sphere_run.evaluation.json` containing MuJoCo comparison data.

Only sphere center/radius fitting is performed. Cylinder, box, free-form fitting,
and meshing remain outside this milestone.

## Calibrated RGB vision-only experiment

The independent fixed RGB camera and transparent vision-only sphere baseline can
be reproduced with:

```bash
MUJOCO_GL=egl python -m vision.experiment \
  --output experiments/vision_baseline/my_run
```

The command refuses to overwrite an existing run. Clean JPEG observations and
algorithm predictions are separated from evaluation-only MuJoCo masks and pose
errors. See `docs/research_rgb_milestone1.md` for the calibration convention,
method, committed four-condition run, and limitations.
