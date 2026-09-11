# General-object object-centric dense-SDF v1

## Scope and representation

This milestone adds a transparent general-surface baseline without changing
sphere methods A–E. The output is a `24 x 24 x 24` float32 signed-distance grid
in `declared_static_object`, bounded by `[-0.065, 0.065] m` on each axis. Its
zero crossings form the predicted surface. The representation has no sphere
radius, primitive class, CAD identity, or learned object-category decoder.

The static experiment assumes a rigid, indexed fixture supplies
`T_world_from_declared_static_object`. This is an input initialization, not a
silently queried MuJoCo pose. The interface permits a future measured pose
tracker with timestamp and covariance, but it rejects `mujoco_ground_truth` as
a source. Temporal fusion is disabled; changing-pose samples are never combined.

## Strict input boundary

`GeneralObjectObservation` contains synchronized:

- clean RGB, calibrated intrinsics and `T_camera_from_object`;
- mutually exclusive foreground, known-background, hand-occluded and
  unreliable masks;
- a frozen ResNet-18 feature map summary and exact weight provenance;
- named scalar tactile observations, joint snapshots, sensing normals,
  representative points, finite patch samples, uncertainty and calibration IDs;
- the declared object frame, timestamp, covariance, grid bounds and resolution.

It cannot contain object identity, radius, CAD, GT pose/surface/contact point or
GT contact normal. Serialized algorithm inputs and evaluation truth live in
separate directories. The experiment reloads the input JSON before calling the
reconstructor and passes truth only to `general_object/evaluation.py` after the
result exists. The explicit perfect-contact oracle is a separate method and its
argument is rejected by every proposed method.

The top-level bundle always carries the full named 20-joint snapshot, including
zero-contact cases. Each active tactile record additionally carries its named
joint subset and an explicit homogeneous `T_object_from_sensor`; transform
direction is serialized beside every pose. This makes the stored input usable
without reconstructing pose from simulator array indices.

## RGB frontend and constraints

The fixed Research RGB Camera calibration is downsampled by four. Focal lengths
are divided by four and principal points follow
`c' = (c + 0.5) / 4 - 0.5`. For object-frame query `x_o`:

```text
x_c = T_camera_from_object [x_o, 1]
u = fx * x_c/x_z + cx
v = fy * y_c/y_z + cy
```

A query projecting to known background is carved as empty. Foreground rays and
hand-occluded/unreliable rays remain possible occupancy within the declared
volume. Occluded pixels are never empty-space penalties. The resulting binary
visual hull is converted to an approximate signed Manhattan distance field.
This single-view silhouette has unresolved depth and is intentionally a weak
baseline, not a learned shape prior.

The modular primary image encoder is frozen torchvision ResNet-18 with ImageNet
weights `resnet18-5c106cde.pth` (SHA-256
`5c106cde386e87d4033832f2996f5493238eda96ccf559d1d62760c4de0613f8`).
Weights are loaded only from an explicit local path and only after the full hash
matches; no network download occurs. A dependency-free RGB/luminance/Sobel
encoder remains available as a fixed analytic ablation. In v1 the encoder is a
modular observation frontend and compute baseline; the transparent geometric
solver uses the masks, not a learned feature-to-shape mapping. Claiming that its
CNN features already improve geometry would therefore be incorrect.

Monocular depth and normal references are optional typed fields but disabled.
MuJoCo depth is forbidden as proposed-method input. Consequently no depth-prior
ablation is reported in this run.

## Tactile constraints and fusion

Only the latest valid measurement per named sensor is effective. An inactive
sensor is not free space. Scalar response is not converted to a 3D force vector,
and the sensing normal comes from calibrated sensor pose rather than GT contact.

For representative constraint `p_i`, the desired residual is `|f(p_i)|`. For a
finite patch `P_i`, v1 selects the sampled point with minimum current
`|f(q)|, q in P_i`; this expresses that the zero level set may occur somewhere
in the region without declaring the center exact. A local update is:

```text
f_new(x) = f_old(x) - w_i f_old(q_i)
           exp(-||x-q_i||^2 / (2 * 0.014^2))
```

The representative and finite-patch baselines use fixed `w_i = 1`. The
reliability-aware mode uses only measured uncertainty and spatial coverage:

```text
w_i = [1 / (1 + (sigma_patch_i / 0.014)^2)] * coverage
coverage = min(Nsensor/6,1) * min(Nfinger/3,1) * min(spread/0.060,1)
```

No GT error controls a weight. This conservative weighting is explicitly
compared with both fixed-weight variants. The tactile-only baseline forms local
12 mm support surfaces around measured representatives; it should not be read as
a complete-object model. The exact-contact oracle is evaluation-only.

## Output, validity, and future hooks

Validity requires a nonempty zero surface with at least 20 extracted crossings.
Diagnostics expose active/effective sensor counts, fingers, spread, coverage,
constraint sources, weights, runtime, grid operations, and whether forbidden
inputs were used. Optional measured object pose, depth and normal prior fields
are already typed, but changing-pose accumulation remains rejected until a
synchronized measured transform exists.
