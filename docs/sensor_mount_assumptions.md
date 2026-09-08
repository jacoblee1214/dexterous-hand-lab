# Provisional tactile-surface mounting assumptions

The sensor definitions in this milestone are STL-informed engineering estimates,
not measured CAD or hardware calibration. The authoritative numerical records are
[`sensor_config.yaml`](../sensors/sensor_config.yaml) for the right hand and
[`sensor_config_left.yaml`](../sensors/sensor_config_left.yaml) for the left hand.
Both files use schema version 3 and contain 18 explicit records per hand.

## Physical interpretation

Each finger has three sensing regions:

1. Two finite rectangular pressure-pad surfaces on Link02 and Link03.
2. One broad curved fingertip region represented by an ellipsoid cap.

The palm additionally has three independently calibrated rectangular pressure
regions. Their initial locations are provisional and intended to be placed with
the sensor-mount calibration UI.

The surfaces are not joint-point markers. Every record stores its parent link,
link-local representative `center_xyz`, unit `surface_u_axis`, unit
`surface_v_axis`, width, height, thickness, and explicit `outward_normal`.
Fingertips additionally store three ellipsoid radii.

The stored right-handed surface frame is `[U V N]`, where:

```text
N = normalize(cross(U, V)) = outward_normal
```

At runtime the world normal is calculated only as:

```text
world_R_parent_link @ outward_normal
```

It is never inferred from a joint axis or a MuJoCo contact normal. The future
positive indentation convention remains:

```text
estimated_contact = surface_point + indentation * sensing_direction
```

Pressure generation is implemented as an isolated scalar adapter. Shape
reconstruction remains intentionally unwired in this milestone.

## Surface coordinates

`SensorMount.surface_point_local(u, v)` provides a link-local point on a region:

- Rectangular pads accept `u, v` independently in `[-1, 1]`. The center is
  `(0, 0)` and the four corners are combinations of `(+/-1, +/-1)`.
- Fingertips accept coordinates in the unit disk, `u^2 + v^2 <= 1`. `(0, 0)` is
  the representative outermost point; other coordinates select different points
  on the outward ellipsoid cap.

This API deliberately prevents a future fingertip contact from being permanently
collapsed to one fixed point.

## Visualization contract

Sensor verification renders rectangular pads as translucent cyan boxes with
bright boundary outlines. Fingertips render as cyan curved ellipsoid regions.
Orange spheres mark representative centers, yellow arrows show outward sensing
directions, labels show IDs, and optional RGB arrows show local frames.

The rendered patches are offset outward by 0.4 mm to avoid mesh z-fighting. That
visual offset does not change the stored mounting transform or FK result.

## Calibration status

Exact per-sensor center, U/V axes, outward normal, dimensions, and tip radii must
be visually checked on both supplied mesh sets. They remain provisional until a
hardware/CAD owner approves them. Use the live editor and save explicitly:

```bash
python -m simulation.mujoco_sim --hand right --sensor-mount-calibration
python -m simulation.mujoco_sim --hand left --sensor-mount-calibration
```

The editor does not modify the STL or URDF. It updates the selected configuration
file only. Regenerate the matching MJCF manually when needed with:

```bash
python -m simulation.model_builder --hand right
python -m simulation.model_builder --hand left
```
