# Minimal read-only real-hardware validation protocol

This is a collection and validation protocol, not a hardware SDK. No real robot
has been connected or validated, and motor output remains disabled.

## 1. Freeze names, units, and clocks

Map physical inputs explicitly to the approved 18-channel registry; never use
array position as identity:

```text
Finger01_Link02_Sensor  Finger01_Link03_Sensor  Finger01_Tip_Sensor
Finger02_Link02_Sensor  Finger02_Link03_Sensor  Finger02_Tip_Sensor
Finger03_Link02_Sensor  Finger03_Link03_Sensor  Finger03_Tip_Sensor
Finger04_Link02_Sensor  Finger04_Link03_Sensor  Finger04_Tip_Sensor
Finger05_Link02_Sensor  Finger05_Link03_Sensor  Finger05_Tip_Sensor
Palm01_Sensor           Palm02_Sensor           Palm03_Sensor
```

The common interface expects joint position in radians, joint velocity in rad/s,
normalized scalar normal-response value in N, pressure in Pa, and indentation in
m. Those are interface units, not assumptions about raw ADC units. Record each
hardware channel's raw unit, bit depth, sign, offset, gain, saturation, sample
rate, and conversion uncertainty before adapting it. Map all 20 named encoders
to URDF joints with sign, zero, limits, and reference pose.

The required registry-to-link mapping is: Finger01 Link02/Link03/Tip to
`thumb_1/thumb_2/thumb_3`; Finger02 to `index_1/index_2/index_3`; Finger03 to
`middle_1/middle_2/middle_3`; Finger04 to `ring_1/ring_2/ring_3`; Finger05 to
`pinky_1/pinky_2/pinky_3`; and Palm01/02/03 to `Palm`. The physical acquisition
channel or bus address corresponding to each name is still required from the
real system and must be recorded in a versioned mapping before collection.

Joint, tactile scan, and RGB timestamps must be source measurement timestamps in
one documented monotonic time base—not packet reception times. Measure tick
frequency, epoch/reset/wrap, drift, latency, and jitter. Collect encoder samples
on both sides of each tactile timestamp and use the existing interpolation layer;
reject extrapolation, excessive bracket gaps, stale samples, and clock resets.

## 2. Calibrate geometry and response before fusion

For each sensor, estimate the rigid transform from named parent link to sensor:
center, orthonormal U/V/normal axes, finite width/height or fingertip radii, and
the sign of indentation along the normal. Use a traceable fixture or metrology
method, repeat mount/remount trials, and retain version, date, operator, residuals,
and covariance. Do not copy simulated mount values and label them measured.

With a normal loading fixture and reference load cell, acquire loading/unloading
curves across the intended range. Estimate raw offset, gain/nonlinearity,
hysteresis, noise, drift, temperature dependence, activation/release thresholds,
saturation, effective area, and the pressure-to-indentation response. Validate on
held-out loads. A scalar channel must remain labeled scalar; do not synthesize a
3D force vector.

Calibrate the actual RGB camera at its actual resolution using a traceable target.
Store intrinsics, distortion model/coefficients, reprojection residuals, and
version. Calibrate `T_robot_base_from_camera_optical` with named frames and report
direction, uncertainty, and date. Verify synchronization with a visible/tactile
timing event. Do not reuse the MuJoCo calibration for real images.

## 3. Safe fixed-object experiment

1. Mechanically fix a sphere of independently measured radius so its pose cannot
   change; locate it within a separately approved reachable workspace.
2. Establish independent evaluation truth without using the reconstruction path:
   calibrated multi-view metrology, optical tracker, CMM/fixture survey, or another
   stated method. Record pose uncertainty and keep truth files evaluation-only.
3. With robot drives disabled or under an independently approved external safety
   controller, begin read-only subscriptions. Confirm provider transitions
   `DISCONNECTED → CONNECTING → STREAMING`; unplug/stop a stream and confirm
   `STALE/ERROR` rather than retained values appearing live.
4. Record several seconds of no-contact baseline, then externally establish safe,
   fixed-object contacts. The dashboard may observe only; it must not command a
   grasp. Record raw packets, converted common states, calibration versions, RGB,
   and receive-time diagnostics.
5. Check all IDs/units, monotonic timestamps, encoder brackets, dropout, saturation,
   and sensor normal directions before running any reconstruction.
6. Run A–D offline on a frozen log. Keep object truth inaccessible to fusion and
   join evaluation by case ID. Report all-attempt success plus common-valid paired
   errors and calibration/timing uncertainty.
7. Repeat across sessions and mount/remount cycles. Compare against the exploratory
   sensitivity curves, but do not claim sim-to-real validation until measured
   perturbation levels and independent truth support it.

Required artifacts are the raw immutable capture, channel-map table, clock report,
sensor-mount calibration, response calibration, camera calibration, independent
truth with uncertainty, common-state replay log, case manifest, algorithm JSON,
and separate evaluation JSON. Motor commands remain out of scope until the real
protocol, mechanical limits, watchdog, emergency stop, risk review, and explicit
approval are supplied.
