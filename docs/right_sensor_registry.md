# Right-hand sensor registry decision

The runtime and `sensors/sensor_config.yaml` both contain 18 enabled scalar
channels. This layout was manually approved by the user before the sphere-fitting
milestone:

- five fingers × Link02, Link03, and fingertip = 15 channels
- three palm pads = 3 channels
- total = 18 channels

The five Link04 rectangular channels are intentionally absent following the
earlier explicit decision that they were unnecessary. Restoring them while
retaining the three palm pads would produce 23 channels, not 20. The approved
18-channel registry is checked at reconstruction-experiment startup so a future
configuration/runtime mismatch cannot pass silently.
