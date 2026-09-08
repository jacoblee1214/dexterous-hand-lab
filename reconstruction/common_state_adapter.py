"""Convert synchronized provider-neutral state into reconstruction input."""

from __future__ import annotations

from robot_data.common import CommonRobotState, ProviderDescription
from reconstruction.reconstruction_input import ReconstructionInput


def reconstruction_input_from_common_state(
    state: CommonRobotState,
    description: ProviderDescription,
) -> ReconstructionInput:
    """Build the only object crossing into the reconstruction core.

    The joint values have already been interpolated to the tactile timestamp by
    the provider synchronization layer. No simulator object or index is visible.
    """
    sensor_values = {
        channel.sensor_id: channel.scalar_value
        for channel in state.tactile_channels
    }
    if set(sensor_values) != set(description.calibration_config):
        raise ValueError("Common-state sensor IDs do not match the calibrated registry")
    if set(state.joint_positions) != {joint.name for joint in description.joints}:
        raise ValueError("Common-state named joints do not match the kinematic model")
    return ReconstructionInput(
        timestamp=state.timestamp,
        joint_state=state.joint_positions,
        sensor_values=sensor_values,
        sensor_config_path=description.sensor_config_path,
        calibration_config=description.calibration_config,
    )
