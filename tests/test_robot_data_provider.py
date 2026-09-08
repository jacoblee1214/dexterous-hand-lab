import inspect
import json
from pathlib import Path

import pytest

from robot_data.common import (
    CameraObservation,
    JointInfo,
    JointStateSample,
    ProviderDescription,
    RobotCommand,
    RobotCommandName,
    TactileChannelSample,
    TactileStateSample,
)
from robot_data.hardware_provider import HardwareRobotDataProvider
from robot_data.provider import ProviderNotImplementedError
from robot_data.recording import CommonStateRecorder
from robot_data.replay_provider import ReplayRobotDataProvider
from robot_data.synchronization import TimestampSynchronizer


ROOT = Path(__file__).parents[1]


def tactile(timestamp, value=1.0):
    return TactileStateSample(
        timestamp,
        (TactileChannelSample(timestamp, "sensor_A", value, value * 10, value > 0,
                              force=value, indentation=value / 1000),),
    )


def test_joint_state_is_interpolated_at_tactile_timestamp_deterministically():
    synchronizer = TimestampSynchronizer()
    synchronizer.add_joint_state(JointStateSample(1.0, {"joint_A": 0.0}, {"joint_A": 2.0}))
    synchronizer.add_joint_state(JointStateSample(3.0, {"joint_A": 4.0}, {"joint_A": 6.0}))
    state = synchronizer.synchronize(tactile(2.0), source="test")
    assert state.timestamp == 2.0
    assert state.joint_positions == {"joint_A": 2.0}
    assert state.joint_velocities == {"joint_A": 4.0}
    assert state.tactile_channels[0].timestamp == state.timestamp


def test_camera_extension_is_metadata_only():
    synchronizer = TimestampSynchronizer()
    synchronizer.add_joint_state(JointStateSample(2.0, {"joint_A": 0.0}, {"joint_A": 0.0}))
    camera = CameraObservation(2.0, "wrist_camera", True, True)
    state = synchronizer.synchronize(tactile(2.0), source="test", camera=camera)
    assert state.camera.to_dict() == {
        "timestamp": 2.0, "frame_id": "wrist_camera",
        "rgb_available": True, "depth_available": True,
    }
    assert "rgb" not in json.dumps(state.to_dict()).lower().replace("rgb_available", "")


def test_common_state_recording_replays_named_values_exactly(tmp_path):
    description = ProviderDescription(
        source="simulation", source_display_name="SIMULATION", handedness="right",
        kinematic_model_path="hand.urdf", sensor_config_path="sensors.yaml",
        calibration_config={"sensor_A": {"gain": 1.0}},
        joints=(JointInfo("joint_A", -1.0, 1.0, "palm", "finger"),),
    )
    synchronizer = TimestampSynchronizer()
    path = tmp_path / "run.jsonl"
    with CommonStateRecorder(path, description) as recorder:
        for timestamp, position in ((1.0, 0.1), (2.0, 0.2)):
            synchronizer.add_joint_state(
                JointStateSample(timestamp, {"joint_A": position}, {"joint_A": 0.1})
            )
            recorder.record(synchronizer.synchronize(tactile(timestamp, position), source="simulation"))
    replay = ReplayRobotDataProvider(path)
    assert replay.description.source_display_name == "REPLAY · SIMULATION"
    assert replay.read_common_state().joint_positions["joint_A"] == pytest.approx(0.1)
    replay.step()
    assert replay.read_common_state().joint_positions["joint_A"] == pytest.approx(0.2)
    replay.send_command(RobotCommand(RobotCommandName.RESET))
    assert replay.current_timestamp == 1.0


@pytest.mark.parametrize(
    "method,arguments",
    [
        ("read_joint_state", ()), ("read_tactile_state", ()),
        ("send_joint_targets", ({"joint_A": 0.0},)),
        ("read_common_state", ()), ("step", ()),
    ],
)
def test_hardware_hooks_return_clean_not_implemented(method, arguments):
    provider = HardwareRobotDataProvider()
    with pytest.raises(ProviderNotImplementedError, match="NOT_IMPLEMENTED"):
        getattr(provider, method)(*arguments)


def test_dashboard_source_contains_no_simulator_runtime_arrays():
    source = (ROOT / "backend" / "dashboard_server.py").read_text(encoding="utf-8")
    for forbidden in ("import mujoco", ".qpos", "mjData", "geom_xpos", "contact["):
        assert forbidden not in source
    assert "RobotDataProvider" in source


def test_hardware_interface_exposes_no_invented_transport_protocol():
    source = inspect.getsource(HardwareRobotDataProvider).lower()
    for invented in ("serial", "socket", "canbus", "ros2"):
        assert invented not in source
