"""Data-level acceptance checks; these do not assert manual visual approval."""

from dataclasses import replace
import json
import math

import numpy as np
import pytest

from backend.dashboard_server import DashboardSimulation, load_dashboard_config
from reconstruction.common_state_adapter import reconstruction_input_from_common_state
from robot_data.common import CameraObservation, CommonRobotState, JointStateSample, RobotCommand, TactileChannelSample, TactileStateSample
from robot_data.hardware_provider import HardwareRobotDataProvider, HardwareDataUnavailableError, HardwareReadOnlyError
from robot_data.recording import read_common_state_log
from robot_data.replay_provider import ReplayRobotDataProvider
from robot_data.synchronization import TimestampSynchronizer, SynchronizationUnavailableError


def command(engine, name, **parameters):
    return engine.handle_command({"schema_version": "1.1.0", "command": name, "parameters": parameters})


class Clock:
    now = 100.1

    def __call__(self):
        return self.now


def hardware():
    clock = Clock()
    provider = HardwareRobotDataProvider(source_time_now=clock, time_base="test_device_seconds",
                                         monotonic=clock, maximum_interpolation_gap_seconds=.2)
    provider.begin_connection()
    return provider, clock


def joints(provider, timestamp, angle=0.0):
    names = [joint.name for joint in provider.description.joints]
    return JointStateSample(timestamp, dict.fromkeys(names, angle), dict.fromkeys(names, .1))


def tactile(provider, timestamp):
    return TactileStateSample(timestamp, tuple(
        TactileChannelSample(timestamp, sensor_id, 0.2, 100.0, True)
        for sensor_id in provider.description.calibration_config
    ))


def stream(provider):
    provider.accept_joint_state(joints(provider, 100.0))
    provider.accept_tactile_state(tactile(provider, 100.05))
    assert provider.connection_status["state"] == "CONNECTING"
    provider.accept_joint_state(joints(provider, 100.1, .2))


def test_hardware_read_only_data_reaches_all_quadrants_and_interpolates():
    provider, clock = hardware()
    engine = DashboardSimulation(provider=provider)
    assert engine.state()["snapshot"]["timestamp"] is None
    stream(provider)
    engine.step()
    engine._sample_history()
    payload = engine.state()
    assert payload["connection"]["state"] == "STREAMING"
    assert payload["source"]["display_name"] == "REAL ROBOT"
    assert not payload["available_commands"]
    assert len(payload["sensors"]) == 18
    assert len(payload["joints"]) == 20
    assert len(payload["current_estimated_contacts"]) == 18
    assert len(payload["hand"]["links"]) > 20
    assert payload["simulation"]["object"] == {}
    assert payload["evaluation"] == {"enabled": False}
    assert all(joint["angle_rad"] == pytest.approx(.1) for joint in payload["joints"])
    assert all(item["timestamp"] == payload["snapshot"]["timestamp"] for item in payload["sensors"])
    assert payload["snapshot"]["synchronization"]["joint_before_timestamp"] == 100.0
    assert payload["snapshot"]["synchronization"]["joint_after_timestamp"] == 100.1
    assert payload["time_series"]["series"][0]["samples"][-1]["time_seconds"] == 100.05
    json.dumps(payload, allow_nan=False)


def test_stale_hardware_freezes_history_and_accumulation_and_recovers():
    provider, clock = hardware()
    stream(provider)
    engine = DashboardSimulation(provider=provider)
    command(engine, "start_accumulation")
    engine._sample_history()
    before = engine.state()
    clock.now += .6
    engine.step()
    engine._sample_history()
    stale = engine.state()
    assert stale["connection"]["state"] == "STALE"
    assert not stale["snapshot"]["valid"]
    assert stale["time_series"] == before["time_series"]
    assert stale["accumulation"] == before["accumulation"]
    with pytest.raises(HardwareDataUnavailableError):
        provider.read_common_state()
    provider.accept_joint_state(joints(provider, clock.now))
    provider.accept_tactile_state(tactile(provider, clock.now))
    engine.step()
    assert engine.state()["snapshot"]["valid"]
    assert engine.state()["accumulation"]["accepted_point_count"] > 0
    provider.disconnect()
    assert engine.state()["connection"]["state"] == "DISCONNECTED"
    assert not engine.state()["snapshot"]["valid"]


@pytest.mark.parametrize("name", ["open_hand", "grasp", "reset", "set_joint_target", "set_grasp_preset"])
def test_hardware_rejects_motor_commands_at_both_boundaries(name):
    provider, _ = hardware()
    stream(provider)
    engine = DashboardSimulation(provider=provider)
    before = provider.read_common_state().to_dict()
    with pytest.raises(HardwareReadOnlyError, match="READ_ONLY"):
        provider.send_command(RobotCommand(name))
    with pytest.raises(HardwareReadOnlyError):
        provider.send_joint_targets({"joint_10": .1})
    with pytest.raises(ValueError, match="READ_ONLY"):
        command(engine, name)
    assert provider.read_common_state().to_dict() == before


@pytest.mark.parametrize("failure", ["missing_joint", "missing_sensor", "future", "old", "duplicate", "gap"])
def test_invalid_hardware_measurements_are_rejected_and_flagged(failure):
    provider, _ = hardware()
    stream(provider)
    with pytest.raises(ValueError):
        if failure == "missing_joint":
            provider.accept_joint_state(JointStateSample(100.1, {}, {}))
        elif failure == "missing_sensor":
            provider.accept_tactile_state(TactileStateSample(100.1, ()))
        elif failure == "future":
            provider.accept_joint_state(joints(provider, 200.0))
        elif failure == "old":
            provider.accept_tactile_state(tactile(provider, 1.0))
        elif failure == "duplicate":
            provider.accept_tactile_state(tactile(provider, 100.05))
        else:
            provider.begin_connection()
            provider.accept_joint_state(joints(provider, 99.7))
            provider.accept_joint_state(joints(provider, 100.1))
            provider.accept_tactile_state(tactile(provider, 100.0))
    assert provider.connection_status["state"] == "ERROR"


def test_receive_clock_is_not_substituted_for_measurement_clock():
    provider = HardwareRobotDataProvider()
    with pytest.raises(ValueError, match="time base"):
        provider.begin_connection()
    source_clock, receive_clock = Clock(), Clock()
    receive_clock.now = 9000.0
    provider = HardwareRobotDataProvider(source_time_now=source_clock, monotonic=receive_clock,
                                         time_base="device_ticks_converted_to_seconds", maximum_interpolation_gap_seconds=.2)
    provider.begin_connection()
    stream(provider)
    assert provider.read_common_state().timestamp == 100.05
    receive_clock.now += 1.0
    assert provider.connection_status["state"] == "STALE"


def test_epoch_timestamps_are_not_relatively_equal():
    with pytest.raises(ValueError, match="timestamp"):
        TactileStateSample(1_800_000_000.0, (TactileChannelSample(1_800_000_000.1, "a", 0, 0, False),))


@pytest.mark.parametrize("query", [float("nan"), float("inf"), -1.0, .5, 3.0])
def test_interpolation_rejects_invalid_query_and_extrapolation(query):
    sync = TimestampSynchronizer()
    sync.add_joint_state(JointStateSample(1.0, {"j": 0}, {"j": 0}))
    sync.add_joint_state(JointStateSample(2.0, {"j": 1}, {"j": 1}))
    with pytest.raises((ValueError, SynchronizationUnavailableError)):
        sync.joint_state_at(query)


def test_camera_reference_requires_calibration_and_never_creates_depth():
    with pytest.raises(ValueError, match="explicit"):
        CameraObservation(1, "camera", True, False, rgb_reference="frames/rgb.png")
    camera = CameraObservation(1, "camera", True, False, rgb_reference="frames/rgb.png",
                               intrinsics={"width": 640, "height": 480, "fx": 400, "fy": 400,
                                           "cx": 320, "cy": 240, "distortion_model": "none"},
                               extrinsics_reference="calibration/camera_to_palm.yaml", time_base="device_seconds")
    restored = CameraObservation.from_dict(camera.to_dict())
    assert restored.to_dict() == camera.to_dict()
    assert not restored.depth_available and restored.depth_reference is None


def test_actual_sphere_run_and_replay_produce_identical_contacts_and_fitting(tmp_path):
    path = tmp_path / "sphere.jsonl"
    config = replace(load_dashboard_config(), recording_enabled=True, recording_path=str(path))
    engine = DashboardSimulation(config)
    try:
        command(engine, "set_grasp_preset", radius_m=.04)
        command(engine, "grasp")
        command(engine, "start_accumulation")
        active = set()
        snapshots = {}
        for _ in range(2600):
            engine.step()
            if engine.current_common_state.timestamp not in snapshots:
                engine._sample_history()
                payload = engine.state()
                active.update(row["sensor_id"] for row in payload["sensors"] if row["active"])
                snapshots[engine.current_common_state.timestamp] = payload["current_estimated_contacts"]
                assert all(row["timestamp"] == payload["snapshot"]["timestamp"] for row in payload["sensors"])
        command(engine, "fit_sphere")
        result = engine.state()
        assert len(active) >= 3
        assert result["accumulation"]["accepted_point_count"] > 0
        assert result["sphere_reconstruction"]["status"] != "NOT_FITTED"
        assert any(abs(joint["angle_rad"]) > .1 for joint in result["joints"])
    finally:
        engine.close()
    _, states = read_common_state_log(path)
    replay = ReplayRobotDataProvider(path)
    replay_engine = DashboardSimulation(provider=replay)
    command(replay_engine, "start_accumulation")
    processed = []
    for _ in range(len(states) - 1):
        replay_engine.step()
        current = replay_engine.state()
        processed.append(current["snapshot"]["timestamp"])
        assert current["current_estimated_contacts"] == snapshots[current["snapshot"]["timestamp"]]
    assert processed == [state.timestamp for state in states[1:]]
    command(replay_engine, "fit_sphere")
    assert replay_engine.state()["accumulated_points"] == result["accumulated_points"]
    assert replay_engine.state()["sphere_reconstruction"] == result["sphere_reconstruction"]
    command(replay_engine, "reset")
    assert replay_engine.state()["snapshot"]["timestamp"] == states[0].timestamp
    replay_engine.step()
    assert replay_engine.state()["snapshot"]["timestamp"] == states[1].timestamp


def test_common_reconstruction_ignores_object_metadata_and_gt_toggle():
    provider, _ = hardware()
    stream(provider)
    engine = DashboardSimulation(provider=provider)
    original = provider.read_common_state()
    altered = replace(original, object_state={"center_xyz": [99, 88, 77], "radius_m": 100})
    first = engine.reconstructor.reconstruct(reconstruction_input_from_common_state(original, provider.description))
    second = engine.reconstructor.reconstruct(reconstruction_input_from_common_state(altered, provider.description))
    assert first == second
    command(engine, "show_ground_truth", enabled=True)
    assert not engine.state()["evaluation"]["enabled"]


def test_unconfigured_hardware_dashboard_runs_without_fabricated_measurements():
    engine = DashboardSimulation(replace(load_dashboard_config(), source="hardware"))
    engine.step()
    payload = engine.state()
    assert payload["connection"]["state"] == "DISCONNECTED"
    assert payload["snapshot"]["timestamp"] is None
    assert payload["joints"] == payload["sensors"] == payload["current_estimated_contacts"] == []
    assert not payload["snapshot"]["valid"]
    json.dumps(payload, allow_nan=False)


def test_same_count_excluded_sensor_substitution_is_rejected():
    from sensors.registry_audit import require_approved_right_sensor_registry
    from sensors.sensor_kinematics import load_sensor_mounts
    provider, _ = hardware()
    mounts = load_sensor_mounts(provider.description.sensor_config_path)
    mount = mounts.pop("Finger01_Link02_Sensor")
    mounts["Finger01_Link04_Sensor"] = replace(mount, sensor_id="Finger01_Link04_Sensor")
    assert len(mounts) == 18
    with pytest.raises(RuntimeError, match="registry"):
        require_approved_right_sensor_registry(mounts, mounts)


def test_camera_reference_attaches_with_own_timestamp_without_fusion():
    provider, _ = hardware()
    observation = CameraObservation(100.0, "camera", True, False,
        rgb_reference="frames/rgb.png", intrinsics={"width": 640, "height": 480,
            "fx": 400, "fy": 400, "cx": 320, "cy": 240, "distortion_model": "none"},
        extrinsics_reference="calibration/camera_to_palm.yaml", time_base=provider.time_base)
    provider.accept_camera_observation(observation)
    stream(provider)
    state = CommonRobotState.from_dict(provider.read_common_state().to_dict())
    assert state.camera.timestamp == 100.0
    assert state.timestamp == 100.05
    assert state.camera.depth_reference is None


def test_duplicate_common_state_and_missing_joint_cannot_reconstruct():
    provider, _ = hardware()
    stream(provider)
    common = provider.read_common_state()
    with pytest.raises(ValueError, match="Duplicate"):
        replace(common, tactile_channels=(common.tactile_channels[0],) * 18)
    incomplete = replace(common, joint_positions={}, joint_velocities={})
    with pytest.raises(ValueError, match="joints"):
        reconstruction_input_from_common_state(incomplete, provider.description)


def test_device_clock_restart_clears_old_snapshot_and_closes_old_log(tmp_path):
    provider, clock = hardware()
    stream(provider)
    path = tmp_path / "old_epoch.jsonl"
    config = replace(load_dashboard_config(), recording_enabled=True, recording_path=str(path))
    engine = DashboardSimulation(config, provider)
    engine._sample_history()
    clock.now = 1.1
    provider.begin_connection()
    waiting = engine.state()
    assert waiting["snapshot"]["timestamp"] is None
    assert not waiting["recording"]["enabled"]
    assert "session change" in waiting["recording"]["detail"]
    provider.accept_joint_state(joints(provider, 1.0))
    provider.accept_tactile_state(tactile(provider, 1.0))
    engine.step()
    assert engine.state()["snapshot"]["timestamp"] == 1.0
    _, old_states = read_common_state_log(path)
    assert [state.timestamp for state in old_states] == [100.05]
    engine.close()
