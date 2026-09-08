"""Pose, original-mesh and replay contract checks for the browser digital twin."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from backend.dashboard_server import DashboardSimulation, load_dashboard_config
from backend.visual_assets import VisualAssets, link_transform
from reconstruction.common_state_adapter import reconstruction_input_from_common_state
from robot_data.common import CommonRobotState, LinkPose
from robot_data.mujoco_provider import MuJoCoRobotDataProvider
from robot_data.recording import CommonStateRecorder
from robot_data.replay_provider import ReplayRobotDataProvider
from robot_data.robot_description import right_hand_description


def test_manifest_maps_all_original_stls_with_unchanged_local_mounts():
    description = right_hand_description()
    assets = VisualAssets(description)
    assert len(assets.manifest["visuals"]) == 21
    assert len(assets.manifest["sensors"]) == 18
    assert assets.manifest["units"] == "metres"
    source = ET.parse(description.kinematic_model_path).getroot()
    for visual in assets.manifest["visuals"]:
        element = source.find(f"link[@name='{visual['link_name']}']/visual/geometry/mesh")
        expected = (Path(description.kinematic_model_path).parent / element.attrib["filename"]).resolve()
        assert assets.files[visual["url"]] == expected
        assert visual["scale_xyz"] == [1, 1, 1]
    from sensors.sensor_kinematics import load_sensor_mounts
    mounts = load_sensor_mounts(description.sensor_config_path)
    for entry in assets.manifest["sensors"]:
        mount = mounts[entry["sensor_id"]]
        assert entry["parent_link"] == mount.parent_link
        np.testing.assert_array_equal(entry["center_xyz"], mount.center_xyz)
        assert entry["width"] == mount.width


def test_streamed_mesh_poses_match_mujoco_at_snapshot_time_and_do_not_change_physics():
    provider = MuJoCoRobotDataProvider()
    engine = DashboardSimulation(provider=provider)
    before = engine.state()["hand"]["links"]
    provider.send_joint_targets({"joint_10": .5})
    for _ in range(100):
        engine.step()
    engine._update_sensor_pipeline(force=True)
    sim = provider.simulation
    original = sim.data.qpos.copy()
    reference = sim.mujoco.MjData(sim.model)
    reference.qpos[:] = sim.data.qpos
    sim.mujoco.mj_kinematics(sim.model, reference)
    payload = engine.state()
    assert payload["hand"]["pose_source"] == "provider"
    assert len(payload["hand"]["links"]) == 21
    for pose in payload["hand"]["links"]:
        body = sim.model.body(pose["name"]).id
        np.testing.assert_allclose(pose["position_xyz"], reference.xpos[body], atol=1e-12)
        np.testing.assert_allclose(pose["quaternion_wxyz"], reference.xquat[body], atol=1e-12)
        assert pose["timestamp"] == payload["snapshot"]["timestamp"]
    assert any(not np.allclose(a["quaternion_wxyz"], b["quaternion_wxyz"])
               for a, b in zip(before, payload["hand"]["links"]))
    np.testing.assert_array_equal(original, sim.data.qpos)
    links = {pose["name"]: pose for pose in payload["hand"]["links"]}
    for sensor in payload["sensors"]:
        mount = engine.mounts[sensor["sensor_id"]]
        transform = link_transform(links[mount.parent_link]) @ mount.link_to_sensor
        np.testing.assert_allclose(sensor["position_world_xyz"], transform[:3, 3], atol=1e-12)
        np.testing.assert_allclose(sensor["sensing_direction_world_xyz"], transform[:3, 2], atol=1e-12)
    engine.close()


def test_recorded_pose_replay_is_exact_and_legacy_logs_use_python_fk(tmp_path):
    provider = MuJoCoRobotDataProvider()
    path = tmp_path / "mesh_run.jsonl"
    samples = []
    with CommonStateRecorder(path, provider.description) as recorder:
        for _ in range(3):
            provider.step()
            sample = provider.read_common_state()
            samples.append(sample)
            recorder.record(sample)
    replay = ReplayRobotDataProvider(path)
    engine = DashboardSimulation(provider=replay)
    for index, sample in enumerate(samples):
        if index:
            engine.step()
        assert engine.state()["hand"]["links"] == [pose.to_dict() for pose in sample.link_poses]
        assert engine.current_common_state.link_poses == sample.link_poses
    legacy = samples[-1].to_dict()
    del legacy["link_poses"]
    engine.current_common_state = CommonRobotState.from_dict(legacy)
    assert len(engine.state()["hand"]["links"]) == 21
    assert engine.state()["hand"]["pose_source"] == "python_fk"


def test_display_poses_cannot_influence_contact_reconstruction():
    provider = MuJoCoRobotDataProvider()
    common = provider.read_common_state()
    altered = replace(common, link_poses=tuple(replace(pose, position_xyz=(100., 200., 300.)) for pose in common.link_poses))
    original = reconstruction_input_from_common_state(common, provider.description)
    other = reconstruction_input_from_common_state(altered, provider.description)
    assert original == other
    assert "link_poses" not in original.__dataclass_fields__


def test_bad_link_pose_contract_rejected():
    with pytest.raises(ValueError, match="unit"):
        LinkPose("Palm", None, 1., (0., 0., 0.), (0., 0., 0., 0.))


def test_native_debug_disabled_by_default_and_shares_simulation(monkeypatch):
    import mujoco.viewer
    calls = []
    class Handle:
        def is_running(self): return True
        def sync(self): calls.append("sync")
        def close(self): calls.append("close")
    def launch(model, data):
        calls.append((model, data))
        return Handle()
    monkeypatch.setattr(mujoco.viewer, "launch_passive", launch)
    provider = MuJoCoRobotDataProvider()
    assert not load_dashboard_config().debug_native_viewer
    provider.read_common_state()
    assert not calls
    provider.open_debug_viewer()
    assert calls[0] == (provider.simulation.model, provider.simulation.data)
    provider.sync_debug_viewer()
    provider.close()
    assert calls[-2:] == ["sync", "close"]
