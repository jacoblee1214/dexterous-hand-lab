import hashlib
from pathlib import Path
import shutil

import pytest
import yaml


mujoco = pytest.importorskip("mujoco")

from backend.dashboard_server import CONTROL_SCHEMA_VERSION, DashboardSimulation
from robot_data.common import RobotCommand, RobotCommandName
from robot_data.mujoco_provider import MuJoCoRobotDataProvider
from simulation.grasp_presets import GraspPresetLibrary
from simulation.model_builder import HAND_VARIANTS
from simulation.mujoco_sim import RIGHT_GRASP_CONFIG


def command(name, **parameters):
    return RobotCommand(RobotCommandName(name), parameters)


def dashboard_message(name, **parameters):
    return {
        "schema_version": CONTROL_SCHEMA_VERSION,
        "command": name,
        "parameters": parameters,
        "request_id": "taught-grasp-test",
    }


def temporary_library(provider, tmp_path: Path) -> Path:
    path = tmp_path / "grasp_config_right.yaml"
    shutil.copy2(RIGHT_GRASP_CONFIG, path)
    variant = HAND_VARIANTS["right"]
    provider.grasp.library = GraspPresetLibrary(path, variant.urdf, variant.joints)
    provider.grasp.selected_preset_name = "sphere_40mm"
    return path


def test_named_joint_mapping_exposes_all_urdf_joints_and_actuators():
    provider = MuJoCoRobotDataProvider()
    state = provider.read_common_state()
    mapping = state.grasp_state["joint_mapping"]
    assert set(mapping) == {f"joint_{finger}{joint}" for finger in range(5) for joint in range(4)}
    for name, details in mapping.items():
        assert details["axis_xyz"] == pytest.approx([0.0, 0.0, 1.0])
        assert details["actuator_name"] == f"servo_{name}"
        assert details["actuator_force_range_n_m"] == pytest.approx([-0.38, 0.38])
    provider.close()


def test_taught_pose_saves_actual_and_actuator_vectors_with_versions(tmp_path):
    provider = MuJoCoRobotDataProvider()
    path = temporary_library(provider, tmp_path)
    provider.send_command(command("set_teach_grasp_mode", enabled=True))
    provider.send_command(command("set_joint_target", joint_name="joint_02", target_rad=0.55))
    for _ in range(3000):
        provider.step()
        if provider.grasp.manual_pose_settled:
            break
    assert provider.grasp.manual_pose_settled
    visible_actual = provider.grasp.actual_joint_positions()
    actuator_targets = provider.grasp.current_actuator_targets()
    provider.send_command(command(
        "save_taught_pose",
        pose_name="approved_visible_pose",
        target_source="actual_measured",
        waypoint_role="final_grasp",
    ))
    saved = provider.grasp.library.presets["approved_visible_pose"]
    assert saved.is_taught
    assert dict(saved.joint_targets) == pytest.approx(visible_actual)
    assert dict(saved.measured_joint_positions) == pytest.approx(visible_actual)
    assert dict(saved.actuator_targets) == pytest.approx(actuator_targets)
    assert saved.metadata["urdf_version"] == hashlib.sha256(
        HAND_VARIANTS["right"].urdf.read_bytes()
    ).hexdigest()
    assert saved.metadata["joint_mapping_version"] == hashlib.sha256(
        HAND_VARIANTS["right"].joints.read_bytes()
    ).hexdigest()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))["presets"]["approved_visible_pose"]
    assert raw["taught"]["target_source"] == "actual_measured"
    assert raw["taught"]["measured_joint_positions"] == pytest.approx(visible_actual)
    assert list((tmp_path / "grasp_backups").glob("*.bak"))
    with pytest.raises(ValueError, match="already exists"):
        provider.send_command(command(
            "save_taught_pose",
            pose_name="approved_visible_pose",
            target_source="actual_measured",
            waypoint_role="final_grasp",
        ))
    provider.close()


def test_unsatisfied_actual_pose_cannot_be_saved_as_settled(tmp_path):
    provider = MuJoCoRobotDataProvider()
    temporary_library(provider, tmp_path)
    provider.send_command(command("set_teach_grasp_mode", enabled=True))
    with pytest.raises(ValueError, match="Automatic Sphere Grasp is disabled"):
        provider.send_command(command("grasp"))
    provider.send_command(command("set_joint_target", joint_name="joint_12", target_rad=1.0))
    with pytest.raises(ValueError, match="not settled"):
        provider.send_command(command(
            "save_taught_pose",
            pose_name="too_early",
            target_source="actual_measured",
            waypoint_role="finger_approach",
        ))
    provider.send_command(command(
        "save_taught_pose",
        pose_name="commanded_waypoint",
        target_source="actuator_target",
        waypoint_role="finger_approach",
    ))
    saved = provider.grasp.library.presets["commanded_waypoint"]
    assert saved.joint_targets["joint_12"] == pytest.approx(1.0)
    assert saved.measured_joint_positions["joint_12"] != pytest.approx(1.0)
    provider.close()


def test_waypoints_execute_in_order_with_bounded_actuator_targets_and_can_stop(tmp_path):
    provider = MuJoCoRobotDataProvider()
    temporary_library(provider, tmp_path)
    provider.send_command(command("set_teach_grasp_mode", enabled=True))
    provider.send_command(command(
        "save_taught_pose", pose_name="open_wp", target_source="actuator_target",
        waypoint_role="reference_open",
    ))
    provider.send_command(command("set_joint_target", joint_name="joint_02", target_rad=0.6))
    provider.send_command(command(
        "save_taught_pose", pose_name="final_wp", target_source="actuator_target",
        waypoint_role="final_grasp",
    ))
    provider.send_command(command("set_joint_target", joint_name="joint_02", target_rad=0.0))
    provider.send_command(command(
        "set_pose_validation_mode", mode="pose_reproduction_no_object"
    ))
    provider.send_command(command(
        "execute_taught_grasp", pose_names=["open_wp", "final_wp"],
        duration_seconds=0.3, tolerance_rad=0.04,
    ))
    before_qpos = provider.grasp.actual_joint_positions()["joint_02"]
    before_ctrl = provider.grasp.current_actuator_targets()["joint_02"]
    provider.step()
    after_qpos = provider.grasp.actual_joint_positions()["joint_02"]
    after_ctrl = provider.grasp.current_actuator_targets()["joint_02"]
    assert after_qpos != pytest.approx(0.6)
    assert abs(after_ctrl - before_ctrl) <= (
        provider.grasp.taught_max_target_velocity_rad_s
        * provider.simulation.model.opt.timestep
        + 1e-9
    )
    assert abs(after_qpos - before_qpos) < 0.1
    for _ in range(6000):
        provider.step()
        if not provider.grasp.running:
            break
    assert provider.grasp.taught_status == "REPRODUCED"
    assert provider.grasp.actual_joint_positions()["joint_02"] == pytest.approx(0.6, abs=0.04)
    assert provider.read_common_state().grasp_state["teach"]["pose_reproduction_validated"]

    provider.send_command(command(
        "execute_taught_grasp", pose_names=["open_wp"], duration_seconds=1.0,
        tolerance_rad=0.04,
    ))
    provider.step()
    provider.send_command(command("stop_taught_grasp"))
    assert not provider.grasp.running
    assert provider.grasp.taught_status == "STOPPED"
    assert provider.grasp.current_actuator_targets() == pytest.approx(
        provider.grasp.actual_joint_positions()
    )
    provider.close()


def test_dashboard_reports_target_actual_error_and_separate_validation_modes(tmp_path):
    engine = DashboardSimulation()
    temporary_library(engine.provider, tmp_path)
    engine.handle_command(dashboard_message("set_teach_grasp_mode", enabled=True))
    engine.handle_command(dashboard_message(
        "save_taught_pose", pose_name="dashboard_pose", target_source="actuator_target",
        waypoint_role="final_grasp",
    ))
    engine.handle_command(dashboard_message("load_taught_pose", pose_name="dashboard_pose"))
    engine.handle_command(dashboard_message(
        "set_pose_validation_mode", mode="pose_reproduction_no_object"
    ))
    state = engine.state()
    assert state["simulation"]["object"]["present"] is False
    assert state["simulation"]["grasp"]["teach"]["validation_mode"] == "pose_reproduction_no_object"
    assert len(state["joints"]) == 20
    assert all("taught_target_rad" in joint and "position_error_rad" in joint for joint in state["joints"])
    engine.handle_command(dashboard_message(
        "set_pose_validation_mode", mode="physical_sphere_grasp"
    ))
    state = engine.state()
    assert state["simulation"]["object"]["present"] is True
    assert state["simulation"]["object"]["experiment_mode"] == "free_object_grasp"
    assert state["simulation"]["grasp"]["teach"]["pose_reproduction_validated"] is False
    engine.close()
