import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import pytest


mujoco = pytest.importorskip("mujoco")

from backend.dashboard_server import CONTROL_SCHEMA_VERSION, DashboardSimulation
from robot_data.common import RobotCommand, RobotCommandName
from robot_data.mujoco_provider import MuJoCoRobotDataProvider
from simulation.grasp_presets import GraspPresetLibrary, SphereGraspController
from simulation.model_builder import HAND_VARIANTS, build_hand_variant
from simulation.mujoco_sim import ObjectController, RIGHT_GRASP_CONFIG


def message(command, **parameters):
    return {
        "schema_version": CONTROL_SCHEMA_VERSION,
        "command": command,
        "parameters": parameters,
        "request_id": "reliable-grasp-test",
    }


def test_right_urdf_limits_axes_and_effort_become_mujoco_force_limits(tmp_path):
    urdf_joints = {
        joint.attrib["name"]: joint
        for joint in ET.parse(HAND_VARIANTS["right"].urdf).getroot().findall("joint")
        if joint.attrib.get("type") != "fixed"
    }
    model = mujoco.MjModel.from_xml_path(str(build_hand_variant("right", tmp_path / "hand.xml")))
    assert len(urdf_joints) == model.nu == 20
    for name, joint in urdf_joints.items():
        joint_id, actuator_id = model.joint(name).id, model.actuator(f"servo_{name}").id
        limit = joint.find("limit")
        np.testing.assert_allclose(model.jnt_axis[joint_id], [0, 0, 1])
        np.testing.assert_allclose(
            model.jnt_range[joint_id], [float(limit.attrib["lower"]), float(limit.attrib["upper"])]
        )
        effort = float(limit.attrib["effort"])
        np.testing.assert_allclose(model.actuator_forcerange[actuator_id], [-effort, effort])


def test_named_preset_save_is_actual_target_vector_backed_up_and_not_overwritten(tmp_path):
    config = tmp_path / "grasp.yaml"
    shutil.copy2(RIGHT_GRASP_CONFIG, config)
    provider = MuJoCoRobotDataProvider()
    provider.grasp.library = GraspPresetLibrary(config, HAND_VARIANTS["right"].urdf)
    provider.grasp.selected_preset_name = "sphere_30mm"
    provider.send_joint_targets({"joint_00": -0.02, "joint_01": -0.73, "joint_02": 0.91})
    provider.send_command(
        RobotCommand(RobotCommandName.SAVE_GRASP_PRESET, {"preset_name": "unique_pose"})
    )
    assert provider.grasp.library.presets["unique_pose"].joint_targets["joint_01"] == pytest.approx(-0.73)
    assert list((tmp_path / "grasp_backups").glob("*.bak"))
    with pytest.raises(ValueError, match="already exists"):
        provider.send_command(
            RobotCommand(RobotCommandName.SAVE_GRASP_PRESET, {"preset_name": "unique_pose"})
        )
    provider.close()


def test_fixed_and_supported_free_object_modes_are_physically_distinct():
    provider = MuJoCoRobotDataProvider()
    original = np.asarray(provider._object_state()["center_xyz"])
    for _ in range(200):
        provider.step()
    np.testing.assert_allclose(provider._object_state()["center_xyz"], original, atol=1e-12)

    provider.send_command(
        RobotCommand(RobotCommandName.SET_EXPERIMENT_MODE, {"mode": "free_object_grasp"})
    )
    cylinder = provider.simulation.model.geom("cylinder_object_geom").id
    sphere_body = provider.simulation.model.body("sphere_object").id
    assert provider.simulation.model.geom_contype[cylinder] == 1
    assert provider.simulation.model.body_gravcomp[sphere_body] == 0
    for _ in range(1000):
        provider.step()
    state = provider.read_common_state()
    assert state.object_state["experiment_mode"] == "free_object_grasp"
    assert state.object_state["center_xyz"][2] > 0.09
    assert state.grasp_state["contact_diagnostics"]["maximum_penetration_m"] < 0.001
    provider.close()


def test_repeat_grasp_parks_free_sphere_until_reference_is_reached():
    provider = MuJoCoRobotDataProvider()
    provider.send_command(
        RobotCommand(RobotCommandName.SET_GRASP_PRESET, {"radius_m": 0.04})
    )
    provider.send_command(
        RobotCommand(RobotCommandName.SET_EXPERIMENT_MODE, {"mode": "free_object_grasp"})
    )
    provider.send_command(RobotCommand(RobotCommandName.GRASP, {}))
    for _ in range(5000):
        provider.step()
        if not provider.grasp.running:
            break
    provider.send_command(RobotCommand(RobotCommandName.GRASP, {}))
    sphere_geom = provider.objects._ids("sphere")[2]
    assert provider.simulation.model.geom_contype[sphere_geom] == 0
    assert provider.read_common_state().object_state["center_xyz"][0] > 1.0
    maximum_penetration = 0.0
    for _ in range(5000):
        provider.step()
        diagnostics = provider._contact_diagnostics()
        maximum_penetration = max(
            maximum_penetration, diagnostics["maximum_penetration_m"]
        )
        if provider.grasp.current_stage >= 1:
            break
    assert provider.simulation.model.geom_contype[sphere_geom] == 1
    np.testing.assert_allclose(
        provider.read_common_state().object_state["center_xyz"],
        provider.grasp.selected_preset.sphere_center_xyz,
        atol=5e-4,
    )
    assert not provider.grasp.reference_timed_out
    assert maximum_penetration < 0.001
    provider.close()


def test_dashboard_named_trials_write_separate_reconstruction_and_evaluation(tmp_path):
    engine = DashboardSimulation()
    engine.experiment.output_root = tmp_path
    engine.handle_command(message("start_experiment", experiment_name="repeatable_30mm"))
    engine.handle_command(message("grasp"))
    for _ in range(100):
        engine.step()
    acknowledgement = engine.handle_command(message("finish_experiment"))
    result = acknowledgement["result"]["last_result"]
    reconstruction_path = Path(result["reconstruction_path"])
    evaluation_path = Path(result["evaluation_path"])
    reconstruction = json.loads(reconstruction_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    assert reconstruction["metadata"]["sphere_radius_m"] == pytest.approx(0.03)
    assert reconstruction["metadata"]["grasp_preset_name"] == "sphere_30mm"
    assert reconstruction["sample_count"] > 0
    assert "ground_truth" not in reconstruction_path.read_text(encoding="utf-8").lower()
    assert evaluation["dataset_type"] == "mujoco_sphere_experiment_evaluation"

    engine.handle_command(message("repeat_experiment"))
    for _ in range(10):
        engine.step()
    engine.handle_command(message("finish_experiment"))
    summary = json.loads((tmp_path / "repeatable_30mm" / "summary.json").read_text(encoding="utf-8"))
    assert summary["number_of_trials"] == 2
    assert summary["distribution"]["accepted_point_count"]["count"] == 2
    engine.close()


def test_free_object_motion_starts_a_new_accumulation_segment():
    engine = DashboardSimulation()
    engine.handle_command(message("set_experiment_mode", mode="free_object_grasp"))
    engine.handle_command(message("start_accumulation"))
    engine.step()
    engine.objects.move(np.array([0.003, 0.0, 0.0]))
    engine._update_sensor_pipeline(force=True)
    state = engine.state()
    assert state["object_motion_guard"]["segment"] >= 1
    assert "reset" in state["object_motion_guard"]["warning"]
    engine.close()
