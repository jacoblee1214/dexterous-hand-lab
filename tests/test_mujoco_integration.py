from pathlib import Path

import numpy as np
import pytest


mujoco = pytest.importorskip("mujoco")

from simulation.model_builder import HAND_VARIANTS, build_hand_variant, build_model
from simulation.mujoco_sim import (
    HandSimulation,
    ObjectController,
    OverlayState,
    apply_hand_control_command,
    configure_viewer,
    finger_joint_groups,
    make_key_callback,
    update_reconstruction_overlays,
    update_sensor_overlays,
    verify_single_joint_numerically,
)
from reconstruction.contact_point_buffer import ContactPointBuffer
from reconstruction.estimated_contact import EstimatedContactPoint
from reconstruction.temporal_observation import TemporalTactileObservation
from evaluation.contact_evaluator import GroundTruthContactPoint
from sensors.pressure_emulator import ScalarSensorReading
from sensors.sensor_kinematics import KinematicTree, load_sensor_mounts


def test_mujoco_loads_and_joint_motion_moves_attached_sites(tmp_path: Path) -> None:
    model_path = build_model(tmp_path / "hand.xml")
    simulation = HandSimulation(model_path)
    initial_sites = simulation.data.site_xpos.copy()

    joint_positions = {
        name: simulation.demonstration_target(
            mujoco.mj_name2id(simulation.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        )
        for name in simulation.joint_names
    }
    simulation.set_joint_positions(joint_positions)
    moved_sites = np.linalg.norm(simulation.data.site_xpos - initial_sites, axis=1)
    root = Path(__file__).resolve().parents[1]
    mounts = load_sensor_mounts(root / "sensors" / "sensor_config.yaml")
    poses = KinematicTree.from_urdf(
        HAND_VARIANTS["right"].urdf
    ).sensor_poses(mounts, joint_positions)
    for sensor_id, expected in poses.items():
        sensor_site = mujoco.mj_name2id(
            simulation.model, mujoco.mjtObj.mjOBJ_SITE, sensor_id
        )
        tip_site = mujoco.mj_name2id(
            simulation.model, mujoco.mjtObj.mjOBJ_SITE, f"{sensor_id}_DirectionTip"
        )
        np.testing.assert_allclose(
            simulation.data.site_xpos[sensor_site], expected.position, atol=1e-9
        )
        rendered_axis = simulation.data.site_xpos[tip_site] - simulation.data.site_xpos[sensor_site]
        rendered_axis /= np.linalg.norm(rendered_axis)
        np.testing.assert_allclose(rendered_axis, expected.sensing_direction, atol=1e-9)
    simulation.step(100)

    assert simulation.model.nbody == 25  # world + three objects + palm + 20 finger links
    assert len(simulation.joint_names) == 20
    assert len(simulation.sensor_site_names) == 18
    assert np.count_nonzero(moved_sites > 0.001) >= 45
    assert np.isfinite(simulation.data.qpos).all()


def test_left_hand_loads_and_sensor_sites_match_left_urdf_fk(tmp_path: Path) -> None:
    model_path = build_hand_variant("left", tmp_path / "hand_left.xml")
    simulation = HandSimulation(model_path)
    variant = HAND_VARIANTS["left"]
    mounts = load_sensor_mounts(variant.sensors)
    joint_positions = {}
    for joint_id, name in enumerate(simulation.joint_names):
        lower, upper = simulation.model.jnt_range[joint_id]
        joint_positions[name] = float(lower + 0.35 * (upper - lower))
    simulation.set_joint_positions(joint_positions)
    poses = KinematicTree.from_urdf(variant.urdf).sensor_poses(mounts, joint_positions)

    for sensor_id, expected in poses.items():
        sensor_site = mujoco.mj_name2id(
            simulation.model, mujoco.mjtObj.mjOBJ_SITE, sensor_id
        )
        tip_site = mujoco.mj_name2id(
            simulation.model, mujoco.mjtObj.mjOBJ_SITE, f"{sensor_id}_DirectionTip"
        )
        np.testing.assert_allclose(
            simulation.data.site_xpos[sensor_site], expected.position, atol=1e-9
        )
        axis = simulation.data.site_xpos[tip_site] - simulation.data.site_xpos[sensor_site]
        axis /= np.linalg.norm(axis)
        np.testing.assert_allclose(axis, expected.sensing_direction, atol=1e-9)

    simulation.step(100)
    assert simulation.model.nbody == 25  # world + three objects + palm + 20 finger links
    assert len(simulation.joint_names) == 20
    assert len(simulation.sensor_site_names) == 18
    assert np.isfinite(simulation.data.qpos).all()


@pytest.mark.parametrize("hand", ["right", "left"])
def test_every_joint_moves_only_its_downstream_chain(hand: str) -> None:
    simulation = HandSimulation(HAND_VARIANTS[hand].output)
    for joint_name in simulation.joint_names:
        changed, descendants = verify_single_joint_numerically(simulation, joint_name)
        assert changed == descendants


@pytest.mark.parametrize("hand", ["right", "left"])
def test_collision_blocks_grasp_at_sphere(hand: str) -> None:
    simulation = HandSimulation(HAND_VARIANTS[hand].output)
    objects = ObjectController(simulation, HAND_VARIANTS[hand].sphere_position)
    objects.activate("sphere")
    sphere_geom = mujoco.mj_name2id(
        simulation.model, mujoco.mjtObj.mjOBJ_GEOM, "sphere_object_geom"
    )
    assert simulation.model.geom_contype[sphere_geom] == 1
    _, sphere_qpos, _ = objects._ids("sphere")
    initial_position = simulation.data.qpos[sphere_qpos : sphere_qpos + 3].copy()
    initial_sphere_contacts = [
        simulation.data.contact[index]
        for index in range(simulation.data.ncon)
        if sphere_geom
        in (
            simulation.data.contact[index].geom1,
            simulation.data.contact[index].geom2,
        )
    ]
    assert initial_sphere_contacts == []

    simulation.data.ctrl[:] = simulation.model.key_ctrl[1]
    sphere_distances = []
    for _ in range(3000):
        simulation.step()
        sphere_distances.extend(
            simulation.data.contact[index].dist
            for index in range(simulation.data.ncon)
            if sphere_geom
            in (
                simulation.data.contact[index].geom1,
                simulation.data.contact[index].geom2,
            )
        )
    assert sphere_distances
    assert min(sphere_distances) > -0.015
    assert np.linalg.norm(
        simulation.data.qpos[sphere_qpos : sphere_qpos + 3] - initial_position
    ) > 0.001  # The object reacts dynamically; it is not a fixed display prop.
    sphere_joint, _, _ = objects._ids("sphere")
    sphere_dof = simulation.model.jnt_dofadr[sphere_joint]
    assert np.linalg.norm(simulation.data.qvel[sphere_dof : sphere_dof + 6]) < 1.0
    assert np.isfinite(simulation.data.qpos).all()


def test_sphere_radius_supports_four_experiment_sizes_and_resets_pose() -> None:
    simulation = HandSimulation(HAND_VARIANTS["right"].output)
    objects = ObjectController(simulation, HAND_VARIANTS["right"].sphere_position)
    objects.activate("sphere")
    for radius in (0.030, 0.040, 0.050, 0.060):
        objects.move(np.array([0.01, 0.0, 0.0]))
        objects.set_sphere_radius(radius)
        assert objects.sphere_radius == pytest.approx(radius)
        _, qpos_address, _ = objects._ids("sphere")
        np.testing.assert_allclose(
            simulation.data.qpos[qpos_address : qpos_address + 3],
            objects.spawn_position,
        )
    with pytest.raises(ValueError, match="one of"):
        objects.set_sphere_radius(0.045)


def test_sensor_overlay_has_arrows_ids_and_coordinate_frames() -> None:
    simulation = HandSimulation(HAND_VARIANTS["right"].output)
    mounts = load_sensor_mounts(HAND_VARIANTS["right"].sensors)
    scene = mujoco.MjvScene(simulation.model, maxgeom=1000)
    update_sensor_overlays(
        simulation,
        scene,
        mounts,
        OverlayState(show_directions=True, show_ids=True, show_frames=True),
    )
    # 13 rectangular surfaces+boundaries, 5 curved tips, 18 centers,
    # 18 sensing arrows, 18 labels, and 54 coordinate-frame arrows.
    assert scene.ngeom == 139


def test_active_sensor_surface_remains_visible_when_base_surfaces_are_hidden() -> None:
    simulation = HandSimulation(HAND_VARIANTS["right"].output)
    mounts = load_sensor_mounts(HAND_VARIANTS["right"].sensors)
    sensor_id = "Finger03_Link02_Sensor"
    mount = mounts[sensor_id]
    reading = ScalarSensorReading(
        sensor_id=sensor_id,
        parent_link=mount.parent_link,
        sensor_type=mount.sensor_type,
        surface_area_m2=mount.width * mount.height,
        normal_contact_force_n=1.0,
        scalar_output_n=1.0,
        pressure_pa=1000.0,
        indentation_m=0.001,
    )
    scene = mujoco.MjvScene(simulation.model, maxgeom=10)
    state = OverlayState(
        show_surfaces=False,
        show_centers=False,
        show_directions=False,
        show_active_sensors=True,
    )
    update_sensor_overlays(
        simulation, scene, mounts, state, readings={sensor_id: reading}
    )

    assert scene.ngeom == 2  # Active rectangular surface plus its boundary.
    np.testing.assert_allclose(scene.geoms[0].rgba, [1.0, 0.06, 0.01, 1.0])


def test_reconstruction_view_uses_a_mouse_controllable_free_camera() -> None:
    simulation = HandSimulation(HAND_VARIANTS["right"].output)

    class ViewerStub:
        cam = mujoco.MjvCamera()
        opt = mujoco.MjvOption()

    viewer = ViewerStub()
    configure_viewer(
        viewer,
        simulation,
        OverlayState(),
        free_camera=True,
        hand="right",
    )
    assert viewer.cam.type == mujoco.mjtCamera.mjCAMERA_FREE
    assert viewer.cam.fixedcamid == -1
    assert viewer.cam.distance == pytest.approx(0.34)


def test_interactive_hand_and_object_controls_update_immediately() -> None:
    simulation = HandSimulation(HAND_VARIANTS["right"].output)
    mounts = load_sensor_mounts(HAND_VARIANTS["right"].sensors)
    objects = ObjectController(simulation, HAND_VARIANTS["right"].sphere_position)
    groups = finger_joint_groups("right", mounts)
    callback = make_key_callback(
        OverlayState(), list(mounts), simulation, mounts, groups, objects
    )

    callback(ord("C"))
    np.testing.assert_allclose(simulation.data.ctrl, simulation.model.key_ctrl[1])
    callback(ord("O"))
    np.testing.assert_allclose(simulation.data.ctrl, simulation.model.key_ctrl[0])
    callback(ord("2"))
    finger_joint_names = groups["Finger02"]
    for name in finger_joint_names:
        actuator = mujoco.mj_name2id(
            simulation.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"servo_{name}"
        )
        joint = mujoco.mj_name2id(
            simulation.model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        assert simulation.data.ctrl[actuator] == simulation.demonstration_target(joint)

    callback(ord("S"))
    assert objects.active_shape == "sphere"
    callback(ord("Y"))
    assert objects.active_shape == "cylinder"
    callback(ord("B"))
    assert objects.active_shape == "box"
    callback(ord("N"))
    assert objects.active_shape is None


def test_reconstruction_controls_toggle_layers_and_keep_sphere_only() -> None:
    simulation = HandSimulation(HAND_VARIANTS["right"].output)
    mounts = load_sensor_mounts(HAND_VARIANTS["right"].sensors)
    objects = ObjectController(simulation, HAND_VARIANTS["right"].sphere_position)
    objects.activate("sphere")
    state = OverlayState()
    point_buffer = ContactPointBuffer()
    point_buffer.start()
    callback = make_key_callback(
        state,
        list(mounts),
        simulation,
        mounts,
        finger_joint_groups("right", mounts),
        objects,
        point_buffer,
        True,
    )

    callback(ord("X"))
    callback(ord("E"))
    callback(ord("M"))
    callback(ord("T"))
    callback(ord("G"))
    callback(ord("A"))
    assert not state.show_active_sensors
    assert not state.show_estimated_contacts
    assert not state.show_accumulated_points
    assert state.point_color_mode == "finger"
    assert state.show_ground_truth_contacts
    assert not point_buffer.accumulating

    point_buffer.start()
    point_buffer.add(
        [
            TemporalTactileObservation(
                timestamp=0.0,
                sensor_id="Finger03_Link02_Sensor",
                finger_id="Finger03",
                parent_link="middle_1",
                joint_state_snapshot={},
                sensor_position=(0.0, 0.0, 0.0),
                sensor_sensing_direction=(0.0, 0.0, 1.0),
                scalar_sensor_value=1.0,
                estimated_pressure=1000.0,
                estimated_indentation=0.001,
                estimated_contact_position=(0.0, 0.0, 0.001),
            )
        ]
    )
    callback(ord("Z"))
    assert point_buffer.points == ()
    assert not point_buffer.accumulating

    callback(ord("Y"))
    callback(ord("B"))
    callback(ord("N"))
    assert objects.active_shape == "sphere"


def test_estimated_and_ground_truth_points_use_separate_default_hidden_layers() -> None:
    simulation = HandSimulation(HAND_VARIANTS["right"].output)
    scene = mujoco.MjvScene(simulation.model, maxgeom=10)
    estimate = EstimatedContactPoint(
        0.0,
        "Finger03_Link02_Sensor",
        "Finger03",
        "middle_1",
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        1.0,
        1000.0,
        0.001,
        (0.0, 0.0, 0.001),
    )
    truth = GroundTruthContactPoint(
        0.0, "Finger03_Link02_Sensor", (0.0, 0.0, 0.002)
    )
    state = OverlayState(show_accumulated_points=False)

    update_reconstruction_overlays(simulation, scene, state, [estimate], (), [truth])
    assert scene.ngeom == 1
    np.testing.assert_allclose(scene.geoms[0].rgba, [1.0, 0.92, 0.02, 1.0])

    scene.ngeom = 0
    state.show_ground_truth_contacts = True
    update_reconstruction_overlays(simulation, scene, state, [estimate], (), [truth])
    assert scene.ngeom == 2
    np.testing.assert_allclose(scene.geoms[1].rgba, [0.10, 1.0, 0.25, 1.0])


def test_interactive_panel_commands_use_actuator_targets_without_overwriting_qpos() -> None:
    simulation = HandSimulation(HAND_VARIANTS["right"].output)
    mounts = load_sensor_mounts(HAND_VARIANTS["right"].sensors)
    groups = finger_joint_groups("right", mounts)
    displaced = {
        name: simulation.demonstration_target(
            mujoco.mj_name2id(simulation.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        )
        for name in simulation.joint_names
    }
    simulation.set_joint_positions(displaced)
    qpos_before = simulation.data.qpos.copy()

    apply_hand_control_command(simulation, groups, ("hand", "reset"))
    np.testing.assert_allclose(simulation.data.qpos, qpos_before)
    np.testing.assert_allclose(simulation.data.ctrl, simulation.model.key_ctrl[0])

    apply_hand_control_command(simulation, groups, ("finger", "Finger03", 0.5))
    for name in groups["Finger03"]:
        actuator = mujoco.mj_name2id(
            simulation.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"servo_{name}"
        )
        joint = mujoco.mj_name2id(
            simulation.model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        expected = 0.5 * (
            simulation.reference_positions()[name]
            + simulation.demonstration_target(joint)
        )
        assert simulation.data.ctrl[actuator] == pytest.approx(expected)


def test_finger_and_advanced_actuator_commands_physically_move_the_hand() -> None:
    simulation = HandSimulation(HAND_VARIANTS["right"].output)
    mounts = load_sensor_mounts(HAND_VARIANTS["right"].sensors)
    groups = finger_joint_groups("right", mounts)

    finger_joint = "joint_21"
    finger_joint_id = mujoco.mj_name2id(
        simulation.model, mujoco.mjtObj.mjOBJ_JOINT, finger_joint
    )
    finger_qpos = simulation.model.jnt_qposadr[finger_joint_id]
    initial = float(simulation.data.qpos[finger_qpos])
    apply_hand_control_command(simulation, groups, ("finger", "Finger03", 1.0))
    simulation.step(500)
    assert abs(float(simulation.data.qpos[finger_qpos]) - initial) > 0.1

    advanced_joint = "joint_32"
    advanced_id = mujoco.mj_name2id(
        simulation.model, mujoco.mjtObj.mjOBJ_JOINT, advanced_joint
    )
    advanced_qpos = simulation.model.jnt_qposadr[advanced_id]
    before = float(simulation.data.qpos[advanced_qpos])
    apply_hand_control_command(simulation, groups, ("joint", advanced_joint, 0.65))
    simulation.step(500)
    assert abs(float(simulation.data.qpos[advanced_qpos]) - before) > 0.1
    simulation.step(2500)
    assert np.max(np.abs(simulation.data.qvel[:20])) < 1e-3
    assert simulation.data.ncon == 0


def test_no_object_is_visible_or_collidable_by_default() -> None:
    simulation = HandSimulation(HAND_VARIANTS["left"].output)
    objects = ObjectController(simulation, HAND_VARIANTS["left"].sphere_position)
    assert objects.active_shape is None
    for shape in objects.SHAPES:
        _, _, geom_id = objects._ids(shape)
        assert simulation.model.geom_contype[geom_id] == 0
        assert simulation.model.geom_conaffinity[geom_id] == 0
        assert simulation.model.geom_rgba[geom_id, 3] == 0
