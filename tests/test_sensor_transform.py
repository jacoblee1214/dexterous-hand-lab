from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sensors.sensor_kinematics import (
    KinematicTree,
    load_sensor_mounts,
)
from simulation.model_builder import HAND_VARIANTS


ROOT = Path(__file__).resolve().parents[1]
URDF = HAND_VARIANTS["right"].urdf
CONFIG = ROOT / "sensors" / "sensor_config.yaml"


def test_zero_angle_sensor_position_matches_explicit_fk() -> None:
    tree = KinematicTree.from_urdf(URDF)
    mounts = load_sensor_mounts(CONFIG)
    mount = mounts["Finger03_Link02_Sensor"]

    pose = tree.sensor_poses({mount.sensor_id: mount})[mount.sensor_id]
    parent = tree.forward_kinematics()[mount.parent_link]
    expected = parent[:3, 3] + parent[:3, :3] @ mount.position_xyz
    np.testing.assert_allclose(pose.position, expected, atol=1e-10)


def test_mount_offset_moves_only_the_edited_sensor() -> None:
    tree = KinematicTree.from_urdf(URDF)
    mounts = load_sensor_mounts(CONFIG)
    original = tree.sensor_poses(mounts)
    sensor_id = "Finger04_Link03_Sensor"
    delta = np.array([0.004, -0.002, 0.001])
    edited = dict(mounts)
    edited[sensor_id] = replace(
        mounts[sensor_id],
        center_xyz=mounts[sensor_id].center_xyz + delta,
    )

    changed = tree.sensor_poses(edited)

    for current_id in mounts:
        if current_id == sensor_id:
            assert not np.allclose(original[current_id].position, changed[current_id].position)
        else:
            np.testing.assert_allclose(original[current_id].position, changed[current_id].position)
    assert np.isclose(
        np.linalg.norm(changed[sensor_id].position - original[sensor_id].position),
        np.linalg.norm(delta),
    )


def test_sensing_axis_rotates_with_parent_link() -> None:
    tree = KinematicTree.from_urdf(URDF)
    mounts = load_sensor_mounts(CONFIG)
    mount = mounts["Finger03_Tip_Sensor"]
    zero = tree.sensor_poses({mount.sensor_id: mount})[mount.sensor_id]
    angle = 0.42
    rotated = tree.sensor_poses(
        {mount.sensor_id: mount},
        {"joint_23": angle},
    )[mount.sensor_id]

    parent_zero = tree.forward_kinematics()[mount.parent_link][:3, :3]
    parent_rotated = tree.forward_kinematics({"joint_23": angle})[mount.parent_link][:3, :3]
    expected = parent_rotated @ parent_zero.T @ zero.sensing_direction
    np.testing.assert_allclose(rotated.sensing_direction, expected, atol=1e-10)
    assert np.isclose(np.linalg.norm(rotated.sensing_direction), 1.0)


def test_all_18_sensor_frames_are_well_formed() -> None:
    tree = KinematicTree.from_urdf(URDF)
    mounts = load_sensor_mounts(CONFIG)
    poses = tree.sensor_poses(mounts)
    assert len(poses) == 18
    assert {mount.finger_id for mount in mounts.values()} == {
        "Finger01",
        "Finger02",
        "Finger03",
        "Finger04",
        "Finger05",
        "Palm",
    }
    for pose in poses.values():
        np.testing.assert_allclose(pose.transform[3], [0, 0, 0, 1])
        assert np.isclose(np.linalg.norm(pose.sensing_direction), 1.0)


def test_sensor_catalog_contains_13_pads_and_5_curved_tip_regions() -> None:
    mounts = load_sensor_mounts(CONFIG)
    rectangular = [mount for mount in mounts.values() if mount.sensor_type == "rectangular"]
    fingertips = [
        mount for mount in mounts.values() if mount.sensor_type == "fingertip_surface"
    ]
    assert len(rectangular) == 13
    assert len(fingertips) == 5
    assert not any("Link04" in sensor_id for sensor_id in mounts)
    assert {sensor_id for sensor_id in mounts if sensor_id.startswith("Palm")} == {
        "Palm01_Sensor",
        "Palm02_Sensor",
        "Palm03_Sensor",
    }
    for mount in mounts.values():
        np.testing.assert_allclose(mount.local_sensing_axis, [0, 0, 1])
        assert mount.surface_width_m > 0
        assert mount.surface_height_m > 0
        computed = np.cross(mount.surface_u_axis, mount.surface_v_axis)
        computed /= np.linalg.norm(computed)
        # YAML persists calibrated axes to microradian-scale decimal precision.
        np.testing.assert_allclose(computed, mount.outward_normal, atol=1e-6)
    for mount in fingertips:
        assert mount.fingertip_radii_xyz is not None
        assert np.all(mount.fingertip_radii_xyz > 0)


def test_surface_coordinates_cover_pad_and_fingertip_regions() -> None:
    mounts = load_sensor_mounts(CONFIG)
    pad = mounts["Finger03_Link02_Sensor"]
    tip = mounts["Finger03_Tip_Sensor"]

    np.testing.assert_allclose(pad.surface_point_local(0.0, 0.0), pad.position_xyz)
    expected_pad_corner = (
        pad.center_xyz
        + pad.surface_u_axis * pad.width / 2
        - pad.surface_v_axis * pad.height / 2
    )
    np.testing.assert_allclose(pad.surface_point_local(1.0, -1.0), expected_pad_corner)

    # The fingertip mount position is the representative outermost point, but
    # non-zero (u, v) selects a different location on the curved region.
    np.testing.assert_allclose(tip.surface_point_local(0.0, 0.0), tip.position_xyz)
    assert not np.allclose(tip.surface_point_local(0.45, -0.25), tip.position_xyz)
    with pytest.raises(ValueError, match="unit disk"):
        tip.surface_point_local(1.0, 1.0)
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        pad.surface_point_local(1.01, 0.0)


def test_right_flexion_sign_moves_fingertips_toward_the_object_side() -> None:
    tree = KinematicTree.from_urdf(URDF)
    mounts = load_sensor_mounts(CONFIG)
    reference = tree.sensor_poses(mounts)
    flexed = tree.sensor_poses(
        mounts,
        {
            joint.name: (
                0.0
                if joint.name in {"joint_10", "joint_20", "joint_30", "joint_40"}
                else 0.5 if joint.name in {"joint_00", "joint_01"} else 0.7
            )
            for joint in tree.joints
        },
    )
    sphere_center = np.array([0.0, -0.070, 0.105])
    tip_ids = [f"Finger{finger:02d}_Tip_Sensor" for finger in range(1, 6)]
    reference_mean_distance = np.mean(
        [np.linalg.norm(reference[sensor_id].position - sphere_center) for sensor_id in tip_ids]
    )
    flexed_mean_distance = np.mean(
        [np.linalg.norm(flexed[sensor_id].position - sphere_center) for sensor_id in tip_ids]
    )
    assert flexed_mean_distance < reference_mean_distance
