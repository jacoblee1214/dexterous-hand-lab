import inspect

from sensors.sensor_kinematics import KinematicTree, SensorMount


def test_sensor_spatialization_api_has_no_contact_ground_truth_input() -> None:
    parameters = list(inspect.signature(KinematicTree.sensor_poses).parameters)
    assert parameters == ["self", "mounts", "joint_positions"]
    mount_fields = set(SensorMount.__dataclass_fields__)
    assert mount_fields == {
        "sensor_id",
        "finger_id",
        "parent_link",
        "sensor_type",
        "center_xyz",
        "surface_u_axis",
        "surface_v_axis",
        "width",
        "height",
        "outward_normal",
        "surface_thickness_m",
        "fingertip_radii_xyz",
    }
