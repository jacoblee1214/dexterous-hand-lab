from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from simulation.model_builder import HAND_VARIANTS, build_hand_variant, build_model


def test_generated_model_has_hand_sphere_sensors_and_actuators(tmp_path: Path) -> None:
    output = build_model(tmp_path / "hand.xml")
    root = ET.parse(output).getroot()

    sensor_markers = [
        site for site in root.findall(".//site")
        if site.attrib["name"].endswith("_Sensor")
    ]
    direction_markers = [
        site for site in root.findall(".//site")
        if site.attrib["name"].endswith("_Direction")
    ]
    model_joints = [
        joint
        for joint in root.findall(".//joint")
        if joint.attrib.get("type") == "hinge"
    ]
    assert len(model_joints) == 20
    assert len(root.findall("./actuator/position")) == 20
    assert len(sensor_markers) == 18
    assert len(direction_markers) == 18
    assert root.find(".//body[@name='sphere_object']") is not None
    assert root.find(".//body[@name='cylinder_object']") is not None
    assert root.find(".//body[@name='box_object']") is not None
    ranges = {
        joint.attrib["name"]: tuple(float(value) for value in joint.attrib["range"].split())
        for joint in root.findall(".//joint")
        if joint.attrib.get("type") == "hinge"
    }
    assert ranges["joint_11"] == (-0.0698131700798, 1.57079632679)
    assert ranges["joint_43"] == (-0.174532925199, 1.3962634016)


def test_updated_right_urdf_ranges_are_preserved_and_movable(tmp_path: Path) -> None:
    variant = HAND_VARIANTS["right"]
    source = ET.parse(variant.urdf).getroot()
    expected = {
        joint.attrib["name"]: tuple(
            float(joint.find("limit").attrib[key]) for key in ("lower", "upper")
        )
        for joint in source.findall("joint")
        if joint.attrib["type"] == "revolute"
    }
    generated = ET.parse(build_model(tmp_path / "hand.xml")).getroot()
    actual = {
        joint.attrib["name"]: tuple(float(value) for value in joint.attrib["range"].split())
        for joint in generated.findall(".//joint")
        if joint.attrib.get("type") == "hinge"
    }
    assert actual.keys() == expected.keys()
    for name in actual:
        assert actual[name] == pytest.approx(expected[name])
        assert actual[name][0] < actual[name][1]


def test_left_mjcf_preserves_valid_left_urdf_joint_ranges(tmp_path: Path) -> None:
    variant = HAND_VARIANTS["left"]
    source = ET.parse(variant.urdf).getroot()
    expected = {
        joint.attrib["name"]: tuple(
            float(joint.find("limit").attrib[key]) for key in ("lower", "upper")
        )
        for joint in source.findall("joint")
        if joint.attrib["type"] == "revolute"
    }
    generated = ET.parse(build_hand_variant("left", tmp_path / "left.xml")).getroot()
    actual = {
        joint.attrib["name"]: tuple(float(value) for value in joint.attrib["range"].split())
        for joint in generated.findall(".//joint")
            if joint.attrib.get("type") == "hinge"
    }
    assert actual.keys() == expected.keys()
    for name in actual:
        assert actual[name] == pytest.approx(expected[name])
    assert len(generated.findall("./actuator/position")) == 20
    assert len(
        [site for site in generated.findall(".//site") if site.attrib["name"].endswith("_Sensor")]
    ) == 18
