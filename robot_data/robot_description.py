"""Portable right-hand geometry and scalar calibration, with no simulator import."""

from pathlib import Path
import xml.etree.ElementTree as ET

from robot_data.common import JointInfo, ProviderDescription
from reconstruction.reconstruction_input import CALIBRATION_FIELDS
from sensors.pressure_emulator import load_pressure_config
from sensors.registry_audit import require_approved_right_sensor_registry
from sensors.sensor_kinematics import load_sensor_mounts


def right_hand_description(source="hardware", display_name="REAL ROBOT"):
    root = Path(__file__).resolve().parents[1]
    urdf = root / "assets/V6_Force_R/urdf/V6_Force_R.urdf"
    sensor_path = root / "sensors/sensor_config.yaml"
    mounts = load_sensor_mounts(sensor_path)
    require_approved_right_sensor_registry(mounts, mounts)
    calibration = load_pressure_config(root / "sensors/pressure_config.yaml", mounts)
    joints = []
    for element in ET.parse(urdf).getroot().findall("joint"):
        if element.attrib["type"] == "fixed":
            continue
        limits = element.find("limit")
        joints.append(JointInfo(element.attrib["name"], float(limits.attrib["lower"]),
                                float(limits.attrib["upper"]), element.find("parent").attrib["link"],
                                element.find("child").attrib["link"]))
    return ProviderDescription(source, display_name, "right", str(urdf), str(sensor_path),
                               {key: {field: getattr(value, field) for field in CALIBRATION_FIELDS}
                                for key, value in calibration.items()}, tuple(joints))
