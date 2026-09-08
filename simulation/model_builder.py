"""Build MuJoCo models from the preserved, updated V6_Force_R/L packages."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from sensors.sensor_kinematics import load_sensor_mounts, rpy_to_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_URDF = PROJECT_ROOT / "assets" / "V6_Force_R" / "urdf" / "V6_Force_R.urdf"
SENSOR_CONFIG = PROJECT_ROOT / "sensors" / "sensor_config.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "simulation" / "hand.xml"
LEFT_REFERENCE_URDF = PROJECT_ROOT / "assets" / "V6_Force_L" / "urdf" / "V6_Force_L.urdf"
LEFT_SENSOR_CONFIG = PROJECT_ROOT / "sensors" / "sensor_config_left.yaml"
LEFT_DEFAULT_OUTPUT = PROJECT_ROOT / "simulation" / "hand_left.xml"
RIGHT_JOINT_CONFIG = PROJECT_ROOT / "simulation" / "joint_config_right.yaml"
LEFT_JOINT_CONFIG = PROJECT_ROOT / "simulation" / "joint_config_left.yaml"
RESEARCH_CAMERA_CONFIG = PROJECT_ROOT / "vision" / "camera_calibration.yaml"


@dataclass(frozen=True)
class HandVariant:
    urdf: Path
    meshes: Path
    sensors: Path
    joints: Path
    output: Path
    sphere_position: str
    camera_position: str
    camera_xyaxes: str


HAND_VARIANTS = {
    "right": HandVariant(
        urdf=REFERENCE_URDF,
        meshes=PROJECT_ROOT / "assets" / "V6_Force_R" / "meshes",
        sensors=SENSOR_CONFIG,
        joints=RIGHT_JOINT_CONFIG,
        output=DEFAULT_OUTPUT,
        sphere_position="0 -0.070 0.105",
        camera_position="0.30 -0.38 0.27",
        camera_xyaxes="0.78 0.62 0 -0.29 0.36 0.89",
    ),
    "left": HandVariant(
        urdf=LEFT_REFERENCE_URDF,
        meshes=PROJECT_ROOT / "assets" / "V6_Force_L" / "meshes",
        sensors=LEFT_SENSOR_CONFIG,
        joints=LEFT_JOINT_CONFIG,
        output=LEFT_DEFAULT_OUTPUT,
        sphere_position="0 0.070 0.105",
        camera_position="0.30 0.38 0.27",
        camera_xyaxes="-0.78 0.62 0 -0.29 -0.36 0.89",
    ),
}

# Backward-compatible fallback for older zero-limited URDF exports. The current
# V6_Force_R/L packages have valid limits, so this table is not used for them.
EXPERIMENTAL_RANGES = {
    1: (-0.35, 0.35),
    # The source axes are -Z. Negative coordinates curl the supplied meshes
    # toward the palmar/object side; positive values visibly hyperextend them.
    2: (-1.40, 0.0),
    3: (-1.40, 0.0),
    4: (-1.40, 0.0),
}


def _numbers(value: str) -> list[float]:
    return [float(item) for item in value.split()]


def _fmt(values) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def _matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to MuJoCo's w-x-y-z quaternion."""
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = np.sqrt(1 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            quaternion = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif index == 1:
            scale = np.sqrt(1 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            quaternion = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = np.sqrt(1 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            quaternion = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    return quaternion / np.linalg.norm(quaternion)


def _joint_range(joint: ET.Element, joint_config: dict) -> tuple[float, float]:
    configured = joint_config.get(joint.attrib["name"], {})
    if "lower" in configured and "upper" in configured:
        return float(configured["lower"]), float(configured["upper"])
    limit = joint.find("limit")
    if limit is not None:
        lower = float(limit.attrib.get("lower", "0"))
        upper = float(limit.attrib.get("upper", "0"))
        if lower < upper:
            return lower, upper
    try:
        return EXPERIMENTAL_RANGES[int(joint.attrib["name"][-1])]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"No usable limit for joint {joint.attrib['name']}") from exc


def _neutral_and_demo(
    joint_name: str,
    range_pair: tuple[float, float],
    joint_config: dict,
) -> tuple[float, float]:
    lower, upper = range_pair
    configured = joint_config.get(joint_name, {})
    neutral = float(np.clip(configured.get("reference", 0.0), lower, upper))
    if "demo_target" in configured:
        return neutral, float(np.clip(configured["demo_target"], lower, upper))
    positive_distance = upper - neutral
    negative_distance = neutral - lower
    target = neutral + 0.55 * positive_distance
    if negative_distance > positive_distance:
        target = neutral - 0.55 * negative_distance
    return neutral, target


def build_model(
    output_path: str | Path = DEFAULT_OUTPUT,
    urdf_path: str | Path = REFERENCE_URDF,
    sensor_config_path: str | Path = SENSOR_CONFIG,
    mesh_directory: str | Path | None = None,
    sphere_position: str | None = None,
    joint_config_path: str | Path | None = RIGHT_JOINT_CONFIG,
    camera_position: str = "0.30 -0.38 0.27",
    camera_xyaxes: str = "0.78 0.62 0 -0.29 0.36 0.89",
    research_camera_config: str | Path | None = RESEARCH_CAMERA_CONFIG,
) -> Path:
    output = Path(output_path)
    urdf_root = ET.parse(urdf_path).getroot()
    mounts = load_sensor_mounts(sensor_config_path)
    joint_config = {}
    if joint_config_path is not None:
        joint_config = yaml.safe_load(Path(joint_config_path).read_text(encoding="utf-8"))["joints"]
    links = {element.attrib["name"]: element for element in urdf_root.findall("link")}
    joints = {element.attrib["name"]: element for element in urdf_root.findall("joint")}
    children: dict[str, list[ET.Element]] = {}
    child_links: set[str] = set()
    for joint in joints.values():
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        children.setdefault(parent, []).append(joint)
        child_links.add(child)
    base_links = set(links) - child_links
    if len(base_links) != 1:
        raise ValueError(f"Expected one base link, found {sorted(base_links)}")
    base_link = base_links.pop()

    mesh_directory = Path(mesh_directory or PROJECT_ROOT / "assets" / "V6_Force_R" / "meshes")
    root = ET.Element(
        "mujoco", {"model": f"{urdf_root.attrib.get('name', 'hand')}_tactile_first_deliverable"}
    )
    ET.SubElement(
        root,
        "compiler",
        {
            "angle": "radian",
            "meshdir": os.path.relpath(mesh_directory, output.parent),
            "autolimits": "true",
            "balanceinertia": "true",
        },
    )
    ET.SubElement(
        root,
        "option",
        {"timestep": "0.002", "gravity": "0 0 -9.81", "integrator": "implicitfast"},
    )
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {"azimuth": "135", "elevation": "-25"})
    ET.SubElement(visual, "quality", {"shadowsize": "2048"})
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", {"damping": "0.08", "armature": "0.001"})
    ET.SubElement(
        default,
        "geom",
        {
            "condim": "3",
            "friction": "0.8 0.01 0.001",
            "solref": "0.005 1",
            "solimp": "0.95 0.99 0.001",
        },
    )

    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        {"type": "skybox", "builtin": "gradient", "rgb1": "0.035 0.05 0.075", "rgb2": "0.005 0.008 0.014", "width": "512", "height": "3072"},
    )
    ET.SubElement(asset, "material", {"name": "hand_material", "rgba": "0.63 0.70 0.79 1", "metallic": "0.15", "roughness": "0.48"})
    ET.SubElement(asset, "material", {"name": "sphere_material", "rgba": "0.22 0.56 0.96 1", "metallic": "0.05", "roughness": "0.32"})
    ET.SubElement(asset, "material", {"name": "cylinder_material", "rgba": "0.96 0.52 0.16 1", "metallic": "0.04", "roughness": "0.36"})
    ET.SubElement(asset, "material", {"name": "box_material", "rgba": "0.48 0.82 0.28 1", "metallic": "0.04", "roughness": "0.40"})
    mesh_assets: dict[str, list[str]] = {}
    for link_name, link in links.items():
        visual_mesh = link.find("./visual/geometry/mesh")
        if visual_mesh is None:
            raise ValueError(f"Link {link_name} has no visual mesh")
        mesh_file = Path(visual_mesh.attrib["filename"]).name
        source_path = mesh_directory / mesh_file
        split_files = sorted(mesh_directory.glob(f"{source_path.stem}_part*.STL"))
        files = split_files or [source_path]
        mesh_assets[link_name] = []
        for part_index, file_path in enumerate(files):
            asset_name = f"mesh_{link_name}_{part_index}"
            mesh_assets[link_name].append(asset_name)
            ET.SubElement(asset, "mesh", {"name": asset_name, "file": file_path.name})

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(worldbody, "light", {"pos": "0 -0.5 0.6", "dir": "0 0.7 -1", "diffuse": "0.85 0.85 0.85", "castshadow": "true"})
    ET.SubElement(worldbody, "light", {"pos": "-0.5 0.2 0.35", "dir": "0.8 -0.2 -0.5", "diffuse": "0.35 0.42 0.55"})
    ET.SubElement(
        worldbody,
        "camera",
        {"name": "overview", "pos": camera_position, "xyaxes": camera_xyaxes},
    )
    if research_camera_config is not None:
        research = yaml.safe_load(
            Path(research_camera_config).read_text(encoding="utf-8")
        )
        pose, image = research["mujoco_pose"], research["image"]
        ET.SubElement(
            worldbody,
            "camera",
            {
                "name": str(research["camera_name"]),
                "pos": _fmt(pose["position_world_m"]),
                "xyaxes": _fmt(pose["xyaxes_world"]),
                "fovy": str(float(image["fovy_degrees"])),
            },
        )
    ET.SubElement(worldbody, "geom", {"name": "floor", "type": "plane", "pos": "0 0 -0.035", "size": "0.5 0.5 0.01", "rgba": "0.055 0.075 0.10 1"})
    sensor_by_link: dict[str, list] = {}
    for mount in mounts.values():
        sensor_by_link.setdefault(mount.parent_link, []).append(mount)

    def add_link(
        parent_xml: ET.Element,
        link_name: str,
        incoming_joint: ET.Element | None = None,
    ) -> None:
        body_attributes = {"name": link_name}
        if incoming_joint is not None:
            origin = incoming_joint.find("origin")
            xyz = _numbers(origin.attrib.get("xyz", "0 0 0"))
            rpy = _numbers(origin.attrib.get("rpy", "0 0 0"))
            body_attributes.update(
                {"pos": _fmt(xyz), "quat": _fmt(_matrix_to_quaternion(rpy_to_matrix(rpy)))}
            )
        body = ET.SubElement(parent_xml, "body", body_attributes)
        link = links[link_name]
        inertial = link.find("inertial")
        if incoming_joint is not None and inertial is not None:
            inertia = inertial.find("inertia").attrib
            ET.SubElement(
                body,
                "inertial",
                {
                    "pos": inertial.find("origin").attrib.get("xyz", "0 0 0"),
                    "mass": inertial.find("mass").attrib["value"],
                    "fullinertia": _fmt(
                        [inertia[key] for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]
                    ),
                },
            )
        if incoming_joint is not None and incoming_joint.attrib["type"] != "fixed":
            lower, upper = _joint_range(incoming_joint, joint_config)
            ET.SubElement(
                body,
                "joint",
                {
                    "name": incoming_joint.attrib["name"],
                    "type": "hinge",
                    "axis": incoming_joint.find("axis").attrib["xyz"],
                    "range": _fmt([lower, upper]),
                },
            )
        for part_index, mesh_asset in enumerate(mesh_assets[link_name]):
            ET.SubElement(
                body,
                "geom",
                {"name": f"visual_{link_name}_{part_index}", "type": "mesh", "mesh": mesh_asset, "material": "hand_material", "contype": "0", "conaffinity": "0", "group": "1"},
            )
            ET.SubElement(
                body,
                "geom",
                {"name": f"collision_{link_name}_{part_index}", "type": "mesh", "mesh": mesh_asset, "rgba": "0 0 0 0", "contype": "1", "conaffinity": "1", "group": "5", "margin": "0.0005"},
            )
        for mount in sensor_by_link.get(link_name, []):
            position = mount.position_xyz
            axis_in_link = mount.outward_normal
            end = position + 0.014 * axis_in_link
            ET.SubElement(body, "site", {"name": mount.sensor_id, "type": "sphere", "pos": _fmt(position), "quat": _fmt(_matrix_to_quaternion(mount.rotation_matrix)), "size": "0.0035", "rgba": "1 0.18 0.05 1", "group": "3"})
            ET.SubElement(body, "site", {"name": f"{mount.sensor_id}_Direction", "type": "capsule", "fromto": _fmt([*position, *end]), "size": "0.00075", "rgba": "1 0.82 0.18 0.92", "group": "4"})
            ET.SubElement(body, "site", {"name": f"{mount.sensor_id}_DirectionTip", "type": "sphere", "pos": _fmt(end), "size": "0.0014", "rgba": "1 0.82 0.18 0.92", "group": "4"})
        for child_joint in children.get(link_name, []):
            add_link(body, child_joint.find("child").attrib["link"], child_joint)

    add_link(worldbody, base_link)

    # Convex collision hulls around mechanically connected parts overlap near
    # their pivots by design. They must not repel one another; external object
    # collisions and non-connected finger-to-finger collisions remain enabled.
    excluded_body_pairs: set[tuple[str, str]] = set()
    for joint in joints.values():
        excluded_body_pairs.add(
            (joint.find("parent").attrib["link"], joint.find("child").attrib["link"])
        )
    # The first flexion link is partially sleeved by the palm/base link in the
    # supplied CAD assembly even though an intermediate ab/adduction link sits
    # between them in the kinematic tree.
    for child_joint in children.get(base_link, []):
        first_link = child_joint.find("child").attrib["link"]
        for grandchild_joint in children.get(first_link, []):
            excluded_body_pairs.add(
                (base_link, grandchild_joint.find("child").attrib["link"])
            )
    contact = ET.SubElement(root, "contact")
    for body1, body2 in sorted(excluded_body_pairs):
        ET.SubElement(contact, "exclude", {"body1": body1, "body2": body2})

    object_specs = [
        ("sphere", {"type": "sphere", "size": "0.030"}),
        ("cylinder", {"type": "cylinder", "size": "0.025 0.035"}),
        ("box", {"type": "box", "size": "0.027 0.027 0.027"}),
    ]
    object_parking_qpos = []
    for object_index, (shape, geom_attributes) in enumerate(object_specs):
        parking_position = [1.5 + 0.2 * object_index, 1.5, 0.2]
        object_body = ET.SubElement(
            worldbody,
            "body",
            {
                "name": f"{shape}_object",
                "pos": _fmt(parking_position),
                "gravcomp": "1",
            },
        )
        ET.SubElement(
            object_body,
            "joint",
            {"name": f"{shape}_object_free", "type": "free", "damping": "2.0"},
        )
        ET.SubElement(
            object_body,
            "geom",
            {
                "name": f"{shape}_object_geom",
                **geom_attributes,
                "material": f"{shape}_material",
                "rgba": "0 0 0 0",
                "mass": "0.500",
                # Compile collision pairs; ObjectController disables inactive
                # shapes at runtime and parks them outside the workspace.
                "contype": "1",
                "conaffinity": "1",
                "group": "2",
            },
        )
        object_parking_qpos.extend([*parking_position, 1.0, 0.0, 0.0, 0.0])

    movable_joints = [
        joint for joint in urdf_root.findall("joint") if joint.attrib["type"] != "fixed"
    ]
    ranges = [_joint_range(joint, joint_config) for joint in movable_joints]
    actuator = ET.SubElement(root, "actuator")
    for joint, (lower, upper) in zip(movable_joints, ranges):
        name = joint.attrib["name"]
        effort = float(joint.find("limit").attrib["effort"])
        ET.SubElement(
            actuator,
            "position",
            {
                "name": f"servo_{name}",
                "joint": name,
                "kp": "2.0",
                "kv": "0.18",
                "ctrlrange": _fmt([lower, upper]),
                "forcerange": _fmt([-effort, effort]),
            },
        )

    poses = [
        _neutral_and_demo(joint.attrib["name"], range_pair, joint_config)
        for joint, range_pair in zip(movable_joints, ranges)
    ]
    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(keyframe, "key", {"name": "open", "qpos": _fmt([pose[0] for pose in poses] + object_parking_qpos), "ctrl": _fmt([pose[0] for pose in poses])})
    ET.SubElement(keyframe, "key", {"name": "grasp", "qpos": _fmt([pose[1] for pose in poses] + object_parking_qpos), "ctrl": _fmt([pose[1] for pose in poses])})

    ET.indent(root, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    return output


def build_hand_variant(hand: str, output_path: str | Path | None = None) -> Path:
    variant = HAND_VARIANTS[hand]
    return build_model(
        output_path=output_path or variant.output,
        urdf_path=variant.urdf,
        sensor_config_path=variant.sensors,
        mesh_directory=variant.meshes,
        sphere_position=variant.sphere_position,
        joint_config_path=variant.joints,
        camera_position=variant.camera_position,
        camera_xyaxes=variant.camera_xyaxes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", choices=sorted(HAND_VARIANTS), default="right")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    generated = build_hand_variant(args.hand, args.output)
    print(f"Generated {args.hand} hand: {generated}")


if __name__ == "__main__":
    main()
