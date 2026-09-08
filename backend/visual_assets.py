"""Allowlisted original STL assets and portable visual transforms for the browser."""

from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

from sensors.sensor_kinematics import load_sensor_mounts


def link_transform(pose):
    w, x, y, z = pose["quaternion_wxyz"]
    matrix = np.eye(4)
    matrix[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                      [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    matrix[:3, 3] = pose["position_xyz"]
    return matrix


class VisualAssets:
    def __init__(self, description):
        urdf = Path(description.kinematic_model_path).resolve()
        approved_root = Path(__file__).resolve().parents[1] / "assets"
        self.files = {}
        visuals = []
        for link in ET.parse(urdf).getroot().findall("link"):
            for index, visual in enumerate(link.findall("visual")):
                mesh = visual.find("geometry/mesh")
                if mesh is None:
                    continue
                path = (urdf.parent / mesh.attrib["filename"]).resolve()
                if not path.is_relative_to(approved_root) or path.suffix.lower() != ".stl" or not path.is_file():
                    raise ValueError(f"Visual mesh is outside the approved STL asset directory: {path}")
                url = f"/robot-assets/meshes/{link.attrib['name']}_{index}.stl"
                self.files[url] = path
                origin = visual.find("origin")
                origin_values = origin.attrib if origin is not None else {}
                visuals.append({"link_name": link.attrib["name"], "url": url,
                                "position_xyz": [float(v) for v in origin_values.get("xyz", "0 0 0").split()],
                                "rotation_rpy": [float(v) for v in origin_values.get("rpy", "0 0 0").split()],
                                "scale_xyz": [float(v) for v in mesh.attrib.get("scale", "1 1 1").split()]})
        sensors = []
        for mount in load_sensor_mounts(description.sensor_config_path).values():
            sensors.append({"sensor_id": mount.sensor_id, "parent_link": mount.parent_link,
                            "sensor_type": mount.sensor_type, "center_xyz": mount.center_xyz.tolist(),
                            "surface_u_axis": mount.surface_u_axis.tolist(), "surface_v_axis": mount.surface_v_axis.tolist(),
                            "outward_normal": mount.outward_normal.tolist(), "width": mount.width, "height": mount.height,
                            "surface_thickness_m": mount.surface_thickness_m,
                            "fingertip_radii_xyz": mount.fingertip_radii_xyz.tolist() if mount.fingertip_radii_xyz is not None else None})
        self.manifest = {"schema_version": "1.0.0", "handedness": description.handedness,
                         "units": "metres", "up_axis": "Z", "visuals": visuals, "sensors": sensors}
