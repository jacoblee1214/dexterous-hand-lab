"""Clean fixed RGB camera rendering, observation metadata, and evaluation masks."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import time

import numpy as np
from PIL import Image

from backend.mujoco_offscreen import RenderedFrame
from robot_data.common import CameraObservation
from vision.baseline import (
    describe_unknown_radius,
    fit_known_radius_sphere,
    segment_blue_sphere,
    sphere_silhouette_mask,
)
from vision.calibration import (
    DEFAULT_CALIBRATION,
    CalibratedCamera,
    calibration_from_mujoco,
    load_camera_config,
)


@dataclass(frozen=True)
class ResearchCapture:
    frame: RenderedFrame
    observation: CameraObservation
    segmentation_mask: np.ndarray
    ground_truth_mask: np.ndarray
    vision_result: dict
    evaluation: dict


class ResearchRGBCamera:
    """A fixed camera with no debug overlays and no depth in algorithm input."""

    def __init__(self, provider, calibration_path: str | Path = DEFAULT_CALIBRATION, *, jpeg_quality: int = 90):
        self.config = load_camera_config(calibration_path)
        self.source = provider.description.source
        self.replay_log_path = getattr(provider, "log_path", None)
        self.jpeg_quality = int(jpeg_quality)
        self._owned_simulation = None
        self._owned_objects = None
        if self.source == "simulation":
            simulation = provider.simulation
        elif self.source == "replay":
            from simulation.model_builder import HAND_VARIANTS, build_hand_variant
            from simulation.mujoco_sim import HandSimulation, ObjectController

            variant = HAND_VARIANTS[provider.description.handedness]
            build_hand_variant(provider.description.handedness)
            simulation = HandSimulation(variant.output)
            self._owned_simulation = simulation
            self._owned_objects = ObjectController(simulation, variant.sphere_position)
            self._owned_objects.activate("sphere")
        else:
            raise ValueError("Research RGB rendering is available only for simulation and replay")
        self.mujoco, self.model = simulation.mujoco, simulation.model
        self.data = self.mujoco.MjData(self.model)
        self._live_data = simulation.data if self.source == "simulation" else None
        self.renderer = self.mujoco.Renderer(
            self.model, height=self.config.height, width=self.config.width
        )
        self.scene_option = self.mujoco.MjvOption()
        self.mujoco.mjv_defaultOption(self.scene_option)
        # Raw research RGB contains visible scene geometry only. Sensor sites,
        # direction sites, and transparent collision hulls are debug artifacts.
        self.scene_option.sitegroup[3] = False
        self.scene_option.sitegroup[4] = False
        self.scene_option.geomgroup[5] = False
        self.sequence = 0

    def _sync_data(self, common_state) -> None:
        if self.source == "simulation":
            self.mujoco.mj_copyData(self.data, self.model, self._live_data)
            return
        if common_state is None:
            return
        self.data.qvel[:] = 0
        for name, value in common_state.joint_positions.items():
            joint_id = self.model.joint(name).id
            self.data.qpos[self.model.jnt_qposadr[joint_id]] = value
        object_state = dict(common_state.object_state or {})
        if object_state.get("center_xyz"):
            joint_id = self.model.joint("sphere_object_free").id
            address = self.model.jnt_qposadr[joint_id]
            self.data.qpos[address : address + 3] = object_state["center_xyz"]
            self.data.qpos[address + 3 : address + 7] = object_state.get(
                "quaternion_wxyz", [1, 0, 0, 0]
            )
            geom_id = self.model.geom("sphere_object_geom").id
            self.model.geom_size[geom_id, 0] = float(object_state.get("radius_m", 0.03))
            visible = bool(object_state.get("present", True))
            self.model.geom_rgba[geom_id] = [0.22, 0.56, 0.96, 1.0 if visible else 0.0]
        self.data.time = common_state.timestamp
        self.mujoco.mj_forward(self.model, self.data)

    def calibration(self) -> CalibratedCamera:
        return calibration_from_mujoco(self.model, self.data, self.config)

    def render_clean_rgb(self, common_state) -> np.ndarray:
        self._sync_data(common_state)
        # Replays use the archived observation as their camera input.  The
        # renderer-only MuJoCo model still provides link/object poses for the
        # dashboard and evaluation, but must not silently replace recorded RGB.
        if self.source == "replay" and common_state is not None and common_state.camera:
            reference = common_state.camera.rgb_reference
            if reference and not reference.startswith(("/", "http://", "https://")):
                image_path = self.replay_log_path.parent / reference
                if not image_path.is_file():
                    raise FileNotFoundError(f"Recorded RGB observation is missing: {image_path}")
                with Image.open(image_path) as archived:
                    rgb = np.asarray(archived.convert("RGB"))
                expected = (self.config.height, self.config.width, 3)
                if rgb.shape != expected:
                    raise ValueError(
                        f"Recorded RGB shape {rgb.shape} does not match calibration {expected}"
                    )
                return rgb.copy()
        self.renderer.update_scene(
            self.data, camera=self.config.camera_name, scene_option=self.scene_option
        )
        return self.renderer.render().copy()

    def _ground_truth_mask(self) -> np.ndarray:
        self.renderer.update_scene(
            self.data, camera=self.config.camera_name, scene_option=self.scene_option
        )
        self.renderer.enable_segmentation_rendering()
        try:
            segmentation = self.renderer.render().copy()
        finally:
            self.renderer.disable_segmentation_rendering()
        sphere_geom = self.model.geom("sphere_object_geom").id
        if segmentation.ndim != 3 or segmentation.shape[2] != 2:
            raise RuntimeError("Unexpected MuJoCo segmentation output")
        # MuJoCo 3.12 returns (object id, object type) per pixel.
        return (
            (segmentation[:, :, 0] == sphere_geom)
            & (segmentation[:, :, 1] == int(self.mujoco.mjtObj.mjOBJ_GEOM))
        )

    def capture(self, engine, *, known_radius: bool = True) -> ResearchCapture:
        common = engine.current_common_state
        if common is None:
            raise ValueError("A synchronized robot state is required before RGB capture")
        rgb = self.render_clean_rgb(common)
        calibration = self.calibration()
        segmentation = segment_blue_sphere(rgb, self.config.segmentation)
        object_state = dict(common.object_state or {})
        if known_radius:
            result = fit_known_radius_sphere(
                segmentation, calibration, float(object_state["radius_m"])
            )
        else:
            result = describe_unknown_radius(segmentation, calibration)
        result = {
            "input_boundary": "clean_rgb_only",
            "camera_timestamp": common.timestamp,
            "camera_id": self.config.camera_id,
            "segmentation": segmentation.diagnostics(),
            "sphere": result,
            "unknown_radius_setting": describe_unknown_radius(segmentation, calibration),
        }
        contacts = []
        if engine.current_estimates:
            positions = np.asarray(
                [estimate.estimated_contact_position for estimate in engine.current_estimates]
            )
            pixels, depth = calibration.project_world(positions)
            contacts = [
                {
                    "sensor_id": estimate.sensor_id,
                    "pixel_uv": pixels[index].tolist(),
                    "depth_camera_cv_m": float(depth[index]),
                    "visible_in_image_bounds": bool(
                        depth[index] > 0
                        and 0 <= pixels[index, 0] < self.config.width
                        and 0 <= pixels[index, 1] < self.config.height
                    ),
                }
                for index, estimate in enumerate(engine.current_estimates)
            ]
        result["projected_estimated_tactile_contacts"] = contacts
        ground_truth = self._ground_truth_mask()
        intersection = int(np.logical_and(segmentation.mask, ground_truth).sum())
        union = int(np.logical_or(segmentation.mask, ground_truth).sum())
        predicted = max(1, int(segmentation.mask.sum()))
        truth = max(1, int(ground_truth.sum()))
        evaluation = {
            "boundary": "evaluation_only_mujoco_segmentation_not_algorithm_input",
            "segmentation_iou": intersection / union if union else 1.0,
            "segmentation_precision": intersection / predicted,
            "segmentation_recall": intersection / truth,
            "ground_truth_visible_pixel_count": int(ground_truth.sum()),
        }
        sphere = result["sphere"]
        center_world = sphere.get("estimated_center_world_m")
        true_center = object_state.get("center_xyz")
        evaluation["sphere_center_error_m"] = (
            float(np.linalg.norm(np.asarray(center_world) - np.asarray(true_center)))
            if center_world is not None and true_center is not None
            else None
        )
        evaluation["radius_error_m"] = None
        evaluation["radius_error_reason"] = "radius is a declared prior, not an estimate"
        if true_center is not None:
            homogeneous = np.r_[np.asarray(true_center, dtype=float), 1.0]
            true_center_cv = (calibration.camera_cv_from_world @ homogeneous)[:3]
            full_silhouette = sphere_silhouette_mask(
                calibration, true_center_cv, float(object_state["radius_m"])
            )
            full_pixels = max(1, int(full_silhouette.sum()))
            visibility = float(ground_truth.sum() / full_pixels)
            evaluation.update(
                {
                    "full_unoccluded_silhouette_pixel_count": full_pixels,
                    "visible_fraction": visibility,
                    "occlusion_condition": (
                        "clear" if visibility >= 0.9 else
                        "partially_occluded" if visibility >= 0.5 else
                        "severely_occluded"
                    ),
                }
            )
        output = BytesIO()
        Image.fromarray(rgb).save(output, format="JPEG", quality=self.jpeg_quality)
        self.sequence += 1
        frame = RenderedFrame(
            self.sequence,
            float(common.timestamp),
            time.time(),
            self.source,
            output.getvalue(),
        )
        observation = CameraObservation(
            timestamp=float(common.timestamp),
            frame_id=self.config.frame_id,
            camera_id=self.config.camera_id,
            rgb_available=True,
            depth_available=False,
            image_width=self.config.width,
            image_height=self.config.height,
            source=self.source,
            calibration_version=self.config.calibration_version,
            intrinsic_calibration_reference="/research-rgb/calibration.json",
            rgb_reference="/research-rgb/latest.jpg",
            intrinsics=calibration.intrinsics,
            extrinsics_reference="/research-rgb/calibration.json#T_world_from_camera_cv",
            time_base=self.config.time_base,
        )
        return ResearchCapture(
            frame, observation, segmentation.mask, ground_truth, result, evaluation
        )

    def close(self) -> None:
        self.renderer.close()
