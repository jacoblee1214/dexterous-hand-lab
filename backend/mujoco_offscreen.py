"""MuJoCo-owned offscreen RGB rendering and a bounded latest-frame buffer."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from io import BytesIO
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class RenderedFrame:
    sequence: int
    simulation_timestamp: float
    rendering_timestamp: float
    source: str
    jpeg: bytes


class LatestFrameBuffer:
    """One-slot frame exchange: consumers never build latency or a queue."""

    def __init__(
        self,
        *,
        mode: str = "MuJoCo Render",
        stream_url: str = "/mujoco-render/stream.mjpg",
    ) -> None:
        self._condition = threading.Condition()
        self._frame: RenderedFrame | None = None
        self._error: str | None = None
        self.mode = mode
        self.stream_url = stream_url

    def publish(self, frame: RenderedFrame) -> None:
        with self._condition:
            self._frame, self._error = frame, None
            self._condition.notify_all()

    def fail(self, error: Exception | str) -> None:
        with self._condition:
            self._error = str(error)
            self._condition.notify_all()

    def wait_after(self, sequence: int, timeout: float = 5.0) -> RenderedFrame | None:
        with self._condition:
            self._condition.wait_for(
                lambda: self._error is not None or (
                    self._frame is not None and self._frame.sequence > sequence
                ),
                timeout=timeout,
            )
            if self._frame is not None and self._frame.sequence > sequence:
                return self._frame
            return None

    @property
    def error(self) -> str | None:
        with self._condition:
            return self._error

    @property
    def latest(self) -> RenderedFrame | None:
        with self._condition:
            return self._frame

    def metadata(self) -> dict:
        with self._condition:
            frame = self._frame
            return {
                "mode": self.mode,
                "status": "ERROR" if self._error else "STREAMING" if frame else "STARTING",
                "error": self._error,
                "frame_sequence": frame.sequence if frame else None,
                "simulation_timestamp": frame.simulation_timestamp if frame else None,
                "rendering_timestamp": frame.rendering_timestamp if frame else None,
                "source": frame.source if frame else None,
                "stream_url": self.stream_url,
            }


class MuJoCoOffscreenRenderer:
    """Render the authoritative model from a synchronized dedicated MjData."""

    def __init__(self, provider, *, width: int = 640, height: int = 480,
                 jpeg_quality: int = 85) -> None:
        if width < 64 or height < 64 or width > 1920 or height > 1080:
            raise ValueError("Offscreen resolution must be between 64x64 and 1920x1080")
        if not 20 <= jpeg_quality <= 95:
            raise ValueError("JPEG quality must be between 20 and 95")
        self.width, self.height, self.jpeg_quality = width, height, jpeg_quality
        self.source = provider.description.source
        self._owned_simulation = None
        self._owned_objects = None
        if self.source == "simulation":
            simulation = provider.simulation
        elif self.source == "replay":
            # This is a renderer-only model whose qpos is overwritten from every
            # recorded sample. It never steps an independent simulation.
            from simulation.model_builder import HAND_VARIANTS, build_hand_variant
            from simulation.mujoco_sim import HandSimulation, ObjectController
            variant = HAND_VARIANTS[provider.description.handedness]
            build_hand_variant(provider.description.handedness)
            simulation = HandSimulation(variant.output)
            self._owned_simulation = simulation
            self._owned_objects = ObjectController(simulation, variant.sphere_position)
            self._owned_objects.activate("sphere")
        else:
            raise ValueError("MuJoCo Render is available only for simulation and replay")
        self.mujoco, self.model = simulation.mujoco, simulation.model
        self.data = self.mujoco.MjData(self.model)
        self._live_data = simulation.data if self.source == "simulation" else None
        self._render_simulation = SimpleNamespace(
            mujoco=self.mujoco, model=self.model, data=self.data
        )
        self.renderer = self.mujoco.Renderer(self.model, height=height, width=width)
        self.camera = self.mujoco.MjvCamera()
        self.mujoco.mjv_defaultCamera(self.camera)
        self._reset_camera()
        self.scene_option = self.mujoco.MjvOption()
        self.mujoco.mjv_defaultOption(self.scene_option)
        self.scene_option.sitegroup[3] = False
        self.scene_option.sitegroup[4] = False
        self.scene_option.geomgroup[5] = False
        self.sequence = 0
        self.show_research_camera_frustum = False

    def _reset_camera(self) -> None:
        self.camera.type = self.mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.fixedcamid = -1
        self.camera.lookat[:] = [0.0, 0.0, 0.095]
        self.camera.distance = 0.34
        self.camera.azimuth = 135.0 if self.source != "hardware" else -135.0
        self.camera.elevation = -25.0

    def camera_command(self, parameters: dict) -> None:
        action = str(parameters.get("action", ""))
        if action == "toggle_research_camera_frustum":
            self.show_research_camera_frustum = bool(parameters.get("enabled", False))
            return
        if action == "reset":
            self._reset_camera()
            return
        if action == "orbit":
            azimuth = float(parameters.get("azimuth_delta_deg", 0.0))
            elevation = float(parameters.get("elevation_delta_deg", 0.0))
            if not np.isfinite([azimuth, elevation]).all():
                raise ValueError("Camera orbit deltas must be finite")
            self.camera.azimuth += float(np.clip(azimuth, -90, 90))
            self.camera.elevation = float(np.clip(self.camera.elevation + elevation, -89, 89))
            return
        if action == "zoom":
            factor = float(parameters.get("factor", 1.0))
            if not np.isfinite(factor) or not 0.25 <= factor <= 4.0:
                raise ValueError("Camera zoom factor must be finite and within [0.25, 4]")
            self.camera.distance = float(np.clip(self.camera.distance * factor, 0.04, 2.0))
            return
        if action == "pan":
            horizontal = float(parameters.get("horizontal", 0.0))
            vertical = float(parameters.get("vertical", 0.0))
            if not np.isfinite([horizontal, vertical]).all():
                raise ValueError("Camera pan deltas must be finite")
            horizontal, vertical = np.clip([horizontal, vertical], -1.0, 1.0)
            azimuth = np.deg2rad(self.camera.azimuth)
            scale = self.camera.distance * 0.35
            right = np.array([np.cos(azimuth), np.sin(azimuth), 0.0])
            self.camera.lookat[:] += scale * horizontal * right
            self.camera.lookat[2] += scale * vertical
            return
        raise ValueError(
            "Camera action must be orbit, zoom, pan, reset, or "
            "toggle_research_camera_frustum"
        )

    def _append_line(self, start, end, rgba, width: float = 0.0007) -> None:
        scene = self.renderer.scene
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        # Fully initialize the user geom before setting its connector endpoints;
        # leaving fields from a previous scene slot undefined can crash GL renderers.
        self.mujoco.mjv_initGeom(
            geom, self.mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3), np.zeros(3), np.eye(3).reshape(-1),
            np.asarray(rgba, dtype=float),
        )
        self.mujoco.mjv_connector(
            geom, self.mujoco.mjtGeom.mjGEOM_CAPSULE, width,
            np.asarray(start, dtype=float), np.asarray(end, dtype=float),
        )
        scene.ngeom += 1

    def _append_research_camera_debug_geometry(self) -> None:
        """Draw calibrated optical frame/frustum only in this debug renderer."""
        if not self.show_research_camera_frustum:
            return
        from vision.calibration import calibration_from_mujoco, load_camera_config

        calibration = calibration_from_mujoco(self.model, self.data, load_camera_config())
        origin = calibration.world_from_camera_cv[:3, 3]
        rotation = calibration.world_from_camera_cv[:3, :3]
        scene = self.renderer.scene
        if scene.ngeom < scene.maxgeom:
            self.mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom], self.mujoco.mjtGeom.mjGEOM_SPHERE,
                np.full(3, 0.005), origin, np.eye(3).reshape(-1),
                np.array([1.0, 0.78, 0.15, 0.9]),
            )
            scene.ngeom += 1
        axis_length = 0.035
        self._append_line(
            origin, origin + rotation[:, 0] * axis_length, [1, 0.1, 0.1, 1], .0012
        )
        self._append_line(
            origin, origin + rotation[:, 1] * axis_length, [0.1, 1, 0.1, 1], .0012
        )
        self._append_line(
            origin, origin + rotation[:, 2] * axis_length, [0.1, 0.45, 1, 1], .0012
        )
        width, height = calibration.config.width, calibration.config.height
        pixels = np.asarray(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=float,
        )
        rays = calibration.pixel_rays_cv(pixels, normalize=False)
        depth = 0.12  # Debug drawing length only; ray angles remain calibration-derived.
        corners = calibration.camera_to_world(rays * depth)
        for corner in corners:
            self._append_line(origin, corner, [1, 0.78, 0.15, 0.8], .0006)
        for start, end in zip(corners, np.roll(corners, -1, axis=0)):
            self._append_line(start, end, [1, 0.78, 0.15, 0.8], .0006)

    def _sync_data(self, common_state) -> None:
        if self.source == "simulation":
            # Called on the same asyncio thread as physics. Copying precedes all
            # rendering, so the renderer never reads live mjData concurrently.
            self.mujoco.mj_copyData(self.data, self.model, self._live_data)
            return
        if common_state is None:
            return
        self.data.qvel[:] = 0
        for name, value in common_state.joint_positions.items():
            joint_id = self.mujoco.mj_name2id(
                self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name
            )
            self.data.qpos[self.model.jnt_qposadr[joint_id]] = value
        object_state = dict(common_state.object_state or {})
        if object_state.get("center_xyz"):
            joint_id = self.mujoco.mj_name2id(
                self.model, self.mujoco.mjtObj.mjOBJ_JOINT, "sphere_object_free"
            )
            address = self.model.jnt_qposadr[joint_id]
            self.data.qpos[address:address + 3] = object_state["center_xyz"]
            self.data.qpos[address + 3:address + 7] = [1, 0, 0, 0]
            geom_id = self.mujoco.mj_name2id(
                self.model, self.mujoco.mjtObj.mjOBJ_GEOM, "sphere_object_geom"
            )
            self.model.geom_size[geom_id, 0] = float(object_state.get("radius_m", .03))
        self.data.time = common_state.timestamp
        self.mujoco.mj_forward(self.model, self.data)

    def render_rgb(self, engine) -> np.ndarray:
        from simulation.mujoco_sim import (
            OverlayState,
            update_reconstruction_overlays,
            update_sensor_overlays,
        )
        self._sync_data(engine.current_common_state)
        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.scene_option)
        overlay = OverlayState(
            show_surfaces=True,
            show_centers=False,
            show_directions=False,
            show_ids=False,
            show_frames=False,
            show_active_sensors=True,
            show_estimated_contacts=True,
            show_accumulated_points=True,
            show_ground_truth_contacts=False,
            show_estimated_sphere=True,
        )
        active = {
            channel.sensor_id: SimpleNamespace(active=channel.active)
            for channel in (engine.current_common_state.tactile_channels
                            if engine.current_common_state else ())
        }
        update_sensor_overlays(
            self._render_simulation,
            self.renderer.scene,
            engine.mounts,
            overlay,
            reset_scene=False,
            live_mount_transforms=True,
            readings=active,
        )
        update_reconstruction_overlays(
            self._render_simulation,
            self.renderer.scene,
            overlay,
            engine.current_estimates,
            tuple(engine.point_buffer.points),
            [],
        )
        self._append_research_camera_debug_geometry()
        result = engine.sphere_session.result
        if (engine.show_estimated_sphere and result is not None
                and result.status.value == "VALID_RECONSTRUCTION"
                and result.estimated_center_xyz is not None
                and self.renderer.scene.ngeom < self.renderer.scene.maxgeom):
            geom = self.renderer.scene.geoms[self.renderer.scene.ngeom]
            self.mujoco.mjv_initGeom(
                geom,
                self.mujoco.mjtGeom.mjGEOM_SPHERE,
                np.full(3, result.estimated_radius),
                np.asarray(result.estimated_center_xyz),
                np.eye(3).reshape(-1),
                np.array([0.05, 0.75, 1.0, 0.24]),
            )
            self.renderer.scene.ngeom += 1
        return self.renderer.render().copy()

    def render_frame(self, engine) -> RenderedFrame:
        rgb = self.render_rgb(engine)
        output = BytesIO()
        Image.fromarray(rgb).save(output, format="JPEG", quality=self.jpeg_quality)
        self.sequence += 1
        timestamp = float(engine.current_common_state.timestamp) if engine.current_common_state else 0.0
        return RenderedFrame(
            self.sequence, timestamp, time.time(), self.source, output.getvalue()
        )

    def close(self) -> None:
        self.renderer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone MuJoCo offscreen rendering proof")
    parser.add_argument("--output", type=Path, default=Path("experiments/mujoco_offscreen_proof.png"))
    parser.add_argument("--backend", choices=("egl", "glfw", "osmesa"), default="egl")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()
    os.environ["MUJOCO_GL"] = args.backend
    from backend.dashboard_server import DashboardSimulation
    from robot_data.mujoco_provider import MuJoCoRobotDataProvider
    provider = MuJoCoRobotDataProvider()
    engine = DashboardSimulation(provider=provider)
    renderer = MuJoCoOffscreenRenderer(provider, width=args.width, height=args.height)
    try:
        rgb = renderer.render_rgb(engine)
        visual_names = [
            renderer.mujoco.mj_id2name(renderer.model, renderer.mujoco.mjtObj.mjOBJ_GEOM, index)
            for index in range(renderer.model.ngeom)
        ]
        visual_count = sum(bool(name and name.startswith("visual_")) for name in visual_names)
        if visual_count < 21 or "sphere_object_geom" not in visual_names:
            raise RuntimeError("Expected hand/sphere visual geometry is missing from the MuJoCo model")
        if float(rgb.std()) < 5 or np.unique(rgb.reshape(-1, 3), axis=0).shape[0] < 100:
            raise RuntimeError("Offscreen output appears blank")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(args.output)
        print(f"Saved {args.output}: {args.width}x{args.height}, {visual_count} hand visuals, RGB std={rgb.std():.3f}")
    finally:
        renderer.close()
        engine.close()


if __name__ == "__main__":
    main()
