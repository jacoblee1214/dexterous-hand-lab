from dataclasses import replace
from io import BytesIO
import json

import numpy as np
import pytest
from PIL import Image


mujoco = pytest.importorskip("mujoco")

from backend.dashboard_server import DashboardSimulation
from robot_data.common import CommonRobotState
from robot_data.mujoco_provider import MuJoCoRobotDataProvider
from robot_data.recording import CommonStateRecorder
from robot_data.replay_provider import ReplayRobotDataProvider
from vision.baseline import (
    describe_unknown_radius,
    fit_known_radius_sphere,
    segment_blue_sphere,
    sphere_silhouette_mask,
)
from vision.calibration import MUJOCO_TO_CV, calibration_from_mujoco, load_camera_config
from vision.research_camera import ResearchRGBCamera


@pytest.fixture
def research_camera():
    provider = MuJoCoRobotDataProvider()
    engine = DashboardSimulation(provider=provider)
    camera = ResearchRGBCamera(provider)
    try:
        yield provider, engine, camera
    finally:
        camera.close()
        engine.close()


def test_intrinsics_and_extrinsics_are_derived_from_actual_mujoco_camera(research_camera):
    provider, _, camera = research_camera
    camera.render_clean_rgb(provider.read_common_state())
    calibration = camera.calibration()
    config = load_camera_config()
    camera_id = camera.model.camera(config.camera_name).id
    assert camera.model.cam_fovy[camera_id] == pytest.approx(config.fovy_degrees)
    np.testing.assert_allclose(camera.data.cam_xpos[camera_id], config.position_world_m)
    assert calibration.intrinsic_matrix[0, 0] == pytest.approx(
        0.5 * config.height / np.tan(np.deg2rad(config.fovy_degrees) / 2)
    )
    np.testing.assert_allclose(
        calibration.world_from_camera_cv @ calibration.camera_cv_from_world,
        np.eye(4), atol=1e-12,
    )
    assert calibration.to_dict()["coordinate_convention"]["mujoco_to_cv_rotation"] == MUJOCO_TO_CV.tolist()


def test_world_camera_projection_and_coordinate_convention_round_trip(research_camera):
    _, engine, camera = research_camera
    camera.render_clean_rgb(engine.current_common_state)
    calibration = camera.calibration()
    points_cv = np.array([[0.0, 0.0, 0.30], [0.03, -0.02, 0.45]])
    points_world = calibration.camera_to_world(points_cv)
    pixels, depth = calibration.project_world(points_world)
    np.testing.assert_allclose(pixels[0], [calibration.intrinsic_matrix[0, 2], calibration.intrinsic_matrix[1, 2]], atol=1e-10)
    np.testing.assert_allclose(depth, points_cv[:, 2], atol=1e-12)
    expected_second = [
        calibration.intrinsic_matrix[0, 2] + calibration.intrinsic_matrix[0, 0] * 0.03 / 0.45,
        calibration.intrinsic_matrix[1, 2] + calibration.intrinsic_matrix[1, 1] * -0.02 / 0.45,
    ]
    np.testing.assert_allclose(pixels[1], expected_second, atol=1e-10)
    # MuJoCo forward -Z becomes computer-vision forward +Z.
    np.testing.assert_allclose(MUJOCO_TO_CV @ [0, 0, -1], [0, 0, 1])


def test_clean_rgb_is_independent_of_debug_sensor_activity(research_camera):
    _, engine, camera = research_camera
    clean = camera.render_clean_rgb(engine.current_common_state)
    common = engine.current_common_state
    channels = list(common.tactile_channels)
    channels[0] = replace(channels[0], active=True)
    changed = camera.render_clean_rgb(replace(common, tactile_channels=tuple(channels)))
    np.testing.assert_array_equal(clean, changed)
    assert clean.shape == (480, 640, 3) and clean.dtype == np.uint8


def test_synthetic_perspective_silhouette_recovers_known_radius_center(research_camera):
    _, engine, camera = research_camera
    camera.render_clean_rgb(engine.current_common_state)
    calibration = camera.calibration()
    true_center = np.array([0.055, -0.025, 0.52])
    radius = 0.04
    mask = sphere_silhouette_mask(calibration, true_center, radius)
    rgb = np.zeros((calibration.config.height, calibration.config.width, 3), dtype=np.uint8)
    rgb[mask] = [40, 130, 245]
    segmentation = segment_blue_sphere(rgb, calibration.config.segmentation)
    result = fit_known_radius_sphere(segmentation, calibration, radius)
    assert result["status"] == "KNOWN_RADIUS_CENTER_ESTIMATED"
    np.testing.assert_allclose(result["estimated_center_camera_cv_m"], true_center, atol=8e-4)
    assert result["radius_m"] == radius
    assert result["radius_label"] == "provided_known_radius_prior"
    assert result["silhouette_residual_pixels"] < 0.6


def test_unknown_radius_never_invents_metric_scale(research_camera):
    _, engine, camera = research_camera
    rgb = camera.render_clean_rgb(engine.current_common_state)
    segmentation = segment_blue_sphere(rgb, camera.config.segmentation)
    result = describe_unknown_radius(segmentation, camera.calibration())
    assert result["status"] == "SCALE_AMBIGUOUS"
    assert result["radius_m"] is None
    assert result["center_depth_m"] is None
    assert result["radius_label"] == "not_estimated"


def test_capture_preserves_timestamp_and_ground_truth_is_evaluation_only(research_camera):
    provider, engine, camera = research_camera
    capture = camera.capture(engine)
    assert capture.observation.timestamp == engine.current_common_state.timestamp
    assert not capture.observation.depth_available
    assert capture.observation.depth_reference is None
    assert capture.vision_result["input_boundary"] == "clean_rgb_only"
    assert "ground_truth" not in json.dumps(capture.vision_result).lower()
    assert capture.evaluation["boundary"].startswith("evaluation_only")
    assert capture.evaluation["radius_error_m"] is None
    provider.publish_camera_observation(capture.observation)
    state = provider.read_common_state()
    assert state.camera.timestamp == capture.observation.timestamp


def test_record_and_replay_preserve_rgb_reference_and_acquisition_timestamp(research_camera, tmp_path):
    provider, engine, camera = research_camera
    capture = camera.capture(engine)
    provider.publish_camera_observation(capture.observation)
    state = provider.read_common_state()
    path = tmp_path / "rgb-replay.jsonl"
    with CommonStateRecorder(path, provider.description) as recorder:
        recorder.record(state)
    replay = ReplayRobotDataProvider(path)
    restored = CommonRobotState.from_dict(replay.read_common_state().to_dict())
    assert restored.camera.to_dict() == state.camera.to_dict()
    assert restored.camera.rgb_reference == "/research-rgb/latest.jpg"
    assert restored.camera.timestamp == state.camera.timestamp


def test_replay_research_camera_decodes_archived_rgb_instead_of_rerendering(
    research_camera, tmp_path
):
    provider, engine, camera = research_camera
    capture = camera.capture(engine)
    archive = tmp_path / "frames" / "rgb.jpg"
    archive.parent.mkdir()
    archive.write_bytes(capture.frame.jpeg)
    observation = replace(capture.observation, rgb_reference="frames/rgb.jpg")
    provider.publish_camera_observation(observation)
    log = tmp_path / "observations.jsonl"
    with CommonStateRecorder(log, provider.description) as recorder:
        recorder.record(provider.read_common_state())
    replay = ReplayRobotDataProvider(log)
    replay_camera = ResearchRGBCamera(replay)
    try:
        actual = replay_camera.render_clean_rgb(replay.read_common_state())
    finally:
        replay_camera.close()
    expected = np.asarray(Image.open(BytesIO(capture.frame.jpeg)).convert("RGB"))
    np.testing.assert_array_equal(actual, expected)
    assert replay.read_common_state().camera.timestamp == observation.timestamp


def test_estimated_contact_projection_uses_same_world_camera_chain(research_camera):
    _, engine, camera = research_camera
    camera.render_clean_rgb(engine.current_common_state)
    calibration = camera.calibration()
    center_world = calibration.camera_to_world(np.array([0.0, 0.0, 0.4]))
    pixels, depth = calibration.project_world(center_world)
    np.testing.assert_allclose(
        pixels[0],
        [calibration.intrinsic_matrix[0, 2], calibration.intrinsic_matrix[1, 2]],
        atol=1e-10,
    )
    assert depth[0] == pytest.approx(0.4)
