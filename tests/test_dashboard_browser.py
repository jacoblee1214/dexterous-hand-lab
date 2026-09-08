"""Opt-in real Chromium/HTTP/WebSocket checks, not manual visual acceptance.

RUN_BROWSER_TESTS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_dashboard_browser.py
Requires: pip install playwright; python -m playwright install chromium
"""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

import pytest


pytestmark = pytest.mark.skipif(os.environ.get("RUN_BROWSER_TESTS") != "1", reason="Opt-in Chromium validation")
ROOT = Path(__file__).parents[1]
BUILD_ID = "rgb-baseline-20260908.1"


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextmanager
def dashboard(source, tmp_path, extra_args=()):
    http_port, ws_port = free_port(), free_port()
    while ws_port == http_port:
        ws_port = free_port()
    with (tmp_path / f"{source}.log").open("w") as output:
        process = subprocess.Popen([
            sys.executable, "-m", "backend.dashboard_server", "--source", source,
            "--http-port", str(http_port), "--websocket-port", str(ws_port),
        ] + list(extra_args), cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 15
            while True:
                try:
                    with urlopen(f"http://127.0.0.1:{http_port}/", timeout=.3):
                        break
                except OSError:
                    if process.poll() is not None or time.monotonic() > deadline:
                        pytest.fail((tmp_path / f"{source}.log").read_text())
                    time.sleep(.1)
            yield f"http://127.0.0.1:{http_port}/?wsPort={ws_port}", process
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def capture(page, name, tmp_path):
    output = Path(os.environ.get("DASHBOARD_SCREENSHOT_DIR", str(tmp_path)))
    output.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(output / name), full_page=True)


def test_actual_browser_sphere_controls_four_panels_and_disconnect(tmp_path):
    from playwright.sync_api import sync_playwright

    with dashboard("simulation", tmp_path) as (url, process), sync_playwright() as driver:
        browser = driver.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1100})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url)
        page.wait_for_function("state?.snapshot.valid && document.querySelectorAll('#sensor-table tbody tr').length === 18")
        page.wait_for_function(
            "state?.render.status === 'STREAMING' && document.querySelector('#mujoco-render').naturalWidth === 640",
            timeout=30000,
        )
        assert page.locator("#render-mode").input_value() == "mujoco"
        assert page.locator("#build-id").inner_text() == f"UI {BUILD_ID} · API {BUILD_ID}"
        assert "mismatch" not in (page.locator("#build-id").get_attribute("class") or "")
        root_response = page.request.get(url)
        assert root_response.headers["x-dashboard-build"] == BUILD_ID
        assert "no-store" in root_response.headers["cache-control"]
        assert page.locator("#mujoco-render").is_visible()
        assert page.evaluate("simulationView === null")
        page.locator(".joint-controls").evaluate("element => element.open = true")
        assert page.locator(".joint-target").count() == 20
        assert page.locator("#joint-diagnostics thead th").count() == 8
        assert page.locator("#execute-taught-grasp").is_visible()
        page.locator("#teach-mode-toggle").click()
        page.wait_for_function("state.simulation.grasp.teach.enabled")
        assert "TEACH" in page.locator("#teach-status").inner_text()
        assert page.locator('[data-command="grasp"]').is_disabled()
        assert page.locator("#execute-taught-grasp").is_enabled()
        page.locator("#teach-mode-toggle").click()
        page.wait_for_function("!state.simulation.grasp.teach.enabled")
        joint_slider = page.locator(".joint-target").filter(has_text="joint_10").locator("input")
        joint_slider.evaluate("element => { element.value = '0.2'; element.dispatchEvent(new Event('change')); }")
        slider_target = float(joint_slider.input_value())
        page.wait_for_function(
            "target => Math.abs(state.joints.find(j => j.name === 'joint_10').target_rad-target) < 1e-9",
            arg=slider_target,
        )
        assert page.locator(".quadrant").count() == 4
        assert page.locator("#data-source").inner_text() == "SIMULATION"
        page.locator('[data-radius="0.04"]').click()
        page.wait_for_function("state.simulation.object.radius_m === .04")
        page.locator('[data-command="start_accumulation"]').click()
        page.locator('[data-command="grasp"]').click()
        grasp_start = page.evaluate("state.simulation.time_seconds")
        contact_handle = page.wait_for_function(
            "state.current_estimated_contacts.length ? structuredClone({"
            "timestamp:state.snapshot.timestamp, contactPoints:state.current_estimated_contacts, "
            "activeSensorIds:state.sensors.filter(sensor => sensor.active).map(sensor => sensor.sensor_id)}) : null",
            timeout=20000,
        )
        contact_frame = contact_handle.json_value()
        contact_sample = contact_frame["contactPoints"]
        assert set(contact["sensor_id"] for contact in contact_sample) <= set(contact_frame["activeSensorIds"])
        selected = contact_sample[0]["sensor_id"]
        page.locator("#sensor-table tbody tr").filter(has_text=selected).click()
        page.wait_for_function("id => state.selected_sensor_id === id", arg=selected)
        page.wait_for_function("start => state.accumulation.accepted_point_count > 0 && state.simulation.grasp.stage === 4 && state.simulation.time_seconds >= start + 5.2", arg=grasp_start, timeout=20000)
        page.locator('[data-command="fit_sphere"]').click()
        page.wait_for_function("state.sphere_reconstruction.status !== 'NOT_FITTED'")
        payload = page.evaluate("state")
        assert len(payload["joints"]) == 20
        assert len(payload["sensors"]) == 18
        assert payload["accumulated_points"]
        assert contact_sample
        assert any(sample["scalar_sensor_value_n"] > 0 for series in payload["time_series"]["series"] for sample in series["samples"])
        timestamps = page.evaluate("Array.from(document.querySelectorAll('.quadrant'), p => p.dataset.snapshotTimestamp)")
        assert len(set(timestamps)) == 1
        # Exercise explicit backend camera orbit and zoom. Neither command can
        # write robot qpos or actuator state.
        image = page.locator("#mujoco-render")
        canvas = image.bounding_box()
        page.mouse.move(canvas["x"] + 100, canvas["y"] + 100)
        page.mouse.down()
        page.mouse.move(canvas["x"] + 160, canvas["y"] + 130)
        page.mouse.up()
        page.mouse.wheel(0, -120)
        page.wait_for_function("dashboardDiagnostics.lastAcknowledgement?.command === 'render_camera'")
        camera_sequence = page.evaluate("state.render.frame_sequence")
        page.wait_for_function("sequence => state.render.frame_sequence > sequence", arg=camera_sequence)
        from io import BytesIO
        from PIL import Image
        import numpy as np
        pixels = np.asarray(Image.open(BytesIO(image.screenshot())))
        assert pixels.std() > 20
        # The fixed research camera is a distinct, clean stream. It remains in
        # the same quadrant and keeps all three tactile panels live.
        page.locator("#render-mode").select_option("research")
        page.wait_for_function(
            "state?.research_rgb.status === 'STREAMING' && "
            "document.querySelector('#research-rgb').naturalWidth === 640 && "
            "state?.vision_only?.segmentation?.status",
            timeout=30000,
        )
        assert page.locator("#research-rgb").is_visible()
        assert page.locator("#mujoco-render").is_hidden()
        assert "Vision-only RGB sphere baseline" in page.locator("#vision-only-metrics").inner_text()
        assert page.evaluate("state.camera.depth_available") is False
        assert page.evaluate("state.camera.timestamp === state.research_rgb.simulation_timestamp")
        base = url.split('/?')[0]
        latest = page.request.get(base + "/research-rgb/latest.jpg")
        calibration = page.request.get(base + "/research-rgb/calibration.json")
        assert latest.status == 200 and latest.headers["content-type"] == "image/jpeg"
        assert "x-acquisition-timestamp" in latest.headers
        assert calibration.status == 200
        assert calibration.json()["calibration_version"] == "research-rgb-camera-v1"
        page.locator("#vision-only-metrics").scroll_into_view_if_needed()
        page.locator("#simulation-toggle").click()
        page.wait_for_function("dashboardDiagnostics.lastAcknowledgement?.command === 'pause_simulation'")
        page.wait_for_function("!state.simulation.running")
        capture(page, "sphere-dashboard.png", tmp_path)
        # Feed status-only hardware fixtures through the same browser schema.
        # These validate presentation, not a real hardware connection.
        for status in ("CONNECTING", "STALE", "ERROR"):
            page.evaluate("""status => {
                socket.onmessage = () => {};
                state = {...state, source: {kind: 'hardware', display_name: 'REAL ROBOT'},
                         connection: {state: status, detail: 'Browser contract fixture', read_only: true},
                         available_commands: [], snapshot: {...state.snapshot, valid: false}};
                pendingState = state;
                lastStateReceived = performance.now(); render();
            }""", status)
            assert page.locator("#connection").inner_text() == status
            assert page.locator('[data-command="grasp"]').is_disabled()
            assert page.locator("#simulation-toggle").is_disabled()
            assert page.locator("#sensor-table tbody tr.active").count() == 0
            assert "unavailable" in page.locator("#stream-status").inner_text()
        page.evaluate("lastStateReceived = performance.now() - 2000; renderStreamStatus()")
        assert page.locator("#connection").inner_text() == "STALE"
        process.terminate()
        process.wait(timeout=5)
        page.wait_for_function("document.querySelector('#connection').textContent === 'DISCONNECTED'")
        capture(page, "disconnected-dashboard.png", tmp_path)
        assert not errors
        browser.close()


def test_actual_browser_unconfigured_hardware_is_read_only_and_empty(tmp_path):
    from playwright.sync_api import sync_playwright

    with dashboard("hardware", tmp_path) as (url, _), sync_playwright() as driver:
        browser = driver.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1100})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url)
        page.wait_for_function("state?.source.kind === 'hardware'")
        assert page.locator("#data-source").inner_text() == "REAL ROBOT"
        assert page.locator("#connection").inner_text() == "DISCONNECTED"
        assert page.locator("#sensor-table tbody tr").count() == 0
        assert page.locator('[data-command="grasp"]').is_disabled()
        assert page.locator('[data-command="reset"]').is_disabled()
        assert page.evaluate("state.snapshot.timestamp") is None
        capture(page, "hardware-waiting.png", tmp_path)
        assert not errors
        browser.close()


def test_replay_stl_meshes_match_named_poses_and_sensor_mounts_in_browser(tmp_path):
    from playwright.sync_api import sync_playwright
    from robot_data.mujoco_provider import MuJoCoRobotDataProvider
    from robot_data.recording import CommonStateRecorder
    from backend.visual_assets import VisualAssets
    import hashlib

    provider = MuJoCoRobotDataProvider()
    path = tmp_path / "mesh-replay.jsonl"
    with CommonStateRecorder(path, provider.description) as recorder:
        recorder.record(provider.read_common_state())
        provider.send_joint_targets({"joint_10": .5})
        for _ in range(100): provider.step()
        recorder.record(provider.read_common_state())
    assets = VisualAssets(provider.description)
    with dashboard("replay", tmp_path, ("--replay-log", str(path))) as (url, _), sync_playwright() as driver:
        browser = driver.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1100})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url)
        page.wait_for_function(
            "state?.connection.replay_finished && state.render.status === 'STREAMING' && document.querySelector('#mujoco-render').naturalWidth === 640",
            timeout=30000,
        )
        assert page.locator("#render-mode").input_value() == "mujoco"
        assert page.locator("#mujoco-render").is_visible()
        page.locator("#render-mode").select_option("mesh")
        page.wait_for_function("simulationView?.loaded", timeout=30000)
        assert page.evaluate("simulationView.meshes.every(m => m.geometry.attributes.position.count > 100)")
        assert page.evaluate("simulationView.meshes.length") == 21
        assert page.evaluate("simulationView.objectSphere.visible")
        # Every original STL request resolves to the same bytes as the URDF asset.
        for asset_url, local_file in assets.files.items():
            response = page.request.get(url.split('/?')[0] + asset_url)
            assert response.status == 200
            assert hashlib.sha256(response.body()).digest() == hashlib.sha256(local_file.read_bytes()).digest()
        assert page.request.get(url.split('/?')[0] + '/robot-assets/missing.stl').status == 404
        error = page.evaluate("""() => {
            simulationView.scene.updateMatrixWorld(true);
            let error = 0;
            for (const pose of state.hand.links) {
                const group = simulationView.links.get(pose.name);
                const position = group.position.toArray();
                const q = group.quaternion;
                [...position, q.w, q.x, q.y, q.z].forEach((v, i) => {
                    error = Math.max(error, Math.abs(v - [...pose.position_xyz, ...pose.quaternion_wxyz][i]));
                });
            }
            for (const sensor of state.sensors) {
                const patch = simulationView.sensors.get(sensor.sensor_id);
                if (patch.parent.name !== sensor.parent_link) throw new Error('Wrong sensor parent');
                const elements = patch.matrixWorld.elements;
                sensor.position_world_xyz.forEach((v, i) => error = Math.max(error, Math.abs(v-elements[12+i])));
                sensor.sensing_direction_world_xyz.forEach((v, i) => error = Math.max(error, Math.abs(v-elements[8+i])));
            }
            return error;
        }""")
        assert error < 1e-10
        capture(page, "replay-mesh.png", tmp_path)
        # Isolated presentation fixture: test all overlay types without claiming
        # the current physical grasp yielded a valid reconstructed sphere.
        page.evaluate("""() => {
            socket.onmessage = () => {};
            const fixture = structuredClone(state);
            fixture.sensors[0].active = true;
            fixture.current_estimated_contacts = [{position_xyz: fixture.sensors[0].position_world_xyz}];
            fixture.accumulated_points = [{position_xyz: fixture.sensors[1].position_world_xyz}];
            fixture.sphere_reconstruction = {...fixture.sphere_reconstruction,
                status: 'VALID_RECONSTRUCTION', show_estimated_sphere: true,
                estimated_center_xyz: fixture.simulation.object.center_xyz,
                estimated_radius_m: fixture.simulation.object.radius_m * .95};
            state = fixture; pendingState = fixture; lastStateReceived = performance.now(); render();
        }""")
        assert page.evaluate("simulationView.estimatedSphere.visible")
        assert page.evaluate("simulationView.pointLayers.get('contacts').count") == 1
        assert page.evaluate("simulationView.pointLayers.get('accumulated').count") == 1
        assert page.evaluate("simulationView.sensors.get(state.sensors[0].sensor_id).material.color.getHexString()") == 'ff4b43'
        assert page.locator('#sensor-table tbody tr.active').count() == 1
        capture(page, "mesh-overlay-fixture.png", tmp_path)
        page.evaluate("lastStateReceived = performance.now()-2000; renderStreamStatus()")
        assert page.evaluate("simulationView.sensors.get(state.sensors[0].sensor_id).material.color.getHexString()") == '35d7e5'
        assert not errors
        browser.close()
