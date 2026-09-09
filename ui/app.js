"use strict";

const SCHEMA_VERSION = "1.1.0";
const FRONTEND_BUILD_ID = "fusion-explainability-v1-20260909.1";
const colors = ["#35d7e5", "#ff8b5c", "#c478ff", "#55ec89", "#ffd84f", "#4d99ff"];
let state = null;
let socket = null;
let requestSequence = 0;
const sensorRows = new Map();
let pendingState = null;
let stateRenderPending = false;
let lastStateReceived = -Infinity;
const dashboardDiagnostics = {lastCommand: null, lastAcknowledgement: null};
window.dashboardDiagnostics = dashboardDiagnostics;
const robotCommands = new Set([
  "open_hand", "grasp", "reset", "set_joint_target", "set_grasp_preset",
  "reset_reference_pose", "execute_grasp_stage", "save_grasp_preset",
  "load_grasp_preset", "set_experiment_mode", "set_teach_grasp_mode",
  "set_pose_validation_mode", "save_taught_pose", "load_taught_pose",
  "execute_taught_grasp", "stop_taught_grasp"
]);
const jointControls = new Map();
let renderMode = null;
let simulationView = null;
let meshLoadPromise = null;
let fusionMethod = "fusion_finite_patch";

function streamValid() {
  return socket?.readyState === WebSocket.OPEN && performance.now() - lastStateReceived < 1500 && state?.snapshot.valid;
}

function renderStreamStatus() {
  const valid = Boolean(streamValid());
  const transportOpen = socket?.readyState === WebSocket.OPEN;
  const transportFresh = performance.now() - lastStateReceived < 1500;
  const status = !transportOpen ? "DISCONNECTED" : !transportFresh ? "STALE" : state?.connection.state || "CONNECTING";
  document.body.dataset.streamValid = String(valid);
  simulationView?.setLive(valid);
  const indicator = document.querySelector("#connection");
  indicator.textContent = status;
  indicator.className = `connection ${valid ? "online" : "offline"}`;
  const detail = transportFresh ? state?.connection.detail || "" : "No recent dashboard state";
  document.querySelector("#stream-status").textContent = `${state?.source.display_name || "SOURCE"} · ${status} · ${detail}${valid ? "" : " · Current values unavailable; any retained view is historical"}`;
  if (state?.recording.detail) document.querySelector("#stream-status").textContent += ` · ${state.recording.detail}`;
  document.querySelectorAll("[data-command]").forEach(button => {
    button.disabled = !valid || (robotCommands.has(button.dataset.command) && !state?.available_commands.includes(button.dataset.command));
  });
  document.querySelectorAll("[data-radius]").forEach(button => button.disabled = !valid || !state?.available_commands.includes("set_grasp_preset"));
  document.querySelectorAll("[data-grasp-stage]").forEach(button => button.disabled = !valid || !state?.available_commands.includes("execute_grasp_stage"));
  for (const id of ["load-grasp-preset", "save-grasp-preset", "experiment-mode"]) {
    const element = document.querySelector(`#${id}`);
    const required = id === "experiment-mode" ? "set_experiment_mode" : `${id.replace("grasp-", "grasp_").replaceAll("-", "_")}`;
    element.disabled = !valid || !state?.available_commands.includes(required);
  }
  const teachRequirements = {
    "teach-mode-toggle":"set_teach_grasp_mode",
    "pose-validation-mode":"set_pose_validation_mode",
    "load-taught-pose":"load_taught_pose",
    "save-taught-pose":"save_taught_pose",
    "execute-taught-grasp":"execute_taught_grasp",
    "stop-taught-grasp":"stop_taught_grasp",
  };
  for (const [id, required] of Object.entries(teachRequirements)) {
    document.querySelector(`#${id}`).disabled = !valid || !state?.available_commands.includes(required);
  }
  const teach = state?.simulation?.grasp?.teach || {};
  if (!teach.enabled) {
    for (const id of ["pose-validation-mode", "load-taught-pose", "save-taught-pose", "execute-taught-grasp", "stop-taught-grasp"])
      document.querySelector(`#${id}`).disabled = true;
  } else {
    document.querySelector('[data-command="grasp"]').disabled = true;
    document.querySelector("#load-grasp-preset").disabled = true;
    document.querySelector("#save-grasp-preset").disabled = true;
    document.querySelectorAll("[data-grasp-stage]").forEach(button => button.disabled = true);
  }
  if (document.querySelector("#taught-pose-source").value === "actual_measured" && !teach.pose_is_settled)
    document.querySelector("#save-taught-pose").disabled = true;
  document.querySelector("#start-experiment").disabled = !valid || state?.experiment?.active;
  document.querySelector("#finish-experiment").disabled = !valid || !state?.experiment?.active;
  document.querySelector("#repeat-experiment").disabled = !valid || state?.experiment?.active || !state?.experiment?.name;
  document.querySelector("#simulation-toggle").disabled = !transportOpen || state?.source.kind === "hardware";
  document.querySelector("#reset-render-camera").disabled = renderMode !== 'mujoco' || state?.render?.status !== 'STREAMING';
  document.querySelector("#show-research-camera-frustum").disabled =
    renderMode !== 'mujoco' || state?.render?.status !== 'STREAMING';
  if (!valid) {
    document.querySelector("#active-count").textContent = "Current activation unavailable";
    document.querySelectorAll("#sensor-table tbody tr").forEach(row => {
      row.classList.remove("active");
      if (row.cells[2]) row.cells[2].textContent = "UNAVAILABLE";
    });
  }
}
setInterval(renderStreamStatus, 200);

function websocketUrl() {
  const params = new URLSearchParams(location.search);
  const port = params.get("wsPort") || "8765";
  return `${location.protocol === "https:" ? "wss" : "ws"}://${location.hostname}:${port}`;
}

function showToast(text) {
  const toast = document.querySelector("#toast");
  toast.textContent = text;
  toast.classList.add("show");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.remove("show"), 2800);
}

function connect() {
  socket = new WebSocket(websocketUrl());
  const indicator = document.querySelector("#connection");
  socket.onopen = () => { lastStateReceived = -Infinity; renderStreamStatus(); };
  socket.onclose = () => {
    indicator.textContent = "Reconnecting…"; indicator.className = "connection offline";
    renderStreamStatus();
    setTimeout(connect, 1200);
  };
  socket.onerror = () => socket.close();
  socket.onmessage = event => {
    const message = JSON.parse(event.data);
    if (message.message_type === "state") {
      if (message.schema_version !== SCHEMA_VERSION) return showToast(`Unsupported state schema ${message.schema_version}`);
      lastStateReceived = performance.now();
      pendingState = message;
      if (!stateRenderPending) {
        stateRenderPending = true;
        requestAnimationFrame(() => {
          stateRenderPending = false;
          state = pendingState;
          render();
        });
      }
    } else if (message.message_type === "command_error") {
      dashboardDiagnostics.lastAcknowledgement = message;
      showToast(message.error);
    } else if (message.message_type === "command_acknowledgement") {
      dashboardDiagnostics.lastAcknowledgement = message;
      const result = message.result;
      if (result?.taught_pose_name) showToast(`Taught pose ready: ${result.taught_pose_name}`);
      else if (result?.preset_name) showToast(`Preset ready: ${result.preset_name}`);
      else if (result?.last_result?.reconstruction_path) showToast(`Trial saved: ${result.last_result.reconstruction_path}`);
    }
  };
}

function command(name, parameters = {}) {
  if (!socket || socket.readyState !== WebSocket.OPEN) return showToast("Dashboard backend is not connected");
  if (robotCommands.has(name) && (!streamValid() || !state.available_commands.includes(name))) return showToast("Robot control is unavailable for this source");
  const request = {schema_version: SCHEMA_VERSION, command: name, parameters, request_id: ++requestSequence};
  dashboardDiagnostics.lastCommand = request;
  socket.send(JSON.stringify(request));
}

document.querySelectorAll("[data-command]").forEach(button => button.addEventListener("click", () => command(button.dataset.command)));
document.querySelectorAll("[data-radius]").forEach(button => button.addEventListener("click", () => command("set_grasp_preset", {radius_m: Number(button.dataset.radius)})));
document.querySelector("#simulation-toggle").addEventListener("click", () => command(state?.simulation.running ? "pause_simulation" : "start_simulation"));
document.querySelector("#show-ground-truth").addEventListener("change", event => command("show_ground_truth", {enabled: event.target.checked}));
document.querySelector("#show-estimated").addEventListener("change", event => command("show_estimated_sphere", {enabled: event.target.checked}));
document.querySelector("#show-projected-contacts").addEventListener("change", () => state && drawResearchContactOverlay());
document.querySelector("#show-research-camera-frustum").addEventListener("change", event =>
  command("render_camera", {action:"toggle_research_camera_frustum", enabled:event.target.checked}));
document.querySelector("#series-scope").addEventListener("change", event => command("set_time_series_scope", {scope: event.target.value}));
document.querySelector("#experiment-mode").addEventListener("change", event => command("set_experiment_mode", {mode:event.target.value}));
document.querySelectorAll("[data-grasp-stage]").forEach(button => button.addEventListener("click", () => command("execute_grasp_stage", {stage_number:Number(button.dataset.graspStage)})));
document.querySelector("#load-grasp-preset").addEventListener("click", () => command("load_grasp_preset", {preset_name:document.querySelector("#grasp-preset").value}));
document.querySelector("#save-grasp-preset").addEventListener("click", () => {
  const name = document.querySelector("#new-grasp-preset-name").value.trim();
  if (!name) return showToast("Enter a new, explicit preset name");
  command("save_grasp_preset", {preset_name:name});
});
document.querySelector("#teach-mode-toggle").addEventListener("click", () =>
  command("set_teach_grasp_mode", {enabled:!Boolean(state?.simulation?.grasp?.teach?.enabled)}));
document.querySelector("#pose-validation-mode").addEventListener("change", event =>
  command("set_pose_validation_mode", {mode:event.target.value}));
document.querySelector("#load-taught-pose").addEventListener("click", () => {
  const poseName = document.querySelector("#taught-pose").value;
  if (!poseName) return showToast("Save or select a taught pose first");
  command("load_taught_pose", {pose_name:poseName});
});
document.querySelector("#save-taught-pose").addEventListener("click", () => {
  const poseName = document.querySelector("#new-taught-pose-name").value.trim();
  if (!poseName) return showToast("Enter a new, explicit taught pose name");
  command("save_taught_pose", {
    pose_name:poseName,
    target_source:document.querySelector("#taught-pose-source").value,
    waypoint_role:document.querySelector("#taught-waypoint-role").value,
  });
});
document.querySelector("#execute-taught-grasp").addEventListener("click", () => {
  const listed = document.querySelector("#taught-waypoint-sequence").value
    .split(",").map(item => item.trim()).filter(Boolean);
  const selected = document.querySelector("#taught-pose").value;
  command("execute_taught_grasp", {
    pose_names:listed.length ? listed : selected ? [selected] : [],
    duration_seconds:Number(document.querySelector("#taught-duration").value),
    tolerance_rad:Number(document.querySelector("#taught-tolerance").value),
  });
});
document.querySelector("#stop-taught-grasp").addEventListener("click", () => command("stop_taught_grasp"));
document.querySelector("#taught-pose-source").addEventListener("change", renderStreamStatus);
document.querySelector("#start-experiment").addEventListener("click", () => {
  const name = document.querySelector("#experiment-name").value.trim();
  if (!name) return showToast("Enter an explicit experiment name");
  command("start_experiment", {experiment_name:name});
});
document.querySelector("#finish-experiment").addEventListener("click", () => command("finish_experiment"));
document.querySelector("#repeat-experiment").addEventListener("click", () => command("repeat_experiment"));

function ensureMeshView() {
  if (meshLoadPromise) return meshLoadPromise;
  document.querySelector('#mesh-status').textContent = 'Loading original STL meshes…';
  meshLoadPromise = import(`./mesh_view.js?v=${FRONTEND_BUILD_ID}`).then(({MeshDigitalTwin}) => {
    simulationView = new MeshDigitalTwin(document.querySelector('#simulation-canvas'));
    if (state) simulationView.draw(state);
    return simulationView;
  }).catch(error => {
    document.querySelector('#mesh-status').textContent = `Mesh renderer unavailable: ${error.message}`;
    document.querySelector('#mesh-status').classList.add('error');
    throw error;
  });
  return meshLoadPromise;
}

function selectRenderMode(mode) {
  renderMode = mode;
  document.querySelector('#render-mode').value = mode;
  if (mode === 'mesh') ensureMeshView();
  if (state) renderPrimaryView();
}

document.querySelector('#render-mode').addEventListener('change', event => selectRenderMode(event.target.value));
document.querySelector('#fusion-method').addEventListener('change', event => {
  fusionMethod = event.target.value;
  if (state) { renderFusion(); reconstructionView.draw(state); }
});
document.querySelector('#reset-render-camera').addEventListener('click', () => command('render_camera', {action:'reset'}));
const renderImage = document.querySelector('#mujoco-render');
let renderDrag = null;
renderImage.addEventListener('contextmenu', event => event.preventDefault());
renderImage.addEventListener('pointerdown', event => {
  renderDrag = {x:event.clientX, y:event.clientY, button:event.button};
  renderImage.setPointerCapture(event.pointerId);
});
renderImage.addEventListener('pointerup', event => {
  if (!renderDrag) return;
  const dx = (event.clientX-renderDrag.x)/Math.max(1,renderImage.clientWidth);
  const dy = (event.clientY-renderDrag.y)/Math.max(1,renderImage.clientHeight);
  if (renderDrag.button === 2) command('render_camera', {action:'pan', horizontal:-dx, vertical:dy});
  else command('render_camera', {action:'orbit', azimuth_delta_deg:-dx*180, elevation_delta_deg:dy*180});
  renderDrag = null;
});
renderImage.addEventListener('pointercancel', () => renderDrag = null);
renderImage.addEventListener('wheel', event => {
  event.preventDefault();
  command('render_camera', {action:'zoom', factor:event.deltaY > 0 ? 1.15 : .87});
}, {passive:false});

class ReconstructionProjectionView {
  constructor(canvas) {
    this.canvas = canvas;
    this.context = canvas.getContext("2d");
    this.yaw = -0.55;
    this.pitch = 0.28;
    this.zoom = 1.5;
    this.drag = null;
    canvas.addEventListener("pointerdown", event => { this.drag = [event.clientX, event.clientY]; canvas.setPointerCapture(event.pointerId); });
    canvas.addEventListener("pointermove", event => {
      if (!this.drag) return;
      this.yaw += (event.clientX - this.drag[0]) * .008;
      this.pitch = Math.max(-1.2, Math.min(1.2, this.pitch + (event.clientY - this.drag[1]) * .006));
      this.drag = [event.clientX, event.clientY];
      if (state) this.draw(state);
    });
    canvas.addEventListener("pointerup", () => this.drag = null);
    canvas.addEventListener("wheel", event => { event.preventDefault(); this.zoom *= event.deltaY > 0 ? .9 : 1.1; this.zoom = Math.max(.35, Math.min(4, this.zoom)); if (state) this.draw(state); }, {passive:false});
  }
  resize() {
    const ratio = window.devicePixelRatio || 1;
    const box = this.canvas.getBoundingClientRect();
    const width = Math.max(1, Math.round(box.width * ratio));
    const height = Math.max(1, Math.round(box.height * ratio));
    if (this.canvas.width !== width || this.canvas.height !== height) { this.canvas.width = width; this.canvas.height = height; }
    this.ratio = ratio; this.width = width; this.height = height;
  }
  transform(point, center) {
    let x = point[0] - center[0], y = point[1] - center[1], z = point[2] - center[2];
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
    const x1 = cy*x - sy*y, depth1 = sy*x + cy*y;
    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    return [x1, cp*z - sp*depth1, sp*z + cp*depth1];
  }
  project(point, center, scale) {
    const p = this.transform(point, center);
    return [this.width/2 + p[0]*scale, this.height/2 - p[1]*scale, p[2]];
  }
  circle(point, radiusM, center, scale, stroke, fill=null, dash=[]) {
    const [x,y] = this.project(point, center, scale), radius = Math.max(2, radiusM*scale);
    const c=this.context; c.beginPath(); c.arc(x,y,radius,0,Math.PI*2); c.setLineDash(dash); c.strokeStyle=stroke; c.lineWidth=2*this.ratio; c.stroke(); c.setLineDash([]); if(fill){c.fillStyle=fill;c.fill();}
  }
  draw(payload) {
    this.resize(); const c=this.context; c.clearRect(0,0,this.width,this.height);
    const center = payload.sphere_reconstruction.estimated_center_xyz || [0,-.045,.105];
    const scale = Math.min(this.width,this.height) * 5.5 * this.zoom;
    c.lineCap="round";
    for(const item of payload.accumulated_points) this.circle(item.position_xyz,.0014,center,scale,"#c478ff","#c478ff");
    const fit=payload.sphere_reconstruction;
    if(fit.show_estimated_sphere && fit.status==="VALID_RECONSTRUCTION" && fit.estimated_center_xyz) this.circle(fit.estimated_center_xyz,fit.estimated_radius_m,center,scale,"#35d7e5",null);
    const methods=payload.fusion?.methods || {};
    const estimates=[
      [methods.rgb_only,"#4d99ff"],
      [methods.tactile_only_representative,"#ff8b5c"],
      [methods[fusionMethod] || methods.fusion_finite_patch,"#55ec89"],
    ];
    for(const [estimate,color] of estimates) if(estimate?.valid && estimate.estimated_center_world_m)
      this.circle(estimate.estimated_center_world_m,estimate.radius_m,center,scale,color,null);
    if(payload.evaluation.enabled && payload.evaluation.ground_truth_sphere) {
      const gt=payload.evaluation.ground_truth_sphere; this.circle(gt.center_xyz,gt.radius_m,center,scale,"#55ec89",null,[7*this.ratio,5*this.ratio]);
    }
  }
}

const reconstructionView = new ReconstructionProjectionView(document.querySelector("#reconstruction-canvas"));

function format(value, digits=5) { return value == null ? "—" : Number(value).toFixed(digits); }
function vector(value) { return value ? `[${value.map(v=>Number(v).toFixed(5)).join(", ")}]` : "none"; }

function renderSensors() {
  const body=document.querySelector("#sensor-table tbody");
  for (const [id, row] of sensorRows) {
    if (!state.sensors.some(sensor => sensor.sensor_id === id)) { row.remove(); sensorRows.delete(id); }
  }
  if (!state.sensors.length) document.querySelector("#selected-sensor").textContent = "Awaiting synchronized tactile measurements";
  for(const sensor of state.sensors) {
    let row=sensorRows.get(sensor.sensor_id);
    if(!row){ row=document.createElement("tr"); row.addEventListener("click",()=>command("select_sensor",{sensor_id:sensor.sensor_id})); body.appendChild(row);sensorRows.set(sensor.sensor_id,row); }
    row.className=`${sensor.active?"active ":""}${state.selected_sensor_id===sensor.sensor_id?"selected":""}`;
    row.innerHTML=`<td>${sensor.sensor_id}</td><td>${sensor.finger_id}</td><td>${sensor.active?"ACTIVE":"Inactive"}</td><td>${format(sensor.scalar_sensor_value_n)}</td><td>${format(sensor.normal_contact_force_n)}</td><td>${format(sensor.estimated_pressure_pa,1)}</td><td>${format(sensor.estimated_indentation_depth_m,7)}</td>`;
  }
  const active=state.sensors.filter(sensor=>sensor.active);
  document.querySelector("#active-count").textContent=`${active.length} / ${state.sensors.length} active`;
  const selected=state.sensors.find(sensor=>sensor.sensor_id===state.selected_sensor_id);
  if(selected) document.querySelector("#selected-sensor").innerHTML=`<h3>${selected.sensor_id}</h3><div class="detail-grid"><b>Finger</b><span>${selected.finger_id}</span><span></span><b>Parent Link</b><span>${selected.parent_link}</span><span></span><b>Sensor Position in World Frame</b><span>${vector(selected.position_world_xyz)}</span><span>x / y / z</span><b>Sensor Sensing Direction in World Frame</b><span>${vector(selected.sensing_direction_world_xyz)}</span><span>nx / ny / nz</span><b>Estimated Three-Dimensional Contact Position</b><span>${vector(selected.estimated_contact_position_xyz)}</span><span>x / y / z</span></div>`;
}

function renderJointTargets() {
  const container = document.querySelector('#joint-target-grid');
  for (const joint of state.joints) {
    let control = jointControls.get(joint.name);
    if (!control) {
      const row = document.createElement('label');
      row.className = 'joint-target';
      const name = document.createElement('span'); name.textContent = joint.name;
      const input = document.createElement('input'); input.type = 'range'; input.step = '.001';
      const output = document.createElement('output');
      input.addEventListener('input', () => output.textContent = Number(input.value).toFixed(3));
      input.addEventListener('pointerdown', () => input.dataset.editing = 'true');
      input.addEventListener('change', () => {
        command('set_joint_target', {joint_name:joint.name, target_rad:Number(input.value)});
        delete input.dataset.editing;
      });
      input.addEventListener('blur', () => delete input.dataset.editing);
      row.append(name,input,output); container.append(row);
      control = {input,output}; jointControls.set(joint.name,control);
    }
    control.input.min = joint.lower_limit_rad;
    control.input.max = joint.upper_limit_rad;
    control.input.disabled = !streamValid() || !state.available_commands.includes('set_joint_target');
    if (!control.input.dataset.editing) control.input.value = joint.target_rad ?? joint.angle_rad;
    control.output.textContent = Number(control.input.value).toFixed(3);
  }
  document.querySelector("#joint-diagnostics tbody").innerHTML = state.joints.map(joint =>
    `<tr><td title="${joint.parent_link} → ${joint.child_link}; axis=${vector(joint.axis_xyz)}; ${joint.actuator_name || 'no actuator'}">${joint.name}</td>`+
    `<td>${format(joint.taught_target_rad,4)}</td><td>${format(joint.target_rad,4)}</td>`+
    `<td>${format(joint.angle_rad,4)}</td><td>${format(joint.position_error_rad,4)}</td>`+
    `<td>${format(joint.velocity_rad_s,4)}</td><td>${format(joint.actuator_force_n_m,4)}</td>`+
    `<td>${joint.tracking_cause || "none"}</td></tr>`
  ).join("");
}

function renderTeachControls(grasp) {
  const teach = grasp.teach || {};
  const toggle = document.querySelector("#teach-mode-toggle");
  toggle.textContent = teach.enabled ? "Exit Teach Grasp Pose" : "Enable Teach Grasp Pose";
  document.querySelector("#pose-validation-mode").value = teach.validation_mode || "physical_sphere_grasp";
  document.querySelector("#taught-tolerance").value = teach.tolerance_rad ?? 0.03;
  const poseSelect = document.querySelector("#taught-pose");
  const radius = state?.simulation?.object?.radius_m;
  const poses = (teach.poses || []).filter(pose => radius == null || Math.abs(pose.radius_m-radius) < 1e-9);
  const poseSignature = JSON.stringify(poses);
  if (poseSelect.dataset.poses !== poseSignature) {
    poseSelect.innerHTML = poses.length
      ? poses.map(pose => `<option value="${pose.name}">${pose.name} · ${pose.waypoint_role} · ${Math.round(pose.radius_m*1000)}mm</option>`).join("")
      : `<option value="">No taught poses saved</option>`;
    poseSelect.dataset.poses = poseSignature;
  }
  if (teach.selected_pose_name) poseSelect.value = teach.selected_pose_name;
  const errorValues = Object.values(teach.position_error_by_joint_rad || {}).map(Math.abs);
  const maximumError = errorValues.length ? Math.max(...errorValues) : null;
  document.querySelector("#teach-status").textContent =
    `${teach.enabled ? "TEACH" : "DISABLED"} · ${teach.validation_mode || "—"} · `+
    `pose=${teach.selected_pose_name || "none"} · status=${teach.execution_status || "IDLE"} · `+
    `settled=${teach.pose_is_settled ? "YES" : "NO"} · max error=${maximumError == null ? "—" : format(maximumError,4)+" rad"}`+
    (teach.failure_reason ? ` · ${teach.failure_reason}` : "");
}

function renderGraspDiagnostics() {
  const grasp = state.simulation.grasp || {};
  const object = state.simulation.object || {};
  const preset = document.querySelector("#grasp-preset");
  const names = grasp.preset_names || [];
  if (preset.dataset.names !== JSON.stringify(names)) {
    preset.innerHTML = names.map(name => `<option value="${name}">${name}</option>`).join("");
    preset.dataset.names = JSON.stringify(names);
  }
  if (grasp.preset_name) preset.value = grasp.preset_name;
  renderTeachControls(grasp);
  document.querySelector("#experiment-mode").value = object.experiment_mode || "fixed_contact_diagnostic";
  const fixed = object.experiment_mode !== "free_object_grasp";
  document.querySelector("#mode-explanation").textContent = fixed ?
    "CALIBRATION ONLY: sphere pose is held fixed; this is not a free grasp." :
    "FREE GRASP: gravity + contact + a visible support are enabled; accumulation resets after object motion.";
  const contact = grasp.contact_diagnostics || {};
  const objectPose = `${vector(object.center_xyz)} · q=${vector(object.quaternion_wxyz)}`;
  const motionWarning = state.object_motion_guard?.warning || "";
  document.querySelector("#grasp-diagnostics").textContent =
    `phase=${grasp.phase_name || "—"} (${grasp.stage || 0}/${grasp.stage_count || 4}) · `+
    `max|qvel|=${format(grasp.maximum_joint_velocity_rad_s,4)} rad/s · settled=${grasp.settled ? "YES" : "NO"} · `+
    `settling=${grasp.settling_time_seconds == null ? "—" : format(grasp.settling_time_seconds,3)+" s"} · `+
    `active sensors=${(grasp.active_sensor_ids || []).join(", ") || "none"} · active fingers=${grasp.active_finger_count || 0} · `+
    `sphere contacts=${contact.sphere_contact_count ?? 0} · penetration=${format(contact.maximum_penetration_m,6)} m · `+
    `normal force=${format(contact.total_normal_contact_force_n,4)} N · object pose (evaluation only)=${objectPose}`+
    (motionWarning ? ` · WARNING: ${motionWarning}` : "");
  const experiment = state.experiment || {};
  document.querySelector("#experiment-status").textContent = experiment.active ?
    `${experiment.name} · trial ${experiment.trial_number} · ${experiment.sample_count} synchronized samples · RECORDING` :
    experiment.last_result ? `${experiment.name} · ${experiment.last_result.number_of_trials} trial(s) · last=${experiment.last_result.reconstruction_status}` :
    "No named trial active";
}

function renderPrimaryView() {
  if (!renderMode) renderMode = state.render?.mode === 'MuJoCo Render' ? 'mujoco' : 'mesh';
  const image = document.querySelector('#mujoco-render');
  const researchImage = document.querySelector('#research-rgb');
  const researchOverlay = document.querySelector('#research-contact-overlay');
  const canvas = document.querySelector('#simulation-canvas');
  const meshStatus = document.querySelector('#mesh-status');
  const error = document.querySelector('#render-error');
  const frameStatus = document.querySelector('#render-frame-status');
  const help = document.querySelector('.canvas-help');
  document.querySelector('#render-mode').value = renderMode;
  document.body.dataset.renderMode = renderMode;
  if (renderMode === 'mujoco') {
    help.textContent = 'Left drag: orbit · right drag: pan · wheel: zoom';
    image.classList.remove('view-hidden'); researchImage.classList.add('view-hidden');
    researchOverlay.classList.add('view-hidden'); canvas.classList.add('view-hidden');
    meshStatus.classList.add('view-hidden');
    const info = state.render || {};
    frameStatus.textContent = info.frame_sequence == null ? info.status || 'UNAVAILABLE' :
      `${info.status} · frame ${info.frame_sequence} · sim t=${Number(info.simulation_timestamp).toFixed(3)} s`;
    if (info.status === 'ERROR' || !info.stream_url) {
      error.textContent = info.error || 'MuJoCo Render is unavailable for this data source. Select Mesh Digital Twin explicitly.';
      error.classList.remove('view-hidden');
      image.removeAttribute('src'); delete image.dataset.streamUrl;
    } else {
      error.classList.add('view-hidden');
      if (image.dataset.streamUrl !== info.stream_url) {
        image.src = info.stream_url; image.dataset.streamUrl = info.stream_url;
      }
    }
  } else if (renderMode === 'research') {
    help.textContent = 'Fixed calibrated camera · clean RGB · no debug overlays';
    image.classList.add('view-hidden'); canvas.classList.add('view-hidden');
    researchImage.classList.remove('view-hidden'); researchOverlay.classList.remove('view-hidden');
    meshStatus.classList.add('view-hidden');
    const info = state.research_rgb || {};
    frameStatus.textContent = info.frame_sequence == null ? info.status || 'UNAVAILABLE' :
      `${info.status} · clean RGB frame ${info.frame_sequence} · acquisition t=${Number(info.simulation_timestamp).toFixed(3)} s`;
    if (info.status === 'ERROR' || !info.stream_url) {
      error.textContent = info.error || 'Research RGB Camera is unavailable for this data source.';
      error.classList.remove('view-hidden');
      researchImage.removeAttribute('src'); delete researchImage.dataset.streamUrl;
    } else {
      error.classList.add('view-hidden');
      if (researchImage.dataset.streamUrl !== info.stream_url) {
        researchImage.src = info.stream_url; researchImage.dataset.streamUrl = info.stream_url;
      }
      drawResearchContactOverlay();
    }
  } else {
    help.textContent = 'Left drag: orbit · right drag: pan · wheel: zoom';
    image.classList.add('view-hidden'); researchImage.classList.add('view-hidden');
    researchOverlay.classList.add('view-hidden'); canvas.classList.remove('view-hidden');
    meshStatus.classList.remove('view-hidden'); error.classList.add('view-hidden');
    frameStatus.textContent = 'Browser STL fallback/debug view';
    ensureMeshView().then(view => state && view.draw(state));
  }
}

function drawResearchContactOverlay() {
  const canvas = document.querySelector('#research-contact-overlay');
  const image = document.querySelector('#research-rgb');
  if (canvas.classList.contains('view-hidden')) return;
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(image.clientWidth * ratio));
  const height = Math.max(1, Math.round(image.clientHeight * ratio));
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  const context = canvas.getContext('2d'); context.clearRect(0,0,width,height);
  if (!document.querySelector('#show-projected-contacts').checked) return;
  const camera = state.camera || {};
  const imageWidth = camera.image_width || 640, imageHeight = camera.image_height || 480;
  const scale = Math.min(width/imageWidth, height/imageHeight);
  const offsetX = (width-imageWidth*scale)/2, offsetY = (height-imageHeight*scale)/2;
  context.fillStyle = '#ffe34f'; context.strokeStyle = '#251d00'; context.lineWidth = 2*ratio;
  for (const contact of state.vision_only?.projected_estimated_tactile_contacts || []) {
    if (!contact.visible_in_image_bounds) continue;
    const [u,v] = contact.pixel_uv, x=offsetX+u*scale, y=offsetY+v*scale;
    context.beginPath(); context.arc(x,y,5*ratio,0,2*Math.PI); context.fill(); context.stroke();
  }
}

function drawPlot(canvas, series, field, unit) {
  const ratio=window.devicePixelRatio||1, box=canvas.getBoundingClientRect(), w=Math.max(1,Math.round(box.width*ratio)), h=Math.max(1,Math.round(box.height*ratio)); if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h}
  const c=canvas.getContext("2d");c.clearRect(0,0,w,h);c.strokeStyle="#263d51";c.lineWidth=1;c.beginPath();c.moveTo(34*ratio,6*ratio);c.lineTo(34*ratio,h-20*ratio);c.lineTo(w-5*ratio,h-20*ratio);c.stroke();
  series = series.map(item => ({...item, samples: item.samples.filter(sample => sample[field] != null && Number.isFinite(sample[field]))}));
  const all=series.flatMap(item=>item.samples.map(sample=>[sample.time_seconds,sample[field]])); if(!all.length){c.fillStyle="#71899b";c.fillText("No samples",45*ratio,h/2);return}
  const tMin=Math.min(...all.map(v=>v[0])), tMax=Math.max(...all.map(v=>v[0])); let yMin=Math.min(0,...all.map(v=>v[1])), yMax=Math.max(...all.map(v=>v[1])); if(yMax-yMin<1e-12)yMax=yMin+1;
  series.forEach((item,index)=>{c.beginPath();item.samples.forEach((sample,i)=>{const x=34*ratio+(sample.time_seconds-tMin)/Math.max(1e-9,tMax-tMin)*(w-42*ratio),y=6*ratio+(yMax-sample[field])/(yMax-yMin)*(h-27*ratio);i?c.lineTo(x,y):c.moveTo(x,y)});c.strokeStyle=colors[index%colors.length];c.lineWidth=1.5*ratio;c.stroke()});
  c.fillStyle="#7890a2";c.font=`${9*ratio}px ui-monospace`;c.fillText(`${format(yMax,field.includes("pressure")?0:5)} ${unit}`,2*ratio,11*ratio);c.fillText(`${format(tMax-tMin,1)} s`,w-42*ratio,h-5*ratio);
}

function renderPlots(){ document.querySelector("#series-scope").value=state.time_series.scope; document.querySelectorAll(".plot canvas").forEach(canvas=>drawPlot(canvas,state.time_series.series,canvas.dataset.field,canvas.dataset.unit)); document.querySelector("#series-legend").innerHTML=state.time_series.series.map((item,index)=>`<span style="color:${colors[index%colors.length]}">● ${item.sensor_id}</span>`).join(" &nbsp; ")||"No currently active sensor series"; }

function renderReconstruction(){
  const r=state.sphere_reconstruction, badge=document.querySelector("#fit-status");badge.textContent=r.status;badge.className=`status-badge ${r.status==="VALID_RECONSTRUCTION"?"valid":r.status==="POOR_SPATIAL_COVERAGE"?"poor":r.status==="INSUFFICIENT_DATA"?"insufficient":""}`;
  const rows=[["Accepted tactile point count",r.number_of_input_points],["Unique sensors",r.number_of_unique_sensors],["Unique fingers",r.number_of_unique_fingers],["Spatial coverage",`${format(r.spatial_coverage_m)} m`],["Estimated center",vector(r.estimated_center_xyz)],["Estimated radius",r.estimated_radius_m==null?"—":`${format(r.estimated_radius_m)} m`],["Sphere fit RMSE",r.fit_residual_rmse_m==null?"—":`${format(r.fit_residual_rmse_m,6)} m`]];
  document.querySelector("#reconstruction-metrics").innerHTML=rows.map(row=>`<dt>${row[0]}</dt><dd>${row[1]}</dd>`).join("");
  const evaluation=document.querySelector("#evaluation-metrics"); if(state.evaluation.enabled){const gt=state.evaluation.ground_truth_sphere,e=state.evaluation.sphere;evaluation.classList.remove("hidden");evaluation.innerHTML=`<h3>Evaluation-only Ground Truth</h3><p>GT center: <span class="mono">${vector(gt.center_xyz)}</span><br>GT radius: <span class="mono">${format(gt.radius_m)} m</span><br>Center error: <span class="mono">${e?format(e.center_error_m,6)+" m":"—"}</span><br>Radius error: <span class="mono">${e?format(e.radius_error_m,6)+" m":"—"}</span><br>Relative radius error: <span class="mono">${e?format(e.relative_radius_error,4):"—"}</span></p>`}else evaluation.classList.add("hidden");
}

function renderVisionOnly() {
  const root = document.querySelector('#vision-only-metrics');
  const vision = state.vision_only || {}, segmentation = vision.segmentation || {};
  const sphere = vision.sphere || {}, unknown = vision.unknown_radius_setting || {};
  const evaluation = state.vision_evaluation || {};
  const radius = sphere.radius_m == null ? '—' : `${format(sphere.radius_m,5)} m (${sphere.radius_label})`;
  root.innerHTML = `<h3>Vision-only RGB sphere baseline</h3><p>`+
    `Segmentation: <span class="mono">${segmentation.status || 'NOT_RUN'}</span> · confidence <span class="mono">${format(segmentation.confidence,3)}</span>`+
    `${evaluation.available ? ` · evaluation IoU <span class="mono">${format(evaluation.segmentation_iou,3)}</span>` : ''}<br>`+
    `Known-radius center: <span class="mono">${vector(sphere.estimated_center_world_m)}</span><br>`+
    `Radius: <span class="mono">${radius}</span> · silhouette residual <span class="mono">${format(sphere.silhouette_residual_pixels,3)} px</span><br>`+
    `Identifiability: <span class="mono">${sphere.status || 'NOT_RUN'}</span> · unknown-radius: <span class="mono">${unknown.status || 'SCALE_AMBIGUOUS'}</span><br>`+
    `Processing: <span class="mono">${format((segmentation.processing_time_ms || 0)+(sphere.processing_time_ms || 0),2)} ms</span>`+
    `</p>`;
}

function renderFusion() {
  const fusion=state.fusion || {}, methods=fusion.methods || {};
  const selector=document.querySelector('#fusion-method');
  if (!methods[fusionMethod] && fusion.selected_method) fusionMethod=fusion.selected_method;
  selector.value=fusionMethod;
  const selected=methods[fusionMethod] || {};
  const labels={rgb_only:'A · RGB only', tactile_only_representative:'B · Tactile only / representative',
    fusion_representative:'C · Fusion / representative', fusion_finite_patch:'D · Fusion / finite sensor patch'};
  const validCount=Object.values(methods).filter(result=>result.valid).length;
  const root=document.querySelector('#fusion-metrics');
  const diagnostics=selected.diagnostics || {};
  const coverage=`${diagnostics.contact_count ?? 0} contacts · ${diagnostics.unique_sensor_count ?? 0} sensors · ${diagnostics.unique_finger_count ?? 0} fingers · spread ${format(diagnostics.spatial_spread_m,4)} m`;
  const uncertainty=(selected.uncertainty_standard_deviation_m || []).map(value=>format(value,6)).join(', ') || 'unavailable';
  root.querySelector(':scope > p').innerHTML=
    `Pipeline: <span class="mono">${fusion.status || 'NOT_RUN'}</span> · valid methods <span class="mono">${validCount} / 4</span><br>`+
    `Selected: <span class="mono">${labels[fusionMethod] || fusionMethod}</span> · status <span class="mono">${selected.status || 'NOT_RUN'}</span><br>`+
    `Center: <span class="mono">${vector(selected.estimated_center_world_m)}</span><br>`+
    `Known radius: <span class="mono">${fusion.known_radius_m == null ? '—' : format(fusion.known_radius_m,5)+' m'}</span> · `+
    `diameter <span class="mono">${fusion.known_diameter_m == null ? '—' : format(fusion.known_diameter_m,5)+' m'}</span> · <span class="mono">${fusion.radius_label || '—'}</span><br>`+
    `Validity: <span class="mono">${selected.validity_reason || 'No result yet'}</span><br>`+
    `RGB residual: <span class="mono">${format(selected.vision_residual_rms_sigma,3)} σ / ${format(diagnostics.vision_residual_rms_px,3)} px</span> · `+
    `tactile residual: <span class="mono">${format(selected.tactile_residual_rms_sigma,3)} σ / ${format(diagnostics.tactile_residual_rms_m,6)} m</span><br>`+
    `Weights λRGB/λT: <span class="mono">${format(fusion.weights?.vision,2)} / ${format(fusion.weights?.tactile,2)}</span> · Huber: <span class="mono">${format(diagnostics.robust_loss?.threshold_sigma ?? fusion.robust_loss?.threshold_sigma,2)} σ</span><br>`+
    `RGB evidence: <span class="mono">${diagnostics.vision_status || '—'} · ${diagnostics.visible_boundary_count ?? 0} boundary samples · angular coverage ${format(diagnostics.visible_angular_coverage,3)}</span><br>`+
    `Contributors: <span class="mono">${coverage}</span><br>`+
    `Sensors: <span class="mono">${(diagnostics.contributing_sensor_ids || []).join(', ') || 'none'}</span><br>`+
    `Fingers: <span class="mono">${(diagnostics.contributing_finger_ids || []).join(', ') || 'none'}</span><br>`+
    `Joint condition: <span class="mono">${format(selected.condition_number,2)}</span> · tactile coverage condition: <span class="mono">${format(diagnostics.jacobian_condition,2)}</span><br>`+
    `Center uncertainty σx/y/z: <span class="mono">${uncertainty} m</span> · init: <span class="mono">${selected.initialization || '—'}</span><br>`+
    `Runtime: <span class="mono">${format(selected.runtime_ms,2)} ms</span>`+
    (fusion.error ? `<br><span class="error">${fusion.error}</span>` : '');
}

function renderCameraCalibration() {
  const root=document.querySelector('#camera-calibration-content');
  const calibration=state.research_camera_calibration;
  document.querySelector('#show-research-camera-frustum').checked=Boolean(state.research_camera_debug_visible);
  if (!calibration) { root.textContent='Waiting for the compiled MuJoCo camera calibration…'; return; }
  const i=calibration.intrinsics || {}, world=calibration.T_world_from_camera_cv || [];
  const cameraFromWorld=calibration.T_camera_cv_from_world || [];
  const position=world.length===4 ? world.slice(0,3).map(row=>row[3]) : null;
  const rotation=world.length===4 ? world.slice(0,3).map(row=>row.slice(0,3)) : null;
  root.innerHTML=`<p><b>Fixed Research RGB Camera</b> is the clean algorithm input. The movable <b>Debug Camera</b> is only the top-left engineering view controlled by drag/pan/zoom; moving it never changes research RGB or qpos.</p>`+
    `<dl class="camera-grid"><dt>Calibration version</dt><dd>${calibration.calibration_version}</dd>`+
    `<dt>Camera / optical frame</dt><dd>${calibration.camera_name} / ${calibration.frame_id}</dd>`+
    `<dt>Parent / transform direction</dt><dd>${calibration.parent_frame} / T_world_from_camera_cv maps CV camera coordinates → world</dd>`+
    `<dt>Position in world (m)</dt><dd>${vector(position)}</dd>`+
    `<dt>Orientation R_world_from_camera_cv</dt><dd><code>${JSON.stringify(rotation)}</code></dd>`+
    `<dt>Extrinsics T_camera_cv_from_world</dt><dd><code>${JSON.stringify(cameraFromWorld)}</code></dd>`+
    `<dt>Resolution / vertical FOV</dt><dd>${i.width}×${i.height} / ${format(calibration.fovy_degrees,3)}°</dd>`+
    `<dt>Intrinsics fx, fy, cx, cy (px)</dt><dd>${format(i.fx,6)}, ${format(i.fy,6)}, ${format(i.cx,3)}, ${format(i.cy,3)}</dd>`+
    `<dt>Convention</dt><dd>CV optical: +X right, +Y down, +Z forward; world frame is MuJoCo world</dd></dl>`;
}

function render(){
  const backendBuild = state.build?.backend || 'LEGACY/RESTART REQUIRED';
  const buildLabel = document.querySelector('#build-id');
  buildLabel.textContent = `UI ${FRONTEND_BUILD_ID} · API ${backendBuild}`;
  buildLabel.classList.toggle('mismatch', backendBuild !== FRONTEND_BUILD_ID);
  document.querySelector("#data-source").textContent=state.source.display_name;
  document.querySelector("#simulation-time").textContent=state.simulation.time_seconds == null ? "Awaiting measurements" : `t = ${state.simulation.time_seconds.toFixed(3)} s`;
  document.querySelector("#simulation-toggle").textContent=state.simulation.running?"Pause Simulation":"Start Simulation";
  document.querySelectorAll("[data-radius]").forEach(button=>button.classList.toggle("selected",Math.abs(Number(button.dataset.radius)-state.simulation.object.radius_m)<1e-6));
  const grasp = state.simulation.grasp || {};
  const graspLabel = grasp.preset_name ? `${grasp.preset_name} · ${grasp.phase_name} · grasp stage ${grasp.stage}/${grasp.stage_count || 4}` : "Read-only robot state";
  document.querySelector("#simulation-summary").textContent=`${graspLabel} · accumulation ${state.accumulation.running?"RUNNING":"PAUSED"} · ${state.accumulation.accepted_point_count} accepted · ${state.accumulation.rejected_duplicate_count} duplicates`;
  document.querySelector("#show-ground-truth").checked=state.evaluation.enabled;document.querySelector("#show-estimated").checked=state.sphere_reconstruction.show_estimated_sphere;
  renderSensors();renderJointTargets();renderGraspDiagnostics();renderPlots();renderReconstruction();renderVisionOnly();renderFusion();renderCameraCalibration();renderPrimaryView();reconstructionView.draw(state);
  document.querySelectorAll(".quadrant").forEach(panel => panel.dataset.snapshotTimestamp = String(state.snapshot.timestamp));
  renderStreamStatus();
}

window.addEventListener("resize",()=>state&&render());
connect();
