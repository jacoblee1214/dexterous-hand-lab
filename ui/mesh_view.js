import * as THREE from 'three';
import { STLLoader } from './vendor/three/examples/jsm/loaders/STLLoader.js';
import { OrbitControls } from './vendor/three/examples/jsm/controls/OrbitControls.js';

// Rendering only: the server supplies world link poses and calibrated local
// mounts. There are no joint graphs, actuator targets, or physics in this scene.
export class MeshDigitalTwin {
  constructor(canvas) {
    this.canvas = canvas;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#091521');
    this.camera = new THREE.PerspectiveCamera(42, 1, .001, 10);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(.32, -.42, .29);
    this.renderer = new THREE.WebGLRenderer({canvas, antialias: true, alpha: false});
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.set(0, -.045, .105);
    this.controls.minDistance = .07;
    this.controls.maxDistance = 2;
    this.controls.addEventListener('change', () => this.render());
    this.controls.update();
    this.controls.saveState();
    const ambient = new THREE.HemisphereLight(0xe8f4ff, 0x334556, 2);
    ambient.position.set(0, 0, 1);
    this.scene.add(ambient);
    for (const [position, intensity] of [[[.2, -.35, .5], 3], [[-.3, .15, .25], 1.6]]) {
      const light = new THREE.DirectionalLight(0xffffff, intensity);
      light.position.fromArray(position);
      this.scene.add(light);
    }
    this.links = new Map();
    this.sensors = new Map();
    this.meshes = [];
    this.loaded = false;
    this.failed = false;
    this.live = false;
    this.latest = null;
    this.pointLayers = new Map();
    this.pointGeometry = new THREE.SphereGeometry(1, 8, 6);
    const sphereGeometry = new THREE.SphereGeometry(1, 40, 24);
    this.objectSphere = new THREE.Mesh(sphereGeometry, new THREE.MeshStandardMaterial({
      color: '#4d99ff', roughness: .45, metalness: .05, transparent: true, opacity: .6,
    }));
    this.estimatedSphere = new THREE.Mesh(sphereGeometry, new THREE.MeshBasicMaterial({
      color: '#35d7e5', wireframe: true, transparent: true, opacity: .45, depthTest: false,
    }));
    this.truthSphere = new THREE.Mesh(sphereGeometry, new THREE.MeshBasicMaterial({
      color: '#55ec89', wireframe: true, transparent: true, opacity: .3, depthTest: false,
    }));
    for (const object of [this.objectSphere, this.estimatedSphere, this.truthSphere]) {
      object.visible = false;
      this.scene.add(object);
    }
    this.observer = new ResizeObserver(() => this.render());
    this.observer.observe(canvas.parentElement);
    canvas.addEventListener('webglcontextlost', event => {
      event.preventDefault();
      this.failed = true;
      this.status('WebGL context lost — reload the dashboard', true);
    });
    this.render();
  }

  // Retains the view-only camera inspection used by the dashboard tests.
  get yaw() { return this.controls.getAzimuthalAngle(); }

  status(text, failed = false) {
    const label = document.querySelector('#mesh-status');
    label.textContent = text;
    label.classList.toggle('error', failed);
  }

  link(name) {
    if (!this.links.has(name)) {
      const group = new THREE.Group();
      group.name = name;
      group.visible = false;
      this.links.set(name, group);
      this.scene.add(group);
    }
    return this.links.get(name);
  }

  async load(url) {
    this.status('Loading original STL meshes…');
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`Manifest HTTP ${response.status}`);
      const manifest = await response.json();
      if (manifest.schema_version !== '1.0.0' || manifest.units !== 'metres') throw new Error('Unsupported mesh manifest');
      this.manifest = manifest;
      const loader = new STLLoader();
      // Load once. Preserve every STL vertex in the supplied link frame: do not
      // recenter geometry or apply MuJoCo's internal mesh inertia transform.
      await Promise.all(manifest.visuals.map(async visual => {
        const geometry = await loader.loadAsync(visual.url);
        geometry.computeBoundingSphere();
        const mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({
          color: '#b9c9d3', metalness: .18, roughness: .6,
        }));
        mesh.name = visual.url;
        mesh.position.fromArray(visual.position_xyz);
        mesh.rotation.set(...visual.rotation_rpy, 'ZYX');
        mesh.scale.fromArray(visual.scale_xyz);
        this.link(visual.link_name).add(mesh);
        this.meshes.push(mesh);
      }));
      for (const mount of manifest.sensors) this.addSensor(mount);
      this.loaded = true;
      this.status(`${this.meshes.length} STL meshes · ${this.sensors.size} sensor surfaces`);
      if (this.latest) this.draw(this.latest);
    } catch (error) {
      this.failed = true;
      // Partial loads are explicitly hidden; never disguise a failed asset load.
      for (const group of this.links.values()) group.visible = false;
      this.status(`Mesh loading failed: ${error.message}`, true);
      this.render();
    }
  }

  addSensor(mount) {
    let geometry;
    if (mount.sensor_type === 'rectangular') {
      geometry = new THREE.PlaneGeometry(mount.width, mount.height);
    } else {
      // Outward half-ellipsoid, with the tip center at local z=0, exactly as in
      // SensorMount.surface_point_local(). U/V/normal form the local basis.
      const positions = [], indices = [], rings = 12, slices = 32;
      const [rx, ry, rz] = mount.fingertip_radii_xyz;
      for (let ring = 0; ring <= rings; ring++) {
        const theta = ring / rings * Math.PI / 2;
        for (let slice = 0; slice <= slices; slice++) {
          const phi = slice / slices * Math.PI * 2;
          positions.push(rx * Math.sin(theta) * Math.cos(phi),
                         ry * Math.sin(theta) * Math.sin(phi), rz * (Math.cos(theta) - 1));
        }
      }
      for (let ring = 0; ring < rings; ring++) for (let slice = 0; slice < slices; slice++) {
        const a = ring * (slices + 1) + slice, b = a + slices + 1;
        indices.push(a, b, a + 1, b, b + 1, a + 1);
      }
      geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
      geometry.setIndex(indices);
      geometry.computeVertexNormals();
    }
    const patch = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({
      color: '#35d7e5', transparent: true, opacity: .38, side: THREE.DoubleSide,
      depthTest: false, depthWrite: false,
    }));
    patch.name = mount.sensor_id;
    patch.renderOrder = 2;
    // The calibration files intentionally remain the authority. Some measured
    // axes differ from a perfectly orthonormal rotation by sub-micrometre
    // rounding. Assign the affine matrix directly so Three.js does not silently
    // normalize those values while converting them through a quaternion.
    const [u, v, n, p] = [mount.surface_u_axis, mount.surface_v_axis,
      mount.outward_normal, mount.center_xyz];
    patch.matrix.set(u[0], v[0], n[0], p[0],
                     u[1], v[1], n[1], p[1],
                     u[2], v[2], n[2], p[2],
                     0, 0, 0, 1);
    patch.matrixAutoUpdate = false;
    this.link(mount.parent_link).add(patch);
    this.sensors.set(mount.sensor_id, patch);
  }

  setSphere(mesh, center, radius, visible) {
    mesh.visible = Boolean(visible && center && radius > 0);
    if (mesh.visible) {
      mesh.position.fromArray(center);
      mesh.scale.setScalar(radius);
    }
  }

  setPoints(name, points, color, radius) {
    let layer = this.pointLayers.get(name);
    if (!layer || layer.instanceMatrix.count < points.length) {
      if (layer) {
        this.scene.remove(layer);
        layer.material.dispose();
        layer.dispose();
      }
      layer = new THREE.InstancedMesh(this.pointGeometry, new THREE.MeshBasicMaterial({
        color, depthTest: false, depthWrite: false,
      }), Math.max(32, points.length * 2));
      layer.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      layer.frustumCulled = false;
      layer.renderOrder = 3;
      this.pointLayers.set(name, layer);
      this.scene.add(layer);
    }
    const matrix = new THREE.Matrix4();
    for (let i = 0; i < points.length; i++) {
      matrix.makeScale(radius, radius, radius);
      matrix.setPosition(...points[i].position_xyz);
      layer.setMatrixAt(i, matrix);
    }
    layer.count = points.length;
    layer.instanceMatrix.needsUpdate = true;
  }

  draw(payload) {
    this.latest = payload;
    if (!this.loading) {
      this.loading = true;
      this.load(payload.hand.manifest_url);
    }
    if (!this.loaded || this.failed) return;
    for (const group of this.links.values()) group.visible = false;
    for (const pose of payload.hand.links) {
      const group = this.links.get(pose.name);
      if (!group) continue;
      group.position.fromArray(pose.position_xyz);
      const [w, x, y, z] = pose.quaternion_wxyz;
      group.quaternion.set(x, y, z, w);
      group.visible = true;
    }
    this.live = payload.snapshot.valid;
    for (const sensor of payload.sensors) {
      const patch = this.sensors.get(sensor.sensor_id);
      if (!patch) continue;
      patch.userData.active = sensor.active;
      patch.material.color.set(sensor.active && this.live ? '#ff4b43' : '#35d7e5');
      patch.material.opacity = sensor.active && this.live ? .85 : .38;
    }
    const object = payload.simulation.object, fit = payload.sphere_reconstruction;
    this.setSphere(this.objectSphere, object?.center_xyz, object?.radius_m, Boolean(object?.center_xyz));
    this.setSphere(this.estimatedSphere, fit.estimated_center_xyz, fit.estimated_radius_m,
                   fit.show_estimated_sphere && fit.status === 'VALID_RECONSTRUCTION');
    const truth = payload.evaluation.ground_truth_sphere;
    this.setSphere(this.truthSphere, truth?.center_xyz, truth?.radius_m, payload.evaluation.enabled);
    this.setPoints('contacts', payload.current_estimated_contacts, '#ffe34f', .0023);
    this.setPoints('accumulated', payload.accumulated_points, '#c478ff', .0013);
    this.setPoints('evaluation', payload.evaluation.enabled ? payload.evaluation.ground_truth_contacts || [] : [], '#55ec89', .0016);
    this.snapshotTimestamp = payload.snapshot.timestamp;
    // Keep an immutable diagnostic of the exact payload applied to this frame.
    // WebSocket state can advance between two Playwright calls, so tests should
    // inspect this snapshot rather than mixing two different simulation frames.
    this.lastApplied = {
      timestamp: payload.snapshot.timestamp,
      activeSensorIds: payload.sensors.filter(sensor => sensor.active).map(sensor => sensor.sensor_id),
      contactPoints: payload.current_estimated_contacts.map(contact => ({
        sensor_id: contact.sensor_id,
        position_xyz: [...contact.position_xyz],
      })),
    };
    this.scene.updateMatrixWorld(true);
    this.render();
  }

  render() {
    if (this.renderPending) return;
    this.renderPending = true;
    const delay = Math.max(0, (this.nextRenderAfter || 0) - performance.now());
    setTimeout(() => requestAnimationFrame(() => {
      this.renderPending = false;
      const started = performance.now();
      this.renderFrame();
      const elapsed = performance.now() - started;
      // Original CAD contains roughly 480k triangles. Preserve it, but leave
      // breathing room for telemetry/WebSocket work on software-rendered GPUs.
      this.nextRenderAfter = performance.now() + Math.min(250, Math.max(0, elapsed * 1.5));
    }), delay);
  }

  renderFrame() {
    if (!this.renderer) return;
    const {width, height} = this.canvas.getBoundingClientRect();
    if (width < 1 || height < 1) return;
    if (width !== this.renderWidth || height !== this.renderHeight) {
      this.renderer.setSize(width, height, false);
      this.renderWidth = width;
      this.renderHeight = height;
    }
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.render(this.scene, this.camera);
  }

  setLive(valid) {
    if (this.live === valid) return;
    this.live = valid;
    for (const patch of this.sensors.values()) {
      const active = valid && patch.userData.active;
      patch.material.color.set(active ? '#ff4b43' : '#35d7e5');
      patch.material.opacity = active ? .85 : .38;
    }
    this.render();
  }
}
