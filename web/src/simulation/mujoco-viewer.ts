/**
 * MuJoCo WASM simulation viewer — adapted from mujoco_wasm-main/src/main.js
 * Loads a generated XML + STL into MuJoCo WASM and renders with Three.js.
 * Visual style, lighting, and drag interaction match the official mujoco_wasm demo.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import GUI from 'lil-gui';

// mujoco-js types (the module doesn't ship TS declarations)
type MujocoModule = any;

let mujocoModule: MujocoModule | null = null;

async function ensureMujoco(): Promise<MujocoModule> {
  if (mujocoModule) return mujocoModule;
  try {
    console.log('[MuJoCo] Loading WASM module...');
    const load_mujoco = (await import('mujoco-js')).default;
    mujocoModule = await load_mujoco();
    mujocoModule.FS.mkdir('/working');
    mujocoModule.FS.mount(mujocoModule.MEMFS, { root: '.' }, '/working');
    console.log('[MuJoCo] WASM module loaded successfully');
    return mujocoModule;
  } catch (e) {
    console.error('[MuJoCo] Failed to load WASM module:', e);
    throw e;
  }
}

/** MuJoCo → Three.js coordinate conversion (Z-up → Y-up) */
function getPosition(buffer: Float64Array, index: number, target: THREE.Vector3): THREE.Vector3 {
  target.set(
    buffer[index * 3 + 0],
    buffer[index * 3 + 2],
    -buffer[index * 3 + 1],
  );
  return target;
}

function getQuaternion(buffer: Float64Array, index: number, target: THREE.Quaternion): THREE.Quaternion {
  target.set(
    -buffer[index * 4 + 1],
    -buffer[index * 4 + 3],
    buffer[index * 4 + 2],
    -buffer[index * 4 + 0],
  );
  return target;
}

/** Three.js → MuJoCo coordinate conversion (Y-up → Z-up) */
function toMujocoPos(v: THREE.Vector3): THREE.Vector3 {
  return v.set(v.x, -v.z, v.y);
}

export class MujocoViewer {
  private mujoco: MujocoModule | null = null;
  private model: any = null;
  private data: any = null;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private controls: OrbitControls;
  private bodies: Map<number, THREE.Group> = new Map();
  private mujocoRoot: THREE.Group | null = null;
  private mujocoTime = 0;
  private paused = false;
  private animationId: number | null = null;
  private canvas: HTMLCanvasElement;

  // GUI for simulation controls + actuators
  private simGui: GUI | null = null;
  private actuatorFolder: GUI | null = null;
  private actuatorGUIs: any[] = [];
  private actuatorParams: Record<string, number> = {};
  private simParams = { paused: false };

  // Drag interaction state
  private raycaster = new THREE.Raycaster();
  private mousePos = new THREE.Vector2();
  private dragActive = false;
  private dragBody: THREE.Object3D | null = null;
  private grabDistance = 0;
  private localHit = new THREE.Vector3();
  private worldHit = new THREE.Vector3();
  private currentWorld = new THREE.Vector3();
  private dragArrow: THREE.ArrowHelper;
  private mouseDown = false;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;

    // Scene — match mujoco_wasm reference
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0.15, 0.25, 0.35);
    this.scene.fog = new THREE.Fog(this.scene.background, 15, 25.5);

    // Camera — match mujoco_wasm reference
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.001, 100);
    this.camera.position.set(2.0, 1.7, 1.7);
    this.scene.add(this.camera);

    // Renderer — match mujoco_wasm: PCFSoftShadowMap
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    // Lighting — match mujoco_wasm: ambient + spotlight
    const ambient = new THREE.AmbientLight(0xffffff, 0.1 * Math.PI);
    this.scene.add(ambient);

    const spot = new THREE.SpotLight();
    spot.angle = 1.11;
    spot.distance = 10000;
    spot.penumbra = 0.5;
    spot.intensity = spot.intensity * Math.PI * 10.0;
    spot.castShadow = true;
    spot.shadow.mapSize.width = 1024;
    spot.shadow.mapSize.height = 1024;
    spot.shadow.camera.near = 0.1;
    spot.shadow.camera.far = 100;
    spot.position.set(0, 3, 3);
    const spotTarget = new THREE.Object3D();
    spotTarget.position.set(0, 1, 0);
    this.scene.add(spotTarget);
    spot.target = spotTarget;
    this.scene.add(spot);

    // OrbitControls — match mujoco_wasm settings
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.set(0, 0.7, 0);
    this.controls.panSpeed = 2;
    this.controls.zoomSpeed = 1;
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.10;
    this.controls.screenSpacePanning = true;
    this.controls.update();

    // Drag arrow helper — match mujoco_wasm
    this.dragArrow = new THREE.ArrowHelper(
      new THREE.Vector3(0, 1, 0), new THREE.Vector3(), 15, 0x666666,
    );
    this.dragArrow.setLength(15, 3, 1);
    (this.dragArrow.line as THREE.Line).material = new THREE.LineBasicMaterial({
      color: 0x666666, transparent: true, opacity: 0.5,
    });
    (this.dragArrow.cone as THREE.Mesh).material = new THREE.MeshBasicMaterial({
      color: 0x666666, transparent: true, opacity: 0.5,
    });
    this.dragArrow.visible = false;
    this.scene.add(this.dragArrow);

    // Pointer event listeners for drag interaction
    canvas.addEventListener('pointerdown', this.onPointerDown.bind(this), true);
    document.addEventListener('pointermove', this.onPointerMove.bind(this), true);
    document.addEventListener('pointerup', this.onPointerUp.bind(this), true);

    window.addEventListener('resize', () => this.resize());

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
      if (e.code === 'Space' && this.animationId !== null) {
        e.preventDefault();
        this.togglePause();
      } else if (e.code === 'Backspace' && this.animationId !== null) {
        e.preventDefault();
        this.reset();
      }
    });
  }

  // ---- Drag interaction (matches mujoco_wasm DragStateManager) ----

  private updateRaycaster(x: number, y: number) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.mousePos.x = ((x - rect.left) / rect.width) * 2 - 1;
    this.mousePos.y = -((y - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.mousePos, this.camera);
  }

  private onPointerDown(evt: PointerEvent) {
    this.mouseDown = true;
    this.dragBody = null;
    this.updateRaycaster(evt.clientX, evt.clientY);

    const intersects = this.raycaster.intersectObjects(this.scene.children, true);
    for (const hit of intersects) {
      let obj: THREE.Object3D | null = hit.object;
      // Walk up to find bodyID
      while (obj && !(obj as any).bodyID) obj = obj.parent;
      if (obj && (obj as any).bodyID > 0) {
        this.dragBody = obj;
        this.grabDistance = hit.distance;
        const hitPoint = this.raycaster.ray.origin.clone()
          .addScaledVector(this.raycaster.ray.direction, this.grabDistance);
        this.localHit.copy(obj.worldToLocal(hitPoint.clone()));
        this.worldHit.copy(hitPoint);
        this.currentWorld.copy(hitPoint);
        this.dragActive = true;
        this.controls.enabled = false;
        this.dragArrow.position.copy(hitPoint);
        this.dragArrow.visible = true;
        break;
      }
    }
  }

  private onPointerMove(evt: PointerEvent) {
    if (!this.mouseDown || !this.dragActive) return;
    this.updateRaycaster(evt.clientX, evt.clientY);
    const hitPoint = this.raycaster.ray.origin.clone()
      .addScaledVector(this.raycaster.ray.direction, this.grabDistance);
    this.currentWorld.copy(hitPoint);
    this.updateDragArrow();
  }

  private onPointerUp(_evt: PointerEvent) {
    this.dragBody = null;
    this.dragActive = false;
    this.controls.enabled = true;
    this.dragArrow.visible = false;
    this.mouseDown = false;
  }

  private updateDragArrow() {
    if (!this.dragBody || !this.dragArrow) return;
    // Recompute worldHit from localHit (body may have moved)
    this.worldHit.copy(this.localHit);
    this.dragBody.localToWorld(this.worldHit);
    this.dragArrow.position.copy(this.worldHit);
    const dir = this.currentWorld.clone().sub(this.worldHit);
    const len = dir.length();
    if (len > 1e-6) {
      this.dragArrow.setDirection(dir.normalize());
      this.dragArrow.setLength(len);
    }
  }

  /** Apply drag force via MuJoCo's mj_applyFT — matches mujoco_wasm main.js */
  private applyDragForce() {
    if (!this.dragActive || !this.dragBody || !this.mujoco || !this.model || !this.data) return;
    const bodyID = (this.dragBody as any).bodyID as number;
    if (!bodyID || bodyID <= 0) return;

    // Update body transforms before computing force
    for (let b = 0; b < this.model.nbody; b++) {
      const group = this.bodies.get(b);
      if (group) {
        getPosition(this.data.xpos, b, group.position);
        getQuaternion(this.data.xquat, b, group.quaternion);
        group.updateWorldMatrix(false, false);
      }
    }

    this.updateDragArrow();

    // Force = mass * 250 * (currentWorld - worldHit), in MuJoCo coords
    const force = toMujocoPos(
      this.currentWorld.clone().sub(this.worldHit)
        .multiplyScalar(this.model.body_mass[bodyID] * 250),
    );
    const point = toMujocoPos(this.worldHit.clone());

    this.mujoco.mj_applyFT(
      this.model, this.data,
      [force.x, force.y, force.z],
      [0, 0, 0],
      [point.x, point.y, point.z],
      bodyID,
      this.data.qfrc_applied,
    );
  }

  private resize() {
    const parent = this.canvas.parentElement;
    if (!parent) return;
    const w = parent.clientWidth, h = parent.clientHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  /** Position camera at a level side-view aimed at the robot's center. */
  private fitCameraToRobot() {
    if (!this.mujocoRoot) return;

    // Compute bounding box of all non-plane meshes (skip ground)
    const box = new THREE.Box3();
    this.mujocoRoot.traverse((obj) => {
      if (obj instanceof THREE.Mesh && !(obj.geometry instanceof THREE.PlaneGeometry)) {
        const meshBox = new THREE.Box3().setFromObject(obj);
        box.union(meshBox);
      }
    });
    if (box.isEmpty()) return;

    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    // Distance so the robot fills roughly 60% of the viewport
    const dist = maxDim * 2.0;

    // Side view: camera at same Y (height) as robot center, offset in X-Z
    this.camera.position.set(
      center.x + dist * 0.7,
      center.y,
      center.z + dist * 0.7,
    );
    this.controls.target.copy(center);
    this.controls.update();
  }

  /** Create/recreate the simulation GUI panel (controls + actuator sliders). */
  private buildSimGUI(
    container: HTMLElement,
    callbacks?: { onReload?: () => void; onStop?: () => void },
  ) {
    // Destroy old GUI
    if (this.simGui) {
      this.simGui.destroy();
      this.simGui = null;
    }
    this.actuatorGUIs = [];
    this.actuatorParams = {};

    const gui = new GUI({ container, title: 'Simulation', width: 270 });
    this.simGui = gui;

    // -- Simulation controls --
    const controlFolder = gui.addFolder('Controls');
    this.simParams.paused = false;
    controlFolder.add(this.simParams, 'paused').name('Paused').listen().onChange((v: boolean) => {
      this.paused = v;
    });
    controlFolder.add({ reset: () => this.reset() }, 'reset').name('Reset');

    // Reload with updated design params
    if (callbacks?.onReload) {
      const btn = controlFolder.add({ reload: callbacks.onReload }, 'reload').name('Reload Model');
      btn.domElement.closest('.controller')?.classList.add('primary');
    }
    // Back to design view
    if (callbacks?.onStop) {
      controlFolder.add({ stop: callbacks.onStop }, 'stop').name('Back to Design');
    }

    // -- Actuator sliders --
    this.actuatorFolder = gui.addFolder('Actuators');
    this.addActuatorSliders();
    this.actuatorFolder.open();
  }

  private addActuatorSliders() {
    if (!this.model || !this.data || !this.actuatorFolder) return;

    const model = this.model;
    const data = this.data;
    const textDecoder = new TextDecoder('utf-8');
    const nullChar = '\0';

    for (let i = 0; i < model.nu; i++) {
      if (!model.actuator_ctrllimited[i]) continue;

      // Read actuator name
      let name: string;
      try {
        name = textDecoder.decode(
          model.names.subarray(model.name_actuatoradr[i]),
        ).split(nullChar)[0];
      } catch {
        name = `Actuator ${i}`;
      }
      if (!name) name = `Actuator ${i}`;

      const rangeMin = model.actuator_ctrlrange[2 * i];
      const rangeMax = model.actuator_ctrlrange[2 * i + 1];

      this.actuatorParams[name] = 0.0;
      data.ctrl[i] = 0.0;

      const slider = this.actuatorFolder!.add(
        this.actuatorParams, name,
        rangeMin, rangeMax, 0.01,
      ).name(name).listen();

      slider.onChange((value: number) => {
        data.ctrl[i] = value;
      });

      this.actuatorGUIs.push(slider);
    }
  }

  /** Load XML string + STL binary into MuJoCo and build Three.js scene. */
  async load(
    xmlString: string,
    stlBuffer: ArrayBuffer,
    guiContainer: HTMLElement,
    callbacks?: { onReload?: () => void; onStop?: () => void },
  ) {
    this.stop();
    this.mujoco = await ensureMujoco();

    try {
      // Write files to virtual filesystem
      console.log('[MuJoCo] Writing XML (%d bytes) and STL (%d bytes) to VFS...', xmlString.length, stlBuffer.byteLength);
      this.mujoco.FS.writeFile('/working/robot.xml', xmlString);
      this.mujoco.FS.writeFile('/working/baselink.stl', new Uint8Array(stlBuffer));

      // Free old model/data
      if (this.data) { this.data.delete(); this.data = null; }
      if (this.model) { this.model = null; }

      console.log('[MuJoCo] Loading model from XML...');
      this.model = this.mujoco.MjModel.loadFromXML('/working/robot.xml');
      this.data = new this.mujoco.MjData(this.model);
      console.log('[MuJoCo] Model loaded: %d bodies, %d geoms, %d joints, %d tendons, %d actuators',
        this.model.nbody, this.model.ngeom, this.model.njnt, this.model.ntendon, this.model.nu);
    } catch (e) {
      console.error('[MuJoCo] Failed to load model:', e);
      console.error('[MuJoCo] XML preview:\n', xmlString.substring(0, 500));
      throw e;
    }

    // Clear old scene objects
    if (this.mujocoRoot) {
      this.scene.remove(this.mujocoRoot);
    }
    this.bodies.clear();

    // Build Three.js geometry from MuJoCo model
    this.mujocoRoot = new THREE.Group();
    this.mujocoRoot.name = 'MuJoCo Root';
    this.scene.add(this.mujocoRoot);

    try {
      this.buildGeometry();
      this.initTendonVis();

      // Forward pass to get initial state
      this.mujoco.mj_forward(this.model, this.data);
      this.updateBodies();
      this.drawTendons();
    } catch (e) {
      console.error('[MuJoCo] Failed to build geometry or initial state:', e);
    }

    // Position camera to face the robot at eye level
    this.fitCameraToRobot();

    // Build actuator GUI
    this.buildSimGUI(guiContainer, callbacks);

    this.resize();
  }

  private buildGeometry() {
    const model = this.model;
    const mujoco = this.mujoco!;

    for (let g = 0; g < model.ngeom; g++) {
      if (model.geom_group[g] >= 3) continue;

      const b = model.geom_bodyid[g];
      const type = model.geom_type[g];
      const size = [
        model.geom_size[g * 3 + 0],
        model.geom_size[g * 3 + 1],
        model.geom_size[g * 3 + 2],
      ];

      if (!this.bodies.has(b)) {
        const group = new THREE.Group();
        (group as any).bodyID = b;
        this.bodies.set(b, group);
      }

      const color = new THREE.Color(
        model.geom_rgba[g * 4 + 0],
        model.geom_rgba[g * 4 + 1],
        model.geom_rgba[g * 4 + 2],
      );
      const alpha = model.geom_rgba[g * 4 + 3];

      // Ground plane — match mujoco_wasm (large reflective plane)
      if (type === mujoco.mjtGeom.mjGEOM_PLANE.value) {
        const planeMat = new THREE.MeshPhysicalMaterial({
          color: 0x444455,
          side: THREE.DoubleSide,
          roughness: 0.7,
          metalness: 0.1,
        });
        const mesh = new THREE.Mesh(new THREE.PlaneGeometry(100, 100), planeMat);
        mesh.rotation.x = -Math.PI / 2;
        mesh.receiveShadow = true;
        this.bodies.get(b)!.add(mesh);
        continue;
      }

      // All other geometry types
      let geometry: THREE.BufferGeometry;
      if (type === mujoco.mjtGeom.mjGEOM_SPHERE.value) {
        geometry = new THREE.SphereGeometry(size[0], 20, 20);
      } else if (type === mujoco.mjtGeom.mjGEOM_CAPSULE.value) {
        geometry = new THREE.CapsuleGeometry(size[0], size[1] * 2, 20, 20);
      } else if (type === mujoco.mjtGeom.mjGEOM_CYLINDER.value) {
        geometry = new THREE.CylinderGeometry(size[0], size[0], size[1] * 2, 20);
      } else if (type === mujoco.mjtGeom.mjGEOM_BOX.value) {
        geometry = new THREE.BoxGeometry(size[0] * 2, size[2] * 2, size[1] * 2);
      } else if (type === mujoco.mjtGeom.mjGEOM_MESH.value) {
        const meshID = model.geom_dataid[g];
        geometry = this.buildMeshGeometry(meshID);
      } else {
        geometry = new THREE.SphereGeometry(size[0] * 0.5);
      }

      // MeshPhysicalMaterial — match mujoco_wasm reference for realistic rendering
      const material = new THREE.MeshPhysicalMaterial({
        color,
        transparent: alpha < 1,
        opacity: alpha,
        side: THREE.DoubleSide,
        roughness: 0.7,
        metalness: 0.1,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      (mesh as any).bodyID = b;

      // Geom position/orientation relative to body
      const pos = new THREE.Vector3(
        model.geom_pos[g * 3 + 0],
        model.geom_pos[g * 3 + 2],
        -model.geom_pos[g * 3 + 1],
      );
      const quat = new THREE.Quaternion(
        -model.geom_quat[g * 4 + 1],
        -model.geom_quat[g * 4 + 3],
        model.geom_quat[g * 4 + 2],
        -model.geom_quat[g * 4 + 0],
      );
      mesh.position.copy(pos);
      mesh.quaternion.copy(quat);
      this.bodies.get(b)!.add(mesh);
    }

    // Ensure all bodies have groups, even those without geoms
    for (let b = 0; b < model.nbody; b++) {
      if (!this.bodies.has(b)) {
        const group = new THREE.Group();
        (group as any).bodyID = b;
        this.bodies.set(b, group);
      }
    }

    // Build hierarchy: body 0 (worldbody) is child of mujocoRoot,
    // all other bodies are children of worldbody (flat, since xpos/xquat are world-frame)
    const worldBody = this.bodies.get(0)!;
    this.mujocoRoot!.add(worldBody);
    for (let b = 1; b < model.nbody; b++) {
      const group = this.bodies.get(b);
      if (group) worldBody.add(group);
    }
  }

  private buildMeshGeometry(meshID: number): THREE.BufferGeometry {
    const model = this.model;
    const geometry = new THREE.BufferGeometry();

    const vertStart = model.mesh_vertadr[meshID] * 3;
    const vertEnd = (model.mesh_vertadr[meshID] + model.mesh_vertnum[meshID]) * 3;
    // Make a copy so we don't mutate the MuJoCo buffer
    const vertexBuffer = new Float32Array(model.mesh_vert.subarray(vertStart, vertEnd));

    // Swizzle Y/Z for Three.js coordinate system
    for (let v = 0; v < vertexBuffer.length; v += 3) {
      const temp = vertexBuffer[v + 1];
      vertexBuffer[v + 1] = vertexBuffer[v + 2];
      vertexBuffer[v + 2] = -temp;
    }

    const faceStart = model.mesh_faceadr[meshID] * 3;
    const faceEnd = (model.mesh_faceadr[meshID] + model.mesh_facenum[meshID]) * 3;
    const faceBuffer = model.mesh_face.subarray(faceStart, faceEnd);

    geometry.setAttribute('position', new THREE.BufferAttribute(vertexBuffer, 3));
    geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(faceBuffer), 1));
    geometry.computeVertexNormals();
    return geometry;
  }

  // ---- Tendon visualization ----

  private initTendonVis() {
    if (!this.mujocoRoot || !this.model) return;
    // Match mujoco_wasm tendon color (0.8, 0.3, 0.3)
    const tendonMat = new THREE.MeshPhongMaterial({ color: new THREE.Color(0.8, 0.3, 0.3) });
    const maxInstances = 1023;
    const cylinders = new THREE.InstancedMesh(
      new THREE.CylinderGeometry(1, 1, 1), tendonMat, maxInstances,
    );
    const spheres = new THREE.InstancedMesh(
      new THREE.SphereGeometry(1, 10, 10), tendonMat, maxInstances,
    );
    cylinders.count = 0;
    spheres.count = 0;
    cylinders.castShadow = true;
    cylinders.receiveShadow = true;
    spheres.castShadow = true;
    spheres.receiveShadow = true;
    (this.mujocoRoot as any).cylinders = cylinders;
    (this.mujocoRoot as any).spheres = spheres;
    this.mujocoRoot.add(cylinders);
    this.mujocoRoot.add(spheres);
  }

  private drawTendons() {
    if (!this.mujocoRoot || !this.model || !this.data) return;
    const root = this.mujocoRoot as any;
    if (!root.cylinders || !root.spheres) return;

    const identityQuat = new THREE.Quaternion();
    const mat = new THREE.Matrix4();
    let numWraps = 0;

    for (let t = 0; t < this.model.ntendon; t++) {
      const startW = this.data.ten_wrapadr[t];
      const r = this.model.tendon_width[t];
      for (let w = startW; w < startW + this.data.ten_wrapnum[t] - 1; w++) {
        const tendonStart = getPosition(this.data.wrap_xpos, w, new THREE.Vector3());
        const tendonEnd = getPosition(this.data.wrap_xpos, w + 1, new THREE.Vector3());
        const tendonAvg = new THREE.Vector3().addVectors(tendonStart, tendonEnd).multiplyScalar(0.5);

        const validStart = tendonStart.length() > 0.01;
        const validEnd = tendonEnd.length() > 0.01;

        if (validStart) {
          root.spheres.setMatrixAt(numWraps, mat.compose(
            tendonStart, identityQuat, new THREE.Vector3(r, r, r)));
        }
        if (validEnd) {
          root.spheres.setMatrixAt(numWraps + 1, mat.compose(
            tendonEnd, identityQuat, new THREE.Vector3(r, r, r)));
        }
        if (validStart && validEnd) {
          mat.compose(
            tendonAvg,
            new THREE.Quaternion().setFromUnitVectors(
              new THREE.Vector3(0, 1, 0),
              tendonEnd.clone().sub(tendonStart).normalize(),
            ),
            new THREE.Vector3(r, tendonStart.distanceTo(tendonEnd), r),
          );
          root.cylinders.setMatrixAt(numWraps, mat);
          numWraps++;
        }
      }
    }

    root.cylinders.count = numWraps;
    root.spheres.count = numWraps > 0 ? numWraps + 1 : 0;
    root.cylinders.instanceMatrix.needsUpdate = true;
    root.spheres.instanceMatrix.needsUpdate = true;
  }

  // ---- Body transform updates ----

  private updateBodies() {
    const pos = new THREE.Vector3();
    const quat = new THREE.Quaternion();
    for (let b = 0; b < this.model.nbody; b++) {
      const group = this.bodies.get(b);
      if (!group) continue;
      getPosition(this.data.xpos, b, pos);
      getQuaternion(this.data.xquat, b, quat);
      group.position.copy(pos);
      group.quaternion.copy(quat);
    }
  }

  // ---- Simulation lifecycle ----

  start() {
    this.paused = false;
    this.simParams.paused = false;
    this.mujocoTime = performance.now();
    if (!this.animationId) {
      this.animate(performance.now());
    }
  }

  stop() {
    this.paused = true;
    this.simParams.paused = true;
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
  }

  togglePause() {
    this.paused = !this.paused;
    this.simParams.paused = this.paused;
    if (!this.paused) {
      this.mujocoTime = performance.now();
    }
  }

  reset() {
    if (this.mujoco && this.model && this.data) {
      this.mujoco.mj_resetData(this.model, this.data);
      // Reset actuator controls and GUI sliders
      for (let i = 0; i < this.model.nu; i++) {
        this.data.ctrl[i] = 0;
      }
      for (const key of Object.keys(this.actuatorParams)) {
        this.actuatorParams[key] = 0;
      }
      this.mujoco.mj_forward(this.model, this.data);
      this.updateBodies();
      this.drawTendons();
      this.mujocoTime = performance.now();
    }
  }

  /** Destroy the simulation GUI panel. */
  destroyGUI() {
    if (this.simGui) {
      this.simGui.destroy();
      this.simGui = null;
      this.actuatorFolder = null;
      this.actuatorGUIs = [];
    }
  }

  private animate(timeMS: number) {
    this.animationId = requestAnimationFrame((t) => this.animate(t));
    this.controls.update();

    try {
      if (!this.paused && this.mujoco && this.model && this.data) {
        const timestep = this.model.opt.timestep;
        // Prevent spiral of death
        if (timeMS - this.mujocoTime > 35) this.mujocoTime = timeMS;

        while (this.mujocoTime < timeMS) {
          // Clear old perturbations, apply new drag forces
          for (let i = 0; i < this.data.qfrc_applied.length; i++) {
            this.data.qfrc_applied[i] = 0.0;
          }
          if (this.dragActive) {
            this.applyDragForce();
          }

          this.mujoco.mj_step(this.model, this.data);
          this.mujocoTime += timestep * 1000;
        }

        this.updateBodies();
        this.drawTendons();
      }
    } catch (e) {
      console.error('[MuJoCo] Simulation step error:', e);
      this.paused = true;
      this.simParams.paused = true;
    }

    this.renderer.render(this.scene, this.camera);
  }

  dispose() {
    this.stop();
    this.destroyGUI();
    if (this.data) { this.data.delete(); this.data = null; }
    this.model = null;
    this.renderer.dispose();
  }
}
