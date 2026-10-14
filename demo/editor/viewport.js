// three.js viewport of the constraint editor (port of the upstream demo's viser
// scene: kimodo/viz/constraint_ui.py, kimodo/viz/playback.py).
//
// Draws, for the motion at the current frame:
// - the character as a "mannequin": one capsule per bone, rigidly attached to
//   the parent joint's frame (so its cross-section follows the bone's twist),
//   plus a head ellipsoid. Light blue (152,189,255) like the upstream light
//   theme, with adjustable opacity (a depth pre-pass keeps the translucent body
//   free of inner capsule overlaps);
// - the skeleton overlay (yellow joints r=0.02, green bones r=0.01, hand-chain
//   joints scaled 0.6 like upstream SkeletonMesh), shown while editing or when
//   showSkeleton is on, drawn on top of everything (depthTest off). With
//   showContacts the foot joints in contact turn red;
// - every constraint of a ConstraintStore: Full-Body ghosts (yellow
//   joints, red bones), End-Effector ghosts (grey skeleton, red root + chain,
//   RGB axes at the chain base), Hips pins (the same ghost; the Hips track has
//   no chain, so the root itself is marked: a larger red root joint, RGB axes
//   at the root for the pinned pelvis rotation and a red drop line to the
//   ground for the pinned hip height), 2D-Root waypoints (red sphere +
//   annulus, plus an arrow for the optional heading), red polylines for 2D-Root
//   intervals and the Smooth Path, CSS2D labels ("Full-Body @ f",
//   "End-Effectors @ f", "Hips @ f" when Hips is the only End-Effector track at
//   f, "2D Root @ f", "<Name> @ [s, e]"), and the blue start-direction marker.
//
// Constraint overlays are rebuilt only when the store changes ('change' event,
// with store.version as a fallback), when numFrames or the overlay options
// change, or on frame changes while onlyCurrent is on - never every frame.
// Ghost skeletons, waypoints and path segments are InstancedMeshes, so a long
// Full-Body interval (one ghost per frame) stays a handful of draw calls.
//
// Conventions: positions in metres, Y-up; quaternions xyzw (three.js order);
// colours are sRGB CSS strings (three.js converts them to its linear working
// space, unlike Color.setRGB which takes linear values).
//
// Additions to the core API: DEFAULT_OPTIONS, DEFAULT_VIEW, Viewport#options,
// #motion, #frame, #skeleton, #store, #pickWaypoint(), #setHighlight(),
// #setInteracting(), #jointScreenPosition(), #worldToClient(),
// #addRenderHook()/#removeRenderHook(), #dispose().

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/CSS2DRenderer.js';
import { EE_TRACKS } from './constraints.js';

/** Upstream camera (kimodo/demo/app.py:367-376). */
export const DEFAULT_VIEW = Object.freeze({ position: Object.freeze([2.7417, 1.879, 7.6757]), target: Object.freeze([0, 0, 0]), fov: 45 });

/** Viewport display options (upstream Visualize tab defaults). */
export const DEFAULT_OPTIONS = Object.freeze({
  showMesh: true,
  meshOpacity: 1,
  showSkeleton: false,
  showContacts: false,
  showLabels: true,
  showStartDirection: true,
  onlyCurrent: false,
  follow: false,
});

const COLORS = {
  background: 0x090d12, // LocalAI stage colour
  gridCenter: 0x2c5862,
  grid: 0x223039,
  mesh: 'rgb(152,189,255)', // upstream LIGHT_THEME mesh (viz/scene.py)
  skelJoint: 'rgb(255,235,0)', // upstream SkeletonMesh defaults
  skelBone: 'rgb(27,106,0)',
  contact: 'rgb(255,0,0)',
  fbJoint: 'rgb(255,235,0)', // Full-Body ghost (constraint_ui.py:272-279)
  fbBone: 'rgb(255,0,0)',
  eeGrey: 'rgb(220,220,220)', // End-Effector ghost (constraint_ui.py:433-495)
  eeRed: 'rgb(255,0,0)',
  axisX: 'rgb(255,0,0)', axisY: 'rgb(0,255,0)', axisZ: 'rgb(0,0,255)',
  waypoint: 'rgb(255,0,0)', // WaypointMesh default
  path: 'rgb(255,0,0)',
  start: 'rgb(0,0,255)', // start-direction marker (app.py:388-396)
  selected: '#3bd3d0', // LocalAI cyan
  hover: '#ffffff',
};
const COL = Object.fromEntries(Object.entries(COLORS).map(([k, v]) => [k, new THREE.Color(v)]));

// Overlay sizes (metres), upstream values.
const SKEL_JOINT_R = 0.02;
const SKEL_BONE_R = 0.01;
const FINGER_SCALE = 0.6;
const AXES_LEN = 0.07;
const AXES_R = 0.007;
const HIPS_ROOT_SCALE = 1.7; // Hips pin: root joint drawn larger than the ghost joints
const HIPS_AXES_LEN = 0.14; // ... with a larger triad (the pelvis sits inside the mannequin)
const WAYPOINT_SPHERE_R = 0.025;
const WAYPOINT_RING_OUTER = 0.2; // annulus 0.10-0.20
const ARROW_LEN = 0.3;
const ARROW_SHAFT_R = 0.01;
const ARROW_HEAD_R = 0.03;
const ARROW_HEAD_H = 0.075;
const PATH_R = 0.013; // "line width 5" (screen px upstream) as tubes ~3 px wide at the default camera distance
const PATH_Y = 0.015; // tube centre: just above the ground
const PICK_R = 0.04; // joint pick spheres are larger than the drawn ones
const PICK_PX = 14; // screen-space fallback pick radius

// Render order: opaque scene < mannequin depth pre-pass < mannequin colour <
// skeleton overlay (depthTest off) < TransformControls (renderOrder Infinity).
const ORDER_MESH_DEPTH = 5;
const ORDER_MESH = 6;
const ORDER_SKEL_BONES = 10;
const ORDER_SKEL_JOINTS = 11;

// Unit primitives shared by every instanced overlay.
const UNIT_SPHERE = new THREE.SphereGeometry(1, 18, 12);
const UNIT_CYLINDER = new THREE.CylinderGeometry(1, 1, 1, 12, 1); // along +Y, centred
const UNIT_CONE = new THREE.ConeGeometry(1, 1, 18).translate(0, 0.5, 0); // base at 0, tip at +Y 1
const UNIT_RING = new THREE.RingGeometry(0.5, 1, 56).rotateX(-Math.PI / 2); // flat on XZ, scaled by the outer radius

// ---------------------------------------------------------------------------
// mannequin bone sizes
// ---------------------------------------------------------------------------

// Per-bone capsule style keyed by the bone's CHILD joint name:
//   [radius, sx, sz, t0 = 0, t1 = 1] or null (no capsule); the capsule axis
//   runs from t0 * offset to t1 * offset. The capsule is built in the parent
//   joint's rest frame along offset[child]; its cross-section axes are
//   Z' = rest forward (+Z, or +Y for bones that point mostly forward/back)
//   made orthogonal to the bone, and X' = boneDir x Z'. sx / sz scale the
//   radius along X' / Z' (flattened torso, hands and feet).
const BONE_STYLES = {
  soma30: {
    // Torso = pelvis block + narrower waist + chest. Adjacent pieces differ
    // clearly in size: nearly coincident capsule surfaces intersect in jagged seams.
    // The chest stops near the shoulder line (t1 = 0.6) instead of reaching the neck.
    Spine1: [0.1, 1.38, 0.9], Spine2: null, Chest: [0.084, 1.32, 0.95], Neck1: [0.118, 1.38, 0.8, 0, 0.6],
    Neck2: [0.05, 1, 1], Head: [0.047, 1, 1], Jaw: null, LeftEye: null, RightEye: null,
    LeftShoulder: null, RightShoulder: null, // chest -> clavicle base: inside the torso capsule
    LeftArm: [0.048, 1, 1], RightArm: [0.048, 1, 1], // clavicles
    LeftForeArm: [0.047, 1, 1], RightForeArm: [0.047, 1, 1], // upper arms
    LeftHand: [0.039, 1, 1], RightHand: [0.039, 1, 1], // forearms
    LeftHandThumbEnd: [0.016, 1, 1], RightHandThumbEnd: [0.016, 1, 1],
    LeftHandMiddleEnd: [0.027, 0.72, 1.45], RightHandMiddleEnd: [0.027, 0.72, 1.45], // flat hands
    LeftLeg: [0.088, 1, 1], RightLeg: [0.088, 1, 1], // pelvis -> hip
    LeftShin: [0.074, 1, 1], RightShin: [0.074, 1, 1], // thighs
    LeftFoot: [0.052, 1, 1], RightFoot: [0.052, 1, 1], // shins
    LeftToeBase: [0.037, 1.18, 0.78], RightToeBase: [0.037, 1.18, 0.78], // feet
  },
  smplx22: {
    left_hip: [0.088, 1, 1], right_hip: [0.088, 1, 1], spine1: [0.1, 1.38, 0.9], spine2: [0.084, 1.32, 0.95],
    spine3: null, neck: [0.112, 1.3, 0.8], head: [0.047, 1, 1],
    left_knee: [0.074, 1, 1], right_knee: [0.074, 1, 1], left_ankle: [0.052, 1, 1], right_ankle: [0.052, 1, 1],
    left_foot: [0.037, 1.18, 0.78], right_foot: [0.037, 1.18, 0.78],
    left_collar: [0.07, 1, 1], right_collar: [0.07, 1, 1], left_shoulder: [0.052, 1, 1], right_shoulder: [0.052, 1, 1],
    left_elbow: [0.047, 1, 1], right_elbow: [0.047, 1, 1], left_wrist: [0.039, 1, 1], right_wrist: [0.039, 1, 1],
  },
};

// Name heuristics for other skeletons (G1, custom): first match wins.
const GENERIC_STYLES = [
  [/eye|jaw/i, null],
  [/thumb|index|middle|ring|pinky|finger|hand_roll|handend/i, [0.018, 1, 1]],
  [/toe/i, [0.036, 1.18, 0.78]],
  [/spine|chest|torso|waist|abdomen/i, [0.1, 1.25, 0.9]],
  [/neck|head/i, [0.05, 1, 1]],
  [/knee|shin|calf/i, [0.07, 1, 1]],
  [/ankle|foot/i, [0.05, 1, 1]],
  [/wrist|hand/i, [0.036, 1, 1]],
  [/elbow|forearm/i, [0.045, 1, 1]],
  [/hip|leg|thigh/i, [0.075, 1, 1]],
];

function boneStyle(skeleton, j) {
  const name = skeleton.names[j];
  const table = BONE_STYLES[skeleton.key];
  if (table && Object.prototype.hasOwnProperty.call(table, name)) return table[name];
  for (const [re, style] of GENERIC_STYLES) if (re.test(name)) return style;
  const o = skeleton.offsets[j];
  const len = Math.hypot(o[0], o[1], o[2]);
  return [Math.min(0.06, Math.max(0.02, 0.18 * len)), 1, 1];
}

/** Capsule along `offset` (parent rest frame), cross-section and extent per `style`. */
function boneCapsule(offset, [radius, sx, sz, t0 = 0, t1 = 1]) {
  const d = new THREE.Vector3(offset[0], offset[1], offset[2]);
  const len = d.length() * (t1 - t0);
  d.normalize();
  const ref = Math.abs(d.z) > 0.75 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(0, 0, 1);
  const zAxis = ref.addScaledVector(d, -ref.dot(d)).normalize();
  const xAxis = new THREE.Vector3().crossVectors(d, zAxis).normalize();
  const geom = new THREE.CapsuleGeometry(radius, len, 12, 32);
  geom.scale(sx, 1, sz);
  geom.applyMatrix4(new THREE.Matrix4().makeBasis(xAxis, d, zAxis));
  const mid = (t0 + t1) / 2;
  geom.translate(offset[0] * mid, offset[1] * mid, offset[2] * mid);
  return geom;
}

// ---------------------------------------------------------------------------
// instanced primitive batches
// ---------------------------------------------------------------------------

const _m4 = new THREE.Matrix4();
const _q = new THREE.Quaternion();
const _qId = new THREE.Quaternion();
const _p = new THREE.Vector3();
const _s = new THREE.Vector3();
const _d = new THREE.Vector3();
const _Y = new THREE.Vector3(0, 1, 0);
const _color = new THREE.Color();

function sphereMatrix(x, y, z, r) {
  return _m4.compose(_p.set(x, y, z), _qId, _s.set(r, r, r));
}

/** Unit cylinder stretched between a and b. */
function segmentMatrix(ax, ay, az, bx, by, bz, r) {
  _d.set(bx - ax, by - ay, bz - az);
  const len = _d.length();
  if (len < 1e-9) _q.identity();
  else _q.setFromUnitVectors(_Y, _d.multiplyScalar(1 / len));
  return _m4.compose(_p.set((ax + bx) / 2, (ay + by) / 2, (az + bz) / 2), _q, _s.set(r, len, r));
}

/** Unit cone (base at a) pointing along unit direction (dx, dy, dz). */
function coneMatrix(ax, ay, az, dx, dy, dz, r, h) {
  _q.setFromUnitVectors(_Y, _d.set(dx, dy, dz));
  return _m4.compose(_p.set(ax, ay, az), _q, _s.set(r, h, r));
}

/**
 * A growable batch of instances of one geometry/material: begin(), push()*,
 * end(). The InstancedMesh is recreated with a larger capacity when needed.
 */
class InstancedBatch {
  constructor(parent, geometry, material, { renderOrder = 0, name = '' } = {}) {
    this.parent = parent;
    this.geometry = geometry;
    this.material = material;
    this.renderOrder = renderOrder;
    this.name = name;
    this.mesh = null;
    this.capacity = 0;
    this.n = 0;
    this._m = new Float32Array(16 * 64);
    this._c = new Float32Array(3 * 64);
    this.tags = []; // optional per-instance payload (e.g. joint index for picking)
  }

  begin() { this.n = 0; this.tags.length = 0; }

  push(matrix, color, tag) {
    if ((this.n + 1) * 16 > this._m.length) {
      const m = new Float32Array(this._m.length * 2); m.set(this._m); this._m = m;
      const c = new Float32Array(this._c.length * 2); c.set(this._c); this._c = c;
    }
    matrix.toArray(this._m, this.n * 16);
    color.toArray(this._c, this.n * 3);
    if (tag !== undefined) this.tags[this.n] = tag;
    this.n++;
  }

  end() {
    const n = this.n;
    if (n > this.capacity) {
      if (this.mesh) { this.mesh.removeFromParent(); this.mesh.dispose(); }
      this.capacity = Math.max(64, 2 ** Math.ceil(Math.log2(n)));
      const mesh = new THREE.InstancedMesh(this.geometry, this.material, this.capacity);
      mesh.name = this.name;
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(this.capacity * 3), 3);
      mesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
      mesh.frustumCulled = false;
      mesh.renderOrder = this.renderOrder;
      if (this.parent) this.parent.add(mesh);
      this.mesh = mesh;
    }
    const mesh = this.mesh;
    if (!mesh) return;
    mesh.count = n;
    mesh.visible = n > 0;
    if (n > 0) {
      mesh.instanceMatrix.array.set(this._m.subarray(0, n * 16));
      mesh.instanceColor.array.set(this._c.subarray(0, n * 3));
      mesh.instanceMatrix.clearUpdateRanges();
      mesh.instanceColor.clearUpdateRanges();
      mesh.instanceMatrix.addUpdateRange(0, n * 16);
      mesh.instanceColor.addUpdateRange(0, n * 3);
      mesh.instanceMatrix.needsUpdate = true;
      mesh.instanceColor.needsUpdate = true;
      mesh.computeBoundingSphere();
    }
  }

  dispose() {
    if (this.mesh) { this.mesh.removeFromParent(); this.mesh.dispose(); this.mesh = null; }
    this.capacity = 0;
  }
}

// ---------------------------------------------------------------------------
// labels
// ---------------------------------------------------------------------------

const LABEL_STYLE_ID = 'kimodo-viewport-style';
function injectStyle() {
  if (typeof document === 'undefined' || document.getElementById(LABEL_STYLE_ID)) return;
  const style = document.createElement('style');
  style.id = LABEL_STYLE_ID;
  style.textContent = `
.kimodo-viewport { position: relative; overflow: hidden; background: #090d12; }
.kimodo-viewport > canvas { display: block; width: 100%; height: 100%; outline: none; touch-action: none; }
.kimodo-viewport-labels { position: absolute; inset: 0; pointer-events: none; overflow: hidden; }
.kimodo-label { font: 600 11px/1.3 Inter, system-ui, sans-serif; color: #f3f6f4; background: rgba(11, 16, 21, .86);
  border: 1px solid #2c3a45; border-radius: 4px; padding: 1px 6px; white-space: nowrap; pointer-events: none;
  box-shadow: 0 2px 8px rgba(0, 0, 0, .45); letter-spacing: .01em; }
.kimodo-label.root2d { border-color: rgba(255, 70, 70, .55); }
.kimodo-label.hips { border-color: rgba(232, 92, 80, .6); }
.kimodo-label.interval { color: #cfe7ee; border-style: dashed; }
`;
  document.head.appendChild(style);
}

// CSS2DObject.center: (0.5, 0.5) centres the element on the point; y = 1 puts
// the element's bottom edge on it.
const ANCHOR = { center: [0.5, 0.5], bottomCenter: [0.5, 1], bottomLeft: [0, 1], topCenter: [0.5, 0] };

// ---------------------------------------------------------------------------
// Viewport
// ---------------------------------------------------------------------------

export class Viewport {
  constructor(canvasContainer, skeleton) {
    if (!canvasContainer) throw new Error('Viewport: container element required');
    injectStyle();
    this.container = canvasContainer;
    canvasContainer.classList.add('kimodo-viewport');
    if (getComputedStyle(canvasContainer).position === 'static') canvasContainer.style.position = 'relative';

    this._options = { ...DEFAULT_OPTIONS };
    this._motion = null;
    this._frame = 0;
    this._store = null;
    this._numFrames = null;
    this._onStoreChange = () => { this._overlayDirty = true; };
    this._overlayDirty = true;
    this._overlayKey = null; // store version / numFrames / frame the overlays were built for
    this._charDirty = true;
    this._drawnMotionVersion = -1;
    this._drawnFrame = -1;
    this._highlight = { hover: -1, selected: -1 };
    this._interacting = false;
    this._pathCache = null; // last Smooth Path polyline, reused while interacting
    this._lastTime = null;
    this._renderHooks = new Set();

    // renderer + label layer
    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.setClearColor(COLORS.background, 1);
    renderer.domElement.tabIndex = 0;
    canvasContainer.appendChild(renderer.domElement);
    this.renderer = renderer;

    const labelRenderer = new CSS2DRenderer();
    labelRenderer.domElement.className = 'kimodo-viewport-labels';
    // The label layer must never swallow pointer events meant for the canvas.
    labelRenderer.domElement.style.pointerEvents = 'none';
    canvasContainer.appendChild(labelRenderer.domElement);
    this.labelRenderer = labelRenderer;

    // scene, camera, controls
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(COLORS.background);
    scene.fog = new THREE.Fog(COLORS.background, 10, 30);
    this.scene = scene;

    const camera = new THREE.PerspectiveCamera(DEFAULT_VIEW.fov, 1, 0.02, 250);
    this.camera = camera;
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.12;
    controls.maxPolarAngle = Math.PI * 0.495; // stay above the ground
    controls.minDistance = 0.3;
    controls.maxDistance = 60;
    controls.zoomToCursor = false;
    this.controls = controls;

    // lights: sky/ground fill + a key light with soft shadows that follows the character
    scene.add(new THREE.HemisphereLight(0xe2ecff, 0x1a222b, 1.9));
    const key = new THREE.DirectionalLight(0xffffff, 2.1);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    const sc = key.shadow.camera;
    sc.left = -3; sc.right = 3; sc.top = 3; sc.bottom = -3; sc.near = 0.5; sc.far = 25;
    key.shadow.bias = -0.0004;
    key.shadow.normalBias = 0.02;
    scene.add(key, key.target);
    this._keyLight = key;
    this._keyOffset = new THREE.Vector3(3.5, 7.5, 4.5);
    const rim = new THREE.DirectionalLight(0x9fdcff, 0.55); // cool back light separates the body from the dark background
    rim.position.set(-4, 3, -5);
    scene.add(rim);

    // ground: shadow catcher + 20 m grid with 0.5 m cells (fades into the fog)
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(400, 400), new THREE.ShadowMaterial({ opacity: 0.42 }));
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    ground.name = 'ground';
    scene.add(ground);
    const grid = new THREE.GridHelper(20, 40, COLORS.gridCenter, COLORS.grid);
    grid.position.y = 0.001;
    grid.material.transparent = true;
    grid.material.opacity = 0.9;
    grid.material.depthWrite = false;
    grid.name = 'grid';
    scene.add(grid);
    this.grid = grid;

    // groups
    this._character = new THREE.Group(); this._character.name = 'character';
    this._skeletonGroup = new THREE.Group(); this._skeletonGroup.name = 'skeleton-overlay';
    this._overlay = new THREE.Group(); this._overlay.name = 'constraints';
    this._labels = new THREE.Group(); this._labels.name = 'labels'; // CSS2DObjects only, as direct children
    this._startMarker = this._buildStartMarker();
    scene.add(this._character, this._skeletonGroup, this._overlay, this._labels, this._startMarker);
    this._labelMap = new Map();
    this._labelPass = 0;

    // materials
    this._meshMat = new THREE.MeshStandardMaterial({ color: COL.mesh, roughness: 0.6, metalness: 0.02 });
    this._depthMat = new THREE.MeshBasicMaterial({ colorWrite: false, transparent: true });
    const skelMat = () => new THREE.MeshLambertMaterial({ depthTest: false, depthWrite: false, transparent: true, fog: false });
    this._ghostMat = new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0 });
    this._ringMat = new THREE.MeshStandardMaterial({ roughness: 0.7, metalness: 0, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: -2 });

    // instanced batches
    this._skelJoints = new InstancedBatch(this._skeletonGroup, UNIT_SPHERE, skelMat(), { renderOrder: ORDER_SKEL_JOINTS, name: 'skeleton-joints' });
    this._skelBones = new InstancedBatch(this._skeletonGroup, UNIT_CYLINDER, skelMat(), { renderOrder: ORDER_SKEL_BONES, name: 'skeleton-bones' });
    this._pick = new InstancedBatch(null, UNIT_SPHERE, new THREE.MeshBasicMaterial(), { name: 'joint-pick' });
    this._ghostSpheres = new InstancedBatch(this._overlay, UNIT_SPHERE, this._ghostMat, { name: 'ghost-joints' });
    this._ghostSegments = new InstancedBatch(this._overlay, UNIT_CYLINDER, this._ghostMat, { name: 'ghost-bones' });
    this._ghostCones = new InstancedBatch(this._overlay, UNIT_CONE, this._ghostMat, { name: 'arrow-heads' });
    this._rings = new InstancedBatch(this._overlay, UNIT_RING, this._ringMat, { name: 'waypoint-rings' });
    this._waypoints = []; // [{frame, x, z}] drawn this pass (for pickWaypoint)

    this._raycaster = new THREE.Raycaster();
    this.setSkeleton(skeleton);
    this.resetCamera();

    // size tracking
    this._resizeObserver = new ResizeObserver(() => this._resize());
    this._resizeObserver.observe(canvasContainer);
    this._resize();
  }

  // -------------------------------------------------------------------------
  // core API
  // -------------------------------------------------------------------------

  get domElement() { return this.renderer.domElement; }
  get options() { return { ...this._options }; }
  get motion() { return this._motion; }
  get frame() { return this._frame; }
  get skeleton() { return this._skeleton; }
  get store() { return this._store; }

  setSkeleton(skeleton) {
    if (!skeleton) throw new Error('Viewport: skeleton required');
    this._skeleton = skeleton;
    if (this._motion && this._motion.skeleton !== skeleton && this._motion.skeleton?.J !== skeleton.J) this._motion = null;
    const J = skeleton.J;
    // Upstream SkeletonMesh scales hand-chain joints other than LeftHand/RightHand by 0.6.
    this._jointScale = new Float32Array(J).fill(1);
    for (const key of ['LeftHand', 'RightHand']) {
      for (const j of skeleton.chains[key] ?? []) if (!/^(LeftHand|RightHand)$/.test(skeleton.names[j])) this._jointScale[j] = FINGER_SCALE;
    }
    // Foot-contact channels [L heel, L toe, R heel, R toe] -> foot chain joints
    // (upstream foot_joint_idx = left foot chain + right foot chain).
    const lf = skeleton.chains.LeftFoot ?? [], rf = skeleton.chains.RightFoot ?? [];
    this._contactJoints = [lf[0], lf.at(-1), rf[0], rf.at(-1)].map((j) => (j === undefined ? -1 : j));
    this._buildMannequin();
    this._charDirty = true;
    this._overlayDirty = true;
  }

  setMotion(motion) {
    if (motion && motion.skeleton && motion.skeleton.J !== this._skeleton.J) {
      throw new Error(`Viewport.setMotion: motion has ${motion.skeleton.J} joints, skeleton has ${this._skeleton.J}`);
    }
    this._motion = motion || null;
    if (this._motion) this._frame = Math.min(this._frame, this._motion.frames - 1);
    this._charDirty = true;
  }

  setFrame(f) {
    const n = this._motion ? this._motion.frames : Infinity;
    f = Math.max(0, Math.min(n - 1, Math.round(Number(f) || 0)));
    if (f === this._frame) return;
    this._frame = f;
    this._charDirty = true;
    if (this._options.onlyCurrent) this._overlayDirty = true;
  }

  /** Shows the constraints of `store` (a ConstraintStore, or null) for a timeline of numFrames. */
  setConstraints(store, numFrames) {
    if (store !== this._store) {
      this._store?.removeEventListener('change', this._onStoreChange);
      store?.addEventListener('change', this._onStoreChange);
      this._store = store || null;
      this._pathCache = null;
    }
    const n = numFrames == null ? null : Math.round(Number(numFrames));
    if (n !== this._numFrames) { this._numFrames = n; this._pathCache = null; }
    this._overlayDirty = true;
  }

  setOptions(opts = {}) {
    const prev = this._options;
    const next = { ...prev };
    for (const k of Object.keys(DEFAULT_OPTIONS)) if (opts[k] !== undefined) next[k] = opts[k];
    const opacity = Number(next.meshOpacity);
    next.meshOpacity = Number.isFinite(opacity) ? Math.max(0, Math.min(1, opacity)) : prev.meshOpacity;
    this._options = next;
    if (next.meshOpacity !== prev.meshOpacity || next.showMesh !== prev.showMesh) this._applyMeshMaterial();
    if (next.showSkeleton !== prev.showSkeleton || next.showContacts !== prev.showContacts) this._charDirty = true;
    if (next.onlyCurrent !== prev.onlyCurrent) this._overlayDirty = true;
    this._labels.visible = next.showLabels;
    this._startMarker.visible = next.showStartDirection;
    return this.options;
  }

  /** Joint index under the pointer (skeleton overlay of the current pose), or -1. Hidden gizmo joints are skipped. */
  pickJoint(clientX, clientY) {
    if (!this._motion || this._pick.n === 0) return -1;
    this._pointerRay(clientX, clientY);
    const mesh = this._pick.mesh;
    if (mesh) {
      const hits = this._raycaster.intersectObject(mesh, false);
      for (const h of hits) {
        const j = this._pick.tags[h.instanceId];
        if (j !== undefined) return j;
      }
    }
    // Fallback: nearest projected joint within PICK_PX pixels (closest to the camera on ties).
    const pose = this._motion.pose(this._frame);
    const rect = this.domElement.getBoundingClientRect();
    let best = -1, bestD = PICK_PX, bestZ = Infinity;
    for (let j = 0; j < this._skeleton.J; j++) {
      if (this._skeleton.hiddenGizmoJoints.has(j)) continue;
      _p.set(pose.positions[3 * j], pose.positions[3 * j + 1], pose.positions[3 * j + 2]).project(this.camera);
      if (_p.z < -1 || _p.z > 1) continue;
      const x = rect.left + (_p.x + 1) / 2 * rect.width, y = rect.top + (1 - _p.y) / 2 * rect.height;
      const dist = Math.hypot(x - clientX, y - clientY);
      if (dist < bestD - 1e-6 || (Math.abs(dist - bestD) < 2 && _p.z < bestZ)) { best = j; bestD = dist; bestZ = _p.z; }
    }
    return best;
  }

  /** Frame of the 2D-Root waypoint (drawn this pass) under the pointer, or -1. */
  pickWaypoint(clientX, clientY) {
    if (!this._waypoints.length) return -1;
    this._pointerRay(clientX, clientY);
    const ray = this._raycaster.ray;
    if (Math.abs(ray.direction.y) < 1e-6) return -1;
    const t = -ray.origin.y / ray.direction.y;
    if (t < 0) return -1;
    const x = ray.origin.x + t * ray.direction.x, z = ray.origin.z + t * ray.direction.z;
    let best = -1, bestD = WAYPOINT_RING_OUTER;
    for (const w of this._waypoints) {
      const d = Math.hypot(w.x - x, w.z - z);
      if (d <= bestD) { best = w.frame; bestD = d; }
    }
    return best;
  }

  /** Upstream start view; with `follow` on, offset to the character's pelvis. */
  resetCamera() {
    const [px, py, pz] = DEFAULT_VIEW.position;
    const [tx, ty, tz] = DEFAULT_VIEW.target;
    let ox = 0, oz = 0;
    if (this._options.follow && this._motion) [ox, oz] = this._motion.rootXZ(this._frame);
    this.camera.fov = DEFAULT_VIEW.fov;
    this.camera.position.set(px + ox, py, pz + oz);
    this.controls.target.set(tx + ox, ty, tz + oz);
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  /** Draws one frame. Call from the app's requestAnimationFrame loop. */
  render() {
    const now = performance.now();
    const dt = this._lastTime === null ? 1 / 60 : Math.min(0.1, (now - this._lastTime) / 1000);
    this._lastTime = now;
    if (this._needsResize) this._resize();
    for (const hook of this._renderHooks) hook(dt);

    const m = this._motion;
    if (m && (this._charDirty || m.version !== this._drawnMotionVersion || this._frame !== this._drawnFrame)) this._drawCharacter();
    else if (!m && this._charDirty) this._drawCharacter();

    const key = this._store ? `${this._store.version}|${this._numFrames}|${this._options.onlyCurrent ? this._frame : ''}` : 'none';
    if (this._overlayDirty || key !== this._overlayKey) { this._rebuildOverlays(); this._overlayKey = key; }

    this._updateFollow(dt);
    this.controls.update(dt);
    this.renderer.render(this.scene, this.camera);
    this.labelRenderer.render(this.scene, this.camera);
  }

  // -------------------------------------------------------------------------
  // additions used by the editor / app
  // -------------------------------------------------------------------------

  /** Highlights joints in the skeleton overlay: {hover, selected} (-1 = none). */
  setHighlight({ hover = this._highlight.hover, selected = this._highlight.selected } = {}) {
    if (hover === this._highlight.hover && selected === this._highlight.selected) return;
    this._highlight = { hover, selected };
    this._charDirty = true;
  }

  /**
   * While a gizmo is dragged the Smooth Path polyline is not recomputed (the
   * smoother costs ~20 ms at 300 frames); it is redrawn when the drag ends.
   */
  setInteracting(on) {
    on = !!on;
    if (on === this._interacting) return;
    this._interacting = on;
    if (!on) this._overlayDirty = true;
  }

  /** fn(dt) runs at the start of every render() (the editor keeps its gizmos on the pose this way). */
  addRenderHook(fn) { this._renderHooks.add(fn); }

  removeRenderHook(fn) { this._renderHooks.delete(fn); }

  /** Client (CSS pixel) coordinates of a world point, or null when behind the camera. */
  worldToClient(x, y, z) {
    _p.set(x, y, z).project(this.camera);
    if (_p.z < -1 || _p.z > 1) return null;
    const rect = this.domElement.getBoundingClientRect();
    return { x: rect.left + (_p.x + 1) / 2 * rect.width, y: rect.top + (1 - _p.y) / 2 * rect.height };
  }

  /** Client coordinates of joint j at the current frame (null without a motion). */
  jointScreenPosition(j) {
    if (!this._motion) return null;
    const p = this._motion.pose(this._frame).positions;
    return this.worldToClient(p[3 * j], p[3 * j + 1], p[3 * j + 2]);
  }

  dispose() {
    this._resizeObserver.disconnect();
    this._store?.removeEventListener('change', this._onStoreChange);
    for (const obj of [...this._labels.children]) obj.removeFromParent();
    this.controls.dispose();
    for (const b of [this._skelJoints, this._skelBones, this._pick, this._ghostSpheres, this._ghostSegments, this._ghostCones, this._rings]) b.dispose();
    this._disposeMannequin();
    this.renderer.dispose();
    this.renderer.domElement.remove();
    this.labelRenderer.domElement.remove();
  }

  // -------------------------------------------------------------------------
  // internals: layout
  // -------------------------------------------------------------------------

  _resize() {
    this._needsResize = false;
    const w = Math.max(1, this.container.clientWidth), h = Math.max(1, this.container.clientHeight);
    if (w === this._w && h === this._h) return;
    this._w = w; this._h = h;
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = '100%';
    this.renderer.domElement.style.height = '100%';
    this.labelRenderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  _pointerRay(clientX, clientY) {
    const rect = this.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1);
    this._raycaster.setFromCamera(ndc, this.camera);
  }

  /** Smoothly keeps the orbit target on the pelvis (x, z) when `follow` is on. */
  _updateFollow(dt) {
    const m = this._motion;
    if (!m) return;
    const [x, z] = m.rootXZ(this._frame);
    // The key light and its shadow frustum always follow the character.
    this._keyLight.target.position.set(x, 0, z);
    this._keyLight.position.set(x + this._keyOffset.x, this._keyOffset.y, z + this._keyOffset.z);
    if (!this._options.follow) return;
    const t = this.controls.target;
    const k = 1 - Math.exp(-dt * 4); // exponential smoothing, ~0.25 s time constant
    const dx = (x - t.x) * k, dz = (z - t.z) * k;
    t.x += dx; t.z += dz;
    this.camera.position.x += dx; this.camera.position.z += dz;
  }

  // -------------------------------------------------------------------------
  // internals: character
  // -------------------------------------------------------------------------

  _disposeMannequin() {
    for (const s of this._segments ?? []) {
      s.mesh.removeFromParent(); s.depth.removeFromParent();
      s.mesh.geometry.dispose();
    }
    this._segments = [];
  }

  /** One capsule per bone (attached to the parent joint) + a head ellipsoid. */
  _buildMannequin() {
    this._disposeMannequin();
    const sk = this._skeleton;
    const add = (geom, joint) => {
      const mesh = new THREE.Mesh(geom, this._meshMat);
      mesh.castShadow = true;
      mesh.matrixAutoUpdate = false;
      mesh.renderOrder = ORDER_MESH;
      const depth = new THREE.Mesh(geom, this._depthMat);
      depth.matrixAutoUpdate = false;
      depth.renderOrder = ORDER_MESH_DEPTH;
      depth.visible = false;
      this._character.add(depth, mesh);
      this._segments.push({ mesh, depth, joint });
    };
    for (let j = 0; j < sk.J; j++) {
      const p = sk.parents[j];
      if (p < 0) continue;
      const style = boneStyle(sk, j);
      const o = sk.offsets[j];
      if (!style || Math.hypot(o[0], o[1], o[2]) < 1e-4) continue;
      add(boneCapsule(o, style), p);
    }
    // Head: ellipsoid in the head joint's own frame (eyes sit ~5 cm above it on SOMA).
    const head = sk.names.findIndex((n) => /^head$/i.test(n));
    if (head >= 0) {
      const geom = new THREE.SphereGeometry(1, 28, 20);
      geom.scale(0.083, 0.106, 0.098);
      geom.translate(0, 0.072, 0.018);
      add(geom, head);
    }
    this._applyMeshMaterial();
  }

  /**
   * Opacity 1: plain opaque material. Below 1: a colour-less depth pre-pass
   * (transparent queue, before the colour pass) makes only the front-most
   * surface blend, so overlapping capsules do not darken the joints.
   */
  _applyMeshMaterial() {
    const { meshOpacity, showMesh } = this._options;
    const translucent = meshOpacity < 0.999;
    const mat = this._meshMat;
    if (mat.transparent !== translucent) { mat.transparent = translucent; mat.needsUpdate = true; }
    mat.opacity = translucent ? meshOpacity : 1;
    mat.depthWrite = !translucent;
    for (const s of this._segments) {
      s.mesh.visible = showMesh && meshOpacity > 0.001;
      s.mesh.castShadow = meshOpacity > 0.3;
      s.depth.visible = s.mesh.visible && translucent;
    }
  }

  _drawCharacter() {
    this._charDirty = false;
    const m = this._motion;
    const sk = this._skeleton;
    if (!m) {
      this._character.visible = false;
      this._skelJoints.begin(); this._skelJoints.end();
      this._skelBones.begin(); this._skelBones.end();
      this._pick.begin(); this._pick.end();
      return;
    }
    this._character.visible = true;
    const f = this._frame;
    const pose = m.pose(f);
    const P = pose.positions, G = pose.globals;
    this._drawnMotionVersion = m.version;
    this._drawnFrame = f;

    // mannequin: segment matrix = T(pos[joint]) * R(global[joint])
    for (const s of this._segments) {
      const j = s.joint;
      _q.set(G[4 * j], G[4 * j + 1], G[4 * j + 2], G[4 * j + 3]);
      s.mesh.matrix.compose(_p.set(P[3 * j], P[3 * j + 1], P[3 * j + 2]), _q, _s.set(1, 1, 1));
      s.depth.matrix.copy(s.mesh.matrix);
      s.mesh.matrixWorldNeedsUpdate = true;
      s.depth.matrixWorldNeedsUpdate = true;
    }

    // pick spheres (always, so the editor can pick even before the overlay is shown)
    this._pick.begin();
    for (let j = 0; j < sk.J; j++) {
      if (sk.hiddenGizmoJoints.has(j)) continue;
      this._pick.push(sphereMatrix(P[3 * j], P[3 * j + 1], P[3 * j + 2], PICK_R), COL.hover, j);
    }
    this._pick.end();

    // skeleton overlay
    const show = this._options.showSkeleton;
    this._skeletonGroup.visible = show;
    this._skelJoints.begin();
    this._skelBones.begin();
    if (show) {
      const contact = new Set();
      if (this._options.showContacts && m.contacts) {
        for (let k = 0; k < 4; k++) if (m.contacts[f * 4 + k] > 0.5 && this._contactJoints[k] >= 0) contact.add(this._contactJoints[k]);
      }
      const { hover, selected } = this._highlight;
      for (let j = 0; j < sk.J; j++) {
        let r = SKEL_JOINT_R * this._jointScale[j];
        let c = contact.has(j) ? COL.contact : COL.skelJoint;
        if (j === selected) { c = COL.selected; r = Math.max(r, SKEL_JOINT_R) * 1.6; }
        else if (j === hover && !sk.hiddenGizmoJoints.has(j)) { c = COL.hover; r = Math.max(r, SKEL_JOINT_R) * 1.35; }
        this._skelJoints.push(sphereMatrix(P[3 * j], P[3 * j + 1], P[3 * j + 2], r), c);
        const p = sk.parents[j];
        if (p >= 0) this._skelBones.push(segmentMatrix(P[3 * p], P[3 * p + 1], P[3 * p + 2], P[3 * j], P[3 * j + 1], P[3 * j + 2], SKEL_BONE_R), COL.skelBone);
      }
    }
    this._skelJoints.end();
    this._skelBones.end();
  }

  // -------------------------------------------------------------------------
  // internals: constraint overlays
  // -------------------------------------------------------------------------

  _buildStartMarker() {
    const g = new THREE.Group();
    g.name = 'start-direction';
    const mat = new THREE.MeshStandardMaterial({ color: COL.start, roughness: 0.6, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: -1 });
    const ring = new THREE.Mesh(UNIT_RING, mat);
    ring.scale.setScalar(WAYPOINT_RING_OUTER);
    ring.position.y = 0.003;
    const dot = new THREE.Mesh(UNIT_SPHERE, mat);
    dot.scale.setScalar(WAYPOINT_SPHERE_R);
    dot.position.y = WAYPOINT_SPHERE_R * 0.5;
    const shaft = new THREE.Mesh(UNIT_CYLINDER, mat);
    shaft.scale.set(ARROW_SHAFT_R, ARROW_LEN, ARROW_SHAFT_R);
    shaft.rotation.x = Math.PI / 2; // +Y -> +Z
    shaft.position.set(0, 0.02, ARROW_LEN / 2);
    const head = new THREE.Mesh(UNIT_CONE, mat);
    head.scale.set(ARROW_HEAD_R, ARROW_HEAD_H, ARROW_HEAD_R);
    head.rotation.x = Math.PI / 2;
    head.position.set(0, 0.02, ARROW_LEN);
    g.add(ring, dot, shaft, head);
    return g;
  }

  _label(key, text, x, y, z, anchor, cls = '') {
    let obj = this._labelMap.get(key);
    if (!obj) {
      const el = document.createElement('div');
      obj = new CSS2DObject(el);
      this._labels.add(obj); // direct child: removeFromParent() also removes the DOM element
      this._labelMap.set(key, obj);
    }
    const el = obj.element;
    const className = `kimodo-label${cls ? ` ${cls}` : ''}`;
    if (el.className !== className) el.className = className;
    if (el.textContent !== text) el.textContent = text;
    obj.position.set(x, y, z);
    obj.center.set(anchor[0], anchor[1]);
    obj.userData.pass = this._labelPass;
  }

  /** Ghost skeleton of a stored pose; jointColor(j) / boneColor(j) give the colours (bone j = parent(j) -> j). */
  _ghost(pose, jointColor, boneColor) {
    const sk = this._skeleton;
    const P = pose.positions;
    for (let j = 0; j < sk.J; j++) {
      this._ghostSpheres.push(sphereMatrix(P[3 * j], P[3 * j + 1], P[3 * j + 2], SKEL_JOINT_R * this._jointScale[j]), jointColor(j));
      const p = sk.parents[j];
      if (p >= 0) this._ghostSegments.push(segmentMatrix(P[3 * p], P[3 * p + 1], P[3 * p + 2], P[3 * j], P[3 * j + 1], P[3 * j + 2], SKEL_BONE_R), boneColor(j));
    }
  }

  /** RGB axes triad (upstream add_batched_axes) at joint j of a stored pose. */
  _axes(pose, j, len = AXES_LEN) {
    const P = pose.positions, G = pose.globals;
    _q.set(G[4 * j], G[4 * j + 1], G[4 * j + 2], G[4 * j + 3]);
    const x = P[3 * j], y = P[3 * j + 1], z = P[3 * j + 2];
    const axis = new THREE.Vector3();
    [[1, 0, 0, COL.axisX], [0, 1, 0, COL.axisY], [0, 0, 1, COL.axisZ]].forEach(([ax, ay, az, c]) => {
      axis.set(ax, ay, az).applyQuaternion(_q).multiplyScalar(len);
      this._ghostSegments.push(segmentMatrix(x, y, z, x + axis.x, y + axis.y, z + axis.z, AXES_R), c);
    });
  }

  /** Red tube polyline through [[x, z], ...] on the ground (null points split the line). */
  _polyline(points) {
    let prev = null;
    for (const p of points) {
      if (p && prev) this._ghostSegments.push(segmentMatrix(prev[0], PATH_Y, prev[1], p[0], PATH_Y, p[1], PATH_R), COL.path);
      prev = p;
    }
  }

  _waypoint(frame, x, z, heading) {
    this._ghostSpheres.push(sphereMatrix(x, WAYPOINT_SPHERE_R * 0.5, z, WAYPOINT_SPHERE_R), COL.waypoint);
    this._rings.push(_m4.compose(_p.set(x, 0.004, z), _qId, _s.setScalar(WAYPOINT_RING_OUTER)), COL.waypoint);
    if (heading !== null && heading !== undefined) {
      // Facing direction of heading theta is (sin theta, 0, cos theta) (upstream compute_heading_angle).
      const dx = Math.sin(heading), dz = Math.cos(heading), y = 0.02;
      this._ghostSegments.push(segmentMatrix(x, y, z, x + dx * ARROW_LEN, y, z + dz * ARROW_LEN, ARROW_SHAFT_R), COL.waypoint);
      this._ghostCones.push(coneMatrix(x + dx * ARROW_LEN, y, z + dz * ARROW_LEN, dx, 0, dz, ARROW_HEAD_R, ARROW_HEAD_H), COL.waypoint);
    }
    this._waypoints.push({ frame, x, z });
  }

  _rebuildOverlays() {
    this._overlayDirty = false;
    this._labelPass++;
    this._waypoints = [];
    for (const b of [this._ghostSpheres, this._ghostSegments, this._ghostCones, this._rings]) b.begin();

    const store = this._store;
    // A store of another skeleton (model switch in progress) is not drawn.
    if (store && store.skeleton?.J === this._skeleton.J) this._buildConstraintOverlays(store);

    for (const b of [this._ghostSpheres, this._ghostSegments, this._ghostCones, this._rings]) b.end();
    // Drop labels that were not produced in this pass.
    for (const [key, obj] of this._labelMap) {
      if (obj.userData.pass !== this._labelPass) { obj.removeFromParent(); this._labelMap.delete(key); }
    }
  }

  _buildConstraintOverlays(store) {
    const sk = this._skeleton;
    const N = this._numFrames ?? store.numFrames ?? Infinity;
    const only = this._options.onlyCurrent;
    const cur = this._frame;
    const vis = (f) => f < N && (!only || f === cur);
    const root = sk.rootIndex;

    // Full-Body ghosts: yellow joints, red bones (one per constrained frame, intervals included).
    const fb = new Map();
    for (const e of store.entries('fullbody')) {
      fb.set(e.frame, e.pose);
      if (vis(e.frame)) this._ghost(e.pose, () => COL.fbJoint, () => COL.fbBone);
    }

    // End-Effector ghosts: grey skeleton, red root + chain joints and chain bones, axes at the chain base.
    // Hips (no chain in Skeleton#chains) marks the root itself: larger joint, axes, drop line to the ground.
    const ee = new Map();
    const eeTypes = new Map(); // frame -> Set of EE tracks (label text)
    for (const e of store.entries('ee')) {
      ee.set(e.frame, e.pose);
      eeTypes.set(e.frame, e.types);
      if (!vis(e.frame)) continue;
      const redJoints = new Set([root]);
      const redBones = new Set();
      const axesAt = [];
      let hips = false;
      for (const track of EE_TRACKS) {
        if (!e.types.has(track)) continue;
        if (track === 'hips') { hips = true; continue; }
        const chain = sk.chainForTrack(track);
        chain.forEach((j, k) => { redJoints.add(j); if (k > 0) redBones.add(j); });
        if (chain.length > 1) axesAt.push(chain[0]);
      }
      this._ghost(e.pose, (j) => (redJoints.has(j) ? COL.eeRed : COL.eeGrey), (j) => (redBones.has(j) ? COL.eeRed : COL.eeGrey));
      for (const j of new Set(axesAt)) this._axes(e.pose, j);
      if (hips) {
        const P = e.pose.positions;
        const x = P[3 * root], y = P[3 * root + 1], z = P[3 * root + 2];
        this._ghostSpheres.push(sphereMatrix(x, y, z, SKEL_JOINT_R * HIPS_ROOT_SCALE), COL.eeRed);
        this._ghostSegments.push(segmentMatrix(x, y, z, x, 0, z, AXES_R), COL.eeRed);
        this._axes(e.pose, root, HIPS_AXES_LEN);
      }
    }
    const eeOnlyHips = (f) => { const t = eeTypes.get(f); return !!t && t.size === 1 && t.has('hips'); };

    // 2D Root entries (frame -> {x, z, heading}).
    const rootEntries = new Map();
    for (const e of store.entries('root2d')) rootEntries.set(e.frame, e);

    // Items: keyframe labels / waypoints, interval labels / polylines.
    const rootAt = (pose) => [pose.positions[3 * root], pose.positions[3 * root + 1], pose.positions[3 * root + 2]];
    const nearestPose = (map, s, e) => {
      const mid = Math.floor((s + e) / 2);
      for (let d = 0; d <= e - s; d++) {
        if (map.has(mid - d) && mid - d >= s) return map.get(mid - d);
        if (map.has(mid + d) && mid + d <= e) return map.get(mid + d);
      }
      return null;
    };
    const eeKeyLabels = new Set();
    for (const item of store.items()) {
      if (item.band) continue; // Smooth Path band: timeline-only
      // Constraints kept past the end of the clip are not drawn (nor sent).
      if (item.start >= N) continue;
      const { track, start: s } = item;
      const e = Math.min(item.end, N - 1);
      const single = s === e;
      if (track === 'fullbody') {
        if (single) {
          if (!vis(s) || !fb.has(s)) continue;
          const [x, y, z] = rootAt(fb.get(s));
          this._label(`fb:${s}`, `Full-Body @ ${s}`, x, y, z, ANCHOR.center);
        } else if (!only) {
          const pose = nearestPose(fb, s, e);
          if (!pose) continue;
          const [x, y, z] = rootAt(pose);
          this._label(`fbi:${s}:${e}`, `Full-Body @ [${s}, ${e}]`, x, y, z, ANCHOR.center, 'interval');
        }
      } else if (track === 'root2d') {
        if (single) {
          const r = rootEntries.get(s);
          if (!vis(s) || !r) continue;
          this._waypoint(s, r.x, r.z, r.heading);
          this._label(`r:${s}`, `2D Root @ ${s}`, r.x, 0, r.z, ANCHOR.bottomLeft, 'root2d');
        } else if (!only) {
          const pts = [];
          for (let f = s; f <= e; f++) {
            const r = rootEntries.get(f);
            pts.push(r ? [r.x, r.z] : null);
          }
          this._polyline(pts);
          const mid = rootEntries.get(Math.floor((s + e) / 2)) ?? rootEntries.get(s);
          if (mid) this._label(`ri:${s}:${e}`, `2D Root @ [${s}, ${e}]`, mid.x, 0, mid.z, ANCHOR.center, 'interval');
        }
      } else {
        // EE tracks (limbs and Hips) share one End-Effectors entry per frame:
        // one keyframe label per frame, "Hips @ f" when Hips is its only track.
        if (single) {
          if (!vis(s) || !ee.has(s) || eeKeyLabels.has(s)) continue;
          eeKeyLabels.add(s);
          const [x, y, z] = rootAt(ee.get(s));
          const hipsOnly = eeOnlyHips(s);
          this._label(`ee:${s}`, `${hipsOnly ? 'Hips' : 'End-Effectors'} @ ${s}`, x, y + 0.05, z, ANCHOR.bottomCenter, hipsOnly ? 'hips' : '');
        } else if (!only) {
          const pose = nearestPose(ee, s, e);
          if (!pose) continue;
          const [x, y, z] = rootAt(pose);
          // A Hips interval label sits below the pelvis so it does not cover a limb interval's label.
          if (track === 'hips') this._label(`hipsi:${s}:${e}`, `Hips @ [${s}, ${e}]`, x, y - 0.05, z, ANCHOR.topCenter, 'interval hips');
          else this._label(`eei:${s}:${e}`, `End-Effectors @ [${s}, ${e}]`, x, y, z, ANCHOR.center, 'interval');
        }
      }
    }

    // Smooth Path (dense) polyline, one sample per integer frame.
    if (store.smoothPath && !only) {
      if (!this._interacting || !this._pathCache) {
        const path = store.densePath(Number.isFinite(N) ? N : undefined);
        this._pathCache = path ? path.map((p) => (p ? [p[0], p[1]] : null)) : null;
      }
      if (this._pathCache) this._polyline(this._pathCache);
    } else if (!store.smoothPath) {
      this._pathCache = null;
    }
  }
}
