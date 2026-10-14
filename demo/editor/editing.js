// Editing mode of the constraint editor (port of the upstream demo's editing
// mode, kimodo/demo/ui.py and kimodo/viz/constraint_ui.py).
//
// While active:
// - the viewport shows the skeleton overlay and the mesh at opacity 0.65;
// - a root ground gizmo (TransformControls translate, world X/Y/Z) sits at
//   (pelvis.x, 0, pelvis.z) and translates the whole pose at the current frame
//   (motion.translate + store.writeThrough(..., {rootTranslateOnly: true})).
//   X/Z put the pelvis under the gizmo; Y (the green arrow) raises or lowers
//   the pose by the drag's height (hip height), and the gizmo goes back to the
//   ground on release. The height reaches the model only through a Full-Body,
//   hand/foot or Hips constraint at that frame (2D Root is x/z only); the
//   'dragend' detail says when a drag changed it without one (heightUnconstrained).
//   The gizmo has the three arrows and the XZ square only (its screen-plane
//   centre and vertical squares would change the height by accident), hides
//   its Y arrow while the hips translate gizmo is up (same vertical line), and
//   a press on a skeleton joint selects the joint even over the Y arrow (which
//   spans the legs); the X/Z arrows keep their presses;
// - clicking a joint of the skeleton overlay selects it and attaches a rotate
//   gizmo to a proxy at the joint whose quaternion is the joint's global
//   rotation. TransformControls composes the drag in the proxy's frame for
//   space 'local' (q = q0 * dq) and in world axes for 'world' (q = dq * q0), so
//   after any drag proxy.quaternion is the new global rotation in both spaces
//   (as upstream); it goes to motion.setJointGlobalRotation and store.writeThrough.
//   The root joint can also translate (hip height): W = translate, E = rotate;
//   the translate gizmo is always in world axes (Y = up), whatever the space.
//   Hidden gizmo joints (skeleton.hiddenGizmoJoints) cannot be selected.
// - clicking the 2D-Root waypoint of the current frame selects it and shows a
//   Y-rotation ring that sets its optional heading (store.setRootHeading).
// - Escape (or a click on empty space) deselects.
//
// Every drag works on the frame captured at drag start (playback or scrubbing
// during a drag cannot redirect it). undoDrag() restores the pose of the last
// drag, resetPose() the pose of the current frame as it was first seen in this
// editing session (a snapshot is taken on every frame change; upstream only
// snapshotted frames reached by scrubbing the ruler); both write through to the
// constraints at that frame (upstream Reset left the constraint edited), with the
// same flags as the edit they revert. snapToConstraint() copies the constraint
// at the current frame into the motion (Full-Body -> End-Effectors -> 2D Root).
//
// Events (CustomEvent, detail in parentheses):
//   'change'     pose / constraint edited ({frame, kind, joint?})
//   'dragstart'  ({frame, kind, joint}),
//   'dragend'    ({frame, kind, joint, heightUnconstrained}: the drag changed
//                the hip height at a frame without a Full-Body/EE constraint)
//   'select'     selection changed ({joint, name} or {waypoint: frame})
//   'modechange' editing entered/left or gizmo mode changed ({active, space, jointMode})
// kind: 'ground' | 'rotate' | 'translate' | 'heading' | 'undo' | 'reset' | 'snap' | 'densePath'.
//
// Not implemented: G1 single-axis hinge projection and joint limits (upstream
// viz/g1_rig.py); on G1 the rotate gizmo space is forced to 'local' as upstream
// (the root translate gizmo is in world axes on every skeleton).
//
// Additions to the core API: space, selectedJoint, jointMode,
// headingSelected, canUndo, canReset, setSpace(), setJointMode(), deselect(),
// selectWaypoint(), clearRootHeading(), refresh(), dispose().

import * as THREE from 'three';
import { TransformControls } from 'three/addons/TransformControls.js';

const EDIT_MESH_OPACITY = 0.65;
const CLICK_SLOP_PX = 4;
const ROOT_TRANSLATE_EPS = 1e-3; // upstream applies root-joint moves above 1 mm
// TransformControls sizes are screen-relative multipliers.
const GIZMO_SIZE = { ground: 0.8, joint: 0.55, rootTranslate: 0.8, heading: 1.1 };

export class Editor extends EventTarget {
  constructor(viewport, getState) {
    super();
    if (!viewport || typeof getState !== 'function') throw new Error('Editor: viewport and getState() required');
    this.viewport = viewport;
    this.getState = getState;
    this.active = false;
    this.space = 'local';
    this.selectedJoint = -1;
    this.jointMode = 'rotate'; // root joint only; other joints always rotate
    this.headingSelected = false;
    this._drag = null;
    this._undo = null;
    this._resetSnaps = new Map(); // frame -> {local, root}
    this._snapMotion = null;
    this._store = null;
    this._press = null;
    this._savedView = null;

    const scene = viewport.scene;
    this._proxies = {};
    for (const name of ['ground', 'joint', 'heading']) {
      const proxy = new THREE.Object3D();
      proxy.name = `editor-${name}-proxy`;
      scene.add(proxy); // TransformControls needs object.parent
      this._proxies[name] = proxy;
    }
    // Created before the pointer listeners below, so the gizmos see a press first.
    this._tc = {
      ground: this._makeControls('ground', { mode: 'translate', space: 'world' }),
      joint: this._makeControls('joint', { mode: 'rotate', space: 'local' }),
      heading: this._makeControls('heading', { mode: 'rotate', space: 'world', showX: false, showZ: false }),
    };
    removeHandles(this._tc.ground, ['XYZ', 'XY', 'YZ']);
    this._groundSuspended = false;

    this._onStoreChange = () => { if (this.active && !this._drag) this.refresh(); };
    // Safety net for apps that forget refresh(): re-sync before a render when
    // the motion, frame or store changed since the last sync.
    this._syncKey = null;
    this._renderHook = () => this._autoSync();
    viewport.addRenderHook(this._renderHook);
    this._onPointerDownCapture = (e) => this._pointerDownCapture(e);
    this._onPointerDown = (e) => this._pointerDown(e);
    this._onPointerUp = (e) => this._pointerUp(e);
    this._onPointerMove = (e) => this._pointerMove(e);
    this._onKeyDown = (e) => this._keyDown(e);
    const dom = viewport.domElement;
    // Capture listeners on the target run before the gizmos' own (bubble) pointerdown.
    dom.addEventListener('pointerdown', this._onPointerDownCapture, { capture: true });
    dom.addEventListener('pointerdown', this._onPointerDown);
    dom.addEventListener('pointerup', this._onPointerUp);
    dom.addEventListener('pointermove', this._onPointerMove);
    window.addEventListener('keydown', this._onKeyDown);
  }

  // -------------------------------------------------------------------------
  // state
  // -------------------------------------------------------------------------

  /** True when undoDrag() has something to restore. */
  get canUndo() { return this._undo !== null; }

  /** True when resetPose() has a snapshot for the current frame. */
  get canReset() {
    const st = this.getState();
    return this.active && !!st?.motion && this._resetSnaps.has(clampFrame(st.frame, st.motion));
  }

  /** Gizmo space for joint rotation; like upstream it cannot change while editing. */
  setSpace(space) {
    if (this.active) return false;
    this.space = space === 'world' ? 'world' : 'local';
    return true;
  }

  // -------------------------------------------------------------------------
  // core API
  // -------------------------------------------------------------------------

  enter(space = this.space) {
    if (this.active) return true;
    const st = this.getState();
    if (!st?.motion) throw new Error('Editor.enter: no motion loaded');
    const sk = this.viewport.skeleton;
    this.space = sk.key === 'g1skel34' ? 'local' : space === 'world' ? 'world' : 'local';
    this.active = true;
    const o = this.viewport.options;
    this._savedView = { showSkeleton: o.showSkeleton, meshOpacity: o.meshOpacity };
    this.viewport.setOptions({ showSkeleton: true, meshOpacity: EDIT_MESH_OPACITY });
    this._resetSnaps.clear();
    this._undo = null;
    this._snapMotion = st.motion;
    this._attach('ground');
    this.refresh();
    this._emit('modechange', { active: true, space: this.space, jointMode: this.jointMode });
    return true;
  }

  exit() {
    if (!this.active) return;
    if (this._drag) this._endDrag(this._drag.name);
    this.deselect();
    for (const name of Object.keys(this._tc)) this._detach(name);
    this.active = false;
    this.viewport.setOptions(this._savedView ?? { showSkeleton: false, meshOpacity: 1 });
    this.viewport.setHighlight({ hover: -1, selected: -1 });
    this.viewport.domElement.style.cursor = '';
    // Upstream re-applies the Smooth Path to the motion on exit (ui.py:2170-2172).
    const st = this.getState();
    if (st?.motion && st.store?.smoothPath) {
      st.store.applyDensePath(st.motion, st.numFrames ?? st.motion.frames);
      this._emit('change', { frame: clampFrame(st.frame, st.motion), kind: 'densePath' });
    }
    this._resetSnaps.clear();
    this._undo = null;
    this._emit('modechange', { active: false, space: this.space, jointMode: this.jointMode });
  }

  /** Selects joint j and attaches its gizmo (root joint: translate/rotate per jointMode). j < 0 deselects. */
  selectJoint(j) {
    if (!this.active) return false;
    if (j === null || j === undefined || j < 0) { this.deselect(); return false; }
    const sk = this.viewport.skeleton;
    if (!Number.isInteger(j) || j >= sk.J || sk.hiddenGizmoJoints.has(j)) return false;
    this._deselectHeading();
    this.selectedJoint = j;
    this._setJointGizmoMode(j === sk.rootIndex ? this.jointMode : 'rotate');
    this._syncGroundY();
    this._syncProxies(null);
    this._attach('joint');
    this.viewport.setHighlight({ selected: j });
    this._emit('select', { joint: j, name: sk.names[j] });
    return true;
  }

  deselect() {
    const had = this.selectedJoint >= 0 || this.headingSelected;
    if (this.selectedJoint >= 0) this._detach('joint');
    this.selectedJoint = -1;
    this._syncGroundY();
    this._deselectHeading();
    this.viewport.setHighlight({ selected: -1 });
    if (had) this._emit('select', { joint: -1, name: null });
  }

  /** Gizmo mode of the root joint: 'translate' (W) or 'rotate' (E). */
  setJointMode(mode) {
    if (mode !== 'translate' && mode !== 'rotate') throw new Error(`Editor.setJointMode: unknown mode ${mode}`);
    if (this._drag) return false;
    this.jointMode = mode;
    if (this.selectedJoint === this.viewport.skeleton.rootIndex) {
      this._setJointGizmoMode(mode);
      this._syncGroundY();
      this._syncProxies(null);
    }
    this._emit('modechange', { active: this.active, space: this.space, jointMode: mode });
    return true;
  }

  /** One-level undo of the last gizmo drag, written through to the constraints. */
  undoDrag() {
    const u = this._undo;
    if (!u || this._drag) return false;
    this._undo = null;
    const st = this.getState();
    if (st?.motion !== u.motion) return false;
    const store = st.store;
    if (u.kind === 'heading') {
      store?.setRootHeading(u.frame, u.heading);
    } else if (!samePose(u.motion.snapshot(u.frame), u.snap)) {
      u.motion.setPose(u.frame, u.snap);
      if (store) {
        if (u.kind === 'ground') store.writeThrough(u.frame, u.motion, { rootTranslateOnly: true });
        else store.writeThrough(u.frame, u.motion, { rootJointMoved: u.kind === 'translate' || u.joint === u.motion.skeleton.rootIndex });
      }
    }
    this.refresh();
    this._emit('change', { frame: u.frame, kind: 'undo' });
    return true;
  }

  /** Restores the current frame's pose to its first-visit snapshot of this session and writes it through. */
  resetPose() {
    const st = this.getState();
    if (!this.active || !st?.motion || this._drag) return false;
    const f = clampFrame(st.frame, st.motion);
    const snap = this._resetSnaps.get(f);
    if (!snap) return false;
    const cur = st.motion.snapshot(f);
    if (samePose(cur, snap)) return false; // nothing edited: leave the constraints untouched
    const rootMoved = cur.root[0] !== snap.root[0] || cur.root[2] !== snap.root[2];
    st.motion.setPose(f, snap);
    st.store?.writeThrough(f, st.motion, { rootJointMoved: rootMoved });
    this.refresh();
    this._emit('change', { frame: f, kind: 'reset' });
    return true;
  }

  /** Copies the constraint at the current frame into the motion (FB -> EE -> 2D Root). Returns the kind or null. */
  snapToConstraint() {
    const st = this.getState();
    if (!st?.motion || !st.store || this._drag) return null;
    const f = clampFrame(st.frame, st.motion);
    const kind = st.store.snapToConstraint(f, st.motion);
    if (kind) {
      this.refresh();
      this._emit('change', { frame: f, kind: 'snap', target: kind });
    }
    return kind;
  }

  // -------------------------------------------------------------------------
  // heading handle (optional 2D-Root heading)
  // -------------------------------------------------------------------------

  /** Selects the 2D-Root waypoint at the current frame and shows its heading ring. */
  selectWaypoint() {
    if (!this.active) return false;
    const st = this.getState();
    if (!st?.motion || !st.store?.entriesAt(clampFrame(st.frame, st.motion)).root2d) return false;
    if (this.selectedJoint >= 0) { this._detach('joint'); this.selectedJoint = -1; this.viewport.setHighlight({ selected: -1 }); }
    this.headingSelected = true;
    this._detach('ground'); // its arrows would overlap the ring
    this._syncProxies(null);
    this._attach('heading');
    this._emit('select', { waypoint: clampFrame(st.frame, st.motion) });
    return true;
  }

  /** Removes the optional heading of the 2D-Root entry at the current frame. */
  clearRootHeading() {
    const st = this.getState();
    if (!st?.motion || !st.store) return false;
    const f = clampFrame(st.frame, st.motion);
    const ok = st.store.setRootHeading(f, null);
    if (ok) { this.refresh(); this._emit('change', { frame: f, kind: 'heading' }); }
    return ok;
  }

  // -------------------------------------------------------------------------
  // frame / motion changes
  // -------------------------------------------------------------------------

  /**
   * Call after the frame or motion changed (and after outside edits): takes
   * the Reset snapshot of a newly visited frame and moves the gizmos to the
   * current pose (never while a gizmo is dragged).
   */
  refresh() {
    const st = this.getState();
    this._watchStore(st?.store ?? null);
    if (!this.active || !st?.motion) return;
    if (st.motion !== this._snapMotion) { // a new motion: old snapshots are meaningless
      this._resetSnaps.clear();
      this._undo = null;
      this._snapMotion = st.motion;
    }
    const f = clampFrame(st.frame, st.motion);
    if (!this._resetSnaps.has(f)) this._resetSnaps.set(f, st.motion.snapshot(f));
    if (this._drag) return;
    if (this.headingSelected && !st.store?.entriesAt(f).root2d) this._deselectHeading(true);
    this._syncProxies(null);
  }

  _autoSync() {
    if (!this.active || this._drag) return;
    const st = this.getState();
    if (!st?.motion) return;
    const key = [st.frame, st.motion.version, st.store?.version, st.numFrames];
    const prev = this._syncKey;
    if (prev && prev.motion === st.motion && prev.key.every((v, i) => v === key[i])) return;
    this._syncKey = { motion: st.motion, key };
    this.refresh();
  }

  dispose() {
    this.exit();
    const dom = this.viewport.domElement;
    dom.removeEventListener('pointerdown', this._onPointerDownCapture, { capture: true });
    dom.removeEventListener('pointerdown', this._onPointerDown);
    dom.removeEventListener('pointerup', this._onPointerUp);
    dom.removeEventListener('pointermove', this._onPointerMove);
    window.removeEventListener('keydown', this._onKeyDown);
    this._watchStore(null);
    this.viewport.removeRenderHook(this._renderHook);
    for (const tc of Object.values(this._tc)) { tc.getHelper().removeFromParent(); tc.dispose(); }
    for (const p of Object.values(this._proxies)) p.removeFromParent();
  }

  // -------------------------------------------------------------------------
  // internals: gizmos
  // -------------------------------------------------------------------------

  _makeControls(name, { mode, space, showX = true, showY = true, showZ = true }) {
    const tc = new TransformControls(this.viewport.camera, this.viewport.domElement);
    tc.setMode(mode);
    tc.setSpace(space);
    tc.setSize(GIZMO_SIZE[name]);
    tc.showX = showX; tc.showY = showY; tc.showZ = showZ;
    tc.enabled = false;
    tc.detach();
    const helper = tc.getHelper(); // r170: the helper goes in the scene, not the controls
    helper.name = `editor-${name}-gizmo`;
    this.viewport.scene.add(helper);
    tc.addEventListener('dragging-changed', (e) => {
      this._syncOrbit();
      if (!e.value && this._drag?.name === name) this._endDrag(name); // pointer released without a mouseUp event
    });
    tc.addEventListener('mouseDown', () => this._beginDrag(name));
    tc.addEventListener('objectChange', () => this._objectChange(name));
    tc.addEventListener('mouseUp', () => this._endDrag(name));
    return tc;
  }

  _setJointGizmoMode(mode) {
    const tc = this._tc.joint;
    tc.setMode(mode);
    // Hip height is a world-Y move; a local translate arrow would follow the pelvis tilt.
    tc.setSpace(mode === 'translate' ? 'world' : this.space);
    tc.setSize(mode === 'translate' ? GIZMO_SIZE.rootTranslate : GIZMO_SIZE.joint);
  }

  /** The ground Y arrow is hidden while the hips translate gizmo (same vertical line) is shown. */
  _syncGroundY() {
    this._tc.ground.showY = !(this.selectedJoint === this.viewport.skeleton.rootIndex && this.jointMode === 'translate');
  }

  _attach(name) {
    const tc = this._tc[name];
    tc.attach(this._proxies[name]);
    tc.enabled = true;
  }

  _detach(name) {
    const tc = this._tc[name];
    tc.detach();
    tc.enabled = false;
  }

  _deselectHeading(emit = false) {
    if (!this.headingSelected) return;
    this.headingSelected = false;
    this._detach('heading');
    if (this.active) this._attach('ground');
    if (emit) this._emit('select', { joint: -1, name: null });
  }

  /** OrbitControls are off while any gizmo is dragged. */
  _syncOrbit() {
    this.viewport.controls.enabled = !Object.values(this._tc).some((tc) => tc.dragging);
  }

  /** Places the gizmo proxies on the pose (skipping the one being dragged). */
  _syncProxies(skip) {
    const d = this._drag;
    const st = d ? null : this.getState();
    const motion = d ? d.motion : st?.motion;
    if (!motion) return;
    const frame = d ? d.frame : clampFrame(st.frame, motion);
    const store = d ? d.store : st.store;
    const pose = motion.pose(frame);
    const P = pose.positions, G = pose.globals;
    const sk = motion.skeleton;
    const r = sk.rootIndex;
    if (skip !== 'ground') {
      const g = this._proxies.ground;
      g.position.set(P[3 * r], 0, P[3 * r + 2]);
      g.quaternion.identity();
      g.updateMatrixWorld();
    }
    const j = this.selectedJoint;
    if (skip !== 'joint' && j >= 0) {
      const p = this._proxies.joint;
      p.position.set(P[3 * j], P[3 * j + 1], P[3 * j + 2]);
      p.quaternion.set(G[4 * j], G[4 * j + 1], G[4 * j + 2], G[4 * j + 3]);
      p.updateMatrixWorld();
    }
    if (skip !== 'heading' && this.headingSelected) {
      const e = store?.entriesAt(frame).root2d;
      if (e) {
        const h = this._proxies.heading;
        const theta = e.heading ?? motion.heading(frame);
        h.position.set(e.x, 0.02, e.z);
        h.quaternion.setFromAxisAngle(new THREE.Vector3(0, 1, 0), theta); // +Z -> (sin, 0, cos)
        h.updateMatrixWorld();
      }
    }
  }

  // -------------------------------------------------------------------------
  // internals: drags
  // -------------------------------------------------------------------------

  _beginDrag(name) {
    const tc = this._tc[name];
    const st = this.getState();
    if (this._drag || !this.active || !st?.motion) {
      // Another gizmo already took this press (first wins), or nothing to edit.
      tc.dragging = false;
      tc.axis = null;
      this._syncOrbit();
      return;
    }
    const motion = st.motion;
    const frame = clampFrame(st.frame, motion);
    if (!this._resetSnaps.has(frame)) this._resetSnaps.set(frame, motion.snapshot(frame));
    const kind = name === 'joint' ? (tc.mode === 'translate' ? 'translate' : 'rotate') : name;
    const joint = name === 'joint' ? this.selectedJoint : motion.skeleton.rootIndex;
    this._drag = {
      name, kind, frame, motion, store: st.store ?? null, numFrames: st.numFrames ?? motion.frames, joint,
      rootY: motion.pose(frame).root[1],
    };
    this._undo = {
      frame, motion, kind, joint, snap: motion.snapshot(frame),
      heading: kind === 'heading' ? st.store?.entriesAt(frame).root2d?.heading ?? null : undefined,
    };
    this.viewport.setInteracting(true);
    this._emit('dragstart', { frame, kind, joint });
  }

  _objectChange(name) {
    const d = this._drag;
    if (!d || d.name !== name) return;
    const { motion, frame, store } = d;
    const proxy = this._proxies[name];
    const root = motion.skeleton.rootIndex;
    switch (d.kind) {
      case 'ground': {
        // X/Z absolute: put the pelvis under the gizmo (no accumulated deltas).
        // Y: the proxy starts each drag on the ground, so its height is the
        // lift since the drag started.
        const [x, z] = motion.rootXZ(frame);
        const dx = proxy.position.x - x, dz = proxy.position.z - z;
        const dy = d.rootY + proxy.position.y - motion.pose(frame).root[1];
        if (dx === 0 && dy === 0 && dz === 0) return;
        motion.translate(frame, dx, dy, dz);
        store?.writeThrough(frame, motion, { rootTranslateOnly: true });
        break;
      }
      case 'rotate': {
        const q = proxy.quaternion;
        motion.setJointGlobalRotation(frame, d.joint, [q.x, q.y, q.z, q.w]);
        store?.writeThrough(frame, motion, { rootJointMoved: d.joint === root });
        break;
      }
      case 'translate': {
        const p = proxy.position;
        const cur = motion.pose(frame).root;
        if (Math.hypot(p.x - cur[0], p.y - cur[1], p.z - cur[2]) <= ROOT_TRANSLATE_EPS) return;
        motion.setRoot(frame, [p.x, p.y, p.z]);
        store?.writeThrough(frame, motion, { rootJointMoved: true });
        break;
      }
      case 'heading': {
        const v = new THREE.Vector3(0, 0, 1).applyQuaternion(proxy.quaternion);
        store?.setRootHeading(frame, Math.atan2(v.x, v.z));
        break;
      }
      default: return;
    }
    this._syncProxies(name); // the other gizmos follow the edited pose
    this._emit('change', { frame, kind: d.kind, joint: d.joint });
  }

  _endDrag(name) {
    const d = this._drag;
    if (!d || d.name !== name) return;
    this._drag = null;
    // Root-gizmo release re-applies the Smooth Path (ui.py:2223-2226).
    if (d.kind === 'ground' && d.store?.smoothPath) {
      d.store.applyDensePath(d.motion, d.numFrames);
      this._emit('change', { frame: d.frame, kind: 'densePath' });
    }
    this.viewport.setInteracting(false);
    this._syncOrbit();
    this.refresh();
    const at = d.store?.entriesAt(d.frame);
    const heightUnconstrained = (d.kind === 'ground' || d.kind === 'translate')
      && d.motion.pose(d.frame).root[1] !== d.rootY && !at?.fullbody && !at?.ee;
    this._emit('dragend', { frame: d.frame, kind: d.kind, joint: d.joint, heightUnconstrained });
  }

  // -------------------------------------------------------------------------
  // internals: pointer / keyboard
  // -------------------------------------------------------------------------

  _anyGizmoBusy() {
    return this._drag !== null || Object.values(this._tc).some((tc) => tc.dragging);
  }

  /**
   * Runs before the gizmos see the press: a press on a skeleton joint that is
   * not on a ground X/Z handle goes to joint selection, so the ground gizmo
   * ignores it (its Y arrow spans the legs). Re-enabled on the next pointerup.
   */
  _pointerDownCapture(e) {
    this._resumeGround();
    const tc = this._tc.ground;
    if (!this.active || this._drag || e.button !== 0 || !tc.enabled || /^(X|Z|XZ)$/.test(tc.axis ?? '')) return;
    if (this.viewport.pickJoint(e.clientX, e.clientY) < 0) return;
    tc.enabled = false;
    this._groundSuspended = true;
  }

  _resumeGround() {
    if (!this._groundSuspended) return;
    this._groundSuspended = false;
    if (this.active && !this.headingSelected) this._tc.ground.enabled = true;
  }

  _pointerDown(e) {
    if (!this.active) return;
    // The gizmos' own pointerdown handlers ran first: a press on a handle is a drag, not a click.
    this._press = { x: e.clientX, y: e.clientY, button: e.button, onGizmo: this._anyGizmoBusy() };
  }

  _pointerUp(e) {
    this._resumeGround();
    const press = this._press;
    this._press = null;
    if (!this.active || !press || press.onGizmo || press.button !== 0 || e.button !== 0) return;
    if (Math.hypot(e.clientX - press.x, e.clientY - press.y) > CLICK_SLOP_PX) return; // orbit drag
    const j = this.viewport.pickJoint(e.clientX, e.clientY);
    if (j >= 0) { this.selectJoint(j); return; }
    const st = this.getState();
    const wf = this.viewport.pickWaypoint(e.clientX, e.clientY);
    if (st?.motion && wf >= 0 && wf === clampFrame(st.frame, st.motion)) { this.selectWaypoint(); return; }
    this.deselect();
  }

  _pointerMove(e) {
    if (!this.active || this._drag || e.buttons !== 0) return;
    // The ground Y arrow yields to joints (see _pointerDownCapture); its other handles do not.
    const ground = this._tc.ground;
    const onHandle = Object.values(this._tc).some((tc) => tc.enabled && tc.axis !== null && !(tc === ground && tc.axis === 'Y'));
    const j = onHandle ? -1 : this.viewport.pickJoint(e.clientX, e.clientY);
    if (j >= 0 && ground.axis === 'Y') ground.axis = null;
    this.viewport.setHighlight({ hover: j });
    this.viewport.domElement.style.cursor = j >= 0 ? 'pointer' : '';
  }

  _keyDown(e) {
    if (!this.active || e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey) return;
    const t = e.target;
    if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
    const isRoot = this.selectedJoint === this.viewport.skeleton.rootIndex;
    switch (e.key) {
      case 'Escape':
        if (this.selectedJoint >= 0 || this.headingSelected) { this.deselect(); e.preventDefault(); }
        break;
      case 'w': case 'W':
        if (isRoot) { this.setJointMode('translate'); e.preventDefault(); }
        break;
      case 'e': case 'E':
        if (isRoot) { this.setJointMode('rotate'); e.preventDefault(); }
        break;
      default:
    }
  }

  _watchStore(store) {
    if (store === this._store) return;
    this._store?.removeEventListener('change', this._onStoreChange);
    store?.addEventListener('change', this._onStoreChange);
    this._store = store;
  }

  _emit(type, detail) {
    this.dispatchEvent(new CustomEvent(type, { detail }));
  }
}

/** Removes the named handles (and their pickers) of a translate TransformControls (three r170 internals). */
function removeHandles(tc, names) {
  const g = tc._gizmo;
  if (!g?.gizmo?.translate || !g?.picker?.translate) throw new Error('Editor: unexpected TransformControls internals');
  for (const group of [g.gizmo.translate, g.picker.translate]) {
    for (const h of [...group.children]) if (names.includes(h.name)) h.removeFromParent();
  }
}

function clampFrame(f, motion) {
  const n = Math.round(Number(f) || 0);
  return Math.max(0, Math.min(motion.frames - 1, n));
}

function samePose(a, b) {
  if (a.local.length !== b.local.length) return false;
  for (let k = 0; k < a.local.length; k++) if (a.local[k] !== b.local[k]) return false;
  return a.root[0] === b.root[0] && a.root[1] === b.root[1] && a.root[2] === b.root[2];
}
