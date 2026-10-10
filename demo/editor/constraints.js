// The editor's constraint model: a port of the upstream NVIDIA Kimodo demo's
// constraint sets (kimodo/viz/constraint_ui.py FullbodyKeyframeSet,
// EEJointsKeyframeSet, RootKeyframe2DSet), the timeline callbacks of
// kimodo/demo/ui.py, and compute_model_constraints_lst
// (kimodo/demo/generation.py) + save format (kimodo/constraints.py, the
// upstream constraints.json format documented in docs/IMPLEMENTATION.md).
//
// Model
// - Six timeline tracks (TRACKS) backed by three per-frame slot maps, like upstream:
//     fullbody: frame -> {pose, ids[], smoothRoot2d}
//     ee:       frame -> {pose, entries[{id, track}], smoothRoot2d}   (shared by the 4 limb tracks)
//     root2d:   frame -> {x, z, heading: [cos, sin] | null, ids[]}
//   `pose` = {local: Float32Array(J*4) xyzw, root: Float32Array(3)} plus derived
//   {positions, globals} (FK, computed when stored). Timeline items
//   {id, track, start, end, locked, opacity} live in a separate map; an interval
//   writes one slot entry per frame, all sharing its id. A keyframe and an
//   interval on the same frame share one slot (last add wins); a slot is freed
//   when its id list is empty.
// - Adding snapshots the motion at the target frame(s), not at the playhead:
//   the full pose for Full-Body / limb tracks, the pelvis (x, z) for 2D Root.
//   Two limb tracks on one frame merge into one EE slot (union of types, pose
//   overwritten by the latest snapshot).
// - move() = remove + re-snapshot at the destination (upstream). Option
//   carryPose keeps the stored data instead: a same-length move shifts it, a
//   resize keeps it on frames that were already covered and snapshots new ones.
// - "Make Smooth Path" (setSmoothPath): two locked invisible 2D-Root anchor
//   keyframes at 0 and numFrames-1 (pelvis snapshots) plus a locked visual-only
//   band item [0, numFrames-1] (no slot data). densePath() = linear interpolation
//   over every 2D-Root entry, then get_smooth_root_pos when there are >= 3.
//
// Deliberate deviations from upstream (documented port decisions):
// - Removing one limb from a shared EE frame recomputes the limb types from the
//   remaining ids (upstream EEJointsKeyframeSet only recomputed joint_names).
// - Timeline edits are clamped to [0, numFrames-1]; add outside the range is
//   rejected (returns null); move/resize is clamped (upstream silently dropped).
// - setNumFrames keeps every constraint (like upstream set_new_duration):
//   frames >= numFrames stay in the store, are not counted, not drawn in the
//   view and not sent (toModelJSON drops them, as upstream does at generation
//   and save time), and come back when the clip grows again. Unlike upstream
//   it re-anchors the Smooth Path end keyframe at numFrames-1 (upstream left
//   the anchor at the old end, which made generation fail after growing).
// - Optional per-frame 2D-Root heading ([cos, sin]); emitted as
//   global_root_heading only where set (a root2d set with mixed frames is split
//   into two objects).
// - loadModelJSON keeps the file's smooth_root_2d for Full-Body/EE (until that
//   frame is re-snapshotted or written through) and root2d headings,
//   sorts frame_indices (and merges objects of the same track) before splitting
//   runs into intervals, rejects duplicate frames inside one object, accepts
//   77-joint SOMA rotations, and validates everything before touching the store.
// - Moving a 2D-Root item keeps its headings (they are not part of the pelvis
//   snapshot).
// - toModelJSON emits frame_indices in ascending order (upstream: dict insertion
//   order; the model treats them as a set) and 30 joints for SOMA.
// - Store methods never mutate the motion except setSmoothPath(true) (unless
//   {applyToMotion: false}), applyDensePath() and snapToConstraint().
//
// Exports beyond the core API: EE_TRACKS, TRACK_COLORS, WARN_FRAMES,
// CONSTRAINT_TYPES, validateModelJSON(); ConstraintStore#version, #numFrames,
// #entries(kind), #hasConstraintAt(f), #warnings(), #applyDensePath(),
// #snapTarget(f), #snapToConstraint(f, motion). Events: 'change'
// (CustomEvent, detail.reason).

import { quatFromAxisAngle, quatToAxisAngle } from './math.js';
import { smoothPathXZ } from './smooth_root.js';
import { SOMA30_FROM_77, isSoma30 } from './soma77.js';

export const TRACKS = Object.freeze(['fullbody', 'root2d', 'left-hand', 'right-hand', 'left-foot', 'right-foot']);
export const EE_TRACKS = Object.freeze(['left-hand', 'right-hand', 'left-foot', 'right-foot']);
export const TRACK_LABELS = Object.freeze({
  fullbody: 'Full-Body', root2d: '2D Root',
  'left-hand': 'Left Hand', 'right-hand': 'Right Hand', 'left-foot': 'Left Foot', 'right-foot': 'Right Foot',
});
/** Upstream timeline colours (kimodo/demo/ui.py:99-171; FB == left foot, 2D root == right foot upstream). */
export const TRACK_COLORS = Object.freeze({
  fullbody: [219, 148, 86], root2d: [150, 100, 200],
  'left-hand': [100, 200, 150], 'right-hand': [200, 100, 150], 'left-foot': [219, 148, 86], 'right-foot': [150, 100, 200],
});
/** Soft limit: >= this many constrained frames per type degrades results (upstream docs, limitations). */
export const WARN_FRAMES = 20;
/** Every `type` accepted by upstream load_constraints_lst (TYPE_TO_CLASS). */
export const CONSTRAINT_TYPES = Object.freeze(['root2d', 'fullbody', 'left-hand', 'right-hand', 'left-foot', 'right-foot', 'end-effector']);

const EE_JOINT_TRACK = Object.freeze({ LeftHand: 'left-hand', RightHand: 'right-hand', LeftFoot: 'left-foot', RightFoot: 'right-foot' });
const EE_JOINT_NAMES = new Set([...Object.keys(EE_JOINT_TRACK), 'Hips']);
const isEE = (track) => EE_TRACKS.includes(track);
const sortedKeys = (map) => [...map.keys()].sort((a, b) => a - b);

export class ConstraintStore extends EventTarget {
  constructor(skeleton) {
    super();
    if (!skeleton || !Number.isInteger(skeleton.J)) throw new Error('ConstraintStore: skeleton required');
    this.skeleton = skeleton;
    this.numFrames = null; // timeline length; set by setNumFrames / setSmoothPath
    this._items = new Map();
    this._fb = new Map();
    this._ee = new Map();
    this._root = new Map();
    this._smooth = null; // {start, end, band} item ids while Smooth Path is on
    this._nextId = 1;
    this._rev = 0;
    this._batchDepth = 0;
    this._pendingReasons = [];
    this._rootRev = 0; // bumps when 2D-root slot data changes (dense path cache key)
    this._denseCache = null;
  }

  // -------------------------------------------------------------------------
  // queries
  // -------------------------------------------------------------------------

  /** Increments on every change. */
  get version() { return this._rev; }

  get smoothPath() { return this._smooth !== null; }

  /**
   * Timeline items [{id, track, start, end, locked, opacity, anchor?, band?}]
   * sorted by track then start. Smooth Path anchors (anchor: true, opacity 0)
   * are invisible keyframes; the band (band: true, opacity 0.3) is visual only.
   */
  items() {
    return [...this._items.values()]
      .sort((a, b) => TRACKS.indexOf(a.track) - TRACKS.indexOf(b.track) || a.start - b.start || a.end - b.end)
      .map((it) => ({ ...it }));
  }

  /**
   * Constrained frames of a track inside the timeline (frames < numFrames,
   * i.e. what the model receives), interval frames counted individually.
   */
  frameCount(track) {
    checkTrack(track);
    const inside = this._inside;
    let n = 0;
    if (track === 'fullbody' || track === 'root2d') {
      for (const f of (track === 'fullbody' ? this._fb : this._root).keys()) if (inside(f)) n++;
      return n;
    }
    for (const [f, slot] of this._ee) if (inside(f) && slot.entries.some((e) => e.track === track)) n++;
    return n;
  }

  /**
   * {track: bool} soft warnings (>= WARN_FRAMES frames). 2D-Root paths are
   * exempt (upstream limitations), so only its single keyframes count.
   */
  warnings() {
    const out = {};
    for (const t of TRACKS) {
      if (t === 'root2d') {
        const singles = [...this._items.values()].filter((it) => it.track === 'root2d' && !it.locked && it.start === it.end && this._inside(it.start)).length;
        out[t] = singles >= WARN_FRAMES;
      } else {
        out[t] = this.frameCount(t) >= WARN_FRAMES;
      }
    }
    return out;
  }

  /** {fullbody?: pose, ee?: {types: Set, pose}, root2d?: {x, z, heading: radians|null, headingCS}} at frame f. */
  entriesAt(f) {
    const out = {};
    const fb = this._fb.get(f);
    if (fb) out.fullbody = fb.pose;
    const ee = this._ee.get(f);
    if (ee) out.ee = { types: eeTypes(ee), pose: ee.pose };
    const r = this._root.get(f);
    if (r) out.root2d = rootView(r);
    return out;
  }

  /** All entries of one slot map ('fullbody' | 'ee' | 'root2d'), ascending frames. */
  entries(kind) {
    if (kind === 'fullbody') return sortedKeys(this._fb).map((f) => ({ frame: f, pose: this._fb.get(f).pose }));
    if (kind === 'ee') return sortedKeys(this._ee).map((f) => ({ frame: f, types: eeTypes(this._ee.get(f)), pose: this._ee.get(f).pose }));
    if (kind === 'root2d') return sortedKeys(this._root).map((f) => ({ frame: f, ...rootView(this._root.get(f)) }));
    throw new Error(`entries: unknown kind ${kind} (fullbody | ee | root2d)`);
  }

  hasConstraintAt(f) {
    return this._fb.has(f) || this._ee.has(f) || this._root.has(f);
  }

  // -------------------------------------------------------------------------
  // timeline edits
  // -------------------------------------------------------------------------

  /** Adds a keyframe snapshotted from `motion` at f. Returns its id, or null when f is out of range. */
  addKeyframe(track, f, motion) {
    checkTrack(track);
    f = toFrame(f, 'frame');
    if (f < 0 || f > this._maxFrame(motion)) return null;
    const id = this._newId();
    this._items.set(id, { id, track, start: f, end: f, locked: false, opacity: 1 });
    this._put(track, f, id, this._snapshot(track, f, motion));
    this._changed('add');
    return id;
  }

  /** Adds an inclusive interval [s, e] (clamped to the timeline). Returns its id, or null. */
  addInterval(track, s, e, motion) {
    checkTrack(track);
    s = toFrame(s, 'start'); e = toFrame(e, 'end');
    if (s > e) [s, e] = [e, s];
    const max = this._maxFrame(motion);
    if (e < 0 || s > max) return null;
    s = Math.max(0, s); e = Math.min(max, e);
    const id = this._newId();
    this._items.set(id, { id, track, start: s, end: e, locked: false, opacity: 1 });
    for (let f = s; f <= e; f++) this._put(track, f, id, this._snapshot(track, f, motion));
    this._changed('add');
    return id;
  }

  /**
   * Moves / resizes item `id` to [s, e]. Default re-snapshots the motion at the
   * destination (upstream); {carryPose: true} keeps the stored data. Locked
   * items do not move. Returns true when something changed.
   */
  move(id, s, e, motion, { carryPose = false } = {}) {
    const item = this._items.get(id);
    if (!item || item.locked) return false;
    s = toFrame(s, 'start'); e = toFrame(e ?? s, 'end');
    if (s > e) [s, e] = [e, s];
    const max = this._maxFrame(motion);
    const len = e - s;
    if (len === item.end - item.start) {
      // Pure move: clamp by shifting so the length is kept.
      if (s < 0) { s = 0; e = len; }
      if (e > max) { e = max; s = Math.max(0, max - len); }
    } else {
      s = Math.max(0, s); e = Math.min(max, e);
    }
    if (s > e || (s === item.start && e === item.end)) return false;

    const shift = e - s === item.end - item.start ? s - item.start : null;
    const old = new Map();
    for (let f = item.start; f <= item.end; f++) {
      // 2D-root headings are not part of the pelvis snapshot, so they travel with
      // the item even when the position is re-snapshotted.
      if (carryPose || item.track === 'root2d') old.set(f, this._slotData(item.track, f));
      this._take(item.track, f, id);
    }
    for (let f = s; f <= e; f++) {
      const prev = shift !== null ? old.get(f - shift) : old.get(f);
      let data = carryPose ? prev : null;
      if (!data) {
        data = this._snapshot(item.track, f, motion);
        if (prev?.heading) data.heading = prev.heading;
      }
      this._put(item.track, f, id, data);
    }
    item.start = s; item.end = e;
    this._changed('move');
    return true;
  }

  /** Removes item `id` (locked items only with {force: true}). */
  remove(id, { force = false } = {}) {
    const item = this._items.get(id);
    if (!item || (item.locked && !force)) return false;
    if (!item.band) for (let f = item.start; f <= item.end; f++) this._take(item.track, f, id);
    this._items.delete(id);
    if (this._smooth && Object.values(this._smooth).includes(id)) {
      // Removing a Smooth Path part (forced) turns Smooth Path off entirely.
      const rest = Object.values(this._smooth).filter((x) => x !== id);
      this._smooth = null;
      for (const other of rest) this._removeQuiet(other);
    }
    this._changed('remove');
    return true;
  }

  /** Clears every constraint and turns Smooth Path off (upstream "Clear All Constraints"). */
  clear() {
    this._items.clear(); this._fb.clear(); this._ee.clear(); this._root.clear();
    this._smooth = null;
    this._rootRev++;
    this._changed('clear');
  }

  // -------------------------------------------------------------------------
  // editing write-through
  // -------------------------------------------------------------------------

  /**
   * After the displayed pose at frame f was edited, updates constraints that
   * already exist at f:
   * - rootTranslateOnly (root ground gizmo): Full-Body/EE keep their stored
   *   rotations and take the motion's root; 2D Root takes the pelvis (x, z).
   * - otherwise (joint gizmo, also Reset/Undo): Full-Body/EE take the full
   *   current pose; 2D Root only when rootJointMoved.
   * Returns true when a constraint changed.
   */
  writeThrough(f, motion, { rootTranslateOnly = false, rootJointMoved = false } = {}) {
    f = toFrame(f, 'frame');
    const snap = motion.snapshot(f);
    let changed = false;
    for (const map of [this._fb, this._ee]) {
      const slot = map.get(f);
      if (!slot) continue;
      slot.pose = this._makePose(rootTranslateOnly ? slot.pose.local : snap.local, snap.root);
      slot.smoothRoot2d = null;
      changed = true;
    }
    const r = this._root.get(f);
    if (r && (rootTranslateOnly || rootJointMoved)) {
      r.x = snap.root[0]; r.z = snap.root[2];
      this._rootRev++;
      changed = true;
    }
    if (changed) this._changed('writeThrough');
    return changed;
  }

  /** Sets (theta radians) or clears (null) the optional heading of the 2D-Root entry at f. */
  setRootHeading(f, theta) {
    const r = this._root.get(f);
    if (!r) return false;
    if (theta === null || theta === undefined) r.heading = null;
    else {
      if (!Number.isFinite(theta)) throw new Error(`setRootHeading: invalid angle ${theta}`);
      r.heading = [Math.cos(theta), Math.sin(theta)];
    }
    this._changed('heading');
    return true;
  }

  // -------------------------------------------------------------------------
  // Smooth Path / duration
  // -------------------------------------------------------------------------

  /**
   * "Make Smooth Path". On: locked anchors at 0 and numFrames-1 (pelvis
   * snapshots from `motion`) + locked band item, then (unless applyToMotion is
   * false) moves the motion's pelvis onto the path like upstream. Off: removes
   * anchors and band; the motion is not reverted.
   */
  setSmoothPath(on, motion, numFrames = this.numFrames ?? motion?.frames, { applyToMotion = true } = {}) {
    this._batch(() => {
      if (!on) {
        if (this._smooth) for (const id of Object.values(this._smooth)) this._removeQuiet(id);
        this._smooth = null;
        this._changed('smoothPath');
        return;
      }
      if (!motion) throw new Error('setSmoothPath: motion required to snapshot the anchors');
      numFrames = toFrame(numFrames, 'numFrames');
      if (numFrames < 1) throw new Error('setSmoothPath: numFrames must be >= 1');
      this.numFrames = numFrames;
      if (this._smooth) this._reanchor(motion);
      else {
        this._smooth = {
          start: this._addLocked('root2d', 0, motion),
          end: this._addLocked('root2d', numFrames - 1, motion),
          band: this._addBand(numFrames),
        };
      }
      this._changed('smoothPath');
      if (applyToMotion) this.applyDensePath(motion, numFrames);
    });
  }

  /**
   * Sets the timeline length and re-anchors the Smooth Path end. Constraints
   * past the end are kept (not counted or sent while outside the clip).
   */
  setNumFrames(n, motion) {
    n = toFrame(n, 'numFrames');
    if (n < 1) throw new Error('setNumFrames: numFrames must be >= 1');
    this._batch(() => {
      this.numFrames = n;
      if (this._smooth) {
        if (!motion) throw new Error('setNumFrames: motion required to re-anchor the smooth path');
        this._reanchor(motion);
      }
      this._changed('numFrames');
    });
  }

  /**
   * Dense Smooth Path per frame: an Array(numFrames) of [x, z] on frames covered
   * by the path (min..max 2D-Root frame) and null elsewhere; null when Smooth
   * Path is off. Linear interpolation over every 2D-Root entry inside the clip
   * (frames < numFrames; entries kept past the end do not bend the path),
   * smoothed with get_smooth_root_pos when there are >= 3 entries
   * (constraint_ui.py:927-956).
   */
  densePath(numFrames = this.numFrames) {
    if (!this._smooth) return null;
    const n = numFrames ?? (sortedKeys(this._root).at(-1) ?? 0) + 1;
    if (this._denseCache && this._denseCache.rootRev === this._rootRev && this._denseCache.n === n) return this._denseCache.path;
    const frames = sortedKeys(this._root).filter((f) => f < n);
    const out = new Array(n).fill(null);
    if (frames.length > 0) {
      const lo = frames[0], hi = frames.at(-1);
      let pts = [];
      let k = 0;
      for (let t = lo; t <= hi; t++) {
        while (frames[k + 1] !== undefined && frames[k + 1] <= t) k++;
        const a = this._root.get(frames[k]);
        if (frames[k] === t || k + 1 >= frames.length) { pts.push([a.x, a.z]); continue; }
        // scipy interp1d(kind="linear"): slope * (t - t0) + y0
        const b = this._root.get(frames[k + 1]);
        const span = frames[k + 1] - frames[k], dt = t - frames[k];
        pts.push([((b.x - a.x) / span) * dt + a.x, ((b.z - a.z) / span) * dt + a.z]);
      }
      if (frames.length >= 3) pts = smoothPathXZ(pts);
      for (let t = lo; t <= hi && t < n; t++) out[t] = pts[t - lo];
    }
    this._denseCache = { rootRev: this._rootRev, n, path: out };
    return out;
  }

  /** Moves the motion's pelvis (x, z) onto the dense path (upstream set_projected_root_pos_path). */
  applyDensePath(motion, numFrames = this.numFrames ?? motion.frames) {
    const path = this.densePath(numFrames);
    if (!path) return false;
    const T = Math.min(path.length, motion.frames);
    for (let f = 0; f < T; f++) {
      const p = path[f];
      if (!p) continue;
      const [x, z] = motion.rootXZ(f);
      const dx = p[0] - x, dz = p[1] - z;
      if (dx !== 0 || dz !== 0) motion.translateXZ(f, dx, dz);
    }
    return true;
  }

  // -------------------------------------------------------------------------
  // snapping
  // -------------------------------------------------------------------------

  /** Constraint to snap to at f: priority Full-Body -> End-Effectors -> 2D Root; or null. */
  snapTarget(f) {
    const fb = this._fb.get(f);
    if (fb) return { kind: 'fullbody', pose: fb.pose };
    const ee = this._ee.get(f);
    if (ee) return { kind: 'ee', pose: ee.pose };
    const r = this._root.get(f);
    if (r) return { kind: 'root2d', x: r.x, z: r.z };
    return null;
  }

  /**
   * Copies the constraint pose at f into the motion (Full-Body/EE), or moves
   * the pelvis to the 2D-Root (x, z) keeping its height. Returns the kind or null.
   */
  snapToConstraint(f, motion) {
    if (f < 0 || f >= motion.frames) return null;
    const t = this.snapTarget(f);
    if (!t) return null;
    if (t.kind === 'root2d') {
      const [x, z] = motion.rootXZ(f);
      motion.translateXZ(f, t.x - x, t.z - z);
    } else {
      motion.setPose(f, { local: t.pose.local, root: t.pose.root });
    }
    return t.kind;
  }

  // -------------------------------------------------------------------------
  // JSON (upstream constraints.json)
  // -------------------------------------------------------------------------

  /**
   * The constraints.json array the model receives (compute_model_constraints_lst
   * + save format): frames >= numFrames dropped; objects ordered fullbody, limb
   * types (first occurrence), root2d.
   */
  toModelJSON(numFrames = this.numFrames) {
    const N = numFrames ?? Infinity;
    const dense = this._smooth ? this.densePath(Number.isFinite(N) ? N : undefined) : null;
    const smoothAt = (f, slot) => dense?.[f] ?? slot.smoothRoot2d ?? [slot.pose.root[0], slot.pose.root[2]];
    const out = [];

    const fbFrames = sortedKeys(this._fb).filter((f) => f < N);
    if (fbFrames.length) out.push(this._poseObject('fullbody', fbFrames, this._fb, smoothAt));

    const groups = new Map();
    for (const f of sortedKeys(this._ee)) {
      if (f >= N) continue;
      for (const t of eeTypes(this._ee.get(f))) {
        if (!groups.has(t)) groups.set(t, []);
        groups.get(t).push(f);
      }
    }
    for (const [type, frames] of groups) out.push(this._poseObject(type, frames, this._ee, smoothAt));

    const rootFrames = dense ? dense.map((p, f) => (p ? f : -1)).filter((f) => f >= 0) : sortedKeys(this._root).filter((f) => f < N);
    const xzAt = (f) => (dense ? [dense[f][0], dense[f][1]] : [this._root.get(f).x, this._root.get(f).z]);
    const withHeading = rootFrames.filter((f) => this._root.get(f)?.heading);
    const without = rootFrames.filter((f) => !this._root.get(f)?.heading);
    if (without.length) out.push({ type: 'root2d', frame_indices: without, smooth_root_2d: without.map(xzAt) });
    if (withHeading.length) {
      out.push({
        type: 'root2d', frame_indices: withHeading, smooth_root_2d: withHeading.map(xzAt),
        global_root_heading: withHeading.map((f) => [...this._root.get(f).heading]),
      });
    }
    return out;
  }

  /**
   * Replaces all constraints with a constraints.json array (upstream
   * load_constraints): per track, consecutive frames become intervals and the
   * rest keyframes (frames of several objects of the same track are merged
   * first; on a frame present in several objects the later object wins).
   * Validates the whole array first and throws Error on malformed data without
   * changing the store. `motion` is not read (poses come from the file).
   * Returns {objects, frames: {track: n}}.
   */
  loadModelJSON(array, motion) { // eslint-disable-line no-unused-vars
    const records = validateModelJSON(array, this.skeleton);
    this._batch(() => {
      this.clear();
      // 1. Timeline items: per track, the union of the frames of every object
      //    routed to it, split into consecutive runs (>= 2 frames -> interval).
      //    Merging across objects keeps e.g. a 2D-root interval whose frames were
      //    split into heading / no-heading objects by toModelJSON in one piece.
      const framesByTrack = new Map();
      for (const rec of records) {
        for (const track of rec.tracks) {
          if (!framesByTrack.has(track)) framesByTrack.set(track, new Set());
          for (const f of rec.frames) framesByTrack.get(track).add(f);
        }
      }
      const idAt = new Map(); // `${track}:${frame}` -> item id
      for (const track of TRACKS) {
        const frames = [...(framesByTrack.get(track) ?? [])].sort((a, b) => a - b);
        for (let k = 0, start = 0; k < frames.length; k++) {
          if (k + 1 < frames.length && frames[k + 1] === frames[k] + 1) continue;
          const id = this._newId();
          this._items.set(id, { id, track, start: frames[start], end: frames[k], locked: false, opacity: 1 });
          for (let i = start; i <= k; i++) idAt.set(`${track}:${frames[i]}`, id);
          start = k + 1;
        }
      }
      // 2. Slot data in file order, so a later object wins on a shared frame (upstream).
      for (const rec of records) {
        rec.frames.forEach((f, idx) => {
          const d = rec.data(idx);
          const data = rec.type === 'root2d' ? d : { pose: this._makePose(d.local, d.root), smoothRoot2d: d.smoothRoot2d };
          for (const track of rec.tracks) this._put(track, f, idAt.get(`${track}:${f}`), data);
        });
      }
      this._changed('load');
    });
    const frames = {};
    for (const t of TRACKS) frames[t] = this.frameCount(t);
    return { objects: records.length, frames };
  }

  // -------------------------------------------------------------------------
  // internals
  // -------------------------------------------------------------------------

  _newId() { return `c${this._nextId++}`; }

  /** True for frames inside the timeline (every frame while numFrames is unset). */
  get _inside() {
    const n = this.numFrames;
    return n == null ? () => true : (f) => f < n;
  }

  _maxFrame(motion) {
    return (this.numFrames ?? motion?.frames ?? Infinity) - 1;
  }

  _makePose(local, root) {
    const pose = { local: Float32Array.from(local), root: Float32Array.of(root[0], root[1], root[2]) };
    const { positions, globals } = this.skeleton.fk(pose.local, pose.root);
    pose.positions = positions; pose.globals = globals;
    return Object.freeze(pose);
  }

  _snapshot(track, f, motion) {
    if (!motion) throw new Error('a motion is required to snapshot constraints');
    if (track === 'root2d') {
      const [x, z] = motion.rootXZ(f);
      return { x, z };
    }
    const s = motion.snapshot(f);
    return { pose: this._makePose(s.local, s.root), smoothRoot2d: null };
  }

  /** Copy of the data stored at (track, f) in the form _put accepts, or null. */
  _slotData(track, f) {
    if (track === 'root2d') {
      const r = this._root.get(f);
      return r ? { x: r.x, z: r.z, heading: r.heading } : null;
    }
    const slot = (track === 'fullbody' ? this._fb : this._ee).get(f);
    return slot ? { pose: slot.pose, smoothRoot2d: slot.smoothRoot2d } : null;
  }

  _put(track, f, id, data) {
    if (track === 'root2d') {
      this._rootRev++;
      let r = this._root.get(f);
      if (!r) this._root.set(f, r = { x: 0, z: 0, heading: null, ids: [] });
      r.x = data.x; r.z = data.z;
      if (data.heading !== undefined) r.heading = data.heading ? [data.heading[0], data.heading[1]] : null;
      if (!r.ids.includes(id)) r.ids.push(id);
      return;
    }
    if (track === 'fullbody') {
      let s = this._fb.get(f);
      if (!s) this._fb.set(f, s = { pose: null, smoothRoot2d: null, ids: [] });
      s.pose = data.pose; s.smoothRoot2d = data.smoothRoot2d ?? null;
      if (!s.ids.includes(id)) s.ids.push(id);
      return;
    }
    let s = this._ee.get(f);
    if (!s) this._ee.set(f, s = { pose: null, smoothRoot2d: null, entries: [] });
    s.pose = data.pose; s.smoothRoot2d = data.smoothRoot2d ?? null;
    if (!s.entries.some((e) => e.id === id)) s.entries.push({ id, track });
  }

  _take(track, f, id) {
    if (track === 'root2d' || track === 'fullbody') {
      const map = track === 'root2d' ? this._root : this._fb;
      const slot = map.get(f);
      if (!slot) return;
      slot.ids = slot.ids.filter((x) => x !== id);
      if (slot.ids.length === 0) {
        map.delete(f);
        if (track === 'root2d') this._rootRev++;
      }
      return;
    }
    const slot = this._ee.get(f);
    if (!slot) return;
    slot.entries = slot.entries.filter((e) => e.id !== id);
    if (slot.entries.length === 0) this._ee.delete(f);
  }

  _removeQuiet(id) {
    const item = this._items.get(id);
    if (!item) return;
    if (!item.band) for (let f = item.start; f <= item.end; f++) this._take(item.track, f, id);
    this._items.delete(id);
  }

  _addLocked(track, f, motion) {
    const id = this._newId();
    this._items.set(id, { id, track, start: f, end: f, locked: true, opacity: 0, anchor: true });
    this._put(track, f, id, this._snapshot(track, f, motion));
    return id;
  }

  _addBand(numFrames) {
    const id = this._newId();
    this._items.set(id, { id, track: 'root2d', start: 0, end: numFrames - 1, locked: true, opacity: 0.3, band: true });
    return id;
  }

  /** Puts the Smooth Path end anchor and band on numFrames-1 (start anchor stays at 0). */
  _reanchor(motion) {
    const last = this.numFrames - 1;
    const end = this._items.get(this._smooth.end);
    if (!end || end.start !== last) {
      this._removeQuiet(this._smooth.end);
      this._smooth.end = this._addLocked('root2d', last, motion);
    }
    if (!this._items.has(this._smooth.start)) this._smooth.start = this._addLocked('root2d', 0, motion);
    const band = this._items.get(this._smooth.band);
    if (band) band.end = last;
    else this._smooth.band = this._addBand(this.numFrames);
  }

  _poseObject(type, frames, map, smoothAt) {
    const J = this.skeleton.J;
    const aa = [0, 0, 0];
    return {
      type,
      frame_indices: frames.slice(),
      local_joints_rot: frames.map((f) => {
        const local = map.get(f).pose.local;
        const rows = new Array(J);
        for (let j = 0; j < J; j++) {
          quatToAxisAngle(local.subarray(4 * j, 4 * j + 4), aa);
          rows[j] = [aa[0], aa[1], aa[2]];
        }
        return rows;
      }),
      root_positions: frames.map((f) => Array.from(map.get(f).pose.root)),
      smooth_root_2d: frames.map((f) => {
        const p = smoothAt(f, map.get(f));
        return [p[0], p[1]];
      }),
    };
  }

  _batch(fn) {
    this._batchDepth++;
    try { fn(); } finally {
      this._batchDepth--;
      if (this._batchDepth === 0 && this._pendingReasons.length) {
        const reasons = this._pendingReasons;
        this._pendingReasons = [];
        this._dispatch(reasons.at(-1), reasons);
      }
    }
  }

  _changed(reason) {
    this._rev++;
    if (this._batchDepth > 0) this._pendingReasons.push(reason);
    else this._dispatch(reason, [reason]);
  }

  _dispatch(reason, reasons) {
    const detail = { reason, reasons };
    const ev = typeof CustomEvent === 'function' ? new CustomEvent('change', { detail }) : Object.assign(new Event('change'), { detail });
    this.dispatchEvent(ev);
  }
}

// ---------------------------------------------------------------------------
// validation of constraints.json
// ---------------------------------------------------------------------------

/**
 * Validates an upstream constraints.json array for `skeleton` and returns
 * normalised records {type, tracks[], frames[], data(k)} (data(k) gives the
 * slot data of the k-th file entry). Throws Error with the object index and
 * field on malformed input.
 */
export function validateModelJSON(array, skeleton) {
  if (!Array.isArray(array)) throw new Error('constraints JSON must be an array of constraint objects');
  const J = skeleton.J;
  const soma = isSoma30(skeleton);
  const records = [];
  array.forEach((obj, i) => {
    const where = (field) => `constraints[${i}]${obj && typeof obj.type === 'string' ? ` (${obj.type})` : ''}${field ? `.${field}` : ''}`;
    const fail = (field, msg) => { throw new Error(`${where(field)}: ${msg}`); };
    if (!obj || typeof obj !== 'object' || Array.isArray(obj)) fail('', 'must be an object');
    const type = obj.type;
    if (!CONSTRAINT_TYPES.includes(type)) fail('type', `unknown type ${JSON.stringify(type)} (expected one of ${CONSTRAINT_TYPES.join(', ')})`);

    const fi = obj.frame_indices;
    if (!Array.isArray(fi)) fail('frame_indices', 'must be an array of frame numbers');
    const frames = fi.map((v, k) => {
      if (typeof v !== 'number' || !Number.isInteger(v) || v < 0) fail(`frame_indices[${k}]`, `must be a non-negative integer (got ${JSON.stringify(v)})`);
      return v;
    });
    const seen = new Set();
    frames.forEach((f, k) => {
      if (seen.has(f)) fail(`frame_indices[${k}]`, `duplicate frame ${f}`);
      seen.add(f);
    });
    const K = frames.length;
    const rows = (field, cols, required = true) => {
      const v = obj[field];
      if (v === undefined || v === null) {
        if (required) fail(field, 'is required');
        return null;
      }
      if (!Array.isArray(v) || v.length !== K) fail(field, `must have ${K} rows (one per frame index), got ${Array.isArray(v) ? v.length : typeof v}`);
      v.forEach((r, k) => {
        if (!Array.isArray(r) || r.length !== cols) fail(`${field}[${k}]`, `must be [${cols} numbers]`);
        r.forEach((x, c) => { if (typeof x !== 'number' || !Number.isFinite(x)) fail(`${field}[${k}][${c}]`, `must be a finite number (got ${JSON.stringify(x)})`); });
      });
      return v;
    };

    if (type === 'root2d') {
      const xz = rows('smooth_root_2d', 2);
      const heading = rows('global_root_heading', 2, false);
      records.push({
        type, tracks: ['root2d'], frames,
        data: (k) => ({ x: xz[k][0], z: xz[k][1], heading: heading ? heading[k] : null }),
      });
      return;
    }

    // fullbody / limb / generic end-effector
    let tracks;
    if (type === 'end-effector') {
      const names = obj.joint_names;
      if (!Array.isArray(names) || names.length === 0) fail('joint_names', 'is required for end-effector constraints');
      names.forEach((n, k) => {
        if (!EE_JOINT_NAMES.has(n)) fail(`joint_names[${k}]`, `unknown joint ${JSON.stringify(n)} (expected ${[...EE_JOINT_NAMES].join(', ')})`);
      });
      tracks = Object.keys(EE_JOINT_TRACK).filter((n) => names.includes(n)).map((n) => EE_JOINT_TRACK[n]);
      if (tracks.length === 0) fail('joint_names', 'needs at least one of LeftHand, RightHand, LeftFoot, RightFoot');
    } else {
      tracks = [type];
    }
    const rot = obj.local_joints_rot;
    if (!Array.isArray(rot) || rot.length !== K) fail('local_joints_rot', `must have ${K} poses (one per frame index)`);
    rot.forEach((pose, k) => {
      if (!Array.isArray(pose)) fail(`local_joints_rot[${k}]`, 'must be a [joints][3] array');
      if (!(pose.length === J || (pose.length === 77 && soma))) {
        fail(`local_joints_rot[${k}]`, `has ${pose.length} joints, expected ${J} (${skeleton.key})${soma ? ' or 77 (SOMA77)' : ''}`);
      }
      pose.forEach((r, j) => {
        if (!Array.isArray(r) || r.length !== 3 || r.some((x) => typeof x !== 'number' || !Number.isFinite(x))) {
          fail(`local_joints_rot[${k}][${j}]`, 'must be an axis-angle [x, y, z] of finite numbers');
        }
      });
    });
    const roots = rows('root_positions', 3);
    const smooth = rows('smooth_root_2d', 2, false);
    records.push({
      type, tracks, frames,
      data: (k) => {
        // 77-joint SOMA rotations: keep the 30 control joints (from_SOMASkeleton77).
        const pose = rot[k];
        const src = pose.length === 77 && soma && J !== 77 ? SOMA30_FROM_77.map((idx) => pose[idx]) : pose;
        const local = new Float32Array(J * 4);
        const q = [0, 0, 0, 1];
        for (let j = 0; j < J; j++) local.set(quatFromAxisAngle(src[j], q), 4 * j);
        return { local, root: roots[k], smoothRoot2d: smooth ? [smooth[k][0], smooth[k][1]] : null };
      },
    });
  });
  return records;
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

function checkTrack(track) {
  if (!TRACKS.includes(track)) throw new Error(`unknown constraint track ${JSON.stringify(track)} (expected ${TRACKS.join(', ')})`);
}

function toFrame(v, what) {
  const n = Number(v);
  if (!Number.isFinite(n)) throw new Error(`invalid ${what}: ${v}`);
  return Math.round(n);
}

function eeTypes(slot) {
  return new Set(slot.entries.map((e) => e.track));
}

function rootView(r) {
  return {
    x: r.x, z: r.z,
    heading: r.heading ? Math.atan2(r.heading[1], r.heading[0]) : null,
    headingCS: r.heading ? [r.heading[0], r.heading[1]] : null,
  };
}
