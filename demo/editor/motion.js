// Editable motion clip: per-frame local rotations + root positions with cached FK.
//
// Data layout matches the server files (/api/animations/<id>/*.f32):
//   local: Float32Array(T*J*4) xyzw, root: Float32Array(T*3),
//   extras.contacts: Float32Array(T*4) | null, extras.smoothRoot: Float32Array(T*3) | null.
// Reads at frames outside [0, T-1] clamp to the nearest frame (upstream
// viz/playback.py get_joints_pos pads with the last frame); writes outside throw.
//
// Every mutation bumps `version` (and a per-frame counter that invalidates the
// FK cache of that frame only).
//
// Additions to the core API: static fromBuffers(), skeleton, local, root,
// contacts, smoothRoot, setLocalRotation(), setRootXZ(), rootXZ(), heading().

import { quatMul, quatNormalize } from './math.js';

export class Motion {
  constructor(skeleton, frames, local, root, extras = {}) {
    const J = skeleton.J;
    if (!Number.isInteger(frames) || frames < 1) throw new Error(`Motion: frames must be a positive integer (got ${frames})`);
    if (!local || local.length !== frames * J * 4) throw new Error(`Motion: local rotations need ${frames}*${J}*4 values (got ${local ? local.length : 0})`);
    if (!root || root.length !== frames * 3) throw new Error(`Motion: root positions need ${frames}*3 values (got ${root ? root.length : 0})`);
    this.skeleton = skeleton;
    this.frames = frames;
    this.local = local instanceof Float32Array ? local : Float32Array.from(local);
    this.root = root instanceof Float32Array ? root : Float32Array.from(root);
    const { contacts = null, smoothRoot = null } = extras || {};
    if (contacts && contacts.length !== frames * 4) throw new Error(`Motion: contacts need ${frames}*4 values`);
    if (smoothRoot && smoothRoot.length !== frames * 3) throw new Error(`Motion: smoothRoot needs ${frames}*3 values`);
    this.contacts = contacts ? Float32Array.from(contacts) : null;
    this.smoothRoot = smoothRoot ? Float32Array.from(smoothRoot) : null;
    this.version = 0;
    this._frameVersion = new Uint32Array(frames);
    this._cacheVersion = new Int32Array(frames).fill(-1);
    this._positions = new Float32Array(frames * J * 3);
    this._globals = new Float32Array(frames * J * 4);
    this._poses = new Array(frames);
  }

  /** Rest pose (Skeleton#restPose) repeated `frames` times. */
  static rest(skeleton, frames) {
    const { local, root } = skeleton.restPose();
    const J = skeleton.J;
    const L = new Float32Array(frames * J * 4);
    const R = new Float32Array(frames * 3);
    for (let f = 0; f < frames; f++) { L.set(local, f * J * 4); R.set(root, f * 3); }
    return new Motion(skeleton, frames, L, R);
  }

  /**
   * Builds a motion from the server's binary files (ArrayBuffer or Float32Array):
   * root_positions.f32 [T,3], local_rotations_xyzw.f32 [T,J,4], optional
   * foot_contacts.f32 [T,4] and smooth_root.f32 [T,3].
   */
  static fromBuffers(skeleton, rootF32, rotF32, contactsF32 = null, smoothRootF32 = null) {
    const f32 = (b) => (b == null ? null : b instanceof Float32Array ? b : new Float32Array(b));
    const root = f32(rootF32), rot = f32(rotF32);
    if (!root || root.length % 3 !== 0 || root.length === 0) throw new Error('Motion.fromBuffers: root_positions must be a non-empty [T,3] float32 array');
    const frames = root.length / 3;
    if (!rot || rot.length !== frames * skeleton.J * 4) {
      throw new Error(`Motion.fromBuffers: expected ${frames}x${skeleton.J}x4 rotations, got ${rot ? rot.length : 0} floats`);
    }
    return new Motion(skeleton, frames, rot.slice(), root.slice(), { contacts: f32(contactsF32), smoothRoot: f32(smoothRootF32) });
  }

  _clamp(f) {
    f = Math.round(f);
    return f < 0 ? 0 : f >= this.frames ? this.frames - 1 : f;
  }

  _check(f) {
    if (!Number.isInteger(f) || f < 0 || f >= this.frames) throw new RangeError(`Motion: frame ${f} outside [0, ${this.frames - 1}]`);
  }

  _touch(f) {
    this._frameVersion[f]++;
    this.version++;
  }

  /**
   * Cached pose at frame f: {local, root, positions, globals} as typed-array
   * views into the motion's storage. Do not mutate them; they change when the
   * frame is edited.
   */
  pose(f) {
    f = this._clamp(f);
    const J = this.skeleton.J;
    if (this._cacheVersion[f] !== this._frameVersion[f]) {
      const p = this._poses[f] ??= {
        local: this.local.subarray(f * J * 4, (f + 1) * J * 4),
        root: this.root.subarray(f * 3, f * 3 + 3),
        positions: this._positions.subarray(f * J * 3, (f + 1) * J * 3),
        globals: this._globals.subarray(f * J * 4, (f + 1) * J * 4),
      };
      this.skeleton.fk(p.local, p.root, p);
      this._cacheVersion[f] = this._frameVersion[f];
    }
    return this._poses[f];
  }

  /** Deep copy {local, root} of frame f (clamped). */
  snapshot(f) {
    f = this._clamp(f);
    const J = this.skeleton.J;
    return {
      local: this.local.slice(f * J * 4, (f + 1) * J * 4),
      root: this.root.slice(f * 3, f * 3 + 3),
    };
  }

  /** Overwrites frame f with {local, root} (either may be omitted). */
  setPose(f, { local, root } = {}) {
    this._check(f);
    const J = this.skeleton.J;
    if (local) {
      if (local.length !== J * 4) throw new Error(`setPose: local needs ${J * 4} values`);
      this.local.set(local, f * J * 4);
    }
    if (root) this.root.set([root[0], root[1], root[2]], f * 3);
    this._touch(f);
  }

  /** Sets the local rotation (xyzw) of joint j at frame f; the subtree follows. */
  setLocalRotation(f, j, q) {
    this._check(f);
    const n = quatNormalize(q);
    this.local.set(n, (f * this.skeleton.J + j) * 4);
    this._touch(f);
  }

  /**
   * Sets the global rotation of joint j at frame f (upstream playback.py
   * update_pose_at_frame with joints_local_rot): local = conj(parent global) * q,
   * or q itself for the root. The root position is kept, so the whole subtree
   * of j swings around j.
   */
  setJointGlobalRotation(f, j, q) {
    this._check(f);
    const sk = this.skeleton;
    const p = sk.parents[j];
    let local;
    if (p < 0) {
      local = quatNormalize(q);
    } else {
      const g = this.pose(f).globals;
      const parentConj = [-g[4 * p], -g[4 * p + 1], -g[4 * p + 2], g[4 * p + 3]];
      local = quatNormalize(quatMul(parentConj, q));
    }
    this.local.set(local, (f * sk.J + j) * 4);
    this._touch(f);
  }

  /** Moves the root (and therefore the whole pose) to xyz at frame f. */
  setRoot(f, xyz) {
    this._check(f);
    this.root[f * 3] = xyz[0]; this.root[f * 3 + 1] = xyz[1]; this.root[f * 3 + 2] = xyz[2];
    this._touch(f);
  }

  /** Moves the root to (x, z) keeping its height. */
  setRootXZ(f, x, z) {
    this._check(f);
    this.root[f * 3] = x; this.root[f * 3 + 2] = z;
    this._touch(f);
  }

  /** Translates the whole pose by (dx, 0, dz). */
  translateXZ(f, dx, dz) {
    this._check(f);
    this.root[f * 3] += dx; this.root[f * 3 + 2] += dz;
    this._touch(f);
  }

  /** Root (pelvis) ground projection [x, z] at frame f (clamped). */
  rootXZ(f) {
    f = this._clamp(f);
    return [this.root[f * 3], this.root[f * 3 + 2]];
  }

  /** Heading (radians) of frame f, see Skeleton#heading. */
  heading(f) {
    return this.skeleton.heading(this.pose(f).positions);
  }

  clone() {
    const m = new Motion(this.skeleton, this.frames, this.local.slice(), this.root.slice(),
      { contacts: this.contacts, smoothRoot: this.smoothRoot });
    return m;
  }
}
