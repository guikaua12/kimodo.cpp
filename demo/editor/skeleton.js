// Control-skeleton description + forward kinematics for the constraint editor.
//
// Built from one /api/models entry:
//   {skeleton_key, names, parents, offsets, root_index, hips: [right, left],
//    chains: {LeftHand: [names], RightHand: [...], LeftFoot: [...], RightFoot: [...]}}
// `hips` and `chains` may hold joint names or indices. When the server does not
// send them yet, the per-skeleton defaults below (upstream
// kimodo/skeleton/definitions.py) are used.
//
// FK convention (upstream kimodo/skeleton/kinematics.py):
//   global[j] = global[parent] * local[j]
//   pos[j]    = pos[parent] + rotate(global[parent], offset[j])
//   root position = `root`, root global rotation = local[root].
// Quaternions are xyzw. FK runs in float64 and writes into the output arrays
// (Float32Array by default).
//
// Additions to the core API: SKELETON_DEFAULTS, EE_CHAIN_KEYS,
// Skeleton#index(name), Skeleton#order, Skeleton#chainForTrack(track),
// Skeleton#endEffectorBase(track), Skeleton#restHeight, facingVector(theta).

/** Per-skeleton defaults used when the /api/models entry lacks hips/chains. */
export const SKELETON_DEFAULTS = Object.freeze({
  soma30: {
    hips: ['RightLeg', 'LeftLeg'],
    chains: {
      LeftHand: ['LeftHand', 'LeftHandMiddleEnd'],
      RightHand: ['RightHand', 'RightHandMiddleEnd'],
      LeftFoot: ['LeftFoot', 'LeftToeBase'],
      RightFoot: ['RightFoot', 'RightToeBase'],
    },
    // upstream viz/playback.py:475-484 (SOMASkeleton30)
    hiddenGizmos: ['RightHandThumbEnd', 'RightHandMiddleEnd', 'LeftHandThumbEnd', 'LeftHandMiddleEnd',
      'LeftEye', 'RightEye', 'Jaw'],
  },
  g1skel34: {
    hips: ['right_hip_pitch_skel', 'left_hip_pitch_skel'],
    chains: {
      LeftHand: ['left_wrist_yaw_skel', 'left_hand_roll_skel'],
      RightHand: ['right_wrist_yaw_skel', 'right_hand_roll_skel'],
      LeftFoot: ['left_ankle_roll_skel', 'left_toe_base'],
      RightFoot: ['right_ankle_roll_skel', 'right_toe_base'],
    },
    // upstream viz/playback.py:453-462 (G1Skeleton34)
    hiddenGizmos: ['left_hand_roll_skel', 'right_hand_roll_skel', 'left_toe_base', 'right_toe_base'],
  },
  smplx22: {
    hips: ['right_hip', 'left_hip'],
    chains: {
      LeftHand: ['left_wrist'],
      RightHand: ['right_wrist'],
      LeftFoot: ['left_ankle', 'left_foot'],
      RightFoot: ['right_ankle', 'right_foot'],
    },
    hiddenGizmos: [],
  },
});
// Upstream class names for the same skeletons.
const KEY_ALIASES = { somaskel30: 'soma30', g1: 'g1skel34', smplx: 'smplx22' };

/** Constraint track id -> chain key in Skeleton#chains. */
export const EE_CHAIN_KEYS = Object.freeze({
  'left-hand': 'LeftHand', 'right-hand': 'RightHand', 'left-foot': 'LeftFoot', 'right-foot': 'RightFoot',
});

/** Facing direction (x, z) for heading theta: (sin theta, cos theta). */
export function facingVector(theta) {
  return [Math.sin(theta), Math.cos(theta)];
}

export class Skeleton {
  constructor(model) {
    if (!model || typeof model !== 'object') throw new Error('Skeleton: model entry required');
    const names = model.names;
    const parents = model.parents;
    const offsets = model.offsets;
    if (!Array.isArray(parents) || parents.length === 0) throw new Error('Skeleton: parents[] required');
    const J = parents.length;
    if (!Array.isArray(names) || names.length !== J) {
      throw new Error(`Skeleton: names[] must have ${J} entries (got ${names ? names.length : 'none'})`);
    }
    const flat = offsets && typeof offsets[0] === 'number' ? offsets : (offsets || []).flat();
    if (flat.length !== J * 3 || flat.some((v) => !Number.isFinite(Number(v)))) {
      throw new Error(`Skeleton: offsets must be ${J} finite [x,y,z] triples`);
    }

    this.key = KEY_ALIASES[model.skeleton_key] || model.skeleton_key || '';
    this.names = names.slice();
    this.parents = parents.map((p) => Number(p));
    this.offsets = Array.from({ length: J }, (_, j) => [Number(flat[3 * j]), Number(flat[3 * j + 1]), Number(flat[3 * j + 2])]);
    this.J = J;
    this._offsetsFlat = Float64Array.from(flat, Number);
    this._nameIndex = new Map(this.names.map((n, i) => [n, i]));

    const roots = this.parents.map((p, i) => (p < 0 ? i : -1)).filter((i) => i >= 0);
    if (roots.length !== 1) throw new Error(`Skeleton: expected exactly one root joint, found ${roots.length}`);
    this.rootIndex = model.root_index ?? roots[0];
    if (this.rootIndex !== roots[0]) throw new Error(`Skeleton: root_index ${this.rootIndex} is not the parent-less joint ${roots[0]}`);

    // Topological order (parents before children); rejects cycles / bad parents.
    this.order = topoOrder(this.parents);

    const defaults = SKELETON_DEFAULTS[this.key] || {};
    const resolve = (v, what) => {
      const i = typeof v === 'number' ? v : this._nameIndex.get(v);
      if (!Number.isInteger(i) || i < 0 || i >= J) throw new Error(`Skeleton ${this.key}: unknown ${what} joint ${JSON.stringify(v)}`);
      return i;
    };
    const hips = model.hips ?? defaults.hips;
    if (!Array.isArray(hips) || hips.length !== 2) throw new Error(`Skeleton ${this.key}: hips must be [right, left]`);
    this.hips = [resolve(hips[0], 'hip'), resolve(hips[1], 'hip')];

    const chains = model.chains ?? defaults.chains ?? {};
    this.chains = {};
    for (const k of ['LeftHand', 'RightHand', 'LeftFoot', 'RightFoot']) {
      const list = chains[k];
      if (!Array.isArray(list) || list.length === 0) throw new Error(`Skeleton ${this.key}: chains.${k} missing`);
      this.chains[k] = list.map((n) => resolve(n, `chains.${k}`));
    }

    const hidden = model.hidden_gizmo_joints ?? defaults.hiddenGizmos ?? [];
    this.hiddenGizmoJoints = new Set(hidden.filter((n) => typeof n === 'number' || this._nameIndex.has(n)).map((n) => resolve(n, 'hidden gizmo')));

    // Height of the root above the lowest joint of the identity pose.
    const fk = this.fk(identityLocal(J), [0, 0, 0]);
    let minY = Infinity;
    for (let j = 0; j < J; j++) minY = Math.min(minY, fk.positions[3 * j + 1]);
    this.restHeight = -minY + 0.0;
  }

  /** Joint index of `name`, or -1. */
  index(name) {
    const i = this._nameIndex.get(name);
    return i === undefined ? -1 : i;
  }

  /** Chain joint indices of an end-effector track ('left-hand', ...). */
  chainForTrack(track) {
    const key = EE_CHAIN_KEYS[track];
    if (!key) throw new Error(`not an end-effector track: ${track}`);
    return this.chains[key];
  }

  /** The rotation-constrained base joint of an end-effector track (first chain joint). */
  endEffectorBase(track) {
    return this.chainForTrack(track)[0];
  }

  /** Identity local rotations, root at the origin in xz and high enough that the lowest joint sits on y = 0. */
  restPose() {
    return { local: identityLocal(this.J), root: Float32Array.of(0, this.restHeight, 0) };
  }

  /**
   * Forward kinematics. local: J*4 xyzw (any array-like / typed view),
   * root: [x,y,z]. out (optional): {positions: J*3, globals: J*4} typed arrays.
   */
  fk(local, root, out) {
    const J = this.J;
    if (local.length < J * 4) throw new Error(`fk: local rotations need ${J * 4} values, got ${local.length}`);
    const positions = out?.positions ?? new Float32Array(J * 3);
    const globals = out?.globals ?? new Float32Array(J * 4);
    const G = this._scratchG ??= new Float64Array(J * 4);
    const P = this._scratchP ??= new Float64Array(J * 3);
    const off = this._offsetsFlat;
    for (const j of this.order) {
      const p = this.parents[j];
      const lx = local[4 * j], ly = local[4 * j + 1], lz = local[4 * j + 2], lw = local[4 * j + 3];
      if (p < 0) {
        G[4 * j] = lx; G[4 * j + 1] = ly; G[4 * j + 2] = lz; G[4 * j + 3] = lw;
        P[3 * j] = root[0]; P[3 * j + 1] = root[1]; P[3 * j + 2] = root[2];
        continue;
      }
      const px = G[4 * p], py = G[4 * p + 1], pz = G[4 * p + 2], pw = G[4 * p + 3];
      // global = parent * local (Hamilton product, xyzw)
      G[4 * j] = pw * lx + px * lw + py * lz - pz * ly;
      G[4 * j + 1] = pw * ly - px * lz + py * lw + pz * lx;
      G[4 * j + 2] = pw * lz + px * ly - py * lx + pz * lw;
      G[4 * j + 3] = pw * lw - px * lx - py * ly - pz * lz;
      // pos = parent pos + rotate(parent global, offset)
      const vx = off[3 * j], vy = off[3 * j + 1], vz = off[3 * j + 2];
      const tx = 2 * (py * vz - pz * vy), ty = 2 * (pz * vx - px * vz), tz = 2 * (px * vy - py * vx);
      P[3 * j] = P[3 * p] + vx + pw * tx + (py * tz - pz * ty);
      P[3 * j + 1] = P[3 * p + 1] + vy + pw * ty + (pz * tx - px * tz);
      P[3 * j + 2] = P[3 * p + 2] + vz + pw * tz + (px * ty - py * tx);
    }
    for (let k = 0; k < J * 4; k++) globals[k] = G[k];
    for (let k = 0; k < J * 3; k++) positions[k] = P[k];
    return { positions, globals };
  }

  /** Local rotations (J*4 xyzw, Float32Array) from global rotations: local = conj(parent) * global. */
  localFromGlobals(globals, out = new Float32Array(this.J * 4)) {
    const J = this.J;
    const tmp = new Float64Array(J * 4);
    for (let j = 0; j < J; j++) {
      const gx = globals[4 * j], gy = globals[4 * j + 1], gz = globals[4 * j + 2], gw = globals[4 * j + 3];
      const p = this.parents[j];
      if (p < 0) { tmp[4 * j] = gx; tmp[4 * j + 1] = gy; tmp[4 * j + 2] = gz; tmp[4 * j + 3] = gw; continue; }
      const px = -globals[4 * p], py = -globals[4 * p + 1], pz = -globals[4 * p + 2], pw = globals[4 * p + 3];
      tmp[4 * j] = pw * gx + px * gw + py * gz - pz * gy;
      tmp[4 * j + 1] = pw * gy - px * gz + py * gw + pz * gx;
      tmp[4 * j + 2] = pw * gz + px * gy - py * gx + pz * gw;
      tmp[4 * j + 3] = pw * gw - px * gx - py * gy - pz * gz;
    }
    for (let k = 0; k < J * 4; k++) out[k] = tmp[k];
    return out;
  }

  /**
   * Heading (radians) of a posed skeleton (upstream
   * compute_heading_angle): d = pos[rightHip] - pos[leftHip], theta = atan2(d.z, -d.x).
   * theta = 0 faces +Z; facing vector (sin theta, cos theta).
   */
  heading(positions) {
    const [r, l] = this.hips;
    const dx = positions[3 * r] - positions[3 * l];
    const dz = positions[3 * r + 2] - positions[3 * l + 2];
    return Math.atan2(dz, -dx);
  }
}

function identityLocal(J) {
  const local = new Float32Array(J * 4);
  for (let j = 0; j < J; j++) local[4 * j + 3] = 1;
  return local;
}

function topoOrder(parents) {
  const J = parents.length;
  const children = Array.from({ length: J }, () => []);
  let root = -1;
  parents.forEach((p, j) => {
    if (p < 0) root = j;
    else if (!Number.isInteger(p) || p >= J) throw new Error(`Skeleton: joint ${j} has invalid parent ${p}`);
    else children[p].push(j);
  });
  const order = [];
  const stack = [root];
  while (stack.length) {
    const j = stack.pop();
    order.push(j);
    for (let k = children[j].length - 1; k >= 0; k--) stack.push(children[j][k]);
  }
  if (order.length !== J) throw new Error('Skeleton: parents[] is not a tree rooted at the root joint');
  return order;
}
