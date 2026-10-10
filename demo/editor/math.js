// Rotation and vector helpers for the constraint editor (no DOM, no three.js).
//
// Conventions:
// - Quaternions are xyzw (same order as the server's local_rotations_xyzw.f32
//   and three.js). Any array-like works as input (plain arrays, Float32Array
//   views, ...). Functions that return a value take an optional `out` argument
//   (aliasing `out` with an input is allowed); otherwise they allocate a plain
//   array.
// - 3x3 matrices are row-major arrays of 9 numbers acting on column vectors:
//   v' = M v, m[r*3+c]. The global rotation of a joint is parent_global * local.
// - Axis-angle ("rotation vector") = axis * angle, angle in radians. This is the
//   `local_joints_rot` encoding of upstream constraints.json.
//
// Core exports: quat mul/conj/normalize/rotateVec/
// fromAxisAngle/toAxisAngle/fromMatrix/toMatrix/slerp and vec helpers. They are
// exported both as flat functions (quatMul, ...) and grouped in the `quat`,
// `vec3` and `mat3` namespace objects.
//
// Additions: quatIdentity, quatInvert, quatDot,
// quatAngleBetween, quatFromYaw, mat3* helpers (identity/mul/transpose/mulVec/
// fromAxisAngle/toAxisAngle), wrapAngle, AXIS_ANGLE_EPS.

/** Epsilon of upstream kimodo/geometry.py quaternion_to_axis_angle. */
export const AXIS_ANGLE_EPS = 1e-6;

// ---------------------------------------------------------------------------
// vec3
// ---------------------------------------------------------------------------

export function vec3Add(a, b, out = [0, 0, 0]) {
  out[0] = a[0] + b[0]; out[1] = a[1] + b[1]; out[2] = a[2] + b[2];
  return out;
}

export function vec3Sub(a, b, out = [0, 0, 0]) {
  out[0] = a[0] - b[0]; out[1] = a[1] - b[1]; out[2] = a[2] - b[2];
  return out;
}

export function vec3Scale(a, s, out = [0, 0, 0]) {
  out[0] = a[0] * s; out[1] = a[1] * s; out[2] = a[2] * s;
  return out;
}

export function vec3Dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function vec3Cross(a, b, out = [0, 0, 0]) {
  const x = a[1] * b[2] - a[2] * b[1];
  const y = a[2] * b[0] - a[0] * b[2];
  const z = a[0] * b[1] - a[1] * b[0];
  out[0] = x; out[1] = y; out[2] = z;
  return out;
}

export function vec3Length(a) {
  return Math.hypot(a[0], a[1], a[2]);
}

export function vec3Distance(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

export function vec3Normalize(a, out = [0, 0, 0]) {
  const n = vec3Length(a);
  if (n === 0) { out[0] = 0; out[1] = 0; out[2] = 0; return out; }
  out[0] = a[0] / n; out[1] = a[1] / n; out[2] = a[2] / n;
  return out;
}

export function vec3Lerp(a, b, t, out = [0, 0, 0]) {
  out[0] = a[0] + (b[0] - a[0]) * t;
  out[1] = a[1] + (b[1] - a[1]) * t;
  out[2] = a[2] + (b[2] - a[2]) * t;
  return out;
}

// ---------------------------------------------------------------------------
// quaternions (xyzw)
// ---------------------------------------------------------------------------

export function quatIdentity(out = [0, 0, 0, 1]) {
  out[0] = 0; out[1] = 0; out[2] = 0; out[3] = 1;
  return out;
}

/** Hamilton product a * b (apply b first, then a). */
export function quatMul(a, b, out = [0, 0, 0, 1]) {
  const ax = a[0], ay = a[1], az = a[2], aw = a[3];
  const bx = b[0], by = b[1], bz = b[2], bw = b[3];
  out[0] = aw * bx + ax * bw + ay * bz - az * by;
  out[1] = aw * by - ax * bz + ay * bw + az * bx;
  out[2] = aw * bz + ax * by - ay * bx + az * bw;
  out[3] = aw * bw - ax * bx - ay * by - az * bz;
  return out;
}

export function quatConj(q, out = [0, 0, 0, 1]) {
  out[0] = -q[0]; out[1] = -q[1]; out[2] = -q[2]; out[3] = q[3];
  return out;
}

/** Inverse of a possibly non-unit quaternion (conjugate / |q|^2). */
export function quatInvert(q, out = [0, 0, 0, 1]) {
  const n2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
  if (n2 === 0) return quatIdentity(out);
  out[0] = -q[0] / n2; out[1] = -q[1] / n2; out[2] = -q[2] / n2; out[3] = q[3] / n2;
  return out;
}

export function quatDot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

/** Normalises q; a zero (or non-finite) quaternion becomes the identity. */
export function quatNormalize(q, out = [0, 0, 0, 1]) {
  const n = Math.hypot(q[0], q[1], q[2], q[3]);
  if (!(n > 0) || !Number.isFinite(n)) return quatIdentity(out);
  out[0] = q[0] / n; out[1] = q[1] / n; out[2] = q[2] / n; out[3] = q[3] / n;
  return out;
}

/** Rotates vector v by unit quaternion q: q v q^-1. */
export function quatRotateVec(q, v, out = [0, 0, 0]) {
  const qx = q[0], qy = q[1], qz = q[2], qw = q[3];
  const vx = v[0], vy = v[1], vz = v[2];
  // t = 2 * (q.xyz x v); v' = v + w t + q.xyz x t
  const tx = 2 * (qy * vz - qz * vy);
  const ty = 2 * (qz * vx - qx * vz);
  const tz = 2 * (qx * vy - qy * vx);
  out[0] = vx + qw * tx + (qy * tz - qz * ty);
  out[1] = vy + qw * ty + (qz * tx - qx * tz);
  out[2] = vz + qw * tz + (qx * ty - qy * tx);
  return out;
}

/**
 * Quaternion from a rotation.
 *   quatFromAxisAngle(rotvec, out?)       rotvec = axis * angle (radians)
 *   quatFromAxisAngle(axis, angle, out?)  axis need not be unit length
 * Exact Rodrigues/half-angle formula with a Taylor series near 0. (Upstream
 * axis_angle_to_matrix divides by angle + 1e-6, which under-rotates tiny angles
 * by < 1e-6 rad; we deliberately use the exact form.)
 */
export function quatFromAxisAngle(v, angleOrOut, maybeOut) {
  let x = v[0], y = v[1], z = v[2];
  let out;
  if (typeof angleOrOut === 'number') {
    out = maybeOut || [0, 0, 0, 1];
    const n = Math.hypot(x, y, z);
    if (n === 0) return quatIdentity(out);
    x = x / n * angleOrOut; y = y / n * angleOrOut; z = z / n * angleOrOut;
  } else {
    out = angleOrOut || [0, 0, 0, 1];
  }
  const theta = Math.hypot(x, y, z);
  let s, w;
  if (theta < 1e-4) {
    // sin(t/2)/t and cos(t/2) series (error < 1e-18 at t = 1e-4).
    const t2 = theta * theta;
    s = 0.5 - t2 / 48;
    w = 1 - t2 / 8;
  } else {
    s = Math.sin(theta / 2) / theta;
    w = Math.cos(theta / 2);
  }
  out[0] = x * s; out[1] = y * s; out[2] = z * s; out[3] = w;
  return out;
}

/**
 * Rotation vector (axis * angle) of q; port of upstream
 * kimodo/geometry.py quaternion_to_axis_angle:
 * flip q when w < -eps, or when |w| <= eps and x < 0; angle = 2 atan2(|xyz|, w);
 * aa = xyz * angle / |xyz|, or 2 xyz when |xyz| < eps. The result therefore
 * has angle in [0, pi] (up to eps at pi). q is normalised first.
 */
export function quatToAxisAngle(q, out = [0, 0, 0]) {
  const n = Math.hypot(q[0], q[1], q[2], q[3]);
  if (!(n > 0)) { out[0] = 0; out[1] = 0; out[2] = 0; return out; }
  let x = q[0] / n, y = q[1] / n, z = q[2] / n, w = q[3] / n;
  const eps = AXIS_ANGLE_EPS;
  if (w < -eps || (Math.abs(w) <= eps && x < 0)) { x = -x; y = -y; z = -z; w = -w; }
  const s = Math.hypot(x, y, z);
  const angle = 2 * Math.atan2(s, w);
  const scale = Math.abs(s) < eps ? 2 : angle / Math.max(s, eps);
  out[0] = x * scale; out[1] = y * scale; out[2] = z * scale;
  return out;
}

/** Row-major 3x3 rotation matrix of q (q need not be unit length). */
export function quatToMatrix(q, out = new Array(9)) {
  const x = q[0], y = q[1], z = q[2], w = q[3];
  const n2 = x * x + y * y + z * z + w * w;
  const s = n2 > 0 ? 2 / n2 : 0;
  const xx = x * x * s, yy = y * y * s, zz = z * z * s;
  const xy = x * y * s, xz = x * z * s, yz = y * z * s;
  const wx = w * x * s, wy = w * y * s, wz = w * z * s;
  out[0] = 1 - (yy + zz); out[1] = xy - wz;       out[2] = xz + wy;
  out[3] = xy + wz;       out[4] = 1 - (xx + zz); out[5] = yz - wx;
  out[6] = xz - wy;       out[7] = yz + wx;       out[8] = 1 - (xx + yy);
  return out;
}

/**
 * Unit quaternion of a row-major rotation matrix. Port of the "best candidate"
 * method of upstream kimodo/geometry.py matrix_to_quaternion (pytorch3d), which
 * is accurate for every rotation including angles near pi. The result is
 * normalised; its sign is whatever the best candidate gives (use
 * quatToAxisAngle for a canonical form).
 */
export function quatFromMatrix(m, out = [0, 0, 0, 1]) {
  const m00 = m[0], m01 = m[1], m02 = m[2];
  const m10 = m[3], m11 = m[4], m12 = m[5];
  const m20 = m[6], m21 = m[7], m22 = m[8];
  const sq = (v) => (v > 0 ? Math.sqrt(v) : 0);
  const qa = [
    sq(1 + m00 + m11 + m22),
    sq(1 + m00 - m11 - m22),
    sq(1 - m00 + m11 - m22),
    sq(1 - m00 - m11 + m22),
  ];
  let best = 0;
  for (let i = 1; i < 4; i++) if (qa[i] > qa[best]) best = i;
  const d = 2 * Math.max(qa[best], 0.1);
  let w, x, y, z; // candidate rows of quat_by_rijk (wxyz order)
  switch (best) {
    case 0: w = qa[0] * qa[0]; x = m21 - m12; y = m02 - m20; z = m10 - m01; break;
    case 1: w = m21 - m12; x = qa[1] * qa[1]; y = m10 + m01; z = m02 + m20; break;
    case 2: w = m02 - m20; x = m10 + m01; y = qa[2] * qa[2]; z = m12 + m21; break;
    default: w = m10 - m01; x = m20 + m02; y = m21 + m12; z = qa[3] * qa[3]; break;
  }
  out[0] = x / d; out[1] = y / d; out[2] = z / d; out[3] = w / d;
  return quatNormalize(out, out);
}

/** Spherical interpolation along the shortest arc (t in [0,1]). */
export function quatSlerp(a, b, t, out = [0, 0, 0, 1]) {
  let bx = b[0], by = b[1], bz = b[2], bw = b[3];
  let cos = a[0] * bx + a[1] * by + a[2] * bz + a[3] * bw;
  if (cos < 0) { cos = -cos; bx = -bx; by = -by; bz = -bz; bw = -bw; }
  let ka, kb;
  if (cos > 0.9995) {
    // Nearly parallel: normalised lerp avoids dividing by sin(~0).
    ka = 1 - t; kb = t;
  } else {
    const theta = Math.acos(Math.min(1, cos));
    const sin = Math.sin(theta);
    ka = Math.sin((1 - t) * theta) / sin;
    kb = Math.sin(t * theta) / sin;
  }
  out[0] = a[0] * ka + bx * kb;
  out[1] = a[1] * ka + by * kb;
  out[2] = a[2] * ka + bz * kb;
  out[3] = a[3] * ka + bw * kb;
  return quatNormalize(out, out);
}

/** Angle (radians, [0, pi]) of the rotation taking a to b. */
export function quatAngleBetween(a, b) {
  const na = Math.hypot(a[0], a[1], a[2], a[3]);
  const nb = Math.hypot(b[0], b[1], b[2], b[3]);
  const d = Math.min(1, Math.abs(quatDot(a, b)) / (na * nb));
  return 2 * Math.acos(d);
}

/**
 * Rotation by theta about +Y. A body facing +Z rotated by quatFromYaw(theta)
 * has heading theta (upstream compute_heading_angle: facing vector (sin theta, 0, cos theta)).
 */
export function quatFromYaw(theta, out = [0, 0, 0, 1]) {
  out[0] = 0; out[1] = Math.sin(theta / 2); out[2] = 0; out[3] = Math.cos(theta / 2);
  return out;
}

/** Wraps an angle to (-pi, pi]. */
export function wrapAngle(a) {
  const twoPi = 2 * Math.PI;
  let r = a - twoPi * Math.floor((a + Math.PI) / twoPi);
  if (r <= -Math.PI) r += twoPi;
  return r;
}

// ---------------------------------------------------------------------------
// mat3 (row-major, column-vector convention)
// ---------------------------------------------------------------------------

export function mat3Identity(out = new Array(9)) {
  out[0] = 1; out[1] = 0; out[2] = 0;
  out[3] = 0; out[4] = 1; out[5] = 0;
  out[6] = 0; out[7] = 0; out[8] = 1;
  return out;
}

export function mat3Mul(a, b, out = new Array(9)) {
  const r = new Array(9);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      r[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
    }
  }
  for (let k = 0; k < 9; k++) out[k] = r[k];
  return out;
}

export function mat3Transpose(a, out = new Array(9)) {
  const t = [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]];
  for (let k = 0; k < 9; k++) out[k] = t[k];
  return out;
}

export function mat3MulVec(m, v, out = [0, 0, 0]) {
  const x = v[0], y = v[1], z = v[2];
  out[0] = m[0] * x + m[1] * y + m[2] * z;
  out[1] = m[3] * x + m[4] * y + m[5] * z;
  out[2] = m[6] * x + m[7] * y + m[8] * z;
  return out;
}

/**
 * Rodrigues: R = I + sin(t) K + (1 - cos(t)) K^2 with K the cross-product
 * matrix of the unit axis (upstream axis_angle_to_matrix), exact near 0.
 */
export function mat3FromAxisAngle(v, out = new Array(9)) {
  return quatToMatrix(quatFromAxisAngle(v), out);
}

/** Port of upstream matrix_to_axis_angle: matrix -> quaternion -> axis-angle. */
export function mat3ToAxisAngle(m, out = [0, 0, 0]) {
  return quatToAxisAngle(quatFromMatrix(m), out);
}

// ---------------------------------------------------------------------------
// namespaces
// ---------------------------------------------------------------------------

export const vec3 = Object.freeze({
  add: vec3Add, sub: vec3Sub, scale: vec3Scale, dot: vec3Dot, cross: vec3Cross,
  length: vec3Length, distance: vec3Distance, normalize: vec3Normalize, lerp: vec3Lerp,
});

export const quat = Object.freeze({
  identity: quatIdentity, mul: quatMul, conj: quatConj, invert: quatInvert, dot: quatDot,
  normalize: quatNormalize, rotateVec: quatRotateVec, fromAxisAngle: quatFromAxisAngle,
  toAxisAngle: quatToAxisAngle, fromMatrix: quatFromMatrix, toMatrix: quatToMatrix,
  slerp: quatSlerp, angleBetween: quatAngleBetween, fromYaw: quatFromYaw,
});

export const mat3 = Object.freeze({
  identity: mat3Identity, mul: mat3Mul, transpose: mat3Transpose, mulVec: mat3MulVec,
  fromAxisAngle: mat3FromAxisAngle, toAxisAngle: mat3ToAxisAngle,
  fromQuat: quatToMatrix, toQuat: quatFromMatrix,
});
