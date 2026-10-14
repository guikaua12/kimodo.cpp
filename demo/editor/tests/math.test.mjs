import test from 'node:test';
import assert from 'node:assert/strict';
import {
  quatMul, quatConj, quatNormalize, quatRotateVec, quatFromAxisAngle, quatToAxisAngle, quatFromMatrix,
  quatToMatrix, quatSlerp, quatAngleBetween, quatFromYaw, quatInvert, mat3Mul, mat3Transpose, mat3MulVec,
  mat3FromAxisAngle, mat3ToAxisAngle, mat3Identity, vec3Cross, vec3Length, vec3Normalize, wrapAngle, quat, vec3, mat3,
} from '../math.js';

const close = (a, b, tol, msg) => {
  assert.equal(a.length, b.length, msg);
  for (let i = 0; i < a.length; i++) assert.ok(Math.abs(a[i] - b[i]) <= tol, `${msg ?? ''} [${i}] ${a[i]} vs ${b[i]} (tol ${tol})`);
};

// Deterministic PRNG (mulberry32) so failures reproduce.
function rng(seed) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = rng(42);
const randUnit = () => vec3Normalize([rand() * 2 - 1, rand() * 2 - 1, rand() * 2 - 1]);
const randQuat = () => quatNormalize([rand() * 2 - 1, rand() * 2 - 1, rand() * 2 - 1, rand() * 2 - 1]);

/** Reference Rodrigues formula (unit axis, exact), as upstream axis_angle_to_matrix. */
function rodrigues(v) {
  const t = vec3Length(v);
  if (t === 0) return mat3Identity();
  const [x, y, z] = v.map((c) => c / t);
  const K = [0, -z, y, z, 0, -x, -y, x, 0];
  const K2 = mat3Mul(K, K);
  const s = Math.sin(t), c = 1 - Math.cos(t);
  return mat3Identity().map((e, i) => e + s * K[i] + c * K2[i]);
}

test('namespaces expose the core names', () => {
  for (const k of ['mul', 'conj', 'normalize', 'rotateVec', 'fromAxisAngle', 'toAxisAngle', 'fromMatrix', 'toMatrix', 'slerp']) {
    assert.equal(typeof quat[k], 'function', k);
  }
  for (const k of ['add', 'sub', 'scale', 'dot', 'cross', 'length', 'normalize', 'lerp']) assert.equal(typeof vec3[k], 'function', k);
  for (const k of ['mul', 'transpose', 'mulVec', 'fromAxisAngle', 'toAxisAngle']) assert.equal(typeof mat3[k], 'function', k);
});

test('toMatrix(q) v == rotateVec(q, v) (handedness, row-major, column vectors)', () => {
  for (let i = 0; i < 200; i++) {
    const q = randQuat(), v = [rand() - 0.5, rand() - 0.5, rand() - 0.5];
    close(mat3MulVec(quatToMatrix(q), v), quatRotateVec(q, v), 1e-12);
  }
  // +90 deg about +Z maps +X to +Y; about +Y maps +Z to +X.
  close(quatRotateVec(quatFromAxisAngle([0, 0, 1], Math.PI / 2), [1, 0, 0]), [0, 1, 0], 1e-15);
  close(quatRotateVec(quatFromYaw(Math.PI / 2), [0, 0, 1]), [1, 0, 0], 1e-15);
});

test('quatMul composes like matrix products and conj inverts', () => {
  for (let i = 0; i < 100; i++) {
    const a = randQuat(), b = randQuat();
    close(quatToMatrix(quatMul(a, b)), mat3Mul(quatToMatrix(a), quatToMatrix(b)), 1e-12);
    close(quatMul(a, quatConj(a)), [0, 0, 0, 1], 1e-12);
    const s = [a[0] * 3, a[1] * 3, a[2] * 3, a[3] * 3];
    close(quatMul(s, quatInvert(s)), [0, 0, 0, 1], 1e-12);
  }
  // aliasing out with an input
  const a = randQuat(), b = randQuat(), expect = quatMul(a, b), a2 = a.slice();
  quatMul(a2, b, a2);
  close(a2, expect, 0);
});

test('fromAxisAngle matches Rodrigues, including tiny angles and both call forms', () => {
  for (const scale of [0, 1e-12, 1e-8, 1e-5, 1e-3, 0.5, 1, 2, 3, Math.PI - 1e-9, Math.PI]) {
    for (let i = 0; i < 20; i++) {
      const v = randUnit().map((c) => c * scale);
      close(quatToMatrix(quatFromAxisAngle(v)), rodrigues(v), 1e-12, `scale ${scale}`);
      close(mat3FromAxisAngle(v), rodrigues(v), 1e-12);
    }
  }
  const axis = [0, 2, 0];
  close(quatFromAxisAngle(axis, Math.PI / 2), quatFromYaw(Math.PI / 2), 1e-15);
  close(quatFromAxisAngle([0, 0, 0]), [0, 0, 0, 1], 0);
});

test('axis-angle <-> quaternion <-> matrix round trips (near 0, generic, near pi)', () => {
  const angles = [0, 1e-10, 1e-7, 1e-6, 2e-6, 1e-4, 0.1, 1, 2, 3, Math.PI - 1e-4, Math.PI - 1e-7, Math.PI];
  for (const angle of angles) {
    for (let i = 0; i < 25; i++) {
      const v = randUnit().map((c) => c * angle);
      const q = quatFromAxisAngle(v);
      const back = quatToAxisAngle(q);
      // The canonical result may differ from v by the equivalent representation at pi; compare rotations.
      close(rodrigues(back), rodrigues(v), 1e-9, `angle ${angle}`);
      assert.ok(vec3Length(back) <= Math.PI + 1e-5, 'angle within [0, pi]');
      if (angle < Math.PI - 1e-3) close(back, v, 1e-9, `aa round trip, angle ${angle}`);
      // via matrix (upstream matrix_to_axis_angle path)
      const viaM = mat3ToAxisAngle(rodrigues(v));
      close(rodrigues(viaM), rodrigues(v), 1e-9);
      // quaternion from matrix equals q up to sign
      const qm = quatFromMatrix(quatToMatrix(q));
      assert.ok(1 - Math.abs(qm[0] * q[0] + qm[1] * q[1] + qm[2] * q[2] + qm[3] * q[3]) < 1e-12);
    }
  }
});

test('toAxisAngle canonicalisation follows upstream (w >= 0, x tie-break at w ~ 0)', () => {
  // w < 0 is flipped: same rotation, angle < pi
  const q = quatFromAxisAngle([0, 0, 1], 1);
  const neg = q.map((c) => -c);
  close(quatToAxisAngle(neg), [0, 0, 1], 1e-12);
  // 180 deg about -x with w = 0: flip so x > 0 -> +pi about +x
  close(quatToAxisAngle([-1, 0, 0, 0]), [Math.PI, 0, 0], 1e-12);
  // small angle branch: |xyz| < 1e-6 -> 2 xyz
  close(quatToAxisAngle([1e-8, 0, 0, 1]), [2e-8, 0, 0], 1e-20);
  // non-normalised input is normalised first
  close(quatToAxisAngle([0, 0, 2 * Math.sin(0.5), 2 * Math.cos(0.5)]), [0, 0, 1], 1e-12);
});

test('fromMatrix is robust for 180 degree rotations about every axis', () => {
  for (const axis of [[1, 0, 0], [0, 1, 0], [0, 0, 1], vec3Normalize([1, 1, 0]), vec3Normalize([-1, 2, 3])]) {
    const R = rodrigues(axis.map((c) => c * Math.PI));
    close(quatToMatrix(quatFromMatrix(R)), R, 1e-12);
  }
});

test('slerp: endpoints, midpoint angle, shortest arc', () => {
  const a = quatFromYaw(0), b = quatFromYaw(Math.PI / 2);
  close(quatSlerp(a, b, 0), a, 1e-15);
  close(quatSlerp(a, b, 1), b, 1e-15);
  close(quatSlerp(a, b, 0.5), quatFromYaw(Math.PI / 4), 1e-12);
  // b and -b are the same rotation: slerp must take the short way.
  close(quatToMatrix(quatSlerp(a, b.map((c) => -c), 0.5)), quatToMatrix(quatFromYaw(Math.PI / 4)), 1e-12);
  assert.ok(Math.abs(quatAngleBetween(a, b) - Math.PI / 2) < 1e-12);
});

test('mat3 helpers and vec3 helpers', () => {
  const R = rodrigues([0.3, -0.2, 0.9]);
  close(mat3Mul(R, mat3Transpose(R)), mat3Identity(), 1e-12);
  close(vec3Cross([1, 0, 0], [0, 1, 0]), [0, 0, 1], 0);
  assert.equal(wrapAngle(3 * Math.PI), Math.PI);
  assert.ok(Math.abs(wrapAngle(-3 * Math.PI / 2) - Math.PI / 2) < 1e-12);
});
