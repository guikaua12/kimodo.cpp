import test from 'node:test';
import assert from 'node:assert/strict';
import { Skeleton } from '../skeleton.js';
import { Motion } from '../motion.js';
import { quatFromAxisAngle, quatMul, quatRotateVec, quatAngleBetween, quatConj } from '../math.js';
import { SOMA30_MODEL } from './fixtures/soma30.mjs';

const sk = new Skeleton(SOMA30_MODEL);
const close = (a, b, tol, msg) => {
  assert.equal(a.length, b.length, msg);
  for (let i = 0; i < a.length; i++) assert.ok(Math.abs(a[i] - b[i]) <= tol, `${msg ?? ''} [${i}] ${a[i]} vs ${b[i]}`);
};
const vec = (arr, j) => Array.from(arr.subarray(3 * j, 3 * j + 3));
const q4 = (arr, j) => Array.from(arr.subarray(4 * j, 4 * j + 4));

function wavyMotion(T = 10) {
  const m = Motion.rest(sk, T);
  for (let f = 0; f < T; f++) {
    for (let j = 0; j < sk.J; j++) {
      m.local.set(quatFromAxisAngle([0.3 * Math.sin(f + j), 0.2 * Math.cos(2 * f - j), 0.25 * Math.sin(j * 1.7)]), (f * sk.J + j) * 4);
    }
    m.root.set([0.1 * f, 0.9, 0.05 * f], f * 3);
  }
  return m;
}

/** Joints in the subtree of j (inclusive). */
function subtree(j) {
  const out = new Set([j]);
  for (let k = 0; k < sk.J; k++) {
    for (let p = sk.parents[k]; p >= 0; p = sk.parents[p]) if (p === j) out.add(k);
  }
  return out;
}

test('rest and fromBuffers', () => {
  const m = Motion.rest(sk, 4);
  assert.equal(m.frames, 4);
  close(m.pose(3).root, sk.restPose().root, 0);
  const root = new Float32Array([0, 1, 0, 1, 1, 1]);
  const rot = new Float32Array(2 * 30 * 4);
  for (let k = 3; k < rot.length; k += 4) rot[k] = 1;
  const contacts = new Float32Array(8).fill(1);
  const b = Motion.fromBuffers(sk, root.buffer, rot.buffer, contacts.buffer);
  assert.equal(b.frames, 2);
  close(b.pose(1).root, [1, 1, 1], 0);
  assert.equal(b.contacts.length, 8);
  assert.equal(b.smoothRoot, null);
  assert.throws(() => Motion.fromBuffers(sk, root, rot.subarray(4)), /rotations/);
});

test('pose(f) caches FK, clamps out-of-range frames and invalidates on mutation', () => {
  const m = wavyMotion(5);
  const p = m.pose(2);
  const again = m.pose(2);
  assert.equal(p, again, 'same cached object');
  close(p.positions, sk.fk(p.local, p.root).positions, 0);
  close(m.pose(99).root, m.pose(4).root, 0);
  close(m.pose(-3).root, m.pose(0).root, 0);
  const v0 = m.version;
  m.translateXZ(2, 1, 0);
  assert.equal(m.version, v0 + 1);
  close(vec(m.pose(2).positions, 0), [0.2 + 1, 0.9, 0.1], 1e-6, 'root moved');
  assert.throws(() => m.setRoot(5, [0, 0, 0]), RangeError);
});

test('snapshot is a deep copy; setPose writes it back', () => {
  const m = wavyMotion(3);
  const s = m.snapshot(1);
  s.local[0] = 42;
  assert.notEqual(m.local[30 * 4 + 0], 42);
  const s2 = m.snapshot(2);
  m.setPose(0, s2);
  close(m.pose(0).positions, m.pose(2).positions, 0);
});

test('setJointGlobalRotation keeps the root, sets the global, swings the subtree only', () => {
  const m = wavyMotion(4);
  const f = 1, j = 11; // LeftArm
  const before = m.pose(f);
  const posBefore = before.positions.slice(), globBefore = before.globals.slice();
  const target = quatMul(quatFromAxisAngle([0, 0, 1], 0.7), q4(globBefore, j));
  m.setJointGlobalRotation(f, j, target);
  const after = m.pose(f);
  assert.ok(quatAngleBetween(q4(after.globals, j), target) < 1e-6, 'global rotation applied');
  const moving = subtree(j);
  for (let k = 0; k < sk.J; k++) {
    if (!moving.has(k)) {
      close(vec(after.positions, k), vec(posBefore, k), 1e-6, `joint ${k} must not move`);
      assert.ok(quatAngleBetween(q4(after.globals, k), q4(globBefore, k)) < 1e-6);
    }
  }
  // The joint itself stays where it was (rotation about its own pivot), its children swing rigidly.
  close(vec(after.positions, j), vec(posBefore, j), 1e-6);
  const delta = quatMul(target, quatConj(q4(globBefore, j)));
  for (const k of moving) {
    if (k === j) continue;
    const rel = [0, 1, 2].map((c) => posBefore[3 * k + c] - posBefore[3 * j + c]);
    const exp = quatRotateVec(delta, rel).map((c, i) => c + posBefore[3 * j + i]);
    close(vec(after.positions, k), exp, 1e-5, `subtree joint ${k}`);
  }
  // Other frames untouched.
  close(m.pose(0).positions, wavyMotion(4).pose(0).positions, 0);
});

test('setJointGlobalRotation on the root rotates the whole body about the pelvis', () => {
  const m = wavyMotion(2);
  const rootBefore = Array.from(m.pose(0).root);
  const q = quatFromAxisAngle([0, 1, 0], Math.PI / 2);
  m.setJointGlobalRotation(0, 0, q);
  close(Array.from(m.pose(0).root), rootBefore, 0);
  close(q4(m.pose(0).globals, 0), q, 1e-7);
  close(q4(m.local, 0), q, 1e-7, 'root local = global');
});

test('setRoot / translateXZ move the whole pose rigidly; clone is independent', () => {
  const m = wavyMotion(3);
  const before = m.pose(1).positions.slice();
  m.translateXZ(1, 0.5, -0.25);
  const after = m.pose(1).positions;
  for (let k = 0; k < sk.J; k++) close(vec(after, k), [before[3 * k] + 0.5, before[3 * k + 1], before[3 * k + 2] - 0.25], 1e-6);
  m.setRoot(1, [0, 2, 0]);
  close(vec(m.pose(1).positions, 0), [0, 2, 0], 0);
  const c = m.clone();
  c.translateXZ(0, 1, 1);
  assert.notDeepEqual(Array.from(c.pose(0).root), Array.from(m.pose(0).root));
  assert.deepEqual(m.rootXZ(1), [0, 0]);
});

test('translate moves the whole pose rigidly in x, y and z (ground gizmo height)', () => {
  const m = wavyMotion(3);
  const before = m.pose(2).positions.slice();
  const other = m.snapshot(1);
  m.translate(2, 0.25, 0.5, -0.125);
  const after = m.pose(2).positions;
  for (let k = 0; k < sk.J; k++) close(vec(after, k), [before[3 * k] + 0.25, before[3 * k + 1] + 0.5, before[3 * k + 2] - 0.125], 1e-6);
  assert.deepEqual(Array.from(m.snapshot(1).root), Array.from(other.root), 'other frames untouched');
  m.translate(2, 0, -0.5, 0);
  close(Array.from(m.pose(2).positions.slice(0, 3)), [before[0] + 0.25, before[1], before[2] - 0.125], 1e-6);
});
