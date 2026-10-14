import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import { Skeleton, SKELETON_DEFAULTS, facingVector } from '../skeleton.js';
import { quatFromAxisAngle, quatFromYaw, quatMul, quatRotateVec } from '../math.js';
import { SOMA30_MODEL, TOY_MODEL } from './fixtures/soma30.mjs';

const close = (a, b, tol, msg) => {
  assert.equal(a.length, b.length, msg);
  for (let i = 0; i < a.length; i++) assert.ok(Math.abs(a[i] - b[i]) <= tol, `${msg ?? ''} [${i}] ${a[i]} vs ${b[i]}`);
};

test('soma30 fixture mirrors demo/skeletons_extra.go', (t) => {
  const go = fileURLToPath(new URL('../../skeletons_extra.go', import.meta.url));
  if (!fs.existsSync(go)) return t.skip('skeletons_extra.go not found');
  let names, parents, offsets;
  try {
    const src = fs.readFileSync(go, 'utf8');
    const start = src.indexOf('skeletonDefinitions["soma30"]');
    if (start < 0) throw new Error('soma30 definition moved');
    const block = src.slice(start, src.indexOf('}}', src.indexOf('offsets:', start)) + 2);
    names = JSON.parse(`[${block.match(/names:\s*\[\]string\{([^}]*)\}/)[1]}]`);
    parents = JSON.parse(`[${block.match(/parents:\s*\[\]int\{([^}]*)\}/)[1]}]`);
    offsets = [...block.match(/offsets:\s*\[\]\[3\]float32\{(.*\})\}/)[1].matchAll(/\{([^{}]*)\}/g)]
      .map((m) => m[1].split(',').map((s) => Number(s.trim())));
  } catch (err) {
    return t.skip(`skeletons_extra.go format changed (${err.message}); fixture not cross-checked`);
  }
  assert.deepEqual(names, SOMA30_MODEL.names);
  assert.deepEqual(parents, SOMA30_MODEL.parents);
  assert.equal(offsets.length, 30);
  offsets.forEach((o, j) => close(o, SOMA30_MODEL.offsets[j], 0, `offset ${j}`));
});

test('constructor resolves hips, chains (names -> indices) and hidden gizmos', () => {
  const sk = new Skeleton(SOMA30_MODEL);
  assert.equal(sk.key, 'soma30');
  assert.equal(sk.J, 30);
  assert.equal(sk.rootIndex, 0);
  assert.deepEqual(sk.hips, [26, 22]);
  assert.deepEqual(sk.chains, { LeftHand: [13, 15], RightHand: [19, 21], LeftFoot: [24, 25], RightFoot: [28, 29] });
  assert.deepEqual([...sk.hiddenGizmoJoints].sort((a, b) => a - b), [7, 8, 9, 14, 15, 20, 21]);
  assert.deepEqual(sk.chainForTrack('left-foot'), [24, 25]);
  assert.equal(sk.endEffectorBase('right-hand'), 19);
  // Defaults apply when the server does not send hips/chains yet.
  const bare = new Skeleton({ ...SOMA30_MODEL, hips: undefined, chains: undefined });
  assert.deepEqual(bare.hips, [26, 22]);
  assert.deepEqual(bare.chains, sk.chains);
});

test('g1 and smplx hidden gizmo sets', () => {
  const g1Names = ['pelvis_skel', 'left_hip_pitch_skel', 'left_hip_roll_skel', 'left_hip_yaw_skel', 'left_knee_skel', 'left_ankle_pitch_skel', 'left_ankle_roll_skel', 'left_toe_base', 'right_hip_pitch_skel', 'right_hip_roll_skel', 'right_hip_yaw_skel', 'right_knee_skel', 'right_ankle_pitch_skel', 'right_ankle_roll_skel', 'right_toe_base', 'waist_yaw_skel', 'waist_roll_skel', 'waist_pitch_skel', 'left_shoulder_pitch_skel', 'left_shoulder_roll_skel', 'left_shoulder_yaw_skel', 'left_elbow_skel', 'left_wrist_roll_skel', 'left_wrist_pitch_skel', 'left_wrist_yaw_skel', 'left_hand_roll_skel', 'right_shoulder_pitch_skel', 'right_shoulder_roll_skel', 'right_shoulder_yaw_skel', 'right_elbow_skel', 'right_wrist_roll_skel', 'right_wrist_pitch_skel', 'right_wrist_yaw_skel', 'right_hand_roll_skel'];
  const g1Parents = [-1, 0, 1, 2, 3, 4, 5, 6, 0, 8, 9, 10, 11, 12, 13, 0, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 17, 26, 27, 28, 29, 30, 31, 32];
  const g1 = new Skeleton({ skeleton_key: 'g1skel34', names: g1Names, parents: g1Parents, offsets: g1Names.map(() => [0, -0.1, 0]) });
  assert.deepEqual([...g1.hiddenGizmoJoints].sort((a, b) => a - b), [7, 14, 25, 33]);
  assert.deepEqual(g1.hips, [8, 1]);
  assert.deepEqual(g1.chains.RightHand, [32, 33]);

  const smplxNames = ['pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'];
  const smplxParents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19];
  const smplx = new Skeleton({ skeleton_key: 'smplx22', names: smplxNames, parents: smplxParents, offsets: smplxNames.map(() => [0, 0.1, 0]) });
  assert.equal(smplx.hiddenGizmoJoints.size, 0);
  assert.deepEqual(smplx.hips, [2, 1]);
  assert.deepEqual(smplx.chains.LeftHand, [20]);
  assert.deepEqual(SKELETON_DEFAULTS.smplx22.chains.LeftFoot, ['left_ankle', 'left_foot']);
});

test('constructor rejects malformed models', () => {
  assert.throws(() => new Skeleton({ ...TOY_MODEL, parents: [-1, 0, 5, 0] }), /invalid parent/);
  assert.throws(() => new Skeleton({ ...TOY_MODEL, parents: [-1, -1, 1, 0] }), /exactly one root/);
  assert.throws(() => new Skeleton({ ...TOY_MODEL, names: ['a'] }), /names/);
  assert.throws(() => new Skeleton({ ...TOY_MODEL, offsets: [[0, 0, 0]] }), /offsets/);
  assert.throws(() => new Skeleton({ ...TOY_MODEL, hips: ['nope', 'a'] }), /unknown hip/);
  assert.throws(() => new Skeleton({ ...TOY_MODEL, parents: [-1, 2, 1, 0] }), /tree/);
});

test('FK matches a hand computation on a toy chain', () => {
  const sk = new Skeleton(TOY_MODEL);
  const local = new Float32Array(16);
  // root: +90 deg about +Z; joint a: +90 deg about +Z; b, c identity.
  const qz = quatFromAxisAngle([0, 0, 1], Math.PI / 2);
  local.set(qz, 0); local.set(qz, 4); local.set([0, 0, 0, 1], 8); local.set([0, 0, 0, 1], 12);
  const { positions, globals } = sk.fk(local, [1, 2, 3]);
  // root at (1,2,3); a = root + Rz90 (1,0,0) = (1,3,3); b = a + Rz180 (0,1,0) = (1,2,3); c = root + Rz90(-1,0,0) = (1,1,3)
  close(positions, [1, 2, 3, 1, 3, 3, 1, 2, 3, 1, 1, 3], 1e-6);
  close(globals.subarray(4, 8), quatMul(qz, qz), 1e-7);
  close(globals.subarray(12, 16), qz, 1e-7);
  // localFromGlobals inverts the rotation part of FK
  close(sk.localFromGlobals(globals), local, 1e-6);
});

test('FK on soma30 agrees with an independent recursive implementation', () => {
  const sk = new Skeleton(SOMA30_MODEL);
  const local = new Float32Array(30 * 4);
  for (let j = 0; j < 30; j++) local.set(quatFromAxisAngle([Math.sin(j), Math.cos(3 * j), 0.5 * Math.sin(7 * j)].map((v) => v * 0.6)), 4 * j);
  const root = [0.3, 0.9, -1.2];
  const { positions } = sk.fk(local, root);
  const memo = new Map();
  const glob = (j) => {
    if (memo.has(j)) return memo.get(j);
    const p = SOMA30_MODEL.parents[j];
    const lq = Array.from(local.subarray(4 * j, 4 * j + 4));
    const r = p < 0 ? { q: lq, x: root } : (() => {
      const P = glob(p);
      const off = quatRotateVec(P.q, SOMA30_MODEL.offsets[j]);
      return { q: quatMul(P.q, lq), x: [P.x[0] + off[0], P.x[1] + off[1], P.x[2] + off[2]] };
    })();
    memo.set(j, r);
    return r;
  };
  for (let j = 0; j < 30; j++) close(positions.subarray(3 * j, 3 * j + 3), glob(j).x, 1e-6, `joint ${j}`);
  close(sk.localFromGlobals(sk.fk(local, root).globals), local, 1e-6);
});

test('restPose: identity rotations, lowest joint at y = 0', () => {
  const sk = new Skeleton(SOMA30_MODEL);
  const rest = sk.restPose();
  for (let j = 0; j < 30; j++) close(rest.local.subarray(4 * j, 4 * j + 4), [0, 0, 0, 1], 0);
  const { positions } = sk.fk(rest.local, rest.root);
  let minY = Infinity;
  for (let j = 0; j < 30; j++) minY = Math.min(minY, positions[3 * j + 1]);
  assert.ok(Math.abs(minY) < 1e-6, `lowest joint at ${minY}`);
  assert.ok(rest.root[1] > 0.8 && rest.root[1] < 1.1, `pelvis height ${rest.root[1]}`);
  assert.equal(rest.root[0], 0); assert.equal(rest.root[2], 0);
});

test('heading: rest faces +Z (~0), +90 deg about +Y gives +pi/2', () => {
  const sk = new Skeleton(SOMA30_MODEL);
  const rest = sk.restPose();
  const h0 = sk.heading(sk.fk(rest.local, rest.root).positions);
  // soma30 hips are not perfectly symmetric in z (2.5e-4 m over 0.2 m): ~1.2e-3 rad.
  assert.ok(Math.abs(h0) < 5e-3, `rest heading ${h0}`);
  const turned = rest.local.slice();
  turned.set(quatFromYaw(Math.PI / 2), 0);
  const h1 = sk.heading(sk.fk(turned, rest.root).positions);
  assert.ok(Math.abs(h1 - h0 - Math.PI / 2) < 1e-6, `rotated heading ${h1}`);
  const [fx, fz] = facingVector(Math.PI / 2);
  close([fx, fz], [1, 0], 1e-15);
  // toy skeleton is exactly symmetric
  const toy = new Skeleton(TOY_MODEL);
  assert.equal(toy.heading(toy.fk(toy.restPose().local, [0, 0, 0]).positions), 0);
});
