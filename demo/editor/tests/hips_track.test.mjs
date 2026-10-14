// The Hips end-effector track: a seventh timeline track routed through the
// shared End-Effector slot map, loaded from / saved as the generic
// {type: 'end-effector', joint_names: ['Hips']} object.
import test from 'node:test';
import assert from 'node:assert/strict';
import { Skeleton } from '../skeleton.js';
import { Motion } from '../motion.js';
import { ConstraintStore, TRACKS, EE_TRACKS, WARN_FRAMES, validateModelJSON } from '../constraints.js';
import { quatFromAxisAngle } from '../math.js';
import { SOMA30_MODEL } from './fixtures/soma30.mjs';

const sk = new Skeleton(SOMA30_MODEL);
const J = sk.J;
const LIMBS = ['left-hand', 'right-hand', 'left-foot', 'right-foot'];
const close = (a, b, tol, msg) => {
  assert.equal(a.length, b.length, msg);
  for (let i = 0; i < a.length; i++) assert.ok(Math.abs(a[i] - b[i]) <= tol, `${msg ?? ''} [${i}] ${a[i]} vs ${b[i]} (tol ${tol})`);
};

/** A motion whose every frame differs: root walks along +z/+x, joints wiggle. */
function makeMotion(T = 60) {
  const m = Motion.rest(sk, T);
  for (let f = 0; f < T; f++) {
    for (let j = 0; j < J; j++) {
      m.local.set(quatFromAxisAngle([0.2 * Math.sin(0.1 * f + j), 0.3 * Math.cos(0.07 * f - j), 0.1 * Math.sin(j)]), (f * J + j) * 4);
    }
    m.root.set([0.01 * f, 0.9 + 0.01 * Math.sin(f), 0.03 * f], f * 3);
  }
  return m;
}

const sameSnapshot = (pose, motion, f) => {
  const s = motion.snapshot(f);
  close(pose.local, s.local, 0);
  close(pose.root, s.root, 0);
};
const itemList = (store) => store.items().map((i) => [i.track, i.start, i.end]);

// Axis-angle poses (30 joints): the zero pose converts exactly; the other one
// goes through float32 quaternions and is compared within 1e-6.
const ZERO = Array.from({ length: J }, () => [0, 0, 0]);
const BENT = Array.from({ length: J }, (_, j) => [0.02 * j, -0.01 * j, 0.03 * Math.sin(j)]);
const HIPS_KEYS = ['type', 'joint_names', 'frame_indices', 'local_joints_rot', 'root_positions', 'smooth_root_2d'];

test('validateModelJSON: a generic end-effector takes any non-empty subset of the five joints', () => {
  const base = { type: 'end-effector', frame_indices: [3], local_joints_rot: [ZERO], root_positions: [[0, 0.9, 0]] };
  const tracksOf = (names) => validateModelJSON([{ ...base, joint_names: names }], sk)[0].tracks;
  assert.deepEqual(tracksOf(['Hips']), ['hips']);
  assert.deepEqual(tracksOf(['RightFoot', 'Hips', 'LeftHand']), ['left-hand', 'right-foot', 'hips']);
  assert.deepEqual(tracksOf(['Hips', 'LeftHand', 'RightHand', 'LeftFoot', 'RightFoot']), EE_TRACKS);
  assert.deepEqual(tracksOf(['LeftFoot', 'RightFoot']), ['left-foot', 'right-foot'], 'limb-only subsets as before');
  assert.deepEqual(tracksOf(['Hips', 'Hips']), ['hips'], 'repeated names route once');
});

test('Hips-only end-effector: load -> save round trip keeps every number', () => {
  const doc = [{
    type: 'end-effector', joint_names: ['Hips'], frame_indices: [31, 0, 30],
    local_joints_rot: [BENT, ZERO, BENT],
    // float32-exact values, so the stored Float32Array root gives them back unchanged
    root_positions: [[0.5, 0.9375, 1.3125], [0, 0.875, 0], [0.5, 0.9375, 1.25]],
    smooth_root_2d: [[0.5, 1.3125], [0, 0], [0.4999, 1.2501]],
  }];
  const m = makeMotion(60);
  const store = new ConstraintStore(sk);
  store.setNumFrames(60, m);
  const summary = store.loadModelJSON(doc, m);
  assert.equal(summary.objects, 1);
  assert.deepEqual(summary.frames, { fullbody: 0, root2d: 0, 'left-hand': 0, 'right-hand': 0, 'left-foot': 0, 'right-foot': 0, hips: 3 });
  assert.deepEqual(itemList(store), [['hips', 0, 0], ['hips', 30, 31]]);
  assert.deepEqual([...store.entriesAt(30).ee.types], ['hips']);
  assert.equal(store.entriesAt(30).fullbody, undefined, 'not a full-body pin');

  const json = store.toModelJSON(60);
  assert.equal(json.length, 1);
  const o = json[0];
  assert.deepEqual(Object.keys(o), HIPS_KEYS);
  assert.equal(o.type, 'end-effector');
  assert.deepEqual(o.joint_names, ['Hips']);
  assert.deepEqual(o.frame_indices, [0, 30, 31]);
  assert.deepEqual(o.root_positions, [[0, 0.875, 0], [0.5, 0.9375, 1.25], [0.5, 0.9375, 1.3125]]);
  assert.deepEqual(o.smooth_root_2d, [[0, 0], [0.4999, 1.2501], [0.5, 1.3125]], 'file smooth_root_2d kept (pelvis exactly on the path point)');
  assert.deepEqual(o.local_joints_rot[0], ZERO);
  for (const k of [1, 2]) close(o.local_joints_rot[k].flat(), BENT.flat(), 1e-6, `rotations @${o.frame_indices[k]}`);

  // Saving is a fixed point: loading the saved file saves exactly the same text
  // (compared as JSON text: the file has no -0).
  const text = JSON.stringify(json);
  const again = new ConstraintStore(sk);
  again.setNumFrames(60, m);
  again.loadModelJSON(JSON.parse(text), m);
  assert.equal(JSON.stringify(again.toModelJSON(60)), text);
  assert.deepEqual(itemList(again), itemList(store));
});

test('mixed ["LeftHand", "Hips"]: routes to both tracks, saved as one object per track', () => {
  const m = makeMotion(40);
  const doc = [{
    type: 'end-effector', joint_names: ['LeftHand', 'Hips'], frame_indices: [10, 11, 20],
    local_joints_rot: [BENT, BENT, ZERO], root_positions: [[0, 0.9, 0.5], [0, 0.9, 0.75], [1, 0.875, 2]],
  }];
  const store = new ConstraintStore(sk);
  store.loadModelJSON(doc, m);
  assert.deepEqual(itemList(store), [['left-hand', 10, 11], ['left-hand', 20, 20], ['hips', 10, 11], ['hips', 20, 20]]);
  for (const f of [10, 11, 20]) assert.deepEqual([...store.entriesAt(f).ee.types].sort(), ['hips', 'left-hand']);
  assert.equal(store.frameCount('left-hand'), 3);
  assert.equal(store.frameCount('hips'), 3);

  const json = store.toModelJSON(40);
  assert.deepEqual(json.map((o) => o.type), ['left-hand', 'end-effector']);
  const [lh, hips] = json;
  assert.deepEqual(Object.keys(hips), HIPS_KEYS);
  assert.deepEqual(hips.joint_names, ['Hips']);
  for (const key of ['frame_indices', 'local_joints_rot', 'root_positions', 'smooth_root_2d']) assert.deepEqual(hips[key], lh[key], key);
  assert.deepEqual(hips.smooth_root_2d, [[0, 0.5], [0, 0.75], [1, 2]], 'no smooth_root_2d in the file: pelvis xz');

  // Name order does not matter; the export reloads to the same store.
  const swapped = new ConstraintStore(sk);
  swapped.loadModelJSON([{ ...doc[0], joint_names: ['Hips', 'LeftHand'] }], m);
  assert.deepEqual(swapped.toModelJSON(40), json);
  const text = JSON.stringify(json);
  const reloaded = new ConstraintStore(sk);
  reloaded.loadModelJSON(JSON.parse(text), m);
  assert.deepEqual(itemList(reloaded), itemList(store));
  assert.equal(JSON.stringify(reloaded.toModelJSON(40)), text);

  // Removing the Hips item at 20 leaves the hand there (types recomputed).
  const hips20 = store.items().find((i) => i.track === 'hips' && i.start === 20).id;
  assert.ok(store.remove(hips20));
  assert.deepEqual([...store.entriesAt(20).ee.types], ['left-hand']);
  assert.deepEqual(store.toModelJSON(40)[1].frame_indices, [10, 11]);
});

test('timeline edits on the hips track: add snapshots the pose, move / interval / remove, out of range', () => {
  const m = makeMotion(40);
  const store = new ConstraintStore(sk);
  store.setNumFrames(40, m);
  const id = store.addKeyframe('hips', 12, m);
  assert.ok(id);
  assert.deepEqual(itemList(store), [['hips', 12, 12]]);
  sameSnapshot(store.entriesAt(12).ee.pose, m, 12);
  close(store.entriesAt(12).ee.pose.positions, m.pose(12).positions, 1e-6, 'FK of the stored pose');
  assert.deepEqual([...store.entriesAt(12).ee.types], ['hips']);
  assert.equal(store.hasConstraintAt(12), true);
  const s12 = m.snapshot(12);
  const json = store.toModelJSON(40);
  assert.deepEqual(json.map((o) => [o.type, o.joint_names, o.frame_indices]), [['end-effector', ['Hips'], [12]]]);
  assert.deepEqual(json[0].root_positions, [Array.from(s12.root)]);
  assert.deepEqual(json[0].smooth_root_2d, [[s12.root[0], s12.root[2]]]);

  // move = re-snapshot at the destination; carryPose keeps the stored pose
  assert.equal(store.move(id, 20, 20, m), true);
  assert.equal(store.entriesAt(12).ee, undefined);
  sameSnapshot(store.entriesAt(20).ee.pose, m, 20);
  assert.equal(store.move(id, 25, 25, m, { carryPose: true }), true);
  sameSnapshot(store.entriesAt(25).ee.pose, m, 20);

  // interval: one entry per frame, clamped to the clip
  const iv = store.addInterval('hips', 35, 50, m);
  assert.deepEqual(store.items().filter((i) => i.id === iv).map((i) => [i.start, i.end]), [[35, 39]]);
  for (let f = 35; f <= 39; f++) sameSnapshot(store.entriesAt(f).ee.pose, m, f);
  assert.equal(store.frameCount('hips'), 6);
  assert.equal(store.addKeyframe('hips', 40, m), null, 'past the end');
  assert.equal(store.remove(iv), true);
  assert.equal(store.frameCount('hips'), 1);

  // Hips and a limb on one frame share the EE slot; removing Hips keeps the foot
  const lf = store.addKeyframe('left-foot', 25, m);
  assert.deepEqual([...store.entriesAt(25).ee.types].sort(), ['hips', 'left-foot']);
  sameSnapshot(store.entriesAt(25).ee.pose, m, 25); // latest snapshot wins, as for limbs
  assert.deepEqual(store.toModelJSON(40).map((o) => o.type), ['end-effector', 'left-foot']);
  store.remove(id);
  assert.deepEqual([...store.entriesAt(25).ee.types], ['left-foot']);
  store.remove(lf);
  assert.equal(store.entriesAt(25).ee, undefined);
  assert.deepEqual(store.toModelJSON(40), []);
});

test('hips track: write-through, snap and counts / warnings', () => {
  const m = makeMotion(60);
  const store = new ConstraintStore(sk);
  store.setNumFrames(60, m);
  store.addKeyframe('hips', 3, m);

  // joint gizmo drag: the stored pose follows the edited motion
  m.setJointGlobalRotation(3, 0, quatFromAxisAngle([0, 1, 0], 0.7));
  assert.equal(store.writeThrough(3, m, { rootJointMoved: true }), true);
  sameSnapshot(store.entriesAt(3).ee.pose, m, 3);
  // root ground translate: stored rotations kept, root taken from the motion
  const storedLocal = store.entriesAt(3).ee.pose.local.slice();
  m.setJointGlobalRotation(3, 11, quatFromAxisAngle([1, 0, 0], 0.5));
  m.translateXZ(3, 1, -1);
  store.writeThrough(3, m, { rootTranslateOnly: true });
  close(store.entriesAt(3).ee.pose.local, storedLocal, 0);
  close(store.entriesAt(3).ee.pose.root, m.pose(3).root, 0);

  // snap: Hips is an End-Effector target (priority Full-Body -> EE -> 2D Root)
  const pose = store.entriesAt(3).ee.pose;
  m.translateXZ(3, 2, 2);
  assert.equal(store.snapTarget(3).kind, 'ee');
  assert.equal(store.snapToConstraint(3, m), 'ee');
  close(m.snapshot(3).root, pose.root, 0);
  close(m.snapshot(3).local, pose.local, 0);

  // counts and the >= WARN_FRAMES soft warning, frames inside the clip only
  store.addInterval('hips', 40, 40 + WARN_FRAMES - 2, m); // 19 frames + the keyframe at 3
  assert.equal(store.frameCount('hips'), WARN_FRAMES);
  const w = store.warnings();
  assert.deepEqual(Object.keys(w), TRACKS);
  assert.equal(w.hips, true);
  for (const t of TRACKS.filter((t) => t !== 'hips')) assert.equal(w[t], false, t);
  for (const t of LIMBS) assert.equal(store.frameCount(t), 0, t);
  store.setNumFrames(50, m);
  assert.equal(store.frameCount('hips'), 11);
  assert.equal(store.warnings().hips, false);
  assert.ok(store.toModelJSON(50)[0].frame_indices.every((f) => f < 50));
});

test('the six upstream tracks are unchanged: no Hips unless the hips track is used', () => {
  const m = makeMotion(40);
  const store = new ConstraintStore(sk);
  store.addKeyframe('root2d', 5, m);
  store.addKeyframe('right-foot', 9, m);
  store.addKeyframe('left-hand', 3, m);
  store.addKeyframe('left-hand', 9, m);
  store.addInterval('fullbody', 20, 22, m);
  store.addKeyframe('fullbody', 2, m);
  const json = store.toModelJSON(40);
  assert.deepEqual(json.map((o) => o.type), ['fullbody', 'left-hand', 'right-foot', 'root2d']);
  assert.ok(json.every((o) => o.joint_names === undefined));
  assert.equal(store.frameCount('hips'), 0);
  assert.ok(store.entries('ee').every((e) => !e.types.has('hips')));
  // Every EE slot type is an EE track; only the four limbs have a skeleton chain.
  assert.ok(store.entries('ee').every((e) => [...e.types].every((t) => EE_TRACKS.includes(t))));
  for (const t of LIMBS) assert.ok(sk.chainForTrack(t).length > 0, t);
  assert.throws(() => sk.chainForTrack('hips'), /not an end-effector track/);

  // A limb-only generic end-effector still loads into limb tracks and saves as limb objects.
  const limbs = new ConstraintStore(sk);
  limbs.loadModelJSON([{ type: 'end-effector', joint_names: ['RightHand', 'LeftFoot'], frame_indices: [4], local_joints_rot: [ZERO], root_positions: [[0, 0.9, 0]] }], m);
  assert.deepEqual(itemList(limbs), [['right-hand', 4, 4], ['left-foot', 4, 4]]);
  assert.deepEqual(limbs.toModelJSON(40).map((o) => o.type), ['right-hand', 'left-foot']);
});

test('writeThrough(rootTranslateOnly) after Motion#translate (the ground gizmo height drag) sends the new root y in Hips / Full-Body root_positions; 2D Root keeps only x, z', () => {
  const m = makeMotion(30);
  const store = new ConstraintStore(sk);
  store.addKeyframe('hips', 10, m);
  store.addKeyframe('fullbody', 12, m);
  store.addKeyframe('root2d', 10, m);
  store.addKeyframe('root2d', 14, m);
  const local10 = store.entriesAt(10).ee.pose.local.slice();
  const [x10, z10] = m.rootXZ(10);
  const y12 = m.pose(12).root[1];

  // lift the whole pose at 10 by 0.4 m (the editor's 'ground' drag), stored rotations kept
  m.translate(10, 0, 0.4, 0);
  assert.equal(store.writeThrough(10, m, { rootTranslateOnly: true }), true);
  close(store.entriesAt(10).ee.pose.root, m.snapshot(10).root, 0);
  close(store.entriesAt(10).ee.pose.local, local10, 0);
  const r = store.entriesAt(10).root2d;
  assert.deepEqual([r.x, r.z], [x10, z10], '2D Root is ground-plane only');

  // lower the Full-Body keyframe at 12
  m.translate(12, 0, -0.25, 0);
  store.writeThrough(12, m, { rootTranslateOnly: true });

  const json = store.toModelJSON(30);
  const hips = json.find((o) => o.type === 'end-effector');
  const fb = json.find((o) => o.type === 'fullbody');
  assert.ok(Math.abs(hips.root_positions[hips.frame_indices.indexOf(10)][1] - m.snapshot(10).root[1]) < 1e-7);
  assert.ok(Math.abs(fb.root_positions[fb.frame_indices.indexOf(12)][1] - (y12 - 0.25)) < 1e-6);

  // a height-only edit where only a 2D Root exists changes no constraint data
  const rootBefore = JSON.stringify(store.toModelJSON(30).find((o) => o.type === 'root2d'));
  m.translate(14, 0, 0.3, 0);
  store.writeThrough(14, m, { rootTranslateOnly: true });
  assert.equal(JSON.stringify(store.toModelJSON(30).find((o) => o.type === 'root2d')), rootBefore);
  assert.equal(store.entriesAt(14).fullbody, undefined);
  assert.equal(store.entriesAt(14).ee, undefined);
});
