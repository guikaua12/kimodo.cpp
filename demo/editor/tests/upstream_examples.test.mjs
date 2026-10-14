// Loads the shipped upstream SOMA example constraint files with a soma30
// skeleton and re-exports them: frame sets, rotations (compared as matrices),
// root positions and smooth roots must round trip.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { Skeleton } from '../skeleton.js';
import { Motion } from '../motion.js';
import { ConstraintStore } from '../constraints.js';
import { mat3FromAxisAngle } from '../math.js';
import { SOMA30_MODEL } from './fixtures/soma30.mjs';

const sk = new Skeleton(SOMA30_MODEL);
const load = (name) => JSON.parse(fs.readFileSync(new URL(`./fixtures/kimodo-soma-rp/${name}.constraints.json`, import.meta.url), 'utf8'));

// numFrames = motion.npz length of each example (upstream assets/demo/examples).
const EXAMPLES = {
  '03_full_body_keyframes': { frames: 150, items: [['fullbody', 79, 79], ['fullbody', 134, 134]] },
  '04_ee_constraint': {
    frames: 151,
    items: [['left-hand', 28, 28], ['left-hand', 94, 94], ['right-hand', 28, 28], ['right-hand', 94, 94],
      ['left-foot', 28, 28], ['left-foot', 94, 94], ['right-foot', 28, 28], ['right-foot', 94, 94]],
  },
  '06_root_waypoints': { frames: 181, items: [['root2d', 0, 0], ['root2d', 90, 90], ['root2d', 180, 180]] },
  '07_mixed_constraints': { frames: 152, items: [['fullbody', 108, 108], ['root2d', 0, 151]] },
};

/** {frame -> value} per type; pose types merge the file's EE objects by type. */
function byType(doc) {
  const out = new Map();
  for (const o of doc) {
    if (!out.has(o.type)) out.set(o.type, new Map());
    o.frame_indices.forEach((f, k) => out.get(o.type).set(f, {
      rot: o.local_joints_rot?.[k], root: o.root_positions?.[k], smooth: o.smooth_root_2d[k], heading: o.global_root_heading?.[k],
    }));
  }
  return out;
}

function maxMatDiff(aaA, aaB) {
  const A = mat3FromAxisAngle(aaA), B = mat3FromAxisAngle(aaB);
  return Math.max(...A.map((v, i) => Math.abs(v - B[i])));
}

for (const [name, spec] of Object.entries(EXAMPLES)) {
  test(`upstream example ${name}: load + re-export round trip`, () => {
    const doc = load(name);
    const motion = Motion.rest(sk, spec.frames);
    const store = new ConstraintStore(sk);
    store.setNumFrames(spec.frames, motion);
    store.loadModelJSON(doc, motion);

    const items = store.items().map((i) => [i.track, i.start, i.end]);
    assert.deepEqual(items, spec.items);

    const exported = store.toModelJSON(spec.frames);
    JSON.parse(JSON.stringify(exported));
    const want = byType(doc), got = byType(exported);
    assert.deepEqual([...got.keys()].sort(), [...want.keys()].sort(), 'same constraint types');
    let worstRot = 0;
    for (const [type, frames] of want) {
      const g = got.get(type);
      assert.deepEqual([...g.keys()].sort((a, b) => a - b), [...frames.keys()].sort((a, b) => a - b), `${type} frame set`);
      for (const [f, w] of frames) {
        const v = g.get(f);
        assert.deepEqual(v.smooth, w.smooth, `${type}@${f} smooth_root_2d`);
        assert.deepEqual(v.heading, w.heading);
        if (!w.rot) continue;
        assert.deepEqual(v.root, w.root, `${type}@${f} root_positions`);
        assert.equal(v.rot.length, 30);
        v.rot.forEach((aa, j) => {
          const d = maxMatDiff(aa, w.rot[j]);
          worstRot = Math.max(worstRot, d);
          assert.ok(d < 1e-5, `${type}@${f} joint ${j}: rotation differs by ${d}`);
        });
        // The examples (and our pelvis default) have smooth_root_2d == root_positions xz.
        assert.deepEqual(v.smooth, [w.root[0], w.root[2]], `${type}@${f} smooth == pelvis xz`);
      }
    }
    assert.ok(worstRot < 1e-5);

    // Order: fullbody, limb types, root2d (last-wins merging gives root2d priority).
    const order = exported.map((o) => o.type);
    const rank = (t) => (t === 'fullbody' ? 0 : t === 'root2d' ? 2 : 1);
    assert.deepEqual(order, [...order].sort((a, b) => rank(a) - rank(b)));
  });
}

test('upstream example 04: four limb files at shared frames merge into one EE slot per frame', () => {
  const store = new ConstraintStore(sk);
  store.loadModelJSON(load('04_ee_constraint'), Motion.rest(sk, 151));
  for (const f of [28, 94]) {
    assert.deepEqual([...store.entriesAt(f).ee.types].sort(), ['left-foot', 'left-hand', 'right-foot', 'right-hand']);
  }
  for (const t of ['left-hand', 'right-hand', 'left-foot', 'right-foot']) assert.equal(store.frameCount(t), 2);
});

test('upstream example 03: stored poses re-derive the file root positions; snapshot-created FB matches pelvis', () => {
  const doc = load('03_full_body_keyframes');
  const store = new ConstraintStore(sk);
  store.loadModelJSON(doc, null);
  const pose = store.entriesAt(79).fullbody;
  assert.deepEqual(Array.from(pose.root), doc[0].root_positions[0].map((v) => Math.fround(v)));
  // Snap the loaded pose into a motion and re-add it from the motion: smooth_root_2d defaults to the pelvis xz.
  const motion = Motion.rest(sk, 150);
  store.snapToConstraint(79, motion);
  const again = new ConstraintStore(sk);
  again.addKeyframe('fullbody', 79, motion);
  const o = again.toModelJSON(150)[0];
  assert.deepEqual(o.smooth_root_2d[0], [o.root_positions[0][0], o.root_positions[0][2]]);
  o.local_joints_rot[0].forEach((aa, j) => assert.ok(maxMatDiff(aa, doc[0].local_joints_rot[0][j]) < 1e-5));
});
