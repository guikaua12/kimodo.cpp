import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { Skeleton } from '../skeleton.js';
import { Motion } from '../motion.js';
import { ConstraintStore, TRACKS, TRACK_LABELS, EE_TRACKS, WARN_FRAMES, validateModelJSON } from '../constraints.js';
import { quatFromAxisAngle, quatToMatrix, mat3FromAxisAngle } from '../math.js';
import { SOMA30_FROM_77 } from '../soma77.js';
import { SOMA30_MODEL } from './fixtures/soma30.mjs';

const sk = new Skeleton(SOMA30_MODEL);
const J = sk.J;
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

test('track constants', () => {
  assert.deepEqual(TRACKS, ['fullbody', 'root2d', 'left-hand', 'right-hand', 'left-foot', 'right-foot']);
  assert.equal(TRACK_LABELS.root2d, '2D Root');
  assert.equal(TRACK_LABELS['left-hand'], 'Left Hand');
  assert.equal(WARN_FRAMES, 20);
});

test('addKeyframe snapshots the motion at the clicked frame; out of range is rejected', () => {
  const m = makeMotion(30);
  const store = new ConstraintStore(sk);
  let events = 0;
  store.addEventListener('change', () => events++);
  const fb = store.addKeyframe('fullbody', 10, m);
  const r = store.addKeyframe('root2d', 12, m);
  assert.equal(events, 2);
  assert.ok(fb && r && fb !== r);
  assert.equal(store.addKeyframe('fullbody', 30, m), null, 'past the end');
  assert.equal(store.addKeyframe('fullbody', -1, m), null);
  assert.throws(() => store.addKeyframe('hands', 3, m), /unknown constraint track/);

  const e10 = store.entriesAt(10);
  sameSnapshot(e10.fullbody, m, 10);
  close(e10.fullbody.positions, m.pose(10).positions, 1e-6, 'FK of stored pose');
  const e12 = store.entriesAt(12);
  assert.deepEqual(Object.keys(e12), ['root2d']);
  close([e12.root2d.x, e12.root2d.z], [m.pose(12).positions[0], m.pose(12).positions[2]], 0, 'pelvis xz');
  assert.equal(e12.root2d.heading, null);
  assert.deepEqual(store.items().map((i) => [i.track, i.start, i.end, i.locked]), [['fullbody', 10, 10, false], ['root2d', 12, 12, false]]);
  assert.equal(store.frameCount('fullbody'), 1);
  assert.ok(store.hasConstraintAt(12) && !store.hasConstraintAt(11));
});

test('intervals: inclusive, clamped, one entry per frame sharing the id; frameCount counts frames', () => {
  const m = makeMotion(30);
  const store = new ConstraintStore(sk);
  const id = store.addInterval('left-foot', 25, 40, m); // clamped to 29
  const it = store.items().find((i) => i.id === id);
  assert.deepEqual([it.start, it.end], [25, 29]);
  assert.equal(store.frameCount('left-foot'), 5);
  assert.equal(store.frameCount('right-foot'), 0);
  for (let f = 25; f <= 29; f++) sameSnapshot(store.entriesAt(f).ee.pose, m, f);
  const swapped = store.addInterval('root2d', 8, 3, m);
  assert.deepEqual(store.items().filter((i) => i.id === swapped).map((i) => [i.start, i.end]), [[3, 8]]);
  assert.equal(store.addInterval('root2d', 40, 50, m), null);
});

test('keyframe and interval on the same frame share one slot (last add wins, freed when empty)', () => {
  const m = makeMotion(30);
  const store = new ConstraintStore(sk);
  const kf = store.addKeyframe('fullbody', 5, m);
  // edit the motion at 5, then add an interval covering 5: the slot takes the latest snapshot
  m.translateXZ(5, 1, 0);
  const iv = store.addInterval('fullbody', 4, 6, m);
  sameSnapshot(store.entriesAt(5).fullbody, m, 5);
  assert.equal(store.frameCount('fullbody'), 3);
  store.remove(iv);
  assert.ok(store.entriesAt(5).fullbody, 'keyframe slot survives');
  sameSnapshot(store.entriesAt(5).fullbody, m, 5); // data stays (upstream)
  assert.equal(store.frameCount('fullbody'), 1);
  store.remove(kf);
  assert.equal(store.entriesAt(5).fullbody, undefined);
  assert.equal(store.items().length, 0);
});

test('end-effector tracks merge at the same frame; removing one recomputes the limb set', () => {
  const m = makeMotion(30);
  const store = new ConstraintStore(sk);
  const lh = store.addKeyframe('left-hand', 7, m);
  m.translateXZ(7, 0, 0.5);
  const rh = store.addKeyframe('right-hand', 7, m);
  const ee = store.entriesAt(7).ee;
  assert.deepEqual([...ee.types].sort(), ['left-hand', 'right-hand']);
  sameSnapshot(ee.pose, m, 7); // latest snapshot wins
  assert.equal(store.frameCount('left-hand'), 1);
  assert.equal(store.frameCount('right-hand'), 1);
  store.remove(lh);
  assert.deepEqual([...store.entriesAt(7).ee.types], ['right-hand'], 'left-hand no longer constrained');
  sameSnapshot(store.entriesAt(7).ee.pose, m, 7);
  const json = store.toModelJSON(30);
  assert.deepEqual(json.map((o) => o.type), ['right-hand']);
  store.remove(rh);
  assert.equal(store.entriesAt(7).ee, undefined);
});

test('move = remove + re-snapshot at the destination; carryPose keeps the stored data', () => {
  const m = makeMotion(40);
  const store = new ConstraintStore(sk);
  const id = store.addKeyframe('fullbody', 10, m);
  const stored = store.entriesAt(10).fullbody;
  assert.equal(store.move(id, 20, 20, m), true);
  assert.equal(store.entriesAt(10).fullbody, undefined);
  sameSnapshot(store.entriesAt(20).fullbody, m, 20);
  assert.equal(store.move(id, 20, 20, m), false, 'no-op move');
  assert.equal(store.move(id, 25, 25, m, { carryPose: true }), true);
  sameSnapshot(store.entriesAt(25).fullbody, m, 20); // carried from frame 20
  // clamped to the timeline
  store.move(id, 99, 99, m);
  assert.deepEqual(store.items()[0].start, 39);
  void stored;

  // interval: pure move keeps the length when clamped; resize with carry keeps covered frames
  const iv = store.addInterval('root2d', 5, 9, m);
  store.move(iv, 37, 41, m);
  assert.deepEqual(store.items().filter((i) => i.id === iv).map((i) => [i.start, i.end]), [[35, 39]]);
  const before = store.entriesAt(36).root2d;
  m.translateXZ(36, 5, 5); // motion changes; carried data must not
  store.move(iv, 33, 39, m, { carryPose: true }); // resize left edge
  assert.deepEqual(store.entriesAt(36).root2d, before);
  close([store.entriesAt(33).root2d.x], [m.rootXZ(33)[0]], 0, 'new frames snapshotted');
  assert.equal(store.frameCount('root2d'), 7);
  // shifted carry
  const iv2 = store.addInterval('left-foot', 0, 2, m);
  const src = [0, 1, 2].map((f) => store.entriesAt(f).ee.pose);
  store.move(iv2, 10, 12, m, { carryPose: true });
  [10, 11, 12].forEach((f, k) => assert.equal(store.entriesAt(f).ee.pose, src[k]));
});

test('writeThrough follows UX section 3.4', () => {
  const m = makeMotion(30);
  const store = new ConstraintStore(sk);
  store.addKeyframe('fullbody', 3, m);
  store.addKeyframe('left-foot', 3, m);
  store.addKeyframe('root2d', 3, m);
  const oldRoot = store.entriesAt(3).root2d;
  const oldLocal = store.entriesAt(3).fullbody.local.slice();

  // joint gizmo drag (non-root): FB + EE take the full pose, 2D root untouched
  m.setJointGlobalRotation(3, 11, quatFromAxisAngle([0, 0, 1], 1.0));
  assert.equal(store.writeThrough(3, m), true);
  sameSnapshot(store.entriesAt(3).fullbody, m, 3);
  sameSnapshot(store.entriesAt(3).ee.pose, m, 3);
  assert.deepEqual(store.entriesAt(3).root2d, oldRoot);
  assert.deepEqual([...store.entriesAt(3).ee.types], ['left-foot']);

  // root joint drag: also writes the 2D root
  m.setRoot(3, [0.4, 0.95, 0.6]);
  store.writeThrough(3, m, { rootJointMoved: true });
  close([store.entriesAt(3).root2d.x, store.entriesAt(3).root2d.z], [0.4, 0.6], 1e-7);

  // root ground translate: FB/EE keep their stored rotations, take the motion root
  const storedLocal = store.entriesAt(3).fullbody.local.slice();
  m.setJointGlobalRotation(3, 12, quatFromAxisAngle([1, 0, 0], 0.5)); // not written through
  m.translateXZ(3, 1, -1);
  store.writeThrough(3, m, { rootTranslateOnly: true });
  close(store.entriesAt(3).fullbody.local, storedLocal, 0);
  close(store.entriesAt(3).fullbody.root, m.pose(3).root, 0);
  close([store.entriesAt(3).root2d.x, store.entriesAt(3).root2d.z], m.rootXZ(3), 0);
  assert.notDeepEqual(Array.from(storedLocal), Array.from(oldLocal));

  assert.equal(store.writeThrough(4, m), false, 'no constraint at that frame');
});

test('locked items cannot be moved or removed; Smooth Path anchors + band', () => {
  const m = makeMotion(50);
  const store = new ConstraintStore(sk);
  store.addKeyframe('root2d', 20, m);
  store.setSmoothPath(true, m, 50, { applyToMotion: false });
  assert.equal(store.smoothPath, true);
  const locked = store.items().filter((i) => i.locked);
  assert.deepEqual(locked.map((i) => [i.start, i.end, i.opacity, !!i.band]).sort(), [[0, 0, 0, false], [0, 49, 0.3, true], [49, 49, 0, false]]);
  for (const it of locked) {
    assert.equal(store.move(it.id, 5, 5, m), false);
    assert.equal(store.remove(it.id), false);
  }
  assert.equal(store.frameCount('root2d'), 3, 'band adds no frames');
  close([store.entriesAt(49).root2d.x, store.entriesAt(49).root2d.z], m.rootXZ(49), 0, 'anchor = pelvis snapshot');

  // dense path: one [x,z] per frame, smoothed (>= 3 entries)
  const path = store.densePath(50);
  assert.equal(path.length, 50);
  assert.ok(path.every((p) => Array.isArray(p) && p.length === 2));
  const json = store.toModelJSON(50);
  const root = json.find((o) => o.type === 'root2d');
  assert.deepEqual(root.frame_indices, Array.from({ length: 50 }, (_, i) => i));
  close(root.smooth_root_2d.flat(), path.flat(), 0);

  store.setSmoothPath(false, m);
  assert.equal(store.smoothPath, false);
  assert.equal(store.items().length, 1);
  assert.equal(store.densePath(50), null);
  assert.deepEqual(store.toModelJSON(50).find((o) => o.type === 'root2d').frame_indices, [20]);
});

test('densePath reproduces upstream interp1d + get_smooth_root_pos (fixture)', () => {
  const fx = JSON.parse(fs.readFileSync(new URL('./fixtures/smooth_root_upstream.json', import.meta.url), 'utf8'));
  for (const c of fx.dense_path) {
    const m = Motion.rest(sk, c.frames);
    for (const [f, x, z] of c.waypoints) m.setRoot(f, [x, 0.9, z]);
    const store = new ConstraintStore(sk);
    for (const [f] of c.waypoints) if (f !== 0 && f !== c.frames - 1) store.addKeyframe('root2d', f, m);
    store.setSmoothPath(true, m, c.frames, { applyToMotion: false });
    const path = store.densePath(c.frames);
    // Waypoints pass through float32 motion storage: compare to 1e-6.
    close(path.flat(), c.path.flat(), 1e-6, `${c.frames} frames, ${c.waypoints.length} waypoints`);
    if (c.waypoints.length === 2) {
      // exactly 2 entries: purely linear, no smoothing
      close(path[45], [c.waypoints[0][1] + (c.waypoints[1][1] - c.waypoints[0][1]) * 45 / 89, c.waypoints[0][2] + (c.waypoints[1][2] - c.waypoints[0][2]) * 45 / 89], 1e-6);
    }
  }
});

test('Smooth Path: applyDensePath moves the pelvis, FB/EE take smooth_root_2d from the path', () => {
  const m = makeMotion(60);
  const store = new ConstraintStore(sk);
  store.addKeyframe('root2d', 30, m);
  m.translateXZ(30, 0.5, 0); // make the path bend
  store.writeThrough(30, m, { rootTranslateOnly: true });
  store.addKeyframe('fullbody', 40, m);
  store.addKeyframe('left-hand', 50, m);
  store.setSmoothPath(true, m, 60); // applyToMotion default
  const path = store.densePath(60);
  for (let f = 0; f < 60; f++) close(m.rootXZ(f), path[f], 1e-6, `pelvis on path at ${f}`);
  const json = store.toModelJSON(60);
  const fb = json.find((o) => o.type === 'fullbody');
  close(fb.smooth_root_2d[0], path[40], 0);
  const lh = json.find((o) => o.type === 'left-hand');
  close(lh.smooth_root_2d[0], path[50], 0);
  // The (expensive) path is cached: Full-Body edits keep it, 2D-root changes recompute it.
  assert.equal(store.densePath(60), path);
  m.setJointGlobalRotation(40, 11, quatFromAxisAngle([0, 0, 1], 0.3));
  store.writeThrough(40, m);
  assert.equal(store.densePath(60), path);
  m.translateXZ(30, 0, 0.2);
  store.writeThrough(30, m, { rootTranslateOnly: true });
  assert.notEqual(store.densePath(60), path);
  // clear() also turns Smooth Path off
  store.clear();
  assert.equal(store.smoothPath, false);
  assert.equal(store.items().length, 0);
});

test('setNumFrames keeps constraints past the end (not counted or sent) and re-anchors the Smooth Path end', () => {
  const m = makeMotion(100);
  const store = new ConstraintStore(sk);
  store.setNumFrames(100, m);
  const a = store.addKeyframe('fullbody', 90, m);
  const b = store.addInterval('left-foot', 50, 80, m);
  const c = store.addKeyframe('root2d', 10, m);
  const d = store.addKeyframe('root2d', 95, m);
  store.setRootHeading(95, 0.5);
  store.setSmoothPath(true, m, 100, { applyToMotion: false });
  const before = store.toModelJSON(100);
  store.setNumFrames(60, m);
  const items = store.items();
  // Kept with their full extent (upstream keeps constraints on a duration change).
  assert.deepEqual(items.filter((i) => i.id === a).map((i) => [i.start, i.end]), [[90, 90]]);
  assert.deepEqual(items.filter((i) => i.id === b).map((i) => [i.start, i.end]), [[50, 80]]);
  assert.ok(items.find((i) => i.id === c) && items.find((i) => i.id === d));
  // ... but only frames inside the clip count and are sent.
  assert.equal(store.frameCount('left-foot'), 10);
  assert.equal(store.frameCount('fullbody'), 0);
  const json = store.toModelJSON(60);
  assert.equal(json.find((o) => o.type === 'fullbody'), undefined);
  assert.deepEqual(json.find((o) => o.type === 'left-foot').frame_indices, [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]);
  assert.ok(json.every((o) => o.frame_indices.every((f) => f < 60)));
  const locked = items.filter((i) => i.locked).map((i) => [i.start, i.end]).sort((x, y) => x[0] - y[0] || x[1] - y[1]);
  assert.deepEqual(locked, [[0, 0], [0, 59], [59, 59]]);
  close([store.entriesAt(59).root2d.x, store.entriesAt(59).root2d.z], m.rootXZ(59), 0);
  assert.equal(store.entriesAt(99).root2d, undefined, 'old anchor slot freed');
  assert.ok(store.entriesAt(95).root2d, 'waypoint past the end kept');
  // A waypoint past the end does not bend the in-clip path: the dense path
  // equals that of a store holding only the in-clip waypoints.
  const ref = new ConstraintStore(sk);
  ref.setNumFrames(60, m);
  ref.addKeyframe('root2d', 10, m);
  ref.setSmoothPath(true, m, 60, { applyToMotion: false });
  assert.deepEqual(store.densePath(60), ref.densePath(60));
  assert.equal(store.addKeyframe('fullbody', 70, m), null, 'beyond the new length');
  // Growing back restores everything (the Smooth Path end is re-anchored).
  store.setNumFrames(100, m);
  assert.ok(store.entriesAt(99).root2d && !store.entriesAt(59).root2d);
  assert.equal(store.densePath().length, 100);
  assert.equal(store.frameCount('left-foot'), 31);
  assert.equal(store.frameCount('fullbody'), 1);
  assert.deepEqual(store.toModelJSON(100), before);
  // An item past the end can be dragged back in (moves are clamped).
  store.setNumFrames(60, m);
  assert.ok(store.move(a, 90, 90, m));
  assert.deepEqual(store.items().filter((i) => i.id === a).map((i) => [i.start, i.end]), [[59, 59]]);
  assert.equal(store.frameCount('fullbody'), 1);
});

test('warnings count only frames inside the clip', () => {
  const m = makeMotion(100);
  const store = new ConstraintStore(sk);
  store.setNumFrames(100, m);
  store.addInterval('fullbody', 70, 99, m);
  for (let f = 50; f < 100; f += 2) store.addKeyframe('root2d', f, m);
  assert.equal(store.warnings().fullbody, true);
  assert.equal(store.warnings().root2d, true);
  store.setNumFrames(75, m);
  assert.equal(store.frameCount('fullbody'), 5);
  assert.equal(store.warnings().fullbody, false);
  assert.equal(store.warnings().root2d, false);
});

test('toModelJSON: order, dropped frames, axis-angle of local rotations, pelvis smooth root', () => {
  const m = makeMotion(40);
  const store = new ConstraintStore(sk);
  store.addKeyframe('root2d', 5, m);
  store.addKeyframe('right-foot', 9, m);
  store.addKeyframe('left-hand', 3, m);
  store.addKeyframe('left-hand', 9, m);
  store.addInterval('fullbody', 20, 22, m);
  store.addKeyframe('fullbody', 2, m);
  const json = store.toModelJSON(21); // frames >= 21 dropped
  assert.deepEqual(json.map((o) => o.type), ['fullbody', 'left-hand', 'right-foot', 'root2d']);
  const fb = json[0];
  assert.deepEqual(Object.keys(fb), ['type', 'frame_indices', 'local_joints_rot', 'root_positions', 'smooth_root_2d']);
  assert.deepEqual(fb.frame_indices, [2, 20]);
  fb.frame_indices.forEach((f, k) => {
    const s = m.snapshot(f);
    assert.equal(fb.local_joints_rot[k].length, J);
    for (let j = 0; j < J; j++) close(mat3FromAxisAngle(fb.local_joints_rot[k][j]), quatToMatrix(s.local.subarray(4 * j, 4 * j + 4)), 1e-6);
    close(fb.root_positions[k], s.root, 0);
    close(fb.smooth_root_2d[k], [s.root[0], s.root[2]], 0);
  });
  assert.deepEqual(json[1].frame_indices, [3, 9]);
  assert.deepEqual(json[2].frame_indices, [9]);
  assert.deepEqual(json[1].local_joints_rot[1], json[2].local_joints_rot[0], 'merged frame: same full pose in both limbs');
  assert.deepEqual(Object.keys(json[3]), ['type', 'frame_indices', 'smooth_root_2d']);
  assert.deepEqual(new ConstraintStore(sk).toModelJSON(10), []);
  JSON.parse(JSON.stringify(json)); // plain JSON
});

test('optional 2D-root heading: emitted only where set (split objects), round trips', () => {
  const m = makeMotion(30);
  const store = new ConstraintStore(sk);
  store.addKeyframe('root2d', 0, m);
  store.addKeyframe('root2d', 10, m);
  store.addInterval('root2d', 20, 22, m);
  assert.equal(store.setRootHeading(10, Math.PI / 2), true);
  assert.equal(store.setRootHeading(21, -0.5), true);
  assert.equal(store.setRootHeading(11, 1), false, 'no entry');
  assert.ok(Math.abs(store.entriesAt(10).root2d.heading - Math.PI / 2) < 1e-15);
  const json = store.toModelJSON(30);
  assert.equal(json.length, 2);
  assert.deepEqual(json[0].frame_indices, [0, 20, 22]);
  assert.equal(json[0].global_root_heading, undefined);
  assert.deepEqual(json[1].frame_indices, [10, 21]);
  close(json[1].global_root_heading.flat(), [Math.cos(Math.PI / 2), 1, Math.cos(-0.5), Math.sin(-0.5)], 1e-15);
  const again = new ConstraintStore(sk);
  again.loadModelJSON(json, m);
  assert.deepEqual(again.toModelJSON(30), json);
  // The interval split across the two objects is merged back into one item.
  assert.deepEqual(again.items().map((i) => [i.track, i.start, i.end]), [['root2d', 0, 0], ['root2d', 10, 10], ['root2d', 20, 22]]);
  // Dragging a waypoint re-snapshots its position but keeps its heading.
  const kf10 = again.items().find((i) => i.start === 10).id;
  again.move(kf10, 15, 15, m);
  assert.ok(Math.abs(again.entriesAt(15).root2d.heading - Math.PI / 2) < 1e-12);
  close([again.entriesAt(15).root2d.x, again.entriesAt(15).root2d.z], m.rootXZ(15), 0);
  const iv = again.items().find((i) => i.start === 20).id;
  again.move(iv, 24, 26, m);
  assert.ok(Math.abs(again.entriesAt(25).root2d.heading + 0.5) < 1e-12, 'interval headings shift with it');
  assert.equal(again.entriesAt(24).root2d.heading, null);
  store.setRootHeading(10, null);
  store.setRootHeading(21, null);
  assert.equal(store.toModelJSON(30).length, 1);
});

test('loadModelJSON: own export round trip, runs -> intervals, one change event', () => {
  const m = makeMotion(50);
  const store = new ConstraintStore(sk);
  store.addKeyframe('fullbody', 4, m);
  store.addInterval('fullbody', 10, 13, m);
  store.addKeyframe('right-hand', 30, m);
  store.addKeyframe('left-foot', 30, m);
  store.addInterval('root2d', 0, 5, m);
  store.addKeyframe('root2d', 40, m);
  const json = store.toModelJSON(50);

  const other = new ConstraintStore(sk);
  let events = 0;
  other.addEventListener('change', (e) => { events++; assert.equal(e.detail.reason, 'load'); });
  const summary = other.loadModelJSON(JSON.parse(JSON.stringify(json)), m);
  assert.equal(events, 1);
  assert.equal(summary.objects, 4);
  assert.deepEqual(summary.frames, { fullbody: 5, root2d: 7, 'left-hand': 0, 'right-hand': 1, 'left-foot': 1, 'right-foot': 0 });
  assert.deepEqual(other.items().map((i) => [i.track, i.start, i.end]),
    [['fullbody', 4, 4], ['fullbody', 10, 13], ['root2d', 0, 5], ['root2d', 40, 40], ['right-hand', 30, 30], ['left-foot', 30, 30]]);
  const back = other.toModelJSON(50);
  assert.deepEqual(back.map((o) => o.type), json.map((o) => o.type));
  back.forEach((o, k) => {
    assert.deepEqual(o.frame_indices, json[k].frame_indices);
    close(o.smooth_root_2d.flat(), json[k].smooth_root_2d.flat(), 1e-12);
    if (o.local_joints_rot) {
      close(o.root_positions.flat(), json[k].root_positions.flat(), 0);
      o.local_joints_rot.forEach((pose, i) => pose.forEach((aa, j) => close(mat3FromAxisAngle(aa), mat3FromAxisAngle(json[k].local_joints_rot[i][j]), 1e-6)));
    }
  });
});

test('loadModelJSON: unsorted frames, generic end-effector, 77-joint SOMA, kept smooth_root_2d', () => {
  const m = makeMotion(20);
  const pose30 = Array.from({ length: J }, (_, j) => [0.01 * j, -0.02 * j, 0.03]);
  const pose77 = Array.from({ length: 77 }, (_, i) => [Math.sin(i), Math.cos(i), 0.5]);
  SOMA30_FROM_77.forEach((i77, j) => { pose77[i77] = pose30[j]; });
  const doc = [
    { type: 'end-effector', joint_names: ['LeftHand', 'RightFoot', 'Hips'], frame_indices: [6, 4, 5],
      local_joints_rot: [pose30, pose30, pose30], root_positions: [[0, 1, 0], [0, 1, 0.1], [0, 1, 0.2]] },
    { type: 'fullbody', frame_indices: [8], local_joints_rot: [pose77], root_positions: [[1, 0.9, 2]], smooth_root_2d: [[1.05, 1.95]] },
  ];
  const store = new ConstraintStore(sk);
  store.loadModelJSON(doc, m);
  assert.deepEqual(store.items().map((i) => [i.track, i.start, i.end]), [['fullbody', 8, 8], ['left-hand', 4, 6], ['right-foot', 4, 6]]);
  assert.deepEqual([...store.entriesAt(5).ee.types].sort(), ['left-hand', 'right-foot']);
  close(store.entriesAt(4).ee.pose.root, [0, 1, 0.1], 1e-7, 'data stays paired with its frame');
  // the 77-joint pose equals the 30-joint pose (non-control joints discarded)
  const fb = store.entriesAt(8).fullbody;
  for (let j = 0; j < J; j++) close(quatToMatrix(fb.local.subarray(4 * j, 4 * j + 4)), mat3FromAxisAngle(pose30[j]), 1e-6);
  const out = store.toModelJSON(20);
  assert.equal(out[0].type, 'fullbody');
  assert.equal(out[0].local_joints_rot[0].length, 30, 'exports the 30-joint control skeleton');
  assert.deepEqual(out[0].smooth_root_2d, [[1.05, 1.95]], 'file smooth_root_2d kept');
  // re-snapshot at that frame drops the override
  store.writeThrough(8, m);
  const s = m.snapshot(8);
  assert.deepEqual(store.toModelJSON(20)[0].smooth_root_2d, [[s.root[0], s.root[2]]]);

  // Objects of the same track merge before run splitting; a later object wins on a shared frame.
  store.loadModelJSON([
    { type: 'root2d', frame_indices: [3, 4], smooth_root_2d: [[0, 0], [1, 1]] },
    { type: 'root2d', frame_indices: [5, 4], smooth_root_2d: [[2, 2], [9, 9]] },
  ], m);
  assert.deepEqual(store.items().map((i) => [i.track, i.start, i.end]), [['root2d', 3, 5]]);
  assert.deepEqual([store.entriesAt(4).root2d.x, store.entriesAt(4).root2d.z], [9, 9]);
});

test('loadModelJSON validates and leaves the store untouched on error', () => {
  const m = makeMotion(10);
  const store = new ConstraintStore(sk);
  store.addKeyframe('fullbody', 1, m);
  const pose = Array.from({ length: J }, () => [0, 0, 0]);
  const good = { type: 'fullbody', frame_indices: [2], local_joints_rot: [pose], root_positions: [[0, 1, 0]] };
  const cases = [
    [{}, /must be an array/],
    [[{ type: 'end-effectors', frame_indices: [] }], /constraints\[0\].*unknown type "end-effectors"/],
    [[null], /must be an object/],
    [[{ ...good, frame_indices: [2.5] }], /frame_indices\[0\].*non-negative integer/],
    [[{ ...good, frame_indices: [2, 2], local_joints_rot: [pose, pose], root_positions: [[0, 1, 0], [0, 1, 0]] }], /duplicate frame 2/],
    [[good, { ...good, local_joints_rot: [pose.slice(1)] }], /constraints\[1\] \(fullbody\)\.local_joints_rot\[0\].*has 29 joints, expected 30 \(soma30\) or 77/],
    [[{ ...good, root_positions: [[0, 1]] }], /root_positions\[0\].*\[3 numbers\]/],
    [[{ ...good, root_positions: [] }], /root_positions.*1 rows/],
    [[{ ...good, local_joints_rot: [pose.map(() => [0, NaN, 0])] }], /axis-angle/],
    [[{ type: 'root2d', frame_indices: [0, 1], smooth_root_2d: [[0, 0]] }], /smooth_root_2d.*2 rows/],
    [[{ type: 'root2d', frame_indices: [0], smooth_root_2d: [[0, 'a']] }], /finite number/],
    [[{ ...good, type: 'end-effector' }], /joint_names.*required/],
    [[{ ...good, type: 'end-effector', joint_names: ['Hips'] }], /at least one of LeftHand/],
    [[{ ...good, type: 'end-effector', joint_names: ['left_hand'] }], /unknown joint "left_hand"/],
  ];
  for (const [doc, re] of cases) {
    assert.throws(() => store.loadModelJSON(doc, m), re, JSON.stringify(doc).slice(0, 80));
  }
  assert.equal(store.items().length, 1, 'store unchanged');
  assert.ok(store.entriesAt(1).fullbody);
  assert.doesNotThrow(() => validateModelJSON([good], sk));
});

test('snapToConstraint priority Full-Body -> End-Effectors -> 2D Root', () => {
  const m = makeMotion(20);
  const store = new ConstraintStore(sk);
  store.addKeyframe('root2d', 3, m);
  store.addKeyframe('left-foot', 3, m);
  const eePose = store.entriesAt(3).ee.pose;
  m.translateXZ(3, 2, 2);
  assert.equal(store.snapTarget(3).kind, 'ee');
  assert.equal(store.snapToConstraint(3, m), 'ee');
  close(m.snapshot(3).root, eePose.root, 0);
  // 2D root only: translate xz, keep height and rotations
  store.addKeyframe('root2d', 5, m);
  const want = store.entriesAt(5).root2d;
  const before = m.snapshot(5);
  m.translateXZ(5, -1, 3);
  assert.equal(store.snapToConstraint(5, m), 'root2d');
  close(m.snapshot(5).root, [want.x, before.root[1], want.z], 1e-6);
  close(m.snapshot(5).local, before.local, 0);
  assert.equal(store.snapToConstraint(6, m), null);
  store.addKeyframe('fullbody', 3, m);
  assert.equal(store.snapTarget(3).kind, 'fullbody');
});

test('warnings: >= 20 frames per type; 2D-root paths are exempt', () => {
  const m = makeMotion(60);
  const store = new ConstraintStore(sk);
  store.addInterval('left-hand', 0, 19, m);
  store.addInterval('root2d', 0, 59, m);
  store.addInterval('fullbody', 0, 18, m);
  const w = store.warnings();
  assert.equal(w['left-hand'], true);
  assert.equal(w.fullbody, false);
  assert.equal(w.root2d, false);
  for (const t of EE_TRACKS.filter((t) => t !== 'left-hand')) assert.equal(w[t], false);
});
