import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {
  FPS, MIN_FRAMES, MAX_FRAMES, DEFAULT_DYNAMICS, PRESETS, presetById, defaultParams, normalizeParams,
  simulatePreset, buildPreset, presetSVG, gaitVerb, previewBuffers,
} from '../presets.js';
import { PRESET_POSES } from '../preset_poses.js';
import { Skeleton } from '../skeleton.js';
import { Motion } from '../motion.js';
import { ConstraintStore, validateModelJSON } from '../constraints.js';
import { quatFromAxisAngle, quatToAxisAngle, wrapAngle } from '../math.js';
import { SOMA30_MODEL, TOY_MODEL } from './fixtures/soma30.mjs';

const sk = new Skeleton(SOMA30_MODEL);
const DEG = Math.PI / 180;
const IDS = ['acceleration', 'constant_speed', 'strafe', 'arc', 'forward_turn', 'strafe_turn', 'start', 'start_turn',
  'stop', 'stop_turn', 'spin', 'turn_in_place'];
const near = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg ?? ''} ${a} vs ${b} (tol ${tol})`);
const close = (a, b, tol, msg) => {
  assert.equal(a.length, b.length, msg);
  for (let i = 0; i < a.length; i++) near(a[i], b[i], tol, `${msg ?? ''} [${i}]`);
};
const sideSign = (s) => (s === 'right' ? -1 : 1);

const probe = JSON.parse(fs.readFileSync(new URL('../../../docs/motion-stage-research/data/probe-paths.json', import.meta.url), 'utf8'));

/** Every number param at its min (or max), sides as given; normalizeParams fits the rest. */
function extreme(id, which, side = 'left') {
  const p = {};
  for (const pd of presetById(id).params) {
    if (pd.type === 'number') p[pd.key] = which === 'min' ? pd.min : pd.max;
    else if (pd.type === 'side') p[pd.key] = side;
  }
  return p;
}
/** Defaults + a few parameter extremes for one preset. */
function variants(id) {
  const out = [{ name: 'defaults', p: defaultParams(id) }, { name: 'min', p: extreme(id, 'min') }, { name: 'max', p: extreme(id, 'max') }];
  const def = presetById(id);
  if (def.params.some((d) => d.key === 'angle')) {
    const a = def.params.find((d) => d.key === 'angle');
    out.push({ name: 'angle max, long clip', p: { ...defaultParams(id), angle: a.max, duration: 8 } });
  }
  if (def.params.some((d) => d.key === 'speed')) out.push({ name: 'run', p: { ...defaultParams(id), speed: 5, speed2: 5.5 } });
  return out;
}
const withSide = (id, p, s) => {
  const q = { ...p };
  for (const pd of presetById(id).params) if (pd.type === 'side') q[pd.key] = s;
  return q;
};

// ---------------------------------------------------------------------------
// probe reproduction (locks every sign)
// ---------------------------------------------------------------------------

test('probe-paths.json has the expected shape', () => {
  for (const k of ['turn_spring', 'line', 'strafe_right']) {
    assert.equal(probe[k].xz.length, 120, k);
    assert.equal(probe[k].heading_rad.length, 120, k);
    assert.equal(probe[k].xz[0].length, 2, k);
  }
});

function matchProbe(key, sim) {
  const ref = probe[key];
  assert.equal(sim.frames, ref.xz.length, `${key} frames`);
  for (let f = 0; f < sim.frames; f++) {
    const d = Math.hypot(sim.xz[f][0] - ref.xz[f][0], sim.xz[f][1] - ref.xz[f][1]);
    assert.ok(d <= 0.005, `${key} frame ${f}: xz ${sim.xz[f]} vs ${ref.xz[f]} (${(d * 1000).toFixed(2)} mm)`);
    const dh = Math.abs(sim.heading[f] - ref.heading_rad[f]) / DEG;
    assert.ok(dh <= 0.5, `${key} frame ${f}: heading ${sim.heading[f]} vs ${ref.heading_rad[f]}`);
  }
}

test('forward_turn reproduces the probe turn_spring path (90 deg left, h 0.09 / 0.07)', () => {
  const sim = simulatePreset('forward_turn', {
    speed: 1.2, angle: 90, side: 'left', first: 2.0, duration: 4.0, turnStyle: 'default',
    turnHalflife: 0.09, facingHalflife: 0.07,
  });
  matchProbe('turn_spring', sim);
  close(sim.xz[119], [2.204, 2.556], 0.001, 'end point');
  near(sim.heading[119], Math.PI / 2, 1e-6, 'end heading');
  assert.ok(sim.xz[119][0] > 0, 'a left turn ends at +X (the character\'s left)');
});

test('constant_speed reproduces the probe line; strafe right reproduces strafe_right', () => {
  matchProbe('line', simulatePreset('constant_speed', { speed: 1.2, duration: 4 }));
  const strafe = simulatePreset('strafe', { speed: 1.0, angle: 90, side: 'right', duration: 4 });
  matchProbe('strafe_right', strafe);
  assert.ok(strafe.xz[119][0] < 0, 'strafing right moves toward -X');
});

// ---------------------------------------------------------------------------
// definitions and params
// ---------------------------------------------------------------------------

test('PRESETS: 12 frozen definitions in Motorica order', () => {
  assert.deepEqual(PRESETS.map((d) => d.id), IDS);
  assert.ok(Object.isFrozen(PRESETS) && Object.isFrozen(PRESETS[0]) && Object.isFrozen(PRESETS[0].params[0]));
  assert.equal(FPS, 30); assert.equal(MIN_FRAMES, 60); assert.equal(MAX_FRAMES, 300);
  assert.deepEqual({ ...DEFAULT_DYNAMICS }, { turnHalflife: 0.12, facingHalflife: 0.12, speedHalflife: 0.3, substeps: 10 });
  const experimental = PRESETS.filter((d) => d.experimental).map((d) => d.id);
  assert.deepEqual(experimental, ['strafe_turn', 'start_turn', 'stop_turn', 'spin']);
  for (const d of PRESETS) {
    assert.equal(typeof d.label, 'string');
    assert.match(d.description, /^[A-Z].*\.$/, `${d.id} description is one sentence`);
    const keys = new Set();
    for (const pd of d.params) {
      assert.ok(!keys.has(pd.key), `${d.id}: duplicate param ${pd.key}`); keys.add(pd.key);
      assert.ok(['number', 'side', 'select', 'bool'].includes(pd.type), `${d.id}.${pd.key} type`);
      assert.ok(['main', 'dynamics', 'pins'].includes(pd.group), `${d.id}.${pd.key} group`);
      if (pd.type === 'number') assert.ok(pd.min <= pd.default && pd.default <= pd.max && pd.step > 0, `${d.id}.${pd.key} range`);
      if (pd.group === 'dynamics') assert.deepEqual([pd.min, pd.max, pd.step, pd.unit], [0.03, 1, 0.01, 's']);
      if (pd.type === 'select' || pd.type === 'side') assert.ok(pd.options.some((o) => o.value === pd.default));
    }
    assert.ok(keys.has('duration'), `${d.id} has a duration`);
  }
  assert.equal(presetById('forward_turn').label, 'Forward turn');
  assert.equal(presetById('arc').label, 'Constant speed/Arc');
  assert.throws(() => presetById('moonwalk'), /unknown preset "moonwalk"/);
});

test('defaultParams returns a fresh object with the spec defaults', () => {
  const a = defaultParams('forward_turn');
  assert.deepEqual(a, { speed: 1.2, angle: 90, side: 'left', first: 2, duration: 4, turnStyle: 'default', turnHalflife: 0.12, facingHalflife: 0.12, groundPin: true });
  a.speed = 3;
  assert.equal(defaultParams('forward_turn').speed, 1.2);
  assert.deepEqual(defaultParams('acceleration'), { speed: 1.2, speed2: 2.6, first: 1.6, duration: 4, speedHalflife: 0.3, groundPin: true });
  assert.deepEqual(defaultParams('turn_in_place'), { angle: 180, side: 'left', duration: 3 });
  assert.equal(defaultParams('stop_turn').duration, 5);
  assert.equal(defaultParams('stop').endIdle, true);
  assert.deepEqual(
    [defaultParams('spin').speed2, defaultParams('spin').angle, defaultParams('spin').spin, defaultParams('strafe_turn').startAngle],
    [1.0, 180, 0.8, 0]);
});

test('normalizeParams: defaults, clamping, rounding, sub-durations, idempotent', () => {
  const p = normalizeParams('forward_turn', { speed: '2.04', angle: 400, side: 'up', first: 9.5, duration: 1, turnStyle: 'post', bogus: 1, groundPin: 'false' });
  assert.deepEqual(p, { speed: 2, angle: 180, side: 'left', first: 1.5, duration: 2, turnStyle: 'post', turnHalflife: 0.12, facingHalflife: 0.12, groundPin: false });
  assert.deepEqual(normalizeParams('forward_turn', p), p);
  assert.equal(normalizeParams('forward_turn', { duration: 12 }).duration, 10);
  assert.equal(normalizeParams('forward_turn', { turnHalflife: 0.004 }).turnHalflife, 0.03);
  assert.equal(normalizeParams('forward_turn', { turnHalflife: 0.123 }).turnHalflife, 0.12);
  assert.equal(normalizeParams('forward_turn', { speed: NaN }).speed, 1.2);
  assert.deepEqual(normalizeParams('turn_in_place', null), defaultParams('turn_in_place'));
  // every event leaves >= 0.5 s
  assert.equal(normalizeParams('stop', { moving: 5, duration: 3 }).moving, 2.5);
  assert.equal(normalizeParams('start', { idle: 3, duration: 2 }).idle, 1.5);
  const spin = normalizeParams('spin', { first: 9, spin: 2, duration: 3 });
  assert.ok(spin.first + spin.spin <= 2.5 + 1e-9 && spin.spin >= 0.3, JSON.stringify(spin));
  const st = normalizeParams('stop_turn', { moving: 9, duration: 2 });
  assert.ok((st.moving + 0.3 + 0.5) * 30 <= 60 - 14 + 1e-9, JSON.stringify(st));
  for (const id of IDS) {
    for (const v of variants(id)) {
      const n = normalizeParams(id, v.p);
      const T = Math.round(n.duration * 30);
      assert.ok(T >= MIN_FRAMES && T <= MAX_FRAMES, `${id} ${v.name}: T ${T}`);
      assert.deepEqual(normalizeParams(id, n), n, `${id} ${v.name} idempotent`);
    }
  }
});

// ---------------------------------------------------------------------------
// simulation invariants
// ---------------------------------------------------------------------------

for (const id of IDS) {
  test(`${id}: frames, origin, finite values, mirror and end heading`, () => {
    for (const v of variants(id)) {
      const tag = `${id} ${v.name}`;
      const left = withSide(id, v.p, 'left');
      const b = buildPreset(id, left);
      const sim = b.sim;
      const T = sim.frames;
      assert.equal(T, b.frames, tag);
      assert.equal(T, Math.round(b.params.duration * 30), tag);
      assert.ok(T >= MIN_FRAMES && T <= MAX_FRAMES, `${tag}: T ${T}`);
      assert.ok(sim.xz.length === T && sim.heading.length === T && sim.speed.length === T, tag);
      assert.deepEqual(sim.xz[0], [0, 0], `${tag}: starts at the origin`);
      assert.equal(sim.heading[0], 0, `${tag}: starts facing +Z`);
      for (let f = 0; f < T; f++) {
        assert.ok(Number.isFinite(sim.xz[f][0]) && Number.isFinite(sim.xz[f][1]) && Number.isFinite(sim.heading[f]) && Number.isFinite(sim.speed[f]), `${tag} frame ${f}`);
      }

      // mirror: right == left with x -> -x, theta -> -theta
      if (presetById(id).params.some((d) => d.type === 'side')) {
        const r = simulatePreset(id, withSide(id, v.p, 'right'));
        for (let f = 0; f < T; f++) {
          near(r.xz[f][0], -sim.xz[f][0], 1e-9, `${tag} mirror x f${f}`);
          near(r.xz[f][1], sim.xz[f][1], 1e-9, `${tag} mirror z f${f}`);
          near(r.heading[f], -sim.heading[f], 1e-9, `${tag} mirror heading f${f}`);
        }
      }

      // end heading vs the requested turn (always facing left here)
      const p = b.params;
      const end = sim.heading[T - 1];
      if (['turn_in_place', 'stop_turn', 'spin'].includes(id)) {
        assert.equal(end, p.angle * DEG, `${tag}: exact end heading`);
      } else if (id === 'forward_turn' || id === 'start_turn') {
        const err = Math.abs(wrapAngle(end - p.angle * DEG)) / DEG;
        const warned = b.warnings.some((w) => /turn does not finish/.test(w));
        if (v.name === 'defaults') assert.ok(err <= 3 && !warned, `${tag}: end heading ${end / DEG}`);
        else assert.ok(err <= 3 || warned, `${tag}: end heading ${end / DEG} without a warning`);
      } else if (['acceleration', 'constant_speed', 'strafe', 'strafe_turn', 'start', 'stop'].includes(id)) {
        assert.ok(sim.heading.every((h) => h === 0), `${tag}: facing stays forward`);
      }
    }
  });
}

test('trajectories go where the recipe says', () => {
  const acc = simulatePreset('acceleration', {});
  near(acc.speed[30], 1.2, 1e-9, 'first part at speed');
  near(acc.speed[119], 2.6, 0.01, 'second part reaches speed2');
  const arc = buildPreset('arc', {});
  near(arc.sim.heading[119] / DEG, 20 * 119 / 30, 5, 'arc heading follows the rate (with spring lag)');
  assert.ok(arc.sim.xz[119][0] > 0, 'left arc curves to +X');
  const st = simulatePreset('strafe_turn', {});
  assert.ok(st.xz[59][0] === 0 && st.xz[119][0] > 1.5, 'forward, then sideways to the left');
  const start = simulatePreset('start', {});
  assert.deepEqual(start.xz[14], [0, 0], 'standing during the idle time');
  assert.ok(start.speed[119] > 1.19);
  const stop = simulatePreset('stop', {});
  assert.ok(stop.speed[59] === 1.2 && stop.speed[119] < 0.01);
  const tip = simulatePreset('turn_in_place', { angle: 90, side: 'right', duration: 3 });
  assert.ok(tip.xz.every(([x, z]) => x === 0 && z === 0), 'turn in place stays at the origin');
  assert.equal(tip.heading[10], 0); assert.equal(tip.heading[76], -90 * DEG);
  const spin = simulatePreset('spin', {});
  assert.ok(spin.xz.every(([x]) => x === 0), 'spin travels straight along +Z');
  assert.equal(spin.heading[60], 0); assert.equal(spin.heading[84], 180 * DEG);
  // turn styles: pre faces the new direction earlier than post
  const pre = simulatePreset('forward_turn', { turnStyle: 'pre' }), post = simulatePreset('forward_turn', { turnStyle: 'post' });
  assert.ok(pre.heading[60] > 0.3 && post.heading[60] === 0 && post.heading[70] > 0);
  assert.deepEqual(pre.xz.slice(0, 61), post.xz.slice(0, 61), 'turn style moves the facing, not the path');
});

// ---------------------------------------------------------------------------
// constraints, pins, store round trip
// ---------------------------------------------------------------------------

test('constraints: pins first, one dense root2d with a heading on every frame, 6 decimals', () => {
  const b = buildPreset('stop', {});
  assert.deepEqual(b.constraints.map((c) => c.type), ['fullbody', 'end-effector', 'root2d']);
  const T = b.frames;
  const root = b.constraints.at(-1);
  assert.deepEqual(root.frame_indices, Array.from({ length: T }, (_, f) => f));
  assert.equal(root.smooth_root_2d.length, T);
  assert.equal(root.global_root_heading.length, T);
  for (let f = 0; f < T; f++) {
    close(root.smooth_root_2d[f], b.sim.xz[f], 5e-7);
    close(root.global_root_heading[f], [Math.cos(b.sim.heading[f]), Math.sin(b.sim.heading[f])], 5e-7);
  }
  const all = JSON.stringify(b.constraints).match(/-?\d+\.\d+(e-?\d+)?/g);
  for (const s of all) assert.ok(Math.abs(Number(s) * 1e6 - Math.round(Number(s) * 1e6)) < 1e-6, `not rounded: ${s}`);
  assert.deepEqual(b.settings, { post_processing: true, root_margin: 0.06 });
  assert.deepEqual(b.pins, [{ frame: 0, kind: 'hips', pose: 'walk' }, { frame: 118, kind: 'fullbody', pose: 'idle.prev' }, { frame: 119, kind: 'fullbody', pose: 'idle.pose' }]);
  assert.deepEqual(b.summary.end, [0, root.smooth_root_2d[T - 1][1]]);
  assert.equal(b.summary.duration_s, 4);
});

test('Hips pin: walk pose, pelvis exactly on (0, 0); only on moving-start presets', () => {
  const walk = PRESET_POSES.soma30.walk;
  for (const id of IDS) {
    const b = buildPreset(id, {});
    const hips = b.constraints.filter((c) => c.type === 'end-effector');
    const standing = ['start', 'start_turn', 'turn_in_place'].includes(id);
    assert.equal(hips.length, standing ? 0 : 1, id);
    if (standing) {
      const fb = b.constraints[0];
      assert.equal(fb.type, 'fullbody', `${id}: frame 0 is a fullbody idle pin`);
      assert.deepEqual(fb.frame_indices.slice(0, 2), [0, 1]);
      continue;
    }
    const h = hips[0];
    assert.deepEqual(h.joint_names, ['Hips']);
    assert.deepEqual(h.frame_indices, [0]);
    assert.equal(h.root_positions[0][0], 0); assert.equal(h.root_positions[0][2], 0);
    assert.equal(h.root_positions[0][1], walk.pelvis[1]);
    assert.deepEqual(h.smooth_root_2d, [[0, 0]]);
    assert.deepEqual(h.local_joints_rot[0], walk.rot);
    const off = buildPreset(id, { groundPin: false });
    assert.ok(!off.constraints.some((c) => c.type === 'end-effector'), `${id}: groundPin off`);
  }
});

/** Hip-line heading (FK) of pose k of a fullbody/EE constraint object. */
function pinHeading(obj, k) {
  const local = new Float32Array(sk.J * 4);
  obj.local_joints_rot[k].forEach((aa, j) => local.set(quatFromAxisAngle(aa), 4 * j));
  return sk.heading(sk.fk(local, obj.root_positions[k]).positions);
}

test('idle pins: placed like clips.py place(); turn_in_place end pins face +-angle (FK)', () => {
  const idle = PRESET_POSES.soma30.idle;
  const start = buildPreset('start', {});
  const fb = start.constraints[0];
  assert.deepEqual(fb.frame_indices, [0, 1]);
  assert.deepEqual(fb.local_joints_rot, [idle.pose.rot, idle.next.rot]);
  assert.deepEqual(fb.root_positions[0], idle.pose.pelvis, 'pelvis offset kept at heading 0');
  assert.deepEqual(fb.smooth_root_2d, [[0, 0], [0, 0]]);
  near(pinHeading(fb, 0), 0, 0.5 * DEG);

  for (const [angle, side] of [[180, 'left'], [90, 'right'], [45, 'left'], [270, 'right'], [360, 'left']]) {
    const b = buildPreset('turn_in_place', { angle, side, duration: 4 });
    const T = b.frames;
    const pins = b.constraints[0];
    assert.equal(pins.type, 'fullbody');
    assert.deepEqual(pins.frame_indices, [0, 1, T - 14, T - 2, T - 1]);
    assert.deepEqual(b.pins.map((p) => p.pose), ['idle.pose', 'idle.next', 'idle.pose', 'idle.prev', 'idle.pose']);
    const target = sideSign(side) * angle * DEG;
    near(pinHeading(pins, 0), 0, 0.5 * DEG, `${angle} ${side} start`);
    for (const k of [2, 3, 4]) {
      near(wrapAngle(pinHeading(pins, k) - target), 0, 0.5 * DEG, `${angle} ${side} end pin ${k}`);
      // pelvis offset rotated with the pose; smooth root on the path point
      const pose = k === 3 ? idle.prev : idle.pose;
      const c = Math.cos(target), s = Math.sin(target);
      close(pins.root_positions[k], [pose.pelvis[0] * c + pose.pelvis[2] * s, pose.pelvis[1], -pose.pelvis[0] * s + pose.pelvis[2] * c], 1e-6);
      assert.deepEqual(pins.smooth_root_2d[k], [0, 0]);
    }
    // non-Hips joints are untouched by the yaw
    assert.deepEqual(pins.local_joints_rot[2].slice(1), idle.pose.rot.slice(1));
  }

  const stopTurn = buildPreset('stop_turn', { side: 'right' });
  const fbs = stopTurn.constraints[0];
  const T = stopTurn.frames;
  assert.deepEqual(fbs.frame_indices, [T - 14, T - 2, T - 1]);
  for (let k = 0; k < 3; k++) {
    near(wrapAngle(pinHeading(fbs, k) + 90 * DEG), 0, 0.5 * DEG, `stop_turn pin ${k}`);
    close(fbs.smooth_root_2d[k], stopTurn.sim.xz[fbs.frame_indices[k]], 1e-6, 'pins sit on the path');
  }
});

test('end idle pins: only once the stop has finished, else a warning', () => {
  assert.ok(!buildPreset('stop', { endIdle: false }).constraints.some((c) => c.type === 'fullbody'));
  const slow = buildPreset('stop', { moving: 3.5, duration: 4, speedHalflife: 1.0 });
  assert.ok(!slow.constraints.some((c) => c.type === 'fullbody'), 'no end pins while still moving');
  assert.ok(slow.warnings.some((w) => /stop is not finished before the end pins/.test(w)), slow.warnings.join('\n'));
  assert.ok(!slow.warnings.some((w) => /does not finish within the clip/.test(w)), 'one warning for one problem');
  const slowNoPins = buildPreset('stop', { moving: 3.5, duration: 4, speedHalflife: 1.0, endIdle: false });
  assert.deepEqual(slowNoPins.warnings, ['The stop does not finish within the clip (1.03 m/s at the end).']);
  const slowTurn = buildPreset('stop_turn', { moving: 3, duration: 4, speedHalflife: 1.0 });
  assert.ok(!slowTurn.constraints.some((c) => c.type === 'fullbody'));
  assert.ok(slowTurn.warnings.some((w) => /not finished/.test(w)));
});

test('non-SOMA skeleton: path only, with a warning', () => {
  for (const id of IDS) {
    const b = buildPreset(id, {}, { skeletonKey: 'g1skel34' });
    assert.deepEqual(b.constraints.map((c) => c.type), ['root2d'], id);
    assert.deepEqual(b.pins, [], id);
    assert.ok(b.warnings.includes('Pose pins need the SOMA skeleton; this preset is sent as a path only.'), id);
  }
  const none = buildPreset('constant_speed', { groundPin: false }, { skeletonKey: 'smplx22' });
  assert.deepEqual(none.warnings, [], 'no pins wanted, nothing to warn about');
});

test('warnings: jog/run speeds, short turn windows, fast spins', () => {
  assert.deepEqual(buildPreset('constant_speed', {}).warnings, []);
  assert.ok(buildPreset('constant_speed', { speed: 2.5 }).warnings.some((w) => /jog\/run quality is untested/.test(w)));
  assert.ok(buildPreset('turn_in_place', { angle: 360, duration: 2 }).warnings.some((w) => /turn window is short/.test(w)));
  assert.ok(!buildPreset('turn_in_place', {}).warnings.some((w) => /turn window/.test(w)));
  assert.ok(buildPreset('spin', { angle: 360, spin: 0.5 }).warnings.some((w) => /spin is very fast/.test(w)));
  const late = buildPreset('forward_turn', { first: 3.5, duration: 4, facingHalflife: 1.0 });
  assert.ok(late.warnings.some((w) => /turn does not finish/.test(w)), late.warnings.join('\n'));
  for (const id of IDS) assert.deepEqual(buildPreset(id, {}).warnings.filter((w) => !/jog\/run/.test(w)), [], `${id} defaults`);
});

for (const id of IDS) {
  test(`${id}: constraints pass validateModelJSON (SOMA) and survive a ConstraintStore round trip`, () => {
    const canon = (aa) => quatToAxisAngle(quatFromAxisAngle(aa));
    for (const v of variants(id)) {
      for (const s of ['left', 'right']) {
        const tag = `${id} ${v.name} ${s}`;
        const b = buildPreset(id, withSide(id, v.p, s));
        validateModelJSON(b.constraints, sk);
        const store = new ConstraintStore(sk);
        store.setNumFrames(b.frames);
        store.loadModelJSON(b.constraints);
        const out = store.toModelJSON();
        assert.deepEqual(out.map((c) => c.type), b.constraints.map((c) => c.type), `${tag}: object order`);
        b.constraints.forEach((c, i) => {
          const o = out[i];
          assert.deepEqual(o.frame_indices, c.frame_indices, `${tag}[${i}] frames`);
          if (c.joint_names) assert.deepEqual(o.joint_names, c.joint_names, `${tag}[${i}] joint_names`);
          c.smooth_root_2d.forEach((xz, k) => close(o.smooth_root_2d[k], xz, 1e-6, `${tag}[${i}] xz ${k}`));
          if (c.type === 'root2d') {
            c.global_root_heading.forEach((h, k) => close(o.global_root_heading[k], h, 1e-6, `${tag} heading ${k}`));
            return;
          }
          c.root_positions.forEach((r, k) => close(o.root_positions[k], r, 1e-6, `${tag}[${i}] root ${k}`));
          c.local_joints_rot.forEach((pose, k) => pose.forEach((aa, j) => close(canon(o.local_joints_rot[k][j]), canon(aa), 1e-6, `${tag}[${i}] rot ${k}.${j}`)));
        });
      }
    }
  });
}

// ---------------------------------------------------------------------------
// prompts
// ---------------------------------------------------------------------------

test('gaitVerb bands', () => {
  assert.equal(gaitVerb(0.3), 'walks');
  assert.equal(gaitVerb(1.99), 'walks');
  assert.equal(gaitVerb(2.0), 'jogs');
  assert.equal(gaitVerb(3.99), 'jogs');
  assert.equal(gaitVerb(4.0), 'runs');
  assert.equal(gaitVerb(6), 'runs');
});

test('prompts: sentence case, number-free, gait and side words agree with the path', () => {
  for (const id of IDS) {
    for (const v of variants(id)) {
      for (const s of ['left', 'right']) {
        const b = buildPreset(id, withSide(id, v.p, s));
        const tag = `${id} ${v.name} ${s}: ${b.prompt}`;
        assert.match(b.prompt, /^A person [a-z].*\.$/, tag);
        assert.doesNotMatch(b.prompt, /\d/, tag);
        if ((b.params.speed ?? 0) >= 2) assert.doesNotMatch(b.prompt.split(/,| and /)[0], /walks/, tag);
        const other = s === 'left' ? 'right' : 'left';
        const hasSide = presetById(id).params.some((d) => d.key === 'side');
        if (hasSide && id !== 'spin' && id !== 'strafe_turn') assert.doesNotMatch(b.prompt, new RegExp(other), tag);
      }
    }
  }
  const P = (id, p = {}) => buildPreset(id, p).prompt;
  assert.equal(P('constant_speed'), 'A person walks forward at a steady pace.');
  assert.equal(P('constant_speed', { speed: 3 }), 'A person jogs forward at a steady pace.');
  assert.equal(P('acceleration'), 'A person walks forward and then breaks into a jog.');
  assert.equal(P('acceleration', { speed2: 1.8 }), 'A person walks forward and gradually speeds up.');
  assert.equal(P('acceleration', { speed2: 4.5 }), 'A person walks forward and then breaks into a run.');
  assert.equal(P('strafe'), 'A person walks at a normal pace.');
  assert.equal(P('strafe', { angle: 10 }), 'A person walks forward at a steady pace.');
  assert.equal(P('strafe', { angle: 45, side: 'right' }), 'A person walks at a normal pace.');
  assert.equal(P('strafe', { speed: 2.5 }), 'A person jogs at a normal pace.');
  assert.equal(P('strafe', { angle: 135 }), 'A person walks at a normal pace.');
  assert.equal(P('strafe', { angle: 180 }), 'A person walks backward.');
  assert.equal(P('arc', { side: 'right' }), 'A person walks forward, curving to the right.');
  assert.equal(P('forward_turn'), 'A person walks forward and turns left.');
  assert.equal(P('forward_turn', { angle: 150, side: 'right' }), 'A person walks forward and turns around to the right.');
  assert.equal(P('strafe_turn'), 'A person walks at a normal pace.');
  assert.equal(P('strafe_turn', { angle: 180 }), 'A person walks forward and then walks backward.');
  assert.equal(P('start'), 'A person stands still and then starts walking forward.');
  assert.equal(P('start_turn'), 'A person stands still, then turns to the left and starts walking.');
  assert.equal(P('stop'), 'A person walks forward, then stops and stands still.');
  assert.equal(P('stop_turn'), 'A person walks forward, stops and turns to the left.');
  assert.equal(P('stop_turn', { angle: 180, side: 'right' }), 'A person walks forward, stops and turns around to the right.');
  assert.equal(P('spin'), 'A person walks forward, spins around to the left and keeps walking backward.');
  assert.equal(P('spin', { angle: 90, side: 'left' }), 'A person walks forward, turns to the left and keeps walking sideways to the right.');
  assert.equal(P('spin', { angle: 360 }), 'A person walks forward, spins all the way around to the left and keeps walking forward.');
  assert.equal(P('turn_in_place'), 'A person stands still and then turns around to the left.');
  assert.equal(P('turn_in_place', { angle: 90, side: 'right' }), 'A person stands still and then turns to the right.');
});

// ---------------------------------------------------------------------------
// SVG
// ---------------------------------------------------------------------------

/** All numeric tokens of an SVG string. */
const svgNumbers = (svg) => (svg.match(/-?(?:\d+\.?\d*|\.\d+)(?:e-?\d+)?|NaN|Infinity/g) || []).map(Number);
const attr = (svg, cls, name) => Number(new RegExp(`class="${cls}"[^>]*\\b${name}="([^"]+)"`).exec(svg)[1]);

test('presetSVG: an <svg> with a path and finite coordinates for every preset', () => {
  for (const id of IDS) {
    const sim = simulatePreset(id, {});
    for (const opts of [{}, { width: 480, height: 360, grid: true, className: 'big', title: 'A <b> & "c"' }]) {
      const svg = presetSVG(sim, opts);
      assert.ok(svg.startsWith('<svg'), id);
      assert.ok(svg.endsWith('</svg>'), id);
      assert.match(svg, /<path class="preset-path" d="M/, id);
      assert.match(svg, /class="preset-facing"/, id);
      assert.match(svg, /class="preset-start"/, id);
      assert.match(svg, /class="preset-end"/, id);
      assert.doesNotMatch(svg, /NaN|Infinity|undefined/, id);
      for (const n of svgNumbers(svg)) assert.ok(Number.isFinite(n), `${id}: ${n}`);
      if (opts.grid) {
        assert.match(svg, /class="preset-grid"/);
        assert.match(svg, /class="preset-svg big"/);
        assert.match(svg, /<title>A &lt;b&gt; &amp; &quot;c&quot;<\/title>/);
      } else {
        assert.doesNotMatch(svg, /preset-grid/);
      }
    }
  }
});

test('presetSVG: forward is up, the character\'s left is screen-left; turn in place is centred', () => {
  const W = 200, H = 100;
  const fwd = presetSVG(simulatePreset('constant_speed', {}), { width: W, height: H });
  assert.ok(attr(fwd, 'preset-end', 'cy') < attr(fwd, 'preset-start', 'cy'), 'forward runs up the screen');
  near(attr(fwd, 'preset-end', 'cx'), attr(fwd, 'preset-start', 'cx'), 1e-9);
  const left = presetSVG(simulatePreset('strafe', { side: 'left' }), { width: W, height: H });
  assert.ok(attr(left, 'preset-end', 'cx') < attr(left, 'preset-start', 'cx'), 'left strafe ends screen-left');
  const right = presetSVG(simulatePreset('strafe', { side: 'right' }), { width: W, height: H });
  assert.ok(attr(right, 'preset-end', 'cx') > attr(right, 'preset-start', 'cx'));
  const tip = presetSVG(simulatePreset('turn_in_place', {}), { width: W, height: H });
  assert.equal(attr(tip, 'preset-start', 'cx'), W / 2);
  assert.equal(attr(tip, 'preset-start', 'cy'), H / 2);
  // one facing triangle every markerEvery frames plus the last frame
  const sim = simulatePreset('constant_speed', {});
  const count = (svg) => (/class="preset-facing" d="([^"]*)"/.exec(svg)[1].match(/Z/g) || []).length;
  assert.equal(count(presetSVG(sim)), 9); // frames 0, 15, ..., 105 and 119
  assert.equal(count(presetSVG(sim, { markerEvery: 60 })), 3); // 0, 60, 119
  // the start triangle of a forward path points up (tip above the start point)
  const tri = /class="preset-facing" d="M([-\d.]+) ([-\d.]+)/.exec(fwd);
  assert.ok(Number(tri[2]) < attr(fwd, 'preset-start', 'cy'));
});

// ---------------------------------------------------------------------------
// preview buffers
// ---------------------------------------------------------------------------

test('previewBuffers (SOMA): anchor poses along the path, turned by the heading', () => {
  for (const id of ['forward_turn', 'start_turn', 'turn_in_place', 'strafe']) {
    const b = buildPreset(id, { side: 'right' });
    const T = b.frames;
    const { local, root } = previewBuffers(b, sk);
    assert.ok(local instanceof Float32Array && local.length === T * sk.J * 4, id);
    assert.ok(root instanceof Float32Array && root.length === T * 3, id);
    const m = new Motion(sk, T, local, root);
    for (let f = 0; f < T; f += 7) {
      near(wrapAngle(m.heading(f) - b.sim.heading[f]), 0, 0.5 * DEG, `${id} heading f${f}`);
      const anchor = b.sim.speed[f] <= 0.3 ? PRESET_POSES.soma30.idle.pose : PRESET_POSES.soma30.walk;
      near(root[3 * f + 1], anchor.pelvis[1], 1e-6, `${id} pelvis height f${f}`);
      const th = b.sim.heading[f], c = Math.cos(th), s = Math.sin(th);
      // idle: offset from the smooth root like the idle pins; walk: pelvis on the path like the Hips pin
      const [dx, , dz] = anchor === PRESET_POSES.soma30.walk ? [0, 0, 0] : anchor.pelvis;
      close([root[3 * f], root[3 * f + 2]], [b.sim.xz[f][0] + dx * c + dz * s, b.sim.xz[f][1] - dx * s + dz * c], 1e-5, `${id} pelvis xz f${f}`);
    }
  }
  // frame 0 of a moving preset is the Hips pin's pose: a snapshot there keeps the pin
  const ft = buildPreset('forward_turn', {});
  const pin = ft.constraints.find((c) => c.type === 'end-effector');
  const pv = previewBuffers(ft, sk);
  close(Array.from(pv.root.subarray(0, 3)), pin.root_positions[0], 1e-6, 'preview f0 = Hips pin root');
  const tip = buildPreset('turn_in_place', {});
  const { local } = previewBuffers(tip, sk);
  // standing preset: idle pose (joint 1 = Spine1 untouched by the yaw)
  close(local.subarray(4, 8), quatFromAxisAngle(PRESET_POSES.soma30.idle.pose.rot[1]), 1e-6);
});

test('previewBuffers (other skeletons): rest pose translated and turned', () => {
  const toy = new Skeleton(TOY_MODEL);
  const b = buildPreset('arc', {}, { skeletonKey: toy.key });
  const { local, root } = previewBuffers(b, toy);
  assert.equal(local.length, b.frames * toy.J * 4);
  const m = new Motion(toy, b.frames, local, root);
  for (let f = 0; f < b.frames; f += 10) {
    close([root[3 * f], root[3 * f + 1], root[3 * f + 2]], [b.sim.xz[f][0], toy.restHeight, b.sim.xz[f][1]], 1e-6);
    near(wrapAngle(m.heading(f) - b.sim.heading[f]), 0, 1e-5, `toy heading f${f}`);
  }
});
