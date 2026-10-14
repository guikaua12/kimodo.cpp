import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {
  POSES, poseById, defaultPoseParams, normalizePoseParams, poseAvailable, pinStride, posePrompt, stripPosePhrases,
  presetPosesFor, guidePins, buildPosedPreset, composeRequest, poseLayer, applyPoseLayer, poseBody, poseSVG, POSE_ARM_JOINTS,
} from '../poses.js';
import { POSE_LIBRARY } from '../pose_library.js';
import { PRESET_POSES } from '../preset_poses.js';
import { PRESETS, defaultParams, buildPreset, previewBuffers } from '../presets.js';
import { Skeleton } from '../skeleton.js';
import { ConstraintStore, validateModelJSON, WARN_FRAMES } from '../constraints.js';
import { quatFromAxisAngle, quatToAxisAngle, wrapAngle } from '../math.js';
import { SOMA30_MODEL } from './fixtures/soma30.mjs';

const sk = new Skeleton(SOMA30_MODEL);
const DEG = Math.PI / 180;
const near = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg ?? ''} ${a} vs ${b} (tol ${tol})`);
const close = (a, b, tol, msg) => {
  assert.equal(a.length, b.length, msg);
  for (let i = 0; i < a.length; i++) near(a[i], b[i], tol, `${msg ?? ''} [${i}]`);
};
const RIFLE = ['rifle_low_ready', 'rifle_collapsed_low_ready'];
/** The cross-body carry (always in the library; its numbers come from anchors/rifle_low_ready_natural). */
const COLLAPSED = 'rifle_collapsed_low_ready';
const anchorOf = (id) => POSE_LIBRARY.soma30.bodies[poseById(id).anchor];
const armsOf = (id) => POSE_LIBRARY.soma30.arms[poseById(id).arms];
const ARM = POSE_ARM_JOINTS.map((n) => sk.index(n));

/** FK of one constraints.json pose (axis-angle rows + root). */
function fkPose(rot, root) {
  const local = new Float32Array(30 * 4);
  rot.forEach((aa, j) => local.set(quatFromAxisAngle(aa), j * 4));
  return sk.fk(local, root).positions;
}
const P = (pos, j) => [pos[3 * j], pos[3 * j + 1], pos[3 * j + 2]];
/** Engine heading of FK positions (RightLeg - LeftLeg). */
const headingOf = (pos) => {
  const r = P(pos, 26), l = P(pos, 22);
  return Math.atan2(r[2] - l[2], -(r[0] - l[0]));
};
/** Joint j relative to the pelvis in the heading frame: [left, up, forward]. */
function rel(pos, j) {
  const h = headingOf(pos);
  const p = P(pos, j), o = P(pos, 0);
  const dx = p[0] - o[0], dz = p[2] - o[2];
  return [dx * Math.cos(h) - dz * Math.sin(h), p[1] - o[1], dx * Math.sin(h) + dz * Math.cos(h)];
}

/** The library without the crouch bodies (an older build), for the degraded paths. */
function noCrouchLibrary() {
  const { crouch_idle: _i, crouch_walk: _w, crouch_walk_next: _n, ...bodies } = POSE_LIBRARY.soma30.bodies;
  return { soma30: { ...POSE_LIBRARY.soma30, bodies } };
}

// ---------------------------------------------------------------------------
// definitions and params
// ---------------------------------------------------------------------------

test('the default poses', () => {
  assert.deepEqual(POSES.map((d) => d.id), ['rifle_low_ready', 'rifle_collapsed_low_ready', 'crouch']);
  assert.deepEqual(POSES.map((d) => d.kind), ['arms', 'arms', 'stance']);
  for (const d of POSES) {
    assert.ok(Object.isFrozen(d) && Object.isFrozen(d.params), d.id);
    assert.ok(d.label && d.description && (d.prompt.phrase || d.prompt.mode === 'none'), d.id);
  }
  assert.throws(() => poseById('nope'), /unknown pose/);
  assert.deepEqual(defaultPoseParams('rifle_low_ready'), { guidePins: true, hold: 100, sway: 100 });
  assert.deepEqual(defaultPoseParams('crouch'), { endAnchor: true });
  // Both rifle poses generate the same way: Kimodo's own cross-body carry as hand pins, with the
  // wording that produces it ("collapsed" itself made it raise the rifle; forward words gave an
  // aiming walk). Only the layer's carry differs.
  for (const id of RIFLE) {
    assert.equal(poseById(id).prompt.phrase, 'holding a rifle in a low ready position with the muzzle pointed down', id);
    assert.equal(poseById(id).anchor, 'rifle_collapsed_walk', id);
  }
  assert.notEqual(poseById('rifle_low_ready').arms, poseById(COLLAPSED).arms);
});

test('normalizePoseParams clamps, rounds, coerces and is idempotent', () => {
  const n = normalizePoseParams('rifle_low_ready', { hold: 123, sway: '42', guidePins: 'false', extra: 1 });
  assert.deepEqual(n, { guidePins: false, hold: 100, sway: 40 });
  assert.deepEqual(normalizePoseParams('rifle_low_ready', n), n);
  assert.deepEqual(normalizePoseParams('rifle_low_ready', null), defaultPoseParams('rifle_low_ready'));
});

test('library data: twelve arm joints under Chest, an upright rifle walk anchor', () => {
  assert.deepEqual(ARM, [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]);
  for (const j of ARM) assert.ok([3, 10, 11, 12, 13, 16, 17, 18, 19].includes(sk.parents[j]), `joint ${j} hangs off Chest`);
  const L = POSE_LIBRARY.soma30;
  assert.ok(poseAvailable('rifle_low_ready') && poseAvailable(COLLAPSED), 'both rifle carries are in the library');
  for (const id of RIFLE) assert.equal(armsOf(id).length, 12, id);
  const w = anchorOf(COLLAPSED);
  near(headingOf(fkPose(w.rot, [0, w.pelvis[1], 0])) / DEG, 0, 0.01, 'pin anchor heading');
  // The collapsed carry is the pin anchor's own arms.
  ARM.forEach((j, k) => close(quatFromAxisAngle(w.rot[j]), quatFromAxisAngle(armsOf(COLLAPSED)[k]), 1e-6, `collapsed arm ${j}`));
  // Collapsed = Kimodo's cross-body carry (tools/mm_factory/anchors/rifle_low_ready_natural/anchor.json):
  // the hand line runs across the body, about 30 cm left and 8 cm forward.
  const c = anchorOf(COLLAPSED);
  const pc = fkPose(c.rot, [0, c.pelvis[1], 0]);
  near(P(pc, 19)[1], 1.2219, 0.002, 'collapsed grip hand height');
  near(P(pc, 13)[1], 1.0011, 0.002, 'collapsed support hand height');
  const yaw = (pos) => {
    const g = rel(pos, 19), sp = rel(pos, 13);
    return Math.atan2(sp[0] - g[0], sp[2] - g[2]) / DEG;
  };
  assert.ok(yaw(pc) > 60, `collapsed muzzle points across the body (${yaw(pc)})`);
  // Low ready on the preset idle body: the muzzle points forward (support hand well ahead of the
  // grip, within 25 deg of straight ahead), angled down, and the stock (the right shoulder) sits
  // about 23 cm behind the grip along the rifle.
  const b = poseBody('rifle_low_ready');
  const pf = fkPose(b.rot, [0, b.pelvis[1], 0]);
  assert.ok(Math.abs(yaw(pf)) < 25, `low-ready muzzle yaw ${yaw(pf)}`);
  const g = rel(pf, 19), sp = rel(pf, 13), sh = rel(pf, 17);
  assert.ok(sp[2] - g[2] > 0.2, 'support hand ahead of the grip');
  const d = [sp[0] - g[0], sp[1] - g[1], sp[2] - g[2]];
  const n = Math.hypot(...d);
  const down = Math.atan2(-d[1], Math.hypot(d[0], d[2])) / DEG;
  assert.ok(down > 10 && down < 35, `muzzle ${down} deg down`);
  const behind = -((sh[0] - g[0]) * d[0] + (sh[1] - g[1]) * d[1] + (sh[2] - g[2]) * d[2]) / n;
  assert.ok(behind > 0.18 && behind < 0.3, `shoulder ${behind} m behind the grip`);
  assert.equal(poseAvailable('crouch'), true);
  assert.equal(poseAvailable('crouch', { library: noCrouchLibrary() }), false);
  // Crouch bodies: upright heading, crouch pelvis heights (research: 0.658 idle, 0.715 walk).
  for (const [k, y] of [['crouch_walk', 0.7145], ['crouch_walk_next', 0.715]]) near(POSE_LIBRARY.soma30.bodies[k].pelvis[1], y, 0.002, k);
  const ci = POSE_LIBRARY.soma30.bodies.crouch_idle.pose;
  near(headingOf(fkPose(ci.rot, [0, ci.pelvis[1], 0])) / DEG, 0, 0.01, 'crouch idle heading');
  near(ci.pelvis[1], 0.658, 0.002, 'crouch idle pelvis');
});

// ---------------------------------------------------------------------------
// prompt
// ---------------------------------------------------------------------------

test('posePrompt splices after the subject, appends otherwise, and swaps cleanly', () => {
  const ph = poseById('rifle_low_ready').prompt.phrase;
  assert.equal(posePrompt('rifle_low_ready', 'A person walks forward at a steady pace.'), `A person ${ph} walks forward at a steady pace.`);
  assert.equal(posePrompt('rifle_low_ready', 'The man stands still.'), `The man ${ph} stands still.`);
  assert.equal(posePrompt('rifle_low_ready', 'walks in a circle'), `walks in a circle ${ph}.`);
  // Crouch adds no words (they make Kimodo hunch); it only swaps standing words.
  assert.equal(posePrompt('crouch', 'A person walks forward.'), 'A person walks forward.');
  assert.equal(posePrompt('crouch', 'A person stands still and then turns.'), 'A person stays still and then turns.');
  const twice = posePrompt('rifle_low_ready', posePrompt('rifle_low_ready', 'A person walks.'));
  assert.equal(twice, `A person ${ph} walks.`, 'idempotent');
  assert.equal(posePrompt('crouch', twice), 'A person walks.', 'switching removes the old phrase');
  assert.equal(posePrompt(null, twice), 'A person walks.');
  assert.equal(stripPosePhrases(`walks in a circle ${ph}.`), 'walks in a circle.');
  assert.equal(posePrompt('rifle_low_ready', ''), `A person ${ph}.`);
  // Only the two places posePrompt writes to are touched; the user's own words stay.
  for (const own of [`A soldier, ${ph}, advances slowly.`, `A person walks ${ph} and then sits down.`, 'A man in a crouch stands up.',
    "A person's arms hang down.", 'A person walks forward,  slowly.']) {
    assert.equal(stripPosePhrases(own), own, own);
  }
  assert.equal(posePrompt('rifle_low_ready', "A person's arms hang."), `A person's arms hang ${ph}.`, 'no splice inside a possessive');
});

// ---------------------------------------------------------------------------
// hand pins
// ---------------------------------------------------------------------------

test('pinStride keeps every track under the 20-frame warning', () => {
  assert.equal(pinStride(60), 8);
  assert.equal(pinStride(120), 8);
  assert.equal(pinStride(150), 9);
  assert.equal(pinStride(300), 18);
  for (let T = 60; T <= 300; T++) {
    const track = { frames: T, xz: Array.from({ length: T }, (_, f) => [0, f * 0.04]), heading: new Array(T).fill(0), speed: new Array(T).fill(1.2) };
    const g = guidePins(COLLAPSED, {}, track);
    assert.ok(g.frames.length < WARN_FRAMES, `T ${T}: ${g.frames.length} pins`);
    assert.equal(g.frames[0], 0);
    assert.ok(T - 1 - g.frames.at(-1) < pinStride(T), `T ${T}: the tail is pinned`);
  }
});

test('guidePins skips frames near fullbody pins, lifts standing frames and honours the toggle', () => {
  const T = 120;
  const track = { frames: T, xz: Array.from({ length: T }, () => [0, 0]), heading: new Array(T).fill(0), speed: new Array(T).fill(0) };
  track.speed.fill(1.1, 60);
  const g = guidePins(COLLAPSED, {}, track, { avoidFrames: [0, 1, 118, 119] });
  assert.ok(g.frames.every((f) => f > 4 && f < 115), JSON.stringify(g.frames));
  const [lh, rh] = g.constraints;
  assert.equal(lh.type, 'left-hand');
  assert.equal(rh.type, 'right-hand');
  assert.deepEqual(lh.root_positions, rh.root_positions);
  const anchorY = anchorOf(COLLAPSED).pelvis[1];
  const idleY = PRESET_POSES.soma30.idle.pose.pelvis[1];
  g.frames.forEach((f, k) => near(lh.root_positions[k][1], f < 60 ? idleY : anchorY, 1e-6, `pelvis y at ${f}`));
  assert.deepEqual(guidePins(COLLAPSED, { guidePins: false }, track), { constraints: [], frames: [] });
  // Frames of unknown facing get no pin.
  const blind = { ...track, heading: track.heading.map((h, f) => (f % 16 ? null : h)) };
  assert.ok(guidePins(COLLAPSED, {}, blind).frames.every((f) => f % 16 === 0));
  assert.deepEqual(guidePins('crouch', {}, track), { constraints: [], frames: [] });
});

// ---------------------------------------------------------------------------
// posed path presets
// ---------------------------------------------------------------------------

for (const poseId of RIFLE) {
  test(`${poseId}: every path preset composes (validation, order, headings, pins, store round trip)`, () => {
    const canon = (aa) => quatToAxisAngle(quatFromAxisAngle(aa));
    const natural = ARM.map((j) => anchorOf(poseId).rot[j]); // the pinned carry
    for (const def of PRESETS) {
      for (const side of ['left', 'right']) {
        const params = { ...defaultParams(def.id) };
        if ('side' in params) params.side = side;
        const tag = `${poseId} ${def.id} ${side}`;
        const plain = buildPreset(def.id, params);
        const b = buildPosedPreset(def.id, params, poseId, {});
        validateModelJSON(b.constraints, sk);
        assert.equal(b.pose.id, poseId);
        assert.ok(b.pose.layer && b.pose.layer.mode === 'hold', tag);
        assert.match(b.prompt, /holding a rifle in a low ready position/, tag);
        // Order: fullbody, Hips end-effector, hands, root2d; the path is untouched.
        const order = ['fullbody', 'end-effector', 'left-hand', 'right-hand', 'root2d'];
        const types = b.constraints.map((c) => c.type);
        assert.deepEqual(types, order.filter((t) => types.includes(t)), `${tag} order`);
        assert.deepEqual(b.constraints.at(-1), plain.constraints.at(-1), `${tag} root2d`);
        const lh = b.constraints.find((c) => c.type === 'left-hand');
        const rh = b.constraints.find((c) => c.type === 'right-hand');
        assert.ok(lh && rh, `${tag} hand pins`);
        assert.deepEqual(lh, { ...rh, type: 'left-hand' }, `${tag}: both hands share frames and poses`);
        assert.ok(lh.frame_indices.length < WARN_FRAMES, tag);
        const fb = b.constraints.find((c) => c.type === 'fullbody');
        for (const f of lh.frame_indices) {
          if (fb) assert.ok(fb.frame_indices.every((g) => Math.abs(g - f) > 3), `${tag}: hand pin ${f} next to a fullbody pin`);
        }
        // The FK heading of every hand pin is the path heading (else the conditioning fights itself).
        lh.frame_indices.forEach((f, k) => {
          const pos = fkPose(lh.local_joints_rot[k], lh.root_positions[k]);
          near(wrapAngle(headingOf(pos) - b.sim.heading[f]) / DEG, 0, 0.01, `${tag} heading at ${f}`);
          close(lh.smooth_root_2d[k], b.sim.xz[f], 1e-6, `${tag} smooth xz at ${f}`);
        });
        // Frame 0 with a Hips pin: the same pose and placement (one store slot).
        const hips = b.constraints.find((c) => c.type === 'end-effector');
        if (hips && lh.frame_indices[0] === 0) {
          assert.deepEqual(lh.local_joints_rot[0], hips.local_joints_rot[0], tag);
          assert.deepEqual(lh.root_positions[0], hips.root_positions[0], tag);
        }
        // Idle pins hold the natural carry's arms on the preset idle body.
        if (fb) {
          fb.local_joints_rot.forEach((rows) => ARM.forEach((j, k) => close(canon(rows[j]), canon(natural[k]), 1e-6, `${tag} idle arm ${j}`)));
          close(fb.local_joints_rot[0][22], plain.constraints[0].local_joints_rot[0][22], 0, `${tag} idle legs kept`);
        }
        // A ConstraintStore round trip keeps every object.
        const store = new ConstraintStore(sk);
        store.setNumFrames(b.frames);
        store.loadModelJSON(b.constraints);
        const out = store.toModelJSON();
        assert.deepEqual(out.map((c) => c.type), types, `${tag} store order`);
        out.forEach((o, i) => {
          const c = b.constraints[i];
          assert.deepEqual(o.frame_indices, c.frame_indices, `${tag}[${i}] frames`);
          if (c.type === 'root2d') return;
          c.root_positions.forEach((r, k) => close(o.root_positions[k], r, 1e-6, `${tag}[${i}] root ${k}`));
          c.local_joints_rot.forEach((pose, k) => pose.forEach((aa, j) => close(canon(o.local_joints_rot[k][j]), canon(aa), 1e-6, `${tag}[${i}] rot ${k}.${j}`)));
        });
      }
    }
  });
}

test('posed presets: hand pins are the natural carry placed on the path', () => {
  const b = buildPosedPreset('forward_turn', { ...defaultParams('forward_turn'), angle: 120 }, 'rifle_collapsed_low_ready', {});
  const lh = b.constraints.find((c) => c.type === 'left-hand');
  const ref = anchorOf(COLLAPSED);
  const refPos = fkPose(ref.rot, [0, ref.pelvis[1], 0]);
  lh.frame_indices.forEach((f, k) => {
    const pos = fkPose(lh.local_joints_rot[k], lh.root_positions[k]);
    for (const j of [13, 15, 19, 21]) close(rel(pos, j), rel(refPos, j), 1e-4, `joint ${j} at frame ${f}`);
  });
  assert.ok(b.pins.some((p) => p.kind === 'hands'));
  assert.equal(b.pins.filter((p) => p.kind === 'hands').length, lh.frame_indices.length);
});

test('posed presets: crouch pins (idle triplet, walk pairs, dense tilted Hips pins), no hand pins, no layer, no words', () => {
  const B = POSE_LIBRARY.soma30.bodies;
  for (const def of PRESETS) {
    for (const endAnchor of [true, false]) {
      const id = def.id;
      const tag = `${id} endAnchor ${endAnchor}`;
      const b = buildPosedPreset(id, defaultParams(id), 'crouch', { endAnchor });
      validateModelJSON(b.constraints, sk);
      assert.equal(b.pose.layer, null, tag);
      assert.ok(!b.constraints.some((c) => c.type === 'left-hand'), tag);
      assert.equal(b.prompt, buildPreset(id, defaultParams(id)).prompt.split('stands still').join('stays still'), tag);
      const T = b.frames;
      const sim = b.sim;
      const fb = b.constraints.find((c) => c.type === 'fullbody');
      const hips = b.constraints.find((c) => c.type === 'end-effector');
      assert.deepEqual(hips.joint_names, ['Hips']);
      // Fullbody pins: crouch bodies only, ascending; the walk pair on a moving start (and end).
      const fbFrames = fb ? fb.frame_indices : [];
      assert.deepEqual(fbFrames, [...fbFrames].sort((a, c) => a - c), tag);
      fb?.root_positions.forEach((r, k) => assert.ok(r[1] < 0.75, `${tag}: fullbody pelvis ${r[1]} at ${fbFrames[k]}`));
      const moving0 = sim.speed[0] > 0.3, movingEnd = sim.speed[T - 1] > 0.3;
      assert.equal(fbFrames.includes(0) && fbFrames.includes(1), moving0 || !!buildPreset(id, defaultParams(id)).pins.some((p) => p.frame === 0 && p.kind === 'fullbody'), tag);
      if (moving0) close(fb.root_positions[fbFrames.indexOf(0)], [B.crouch_walk.pelvis[0], B.crouch_walk.pelvis[1], B.crouch_walk.pelvis[2]], 1e-6, `${tag} walk @0`);
      assert.equal(fbFrames.includes(T - 1) && fbFrames.includes(T - 2), (moving0 && movingEnd && endAnchor) || buildPreset(id, defaultParams(id)).pins.some((p) => p.frame === T - 1), tag);
      // Dense Hips pins: no frame-0 ground pin of the preset, none near a fullbody pin, under 20 frames.
      assert.ok(hips.frame_indices.length < WARN_FRAMES, tag);
      hips.frame_indices.forEach((f, k) => {
        assert.ok(fbFrames.every((g) => Math.abs(g - f) > 2), `${tag}: Hips pin ${f}`);
        const h = 0.66 + 0.04 * Math.min(1, Math.max(0, sim.speed[f] / 0.6));
        near(hips.root_positions[k][1], h, 1e-6, `${tag} height at ${f}`);
        close([hips.root_positions[k][0], hips.root_positions[k][2]], sim.xz[f], 1e-6, `${tag} on the path at ${f}`);
        // Pelvis tilted 15 degrees forward, turned by the path heading.
        const q = quatFromAxisAngle(hips.local_joints_rot[k][0]);
        const up = [2 * (q[0] * q[1] - q[3] * q[2]), 1 - 2 * (q[0] * q[0] + q[2] * q[2]), 2 * (q[1] * q[2] + q[3] * q[0])];
        near(Math.acos(Math.min(1, up[1])) / DEG, 15, 0.01, `${tag} tilt at ${f}`);
        const fwd = [Math.sin(sim.heading[f]), Math.cos(sim.heading[f])];
        assert.ok(up[0] * fwd[0] + up[2] * fwd[1] > 0, `${tag}: tilted forward at ${f}`);
      });
      // A ConstraintStore round trip keeps it.
      const store = new ConstraintStore(sk);
      store.setNumFrames(T);
      store.loadModelJSON(b.constraints);
      assert.deepEqual(store.toModelJSON().map((c) => c.frame_indices), b.constraints.map((c) => c.frame_indices), tag);
      if (!['constant_speed', 'strafe', 'forward_turn', 'start', 'stop', 'turn_in_place'].includes(id)) assert.ok(b.warnings.some((w) => /untested with it/.test(w)), tag);
    }
  }
});

test('posed presets: crouch matches the GPU-validated research requests exactly (r6)', () => {
  // docs/motion-stage-research/probe/poses: build_crouch_requests.mjs r6 (seed 3; final recipe).
  const lines = fs.readFileSync(new URL('../../../docs/motion-stage-research/probe/poses/requests/r6.jsonl', import.meta.url), 'utf8').trim().split('\n').map((l) => JSON.parse(l));
  const cases = [
    ['constant_speed', { speed: 1.0, duration: 4 }, false, 'r6_constant_speed_final_s3'], ['strafe', { speed: 1.0 }, false, 'r6_strafe_final_s3'],
    ['forward_turn', { speed: 1.0 }, false, 'r6_forward_turn_final_s3'], ['stop', { speed: 1.0 }, false, 'r6_stop_final_s3'],
    ['start', { speed: 1.0 }, false, 'r6_start_final_s3'], ['turn_in_place', {}, false, 'r6_turn_in_place_final_s3'],
    ['constant_speed', { speed: 1.0, duration: 4 }, true, 'r6_constant_speed_anchor2_s3'], ['strafe', { speed: 1.0 }, true, 'r6_strafe_anchor2_s3'],
    ['forward_turn', { speed: 1.0 }, true, 'r6_forward_turn_anchor2_s3'],
  ];
  for (const [id, params, endAnchor, name] of cases) {
    const ref = lines.find((r) => r.output.endsWith(`/${name}`));
    assert.ok(ref, name);
    const b = buildPosedPreset(id, params, 'crouch', { endAnchor });
    assert.equal(b.prompt, ref.segments[0].prompt, name);
    assert.deepEqual(b.constraints, ref.constraints, name);
  }
});

test('posed presets: missing pose data or another skeleton degrade to the prompt with a note', () => {
  const library = noCrouchLibrary();
  const noCrouch = buildPosedPreset('start', defaultParams('start'), 'crouch', {}, { library });
  assert.deepEqual(noCrouch.constraints, buildPreset('start', defaultParams('start')).constraints);
  assert.ok(noCrouch.warnings.some((w) => /pose data is missing/.test(w)));
  assert.throws(() => presetPosesFor('crouch', { library }), /missing/);
  const g1 = buildPosedPreset('constant_speed', defaultParams('constant_speed'), 'rifle_low_ready', {}, { skeletonKey: 'g1skel34' });
  assert.ok(!g1.constraints.some((c) => c.type === 'left-hand'));
  assert.equal(g1.pose.layer, null);
  assert.ok(g1.warnings.some((w) => /needs the SOMA skeleton/.test(w)));
  const none = buildPosedPreset('stop', defaultParams('stop'), null);
  assert.equal(none.pose, null);
  assert.deepEqual(none.constraints, buildPreset('stop', defaultParams('stop')).constraints);
  assert.ok(buildPosedPreset('circle', defaultParams('circle'), 'rifle_low_ready', {}).warnings.some((w) => /walking speeds/.test(w)));
});

test('presetPosesFor + previewBuffers: the preview carries the rifle', () => {
  const poses = presetPosesFor(COLLAPSED);
  const b = buildPosedPreset('start', defaultParams('start'), COLLAPSED, {});
  const { local, root } = previewBuffers(b, sk, { poses });
  const at = (f) => {
    const pos = sk.fk(local.subarray(f * 120, f * 120 + 120), root.subarray(f * 3, f * 3 + 3)).positions;
    return P(pos, 19)[1];
  };
  near(at(0), 1.26, 0.06, 'grip hand at idle');
  near(at(b.frames - 1), 1.22, 0.03, 'grip hand while walking');
});

// ---------------------------------------------------------------------------
// request-time composition
// ---------------------------------------------------------------------------

test('composeRequest: arms on fullbody keyframes, pins along a dense 2D root, notes otherwise', () => {
  const plain = buildPreset('stop', defaultParams('stop'));
  const T = plain.frames;
  const segments = [{ prompt: 'A person walks forward, then stops.', frames: T }];
  const r = composeRequest({ segments, constraints: plain.constraints, frames: T }, COLLAPSED, {});
  assert.match(r.segments[0].prompt, /^A person holding a rifle in a low ready position/);
  validateModelJSON(r.constraints, sk);
  const fb = r.constraints.find((c) => c.type === 'fullbody');
  const natural = armsOf(COLLAPSED);
  ARM.forEach((j, k) => assert.deepEqual(fb.local_joints_rot[0][j], natural[k]));
  const lh = r.constraints.find((c) => c.type === 'left-hand');
  assert.ok(lh && lh.frame_indices.every((f) => f < T - 5), 'pins away from the end idle pins');
  assert.ok(r.pose.layer && r.notes.some((n) => /hand guide pins/.test(n)));
  // Frames that already hold a hand constraint keep theirs.
  const own = { ...lh, type: 'right-hand', frame_indices: [8], local_joints_rot: [lh.local_joints_rot[1]], root_positions: [lh.root_positions[1]], smooth_root_2d: [lh.smooth_root_2d[1]] };
  const r2 = composeRequest({ segments, constraints: [...plain.constraints, own], frames: T }, COLLAPSED, {});
  assert.ok(!r2.constraints.filter((c) => c !== own && c.type === 'left-hand').some((c) => c.frame_indices.includes(8)));
  // A dense 2D root without headings (Make Smooth Path): the facing is unknown, so no pins.
  const noHeadings = plain.constraints.map((c) => (c.type === 'root2d' ? { ...c, global_root_heading: undefined } : c));
  const r6 = composeRequest({ segments, constraints: noHeadings, frames: T }, COLLAPSED, {});
  assert.ok(!r6.constraints.some((c) => c.type === 'left-hand'));
  assert.ok(r6.notes.some((n) => /no headings/.test(n)));
  // A sparse 2D root: no pins, a note.
  const sparse = [{ type: 'root2d', frame_indices: [0, T - 1], smooth_root_2d: [[0, 0], [0, 3]] }];
  const r3 = composeRequest({ segments, constraints: sparse, frames: T }, COLLAPSED, {});
  assert.deepEqual(r3.constraints, sparse);
  assert.ok(r3.notes.some((n) => /need a 2D root on every frame/.test(n)));
  // Crouch with own constraints: prompt only.
  const r4 = composeRequest({ segments, constraints: sparse, frames: T }, 'crouch', {});
  assert.deepEqual(r4.constraints, sparse);
  assert.equal(r4.segments[0].prompt, 'A person walks forward, then stops.');
  assert.ok(r4.notes.some((n) => /comes with the path presets/.test(n)));
  // No pose: only the phrase is stripped.
  const r5 = composeRequest({ segments: r.segments, constraints: sparse, frames: T }, null, {});
  assert.equal(r5.segments[0].prompt, 'A person walks forward, then stops.');
  assert.equal(r5.pose, null);
});

// ---------------------------------------------------------------------------
// layer
// ---------------------------------------------------------------------------

test('composeRequest never puts a second body on a frame (Hips, feet, hands keep theirs) and respects the 20-frame budget', () => {
  const plain = buildPreset('constant_speed', defaultParams('constant_speed'));
  const T = plain.frames;
  const segments = [{ prompt: 'A person walks forward.', frames: T }];
  const lh = plain.constraints.find((c) => c.type === 'end-effector');
  const foot = { ...lh, type: 'left-foot', joint_names: undefined, frame_indices: [16] };
  delete foot.joint_names;
  const r = composeRequest({ segments, constraints: [...plain.constraints, foot], frames: T }, COLLAPSED, {});
  const pinned = r.constraints.filter((c) => c.type === 'left-hand').flatMap((c) => c.frame_indices);
  const others = [0, 16]; // the Hips pin and the foot pin
  assert.ok(pinned.length > 5);
  for (const f of pinned) assert.ok(others.every((g) => Math.abs(g - f) > 3), `hand pin at ${f}`);
  // Through a ConstraintStore: every end-effector frame keeps exactly one pose, and the foot pin is untouched.
  const store = new ConstraintStore(sk);
  store.setNumFrames(T);
  store.loadModelJSON(r.constraints);
  const back = store.toModelJSON();
  close(back.find((c) => c.type === 'left-foot').root_positions[0], foot.root_positions[0], 1e-6, 'foot pin root');
  // Many own hand pins: the guide pins are thinned so a hand track stays under WARN_FRAMES.
  const many = Array.from({ length: 12 }, (_, k) => 4 + k * 9);
  const own = { ...lh, type: 'right-hand', frame_indices: many, local_joints_rot: many.map(() => lh.local_joints_rot[0]), root_positions: many.map(() => lh.root_positions[0]), smooth_root_2d: many.map(() => lh.smooth_root_2d[0]) };
  delete own.joint_names;
  const r2 = composeRequest({ segments, constraints: [...plain.constraints, own], frames: T }, COLLAPSED, {});
  const added = r2.constraints.filter((c) => c.type === 'right-hand' && c !== own).flatMap((c) => c.frame_indices);
  assert.ok(many.length + added.length < WARN_FRAMES, `${many.length} + ${added.length}`);
});

test('applyPoseLayer rejects malformed target rotations', () => {
  const local = new Float32Array(30 * 4);
  for (let j = 0; j < 30; j++) local[j * 4 + 3] = 1;
  const bad = { mode: 'hold', joints: ['LeftArm'], rotations: [[0.3, -0.2, -0.1]] };
  assert.throws(() => applyPoseLayer(local, 1, sk.names, bad), /4 finite numbers/);
});

test('poseLayer: hold targets for rifle poses, none for crouch or hold 0', () => {
  for (const id of RIFLE) {
    const L = poseLayer(id, {});
    assert.deepEqual(L.joints, POSE_ARM_JOINTS);
    assert.equal(L.rotations.length, 12);
    for (const q of L.rotations) near(Math.hypot(...q), 1, 1e-5);
    assert.deepEqual([L.strength, L.sway, L.max_sway_deg], [1, 1, 15]);
  }
  assert.equal(poseLayer('crouch', {}), null);
  assert.equal(poseLayer('rifle_low_ready', { hold: 0 }), null);
  assert.deepEqual([poseLayer('rifle_low_ready', { hold: 50, sway: 25 }).strength, poseLayer('rifle_low_ready', { sway: 25 }).sway], [0.5, 0.25]);
});

test('applyPoseLayer matches the shared fixture (the Go twin checks the same numbers)', () => {
  const fx = JSON.parse(fs.readFileSync(new URL('./fixtures/pose_layer.json', import.meta.url), 'utf8'));
  for (const c of fx.cases) {
    const out = applyPoseLayer(Float32Array.from(fx.rotations), fx.frames, fx.names, c.layer);
    close(Array.from(out), c.expected, 1e-7);
  }
});

test('applyPoseLayer: the collapsed layer keeps the pinned carry; the low-ready layer moves the hands onto the forward carry', () => {
  const chestLocal = (pos, glob, j) => {
    // Position in the Chest (joint 3) frame: R_chest^T (p_j - p_chest).
    const q = [glob[12], glob[13], glob[14], glob[15]];
    const d = [pos[3 * j] - pos[9], pos[3 * j + 1] - pos[10], pos[3 * j + 2] - pos[11]];
    const inv = [-q[0], -q[1], -q[2], q[3]];
    const t = [2 * (inv[1] * d[2] - inv[2] * d[1]), 2 * (inv[2] * d[0] - inv[0] * d[2]), 2 * (inv[0] * d[1] - inv[1] * d[0])];
    return [d[0] + inv[3] * t[0] + (inv[1] * t[2] - inv[2] * t[1]), d[1] + inv[3] * t[1] + (inv[2] * t[0] - inv[0] * t[2]), d[2] + inv[3] * t[2] + (inv[0] * t[1] - inv[1] * t[0])];
  };
  const toLocal = (rot) => {
    const local = new Float32Array(30 * 4);
    rot.forEach((aa, j) => local.set(quatFromAxisAngle(aa), j * 4));
    return local;
  };
  const w = anchorOf(COLLAPSED);
  const local = toLocal(w.rot), root = [0, w.pelvis[1], 0];
  const before = sk.fk(local, root);
  // One frame: its mean is itself. The collapsed layer on its own pinned carry changes nothing.
  const same = applyPoseLayer(local, 1, sk.names, poseLayer(COLLAPSED, {}));
  close(Array.from(sk.fk(same, root).positions), Array.from(before.positions), 1e-5, 'collapsed on its anchor');
  // The low-ready layer: the hands land where the forward carry holds them relative to the chest
  // (chest-relative positions depend only on the arm rotations), nothing else moves.
  const moved = sk.fk(applyPoseLayer(local, 1, sk.names, poseLayer('rifle_low_ready', {})), root);
  const ref = poseBody('rifle_low_ready');
  const refFk = sk.fk(toLocal(ref.rot), [0, ref.pelvis[1], 0]);
  for (const j of [13, 15, 19, 21]) close(chestLocal(moved.positions, moved.globals, j), chestLocal(refFk.positions, refFk.globals, j), 1e-4, `hand ${j} in the chest frame`);
  for (const j of [0, 3, 6, 24, 28]) close(P(moved.positions, j), P(before.positions, j), 1e-6, `joint ${j} untouched`);
  assert.throws(() => applyPoseLayer(local, 1, ['Hips'], poseLayer(COLLAPSED, {})), /expected/);
});

test('both rifle poses build the same request; only the layer differs', () => {
  for (const id of ['constant_speed', 'start', 'turn_in_place']) {
    const a = buildPosedPreset(id, defaultParams(id), 'rifle_low_ready', {});
    const c = buildPosedPreset(id, defaultParams(id), COLLAPSED, {});
    assert.deepEqual(a.constraints, c.constraints, id);
    assert.equal(a.prompt, c.prompt, id);
    assert.notDeepEqual(a.pose.layer.rotations, c.pose.layer.rotations, id);
  }
});

// ---------------------------------------------------------------------------
// thumbnails
// ---------------------------------------------------------------------------

test('poseSVG draws a stick figure (with a rifle line for the carries)', () => {
  for (const id of RIFLE) {
    const svg = poseSVG(sk, id, { title: id });
    assert.match(svg, /^<svg[^>]+class="pose-svg"/);
    assert.match(svg, /class="pose-prop"/);
    assert.match(svg, /<title>/);
  }
  assert.ok(!/pose-prop/.test(poseSVG(sk, null)));
  assert.ok(!/pose-prop|no preview/.test(poseSVG(sk, 'crouch')), 'crouch draws a figure, no rifle');
  assert.ok(poseBody('rifle_low_ready').rot.length === 30);
});
