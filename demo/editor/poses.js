// Pose presets: selectable body poses (two rifle carries and a crouch) that a
// generation keeps from start to end. The carries follow the usual tactical
// meaning: low ready = rifle pointing forward, muzzle angled down at the ground
// ahead; collapsed low ready = rifle pulled in across the body, muzzle down. They compose with the parametric path
// presets (presets.js) and with hand-made constraints.
// DOM-free; runs unchanged in node (tests: node --test demo/editor/tests/).
//
// Why not only text (docs/motion-stage-research/09-experiment-log.md, "Pose
// presets", and the tools/mm_factory rifle_low_ready set):
// - "...holding a rifle in a low ready position..." gave Kimodo's own carry on
//   about 1 of 12 seed/wording pairs; other seeds raised the rifle or clasped
//   the hands behind the back, and "collapsed" wording made it worse (0 of 4).
// - Hand pins (left-hand + right-hand sets) at one of Kimodo's OWN carries
//   every ~8 frames along the path make the body move like it holds the rifle
//   on every seed. Pinning hands where the model does not want them (lower, or further
//   forward) makes it lean 10-26 degrees and bow the head to reach them, so
//   BOTH rifle poses pin Kimodo's own reading of "low ready": the cross-body
//   carry (anchor rifle_collapsed_walk), which keeps the body upright.
// - A pose layer, applied by the server after generation (demo/pose_layer.go;
//   applyPoseLayer here is its twin for previews and tests), holds the arms
//   under Chest on the pose's carry on every frame, so the carry is identical
//   across seeds and clips:
//   - collapsed low ready: the cross-body carry itself;
//   - low ready: the muzzle-forward carry. Kimodo never generates it (0 of 72
//     unpinned screen clips; probe/poses/report_low_ready.md), so it is
//     authored (tools/mm_factory/anchors/rifle_low_ready_forward) and reached
//     only through the layer.
//
// A pose has up to four parts:
// - prompt: a phrase spliced after the leading subject of every prompt
//   segment ("A person" -> "A person holding a rifle ..."), or appended.
//   Crouch has none: crouch wordings ("crouch", "low", "bent knees") gave a
//   60-80 degree hunch without pins and 51-69 degrees with Hips pins, "sneak"
//   barely crouched, and "squat" gave a full squat (research 09, "Pose
//   presets"; probe/poses/report_crouch.md). It only swaps "stands still" for
//   "stays still", and only in path-preset prompts.
// - bodies: the poses the path presets pin (buildPreset's `poses` option).
//   Rifle: the idle pins hold the preset idle body with the pinned (anchor)
//   carry's arms, and the frame-0 Hips pin holds the anchor. Crouch: the crouch
//   idle triplet in the idle pins.
// - crouch pins (with a path preset): a fullbody pair of the crouch walk pose
//   (crouch_walk @0, crouch_walk_next @1, placed on the path) on presets that
//   start moving, and the same pair at T-2, T-1 when the clip still moves
//   there (option endAnchor); the preset's frame-0 Hips pin is replaced by
//   dense Hips end-effector pins every hipsStride(T) frames (none within
//   CROUCH_GAP of a fullbody pin): a gait-neutral pelvis tilted 15 degrees
//   forward, on the path point, at 0.66 m standing rising to 0.70 m from
//   0.6 m/s. The fullbody crouch pins are what lowers the hunch (40-49
//   degrees with the walk pair, about 39 on idle and turn presets); the dense
//   Hips pins hold the pelvis height to about 2 cm.
// - pins (rifle): left-hand and right-hand sets of the pose's walk anchor,
//   placed on the path like the factory does (place(): yaw = path heading,
//   pelvis offset turned with it) every pinStride(T) frames. They are lifted
//   toward the standing pelvis height as the path slows (factory
//   standing_lift). Frames within PIN_GAP of a fullbody pin are skipped, so a
//   fullbody pin and a hand pin never ask for two bodies at once. With a Hips
//   pin at frame 0, the frame-0 hand pins use exactly its placement: the
//   ConstraintStore keeps one pose per frame for all end-effector tracks.
// - layer (rifle): {mode: 'hold', joints, rotations, strength, sway,
//   max_sway_deg}. Per arm joint, the clip-mean local rotation is replaced by
//   the authored one, and the model's per-frame deviation from its mean (the
//   arm sway, clamped to max_sway_deg and scaled by `sway`) is kept on top:
//   out = target * clamp(inv(mean) * q). The result is slerped from q by
//   `strength`. With sway 0 the arms are rigid in the chest frame. Measured
//   (factory rifle clips): hands move 0.4 cm (idle) to 2-4 cm (walk, start)
//   relative to the chest, so the sway is small but visible.
//
// Conventions are presets.js's (Y up, 30 fps, +theta = the character's left,
// wire heading [cos, sin], frame 0 faces +Z). Joint names and indices are
// SOMA-30's (src/skeleton.hpp); poses other than prompt-only need soma30.
//
// API
//   POSES                     frozen PoseDef[]
//   poseById(id)              PoseDef, or throws
//   defaultPoseParams(id), normalizePoseParams(id, p)
//   poseAvailable(id, {library})     false when the library lacks the pose's data
//   pinStride(T)              hand-pin spacing (frames) for a T-frame clip
//   posePrompt(id, prompt)    prompt with any pose phrase removed and `id`'s spliced in
//   stripPosePhrases(prompt)  prompt without any pose phrase
//   presetPosesFor(id)        {soma30: {idle: {prev, pose, next}, walk}} for buildPreset({poses})
//   hipsStride(T)             crouch Hips-pin spacing (frames) for a T-frame clip
//   guidePins(id, params, track, {fullbodyFrames, hipsAt0})
//                             {constraints: [left-hand, right-hand], frames}
//   buildPosedPreset(presetId, presetParams, poseId, poseParams, {skeletonKey})
//                             presets.js Built + {pose: PoseApplied | null}
//   composeRequest({segments, constraints, frames}, poseId, poseParams, {skeletonKey})
//                             request-time composition for constraints NOT made by a
//                             path preset: {segments, constraints, pose, notes}
//   poseLayer(id, params)     layer JSON for /api/generate `pose.layer`, or null
//   applyPoseLayer(local, T, names, layer)
//                             new Float32Array(T*J*4): the layer applied (server twin)
//   poseBody(id)              {rot, pelvis} body drawn for the pose (thumbnails, preview)
//   poseSVG(skeleton, id, opts)  front + side stick figure SVG string (no DOM)
//
// PoseDef: {id, label, group, description, kind: 'arms' | 'stance',
//   experimental, prompt: {phrase, mode: 'subject' | 'end'}, params: ParamDef[]}
// (ParamDef as in presets.js; groups 'main', 'pins', 'layer').
// PoseApplied: {id, label, params, layer, pins: [{frame, kind, pose}], notes}.

import { quatFromAxisAngle, quatInvert, quatMul, quatNormalize, quatSlerp } from './math.js';
import { PRESET_POSES } from './preset_poses.js';
import { POSE_LIBRARY } from './pose_library.js';
import { FPS, buildPreset, place, poseObject } from './presets.js';
import { WARN_FRAMES } from './constraints.js';

/** Hand pins every this many frames at least (factory hand_anchor every 8). */
const MIN_STRIDE = 8;
/** At most this many intervals, so a track stays under the 20-frame warning (incl. the last frame). */
const MAX_INTERVALS = 17;
/** No hand pin within this many frames of a fullbody pin. */
const PIN_GAP = 3;
/** Speed (m/s) at which the rifle walk anchor needs no lift (the factory set's walk speed). */
const ANCHOR_SPEED = 1.1;
/** Above this speed (m/s) a pose warns that it was validated at walking speeds only. */
const WALK_MAX = 2.0;
const DEG = Math.PI / 180;

const ARM_JOINTS = Object.freeze(POSE_LIBRARY.soma30.arms.joints.slice());
const SOMA_NAMES = Object.freeze([
  'Hips', 'Spine1', 'Spine2', 'Chest', 'Neck1', 'Neck2', 'Head', 'Jaw', 'LeftEye', 'RightEye',
  'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHandThumbEnd', 'LeftHandMiddleEnd',
  'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RightHandThumbEnd', 'RightHandMiddleEnd',
  'LeftLeg', 'LeftShin', 'LeftFoot', 'LeftToeBase', 'RightLeg', 'RightShin', 'RightFoot', 'RightToeBase',
]);
const ARM_INDICES = Object.freeze(ARM_JOINTS.map((n) => SOMA_NAMES.indexOf(n)));

// ---------------------------------------------------------------------------
// definitions
// ---------------------------------------------------------------------------

/**
 * The wording that goes with the pinned cross-body carry (research 09). Both
 * rifle poses use it: muzzle-forward words gave an aiming walk (rifle at chin
 * height, head down, 10-23 degrees of lean) instead (report_low_ready.md).
 */
const RIFLE_PHRASE = 'holding a rifle in a low ready position with the muzzle pointed down';

const GUIDE_PINS = {
  key: 'guidePins', label: 'Hand guide pins', type: 'bool', default: true, group: 'pins',
  hint: "Pins both hands to Kimodo's own low-ready carry every ~8 frames along the path, so the body moves like it holds the rifle. Needs a path with known facing: a path preset, or a 2D root on every frame with headings (pins go only where the facing is known).",
};
const HOLD = {
  key: 'hold', label: 'Hold the carry', type: 'number', default: 100, min: 0, max: 100, step: 5, unit: '%', group: 'layer',
  hint: 'After generation, how far the arms are moved onto the authored carry (100% = exactly).',
};
const SWAY = {
  key: 'sway', label: 'Arm sway', type: 'number', default: 100, min: 0, max: 100, step: 5, unit: '%', group: 'layer',
  hint: "How much of the model's own arm motion stays on top of the held carry (0% = arms rigid in the chest frame).",
};
/** The model's arm deviation from its clip mean is clamped to this (degrees per joint). */
const MAX_SWAY_DEG = 15;

const END_ANCHOR = {
  key: 'endAnchor', label: 'Anchor the end too', type: 'bool', default: true, group: 'pins',
  hint: 'On presets that start moving and are still moving at the end, also pins the crouch walk pose on the last two frames (no effect on Start / Start turn). It lowered the hunch by 2.5-7 degrees on 2 seeds per preset (Constant Speed, Strafe, Forward turn); turn it off if the feet slide at the end.',
};
/** Crouch pins (GPU recipe, research 09 "Crouch"): pelvis heights (m), tilt, Hips spacing, gap. */
const CROUCH = Object.freeze({ idleHeight: 0.66, walkHeight: 0.70, rampSpeed: 0.6, tiltDeg: 15, every: 10, gap: 2, movingSpeed: 0.05 });
/** Presets the crouch recipe was checked on with the GPU (3 seeds each). */
const CROUCH_TESTED = new Set(['constant_speed', 'strafe', 'forward_turn', 'start', 'stop', 'turn_in_place']);
/** The crouch was validated at 1.0 m/s (3 seeds); 0.8 and 1.2 m/s only on one seed of Constant Speed. */
const CROUCH_SPEED = 1.0;

const DEFS = [
  {
    id: 'rifle_low_ready', label: 'Rifle: low ready', group: 'Rifle', kind: 'arms', experimental: false,
    description: 'Rifle pointing forward with the muzzle angled down at the ground ahead: stock at the shoulder, grip hand at the chest, support hand forward on the handguard. Kimodo never generates this carry, so the body is generated with its own rifle carry and the server moves the arms onto the low ready on every frame.',
    arms: 'rifle_low_ready', anchor: 'rifle_collapsed_walk',
    prompt: { phrase: RIFLE_PHRASE, mode: 'subject' },
    params: [GUIDE_PINS, HOLD, SWAY],
  },
  {
    id: 'rifle_collapsed_low_ready', label: 'Rifle: collapsed low ready', group: 'Rifle', kind: 'arms', experimental: false,
    description: 'Rifle pulled in across the body, muzzle down and to the left: grip hand at the chest, support hand at the left hip. Hand pins guide the body; the arms are held on every frame.',
    arms: 'rifle_collapsed_low_ready', anchor: 'rifle_collapsed_walk',
    prompt: { phrase: RIFLE_PHRASE, mode: 'subject' },
    params: [GUIDE_PINS, HOLD, SWAY],
  },
  {
    id: 'crouch', label: 'Crouch', group: 'Stance', kind: 'stance', experimental: true,
    description: 'Knees bent about 100 degrees, pelvis at 0.66 m standing and 0.70 m walking, back leaning about 38-49 degrees. Made with pins, not words: crouch words made Kimodo hunch 60-80 degrees, so the prompt is left as it is. Works with the path presets.',
    // No phrase; standing words become stance-free (the GPU recipe's wording).
    prompt: { phrase: '', mode: 'none', swap: [['stands still', 'stays still']] },
    params: [END_ANCHOR],
  },
];

for (const d of DEFS) {
  Object.freeze(d.prompt);
  d.params.forEach(Object.freeze);
  Object.freeze(d.params);
  Object.freeze(d);
}
export const POSES = Object.freeze(DEFS);
const BY_ID = new Map(POSES.map((d) => [d.id, d]));

export function poseById(id) {
  const def = BY_ID.get(id);
  if (!def) throw new Error(`unknown pose ${JSON.stringify(id)} (expected one of ${[...BY_ID.keys()].join(', ')})`);
  return def;
}

export function defaultPoseParams(id) {
  const out = {};
  for (const pd of poseById(id).params) out[pd.key] = pd.default;
  return out;
}

/** Defaults filled, types coerced, numbers clamped and rounded to their step. Unknown keys dropped. Idempotent. */
export function normalizePoseParams(id, p = {}) {
  const src = p && typeof p === 'object' ? p : {};
  const out = {};
  for (const pd of poseById(id).params) out[pd.key] = normalizeValue(pd, src[pd.key]);
  return out;
}

function normalizeValue(pd, v) {
  if (pd.type === 'number') {
    const n = typeof v === 'number' ? v : typeof v === 'string' && v.trim() !== '' ? Number(v) : NaN;
    if (!Number.isFinite(n)) return pd.default;
    const c = Math.min(pd.max, Math.max(pd.min, n));
    return Math.min(pd.max, Math.max(pd.min, Math.round(c / pd.step) * pd.step));
  }
  if (pd.type === 'bool') {
    if (typeof v === 'boolean') return v;
    if (v === 'true' || v === 1) return true;
    if (v === 'false' || v === 0) return false;
    return pd.default;
  }
  if (pd.type === 'select') return pd.options.some((o) => o.value === v) ? v : pd.default;
  return pd.default;
}

/** False when the generated pose library lacks the data the pose needs (e.g. crouch bodies). */
export function poseAvailable(id, { library = POSE_LIBRARY } = {}) {
  const def = poseById(id);
  const L = library && library.soma30;
  if (!L) return false;
  if (def.kind === 'arms') return !!(L.arms && L.arms[def.arms] && L.bodies && L.bodies[def.anchor]);
  return !!(L.bodies && L.bodies.crouch_idle && L.bodies.crouch_walk && L.bodies.crouch_walk_next);
}

// ---------------------------------------------------------------------------
// prompt
// ---------------------------------------------------------------------------

const SUBJECT = /^(\s*)(an?|the)\s+(person|man|woman|character|soldier|figure)(?=\s|$)/i;
const PHRASES = [...new Set(POSES.map((d) => d.prompt.phrase).filter(Boolean))].sort((a, b) => b.length - a.length);
/** Phrases posePrompt may put right after the subject (mode 'subject'); every phrase may end a prompt. */
const SUBJECT_PHRASES = new Set(POSES.filter((d) => d.prompt.mode === 'subject').map((d) => d.prompt.phrase));
const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

/**
 * The prompt without the pose phrases posePrompt inserts, and only in the two
 * places it inserts them: right after the leading subject, and as the last
 * words before the final punctuation. The user's own wording elsewhere is
 * left alone, and a prompt holding no phrase is returned unchanged.
 */
export function stripPosePhrases(prompt) {
  let s = String(prompt ?? '');
  for (const ph of PHRASES) {
    const m = s.match(SUBJECT);
    if (m && SUBJECT_PHRASES.has(ph) && new RegExp(`^ ${escapeRe(ph)}(?=\\s|[.!?]|$)`, 'i').test(s.slice(m[0].length))) {
      s = s.slice(0, m[0].length) + s.slice(m[0].length + 1 + ph.length);
    }
    s = s.replace(new RegExp(`\\s${escapeRe(ph)}([.!?]*)\\s*$`, 'i'), '$1');
  }
  return s;
}

/**
 * `prompt` with any pose phrase removed and pose `id`'s phrase spliced in:
 * mode 'subject' inserts it after a leading "A person" (also a/an/the + man,
 * woman, character, soldier, figure), mode 'end' (or no subject found) appends
 * it before the final punctuation. id null/undefined only strips; a pose
 * without a phrase (crouch) strips and applies its word swaps
 * ("stands still" -> "stays still").
 */
export function posePrompt(id, prompt) {
  const base = stripPosePhrases(prompt).trim();
  const { phrase, mode, swap = [] } = id == null ? { phrase: '', mode: 'none' } : poseById(id).prompt;
  if (!phrase || mode === 'none') return swap.reduce((t, [from, to]) => t.split(from).join(to), base);
  if (!base) return `A person ${phrase}.`;
  const m = mode === 'subject' ? base.match(SUBJECT) : null;
  if (m) return `${base.slice(0, m[0].length)} ${phrase}${base.slice(m[0].length)}`;
  const end = base.match(/[.!?]+$/);
  const body = end ? base.slice(0, -end[0].length) : base;
  return `${body} ${phrase}${end ? end[0] : '.'}`;
}

// ---------------------------------------------------------------------------
// bodies
// ---------------------------------------------------------------------------

/** The twelve arm rows of the carry a pose PINS (its anchor body's own arms). */
const pinArms = (def, library) => ARM_INDICES.map((j) => library.soma30.bodies[def.anchor].rot[j]);

/** A {rot, pelvis} body with its twelve arm joints replaced by `arms` (axis-angle rows). */
function withArms(body, arms) {
  const rot = body.rot.map((r) => r.slice());
  ARM_INDICES.forEach((j, k) => { rot[j] = arms[k].slice(); });
  return { rot, pelvis: body.pelvis.slice() };
}

/**
 * The `poses` option of presets.js buildPreset/previewBuffers for a pose:
 * rifle = the preset idle triplet with the natural carry's arms (both rifle
 * poses pin the natural carry; the collapse is the layer's job) and the rifle
 * walk anchor as the Hips-pin walk pose; crouch = the crouch idle and walk
 * bodies; null id = the base poses unchanged.
 */
export function presetPosesFor(id, { base = PRESET_POSES, library = POSE_LIBRARY } = {}) {
  if (id == null) return base;
  const def = poseById(id);
  if (!poseAvailable(id, { library })) throw new Error(`pose ${id}: its data is missing from pose_library.js`);
  const L = library.soma30;
  const B = base.soma30;
  if (def.kind === 'arms') {
    const arms = pinArms(def, library);
    const idle = {};
    for (const k of ['prev', 'pose', 'next']) idle[k] = withArms(B.idle[k], arms);
    return { soma30: { idle, walk: L.bodies[def.anchor] } };
  }
  return { soma30: { idle: L.bodies.crouch_idle, walk: L.bodies.crouch_walk } };
}

/** The body drawn for a pose: the preset idle with the pose's own arms (rifle), or the crouch idle. */
export function poseBody(id, { base = PRESET_POSES, library = POSE_LIBRARY } = {}) {
  if (id == null) return base.soma30.idle.pose;
  const def = poseById(id);
  if (!poseAvailable(id, { library })) return null;
  if (def.kind === 'arms') return withArms(base.soma30.idle.pose, library.soma30.arms[def.arms]);
  return library.soma30.bodies.crouch_idle.pose;
}

// ---------------------------------------------------------------------------
// hand pins
// ---------------------------------------------------------------------------

/** Hand-pin spacing: every 8 frames, wider on long clips so a track keeps at most 19 frames. */
export function pinStride(T) {
  return Math.max(MIN_STRIDE, Math.ceil((T - 1) / MAX_INTERVALS));
}

/** Crouch Hips-pin spacing: every 10 frames, wider on long clips so the track keeps at most 18 frames. */
export function hipsStride(T) {
  return Math.max(CROUCH.every, Math.ceil((T - 1) / MAX_INTERVALS));
}

/**
 * Candidate hand-pin frames of a T-frame clip: 0, s, 2s, ... and the last
 * frame when it is at least s/2 past. A grid frame whose facing is unknown
 * (known(f) false) moves to the nearest known frame within s/2, so sparse
 * root2d headings (waypoints) still get pins; then excluded frames drop out.
 */
function pinFrames(T, exclude, known = () => true) {
  const s = pinStride(T);
  const grid = [];
  for (let f = 0; f < T; f += s) grid.push(f);
  if (T - 1 - grid[grid.length - 1] >= s / 2) grid.push(T - 1);
  const out = [];
  for (const g of grid) {
    let f = null;
    for (let d = 0; d <= Math.floor(s / 2) && f === null; d++) {
      if (g - d >= 0 && known(g - d)) f = g - d;
      else if (g + d < T && known(g + d)) f = g + d;
    }
    if (f !== null && !out.includes(f) && !exclude(f)) out.push(f);
  }
  return out.sort((a, b) => a - b);
}

/**
 * Hand guide pins of an arms pose along a path. track: {frames, xz: [x, z][T],
 * heading: rad[T] (null = unknown facing: no pin there), speed: m/s[T]}.
 * avoidFrames: frames holding other posed constraints (fullbody, Hips, feet,
 * hands); no hand pin within PIN_GAP of them, since a frame keeps one body
 * (ConstraintStore stores one pose per frame for every end-effector track).
 * hipsAt0: the frame-0 Hips pin of a posed path preset holds the walk anchor
 * with its pelvis exactly on the path point, so the frame-0 hand pins use that
 * placement (frame 0 must then not be in avoidFrames). Returns
 * {constraints: [left-hand, right-hand] (empty when no frame is left), frames}.
 */
export function guidePins(id, params, track, { avoidFrames = [], hipsAt0 = false, library = POSE_LIBRARY, base = PRESET_POSES } = {}) {
  const def = poseById(id);
  const p = normalizePoseParams(id, params);
  if (def.kind !== 'arms' || !p.guidePins) return { constraints: [], frames: [] };
  const T = track.frames;
  const avoid = [...new Set(avoidFrames)];
  const frames = pinFrames(T, (f) => avoid.some((g) => Math.abs(g - f) <= PIN_GAP), (f) => track.heading[f] != null);
  if (!frames.length) return { constraints: [], frames: [] };
  const anchor = library.soma30.bodies[def.anchor];
  const standingLift = base.soma30.idle.pose.pelvis[1] - anchor.pelvis[1];
  const placed = frames.map((f) => {
    if (f === 0 && hipsAt0) return { rot: anchor.rot, root: [track.xz[0][0], anchor.pelvis[1], track.xz[0][1]], smooth: track.xz[0].slice() };
    const pl = place(anchor, track.xz[f], track.heading[f]);
    const still = Math.min(1, Math.max(0, 1 - track.speed[f] / ANCHOR_SPEED));
    pl.root[1] += standingLift * still;
    return pl;
  });
  return {
    constraints: [poseObject('left-hand', frames, placed), poseObject('right-hand', frames, placed)],
    frames,
  };
}

// ---------------------------------------------------------------------------
// composition
// ---------------------------------------------------------------------------

/**
 * A path preset built with a pose (null poseId = plain buildPreset): the
 * pose's bodies in the idle and Hips pins, hand guide pins (arms poses), the
 * pose phrase in the prompt and `pose` (PoseApplied) for the request's layer.
 * Constraint order: fullbody, Hips end-effector, left-hand, right-hand, root2d
 * (ConstraintStore#toModelJSON order for frame-0 pins: Hips before the hands).
 */
export function buildPosedPreset(presetId, presetParams, poseId, poseParams, { skeletonKey = 'soma30', library = POSE_LIBRARY, base = PRESET_POSES } = {}) {
  if (poseId == null) {
    const plain = buildPreset(presetId, presetParams, { skeletonKey, poses: base });
    return { ...plain, basePrompt: plain.prompt, pose: null };
  }
  const def = poseById(poseId);
  const params = normalizePoseParams(poseId, poseParams);
  const soma = skeletonKey === 'soma30';
  const usable = soma && poseAvailable(poseId, { library });
  const stance = usable && def.kind === 'stance';
  // The user's preset params pass through unchanged (a crouch build drops the
  // frame-0 ground pin below; the preset keeps its groundPin setting).
  const built = buildPreset(presetId, presetParams, { skeletonKey, poses: usable ? presetPosesFor(poseId, { base, library }) : base });
  const notes = [];
  // Crouch: dense Hips pins replace the frame-0 ground pin (research 09, "Pose presets").
  const pins = built.pins.filter((pin) => !(stance && pin.kind === 'hips'))
    .map((pin) => ({ ...pin, pose: pin.kind === 'fullbody' ? `${poseId}.${pin.pose}` : `${poseId}.walk` }));
  let constraints = built.constraints;
  if (!soma) notes.push(`${def.label} needs the SOMA skeleton; only its prompt is used.`);
  else if (!usable) notes.push(`${def.label}: its pose data is missing (scripts/extract_pose_library.py); only its prompt is used.`);
  if (usable && def.kind === 'arms') {
    const avoidFrames = built.pins.filter((pn) => pn.kind === 'fullbody').map((pn) => pn.frame);
    const hipsAt0 = built.pins.some((pn) => pn.kind === 'hips' && pn.frame === 0);
    const g = guidePins(poseId, params, built.sim, { avoidFrames, hipsAt0, library, base });
    if (g.frames.length) {
      const root = constraints.filter((c) => c.type === 'root2d');
      constraints = [...constraints.filter((c) => c.type !== 'root2d'), ...g.constraints, ...root];
      for (const f of g.frames) pins.push({ frame: f, kind: 'hands', pose: def.anchor });
      pins.sort((a, b) => a.frame - b.frame || kindOrder(a.kind) - kindOrder(b.kind));
    }
  }
  if (stance) {
    const c = crouchPins(built, params, library);
    constraints = c.constraints;
    pins.push(...c.pins);
    pins.sort((a, b) => a.frame - b.frame || kindOrder(a.kind) - kindOrder(b.kind));
  }
  const warnings = built.warnings.slice();
  if (stance) warnings.push(...crouchWarnings(presetId, built));
  else if (built.summary.max_speed >= WALK_MAX) warnings.push(`${def.label} was validated at walking speeds; jogs and runs with it are untested.`);
  warnings.push(...notes);
  return {
    ...built,
    basePrompt: built.prompt, // the preset's own prompt, before the pose's words
    prompt: posePrompt(poseId, built.prompt),
    constraints,
    pins,
    warnings,
    pose: { id: poseId, label: def.label, params, layer: usable ? poseLayer(poseId, params) : null, notes },
  };
}

const kindOrder = (k) => ({ fullbody: 0, hips: 1, hands: 2 })[k] ?? 3;

/**
 * The crouch pins on a preset built with the crouch bodies (idle pins already
 * crouch idle, no Hips pin): the crouch walk pair at the start (and the end,
 * endAnchor) of a moving clip, merged into the one fullbody object in frame
 * order, then dense Hips pins. Returns {constraints (fullbody, Hips
 * end-effector, root2d), pins}.
 */
function crouchPins(built, params, library) {
  const B = library.soma30.bodies;
  const sim = built.sim;
  const T = sim.frames;
  const pins = [];
  const fb = built.constraints.find((c) => c.type === 'fullbody');
  const entries = fb ? fb.frame_indices.map((f, k) => ({ f, pl: { rot: fb.local_joints_rot[k], root: fb.root_positions[k], smooth: fb.smooth_root_2d[k] } })) : [];
  if (sim.speed[0] > CROUCH.movingSpeed) {
    const pairs = [[0, B.crouch_walk, 'crouch.walk'], [1, B.crouch_walk_next, 'crouch.walk_next']];
    if (params.endAnchor && sim.speed[T - 1] > CROUCH.movingSpeed) pairs.push([T - 2, B.crouch_walk, 'crouch.walk'], [T - 1, B.crouch_walk_next, 'crouch.walk_next']);
    for (const [f, pose, name] of pairs) {
      entries.push({ f, pl: place(pose, sim.xz[f], sim.heading[f]) });
      pins.push({ frame: f, kind: 'fullbody', pose: name });
    }
    entries.sort((a, b) => a.f - b.f);
  }
  const constraints = [];
  if (entries.length) constraints.push(poseObject('fullbody', entries.map((e) => e.f), entries.map((e) => e.pl)));
  const fullFrames = entries.map((e) => e.f);
  const frames = [];
  const s = hipsStride(T);
  for (let f = 0; f < T; f += s) if (!fullFrames.some((g) => Math.abs(g - f) <= CROUCH.gap)) frames.push(f);
  if (frames.length) {
    const tilt = [CROUCH.tiltDeg * DEG, 0, 0];
    const placed = frames.map((f) => {
      const h = CROUCH.idleHeight + (CROUCH.walkHeight - CROUCH.idleHeight) * Math.min(1, Math.max(0, sim.speed[f] / CROUCH.rampSpeed));
      // A gait-neutral pelvis: tilted forward, on the path point (the other joints are unused by a Hips pin).
      const rot = B.crouch_walk.rot.map((r) => r.slice());
      rot[0] = tilt;
      const pl = place({ rot, pelvis: [0, h, 0] }, sim.xz[f], sim.heading[f]);
      if (f === 0) pl.root = [sim.xz[0][0], h, sim.xz[0][1]];
      return pl;
    });
    constraints.push(poseObject('end-effector', frames, placed, { joint_names: ['Hips'] }));
    for (const f of frames) pins.push({ frame: f, kind: 'hips', pose: 'crouch.pelvis' });
  }
  constraints.push(...built.constraints.filter((c) => c.type === 'root2d'));
  return { constraints, pins };
}

/** The crouch's GPU caveats for a preset (probe/poses/report_crouch.md). */
function crouchWarnings(presetId, built) {
  const out = [];
  const sim = built.sim;
  const moving = sim.speed.some((v) => v > CROUCH.movingSpeed);
  if (!CROUCH_TESTED.has(presetId)) out.push('Crouch was checked on Constant Speed, Strafe, Forward turn, Start, Stop and Turn in place; this preset is untested with it.');
  if (moving && Math.abs(built.summary.max_speed - CROUCH_SPEED) > 0.05) {
    out.push(`Crouch was validated at ${CROUCH_SPEED} m/s (3 seeds per preset); 0.8 and 1.2 m/s were checked on one seed of Constant Speed only, and other speeds are untested.`);
  }
  if (presetId === 'start' || presetId === 'start_turn') {
    out.push('Crouch starts lean about 41-45 degrees, and on 2 of 3 seeds the first steps sank 5-7 cm into the floor for about a second.');
  } else if (sim.speed[0] > CROUCH.movingSpeed && presetId !== 'stop' && presetId !== 'stop_turn') {
    out.push('Moving crouch clips read as a creeping crouch: the back leans 40-49 degrees (40-43 with the end anchor), against about 39 on the idle, stop and turn presets.');
  }
  return out;
}

/**
 * Request-time composition for constraints that do NOT come from a posed path
 * preset (hand-made keyframes, a plain prompt, a gallery request), applied to
 * the request only (the store is not changed). Arms poses: every SOMA-30
 * fullbody keyframe gets the natural carry's arms, and hand guide pins follow
 * the 2D root when it covers every frame, on frames whose facing is known (a
 * root2d heading, or frame 0, which always faces +Z). No pin lands within
 * PIN_GAP of a frame that already holds a fullbody or end-effector constraint,
 * so the user's pins (or a posed preset's own guide pins) keep their frames,
 * and the pins are thinned so a hand track stays under the 20-frame warning.
 * Stance poses (crouch) add nothing here (a note says so): their pins come
 * with the path presets. Returns {segments (prompts with the phrase spliced;
 * the editor sends its own segments), constraints, pose: PoseApplied | null, notes}.
 */
export function composeRequest({ segments, constraints = [], frames }, poseId, poseParams, { skeletonKey = 'soma30', library = POSE_LIBRARY, base = PRESET_POSES } = {}) {
  const segs = segments.map((s) => ({ ...s, prompt: posePrompt(poseId, s.prompt) }));
  if (poseId == null) return { segments: segs, constraints, pose: null, notes: [] };
  const def = poseById(poseId);
  const params = normalizePoseParams(poseId, poseParams);
  const notes = [];
  let out = constraints;
  const usable = skeletonKey === 'soma30' && poseAvailable(poseId, { library });
  if (skeletonKey !== 'soma30') notes.push(`${def.label} needs the SOMA skeleton; only its prompt is used.`);
  else if (!usable) notes.push(`${def.label}: its pose data is missing; only its prompt is used.`);
  else if (def.kind === 'stance') notes.push(`${def.label} comes with the path presets (its idle, walk and pelvis pins); with your own constraints it adds nothing.`);
  else {
    const arms = pinArms(def, library);
    out = constraints.map((c) => (c.type === 'fullbody'
      ? { ...c, local_joints_rot: c.local_joints_rot.map((rows) => (rows.length !== SOMA_NAMES.length ? rows : rows.map((r, j) => {
        const k = ARM_INDICES.indexOf(j);
        return k >= 0 ? arms[k].slice() : r;
      }))) }
      : c));
    if (params.guidePins) {
      const track = pathTrack(constraints, frames);
      if (!track) {
        notes.push('No hand guide pins: they need a 2D root on every frame with headings (a path preset, or Make Smooth Path with waypoint headings).');
      } else if (track.heading.filter((h) => h != null).length <= 1) {
        notes.push('No hand guide pins: the 2D root has no headings, so the facing on its frames is unknown (set headings on its waypoints, or use a path preset).');
      } else {
        const posed = constraints.filter((c) => c.type !== 'root2d');
        const avoidFrames = posed.flatMap((c) => c.frame_indices);
        const handName = { 'left-hand': 'LeftHand', 'right-hand': 'RightHand' };
        const own = (side) => new Set(posed.filter((c) => c.type === side || (c.type === 'end-effector' && (c.joint_names || []).includes(handName[side])))
          .flatMap((c) => c.frame_indices).filter((f) => f < frames)).size;
        const userHands = Math.max(own('left-hand'), own('right-hand'));
        const g = guidePins(poseId, params, track, { avoidFrames, library, base });
        let keep = g.frames.map((_, k) => k);
        let thinned = false;
        while (keep.length && userHands + keep.length >= WARN_FRAMES) {
          keep = keep.length === 1 ? [] : keep.filter((_, i) => i % 2 === 0);
          thinned = true;
        }
        if (keep.length) {
          const pick = (c) => ({
            ...c,
            frame_indices: keep.map((k) => c.frame_indices[k]),
            local_joints_rot: keep.map((k) => c.local_joints_rot[k]),
            root_positions: keep.map((k) => c.root_positions[k]),
            smooth_root_2d: keep.map((k) => c.smooth_root_2d[k]),
          });
          out = [...out, ...g.constraints.map(pick)];
          notes.push(`Added ${keep.length} hand guide pins per hand along the 2D root${thinned ? ' (thinned to stay under 20 hand frames)' : ''}.`);
        } else if (userHands > 0) {
          notes.push(`No hand guide pins added: your own hand constraints (${userHands} frames per hand) are sent as they are.`);
        } else if (!g.frames.length) {
          notes.push('No hand guide pins: every candidate frame is next to one of your own constraints.');
        } else {
          notes.push('No hand guide pins: your own hand constraints already fill the 20-frame budget.');
        }
      }
    }
  }
  return { segments: segs, constraints: out, pose: { id: poseId, label: def.label, params, layer: usable ? poseLayer(poseId, params) : null, notes }, notes };
}

/**
 * The 2D-root track (xz, heading, speed per frame) when root2d constraints
 * cover every frame 0..T-1, else null. heading[f] is the root2d heading where
 * one is set and null elsewhere; frame 0 is always 0 (the engine starts every
 * clip facing +Z and ignores a frame-0 heading).
 */
function pathTrack(constraints, T) {
  const xz = new Array(T), heading = new Array(T).fill(null);
  for (const c of constraints) {
    if (c.type !== 'root2d') continue;
    c.frame_indices.forEach((f, k) => {
      if (f >= T) return;
      xz[f] = [c.smooth_root_2d[k][0], c.smooth_root_2d[k][1]];
      const h = c.global_root_heading && c.global_root_heading[k];
      if (h) heading[f] = Math.atan2(h[1], h[0]);
    });
  }
  for (let f = 0; f < T; f++) if (!xz[f]) return null;
  heading[0] = 0;
  const speed = new Array(T);
  for (let f = 0; f < T; f++) {
    const a = xz[Math.max(0, f - 1)], b = xz[Math.min(T - 1, f + 1)];
    const n = Math.min(T - 1, f + 1) - Math.max(0, f - 1);
    speed[f] = n ? (Math.hypot(b[0] - a[0], b[1] - a[1]) * FPS) / n : 0;
  }
  return { frames: T, xz, heading, speed };
}

// ---------------------------------------------------------------------------
// layer
// ---------------------------------------------------------------------------

/** Layer JSON of a pose for the generate request (`pose.layer`), or null (no layer). */
export function poseLayer(id, params, { library = POSE_LIBRARY } = {}) {
  const def = poseById(id);
  if (def.kind !== 'arms') return null;
  const p = normalizePoseParams(id, params);
  if (p.hold <= 0) return null;
  const arms = library.soma30.arms[def.arms];
  const r6 = (v) => (Math.round(v * 1e6) / 1e6) || 0;
  return {
    mode: 'hold',
    joints: ARM_JOINTS.slice(),
    rotations: arms.map((aa) => quatFromAxisAngle(aa).map(r6)),
    strength: p.hold / 100,
    sway: p.sway / 100,
    max_sway_deg: MAX_SWAY_DEG,
  };
}

/** Scales a unit quaternion's rotation angle: min(angle, maxRad) * scale (canonical w >= 0). */
function clampScale(q, maxRad, scale, out) {
  let [x, y, z, w] = q;
  if (w < 0) { x = -x; y = -y; z = -z; w = -w; }
  const s = Math.hypot(x, y, z);
  if (s < 1e-12) { out[0] = 0; out[1] = 0; out[2] = 0; out[3] = 1; return out; }
  const a = Math.min(2 * Math.atan2(s, w), maxRad) * scale;
  const k = Math.sin(a / 2) / s;
  out[0] = x * k; out[1] = y * k; out[2] = z * k; out[3] = Math.cos(a / 2);
  return out;
}

/**
 * The pose layer applied to a [T][J] xyzw rotation buffer (twin of
 * demo/pose_layer.go applyPoseLayer; same maths, float64). names: joint names
 * of the skeleton. Returns a new Float32Array; joints the skeleton lacks are
 * an error.
 */
export function applyPoseLayer(local, T, names, layer) {
  const J = names.length;
  if (local.length !== T * J * 4) throw new Error(`applyPoseLayer: expected ${T * J * 4} values, got ${local.length}`);
  const out = Float32Array.from(local);
  if (!layer || T === 0) return out;
  if (layer.mode !== 'hold') throw new Error(`applyPoseLayer: unknown mode ${JSON.stringify(layer.mode)}`);
  const strength = Math.min(1, Math.max(0, layer.strength ?? 1));
  const sway = Math.min(1, Math.max(0, layer.sway ?? 1));
  const maxRad = Math.max(0, layer.max_sway_deg ?? MAX_SWAY_DEG) * DEG;
  const q = [0, 0, 0, 1], inv = [0, 0, 0, 1], r = [0, 0, 0, 1], o = [0, 0, 0, 1], s = [0, 0, 0, 1];
  layer.joints.forEach((name, k) => {
    const j = names.indexOf(name);
    if (j < 0) throw new Error(`applyPoseLayer: no joint ${JSON.stringify(name)}`);
    const tq = layer.rotations[k];
    if (!Array.isArray(tq) || tq.length !== 4 || !tq.every(Number.isFinite) || Math.hypot(...tq) < 1e-6) {
      throw new Error(`applyPoseLayer: rotation ${k} must be 4 finite numbers (xyzw)`);
    }
    const target = quatNormalize(tq);
    // Clip mean, each frame sign-aligned to frame 0.
    const m = [0, 0, 0, 0];
    const q0 = [local[j * 4], local[j * 4 + 1], local[j * 4 + 2], local[j * 4 + 3]];
    for (let t = 0; t < T; t++) {
      const i = (t * J + j) * 4;
      const d = local[i] * q0[0] + local[i + 1] * q0[1] + local[i + 2] * q0[2] + local[i + 3] * q0[3];
      const sg = d < 0 ? -1 : 1;
      for (let c = 0; c < 4; c++) m[c] += sg * local[i + c];
    }
    const mean = Math.hypot(...m) > 1e-9 ? quatNormalize(m) : quatNormalize(q0);
    quatInvert(mean, inv);
    for (let t = 0; t < T; t++) {
      const i = (t * J + j) * 4;
      q[0] = local[i]; q[1] = local[i + 1]; q[2] = local[i + 2]; q[3] = local[i + 3];
      quatNormalize(q, q);
      quatMul(inv, q, r);
      clampScale(r, maxRad, sway, r);
      quatMul(target, r, o);
      quatSlerp(q, o, strength, s);
      quatNormalize(s, s);
      out[i] = s[0]; out[i + 1] = s[1]; out[i + 2] = s[2]; out[i + 3] = s[3];
    }
  });
  return out;
}

// ---------------------------------------------------------------------------
// thumbnail
// ---------------------------------------------------------------------------

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const fmt = (v) => (Number.isFinite(v) ? String(Math.round(v * 10) / 10) : '0');

/**
 * Front and side stick figures of a pose's body as an SVG string (no DOM).
 * skeleton: a skeleton.js Skeleton (SOMA-30). Front view: the character faces
 * the viewer (its left on screen right); side view: seen from its left, facing
 * screen right. Arm bones get class pose-arm; rifle poses add a rifle line
 * (pose-prop) along grip -> support hand. opts: {width = 160, height = 120,
 * className, title}.
 */
export function poseSVG(skeleton, id, opts = {}) {
  const W = opts.width ?? 160, H = opts.height ?? 120;
  const body = poseBody(id);
  const cls = ['pose-svg', opts.className].filter(Boolean).join(' ');
  const label = opts.title ? ` role="img" aria-label="${esc(opts.title)}"` : ' aria-hidden="true"';
  const parts = [`<svg xmlns="http://www.w3.org/2000/svg" class="${esc(cls)}" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}"${label}>`];
  if (opts.title) parts.push(`<title>${esc(opts.title)}</title>`);
  if (!body || skeleton.J !== 30) {
    parts.push(`<text x="${W / 2}" y="${H / 2}" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.6">no preview</text></svg>`);
    return parts.join('');
  }
  const local = new Float32Array(30 * 4);
  body.rot.forEach((aa, j) => local.set(quatFromAxisAngle(aa), j * 4));
  const { positions: P } = skeleton.fk(local, [0, body.pelvis[1], 0]);
  const pad = 8, top = 1.85;
  const scale = (H - 2 * pad) / top;
  const ground = H - pad;
  const panels = [
    { cx: W * 0.28, x: (j) => P[3 * j] },          // front: +X (its left) to screen right
    { cx: W * 0.72, x: (j) => P[3 * j + 2] },      // side: +Z (forward) to screen right
  ];
  const isArm = new Set(ARM_INDICES);
  for (const pn of panels) {
    const sx = (j) => pn.cx + pn.x(j) * scale;
    const sy = (j) => ground - P[3 * j + 1] * scale;
    let bones = '', arms = '';
    for (let j = 1; j < 30; j++) {
      const p = skeleton.parents[j];
      if (j >= 7 && j <= 9) continue; // jaw and eyes
      const seg = `M${fmt(sx(p))} ${fmt(sy(p))}L${fmt(sx(j))} ${fmt(sy(j))}`;
      if (isArm.has(j)) arms += seg; else bones += seg;
    }
    parts.push(`<path class="pose-bone" d="${bones}" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" stroke-opacity="0.75"/>`);
    parts.push(`<path class="pose-arm" d="${arms}" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>`);
    parts.push(`<circle class="pose-head" cx="${fmt(sx(6))}" cy="${fmt(sy(6) - 0.06 * scale)}" r="${fmt(0.09 * scale)}" fill="none" stroke="currentColor" stroke-width="1.6" stroke-opacity="0.75"/>`);
    if (id != null && poseById(id).kind === 'arms') {
      const g = 19, s = 13; // grip (right hand) -> support (left hand)
      const d = [P[3 * s] - P[3 * g], P[3 * s + 1] - P[3 * g + 1], P[3 * s + 2] - P[3 * g + 2]];
      const n = Math.hypot(...d) || 1;
      const at = (t) => [P[3 * g] + (d[0] / n) * t, P[3 * g + 1] + (d[1] / n) * t, P[3 * g + 2] + (d[2] / n) * t];
      const a = at(-0.25), b = at(n + 0.35);
      const px = (v) => pn.cx + (pn === panels[0] ? v[0] : v[2]) * scale;
      const py = (v) => ground - v[1] * scale;
      parts.push(`<path class="pose-prop" d="M${fmt(px(a))} ${fmt(py(a))}L${fmt(px(b))} ${fmt(py(b))}" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round"/>`);
    }
  }
  parts.push(`<path class="pose-ground" d="M${pad} ${fmt(ground)}H${W - pad}" stroke="currentColor" stroke-opacity="0.2" stroke-width="1"/>`);
  parts.push('</svg>');
  return parts.join('');
}

export const POSE_ARM_JOINTS = ARM_JOINTS;
