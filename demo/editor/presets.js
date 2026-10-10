// Parametric path presets (Motorica-style "Parametric presets"): a preset id +
// a few parameters become a Kimodo request built from constraints - a dense
// root2d path with a heading on every frame, pose pins, a matching prompt and
// the generation settings that were measured to follow such paths.
// DOM-free; runs unchanged in node (tests: node --test demo/editor/tests/).
//
// Background: docs/motion-stage-research/05-root-paths-presets-and-motion-factory.md
// (A: spring maths, B: presets and requests, C: measurements) and
// 09-experiment-log.md. Pin recipes are ports of tools/mm_factory (clips.py
// place / fullbody / hips_pin, factory.py start_stop / turn). Springs: spring.js
// (Daniel Holden's MIT spring maths). Our own defaults; no Motorica code or
// constants.
//
// Conventions (do not change; checked against generated outputs)
// - Y up, 30 fps, metres. Frame 0 at (x, z) = (0, 0) facing +Z (Kimodo ignores
//   a non-zero frame-0 heading, so every preset starts facing +Z).
// - Heading theta (radians): +theta turns to the character's LEFT, and its left
//   is +X. Forward = (sin theta, 0, cos theta); wire format
//   global_root_heading = [cos theta, sin theta]; path tangent (vx, vz) ->
//   theta = atan2(vx, vz). Headings are UNWRAPPED (a 180 degree turn ends at
//   +-pi, a 360 degree turn at +-2 pi).
// - Angle params are magnitudes in degrees with a separate side param
//   ('left' | 'right'; left = +theta). Durations are seconds in the UI and become
//   frames with Math.round(s * 30); one clip is 60..300 frames (Go server limit
//   for one prompt segment).
// - Rotating a point about +Y by theta: x' = x cos + z sin, z' = -x sin + z cos.
//   A pose is turned by left-multiplying the Hips (joint 0) local rotation by
//   q_yaw(theta) = (0, sin theta/2, 0, cos theta/2) (xyzw).
// - Top-down drawings (presetSVG): forward (+Z) is screen-up and the
//   character's left (+X) is screen-LEFT, as seen from above:
//   sx = cx - X * scale, sy = cy - Z * scale; a facing theta points along
//   screen (-sin theta, -cos theta).
//
// Trail simulator (research 05 A.11): per frame the state is EMITTED first
// (frame 0 = initial state), then advanced `substeps` steps of dt = 1/30/substeps.
// State: position, velocity and acceleration per axis (springCharacterUpdate,
// velocity goal = speedGoal * (sin d, cos d) with d the travel-direction goal),
// facing theta and its rate (springDamperExact toward the facing goal), or an
// explicit cosine-eased facing. Goal switches happen on substep indices
// (Math.round(seconds * 30 * substeps)), so they do not drift with float sums.
// Half-life routing: the velocity spring uses turnHalflife when the travel
// direction changes (Arc, Forward turn, Strafe turn) and speedHalflife when the
// speed goal changes (Acceleration, Start, Start turn, Stop, Stop turn, Spin);
// the facing spring uses facingHalflife.
//
// Pins (SOMA-30 only; anchor poses from preset_poses.js)
// - Hips pin (moving-start presets, pin option groundPin): one generic
//   end-effector ['Hips'] at frame 0 with the walk pose, pelvis EXACTLY on the
//   path point (0, 0) at the walk pose's pelvis height. It grounds the start of
//   turning paths (the measured "floating start", research 05 C.4).
// - Idle pins: ONE fullbody object holding every pinned frame in ascending
//   order, poses placed with place() (port of clips.py place: Hips rotation
//   left-multiplied by q_yaw, pelvis offset from the smooth root rotated too).
// - Other skeletons: no pins (path only) and a warning.
//
// API
//   FPS, MIN_FRAMES, MAX_FRAMES, DEFAULT_DYNAMICS
//   PRESETS                          frozen PresetDef[] in Motorica's order
//   presetById(id)                   PresetDef, or throws
//   defaultParams(id)                fresh {key: default}
//   normalizeParams(id, p)           defaults filled, clamped to the ParamDef ranges,
//                                    rounded to step, sub-durations fitted into the clip
//   simulatePreset(id, p)            Sim {frames, xz: [x, z][T], heading: rad[T] (unwrapped), speed: m/s[T]}
//   buildPreset(id, p, {skeletonKey = 'soma30', poses = PRESET_POSES})
//                                    Built {id, label, params, frames, prompt, sim, constraints,
//                                    pins: [{frame, kind: 'hips' | 'fullbody', pose}],
//                                    settings: {post_processing, root_margin},
//                                    summary: {duration_s, distance_m, end: [x, z], net_turn_deg, max_speed},
//                                    warnings: string[]}
//   presetSVG(sim, opts)             top-down SVG markup string (no DOM)
//   gaitVerb(speed)                  'walks' (< 2 m/s) | 'jogs' (< 4) | 'runs'
//   previewBuffers(built, skeleton, {poses})
//                                    {local: Float32Array(T*J*4) xyzw, root: Float32Array(T*3)},
//                                    the anchor pose sliding along the path (Motion layout)
//
// PresetDef: {id, label, description, experimental, params: ParamDef[]}.
// ParamDef: {key, label, type: 'number' | 'side' | 'select' | 'bool', default,
//   min?, max?, step?, unit?, hint?, options?: [{value, label}],
//   group: 'main' | 'dynamics' | 'pins'}. A preset lists only the dynamics
//   half-lives its recipe uses (Constant Speed, Strafe and Turn in place use none).
//
// constraints order: fullbody object (if any), Hips end-effector (if any), then
// one root2d object over frames 0..T-1 with a heading on every frame - the order
// ConstraintStore#toModelJSON emits, so a store round trip is stable. Numbers
// are rounded to 6 decimals.

import { springCharacterUpdate, springDamperExact } from './spring.js';
import { quatFromAxisAngle, quatFromYaw, quatMul, quatToAxisAngle, wrapAngle } from './math.js';
import { PRESET_POSES } from './preset_poses.js';

export const FPS = 30;
export const MIN_FRAMES = 60;
export const MAX_FRAMES = 300;
export const DEFAULT_DYNAMICS = Object.freeze({ turnHalflife: 0.12, facingHalflife: 0.12, speedHalflife: 0.3, substeps: 10 });

/** Generation settings every preset sends (research 05 B.3 step 9). */
const SETTINGS = Object.freeze({ post_processing: true, root_margin: 0.06 });
/** Time an event leaves at least before the end of the clip (s). */
const MIN_TAIL_S = 0.5;
/** Facing lead/lag of the pre/post turn styles (s), research 05 A.7. */
const TURN_LEAD_S = 0.25;
const TURN_STYLE = Object.freeze({ pre: 0, default: 0.5, post: 1 });
/** In-place turns: eased from frame TURN_START to frame T - TURN_END_OFFSET (factory.py turn()). */
const TURN_START = 10;
const TURN_END_OFFSET = 14;
/** Stop turn: the turn starts this long after the speed goal drops to 0 (s). */
const STOP_TURN_DELAY_S = 0.3;
/** Below this speed (m/s) a stop counts as finished (end idle pins allowed). */
const STOP_SPEED = 0.05;
/** previewBuffers: idle pose at or below this speed (m/s), walk pose above. */
const IDLE_PREVIEW_SPEED = 0.3;
/** Average turn rates (deg/s) above which a warning is added. */
const MAX_INPLACE_RATE = 150;
const MAX_SPIN_RATE = 360;
/** End-heading tolerance (deg) of spring-driven turns before a warning. */
const TURN_SETTLE_DEG = 3;
/** Gait bands (m/s): walk below WALK_MAX, jog below JOG_MAX, run above. */
const WALK_MAX = 2.0;
const JOG_MAX = 4.0;

const DEG = Math.PI / 180;
const NON_SOMA_WARNING = 'Pose pins need the SOMA skeleton; this preset is sent as a path only.';

// ---------------------------------------------------------------------------
// parameter definitions
// ---------------------------------------------------------------------------

const SIDE_OPTIONS = Object.freeze([{ value: 'left', label: 'Left' }, { value: 'right', label: 'Right' }]);

function num(key, label, def, min, max, step, unit, extra = {}) {
  return { key, label, type: 'number', default: def, min, max, step, unit, group: 'main', ...extra };
}
const speed = (def, key = 'speed', label = 'Speed') =>
  num(key, label, def, 0.3, 6, 0.1, 'm/s', { hint: 'Walk below 2 m/s, jog from 2 to 4 m/s, run above.' });
const duration = (def = 4, min = 2, max = 10) =>
  num('duration', 'Duration', def, min, max, 0.1, 's', { hint: 'Clip length (60 to 300 frames at 30 fps).' });
const angle = (def, min, max, label = 'Angle', hint) => num('angle', label, def, min, max, 1, '°', hint ? { hint } : {});
const side = (key = 'side', label = 'Side') => ({ key, label, type: 'side', default: 'left', options: SIDE_OPTIONS, group: 'main' });
const time = (key, label, def, min, max, hint) => num(key, label, def, min, max, 0.1, 's', { hint });

const DYN = {
  turnHalflife: num('turnHalflife', 'Turn half-life', DEFAULT_DYNAMICS.turnHalflife, 0.03, 1, 0.01, 's',
    { group: 'dynamics', hint: 'How fast the travel direction follows a change (smaller = sharper corner).' }),
  facingHalflife: num('facingHalflife', 'Facing half-life', DEFAULT_DYNAMICS.facingHalflife, 0.03, 1, 0.01, 's',
    { group: 'dynamics', hint: 'How fast the body turns toward its goal facing (smaller = snappier).' }),
  speedHalflife: num('speedHalflife', 'Speed half-life', DEFAULT_DYNAMICS.speedHalflife, 0.03, 1, 0.01, 's',
    { group: 'dynamics', hint: 'How fast the speed follows a change, start or stop (smaller = quicker).' }),
};
const dynamics = (...keys) => keys.map((k) => DYN[k]);

const GROUND_PIN = {
  key: 'groundPin', label: 'Ground the start (Hips pin)', type: 'bool', default: true, group: 'pins',
  hint: 'Pins pelvis height, position and rotation at frame 0 to a walking pose; prevents a floating start on turning paths.',
};
const END_IDLE = {
  key: 'endIdle', label: 'End on the idle pose', type: 'bool', default: true, group: 'pins',
  hint: 'Pins the last frames to the standing idle pose once the stop has finished.',
};

const DEFS = [
  {
    id: 'acceleration', label: 'Acceleration', experimental: false,
    description: 'Moves forward and changes speed partway through.',
    params: [speed(1.2), speed(2.6, 'speed2', 'Speed in 2nd part'),
      time('first', 'First part', 1.6, 0.5, 9.5, 'Time at the first speed before the change starts.'),
      duration(), ...dynamics('speedHalflife'), GROUND_PIN],
  },
  {
    id: 'constant_speed', label: 'Constant Speed', experimental: false,
    description: 'Moves straight forward at one steady speed.',
    params: [speed(1.2), duration(), GROUND_PIN],
  },
  {
    id: 'strafe', label: 'Strafe', experimental: false,
    description: 'Travels in a fixed direction while the body keeps facing forward.',
    params: [speed(1.0), angle(90, 0, 180, 'Angle', 'Travel direction relative to the facing: 0 forward, 90 sideways, 180 backward.'),
      side(), duration(), GROUND_PIN],
  },
  {
    id: 'arc', label: 'Constant speed/Arc', experimental: false,
    description: 'Moves at a steady speed along a curve with a constant turn rate.',
    params: [speed(1.2), num('rate', 'Arc rate', 20, 1, 90, 1, '°/s', { hint: 'How fast the travel direction rotates.' }),
      side(), duration(), ...dynamics('turnHalflife', 'facingHalflife'), GROUND_PIN],
  },
  {
    id: 'forward_turn', label: 'Forward turn', experimental: false,
    description: 'Moves forward, then changes direction and turns the body to follow.',
    params: [speed(1.2), angle(90, 0, 180), side(),
      time('first', 'Turn at', 2.0, 0.5, 9.5, 'Time of the turn.'), duration(),
      {
        key: 'turnStyle', label: 'Turn style', type: 'select', default: 'default', group: 'main',
        options: [{ value: 'pre', label: 'Pre-turn' }, { value: 'default', label: 'Default' }, { value: 'post', label: 'Post-turn' }],
        hint: 'Pre-turn faces the new direction before the path bends; post-turn bends the path first.',
      },
      ...dynamics('turnHalflife', 'facingHalflife'), GROUND_PIN],
  },
  {
    id: 'strafe_turn', label: 'Strafe turn', experimental: true,
    description: 'Travels while facing forward and changes travel direction partway through.',
    params: [speed(1.0),
      num('startAngle', 'Starting direction', 0, 0, 180, 1, '°', { hint: 'Travel direction of the first part relative to the facing.' }),
      side('startSide', 'Starting side'), angle(90, 0, 180, 'Turn angle'), side(),
      time('first', 'Turn at', 2.0, 0.5, 9.5, 'Time of the direction change.'), duration(),
      ...dynamics('turnHalflife'), GROUND_PIN],
  },
  {
    id: 'start', label: 'Start', experimental: false,
    description: 'Stands still, then starts moving forward.',
    params: [speed(1.2), time('idle', 'Standing time', 0.5, 0.1, 3, 'Time standing before the first step.'),
      duration(), ...dynamics('speedHalflife')],
  },
  {
    id: 'start_turn', label: 'Start turn', experimental: true,
    description: 'Stands still, then sets off in a new direction while turning to face it.',
    params: [speed(1.2), angle(90, 0, 180), side(),
      time('idle', 'Standing time', 0.5, 0.1, 3, 'Time standing before setting off.'),
      duration(), ...dynamics('speedHalflife', 'facingHalflife')],
  },
  {
    id: 'stop', label: 'Stop', experimental: false,
    description: 'Moves forward, then comes to a stop and stands still.',
    params: [speed(1.2), time('moving', 'Moving time', 2.0, 0.5, 9.5, 'Time moving before the stop starts.'),
      duration(), ...dynamics('speedHalflife'), GROUND_PIN, END_IDLE],
  },
  {
    id: 'stop_turn', label: 'Stop turn', experimental: true,
    description: 'Moves forward, stops, then turns on the spot.',
    params: [speed(1.2), angle(90, 15, 360), side(),
      time('moving', 'Moving time', 1.5, 0.5, 9.5, 'Time moving before the stop starts.'),
      duration(5), ...dynamics('speedHalflife'), GROUND_PIN, END_IDLE],
  },
  {
    id: 'spin', label: 'Spin', experimental: true,
    description: 'Spins the body around while it keeps travelling in the same direction.',
    params: [speed(1.2), speed(1.0, 'speed2', 'Speed in 2nd part'), angle(180, 90, 360, 'Spin angle'), side(),
      time('first', 'Spin at', 2.0, 0.5, 9.5, 'Time the spin starts.'),
      time('spin', 'Spin time', 0.8, 0.3, 2, 'How long the spin takes.'),
      duration(), ...dynamics('speedHalflife'), GROUND_PIN],
  },
  {
    id: 'turn_in_place', label: 'Turn in place', experimental: false,
    description: 'Stands on the spot and turns to a new facing.',
    params: [angle(180, 15, 360), side(), duration(3)],
  },
];

function deepFreeze(o) {
  if (o && typeof o === 'object' && !Object.isFrozen(o)) {
    Object.freeze(o);
    for (const v of Object.values(o)) deepFreeze(v);
  }
  return o;
}

/** Every preset, in Motorica's order. */
export const PRESETS = deepFreeze(DEFS);
const BY_ID = new Map(PRESETS.map((d) => [d.id, d]));

export function presetById(id) {
  const def = BY_ID.get(id);
  if (!def) throw new Error(`unknown preset ${JSON.stringify(id)} (expected one of ${[...BY_ID.keys()].join(', ')})`);
  return def;
}

export function defaultParams(id) {
  const out = {};
  for (const pd of presetById(id).params) out[pd.key] = pd.default;
  return out;
}

/**
 * Fills defaults, coerces types, clamps numbers to their ParamDef range and
 * rounds them to the step, then fits the sub-durations into the clip (each
 * event leaves at least 0.5 s). Unknown keys are dropped. Idempotent.
 */
export function normalizeParams(id, p = {}) {
  const def = presetById(id);
  const src = p && typeof p === 'object' ? p : {};
  const out = {};
  for (const pd of def.params) out[pd.key] = normalizeValue(pd, src[pd.key]);
  RECIPES[id].fit?.(out);
  return out;
}

function normalizeValue(pd, v) {
  if (pd.type === 'number') {
    const n = typeof v === 'number' ? v : typeof v === 'string' && v.trim() !== '' ? Number(v) : NaN;
    if (!Number.isFinite(n)) return pd.default;
    return clamp(roundStep(clamp(n, pd.min, pd.max), pd.step), pd.min, pd.max);
  }
  if (pd.type === 'bool') {
    if (typeof v === 'boolean') return v;
    if (v === 'true' || v === 1) return true;
    if (v === 'false' || v === 0) return false;
    return pd.default;
  }
  if (pd.type === 'side') return v === 'left' || v === 'right' ? v : pd.default;
  if (pd.type === 'select') return pd.options.some((o) => o.value === v) ? v : pd.default;
  return pd.default;
}

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
const decimals = (step) => (String(step).split('.')[1] || '').length;
function roundStep(v, step) {
  if (!step) return v;
  return Number((Math.round(v / step) * step).toFixed(decimals(step)));
}
/** Largest multiple of step <= v (with a little float slack). */
function floorStep(v, step) {
  return Number((Math.floor(v / step + 1e-9) * step).toFixed(decimals(step)));
}
/** Clamps time param `key` of `p` to [its min, max], max rounded down to the 0.1 s grid. */
function fitTime(p, id, key, max) {
  const pd = presetById(id).params.find((d) => d.key === key);
  p[key] = Math.max(pd.min, Math.min(p[key], floorStep(max, pd.step)));
}

// ---------------------------------------------------------------------------
// simulation
// ---------------------------------------------------------------------------

const framesOf = (p) => clamp(Math.round(p.duration * FPS), MIN_FRAMES, MAX_FRAMES);
const sign = (s) => (s === 'right' ? -1 : 1);
const dynOf = (p) => ({
  turnHalflife: p.turnHalflife ?? DEFAULT_DYNAMICS.turnHalflife,
  facingHalflife: p.facingHalflife ?? DEFAULT_DYNAMICS.facingHalflife,
  speedHalflife: p.speedHalflife ?? DEFAULT_DYNAMICS.speedHalflife,
  substeps: DEFAULT_DYNAMICS.substeps,
});

/**
 * The trail simulator. velocity(k, t) -> {speed, dir, halflife} at substep k
 * (time t = k dt); facing: {goal(k, t), halflife} drives the facing spring;
 * heading(f) gives an explicit facing per frame instead; neither = facing 0.
 * v0 = initial velocity [vx, vz]. Returns a Sim.
 */
function simulateTrail({ frames, substeps, v0 = [0, 0], velocity, facing = null, heading = null }) {
  const n = substeps;
  const dt = 1 / FPS / n;
  const xz = new Array(frames), hd = new Array(frames), spd = new Array(frames);
  let px = 0, pz = 0, vx = v0[0], vz = v0[1], ax = 0, az = 0, th = 0, w = 0;
  for (let f = 0; f < frames; f++) {
    xz[f] = [px, pz];
    hd[f] = heading ? heading(f) : th;
    spd[f] = Math.hypot(vx, vz);
    if (f === frames - 1) break;
    for (let s = 0; s < n; s++) {
      const k = f * n + s;
      const t = k / (FPS * n);
      const g = velocity(k, t);
      [px, vx, ax] = springCharacterUpdate(px, vx, ax, g.speed * Math.sin(g.dir), g.halflife, dt);
      [pz, vz, az] = springCharacterUpdate(pz, vz, az, g.speed * Math.cos(g.dir), g.halflife, dt);
      if (facing) [th, w] = springDamperExact(th, w, facing.goal(k, t), facing.halflife, dt);
    }
  }
  return { frames, xz, heading: hd, speed: spd };
}

/** factory.py eased_heading: 0 until frame t0, cosine ease to theta at t1, theta after. */
function easedHeading(f, t0, t1, theta) {
  if (f <= t0) return 0;
  if (f >= t1) return theta;
  return theta * 0.5 * (1 - Math.cos((Math.PI * (f - t0)) / (t1 - t0)));
}

/** Substep index of an event at `seconds`. */
const eventStep = (seconds, substeps) => Math.round(seconds * FPS * substeps);
const unit = (speedV, dir) => [speedV * Math.sin(dir), speedV * Math.cos(dir)];

// Per preset recipe:
//   fit(p)             fits the sub-durations into the clip (normalizeParams)
//   simulate(p, dyn)   -> Sim
//   prompt(p, sim)     -> generated prompt
//   moving             starts at speed; gets the frame-0 Hips pin (pin option groundPin)
//   standingStart      starts on the idle pose: fullbody idle.pose@0, idle.next@1
//   endPins(T)         [[frame, idle pose name]] of the end idle pins; endYaw(p, sim) their yaw
//   target(p)          goal facing of a spring-driven turn (end-heading warning)
const RECIPES = {
  acceleration: {
    moving: true,
    fit(p) { fitTime(p, 'acceleration', 'first', p.duration - MIN_TAIL_S); },
    simulate(p, d) {
      const kFirst = eventStep(p.first, d.substeps);
      return simulateTrail({
        frames: framesOf(p), substeps: d.substeps, v0: [0, p.speed],
        velocity: (k) => ({ speed: k >= kFirst ? p.speed2 : p.speed, dir: 0, halflife: d.speedHalflife }),
      });
    },
    prompt(p) {
      const v1 = gaitVerb(p.speed), v2 = gaitVerb(p.speed2);
      const b1 = gaitBand(p.speed), b2 = gaitBand(p.speed2);
      if (b2 > b1) return `A person ${v1} forward and then breaks into a ${GAIT_NOUN[v2]}.`;
      if (b2 < b1) return `A person ${v1} forward and then slows down to a ${GAIT_NOUN[v2]}.`;
      if (p.speed2 > p.speed) return `A person ${v1} forward and gradually speeds up.`;
      if (p.speed2 < p.speed) return `A person ${v1} forward and gradually slows down.`;
      return `A person ${v1} forward at a steady pace.`;
    },
  },
  constant_speed: {
    moving: true,
    simulate(p, d) {
      return simulateTrail({
        frames: framesOf(p), substeps: d.substeps, v0: [0, p.speed],
        velocity: () => ({ speed: p.speed, dir: 0, halflife: d.speedHalflife }),
      });
    },
    prompt: (p) => `A person ${gaitVerb(p.speed)} forward at a steady pace.`,
  },
  strafe: {
    moving: true,
    simulate(p, d) {
      const dir = sign(p.side) * p.angle * DEG;
      return simulateTrail({
        frames: framesOf(p), substeps: d.substeps, v0: unit(p.speed, dir),
        velocity: () => ({ speed: p.speed, dir, halflife: d.turnHalflife }),
      });
    },
    prompt: (p) => `A person ${fixedFacingWords(sign(p.side) * p.angle * DEG, gaitVerb(p.speed))}.`,
  },
  arc: {
    moving: true,
    simulate(p, d) {
      const rate = sign(p.side) * p.rate * DEG;
      const dir = (k, t) => rate * t;
      return simulateTrail({
        frames: framesOf(p), substeps: d.substeps, v0: [0, p.speed],
        velocity: (k, t) => ({ speed: p.speed, dir: dir(k, t), halflife: d.turnHalflife }),
        facing: { goal: dir, halflife: d.facingHalflife },
      });
    },
    prompt: (p) => `A person ${gaitVerb(p.speed)} forward, curving to the ${p.side}.`,
  },
  forward_turn: {
    moving: true,
    fit(p) { fitTime(p, 'forward_turn', 'first', p.duration - MIN_TAIL_S); },
    simulate(p, d) {
      const target = sign(p.side) * p.angle * DEG;
      const kTurn = eventStep(p.first, d.substeps);
      const kFace = eventStep(p.first + (TURN_STYLE[p.turnStyle] - 0.5) * 2 * TURN_LEAD_S, d.substeps);
      return simulateTrail({
        frames: framesOf(p), substeps: d.substeps, v0: [0, p.speed],
        velocity: (k) => ({ speed: p.speed, dir: k >= kTurn ? target : 0, halflife: d.turnHalflife }),
        facing: { goal: (k) => (k >= kFace ? target : 0), halflife: d.facingHalflife },
      });
    },
    target: (p) => sign(p.side) * p.angle * DEG,
    prompt(p) {
      const v = gaitVerb(p.speed);
      if (p.angle < 1) return `A person ${v} forward at a steady pace.`;
      if (p.angle < 30) return `A person ${v} forward and veers slightly to the ${p.side}.`;
      if (p.angle < 135) return `A person ${v} forward and turns ${p.side}.`;
      return `A person ${v} forward and turns around to the ${p.side}.`;
    },
  },
  strafe_turn: {
    moving: true,
    fit(p) { fitTime(p, 'strafe_turn', 'first', p.duration - MIN_TAIL_S); },
    simulate(p, d) {
      const d0 = sign(p.startSide) * p.startAngle * DEG;
      const d1 = d0 + sign(p.side) * p.angle * DEG;
      const kTurn = eventStep(p.first, d.substeps);
      return simulateTrail({
        frames: framesOf(p), substeps: d.substeps, v0: unit(p.speed, d0),
        velocity: (k) => ({ speed: p.speed, dir: k >= kTurn ? d1 : d0, halflife: d.turnHalflife }),
      });
    },
    prompt(p) {
      const v = gaitVerb(p.speed);
      const d0 = sign(p.startSide) * p.startAngle * DEG;
      const d1 = d0 + sign(p.side) * p.angle * DEG;
      const first = fixedFacingWords(d0, v), second = fixedFacingWords(d1, v);
      if (first === second) return `A person ${first}.`;
      if (travelBand(d0).side || travelBand(d1).side) return `A person ${v} at a normal pace.`;
      return `A person ${travelWords(d0, v)} and then ${travelWords(d1, v)}.`;
    },
  },
  start: {
    standingStart: true,
    fit(p) { fitTime(p, 'start', 'idle', p.duration - MIN_TAIL_S); },
    simulate(p, d) {
      const kGo = eventStep(p.idle, d.substeps);
      return simulateTrail({
        frames: framesOf(p), substeps: d.substeps,
        velocity: (k) => ({ speed: k >= kGo ? p.speed : 0, dir: 0, halflife: d.speedHalflife }),
      });
    },
    prompt: (p) => `A person stands still and then starts ${GAIT_GERUND[gaitVerb(p.speed)]} forward.`,
  },
  start_turn: {
    standingStart: true,
    fit(p) { fitTime(p, 'start_turn', 'idle', p.duration - MIN_TAIL_S); },
    simulate(p, d) {
      const target = sign(p.side) * p.angle * DEG;
      const kGo = eventStep(p.idle, d.substeps);
      return simulateTrail({
        frames: framesOf(p), substeps: d.substeps,
        velocity: (k) => ({ speed: k >= kGo ? p.speed : 0, dir: target, halflife: d.speedHalflife }),
        facing: { goal: (k) => (k >= kGo ? target : 0), halflife: d.facingHalflife },
      });
    },
    target: (p) => sign(p.side) * p.angle * DEG,
    prompt(p) {
      const g = GAIT_GERUND[gaitVerb(p.speed)];
      if (p.angle < 22.5) return `A person stands still and then starts ${g} forward.`;
      return `A person stands still, then ${turnWords(p.angle, p.side)} and starts ${g}.`;
    },
  },
  stop: {
    moving: true,
    fit(p) { fitTime(p, 'stop', 'moving', p.duration - MIN_TAIL_S); },
    simulate(p, d) {
      const kStop = eventStep(p.moving, d.substeps);
      return simulateTrail({
        frames: framesOf(p), substeps: d.substeps, v0: [0, p.speed],
        velocity: (k) => ({ speed: k >= kStop ? 0 : p.speed, dir: 0, halflife: d.speedHalflife }),
      });
    },
    endPins: (T) => [[T - 2, 'prev'], [T - 1, 'pose']],
    endYaw: (p, sim) => sim.heading[sim.frames - 1],
    prompt: (p) => `A person ${gaitVerb(p.speed)} forward, then stops and stands still.`,
  },
  stop_turn: {
    moving: true,
    fit(p) {
      const T = framesOf(p);
      fitTime(p, 'stop_turn', 'moving', (T - TURN_END_OFFSET) / FPS - STOP_TURN_DELAY_S - MIN_TAIL_S);
    },
    simulate(p, d) {
      const T = framesOf(p);
      const kStop = eventStep(p.moving, d.substeps);
      const [t0, t1] = stopTurnWindow(p);
      const target = sign(p.side) * p.angle * DEG;
      return simulateTrail({
        frames: T, substeps: d.substeps, v0: [0, p.speed],
        velocity: (k) => ({ speed: k >= kStop ? 0 : p.speed, dir: 0, halflife: d.speedHalflife }),
        heading: (f) => easedHeading(f, t0, t1, target),
      });
    },
    endPins: (T) => [[T - TURN_END_OFFSET, 'pose'], [T - 2, 'prev'], [T - 1, 'pose']],
    endYaw: (p) => sign(p.side) * p.angle * DEG,
    prompt: (p) => `A person ${gaitVerb(p.speed)} forward, stops and ${turnWords(p.angle, p.side)}.`,
  },
  spin: {
    moving: true,
    fit(p) {
      const spinMin = presetById('spin').params.find((d) => d.key === 'spin').min;
      fitTime(p, 'spin', 'first', p.duration - MIN_TAIL_S - spinMin);
      fitTime(p, 'spin', 'spin', p.duration - MIN_TAIL_S - p.first);
    },
    simulate(p, d) {
      const kFirst = eventStep(p.first, d.substeps);
      const [t0, t1] = spinWindow(p);
      const target = sign(p.side) * p.angle * DEG;
      return simulateTrail({
        frames: framesOf(p), substeps: d.substeps, v0: [0, p.speed],
        velocity: (k) => ({ speed: k >= kFirst ? p.speed2 : p.speed, dir: 0, halflife: d.speedHalflife }),
        heading: (f) => easedHeading(f, t0, t1, target),
      });
    },
    prompt(p) {
      const turn = p.angle >= 300 ? `spins all the way around to the ${p.side}`
        : p.angle >= 135 ? `spins around to the ${p.side}` : `turns to the ${p.side}`;
      // continuation: travel direction (+Z) relative to the final facing
      const rel = -sign(p.side) * p.angle * DEG;
      const rest = travelWords(rel, `keeps ${GAIT_GERUND[gaitVerb(p.speed2)]}`);
      return `A person ${gaitVerb(p.speed)} forward, ${turn} and ${rest}.`;
    },
  },
  turn_in_place: {
    standingStart: true,
    simulate(p) {
      const T = framesOf(p);
      const target = sign(p.side) * p.angle * DEG;
      const xz = [], heading = [], spd = [];
      for (let f = 0; f < T; f++) {
        xz.push([0, 0]);
        heading.push(easedHeading(f, TURN_START, T - TURN_END_OFFSET, target));
        spd.push(0);
      }
      return { frames: T, xz, heading, speed: spd };
    },
    endPins: (T) => [[T - TURN_END_OFFSET, 'pose'], [T - 2, 'prev'], [T - 1, 'pose']],
    endYaw: (p) => sign(p.side) * p.angle * DEG,
    prompt: (p) => `A person stands still and then ${turnWords(p.angle, p.side)}.`,
  },
};

const stopTurnWindow = (p) => [Math.round((p.moving + STOP_TURN_DELAY_S) * FPS), framesOf(p) - TURN_END_OFFSET];
const spinWindow = (p) => [Math.round(p.first * FPS), Math.round((p.first + p.spin) * FPS)];

export function simulatePreset(id, p) {
  const params = normalizeParams(id, p);
  return RECIPES[id].simulate(params, dynOf(params));
}

// ---------------------------------------------------------------------------
// prompts
// ---------------------------------------------------------------------------

const GAIT_GERUND = Object.freeze({ walks: 'walking', jogs: 'jogging', runs: 'running' });
const GAIT_NOUN = Object.freeze({ walks: 'walk', jogs: 'jog', runs: 'run' });

/** Gait verb of a speed band: walk below 2 m/s, jog below 4 m/s, run above. */
export function gaitVerb(speedMps) {
  const s = Math.abs(speedMps);
  return s < WALK_MAX ? 'walks' : s < JOG_MAX ? 'jogs' : 'runs';
}
const gaitBand = (s) => ['walks', 'jogs', 'runs'].indexOf(gaitVerb(s));

/** Band of a travel direction relative to the facing (+ = left). */
function travelBand(rel) {
  const r = wrapAngle(rel);
  const deg = Math.abs(r) / DEG;
  const s = r >= 0 ? 'left' : 'right';
  if (deg < 22.5) return { words: 'forward', side: null };
  if (deg < 67.5) return { words: 'diagonally forward', side: s };
  if (deg < 112.5) return { words: 'sideways', side: s };
  if (deg < 157.5) return { words: 'diagonally backward', side: s };
  return { words: 'backward', side: null };
}

/**
 * Prompt words for travel at a fixed (forward) facing. Sideways and diagonal
 * travel get a direction-free gait ("walks at a normal pace"): the heading on
 * every frame holds the facing, and on the GPU (2026-10-10, 2 seeds each)
 * "walks sideways ... while facing forward" crouched the torso 13-18 degrees,
 * "sidesteps" raised a boxing guard, while the neutral wording held the
 * facing best (0.4-0.9 degrees mean) with 6-8 degrees of pitch (research 09,
 * "Parametric presets in the web editor").
 */
function fixedFacingWords(rel, verb) {
  const b = travelBand(rel);
  if (b.words === 'forward') return `${verb} forward at a steady pace`;
  if (b.words === 'backward') return `${verb} backward`;
  return `${verb} at a normal pace`;
}

function travelWords(rel, verb) {
  const b = travelBand(rel);
  return b.side ? `${verb} ${b.words} to the ${b.side}` : `${verb} ${b.words}`;
}

function turnWords(angleDeg, s) {
  if (angleDeg >= 300) return `turns all the way around to the ${s}`;
  if (angleDeg >= 135) return `turns around to the ${s}`;
  return `turns to the ${s}`;
}

// ---------------------------------------------------------------------------
// pins
// ---------------------------------------------------------------------------

const r6 = (v) => {
  const r = Math.round(v * 1e6) / 1e6;
  return r === 0 ? 0 : r; // no -0 in the JSON
};

/**
 * Port of tools/mm_factory/clips.py place(): the pose at smooth-root target
 * [x, z], turned by yaw about +Y (Hips rotation left-multiplied by q_yaw, the
 * pelvis offset rotated). Returns {rot: [J][3] axis-angle, root: [x, y, z], smooth: [x, z]}.
 */
function place(pose, target, yaw = 0) {
  const rot = pose.rot.map((r) => [r[0], r[1], r[2]]);
  let ox = pose.pelvis[0], oz = pose.pelvis[2];
  if (yaw) {
    const q = quatMul(quatFromYaw(yaw), quatFromAxisAngle(pose.rot[0]));
    rot[0] = quatToAxisAngle(q);
    const c = Math.cos(yaw), s = Math.sin(yaw);
    [ox, oz] = [ox * c + oz * s, -ox * s + oz * c];
  }
  return { rot, root: [target[0] + ox, pose.pelvis[1], target[1] + oz], smooth: [target[0], target[1]] };
}

function poseObject(type, frames, placed, extra = {}) {
  return {
    type,
    ...extra,
    frame_indices: frames.slice(),
    local_joints_rot: placed.map((pl) => pl.rot.map((r) => [r6(r[0]), r6(r[1]), r6(r[2])])),
    root_positions: placed.map((pl) => pl.root.map(r6)),
    smooth_root_2d: placed.map((pl) => pl.smooth.map(r6)),
  };
}

// ---------------------------------------------------------------------------
// build
// ---------------------------------------------------------------------------

export function buildPreset(id, p, { skeletonKey = 'soma30', poses = PRESET_POSES } = {}) {
  const def = presetById(id);
  const recipe = RECIPES[id];
  const params = normalizeParams(id, p);
  const sim = recipe.simulate(params, dynOf(params));
  const T = sim.frames;
  const warnings = [];

  // ---- pins: idle pins wanted [{frame, pose: 'pose' | 'next' | 'prev', yaw}] (one fullbody object) + Hips pin
  const idleWanted = [];
  if (recipe.standingStart) {
    idleWanted.push({ frame: 0, pose: 'pose', yaw: sim.heading[0] }, { frame: 1, pose: 'next', yaw: sim.heading[1] });
  }
  if (recipe.endPins && (id === 'turn_in_place' || params.endIdle)) {
    const ends = recipe.endPins(T);
    const first = ends[0][0];
    if (id !== 'turn_in_place' && !(sim.speed[first] < STOP_SPEED)) {
      warnings.push(`The stop is not finished before the end pins (${sim.speed[first].toFixed(2)} m/s at frame ${first}), `
        + 'so the end idle pins were left out; lengthen the clip, shorten the moving time or the speed half-life.');
    } else {
      const yaw = recipe.endYaw(params, sim);
      for (const [frame, pose] of ends) idleWanted.push({ frame, pose, yaw });
    }
  }
  const hipsWanted = !!(recipe.moving && params.groundPin);

  const constraints = [];
  const pins = [];
  const soma = skeletonKey === 'soma30' && poses && poses.soma30;
  if ((idleWanted.length || hipsWanted) && !soma) {
    warnings.push(NON_SOMA_WARNING);
  } else if (soma) {
    const P = poses.soma30;
    if (idleWanted.length) {
      idleWanted.sort((a, b) => a.frame - b.frame);
      const placed = idleWanted.map((w) => place(P.idle[w.pose], sim.xz[w.frame], w.yaw));
      constraints.push(poseObject('fullbody', idleWanted.map((w) => w.frame), placed));
      for (const w of idleWanted) pins.push({ frame: w.frame, kind: 'fullbody', pose: `idle.${w.pose}` });
    }
    if (hipsWanted) {
      // Pelvis EXACTLY on the path point (0, 0) at the walk pose's pelvis height.
      const walk = P.walk;
      const placed = { rot: walk.rot, root: [0, walk.pelvis[1], 0], smooth: [0, 0] };
      constraints.push(poseObject('end-effector', [0], [placed], { joint_names: ['Hips'] }));
      pins.push({ frame: 0, kind: 'hips', pose: 'walk' });
    }
    pins.sort((a, b) => a.frame - b.frame || (a.kind === b.kind ? 0 : a.kind === 'fullbody' ? -1 : 1));
  }
  constraints.push({
    type: 'root2d',
    frame_indices: sim.xz.map((_, f) => f),
    smooth_root_2d: sim.xz.map(([x, z]) => [r6(x), r6(z)]),
    global_root_heading: sim.heading.map((th) => [r6(Math.cos(th)), r6(Math.sin(th))]),
  });

  warnings.push(...presetWarnings(id, params, sim));

  let distance = 0, maxSpeed = 0;
  for (let f = 0; f < T; f++) {
    if (f > 0) distance += Math.hypot(sim.xz[f][0] - sim.xz[f - 1][0], sim.xz[f][1] - sim.xz[f - 1][1]);
    maxSpeed = Math.max(maxSpeed, sim.speed[f]);
  }
  return {
    id, label: def.label, params, frames: T,
    prompt: recipe.prompt(params, sim),
    sim, constraints, pins,
    settings: { ...SETTINGS },
    summary: {
      duration_s: r6(T / FPS),
      distance_m: r6(distance),
      end: [r6(sim.xz[T - 1][0]), r6(sim.xz[T - 1][1])],
      net_turn_deg: r6((sim.heading[T - 1] - sim.heading[0]) / DEG),
      max_speed: r6(maxSpeed),
    },
    warnings,
  };
}

function presetWarnings(id, p, sim) {
  const out = [];
  const T = sim.frames;
  const speeds = ['speed', 'speed2'].filter((k) => k in p).map((k) => p[k]);
  if (speeds.some((s) => s >= WALK_MAX)) {
    out.push('Speeds of 2 m/s and above ask for a jog or run; jog/run quality is untested (the path tests used walking speeds).');
  }
  const recipe = RECIPES[id];
  if (recipe.target) {
    const short = Math.abs(wrapAngle(sim.heading[T - 1] - recipe.target(p))) / DEG;
    if (short > TURN_SETTLE_DEG) {
      out.push(`The turn does not finish before the end of the clip (${short.toFixed(1)}° short); lengthen the clip, `
        + 'turn earlier or shorten the facing half-life.');
    }
  }
  if (id === 'turn_in_place' || id === 'stop_turn') {
    const [t0, t1] = id === 'turn_in_place' ? [TURN_START, T - TURN_END_OFFSET] : stopTurnWindow(p);
    const rate = p.angle / ((t1 - t0) / FPS);
    if (rate > MAX_INPLACE_RATE) {
      out.push(`The turn window is short for the angle (${Math.round(rate)}°/s on average, more than ${MAX_INPLACE_RATE}°/s); `
        + (id === 'stop_turn' ? 'lengthen the clip or shorten the moving time.' : 'lengthen the clip.'));
    }
  }
  if (id === 'spin') {
    const rate = p.angle / p.spin;
    if (rate > MAX_SPIN_RATE) out.push(`The spin is very fast (${Math.round(rate)}°/s on average); lengthen the spin time.`);
  }
  // (with endIdle on, buildPreset already warned that the end pins were left out)
  if ((id === 'stop' || id === 'stop_turn') && !p.endIdle && !(sim.speed[T - 1] < STOP_SPEED)) {
    out.push(`The stop does not finish within the clip (${sim.speed[T - 1].toFixed(2)} m/s at the end).`);
  }
  return out;
}

// ---------------------------------------------------------------------------
// preview motion
// ---------------------------------------------------------------------------

/**
 * Preview motion buffers for a Built preset (Motion layout): for SOMA-30 the
 * anchor pose (idle at speeds <= 0.3 m/s, else walk) placed at every frame's
 * path point and turned by that frame's heading; for other skeletons the rest
 * pose translated and turned the same way.
 */
export function previewBuffers(built, skeleton, { poses = PRESET_POSES } = {}) {
  const sim = built.sim;
  const T = built.frames;
  const J = skeleton.J;
  const local = new Float32Array(T * J * 4);
  const root = new Float32Array(T * 3);
  const yawQ = [0, 0, 0, 1], q = [0, 0, 0, 1];
  const soma = skeleton.key === 'soma30' && J === 30 && poses && poses.soma30;
  if (soma) {
    const prep = (pose) => ({ q: pose.rot.map((r) => quatFromAxisAngle(r)), pelvis: pose.pelvis });
    const idle = prep(poses.soma30.idle.pose), walk = prep(poses.soma30.walk);
    for (let f = 0; f < T; f++) {
      const a = sim.speed[f] <= IDLE_PREVIEW_SPEED ? idle : walk;
      const th = sim.heading[f];
      for (let j = 0; j < J; j++) local.set(a.q[j], (f * J + j) * 4);
      local.set(quatMul(quatFromYaw(th, yawQ), a.q[0], q), f * J * 4);
      const c = Math.cos(th), s = Math.sin(th);
      // The walk pose stands with its pelvis ON the path, like the Hips pin, so
      // a snapshot of the preview at frame 0 (timeline add, write-through)
      // keeps the pin where buildPreset put it; the idle pose keeps its offset
      // from the smooth root, like the idle pins (place()).
      const ox = a === walk ? 0 : a.pelvis[0], oz = a === walk ? 0 : a.pelvis[2];
      root[3 * f] = sim.xz[f][0] + ox * c + oz * s;
      root[3 * f + 1] = a.pelvis[1];
      root[3 * f + 2] = sim.xz[f][1] - ox * s + oz * c;
    }
  } else {
    const rest = skeleton.restPose();
    const r = skeleton.rootIndex ?? 0;
    const q0 = Array.from(rest.local.subarray(4 * r, 4 * r + 4));
    for (let f = 0; f < T; f++) {
      local.set(rest.local, f * J * 4);
      local.set(quatMul(quatFromYaw(sim.heading[f], yawQ), q0, q), (f * J + r) * 4);
      root[3 * f] = sim.xz[f][0];
      root[3 * f + 1] = rest.root[1];
      root[3 * f + 2] = sim.xz[f][1];
    }
  }
  return { local, root };
}

// ---------------------------------------------------------------------------
// top-down SVG
// ---------------------------------------------------------------------------

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const fmt = (v) => (Number.isFinite(v) ? String(Math.round(v * 100) / 100) : '0');

/**
 * Top-down drawing of a Sim as an SVG string (no DOM). opts: {width = 160,
 * height = 120, padding, markerEvery = 15 (frames between facing triangles),
 * grid = false (metre grid), className, title, minSpan = 1.5 (m; the least
 * extent shown, so a turn in place is a centred marker), strokeWidth,
 * markerSize}. Classes for theming: preset-svg (root), preset-grid,
 * preset-path, preset-facing, preset-start, preset-end; colours default to
 * currentColor.
 */
export function presetSVG(sim, opts = {}) {
  const W = opts.width ?? 160;
  const H = opts.height ?? 120;
  const small = Math.min(W, H);
  const pad = clamp(opts.padding ?? Math.round(small * 0.12), 0, small / 2 - 1);
  const every = Math.max(1, Math.round(opts.markerEvery ?? 15));
  const minSpan = opts.minSpan ?? 1.5;
  const sw = opts.strokeWidth ?? Math.max(1.25, small / 70);
  const L = opts.markerSize ?? Math.max(3, small / 22);
  const xz = sim.xz;
  const T = xz.length;

  let x0 = Infinity, x1 = -Infinity, z0 = Infinity, z1 = -Infinity;
  for (const [x, z] of xz) {
    if (!Number.isFinite(x) || !Number.isFinite(z)) continue;
    x0 = Math.min(x0, x); x1 = Math.max(x1, x); z0 = Math.min(z0, z); z1 = Math.max(z1, z);
  }
  if (!(x0 <= x1)) { x0 = x1 = 0; z0 = z1 = 0; }
  // minSpan caps the zoom, so a path of (almost) zero extent stays a centred marker.
  const spanX = Math.max(x1 - x0, minSpan), spanZ = Math.max(z1 - z0, minSpan);
  const scale = Math.min((W - 2 * pad) / spanX, (H - 2 * pad) / spanZ);
  const mx = (x0 + x1) / 2, mz = (z0 + z1) / 2;
  const sx = (X) => W / 2 - (X - mx) * scale;
  const sy = (Z) => H / 2 - (Z - mz) * scale;

  const parts = [];
  const cls = ['preset-svg', opts.className].filter(Boolean).join(' ');
  const label = opts.title ? ` role="img" aria-label="${esc(opts.title)}"` : ' aria-hidden="true"';
  parts.push(`<svg xmlns="http://www.w3.org/2000/svg" class="${esc(cls)}" viewBox="0 0 ${fmt(W)} ${fmt(H)}" width="${fmt(W)}" height="${fmt(H)}"${label}>`);
  if (opts.title) parts.push(`<title>${esc(opts.title)}</title>`);

  if (opts.grid) {
    const gx0 = Math.ceil(mx - W / 2 / scale), gx1 = Math.floor(mx + W / 2 / scale);
    const gz0 = Math.ceil(mz - H / 2 / scale), gz1 = Math.floor(mz + H / 2 / scale);
    let d = '';
    for (let k = gx0; k <= gx1; k++) d += `M${fmt(sx(k))} 0V${fmt(H)}`;
    for (let k = gz0; k <= gz1; k++) d += `M0 ${fmt(sy(k))}H${fmt(W)}`;
    if (d) parts.push(`<path class="preset-grid" d="${d}" fill="none" stroke="currentColor" stroke-opacity="0.15" stroke-width="1"/>`);
  }

  let d = '';
  for (let f = 0; f < T; f++) d += `${f ? 'L' : 'M'}${fmt(sx(xz[f][0]))} ${fmt(sy(xz[f][1]))}`;
  parts.push(`<path class="preset-path" d="${d}" fill="none" stroke="currentColor" stroke-width="${fmt(sw)}" stroke-linecap="round" stroke-linejoin="round"/>`);

  const marks = [];
  for (let f = 0; f < T; f += every) marks.push(f);
  if (marks.at(-1) !== T - 1) marks.push(T - 1);
  let tri = '';
  for (const f of marks) {
    const th = sim.heading[f];
    const ux = -Math.sin(th), uy = -Math.cos(th); // screen direction of the facing
    // Standing frames push their triangle out along the facing, so an in-place
    // turn reads as a fan around the spot instead of a pile of triangles.
    const still = sim.speed ? sim.speed[f] < STOP_SPEED : false;
    const off = still ? L * 1.4 : 0;
    const cx = sx(xz[f][0]) + ux * off, cy = sy(xz[f][1]) + uy * off;
    const tipX = cx + ux * L * 1.6, tipY = cy + uy * L * 1.6;
    const bx = cx + uy * L * 0.55, by = cy - ux * L * 0.55; // base corners: +-normal
    const ex = cx - uy * L * 0.55, ey = cy + ux * L * 0.55;
    tri += `M${fmt(tipX)} ${fmt(tipY)}L${fmt(bx)} ${fmt(by)}L${fmt(ex)} ${fmt(ey)}Z`;
  }
  parts.push(`<path class="preset-facing" d="${tri}" fill="currentColor" fill-opacity="0.85"/>`);

  const r = Math.max(2, L * 0.6);
  parts.push(`<circle class="preset-start" cx="${fmt(sx(xz[0][0]))}" cy="${fmt(sy(xz[0][1]))}" r="${fmt(r)}" fill="currentColor"/>`);
  parts.push(`<circle class="preset-end" cx="${fmt(sx(xz[T - 1][0]))}" cy="${fmt(sy(xz[T - 1][1]))}" r="${fmt(r)}" fill="none" stroke="currentColor" stroke-width="${fmt(sw)}"/>`);
  parts.push('</svg>');
  return parts.join('');
}
