// Critically damped springs for the parametric path presets (presets.js).
//
// The maths is Daniel Holden's (MIT licence): "Spring-It-On: The Game
// Developer's Spring-Roll-Call", https://theorangeduck.com/page/spring-roll-call
// and https://github.com/orangeduck/Spring-It-On (simple_spring_damper_exact,
// spring_character_update). Re-implemented here from the published formulas;
// see docs/motion-stage-research/05-root-paths-presets-and-motion-factory.md
// sections A.2-A.4 and A.8. Every exponential is the exact Math.exp (no
// fast_negexp approximation), matching the probe path generator
// (docs/motion-stage-research/probe/scripts/build_requests.py).
//
// Conventions
// - Scalars only: run a spring per axis (x and z of a velocity) or on an
//   unwrapped yaw angle. Time is in seconds, half-lives in seconds.
// - A half-life h is the time in which the remaining error of a pure
//   exponential halves; the decay rate is y = halflifeToDecay(h) ~ 2 ln2 / h.
//   The critically damped step response is r(t) = delta (1 + y t) e^(-y t).
// - The solutions are exact for any dt while the goal is constant within the
//   step; callers substep when the goal moves continuously.
//
// API
//   LN2
//   halflifeToDamping(h, eps = 1e-5)   4 ln2 / (h + eps)
//   halflifeToDecay(h)                 damping / 2 (y)
//   springDamperExact(x, v, goal, halflife, dt)            -> [x, v]
//   springCharacterUpdate(x, v, a, vGoal, halflife, dt)    -> [x, v, a]
//   settleTime50(h), settleTime95(h)   time to 50% / 95% of a step (~1.211 h, ~3.422 h)
//   peakRate(delta, h)                 peak rate of change of a step response, delta y / e
//   stopDistance(speed, h)             distance lost against an instant speed change, 2 speed / y

export const LN2 = Math.LN2;

/** Holden's halflife_to_damping: 4 ln2 / (h + eps). */
export function halflifeToDamping(h, eps = 1e-5) {
  return (4 * LN2) / (h + eps);
}

/** Decay rate y of the critically damped spring (damping / 2). */
export function halflifeToDecay(h) {
  return halflifeToDamping(h) / 2;
}

/**
 * Holden's simple_spring_damper_exact: critically damped spring of x (rate v)
 * toward a constant goal. Returns [x, v] after dt.
 */
export function springDamperExact(x, v, goal, halflife, dt) {
  const y = halflifeToDecay(halflife);
  const j0 = x - goal;
  const j1 = v + j0 * y;
  const eydt = Math.exp(-y * dt);
  return [eydt * (j0 + j1 * dt) + goal, eydt * (v - j1 * y * dt)];
}

/**
 * Holden's spring_character_update: a critically damped spring on the
 * velocity v (acceleration a) toward vGoal that integrates the position x.
 * Returns [x, v, a] after dt.
 */
export function springCharacterUpdate(x, v, a, vGoal, halflife, dt) {
  const y = halflifeToDecay(halflife);
  const j0 = v - vGoal;
  const j1 = a + j0 * y;
  const eydt = Math.exp(-y * dt);
  return [
    eydt * ((-j1 / (y * y)) + ((-j0 - j1 * dt) / y)) + (j1 / (y * y)) + j0 / y + vGoal * dt + x,
    eydt * (j0 + j1 * dt) + vGoal,
    eydt * (a - j1 * y * dt),
  ];
}

// ---------------------------------------------------------------------------
// closed forms (research 05 section A.8)
// ---------------------------------------------------------------------------

/** Solves (1 + u) e^-u = r for u > 0 (the step response has r(u) falling from 1 to 0). */
function solveRemaining(r) {
  let lo = 0, hi = 1;
  while ((1 + hi) * Math.exp(-hi) > r) hi *= 2;
  for (let k = 0; k < 200; k++) {
    const mid = (lo + hi) / 2;
    if ((1 + mid) * Math.exp(-mid) > r) lo = mid; else hi = mid;
  }
  return (lo + hi) / 2;
}

const U50 = solveRemaining(0.5); // y t ~ 1.678
const U95 = solveRemaining(0.05); // y t ~ 4.744

/** Time for a critically damped step response to cover 50% of the step (~1.211 h). */
export function settleTime50(h) {
  return U50 / halflifeToDecay(h);
}

/** Time for a critically damped step response to cover 95% of the step (~3.422 h). */
export function settleTime95(h) {
  return U95 / halflifeToDecay(h);
}

/** Peak rate of change (units/s) of the response to a step of size delta: delta y / e, at t = 1/y. */
export function peakRate(delta, h) {
  return (delta * halflifeToDecay(h)) / Math.E;
}

/**
 * Distance lost against an instant change of speed (also the stopping
 * distance from `speed` with a speed goal of 0): 2 speed / y ~ speed h / ln2.
 */
export function stopDistance(speed, h) {
  return (2 * speed) / halflifeToDecay(h);
}
