import test from 'node:test';
import assert from 'node:assert/strict';
import {
  LN2, halflifeToDamping, halflifeToDecay, springDamperExact, springCharacterUpdate,
  settleTime50, settleTime95, peakRate, stopDistance,
} from '../spring.js';

const near = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg ?? ''} ${a} vs ${b} (tol ${tol})`);

test('damping and decay follow Holden (eps 1e-5)', () => {
  near(LN2, 0.6931471805599453, 1e-15);
  near(halflifeToDamping(0.09), (4 * Math.log(2)) / (0.09 + 1e-5), 1e-12);
  near(halflifeToDamping(0.09, 0), (4 * Math.log(2)) / 0.09, 1e-12);
  near(halflifeToDecay(0.07), halflifeToDamping(0.07) / 2, 1e-12);
  near(halflifeToDecay(0.07), 19.80, 0.01, 'research table: y at h = 0.07');
});

test('closed forms: settle times, peak rate, stop distance', () => {
  for (const h of [0.05, 0.07, 0.12, 0.3, 1.0]) {
    near(settleTime50(h) / h, 1.211, 0.005, `T50/h at ${h}`);
    near(settleTime95(h) / h, 3.422, 0.005, `T95/h at ${h}`);
    near(stopDistance(1.2, h), (1.2 * (h + 1e-5)) / LN2, 1e-12, `stop distance at ${h}`);
    near(stopDistance(1.2, h), (1.2 * h) / LN2, 1e-3 * (1.2 * h) / LN2, `~speed h / ln2 at ${h}`);
  }
  // research 05 A.8 table: 90 degree step at h = 0.07 peaks at ~656 deg/s; 1.2 m/s lag at h = 0.09 is 0.156 m.
  near(peakRate(90, 0.07), 656, 1);
  near(stopDistance(1.2, 0.09), 0.156, 0.0005);
});

test('springDamperExact reaches the goal like the closed-form step response', () => {
  const h = 0.12, y = halflifeToDecay(h);
  let x = 0, v = 0;
  const dt = 1 / 300;
  let t = 0;
  for (let k = 0; k < 300; k++) {
    [x, v] = springDamperExact(x, v, 1, h, dt);
    t += dt;
    const expected = 1 - (1 + y * t) * Math.exp(-y * t);
    near(x, expected, 1e-12, `step response at t=${t}`);
  }
  // 50% at settleTime50, 95% at settleTime95
  const at = (T) => { let p = 0, w = 0; const n = 1000; for (let k = 0; k < n; k++) [p, w] = springDamperExact(p, w, 1, h, T / n); return p; };
  near(at(settleTime50(h)), 0.5, 1e-9);
  near(at(settleTime95(h)), 0.95, 1e-9);
  // exact for any dt with a constant goal: one big step == many small ones
  const [x1, v1] = springDamperExact(0.3, -2, 1.5, 0.2, 0.5);
  let xs = 0.3, vs = -2;
  for (let k = 0; k < 50; k++) [xs, vs] = springDamperExact(xs, vs, 1.5, 0.2, 0.01);
  near(xs, x1, 1e-12); near(vs, v1, 1e-12);
});

test('springCharacterUpdate: velocity follows the goal, position lag converges to dv h / ln2', () => {
  for (const h of [0.09, 0.3, 0.6]) {
    const dv = 1.2;
    let x = 0, v = 0, a = 0;
    const dt = 1 / 300;
    const T = 20 * h + 2;
    const n = Math.round(T / dt);
    for (let k = 0; k < n; k++) [x, v, a] = springCharacterUpdate(x, v, a, dv, h, dt);
    near(v, dv, 1e-9, 'velocity settles');
    near(a, 0, 1e-8, 'acceleration settles');
    const lag = dv * n * dt - x;
    const expected = (dv * h) / LN2;
    assert.ok(Math.abs(lag - expected) <= 0.01 * expected, `lag ${lag} vs ${expected} at h=${h}`);
    near(lag, stopDistance(dv, h), 1e-6, 'lag == stopDistance');
  }
  // a constant velocity with zero acceleration integrates exactly
  let x = 0, v = 1.2, a = 0;
  for (let k = 0; k < 300; k++) [x, v, a] = springCharacterUpdate(x, v, a, 1.2, 0.12, 1 / 300);
  near(x, 1.2, 1e-12); near(v, 1.2, 0); near(a, 0, 0);
});

test('springCharacterUpdate matches the probe generator formula token for token', () => {
  // build_requests.py spring_character_update, transcribed
  const ref = (x, v, a, vGoal, halflife, dt) => {
    const y = ((4 * Math.log(2)) / (halflife + 1e-5)) / 2;
    const j0 = v - vGoal, j1 = a + j0 * y, e = Math.exp(-y * dt);
    return [e * (((-j1) / (y * y)) + ((-j0 - j1 * dt) / y)) + (j1 / (y * y)) + j0 / y + vGoal * dt + x,
      e * (j0 + j1 * dt) + vGoal, e * (a - j1 * y * dt)];
  };
  const args = [[0.4, 1.1, -0.3, 0.2, 0.09, 1 / 300], [-2, 0, 0, 1.2, 0.3, 1 / 30], [5, -1, 4, -3, 0.03, 0.01]];
  for (const arg of args) assert.deepEqual(springCharacterUpdate(...arg), ref(...arg));
});
