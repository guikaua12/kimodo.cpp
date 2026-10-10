import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { smoothSignal, smoothPathXZ, getSmoothRootPos, TrajectorySmoother, SMOOTH_ROOT_MARGIN } from '../smooth_root.js';

const fixture = JSON.parse(fs.readFileSync(new URL('./fixtures/smooth_root_upstream.json', import.meta.url), 'utf8'));
const maxDiff = (a, b) => {
  assert.equal(a.length, b.length);
  let m = 0;
  a.forEach((r, i) => r.forEach((v, d) => { m = Math.max(m, Math.abs(v - b[i][d])); }));
  return m;
};

test('TrajectorySmoother matches upstream single-level ADMM (incl. rho rule)', () => {
  for (const c of fixture.trajectory_smoother) {
    const s = new TrajectorySmoother(new Array(c.N).fill(SMOOTH_ROOT_MARGIN), {
      posWeight: c.pos_weight, admmIters: c.iters, alphaOverrelax: c.alpha,
    });
    assert.ok(Math.abs(s.admmStepsize - c.rho) < 1e-15, `rho N=${c.N}`);
    const d = maxDiff(s.smooth(c.targets, c.x0), c.out);
    assert.ok(d < 1e-9, `N=${c.N}: max diff ${d}`);
  }
});

test('smoothSignal matches upstream smooth_signal on random/straight/zigzag/circle paths', () => {
  for (const c of fixture.smooth_signal) {
    const d = maxDiff(smoothSignal(c.x, SMOOTH_ROOT_MARGIN), c.out);
    assert.ok(d < 1e-9, `${c.name}: max diff ${d}`);
    assert.ok(maxDiff(smoothPathXZ(c.x), c.out) < 1e-9);
  }
});

test('margin semantics: output stays within ~margin of the input (plus the half-frame shift)', () => {
  const c = fixture.smooth_signal.find((x) => x.name === 'random-300');
  const out = smoothSignal(c.x, SMOOTH_ROOT_MARGIN);
  // The final level shifts samples forward by half a frame (upstream quirk), so
  // compare against the midpoint of consecutive inputs.
  for (let i = 0; i < out.length - 2; i++) {
    const mid = [(c.x[i][0] + c.x[i + 1][0]) / 2, (c.x[i][1] + c.x[i + 1][1]) / 2];
    assert.ok(Math.hypot(out[i][0] - mid[0], out[i][1] - mid[1]) < SMOOTH_ROOT_MARGIN + 0.01, `frame ${i}`);
  }
});

test('last-level aliasing quirk: out[T-2] averages with the already extrapolated last sample', () => {
  // A reading of upstream smooth_signal suggests out[i] = (s[i] + s[i+1]) / 2 for i < T-1, but
  // numpy extrapolates xs[T-1] in place first, so out[T-2] reads the new value.
  // With the non-aliased formula the upstream vectors would differ by
  // (s[T-1] - s[T-2]) / 4 at T-2 (centimetres on a walking path); they match to 1e-9.
  // Recompute T = 8 by hand with the exported single-level smoother:
  // level step 2 (samples 0,2,4,6), interpolate, then the full-resolution level.
  const x = [[0, 0], [0.1, 0.02], [0.25, 0.01], [0.3, 0.2], [0.5, 0.25], [0.55, 0.4], [0.8, 0.45], [0.9, 0.7]];
  const T = x.length, opts = { posWeight: 0, alphaOverrelax: 1.8, admmIters: 500 };
  const mean = [0, 1].map((d) => x.reduce((s, r) => s + r[d], 0) / T);
  const xs = x.map(() => mean.slice());
  const coarse = new TrajectorySmoother([0.06, 0.06, 0.06, 0.06], opts).smooth([x[0], x[2], x[4], x[6]], [xs[0], xs[2], xs[4], xs[6]]);
  coarse.forEach((r, k) => { xs[2 * k] = r; });
  xs[7] = coarse[3].map((v, d) => v + (v - coarse[2][d]) / 2); // 4 interleaved == 4 coarse: extrapolate
  for (let k = 0; k < 3; k++) xs[2 * k + 1] = coarse[k].map((v, d) => (v + coarse[k + 1][d]) / 2);
  const s = new TrajectorySmoother(new Array(T).fill(0.06), opts).smooth(x, xs);
  const last = s[T - 1].map((v, d) => v + (v - s[T - 2][d]) / 2);
  const expected = s.slice(0, T - 1).map((r, i) => r.map((v, d) => (v + (i + 1 === T - 1 ? last : s[i + 1])[d]) / 2));
  expected.push(last);
  const out = smoothSignal(x, SMOOTH_ROOT_MARGIN);
  assert.ok(maxDiff(out, expected) < 1e-12, `aliased last-level interpolation, diff ${maxDiff(out, expected)}`);
  const naive = (s[T - 2][0] + s[T - 1][0]) / 2;
  assert.ok(Math.abs(naive - out[T - 2][0]) > 1e-4, 'the non-aliased closed form would differ here');
});

test('short signals (T < 5, singular upstream) are returned unchanged; singular level throws', () => {
  for (const T of [0, 1, 2, 3, 4]) {
    const x = Array.from({ length: T }, (_, i) => [i, -i]);
    assert.deepEqual(smoothSignal(x, 0.06), x);
  }
  assert.throws(() => new TrajectorySmoother([0.06, 0.06]), /singular/);
  assert.doesNotThrow(() => smoothSignal([[0, 0], [1, 0], [2, 1], [3, 1], [4, 2]], 0.06));
});

test('getSmoothRootPos smooths x/z with the 0.06 margin and keeps y', () => {
  const c = fixture.smooth_signal.find((x) => x.name === 'circle-150');
  const pos = c.x.map(([x, z], i) => [x, 0.9 + 0.01 * Math.sin(i), z]);
  const out = getSmoothRootPos(pos);
  out.forEach((p, i) => {
    assert.equal(p[1], pos[i][1]);
    assert.ok(Math.abs(p[0] - c.out[i][0]) < 1e-9 && Math.abs(p[2] - c.out[i][1]) < 1e-9);
  });
});
