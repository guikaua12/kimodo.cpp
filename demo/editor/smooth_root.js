// Smooth root trajectory: exact port of upstream NVIDIA Kimodo
// kimodo/motion_rep/smooth_root.py (Apache-2.0): TrajectorySmoother (ADMM),
// smooth_signal (multigrid) and get_smooth_root_pos (xz smoothing, y kept).
// The demo's "Make Smooth Path" runs get_smooth_root_pos on the linearly
// interpolated waypoint path (viz/constraint_ui.py interpolate_path).
//
// Differences from upstream, all numerical-only:
// - The pentadiagonal SPD system M = posWeight I + A^T A + rho I is solved with
//   a banded Cholesky factorisation instead of scipy's sparse LU (splu).
// - Signals shorter than 5 samples are returned unchanged: upstream builds a
//   singular system there (every multigrid level has N <= 2 samples, so
//   A^T A = 0 and rho = 0) and scipy raises.
// Validated against the upstream Python on random/straight/zigzag paths
// (tests/smooth_root.test.mjs, fixtures/smooth_root_upstream.json).
//
// Signals are arrays of rows ([T][D] number arrays); computation is float64.
//
// Exports: TrajectorySmoother, smoothSignal, getSmoothRootPos, smoothPathXZ,
// SMOOTH_ROOT_MARGIN. (The editor itself only needs smoothPathXZ.)

/** Margin (metres) used by get_smooth_root_pos on every frame. */
export const SMOOTH_ROOT_MARGIN = 0.06;

/**
 * ADMM smoother for one multigrid level (smooth_root.py TrajectorySmoother,
 * loop=False, circle_project=False). margins: number[N].
 */
export class TrajectorySmoother {
  constructor(margins, { posWeight = 0, admmIters = 100, alphaOverrelax = 1.0 } = {}) {
    const N = margins.length;
    this.N = N;
    this.posWeight = posWeight;
    this.admmIters = admmIters;
    this.alphaOverrelax = alphaOverrelax;
    this.marginVals = Float64Array.from(margins);

    // A has rows i = 1..N-2 equal to [-1, 2, -1] at columns i-1, i, i+1; rows 0
    // and N-1 are zero. M0 = posWeight I + A^T A is symmetric pentadiagonal; we
    // keep its diagonal and first/second sub-diagonals: d0[i] = M[i][i],
    // d1[i] = M[i][i-1], d2[i] = M[i][i-2].
    const d0 = new Float64Array(N), d1 = new Float64Array(N), d2 = new Float64Array(N);
    const row = [-1, 2, -1];
    for (let i = 1; i < N - 1; i++) {
      // A^T A += a_i^T a_i where a_i touches columns i-1, i, i+1.
      for (let a = 0; a < 3; a++) {
        for (let b = 0; b <= a; b++) {
          const r = i - 1 + a, c = i - 1 + b, v = row[a] * row[b];
          if (r === c) d0[r] += v; else if (r - c === 1) d1[r] += v; else d2[r] += v;
        }
      }
    }
    for (let i = 0; i < N; i++) d0[i] += posWeight;

    let diagMax = 0;
    for (let i = 0; i < N; i++) diagMax = Math.max(diagMax, Math.abs(d0[i]));
    this.admmStepsize = 0.25 * Math.sqrt(diagMax);
    for (let i = 0; i < N; i++) d0[i] += this.admmStepsize;

    // Banded Cholesky M = L L^T, L lower triangular with bandwidth 2.
    const l0 = new Float64Array(N), l1 = new Float64Array(N), l2 = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      if (i >= 2) l2[i] = d2[i] / l0[i - 2];
      if (i >= 1) l1[i] = (d1[i] - (i >= 2 ? l2[i] * l1[i - 1] : 0)) / l0[i - 1];
      const diag = d0[i] - l1[i] * l1[i] - l2[i] * l2[i];
      if (!(diag > 0)) {
        throw new Error(`TrajectorySmoother: system matrix is singular (N=${N}, posWeight=${posWeight}); need N >= 3 or posWeight > 0`);
      }
      l0[i] = Math.sqrt(diag);
    }
    this._l0 = l0; this._l1 = l1; this._l2 = l2;
  }

  /** Solves M x = r in place for one column stored with stride D at offset c. */
  _solveColumn(r, D, c) {
    const N = this.N, l0 = this._l0, l1 = this._l1, l2 = this._l2;
    // forward: L y = r
    for (let i = 0; i < N; i++) {
      let s = r[i * D + c];
      if (i >= 1) s -= l1[i] * r[(i - 1) * D + c];
      if (i >= 2) s -= l2[i] * r[(i - 2) * D + c];
      r[i * D + c] = s / l0[i];
    }
    // backward: L^T x = y
    for (let i = N - 1; i >= 0; i--) {
      let s = r[i * D + c];
      if (i + 1 < N) s -= l1[i + 1] * r[(i + 1) * D + c];
      if (i + 2 < N) s -= l2[i + 2] * r[(i + 2) * D + c];
      r[i * D + c] = s / l0[i];
    }
  }

  /**
   * ADMM iterations (smooth_root.py smooth): z-update (project x + u onto the
   * margin ball around the target), u-update (over-relaxed), x-update (solve).
   * targets, x0: Float64Array(N*D) row-major. Returns a new Float64Array.
   */
  smoothFlat(targets, x0, D) {
    const N = this.N;
    const x = Float64Array.from(x0);
    const z = new Float64Array(N * D);
    const u = new Float64Array(N * D);
    const r = new Float64Array(N * D);
    const margins = this.marginVals, alpha = this.alphaOverrelax, rho = this.admmStepsize, pw = this.posWeight;
    for (let it = 0; it < this.admmIters; it++) {
      // z = x + u - t; rows farther than margin from the target are scaled back.
      for (let i = 0; i < N; i++) {
        let n2 = 0;
        for (let d = 0; d < D; d++) {
          const k = i * D + d;
          z[k] = x[k] + u[k] - targets[k];
          n2 += z[k] * z[k];
        }
        const norm = Math.sqrt(n2);
        if (norm > margins[i]) {
          const s = margins[i] / norm;
          for (let d = 0; d < D; d++) z[i * D + d] *= s;
        }
        for (let d = 0; d < D; d++) z[i * D + d] += targets[i * D + d];
      }
      // u += alpha (x - z)
      for (let k = 0; k < N * D; k++) u[k] += alpha * (x[k] - z[k]);
      // x = M^-1 (posWeight t + rho (z - u))
      for (let k = 0; k < N * D; k++) r[k] = pw * targets[k] + rho * (z[k] - u[k]);
      for (let d = 0; d < D; d++) this._solveColumn(r, D, d);
      x.set(r);
    }
    return x;
  }

  /** Row-array convenience wrapper: targets, x0: [N][D] -> [N][D]. */
  smooth(targets, x0) {
    const D = targets[0].length;
    const out = this.smoothFlat(Float64Array.from(targets.flat()), Float64Array.from(x0.flat()), D);
    return Array.from({ length: this.N }, (_, i) => Array.from(out.subarray(i * D, i * D + D)));
  }
}

/**
 * Multigrid smoothing with margin constraints (smooth_root.py smooth_signal).
 * x: [T][D] rows; margins: number[T] or a single number for every frame.
 * Returns new [T][D] rows.
 */
export function smoothSignal(x, margins, { posWeight = 0, alphaOverrelax = 1.8, admmIters = 500 } = {}) {
  const T = x.length;
  if (T === 0) return [];
  const D = x[0].length;
  const marg = typeof margins === 'number' ? new Float64Array(T).fill(margins) : Float64Array.from(margins);
  if (marg.length !== T) throw new Error(`smoothSignal: ${marg.length} margins for ${T} samples`);
  // Upstream raises (singular system) below 5 samples; keep the input instead.
  if (T < 5 && posWeight === 0) return x.map((r) => Array.from(r, Number));

  const X = new Float64Array(T * D);
  for (let i = 0; i < T; i++) for (let d = 0; d < D; d++) X[i * D + d] = x[i][d];

  // xs[:] = mean over time
  const xs = new Float64Array(T * D);
  for (let d = 0; d < D; d++) {
    let s = 0;
    for (let i = 0; i < T; i++) s += X[i * D + d];
    const m = s / T;
    for (let i = 0; i < T; i++) xs[i * D + d] = m;
  }

  // levels = max(floor(log2(T)) - 4, 1); stepsize = 2**levels
  const levels = Math.max((31 - Math.clz32(T)) - 4, 1);
  let step = 2 ** levels;
  for (;;) {
    // Coarse samples xs[::step] (N of them) are smoothed against x[::step].
    const N = Math.ceil(T / step);
    const subMargins = new Float64Array(N);
    const targets = new Float64Array(N * D);
    const x0 = new Float64Array(N * D);
    for (let k = 0; k < N; k++) {
      subMargins[k] = marg[k * step];
      for (let d = 0; d < D; d++) {
        targets[k * D + d] = X[k * step * D + d];
        x0[k * D + d] = xs[k * step * D + d];
      }
    }
    const smoother = new TrajectorySmoother(subMargins, { posWeight, alphaOverrelax, admmIters });
    const res = smoother.smoothFlat(targets, x0, D);
    for (let k = 0; k < N; k++) for (let d = 0; d < D; d++) xs[k * step * D + d] = res[k * D + d];

    // Interpolate to the next level: xs[half::step] are the in-between samples.
    // At step == 1, half == 0 and xs[0::1] aliases the whole array; like numpy we
    // first extrapolate the last sample in place and only then evaluate the
    // midpoints, which therefore read the extrapolated last value (this shifts
    // the final output forward by half a frame, as upstream does).
    const half = step >> 1;
    let nInt = half < T ? Math.ceil((T - half) / step) : 0;
    const at = (k, d) => xs[k * step * D + d]; // xs[::step][k]
    if (nInt === N) {
      const last = (half + (nInt - 1) * step) * D;
      for (let d = 0; d < D; d++) xs[last + d] = at(N - 1, d) + (at(N - 1, d) - at(N - 2, d)) / 2;
      nInt -= 1;
    }
    const mids = new Float64Array(Math.max(N - 1, 0) * D);
    for (let k = 0; k < N - 1; k++) for (let d = 0; d < D; d++) mids[k * D + d] = (at(k, d) + at(k + 1, d)) / 2;
    for (let k = 0; k < nInt; k++) for (let d = 0; d < D; d++) xs[(half + k * step) * D + d] = mids[k * D + d];

    if (step === 1) break;
    step >>= 1;
  }
  return Array.from({ length: T }, (_, i) => Array.from(xs.subarray(i * D, i * D + D)));
}

/**
 * get_smooth_root_pos for one sequence: rows [x, y, z]; x/z are smoothed with a
 * 0.06 m margin on every frame, y is kept.
 */
export function getSmoothRootPos(positions) {
  const xz = positions.map((p) => [p[0], p[2]]);
  const s = smoothSignal(xz, SMOOTH_ROOT_MARGIN);
  return positions.map((p, i) => [s[i][0], p[1], s[i][1]]);
}

/** Smooths a ground path [T][2] (x, z) exactly like get_smooth_root_pos. */
export function smoothPathXZ(points) {
  return smoothSignal(points, SMOOTH_ROOT_MARGIN);
}
