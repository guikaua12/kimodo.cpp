// Ports of NVIDIA Kimodo's motion-feature encoder (KimodoMotionRep.__call__,
// kimodo/motion_rep/reps/kimodo_motionrep.py:52-111) and its smooth-root
// solver (kimodo/motion_rep/smooth_root.py), used to re-encode post-processed
// motion.
#include "postprocess.hpp"

#include "skeleton.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace kimodo::detail {
namespace {

// Cholesky factor of a symmetric positive-definite pentadiagonal matrix, used
// for TrajectorySmoother's (A^T A + rho I) system (upstream: scipy splu; the
// factorizations agree to ~1e-15 relative, below the float32 storage).
class pentadiagonal_cholesky {
public:
    // diag[i] = M[i][i], sub1[i] = M[i][i-1], sub2[i] = M[i][i-2].
    pentadiagonal_cholesky(const std::vector<double> &diag, const std::vector<double> &sub1,
                           const std::vector<double> &sub2)
        : n_(diag.size()), d_(n_), l1_(n_), l2_(n_) {
        for (std::size_t i = 0; i < n_; ++i) {
            l2_[i] = i >= 2 ? sub2[i] / d_[i - 2] : 0.;
            l1_[i] = i >= 1 ? (sub1[i] - (i >= 2 ? l2_[i] * l1_[i - 1] : 0.)) / d_[i - 1] : 0.;
            d_[i] = std::sqrt(diag[i] - l2_[i] * l2_[i] - l1_[i] * l1_[i]);
        }
    }
    void solve(std::vector<double> &x) const {
        for (std::size_t i = 0; i < n_; ++i) {
            double v = x[i];
            if (i >= 1) v -= l1_[i] * x[i - 1];
            if (i >= 2) v -= l2_[i] * x[i - 2];
            x[i] = v / d_[i];
        }
        for (std::size_t i = n_; i-- > 0;) {
            double v = x[i];
            if (i + 1 < n_) v -= l1_[i + 1] * x[i + 1];
            if (i + 2 < n_) v -= l2_[i + 2] * x[i + 2];
            x[i] = v / d_[i];
        }
    }

private:
    std::size_t n_;
    std::vector<double> d_, l1_, l2_; // L[i][i], L[i][i-1], L[i][i-2]
};

// TrajectorySmoother(margins = 0.06, pos_weight = 0, loop = False,
// admm_iters = 500, alpha_overrelax = 1.8).smooth(targets, x0) on N samples of
// 2-D points. Storage dtypes follow upstream under NumPy 2: x, z, u and the
// targets are float32 (torch float32 input); the right-hand side
// admm_stepsize * (z - u) is float64 (admm_stepsize is a NumPy float64 scalar)
// and the solve runs in float64.
void trajectory_smooth(std::vector<float> &x, const std::vector<float> &targets, std::size_t n) {
    constexpr double margin = .06;
    constexpr float alpha = 1.8F;
    constexpr int iterations = 500;
    // A has rows i = 1..N-2 equal to [-1, 2, -1] at columns i-1, i, i+1; rows 0
    // and N-1 are empty. M0 = A^T A.
    std::vector<double> diag(n, 0.), sub1(n, 0.), sub2(n, 0.);
    constexpr std::array<double, 3> stencil{-1., 2., -1.};
    for (std::size_t row = 1; row + 1 < n; ++row)
        for (std::size_t a = 0; a < 3; ++a)
            for (std::size_t b = 0; b <= a; ++b) {
                const std::size_t i = row - 1 + a, gap = a - b;
                (gap == 0 ? diag : gap == 1 ? sub1 : sub2)[i] += stencil[a] * stencil[b];
            }
    const double diag_max = *std::ranges::max_element(diag);
    const double rho = .25 * std::sqrt(diag_max); // admm_stepsize
    for (double &d : diag) d += rho;
    const pentadiagonal_cholesky system(diag, sub1, sub2);

    std::vector<float> z(n * 2, 0.F), u(n * 2, 0.F);
    std::vector<double> rhs(n);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        // z = x + u - t; project rows with |z| > margin onto the margin; z += t.
        for (std::size_t i = 0; i < n; ++i) {
            float *zi = z.data() + i * 2;
            for (std::size_t k = 0; k < 2; ++k) zi[k] = (x[i * 2 + k] + u[i * 2 + k]) - targets[i * 2 + k];
            const float norm = std::sqrt(zi[0] * zi[0] + zi[1] * zi[1]);
            if (static_cast<double>(norm) > margin) {
                const double scale = margin / static_cast<double>(norm);
                for (std::size_t k = 0; k < 2; ++k) zi[k] = static_cast<float>(static_cast<double>(zi[k]) * scale);
            }
            for (std::size_t k = 0; k < 2; ++k) zi[k] += targets[i * 2 + k];
        }
        // u += alpha * (x - z)
        for (std::size_t i = 0; i < n * 2; ++i) u[i] += alpha * (x[i] - z[i]);
        // x = M^-1 (pos_weight * t + rho * (z - u)), pos_weight = 0.
        for (std::size_t k = 0; k < 2; ++k) {
            for (std::size_t i = 0; i < n; ++i) rhs[i] = rho * static_cast<double>(z[i * 2 + k] - u[i * 2 + k]);
            system.solve(rhs);
            for (std::size_t i = 0; i < n; ++i) x[i * 2 + k] = static_cast<float>(rhs[i]);
        }
    }
}

// smooth_signal(x, margins = 0.06) for a [T,2] float32 signal, T >= 5.
std::vector<float> smooth_signal(const std::vector<float> &x, std::size_t frames) {
    // x_smoothed[:] = x.mean(axis=0): float32 accumulation along the strided
    // axis, divided by the integer count.
    std::array<float, 2> sum{0.F, 0.F};
    for (std::size_t t = 0; t < frames; ++t)
        for (std::size_t k = 0; k < 2; ++k) sum[k] += x[t * 2 + k];
    std::vector<float> smoothed(frames * 2);
    for (std::size_t t = 0; t < frames; ++t)
        for (std::size_t k = 0; k < 2; ++k)
            smoothed[t * 2 + k] = static_cast<float>(static_cast<double>(sum[k]) / static_cast<double>(frames));

    // Multigrid: levels = max(floor(log2 T) - 4, 1), from coarse to full rate.
    std::size_t log2 = 0;
    while ((std::size_t{2} << log2) <= frames) ++log2;
    std::size_t step = std::size_t{1} << std::max<std::size_t>(log2 >= 4 ? log2 - 4 : 0, 1);
    std::vector<float> sub, targets, interpolated;
    for (;;) {
        const std::size_t n = (frames + step - 1) / step;
        sub.resize(n * 2);
        targets.resize(n * 2);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t k = 0; k < 2; ++k) {
                sub[i * 2 + k] = smoothed[i * step * 2 + k];
                targets[i * 2 + k] = x[i * step * 2 + k];
            }
        trajectory_smooth(sub, targets, n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t k = 0; k < 2; ++k) smoothed[i * step * 2 + k] = sub[i * 2 + k];

        // Interpolate to the next level. At the last level (step 1) the
        // "interleaved" slots x_smoothed[0::1] alias x_smoothed[::1]: upstream
        // first overwrites the last sample with the extrapolation and then
        // averages neighbours from the updated array, which shifts the output
        // forward by half a frame (and the second-to-last sample reads the
        // extrapolated last one). Reproduced as-is.
        const std::size_t half = step / 2;
        std::size_t interleaved = half < frames ? (frames - half + step - 1) / step : 0;
        if (interleaved == n) {
            const std::size_t last = (n - 1) * step, before = (n - 2) * step;
            for (std::size_t k = 0; k < 2; ++k) {
                const float a = smoothed[last * 2 + k], b = smoothed[before * 2 + k];
                smoothed[(half + (interleaved - 1) * step) * 2 + k] = a + (a - b) / 2.F;
            }
            --interleaved;
        }
        interpolated.resize(interleaved * 2);
        for (std::size_t i = 0; i < interleaved; ++i)
            for (std::size_t k = 0; k < 2; ++k)
                interpolated[i * 2 + k] = (smoothed[i * step * 2 + k] + smoothed[(i + 1) * step * 2 + k]) / 2.F;
        for (std::size_t i = 0; i < interleaved; ++i)
            for (std::size_t k = 0; k < 2; ++k) smoothed[(half + i * step) * 2 + k] = interpolated[i * 2 + k];
        if (step == 1) break;
        step /= 2;
    }
    return smoothed;
}

struct mat3 {
    std::array<double, 9> v{}; // row-major, column-vector convention
};
mat3 mul(const mat3 &a, const mat3 &b) {
    mat3 r;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            for (std::size_t k = 0; k < 3; ++k) r.v[i * 3 + j] += a.v[i * 3 + k] * b.v[k * 3 + j];
    return r;
}
// Upstream quaternion_to_matrix (geometry.py:190-216), here for x, y, z, w;
// tolerates non-unit quaternions like upstream (two_s = 2 / |q|^2).
mat3 quaternion_matrix(const float *q) {
    const double x = q[0], y = q[1], z = q[2], w = q[3];
    const double s = 2. / (x * x + y * y + z * z + w * w);
    return {{1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w),
             s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w),
             s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)}};
}

} // namespace

std::vector<float> smooth_root_positions(std::span<const float> root_positions, std::size_t frames) {
    std::vector<float> out(root_positions.begin(), root_positions.end());
    if (frames <= 4 || root_positions.size() != frames * 3) return out; // upstream: singular coarse solve
    std::vector<float> xz(frames * 2);
    for (std::size_t t = 0; t < frames; ++t) {
        xz[t * 2] = root_positions[t * 3];
        xz[t * 2 + 1] = root_positions[t * 3 + 2];
    }
    const auto smoothed = smooth_signal(xz, frames);
    for (std::size_t t = 0; t < frames; ++t) {
        out[t * 3] = smoothed[t * 2];
        out[t * 3 + 2] = smoothed[t * 2 + 1];
    }
    return out;
}

std::expected<std::vector<float>, std::string> encode_motion(
    const skeleton_spec &s, std::span<const float> local_xyzw, std::span<const float> root_positions,
    std::size_t frames, float fps) {
    const std::size_t J = s.joints(), D = s.motion_dim();
    if (local_xyzw.size() != frames * J * 4 || root_positions.size() != frames * 3)
        return std::unexpected("invalid motion size for feature encoding");
    if (frames < 5) return std::unexpected("feature encoding requires at least 5 frames (smooth root)");
    if (!(fps > 0.F) || !std::isfinite(fps)) return std::unexpected("invalid frame rate for feature encoding");
    for (std::size_t j = 0; j < J; ++j)
        if (j == 0 ? s.parents[j] != -1 : s.parents[j] < 0 || static_cast<std::size_t>(s.parents[j]) >= j)
            return std::unexpected("feature encoding requires a root-first topological skeleton");

    // Forward kinematics (kimodo/skeleton/kinematics.py): global rotations and
    // positions relative to the pelvis, which sits at the origin.
    std::vector<mat3> global(frames * J);
    std::vector<std::array<double, 3>> relative(frames * J), posed(frames * J);
    for (std::size_t t = 0; t < frames; ++t) {
        mat3 *g = global.data() + t * J;
        auto *rel = relative.data() + t * J;
        for (std::size_t j = 0; j < J; ++j) {
            const mat3 local = quaternion_matrix(local_xyzw.data() + (t * J + j) * 4);
            if (j == 0) {
                g[0] = local;
                rel[0] = {0., 0., 0.};
            } else {
                const auto p = static_cast<std::size_t>(s.parents[j]);
                g[j] = mul(g[p], local);
                for (std::size_t r = 0; r < 3; ++r)
                    rel[j][r] = rel[p][r] + g[p].v[r * 3] * s.offsets[j][0] + g[p].v[r * 3 + 1] * s.offsets[j][1] +
                        g[p].v[r * 3 + 2] * s.offsets[j][2];
            }
            for (std::size_t r = 0; r < 3; ++r)
                posed[t * J + j][r] = rel[j][r] + static_cast<double>(root_positions[t * 3 + r]);
        }
    }
    const std::vector<float> smooth = smooth_root_positions(root_positions, frames);

    std::vector<float> out(frames * D);
    const std::size_t positions_begin = 5, rotations_begin = 5 + 3 * J, velocities_begin = 5 + 9 * J,
                      contacts_begin = 5 + 12 * J;
    const std::array<std::size_t, 4> contact_joints{static_cast<std::size_t>(s.chains[0][0]),
                                                    static_cast<std::size_t>(s.chains[0][1]),
                                                    static_cast<std::size_t>(s.chains[1][0]),
                                                    static_cast<std::size_t>(s.chains[1][1])};
    for (std::size_t t = 0; t < frames; ++t) {
        float *row = out.data() + t * D;
        const auto *p = posed.data() + t * J;
        const float *root = root_positions.data() + t * 3;
        // smooth_root_pos = (x_s, root y, z_s)
        std::copy_n(smooth.data() + t * 3, 3, row);
        // global_root_heading from the hips (right - left)
        const double dx = p[s.hips[0]][0] - p[s.hips[1]][0], dz = p[s.hips[0]][2] - p[s.hips[1]][2];
        const double heading = std::atan2(dz, -dx);
        row[3] = static_cast<float>(std::cos(heading));
        row[4] = static_cast<float>(std::sin(heading));
        // local_joints_positions = pelvis-relative FK + (root - smooth) in x,z
        // and the root height in y.
        const std::array<double, 3> hips_offset{static_cast<double>(root[0] - smooth[t * 3]), static_cast<double>(root[1]),
                                                static_cast<double>(root[2] - smooth[t * 3 + 2])};
        for (std::size_t j = 0; j < J; ++j)
            for (std::size_t r = 0; r < 3; ++r)
                row[positions_begin + j * 3 + r] = static_cast<float>(relative[t * J + j][r] + hips_offset[r]);
        // global_rot_data: first two columns of each global rotation matrix
        for (std::size_t j = 0; j < J; ++j) {
            const auto &g = global[t * J + j].v;
            float *r6 = row + rotations_begin + j * 6;
            r6[0] = static_cast<float>(g[0]);
            r6[1] = static_cast<float>(g[3]);
            r6[2] = static_cast<float>(g[6]);
            r6[3] = static_cast<float>(g[1]);
            r6[4] = static_cast<float>(g[4]);
            r6[5] = static_cast<float>(g[7]);
        }
        // velocities: fps * (p[t+1] - p[t]); the last frame repeats the previous one
        const std::size_t from = t + 1 < frames ? t : t - 1;
        for (std::size_t j = 0; j < J; ++j)
            for (std::size_t r = 0; r < 3; ++r)
                row[velocities_begin + j * 3 + r] =
                    static_cast<float>(static_cast<double>(fps) * (posed[(from + 1) * J + j][r] - posed[from * J + j][r]));
        // foot contacts: |velocity| < 0.15 m/s and height < 0.10 m on
        // [left foot, left toe, right foot, right toe]
        for (std::size_t c = 0; c < 4; ++c) {
            const float *v = row + velocities_begin + contact_joints[c] * 3;
            const float speed = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            const auto height = static_cast<float>(p[contact_joints[c]][1]);
            row[contacts_begin + c] = speed < .15F && height < .10F ? 1.F : 0.F;
        }
    }
    return out;
}

} // namespace kimodo::detail
