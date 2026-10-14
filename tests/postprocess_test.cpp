// Post-processing (NVIDIA MotionCorrection port), feature re-encoding and
// smooth-root tests. No GPU or model files needed.
#include "motion_decode.hpp"
#include "postprocess.hpp"
#include "skeleton.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <span>
#include <string>
#include <vector>

namespace {

using kimodo::constraint_set;
using kimodo::constraint_type;
using namespace kimodo::detail;

int failures = 0;
#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
            ++failures;                                                                \
        }                                                                              \
    } while (0)

using quat = std::array<float, 4>; // x, y, z, w
using mat = std::array<double, 9>;  // row-major, column-vector convention

quat axis_angle(double ax, double ay, double az, double angle) {
    const double n = std::sqrt(ax * ax + ay * ay + az * az), s = std::sin(angle / 2) / n;
    return {static_cast<float>(ax * s), static_cast<float>(ay * s), static_cast<float>(az * s),
            static_cast<float>(std::cos(angle / 2))};
}
mat to_matrix(const float *q) {
    const double x = q[0], y = q[1], z = q[2], w = q[3], s = 2. / (x * x + y * y + z * z + w * w);
    return {1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w), s * (x * y + z * w),
            1 - s * (x * x + z * z), s * (y * z - x * w), s * (x * z - y * w), s * (y * z + x * w),
            1 - s * (x * x + y * y)};
}
mat mul(const mat &a, const mat &b) {
    mat r{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) r[i * 3 + j] += a[i * 3 + k] * b[k * 3 + j];
    return r;
}

// Independent FK of one pose: global positions [J][3] and rotations [J].
struct posed {
    std::vector<std::array<double, 3>> position;
    std::vector<mat> rotation;
};
posed fk(const skeleton_spec &s, const float *local, const float *root) {
    const std::size_t J = s.joints();
    posed p{std::vector<std::array<double, 3>>(J), std::vector<mat>(J)};
    for (std::size_t j = 0; j < J; ++j) {
        const mat l = to_matrix(local + j * 4);
        if (s.parents[j] < 0) {
            p.rotation[j] = l;
            p.position[j] = {root[0], root[1], root[2]};
            continue;
        }
        const auto parent = static_cast<std::size_t>(s.parents[j]);
        const mat &g = p.rotation[parent];
        p.rotation[j] = mul(g, l);
        for (int r = 0; r < 3; ++r)
            p.position[j][r] = p.position[parent][r] + g[r * 3] * s.offsets[j][0] + g[r * 3 + 1] * s.offsets[j][1] +
                g[r * 3 + 2] * s.offsets[j][2];
    }
    return p;
}

struct motion {
    std::size_t frames = 0;
    std::vector<float> local, root;
};

// Smooth, plausible synthetic motion: small oscillating joint rotations and a
// root walking forward along +z with lateral sway.
motion synthetic(const skeleton_spec &s, std::size_t T, double phase = 0., double amplitude = .25) {
    const std::size_t J = s.joints();
    motion m{T, std::vector<float>(T * J * 4), std::vector<float>(T * 3)};
    const double root_y = make_motion_correction_rig(s).rest_translations[1];
    for (std::size_t t = 0; t < T; ++t) {
        const double u = static_cast<double>(t) / 30.;
        for (std::size_t j = 0; j < J; ++j) {
            const double fj = static_cast<double>(j);
            const quat q = axis_angle(std::sin(fj + 1), std::cos(fj * 1.3), .4, amplitude * std::sin(2. * u * (1 + j % 3) + fj + phase));
            std::copy(q.begin(), q.end(), m.local.begin() + static_cast<std::ptrdiff_t>((t * J + j) * 4));
        }
        m.root[t * 3] = static_cast<float>(.05 * std::sin(3. * u + phase));
        m.root[t * 3 + 1] = static_cast<float>(root_y - .03 + .01 * std::sin(6. * u));
        m.root[t * 3 + 2] = static_cast<float>(1.1 * u);
    }
    return m;
}

double quat_distance(const float *a, const float *b) { // sign-free
    double plus = 0, minus = 0;
    for (int k = 0; k < 4; ++k) {
        plus = std::max(plus, std::abs(static_cast<double>(a[k]) - b[k]));
        minus = std::max(minus, std::abs(static_cast<double>(a[k]) + b[k]));
    }
    return std::min(plus, minus);
}

constraint_set pose_constraint(constraint_type type, const skeleton_spec &s, const motion &m, std::size_t source,
                               unsigned frame) {
    const std::size_t J = s.joints();
    constraint_set c;
    c.type = type;
    c.frame_indices = {frame};
    c.local_rotations_xyzw.assign(m.local.begin() + static_cast<std::ptrdiff_t>(source * J * 4),
                                  m.local.begin() + static_cast<std::ptrdiff_t>((source + 1) * J * 4));
    c.root_positions.assign(m.root.begin() + static_cast<std::ptrdiff_t>(source * 3),
                            m.root.begin() + static_cast<std::ptrdiff_t>(source * 3 + 3));
    return c;
}

const std::array<const skeleton_spec *, 3> skeletons{&smplx22_spec, &soma30_spec, &g1skel34_spec};

void test_rig_constants() {
    // Upstream values (create_working_rig_from_skeleton, re-computed from the
    // upstream rig): rig root y and the contact
    // minHeights the solver derives from it with float32 sums (right foot, left
    // foot, right toe used for both toes). Root y equals upstream's float64 rig
    // value cast to float exactly, for all three skeletons.
    struct expected {
        const skeleton_spec *s;
        double root_y, right_foot, left_foot, toe;
    };
    const std::array<expected, 3> table{{{&soma30_spec, 1.008708477, .07095987, .07059473, .02016377},
                                         {&smplx22_spec, 1.023246646, .07228649, .07528615, .00699997},
                                         {&g1skel34_spec, .794193029, .04200006, .04200006, .00700003}}};
    for (const auto &e : table) {
        const auto rig = make_motion_correction_rig(*e.s);
        CHECK(rig.rest_translations[1] == static_cast<float>(e.root_y));
        CHECK(rig.rest_translations[0] == 0.F && rig.rest_translations[2] == 0.F);
        CHECK(rig.left_foot == static_cast<int>(e.s->end_effectors[0]) && rig.right_hand == static_cast<int>(e.s->end_effectors[3]));
        CHECK(rig.has_double_ankle_joints == (e.s == &g1skel34_spec));
#ifdef KIMODO_HAVE_MOTION_CORRECTION
        const auto heights = motion_correction_min_heights(*e.s);
        CHECK(heights.has_value());
        if (heights) {
            std::printf("rig %-8s root_y %.8f minHeight R %.8f L %.8f toe %.8f\n", std::string(e.s->key).c_str(),
                        rig.rest_translations[1], (*heights)[0], (*heights)[1], (*heights)[2]);
            CHECK(std::abs((*heights)[0] - e.right_foot) < 1e-8);
            CHECK(std::abs((*heights)[1] - e.left_foot) < 1e-8);
            CHECK(std::abs((*heights)[2] - e.toe) < 1e-8);
        }
#endif
    }
}

#ifdef KIMODO_HAVE_MOTION_CORRECTION
void test_identity() {
    for (const auto *s : skeletons) {
        const std::size_t T = 64, J = s->joints();
        const motion m = synthetic(*s, T);
        auto local = m.local, root = m.root;
        const std::vector<float> contacts(T * 4, 0.F);
        CHECK(post_process_motion(*s, T, local, root, contacts, {}).has_value());
        CHECK(root == m.root); // bit-exact: nothing pinned, no margins, no contacts
        double worst = 0;
        for (std::size_t i = 0; i < T * J; ++i) worst = std::max(worst, quat_distance(&local[i * 4], &m.local[i * 4]));
        std::printf("identity %-8s max rotation change %.3g\n", std::string(s->key).c_str(), worst);
        CHECK(worst < 1e-6);
    }
}

void test_fullbody_keyframe() {
    const auto &s = soma30_spec;
    const std::size_t T = 80, J = s.joints();
    const motion m = synthetic(s, T), other = synthetic(s, T, 1.7, .4);
    auto target = pose_constraint(constraint_type::fullbody, s, other, 52, 30);
    target.root_positions[0] += .2F;
    target.root_positions[2] -= .15F;
    auto local = m.local, root = m.root;
    CHECK(post_process_motion(s, T, local, root, {}, std::span(&target, 1)).has_value());
    double root_error = 0, rotation_error = 0;
    for (int k = 0; k < 3; ++k) root_error = std::max(root_error, std::abs(static_cast<double>(root[30 * 3 + k]) - target.root_positions[k]));
    for (std::size_t j = 0; j < J; ++j)
        rotation_error = std::max(rotation_error, quat_distance(&local[(30 * J + j) * 4], &target.local_rotations_xyzw[j * 4]));
    std::printf("fullbody keyframe: root error %.3g, rotation error %.3g\n", root_error, rotation_error);
    CHECK(root_error < 1e-5);
    CHECK(rotation_error < 1e-5);
    // Frames away from the keyframe stay close to the input motion.
    CHECK(std::abs(root[0] - m.root[0]) < .25F);
}

void test_root2d() {
    const auto &s = soma30_spec;
    const std::size_t T = 80;
    const motion m = synthetic(s, T);
    constraint_set c;
    c.type = constraint_type::root2d;
    c.frame_indices = {40, 200}; // 200 is past the end and must be ignored
    const float tx = m.root[40 * 3] + .3F, tz = m.root[40 * 3 + 2] - .2F;
    c.smooth_root_2d = {tx, tz, 9.F, 9.F};
    auto local = m.local, root = m.root;
    const float margin = .04F;
    CHECK(post_process_motion(s, T, local, root, {}, std::span(&c, 1), margin).has_value());
    const double before = std::hypot(m.root[40 * 3] - tx, m.root[40 * 3 + 2] - tz);
    const double after = std::hypot(root[40 * 3] - tx, root[40 * 3 + 2] - tz);
    std::printf("root2d: distance to target %.4f -> %.4f (margin %.2f)\n", before, after, margin);
    CHECK(after <= margin + 1e-3);
    CHECK(root[40 * 3 + 1] == m.root[40 * 3 + 1] || std::abs(root[40 * 3 + 1] - m.root[40 * 3 + 1]) < 1e-6); // y untouched
}

// Spread of a joint's x,z positions over [first, last]: max distance to the mean.
double xz_spread(const skeleton_spec &s, const std::vector<float> &local, const std::vector<float> &root,
                 std::size_t joint, std::size_t first, std::size_t last) {
    const std::size_t J = s.joints();
    std::vector<std::array<double, 2>> xz;
    for (std::size_t t = first; t <= last; ++t) {
        const auto p = fk(s, &local[t * J * 4], &root[t * 3]);
        xz.push_back({p.position[joint][0], p.position[joint][2]});
    }
    double mx = 0, mz = 0;
    for (const auto &v : xz) mx += v[0], mz += v[1];
    mx /= static_cast<double>(xz.size());
    mz /= static_cast<double>(xz.size());
    double spread = 0;
    for (const auto &v : xz) spread = std::max(spread, std::hypot(v[0] - mx, v[1] - mz));
    return spread;
}

void test_foot_contact() {
    // Rest pose with slightly bent-down root height, root sliding 1 cm/frame in
    // x: the feet slide. A left-foot contact interval must plant the left foot.
    const auto &s = soma30_spec;
    const std::size_t T = 60, J = s.joints(), left_foot = s.end_effectors[0];
    const float root_y = make_motion_correction_rig(s).rest_translations[1] - .05F;
    std::vector<float> local(T * J * 4, 0.F), root(T * 3), contacts(T * 4, 0.F);
    for (std::size_t i = 3; i < local.size(); i += 4) local[i] = 1.F;
    for (std::size_t t = 0; t < T; ++t) {
        root[t * 3] = .01F * static_cast<float>(t);
        root[t * 3 + 1] = root_y;
        root[t * 3 + 2] = 0.F;
    }
    const std::size_t first = 20, last = 31;
    for (std::size_t t = first; t <= last; ++t) contacts[t * 4] = contacts[t * 4 + 1] = .9F; // binarized to 1
    const double before = xz_spread(s, local, root, left_foot, first, last);
    auto fixed_local = local, fixed_root = root;
    CHECK(post_process_motion(s, T, fixed_local, fixed_root, contacts, {}).has_value());
    const double after = xz_spread(s, fixed_local, fixed_root, left_foot, first, last);
    const double right_after = xz_spread(s, fixed_local, fixed_root, s.end_effectors[1], first, last);
    std::printf("foot contact: left foot xz spread %.4f -> %.4f m (right foot, no contact: %.4f)\n", before, after, right_after);
    CHECK(after < .25 * before);
    CHECK(right_after > .5 * before);
    // Raw contacts below 0.5 are not contacts (binarization), so nothing moves.
    for (std::size_t t = first; t <= last; ++t) contacts[t * 4] = contacts[t * 4 + 1] = .3F;
    auto weak_local = local, weak_root = root;
    CHECK(post_process_motion(s, T, weak_local, weak_root, contacts, {}).has_value());
    CHECK(std::abs(xz_spread(s, weak_local, weak_root, left_foot, first, last) - before) < 1e-6);
}

void test_effectors() {
    const auto &s = soma30_spec;
    const std::size_t T = 70, J = s.joints(), hand = s.end_effectors[2], frame = 35;
    const motion m = synthetic(s, T);
    // Target: the same frame with the left forearm bent further (reachable);
    // smooth_root_2d defaults to the pelvis.
    auto c = pose_constraint(constraint_type::left_hand, s, m, frame, static_cast<unsigned>(frame));
    const quat bend = axis_angle(0, 1, 0, .5);
    float *forearm = &c.local_rotations_xyzw[12 * 4];
    const quat q{forearm[0], forearm[1], forearm[2], forearm[3]};
    const quat r{bend[3] * q[0] + bend[0] * q[3] + bend[1] * q[2] - bend[2] * q[1],
                 bend[3] * q[1] - bend[0] * q[2] + bend[1] * q[3] + bend[2] * q[0],
                 bend[3] * q[2] + bend[0] * q[1] - bend[1] * q[0] + bend[2] * q[3],
                 bend[3] * q[3] - bend[0] * q[0] - bend[1] * q[1] - bend[2] * q[2]};
    std::copy(r.begin(), r.end(), forearm);
    const auto target = fk(s, c.local_rotations_xyzw.data(), c.root_positions.data());
    auto local = m.local, root = m.root;
    CHECK(post_process_motion(s, T, local, root, {}, std::span(&c, 1)).has_value());
    const auto got = fk(s, &local[frame * J * 4], &root[frame * 3]);
    const auto was = fk(s, &m.local[frame * J * 4], &m.root[frame * 3]);
    double error = 0, start = 0, rotation = 0;
    for (int k = 0; k < 3; ++k) {
        error = std::max(error, std::abs(got.position[hand][k] - target.position[hand][k]));
        start = std::max(start, std::abs(was.position[hand][k] - target.position[hand][k]));
    }
    for (int k = 0; k < 9; ++k) rotation = std::max(rotation, std::abs(got.rotation[hand][k] - target.rotation[hand][k]));
    std::printf("left-hand pin: hand error %.4f -> %.3g m, global rotation error %.3g\n", start, error, rotation);
    CHECK(start > .05);
    CHECK(error < 1e-3);
    CHECK(rotation < 1e-3);

    // The generic end-effector type contributes targets but no mask: no change
    // compared to running without constraints.
    auto generic = c;
    generic.type = constraint_type::end_effector;
    generic.joint_names = {"LeftHand"};
    auto a_local = m.local, a_root = m.root, b_local = m.local, b_root = m.root;
    CHECK(post_process_motion(s, T, a_local, a_root, {}, std::span(&generic, 1)).has_value());
    CHECK(post_process_motion(s, T, b_local, b_root, {}, {}).has_value());
    CHECK(a_local == b_local && a_root == b_root);

    // Full-body sets are applied after effector sets, whatever the list order:
    // the full-body root wins at a shared frame.
    auto fullbody = pose_constraint(constraint_type::fullbody, s, m, frame, static_cast<unsigned>(frame));
    fullbody.root_positions[0] -= .1F;
    auto effector = c;
    effector.smooth_root_2d = {c.root_positions[0] + .3F, c.root_positions[2]};
    const std::array<constraint_set, 2> both{fullbody, effector};
    local = m.local;
    root = m.root;
    CHECK(post_process_motion(s, T, local, root, {}, both).has_value());
    CHECK(std::abs(root[frame * 3] - fullbody.root_positions[0]) < 1e-5);
}

void test_errors() {
    const auto &s = soma30_spec;
    motion m = synthetic(s, 1);
    CHECK(!post_process_motion(s, 1, m.local, m.root, {}, {}).has_value());
    m = synthetic(s, 10);
    constraint_set bad;
    bad.type = constraint_type::fullbody;
    bad.frame_indices = {1};
    bad.root_positions = {0, 1, 0};
    CHECK(!post_process_motion(s, 10, m.local, m.root, {}, std::span(&bad, 1)).has_value());
    const std::vector<float> contacts(7, 0.F);
    CHECK(!post_process_motion(s, 10, m.local, m.root, contacts, {}).has_value());
}
#endif

void test_encode_decode() {
    for (const auto *s : skeletons) {
        const std::size_t T = 40, J = s->joints(), D = s->motion_dim();
        const motion m = synthetic(*s, T);
        const auto features = encode_motion(*s, m.local, m.root, T);
        CHECK(features.has_value() && features->size() == T * D);
        if (!features) continue;
        const auto smooth = smooth_root_positions(m.root, T);
        double rot6 = 0, pos = 0, head = 0, vel = 0, smooth_error = 0;
        std::vector<posed> poses;
        for (std::size_t t = 0; t < T; ++t) poses.push_back(fk(*s, &m.local[t * J * 4], &m.root[t * 3]));
        for (std::size_t t = 0; t < T; ++t) {
            const float *f = features->data() + t * D;
            const auto &p = poses[t];
            CHECK(f[1] == m.root[t * 3 + 1]);
            smooth_error = std::max({smooth_error, std::abs(static_cast<double>(f[0]) - smooth[t * 3]),
                                     std::abs(static_cast<double>(f[2]) - smooth[t * 3 + 2])});
            const double angle = std::atan2(p.position[s->hips[0]][2] - p.position[s->hips[1]][2],
                                            -(p.position[s->hips[0]][0] - p.position[s->hips[1]][0]));
            head = std::max({head, std::abs(f[3] - std::cos(angle)), std::abs(f[4] - std::sin(angle))});
            for (std::size_t j = 0; j < J; ++j) {
                const float *r6 = f + 5 + 3 * J + j * 6;
                const mat &g = p.rotation[j];
                const std::array<double, 6> cols{g[0], g[3], g[6], g[1], g[4], g[7]};
                for (int k = 0; k < 6; ++k) rot6 = std::max(rot6, std::abs(r6[k] - cols[k]));
                const float *lp = f + 5 + j * 3;
                pos = std::max({pos, std::abs(lp[0] + f[0] - p.position[j][0]), std::abs(lp[1] - p.position[j][1]),
                                std::abs(lp[2] + f[2] - p.position[j][2])});
                const std::size_t from = t + 1 < T ? t : t - 1; // last velocity repeats the previous one
                for (int k = 0; k < 3; ++k)
                    vel = std::max(vel, std::abs(f[5 + 9 * J + j * 3 + k] -
                                                 30. * (poses[from + 1].position[j][k] - poses[from].position[j][k])));
            }
        }
        std::printf("encode %-8s: 6D %.2g, positions %.2g, heading %.2g, velocity %.2g, smooth %.2g\n",
                    std::string(s->key).c_str(), rot6, pos, head, vel, smooth_error);
        CHECK(rot6 < 1e-6 && pos < 1e-5 && head < 1e-5 && vel < 1e-3 && smooth_error == 0.);
        // kimodo.cpp's decoder (upstream inverse) gives back the root positions,
        // the local rotations, the smooth root and the contacts.
        const auto decoded = decode_motion_rows(*features, T, *s);
        CHECK(decoded.has_value());
        if (!decoded) continue;
        double root_error = 0, rotation_error = 0;
        for (std::size_t i = 0; i < T * 3; ++i) root_error = std::max(root_error, std::abs(static_cast<double>(decoded->root_positions[i]) - m.root[i]));
        for (std::size_t i = 0; i < T * J; ++i)
            rotation_error = std::max(rotation_error, quat_distance(&decoded->local_xyzw[i * 4], &m.local[i * 4]));
        std::printf("decode %-8s: root %.2g, rotations %.2g\n", std::string(s->key).c_str(), root_error, rotation_error);
        CHECK(root_error < 1e-5 && rotation_error < 1e-5);
        CHECK(decoded->smooth_root == smooth);
        for (std::size_t t = 0; t < T; ++t)
            for (std::size_t c = 0; c < 4; ++c) CHECK(decoded->foot_contacts[t * 4 + c] == (*features)[t * D + D - 4 + c]);
    }
    // Contacts: standing still at rest height -> all four in contact; lifted
    // by 20 cm -> none; T < 5 cannot be encoded (smooth root).
    const auto &s = soma30_spec;
    const std::size_t T = 8, J = s.joints(), D = s.motion_dim();
    std::vector<float> local(T * J * 4, 0.F), root(T * 3, 0.F);
    for (std::size_t i = 3; i < local.size(); i += 4) local[i] = 1.F;
    for (const float lift : {0.F, .2F}) {
        for (std::size_t t = 0; t < T; ++t) root[t * 3 + 1] = make_motion_correction_rig(s).rest_translations[1] + lift;
        const auto f = encode_motion(s, local, root, T);
        CHECK(f.has_value());
        if (f)
            for (std::size_t t = 0; t < T; ++t)
                for (std::size_t c = 0; c < 4; ++c) CHECK((*f)[t * D + 5 + 12 * J + c] == (lift == 0.F ? 1.F : 0.F));
    }
    CHECK(!encode_motion(s, std::span(local).first(4 * J * 4), std::span(root).first(12), 4).has_value());
}

void test_smooth_root() {
    // Straight constant-velocity line: any straight line within the 6 cm
    // margin has zero acceleration, so the output is a straight line within
    // the margin (ADMM started from the mean does not return the input itself),
    // shifted forward by half a frame by upstream's final interpolation (the
    // second-to-last sample reads the extrapolated last one).
    const std::size_t T = 100;
    std::vector<float> line(T * 3);
    const auto at = [](double t) { return std::array<double, 2>{.02 * t - .3, .5 + .013 * t}; };
    for (std::size_t t = 0; t < T; ++t) {
        const auto p = at(static_cast<double>(t));
        line[t * 3] = static_cast<float>(p[0]);
        line[t * 3 + 1] = .9F + .01F * static_cast<float>(t % 7);
        line[t * 3 + 2] = static_cast<float>(p[1]);
    }
    const auto out = smooth_root_positions(line, T);
    double worst = 0, bend = 0;
    for (std::size_t t = 0; t < T; ++t) {
        const double shift = t + 2 < T ? .5 : t + 1 < T ? .75 : .5;
        const auto p = at(static_cast<double>(t) + shift);
        worst = std::max(worst, std::hypot(out[t * 3] - p[0], out[t * 3 + 2] - p[1]));
        CHECK(out[t * 3 + 1] == line[t * 3 + 1]);
        if (t >= 1 && t + 3 < T)
            for (const std::size_t k : {std::size_t{0}, std::size_t{2}})
                bend = std::max(bend, std::abs(static_cast<double>(out[(t - 1) * 3 + k]) - 2. * out[t * 3 + k] + out[(t + 1) * 3 + k]));
    }
    std::printf("smooth root, straight line: max distance to line(t + 1/2) %.4f m, max second difference %.2g m\n", worst, bend);
    CHECK(worst < .06 + 2e-3); // 500 ADMM iterations: the margin is met approximately
    CHECK(bend < 1e-4);

    // Zigzag of +-5 cm around a straight walk: within the 6 cm margin of the
    // (half-frame shifted) input, and much flatter than the input.
    const std::size_t Z = 150;
    std::vector<float> zig(Z * 3);
    for (std::size_t t = 0; t < Z; ++t) {
        const double phase = static_cast<double>(t % 20) / 20.;
        zig[t * 3] = static_cast<float>(.03 * static_cast<double>(t));
        zig[t * 3 + 1] = 1.F;
        zig[t * 3 + 2] = static_cast<float>(.05 * (phase < .5 ? 4 * phase - 1 : 3 - 4 * phase));
    }
    const auto smoothed = smooth_root_positions(zig, Z);
    double margin = 0, amplitude = 0;
    for (std::size_t t = 0; t + 1 < Z; ++t) {
        const double mx = (zig[t * 3] + zig[t * 3 + 3]) / 2., mz = (zig[t * 3 + 2] + zig[t * 3 + 5]) / 2.;
        margin = std::max(margin, std::hypot(smoothed[t * 3] - mx, smoothed[t * 3 + 2] - mz));
        if (t > 10 && t + 10 < Z) amplitude = std::max(amplitude, std::abs(static_cast<double>(smoothed[t * 3 + 2])));
    }
    std::printf("smooth root, zigzag: max distance to shifted input %.4f m, residual |z| %.4f m (input 0.05)\n", margin, amplitude);
    CHECK(margin < .06 + 2e-3);
    CHECK(amplitude < .02);

    // Too short for upstream's solver: returned unchanged.
    const std::vector<float> tiny{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    CHECK(smooth_root_positions(tiny, 4) == tiny);
}

} // namespace

int main() {
    test_rig_constants();
#ifdef KIMODO_HAVE_MOTION_CORRECTION
    test_identity();
    test_fullbody_keyframe();
    test_root2d();
    test_foot_contact();
    test_effectors();
    test_errors();
#else
    {
        motion m = synthetic(soma30_spec, 10);
        if (post_process_motion(soma30_spec, 10, m.local, m.root, {}, {}).has_value()) ++failures;
        std::puts("motion correction unavailable in this build: solver tests skipped");
    }
#endif
    test_encode_decode();
    test_smooth_root();
    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::puts("postprocess tests passed");
    return 0;
}
