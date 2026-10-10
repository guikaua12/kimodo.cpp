// Port of NVIDIA Kimodo kimodo/postprocess.py (post_process_motion,
// extract_input_motion_from_constraints, create_working_rig_from_skeleton) on
// top of the vendored MotionCorrection solver (src/motion_correction_glue.hpp).
#include "postprocess.hpp"

#include "skeleton.hpp"

#ifdef KIMODO_HAVE_MOTION_CORRECTION
#include "motion_correction_glue.hpp"
#endif

#include <algorithm>
#include <charconv>
#include <cmath>
#include <system_error>
#include <string_view>

namespace kimodo::detail {
namespace {

std::string_view type_name(constraint_type type) noexcept {
    switch (type) {
    case constraint_type::root2d: return "root2d";
    case constraint_type::fullbody: return "fullbody";
    case constraint_type::left_hand: return "left-hand";
    case constraint_type::right_hand: return "right-hand";
    case constraint_type::left_foot: return "left-foot";
    case constraint_type::right_foot: return "right-foot";
    case constraint_type::end_effector: return "end-effector";
    }
    return "unknown";
}

std::expected<void, std::string> check_constraint(const constraint_set &c, std::size_t joints) {
    const std::size_t K = c.frame_indices.size();
    const auto fail = [&](std::string_view what) {
        return std::unexpected(std::string("invalid ") + std::string(type_name(c.type)) + " constraint: " + std::string(what));
    };
    if (c.type == constraint_type::root2d) {
        if (c.smooth_root_2d.size() != K * 2) return fail("smooth_root_2d must be [K,2]");
        return {};
    }
    if (c.local_rotations_xyzw.size() != K * joints * 4) return fail("local rotations must be [K,J,4]");
    if (c.root_positions.size() != K * 3) return fail("root_positions must be [K,3]");
    if (!c.smooth_root_2d.empty() && c.smooth_root_2d.size() != K * 2) return fail("smooth_root_2d must be [K,2]");
    return {};
}

#ifdef KIMODO_HAVE_MOTION_CORRECTION
struct solver_inputs {
    // Upstream mask dict (postprocess.py:218-252), each [T].
    std::vector<float> full_body, left_hand, right_hand, left_foot, right_foot, root;
    std::vector<float> target_root;       // [T*3], zeros where unconstrained
    std::vector<float> target_local_xyzw; // [T*J*4], identity where unconstrained
};

// extract_input_motion_from_constraints (postprocess.py:27-114) and the mask
// dict (218-252) for one sequence of `frames` frames.
solver_inputs build_solver_inputs(std::span<const constraint_set> constraints, std::size_t frames, std::size_t joints) {
    solver_inputs in;
    for (auto *mask : {&in.full_body, &in.left_hand, &in.right_hand, &in.left_foot, &in.right_foot, &in.root})
        mask->assign(frames, 0.F);
    in.target_root.assign(frames * 3, 0.F);
    in.target_local_xyzw.assign(frames * joints * 4, 0.F);
    for (std::size_t i = 3; i < in.target_local_xyzw.size(); i += 4) in.target_local_xyzw[i] = 1.F;

    // Masks by constraint name; the generic end-effector set has none.
    for (const auto &c : constraints) {
        std::vector<float> *mask = nullptr;
        switch (c.type) {
        case constraint_type::fullbody: mask = &in.full_body; break;
        case constraint_type::left_foot: mask = &in.left_foot; break;
        case constraint_type::right_foot: mask = &in.right_foot; break;
        case constraint_type::left_hand: mask = &in.left_hand; break;
        case constraint_type::right_hand: mask = &in.right_hand; break;
        case constraint_type::root2d: mask = &in.root; break;
        case constraint_type::end_effector: break;
        }
        if (mask)
            for (const unsigned frame : c.frame_indices)
                if (frame < frames) (*mask)[frame] = 1.F;
    }

    // Targets: sorted(key=isinstance(c, FullBodyConstraintSet)) is a stable
    // partition, so full-body sets are applied last and win on shared frames.
    std::vector<const constraint_set *> order;
    order.reserve(constraints.size());
    for (const auto &c : constraints) order.push_back(&c);
    std::stable_partition(order.begin(), order.end(),
                          [](const constraint_set *c) { return c->type != constraint_type::fullbody; });
    for (const constraint_set *c : order) {
        // Effector sets whose joints do not include "Hips" take the root x,z
        // from smooth_root_2d (which defaults to the pelvis x,z), since the
        // model conditioning does not pin the hips themselves. The four sided
        // types never include Hips (upstream class-level joint_names).
        const bool smooth_xz = c->type != constraint_type::fullbody && c->type != constraint_type::root2d &&
            !(c->type == constraint_type::end_effector &&
              std::ranges::any_of(c->joint_names, [](const std::string &name) { return name == "Hips"; })) &&
            !c->smooth_root_2d.empty();
        for (std::size_t k = 0; k < c->frame_indices.size(); ++k) {
            const std::size_t frame = c->frame_indices[k];
            if (frame >= frames) continue;
            float *hip = in.target_root.data() + frame * 3;
            if (c->type == constraint_type::root2d) {
                hip[0] = c->smooth_root_2d[k * 2];
                hip[2] = c->smooth_root_2d[k * 2 + 1];
                continue;
            }
            std::copy_n(c->root_positions.data() + k * 3, 3, hip);
            if (smooth_xz) {
                hip[0] = c->smooth_root_2d[k * 2];
                hip[2] = c->smooth_root_2d[k * 2 + 1];
            }
            std::copy_n(c->local_rotations_xyzw.data() + k * joints * 4, joints * 4,
                        in.target_local_xyzw.data() + frame * joints * 4);
        }
    }
    return in;
}

mc::rig solver_rig(const motion_correction_rig &rig) noexcept {
    mc::rig r;
    r.joints = rig.parents.size();
    r.parents = rig.parents.data();
    r.rest_translations = rig.rest_translations.data();
    r.left_hand = rig.left_hand;
    r.right_hand = rig.right_hand;
    r.left_foot = rig.left_foot;
    r.right_foot = rig.right_foot;
    r.has_double_ankle_joints = rig.has_double_ankle_joints;
    return r;
}
#endif

// skeleton.hpp offsets are float32 roundings of upstream's float64 rest
// offsets (joints.p). Their shortest round-trip decimals recover upstream's
// decimal rest layout, so summing those in double reproduces upstream's float64
// rig root height, and hence its float32 value, exactly for all three
// skeletons. (Summing the float values themselves lands one float32 ulp low
// for G1, which the G1 double-ankle IK amplifies to ~1e-4 near full leg
// extension.)
double rest_decimal(float value) noexcept {
    char buffer[32];
    const auto written = std::to_chars(buffer, buffer + sizeof(buffer), value);
    double parsed = static_cast<double>(value);
    if (written.ec == std::errc()) std::from_chars(buffer, written.ptr, parsed);
    return parsed;
}

} // namespace

motion_correction_rig make_motion_correction_rig(const skeleton_spec &s) {
    const std::size_t J = s.joints();
    motion_correction_rig rig;
    rig.parents.assign(s.parents.begin(), s.parents.end());
    rig.rest_translations.resize(J * 3);
    // Rest heights from the parent-local offsets with the pelvis at the origin.
    // Only the root rest y reaches the solver's outputs (foot/toe minHeight);
    // upstream's pelvis x,z offset is overwritten on every frame.
    std::vector<double> height(J, 0.);
    double lowest = 0.;
    for (std::size_t j = 0; j < J; ++j) {
        for (std::size_t k = 0; k < 3; ++k) rig.rest_translations[j * 3 + k] = s.offsets[j][k];
        if (s.parents[j] >= 0 && static_cast<std::size_t>(s.parents[j]) < j)
            height[j] = height[static_cast<std::size_t>(s.parents[j])] + rest_decimal(s.offsets[j][1]);
        lowest = std::min(lowest, height[j]);
    }
    // Larger offset for SOMA, "since model tends to generate lower to the ground".
    const double above_ground_offset = s.key == "soma30" ? .02 : .007;
    rig.rest_translations[0] = 0.F;
    rig.rest_translations[1] = static_cast<float>(-lowest + above_ground_offset);
    rig.rest_translations[2] = 0.F;
    rig.left_foot = static_cast<int>(s.end_effectors[0]);
    rig.right_foot = static_cast<int>(s.end_effectors[1]);
    rig.left_hand = static_cast<int>(s.end_effectors[2]);
    rig.right_hand = static_cast<int>(s.end_effectors[3]);
    rig.has_double_ankle_joints = s.key == "g1skel34";
    return rig;
}

std::expected<std::array<float, 3>, std::string> motion_correction_min_heights(const skeleton_spec &s) {
#ifdef KIMODO_HAVE_MOTION_CORRECTION
    const auto rig = make_motion_correction_rig(s);
    const auto r = solver_rig(rig);
    int right_toe = -1;
    for (std::size_t j = 0; j < rig.parents.size(); ++j)
        if (rig.parents[j] == rig.right_foot) right_toe = static_cast<int>(j);
    const std::array<float, 3> heights{mc::rest_joint_height(r, rig.right_foot), mc::rest_joint_height(r, rig.left_foot),
                                       right_toe < 0 ? 0.F : mc::rest_joint_height(r, right_toe)};
    if (std::ranges::any_of(heights, [](float h) { return std::isnan(h); }))
        return std::unexpected("invalid motion-correction rig");
    return heights;
#else
    (void)s;
    return std::unexpected("post-processing is unavailable in this build (x86 only)");
#endif
}

bool post_processing_available() noexcept {
#ifdef KIMODO_HAVE_MOTION_CORRECTION
    return true;
#else
    return false;
#endif
}

std::expected<void, std::string> post_process_motion(
    const skeleton_spec &s, std::size_t frames, std::vector<float> &local_xyzw, std::vector<float> &root_positions,
    std::span<const float> foot_contacts, std::span<const constraint_set> constraints, float root_margin,
    float contact_threshold) {
    const std::size_t J = s.joints();
    if (frames < 2) return std::unexpected("post-processing requires at least 2 frames");
    if (local_xyzw.size() != frames * J * 4 || root_positions.size() != frames * 3)
        return std::unexpected("invalid motion size for post-processing");
    if (!foot_contacts.empty() && foot_contacts.size() != frames * 4)
        return std::unexpected("foot contacts must be [T,4]");
    if (!std::isfinite(root_margin) || !std::isfinite(contact_threshold))
        return std::unexpected("invalid post-processing margin or contact threshold");
    if (J == 0 || s.parents[0] != -1) return std::unexpected("post-processing requires joint 0 to be the root");
    for (const auto &c : constraints)
        if (auto valid = check_constraint(c, J); !valid) return valid;
#ifdef KIMODO_HAVE_MOTION_CORRECTION
    const solver_inputs in = build_solver_inputs(constraints, frames, J);
    // Upstream contacts are the decoder's `foot_contacts > 0.5` booleans as
    // float 0/1 (kimodo_motionrep.py:199); the solver ORs heel and toe, so raw
    // values must be binarized first.
    std::vector<float> contacts(frames * 4, 0.F);
    for (std::size_t i = 0; i < foot_contacts.size(); ++i) contacts[i] = foot_contacts[i] > .5F ? 1.F : 0.F;
    const auto rig = make_motion_correction_rig(s);
    const mc::masks m{in.full_body.data(), in.left_hand.data(), in.right_hand.data(),
                      in.left_foot.data(), in.right_foot.data(), in.root.data()};
    const int status = mc::correct_motion(frames, solver_rig(rig), root_positions.data(), local_xyzw.data(),
                                          in.target_root.data(), in.target_local_xyzw.data(), m, contacts.data(),
                                          contact_threshold, root_margin);
    if (status == mc::invalid_input) return std::unexpected("motion correction rejected its inputs");
    if (status != mc::ok) return std::unexpected("motion correction failed");
    return {};
#else
    (void)foot_contacts;
    (void)root_margin;
    (void)contact_threshold;
    return std::unexpected("post-processing is unavailable in this build (x86 only)");
#endif
}

} // namespace kimodo::detail

namespace kimodo {
bool post_processing_available() noexcept { return detail::post_processing_available(); }
} // namespace kimodo
