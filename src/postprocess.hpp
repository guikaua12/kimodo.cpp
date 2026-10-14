#pragma once

#include <kimodo/kimodo.hpp>

#include <array>
#include <cstddef>
#include <expected>
#include <span>
#include <string>
#include <vector>

namespace kimodo::detail {
struct skeleton_spec;

// Upstream kimodo.motion_rep.smooth_root.get_smooth_root_pos for one sequence:
// root (pelvis) positions [T,3] -> smoothed (x_s, y, z_s) [T,3]. ADMM multigrid
// with a 0.06 m margin, reproducing upstream quirks (half-frame shift).
// Upstream fails for T <= 4 (singular coarse system); this returns the input
// unchanged for those lengths (and for a size mismatch).
std::vector<float> smooth_root_positions(std::span<const float> root_positions, std::size_t frames);

// Upstream KimodoMotionRep.__call__ (unnormalized, not canonicalized): local
// rotations [T,J,4] xyzw + root positions [T,3] -> raw features [T, 9+12J]
// (smooth root, heading, local positions, 6D global rotations, velocities,
// foot contacts with thresholds 0.15 m/s-scaled and 0.10 m).
// Requires T >= 5, like upstream's smooth-root solve.
std::expected<std::vector<float>, std::string> encode_motion(
    const skeleton_spec &skeleton, std::span<const float> local_xyzw,
    std::span<const float> root_positions, std::size_t frames, float fps = 30.F);

// Upstream kimodo.postprocess.post_process_motion for one sequence, in place:
// foot-skate cleanup plus constraint enforcement through the vendored NVIDIA
// MotionCorrection solver. `foot_contacts` is [T,4] (left heel, left toe,
// right heel, right toe; > 0.5 means contact) or empty for no contacts.
// `constraints` use frame indices relative to this sequence; frames >= T are
// ignored. Requires T >= 2. Upstream's demo and CLI never post-process G1
// motions; `model::generate` follows that policy.
std::expected<void, std::string> post_process_motion(
    const skeleton_spec &skeleton, std::size_t frames,
    std::vector<float> &local_xyzw, std::vector<float> &root_positions,
    std::span<const float> foot_contacts, std::span<const constraint_set> constraints,
    float root_margin = .04F, float contact_threshold = .5F);

// Working rig handed to the solver (upstream create_working_rig_from_skeleton):
// parent-local rest translations [J*3] from skeleton_spec::offsets, the root
// lifted so that the lowest rest joint sits `above_ground_offset` above y = 0
// (0.02 for SOMA, 0.007 otherwise), identity rest rotations.
struct motion_correction_rig {
    std::vector<int> parents;
    std::vector<float> rest_translations;
    int left_hand = -1, right_hand = -1, left_foot = -1, right_foot = -1;
    bool has_double_ankle_joints = false;
};
motion_correction_rig make_motion_correction_rig(const skeleton_spec &skeleton);

// True when the NVIDIA MotionCorrection solver is part of this build (x86 and
// KIMODO_ENABLE_POSTPROCESS=ON); post_process_motion fails otherwise.
bool post_processing_available() noexcept;

// Contact minHeights exactly as the solver computes them from the rig: right
// foot, left foot, toes (upstream uses the right toe's rest height for both
// toes). Fails when the solver is not part of this build.
std::expected<std::array<float, 3>, std::string> motion_correction_min_heights(const skeleton_spec &skeleton);

} // namespace kimodo::detail
