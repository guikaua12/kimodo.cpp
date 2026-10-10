#pragma once

#include <kimodo/kimodo.hpp>

#include <cstddef>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace kimodo::detail {
struct skeleton_spec;
namespace json { struct value; }

// Unnormalized model condition over a timeline: row-major [frames, motion_dim]
// observed feature values and a 0/1 feature mask (upstream create_conditions).
struct motion_conditions {
    std::size_t frames = 0;
    std::vector<float> observed;
    std::vector<float> mask;
    [[nodiscard]] bool any() const noexcept;
    // True when frame `frame` has at least one masked feature.
    [[nodiscard]] bool frame_constrained(std::size_t frame, std::size_t motion_dim) const noexcept;
};

// Forward kinematics of one control-skeleton pose. `local_xyzw` is [J,4],
// `root` is [3]. Writes global joint positions [J,3] and global rotation
// matrices [J,9] (row-major, column-vector convention) when non-null.
void forward_kinematics(const skeleton_spec &skeleton, const float *local_xyzw, const float *root,
                        float *positions, float *rotations);

// Structural validation against the skeleton and a timeline of `frames`
// frames: sizes, finite values, unit-ish quaternions, frame range, names.
std::expected<void, std::string> validate_constraints(
    const skeleton_spec &skeleton, std::span<const constraint_set> constraints, std::size_t frames);

// Upstream KimodoMotionRep.create_conditions over global frames [0, frames):
// constraints are applied in list order and, for duplicate (frame, feature)
// entries, the last appended value wins (PyTorch CPU scatter behaviour).
std::expected<motion_conditions, std::string> build_conditions(
    const skeleton_spec &skeleton, std::span<const constraint_set> constraints, std::size_t frames);

// Upstream crop_move: keep frames in [begin, end) and shift them by -begin.
std::vector<constraint_set> crop_constraints(std::span<const constraint_set> constraints,
                                             std::size_t begin, std::size_t end);

// upstream constraints.json <-> constraint sets on the model control skeleton.
// SOMA files may carry 77-joint rotations; they are reduced to the 30-joint
// control skeleton with the upstream name slice. Output always uses the
// control-skeleton joint count.
std::expected<std::vector<constraint_set>, std::string> parse_constraints_json(
    const skeleton_spec &skeleton, std::string_view json);
// Same, for an already parsed JSON array (e.g. inline in a generation request).
std::expected<std::vector<constraint_set>, std::string> parse_constraints_json(
    const skeleton_spec &skeleton, const json::value &document);
std::string constraints_to_json(const skeleton_spec &skeleton, std::span<const constraint_set> constraints);

} // namespace kimodo::detail
