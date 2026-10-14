#pragma once
#include <expected>
#include <span>
#include <string>
#include <vector>
namespace kimodo::detail {
struct skeleton_spec;
// Upstream KimodoMotionRep.inverse: local rotations [T,J,4] xyzw, root (pelvis)
// positions [T,3], foot contacts [T,4] as 0/1 (unnormalized feature > 0.5;
// left heel, left toe, right heel, right toe) and the smooth root [T,3]
// (features 0..2: smooth x, pelvis y, smooth z).
struct decoded_motion { std::vector<float> local_xyzw, root_positions, foot_contacts, smooth_root; };
std::expected<decoded_motion,std::string> decode_motion(std::span<const float> normalized, std::size_t frames, const skeleton_spec &skeleton, std::span<const float> global_mean, std::span<const float> global_std, std::span<const float> body_mean, std::span<const float> body_std);
// Same decode for already unnormalized feature rows [T, 9+12J].
std::expected<decoded_motion,std::string> decode_motion_rows(std::span<const float> raw, std::size_t frames, const skeleton_spec &skeleton);
std::expected<decoded_motion,std::string> decode_smplx22(std::span<const float> normalized, std::size_t frames, std::span<const float> global_mean, std::span<const float> global_std, std::span<const float> body_mean, std::span<const float> body_std);
}
