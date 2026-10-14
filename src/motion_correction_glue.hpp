#pragma once

// C++17 boundary to the vendored NVIDIA MotionCorrection solver
// (third_party/motion_correction). Compiled inside the kimodo_motion_correction
// target; only plain pointers cross it, never Math/Eigen types.

#include <cstddef>

namespace kimodo::detail::mc {

// Working rig of upstream create_working_rig_from_skeleton. Joint 0 must be the
// root (parents[0] == -1). rest_translations is [J*3] parent-local rest
// translations; entry 0 is the root rest translation. Rest rotations are the
// identity, as upstream passes them.
struct rig {
    std::size_t joints = 0;
    const int *parents = nullptr;
    const float *rest_translations = nullptr;
    int left_hand = -1, right_hand = -1, left_foot = -1, right_foot = -1;
    bool has_double_ankle_joints = false;
};

// Per-frame 0/1 masks, each [T] (upstream FullBody, LeftHand, RightHand,
// LeftFoot, RightFoot, Root).
struct masks {
    const float *full_body = nullptr, *left_hand = nullptr, *right_hand = nullptr;
    const float *left_foot = nullptr, *right_foot = nullptr, *root = nullptr;
};

enum status : int { ok = 0, invalid_input = 1, solver_exception = 2 };

// Exact re-implementation of upstream MotionCorrection BindingsPython.cpp
// correct_motion (lines 28-359), with xyzw quaternions instead of the
// binding's wxyz. root_xyz [T*3] and local_xyzw [T*J*4] are corrected in place;
// targets have the same layouts; contacts is [T*4] (left heel, left toe, right
// heel, right toe). Requires frames >= 2.
int correct_motion(std::size_t frames, const rig &r, float *root_xyz, float *local_xyzw,
                   const float *target_root_xyz, const float *target_local_xyzw, const masks &m,
                   const float *contacts, float contact_threshold, float root_margin) noexcept;

// Rest-pose global height of `joint` as the solver computes it
// (Animation::JointLocalToGlobal over the identity-rotation rig). Upstream
// uses it as the foot/toe contact minHeight. Returns NaN on invalid input.
float rest_joint_height(const rig &r, int joint) noexcept;

} // namespace kimodo::detail::mc
