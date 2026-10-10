// C++17 glue between kimodo.cpp and the vendored NVIDIA MotionCorrection
// solver. correct_motion re-implements MotionCorrection/src/cpp/
// BindingsPython.cpp:28-359 (nv-tlabs/kimodo @ 58e78189, Apache-2.0) without
// pybind11, taking xyzw quaternions directly: the binding's wxyz shuffle only
// exists because torch quaternions are real-first. Line references below are
// to that file.

#include "motion_correction_glue.hpp"

#include "AnimProcessing/Utility.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace kimodo::detail::mc {
namespace {

using Pose = std::vector<Math::Transform>;

// strip_nan_inf (21-26).
float strip_nan_inf(float x) noexcept {
    if (std::isnan(x)) return 0;
    if (std::isinf(x)) return 0;
    return x;
}

bool valid_rig(const rig &r) noexcept {
    const std::size_t J = r.joints;
    if (J == 0 || !r.parents || !r.rest_translations || r.parents[0] != -1) return false;
    for (std::size_t j = 0; j < J; ++j)
        if (r.parents[j] < -1 || r.parents[j] >= static_cast<int>(J)) return false; // 130-133
    for (const int index : {r.left_hand, r.right_hand, r.left_foot, r.right_foot})
        if (index < 0 || index >= static_cast<int>(J)) return false; // 60-67
    return true;
}

// defaultPose (69-120): parent-local rest translations, identity rest rotations
// (upstream passes [0, 0, 0, 1] in x, y, z, w order, postprocess.py:171).
Pose default_pose(const rig &r) {
    Pose pose(r.joints);
    for (std::size_t j = 0; j < r.joints; ++j) {
        const float *t = r.rest_translations + 3 * j;
        pose[j].SetTranslation(Math::Vector(t[0], t[1], t[2]));
        pose[j].SetRotation(Math::Quaternion(0.0f, 0.0f, 0.0f, 1.0f));
    }
    return pose;
}

// setTransforms (278-308): joint 0 takes the root translation, every joint
// takes its normalized local rotation. Input is x, y, z, w per joint (the
// binding reads w, x, y, z and passes x, y, z, w to the same constructor).
void set_transforms(std::vector<Pose> &poses, std::size_t frames, std::size_t joints,
                    const float *root, const float *rotations) {
    for (std::size_t f = 0; f < frames; ++f)
        poses[f][0].SetTranslation({strip_nan_inf(root[3 * f + 0]), strip_nan_inf(root[3 * f + 1]),
                                    strip_nan_inf(root[3 * f + 2])});
    for (std::size_t f = 0; f < frames; ++f)
        for (std::size_t j = 0; j < joints; ++j) {
            const float *a = rotations + 4 * (joints * f + j);
            Math::Quaternion q(strip_nan_inf(a[0]), strip_nan_inf(a[1]), strip_nan_inf(a[2]),
                               strip_nan_inf(a[3]));
            q.Normalize();
            poses[f][j].SetRotation(q);
        }
}

} // namespace

int correct_motion(std::size_t frames, const rig &r, float *root_xyz, float *local_xyzw,
                   const float *target_root_xyz, const float *target_local_xyzw, const masks &m,
                   const float *contacts, float contact_threshold, float root_margin) noexcept {
    // CorrectMotion reads velocity_weights[1] (Utility.cpp:1136), so a single
    // frame would index out of bounds.
    if (frames < 2 || !valid_rig(r) || !root_xyz || !local_xyzw || !target_root_xyz ||
        !target_local_xyzw || !contacts || !m.full_body || !m.left_hand || !m.right_hand ||
        !m.left_foot || !m.right_foot || !m.root)
        return invalid_input;
    try {
        const std::size_t num_joints = r.joints, gen_length = frames;
        const Pose defaultPose = default_pose(r);
        const std::vector<int> joint_parents_vec(r.parents, r.parents + num_joints);

        // End-effector pins (168-191).
        std::vector<Animation::ContactInfo> endEffectorPins(4);
        endEffectorPins[0].jointIndex = r.left_hand;
        endEffectorPins[0].hintOffset = Math::Vector(0.0f, 0.0f, -0.1f);
        endEffectorPins[1].jointIndex = r.right_hand;
        endEffectorPins[1].hintOffset = Math::Vector(0.0f, 0.0f, -0.1f);
        endEffectorPins[2].jointIndex = r.left_foot;
        endEffectorPins[2].hintOffset = Math::Vector(0.0f, 0.0f, 0.1f);
        endEffectorPins[3].jointIndex = r.right_foot;
        endEffectorPins[3].hintOffset = Math::Vector(0.0f, 0.0f, 0.1f);
        for (auto &pin : endEffectorPins) pin.contactMask.reserve(gen_length);
        for (std::size_t i = 0; i < gen_length; ++i) {
            endEffectorPins[0].contactMask.push_back((1.0f - m.full_body[i]) * m.left_hand[i]);
            endEffectorPins[1].contactMask.push_back((1.0f - m.full_body[i]) * m.right_hand[i]);
            endEffectorPins[2].contactMask.push_back((1.0f - m.full_body[i]) * m.left_foot[i]);
            endEffectorPins[3].contactMask.push_back((1.0f - m.full_body[i]) * m.right_foot[i]);
        }

        // Two-bone foot contacts: [0] right foot, [1] left foot (193-229).
        std::vector<Animation::ContactInfo> contactInfo(2);
        auto footTranslation = Animation::JointLocalToGlobal(joint_parents_vec, r.right_foot, defaultPose).GetTranslation();
        contactInfo[0].jointIndex = r.right_foot;
        contactInfo[0].hintOffset = Math::Vector(0.0f, 0.0f, 0.1f);
        contactInfo[0].minHeight = footTranslation.GetY();
        footTranslation = Animation::JointLocalToGlobal(joint_parents_vec, r.left_foot, defaultPose).GetTranslation();
        contactInfo[1].jointIndex = r.left_foot;
        contactInfo[1].hintOffset = Math::Vector(0.0f, 0.0f, 0.1f);
        contactInfo[1].minHeight = footTranslation.GetY();
        auto &rContacts = contactInfo[0].contactMask;
        auto &lContacts = contactInfo[1].contactMask;
        rContacts.resize(gen_length);
        lContacts.resize(gen_length);
        for (std::size_t i = 0; i < gen_length; ++i) {
            // A foot pinned by an end-effector constraint is not a contact.
            rContacts[i] = m.right_foot[i] ? 0 : contacts[4 * i + 2];
            lContacts[i] = m.left_foot[i] ? 0 : contacts[4 * i + 0];
            // The heel is a contact whenever the toe is.
            rContacts[i] = std::min((m.right_foot[i] ? 0 : contacts[4 * i + 3]) + rContacts[i], 1.0f);
            lContacts[i] = std::min((m.left_foot[i] ? 0 : contacts[4 * i + 1]) + lContacts[i], 1.0f);
        }

        // Toes are the last joint whose parent is the foot (231-243).
        int left_toe_idx = -1, right_toe_idx = -1;
        for (int i = 0; i < static_cast<int>(num_joints); ++i) {
            if (joint_parents_vec[static_cast<std::size_t>(i)] == r.left_foot) left_toe_idx = i;
            if (joint_parents_vec[static_cast<std::size_t>(i)] == r.right_foot) right_toe_idx = i;
        }
        // One-bone toe contacts: [2] right toe, [3] left toe (245-275). Both use
        // the RIGHT toe's rest height as minHeight (256, 260), as upstream.
        if (left_toe_idx != -1 && right_toe_idx != -1) {
            const auto toeTranslation = Animation::JointLocalToGlobal(joint_parents_vec, right_toe_idx, defaultPose).GetTranslation();
            contactInfo.resize(4);
            contactInfo[2].jointIndex = right_toe_idx;
            contactInfo[2].contactType = Animation::kOneBone;
            contactInfo[2].minHeight = toeTranslation.GetY();
            contactInfo[3].jointIndex = left_toe_idx;
            contactInfo[3].contactType = Animation::kOneBone;
            contactInfo[3].minHeight = toeTranslation.GetY();
            auto &rToeContacts = contactInfo[2].contactMask;
            auto &lToeContacts = contactInfo[3].contactMask;
            rToeContacts.resize(gen_length);
            lToeContacts.resize(gen_length);
            for (std::size_t i = 0; i < gen_length; ++i) {
                rToeContacts[i] = m.right_foot[i] ? 0 : contacts[4 * i + 3];
                lToeContacts[i] = m.left_foot[i] ? 0 : contacts[4 * i + 1];
            }
        }

        // Generated and target poses start from the rest pose (310-314).
        std::vector<Pose> posesFixed(gen_length, defaultPose);
        set_transforms(posesFixed, gen_length, num_joints, root_xyz, local_xyzw);
        std::vector<Pose> posesTarget(gen_length, defaultPose);
        set_transforms(posesTarget, gen_length, num_joints, target_root_xyz, target_local_xyzw);

        // 316-322.
        const std::vector<float> fullPoseMask_vec(m.full_body, m.full_body + gen_length);
        const std::vector<float> rootMask_vec(m.root, m.root + gen_length);

        Animation::CorrectMotion(posesFixed, posesTarget, fullPoseMask_vec, rootMask_vec, contactInfo,
                                 endEffectorPins, joint_parents_vec, defaultPose, contact_threshold,
                                 root_margin, r.has_double_ankle_joints); // 324-336

        // Write back (338-357): the binding reads the quaternion's memory lanes
        // [0..3] = x, y, z, w and stores them as w, x, y, z; here they stay xyzw.
        for (std::size_t f = 0; f < gen_length; ++f) {
            const auto t = posesFixed[f][0].GetTranslation();
            root_xyz[3 * f + 0] = t.GetX();
            root_xyz[3 * f + 1] = t.GetY();
            root_xyz[3 * f + 2] = t.GetZ();
        }
        for (std::size_t f = 0; f < gen_length; ++f)
            for (std::size_t j = 0; j < num_joints; ++j) {
                const auto q = posesFixed[f][j].GetRotation();
                float lanes[4];
                static_assert(sizeof(q) >= sizeof(lanes));
                std::memcpy(lanes, &q, sizeof(lanes));
                std::memcpy(local_xyzw + 4 * (num_joints * f + j), lanes, sizeof(lanes));
            }
        return ok;
    } catch (...) {
        return solver_exception;
    }
}

float rest_joint_height(const rig &r, int joint) noexcept {
    if (!valid_rig(r) || joint < 0 || joint >= static_cast<int>(r.joints))
        return std::numeric_limits<float>::quiet_NaN();
    try {
        const Pose defaultPose = default_pose(r);
        const std::vector<int> joint_parents_vec(r.parents, r.parents + r.joints);
        return Animation::JointLocalToGlobal(joint_parents_vec, joint, defaultPose).GetTranslation().GetY();
    } catch (...) {
        return std::numeric_limits<float>::quiet_NaN();
    }
}

} // namespace kimodo::detail::mc
