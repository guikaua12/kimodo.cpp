// Constraint semantics against NVIDIA Kimodo (kimodo/constraints.py,
// KimodoMotionRep.create_conditions): mask index sets, condition values,
// merge rules, cropping, constraints.json import/export and validation.
// Needs no model or GPU. Usage: kimodo-constraints-test TESTS/DATA/CONSTRAINTS
#include "constraints.hpp"
#include "json.hpp"
#include "motion_decode.hpp"
#include "skeleton.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace kimodo;
using namespace kimodo::detail;

namespace {
int failures = 0, checks = 0;
void check(bool ok, const std::string &what) {
    ++checks;
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL: %s\n", what.c_str()); }
}
bool near(double a, double b, double tolerance) { return std::abs(a - b) <= tolerance; }

// Verbatim reference copy of src/sequence.cpp condition_row (the upstream
// `_multiprompt` transition condition); our FK must reproduce it.
namespace reference {
struct mat { double v[9]; };
mat mul(const mat&a,const mat&b){mat r{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int k=0;k<3;++k)r.v[i*3+j]+=a.v[i*3+k]*b.v[k*3+j];return r;}
mat trans(const mat&a){mat r{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)r.v[i*3+j]=a.v[j*3+i];return r;}
mat cont6(const float*x){double a[3]={x[0],x[1],x[2]},n=std::sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]);for(double&q:a)q/=n;double z[3]={a[1]*x[5]-a[2]*x[4],a[2]*x[3]-a[0]*x[5],a[0]*x[4]-a[1]*x[3]};n=std::sqrt(z[0]*z[0]+z[1]*z[1]+z[2]*z[2]);for(double&q:z)q/=n;double b[3]={z[1]*a[2]-z[2]*a[1],z[2]*a[0]-z[0]*a[2],z[0]*a[1]-z[1]*a[0]};return{{a[0],b[0],z[0],a[1],b[1],z[1],a[2],b[2],z[2]}};}
void rotate(const mat&m,const std::array<float,3>&x,float*o){for(int i=0;i<3;++i)o[i]=static_cast<float>(m.v[i*3]*x[0]+m.v[i*3+1]*x[1]+m.v[i*3+2]*x[2]);}
float condition_row(const float *raw, const skeleton_spec &s, float *value) {
    const size_t D=s.motion_dim(),J=s.joints(),rotation_begin=5+3*J;
    std::copy_n(raw,D,value);
    std::vector<mat> decoded(J),local(J),global(J);
    for(size_t j=0;j<J;++j)decoded[j]=cont6(value+rotation_begin+j*6);
    for(size_t j=0;j<J;++j)local[j]=s.parents[j]<0?decoded[j]:mul(trans(decoded[static_cast<size_t>(s.parents[j])]),decoded[j]);
    const float root[3]={value[0]+value[5],value[6],value[2]+value[7]};
    std::vector<std::array<float,3>> posed(J);
    for(size_t j=0;j<J;++j){const int parent=s.parents[j];if(parent<0){global[j]=local[j];posed[j]={root[0],root[1],root[2]};}else{global[j]=mul(global[static_cast<size_t>(parent)],local[j]);float offset[3];rotate(global[static_cast<size_t>(parent)],s.offsets[j],offset);for(int k=0;k<3;++k)posed[j][k]=posed[static_cast<size_t>(parent)][k]+offset[k];}}
    const auto right=s.hips[0],left=s.hips[1];
    const float angle=std::atan2(posed[right][2]-posed[left][2],-(posed[right][0]-posed[left][0]));
    value[1]=root[1];value[3]=std::cos(angle);value[4]=std::sin(angle);
    for(size_t j=0;j<J;++j){value[5+j*3]=posed[j][0]-value[0];value[6+j*3]=posed[j][1];value[7+j*3]=posed[j][2]-value[2];}
    for(unsigned joint:s.end_effectors)for(int d=0;d<6;++d)value[rotation_begin+joint*6+static_cast<size_t>(d)]=static_cast<float>(global[joint].v[(d%3)*3+d/3]);
    return angle;
}
} // namespace reference

using index_set = std::set<std::size_t>;
index_set ids(std::size_t first, std::size_t last) { index_set out; for (auto i = first; i <= last; ++i) out.insert(i); return out; }
index_set unite(std::initializer_list<index_set> sets) { index_set out; for (const auto &s : sets) out.insert(s.begin(), s.end()); return out; }
index_set mask_row(const motion_conditions &m, std::size_t frame, std::size_t D) {
    index_set out; for (std::size_t d = 0; d < D; ++d) if (m.mask[frame*D+d] != 0.F) out.insert(d); return out;
}
std::string show(const index_set &s) { std::string out; for (auto i : s) out += std::to_string(i) + ","; return out; }

std::array<double, 9> quaternion_matrix(const float *q) {
    double x=q[0], y=q[1], z=q[2], w=q[3]; const double n=std::sqrt(x*x+y*y+z*z+w*w); x/=n; y/=n; z/=n; w/=n;
    return {1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w), 2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w), 2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)};
}
// Exact Rodrigues (no epsilon) as an independent axis-angle reference.
std::array<double, 9> rodrigues(double ax, double ay, double az) {
    const double angle = std::sqrt(ax*ax+ay*ay+az*az);
    if (angle < 1e-12) return {1,0,0,0,1,0,0,0,1};
    const double x=ax/angle, y=ay/angle, z=az/angle, s=std::sin(angle), c=1-std::cos(angle);
    return {1+c*(-y*y-z*z), -s*z+c*x*y, s*y+c*x*z, s*z+c*x*y, 1+c*(-x*x-z*z), -s*x+c*y*z, -s*y+c*x*z, s*x+c*y*z, 1+c*(-x*x-y*y)};
}
double matrix_distance(const std::array<double, 9> &a, const std::array<double, 9> &b) {
    double worst = 0; for (int i = 0; i < 9; ++i) worst = std::max(worst, std::abs(a[i]-b[i])); return worst;
}

std::vector<float> random_pose(std::size_t J, std::mt19937 &rng, float spread) {
    std::normal_distribution<float> normal(0.F, 1.F);
    std::vector<float> q(J*4);
    for (std::size_t j = 0; j < J; ++j) {
        float v[4]{normal(rng)*spread, normal(rng)*spread, normal(rng)*spread, 1.F};
        const float n = std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]+v[3]*v[3]);
        for (int k = 0; k < 4; ++k) q[j*4+static_cast<std::size_t>(k)] = v[k]/n;
    }
    return q;
}

constraint_set posed(const skeleton_spec &s, constraint_type type, std::vector<unsigned> frames,
                     std::vector<std::string> names = {}, unsigned seed = 1) {
    std::mt19937 rng(seed);
    constraint_set c;
    c.type = type; c.frame_indices = std::move(frames); c.joint_names = std::move(names);
    for (std::size_t k = 0; k < c.frame_indices.size(); ++k) {
        const auto pose = random_pose(s.joints(), rng, .3F);
        c.local_rotations_xyzw.insert(c.local_rotations_xyzw.end(), pose.begin(), pose.end());
        c.root_positions.insert(c.root_positions.end(), {.1F*float(k)+.3F, .9F, -.2F*float(k)+.1F});
    }
    return c;
}

std::string read_file(const std::filesystem::path &path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), {}};
}

struct ee_table { const skeleton_spec *s; index_set lf, rf, lh, rh, hips; std::size_t fullbody_last; };

// (a) Feature indices per skeleton transcribed from upstream (KimodoMotionRep
// feature layout, skeleton/base.py expand_joint_names), not recomputed here.
void test_masks() {
    const std::array<ee_table, 3> tables{{
        {&smplx22_spec, unite({ids(26,28), ids(35,37), ids(113,118)}), unite({ids(29,31), ids(38,40), ids(119,124)}),
         unite({ids(65,67), ids(191,196)}), unite({ids(68,70), ids(197,202)}), unite({ids(5,7), ids(71,76)}), 70},
        {&soma30_spec, unite({ids(77,82), ids(239,244)}), unite({ids(89,94), ids(263,268)}),
         unite({ids(44,46), ids(50,52), ids(173,178)}), unite({ids(62,64), ids(68,70), ids(209,214)}), unite({ids(5,7), ids(95,100)}), 94},
        {&g1skel34_spec, unite({ids(23,28), ids(143,148)}), unite({ids(44,49), ids(185,190)}),
         unite({ids(77,82), ids(251,256)}), unite({ids(101,106), ids(299,304)}), unite({ids(5,7), ids(107,112)}), 106},
    }};
    const index_set base = ids(0, 4);
    for (const auto &t : tables) {
        const auto &s = *t.s;
        const std::size_t D = s.motion_dim();
        const std::string key(s.key);
        for (std::size_t j = 0; j < s.joints(); ++j) check(s.parents[j] < static_cast<int>(j), key + ": parents are topologically ordered");
        for (std::size_t limb = 0; limb < 4; ++limb) check(s.chains[limb][0] == static_cast<int>(s.end_effectors[limb]), key + ": chain base is the end effector");
        auto masks = [&](const constraint_set &c) {
            const std::array<constraint_set, 1> list{c};
            auto built = build_conditions(s, list, 3);
            check(built.has_value(), key + ": build " + (built ? "" : built.error()));
            if (!built) return index_set{};
            check(mask_row(*built, 0, D).empty() && mask_row(*built, 2, D).empty(), key + ": only the constrained frame is masked");
            check(built->frame_constrained(1, D) && !built->frame_constrained(0, D) && built->any(), key + ": frame_constrained/any");
            return mask_row(*built, 1, D);
        };
        const std::pair<constraint_type, index_set> cases[]{
            {constraint_type::left_foot, unite({base, t.lf})}, {constraint_type::right_foot, unite({base, t.rf})},
            {constraint_type::left_hand, unite({base, t.lh})}, {constraint_type::right_hand, unite({base, t.rh})},
            {constraint_type::fullbody, ids(0, t.fullbody_last)}};
        for (const auto &[type, expected] : cases) {
            const auto got = masks(posed(s, type, {1}));
            check(got == expected, key + ": mask of type " + std::to_string(static_cast<int>(type)) + " got " + show(got));
        }
        check(masks(posed(s, constraint_type::end_effector, {1}, {"Hips"})) == unite({base, t.hips}), key + ": Hips end-effector mask");
        check(masks(posed(s, constraint_type::end_effector, {1}, {"RightHand", "LeftFoot"})) == unite({base, t.rh, t.lf}), key + ": two-limb end-effector mask");
        check(masks(posed(s, constraint_type::end_effector, {1}, {"LeftFoot", "RightFoot", "LeftHand", "RightHand", "Hips"})) ==
              unite({base, t.lf, t.rf, t.lh, t.rh, t.hips}), key + ": all end-effectors mask");
        check(masks(posed(s, constraint_type::end_effector, {1}, {})) == base, key + ": empty end-effector pins only the root");
        constraint_set root; root.type = constraint_type::root2d; root.frame_indices = {1}; root.smooth_root_2d = {.5F, -.5F};
        check(masks(root) == index_set{0, 2}, key + ": root2d mask");
        root.global_root_heading = {0.F, 1.F};
        check(masks(root) == index_set{0, 2, 3, 4}, key + ": root2d+heading mask");
    }
}

// (b) Full-body values reproduce condition_row (FK from decoded 6D rows).
void test_condition_row_parity() {
    for (const auto *sp : {&smplx22_spec, &soma30_spec, &g1skel34_spec}) {
        const auto &s = *sp;
        const std::string key(s.key);
        const std::size_t D = s.motion_dim(), J = s.joints(), rotation_begin = 5+3*J;
        std::mt19937 rng(11);
        std::uniform_real_distribution<float> u(-1.F, 1.F);
        for (int trial = 0; trial < 8; ++trial) {
            std::vector<float> raw(D);
            for (auto &v : raw) v = u(rng)*.1F;
            raw[0] = u(rng)*2.F; raw[1] = .93F; raw[2] = u(rng)*2.F; raw[6] = .92F + u(rng)*.05F;
            const auto pose = random_pose(J, rng, .6F);
            for (std::size_t j = 0; j < J; ++j) {
                const auto m = quaternion_matrix(pose.data()+j*4);
                for (std::size_t d = 0; d < 6; ++d) raw[rotation_begin+j*6+d] = static_cast<float>(m[(d%3)*3+d/3]) + u(rng)*2e-3F;
            }
            for (std::size_t k = 0; k < 4; ++k) raw[D-4+k] = u(rng);
            std::vector<float> value(D);
            const float angle = reference::condition_row(raw.data(), s, value.data());
            auto decoded = decode_motion_rows(raw, 1, s);
            check(decoded.has_value(), key + ": decode_motion_rows");
            if (!decoded) return;
            for (std::size_t k = 0; k < 3; ++k) check(decoded->smooth_root[k] == raw[k], key + ": decoded smooth root");
            for (std::size_t k = 0; k < 4; ++k) check(decoded->foot_contacts[k] == (raw[D-4+k] > .5F ? 1.F : 0.F), key + ": decoded contacts");
            constraint_set full; full.type = constraint_type::fullbody; full.frame_indices = {0};
            full.local_rotations_xyzw = decoded->local_xyzw; full.root_positions = decoded->root_positions;
            constraint_set ee = full; ee.type = constraint_type::end_effector; ee.joint_names = {"LeftFoot", "RightFoot", "LeftHand", "RightHand"};
            const std::array<constraint_set, 2> list{full, ee};
            auto built = build_conditions(s, list, 1);
            check(built.has_value(), key + ": build full-body");
            if (!built) return;
            const auto &o = built->observed;
            double worst = 0;
            for (std::size_t j = 0; j < J; ++j) {
                worst = std::max(worst, std::abs(double(o[0]+o[5+3*j]) - (value[0]+value[5+3*j])));
                worst = std::max(worst, std::abs(double(o[6+3*j]) - value[6+3*j]));
                worst = std::max(worst, std::abs(double(o[2]+o[7+3*j]) - (value[2]+value[7+3*j])));
            }
            check(worst <= 1e-5, key + ": global joint positions vs condition_row, max error " + std::to_string(worst));
            check(near(o[1], value[1], 1e-6), key + ": root y");
            check(near(o[3], value[3], 1e-5) && near(o[4], value[4], 1e-5) && near(o[3], std::cos(angle), 1e-5), key + ": heading");
            // Our smooth root is the pelvis xz by design (condition_row keeps the feature's).
            check(o[0] == decoded->root_positions[0] && o[2] == decoded->root_positions[2] && o[5] == 0.F && o[7] == 0.F, key + ": default smooth root is pelvis xz");
            double rotation_worst = 0;
            for (unsigned joint : s.end_effectors)
                for (std::size_t d = 0; d < 6; ++d) rotation_worst = std::max(rotation_worst, std::abs(double(o[rotation_begin+joint*6+d]) - value[rotation_begin+joint*6+d]));
            check(rotation_worst <= 1e-5, key + ": end-effector 6D rotations vs condition_row, max error " + std::to_string(rotation_worst));
        }
    }
}

// (c) root2d with and without heading: values written as given.
void test_root2d() {
    const auto &s = soma30_spec; const std::size_t D = s.motion_dim();
    constraint_set r; r.type = constraint_type::root2d; r.frame_indices = {1, 4}; r.smooth_root_2d = {.5F, -.25F, 1.5F, 2.F};
    std::array<constraint_set, 1> list{r};
    auto built = build_conditions(s, list, 6);
    check(built && built->observed[D] == .5F && built->observed[D+2] == -.25F && built->observed[4*D] == 1.5F && built->observed[4*D+2] == 2.F, "root2d values");
    check(built && built->observed[D+1] == 0.F && built->mask[D+1] == 0.F && mask_row(*built, 4, D) == index_set{0, 2}, "root2d leaves root y and heading free");
    list[0].global_root_heading = {.6F, .8F, 2.F, 0.F};
    built = build_conditions(s, list, 6);
    check(built && built->observed[D+3] == .6F && built->observed[D+4] == .8F && built->observed[4*D+3] == 2.F && built->observed[4*D+4] == 0.F,
          "root2d heading stored without renormalization");
    check(built && mask_row(*built, 0, D).empty() && mask_row(*built, 5, D).empty(), "root2d other frames unmasked");
}

// (d) Last appended wins; positions are relative to the merged smooth root.
void test_merge_rules() {
    const auto &s = soma30_spec; const std::size_t D = s.motion_dim(), J = s.joints();
    auto pose_at = [&](unsigned frame, float x, float y, float z, unsigned seed) {
        auto c = posed(s, constraint_type::fullbody, {frame}, {}, seed); c.root_positions = {x, y, z}; return c;
    };
    constraint_set root; root.type = constraint_type::root2d; root.frame_indices = {2}; root.smooth_root_2d = {5.F, 7.F};
    const auto full = pose_at(2, 1.F, .9F, 2.F, 3);
    std::vector<float> positions(J*3); forward_kinematics(s, full.local_rotations_xyzw.data(), full.root_positions.data(), positions.data(), nullptr);
    {
        const std::array<constraint_set, 2> list{full, root};
        auto b = build_conditions(s, list, 4);
        check(b && b->observed[2*D] == 5.F && b->observed[2*D+2] == 7.F, "root2d after full-body wins the smooth root");
        check(b && b->observed[2*D+5] == 1.F-5.F && b->observed[2*D+7] == 2.F-7.F && b->observed[2*D+6] == .9F && b->observed[2*D+1] == .9F,
              "full-body positions relative to the merged (root2d) smooth root");
        check(b && near(b->observed[2*D+5+3*13], positions[13*3]-5.F, 1e-6) && near(b->observed[2*D+7+3*13], positions[13*3+2]-7.F, 1e-6),
              "hand position relative to the merged smooth root");
    }
    {
        const std::array<constraint_set, 2> list{root, full};
        auto b = build_conditions(s, list, 4);
        check(b && b->observed[2*D] == 1.F && b->observed[2*D+2] == 2.F && b->observed[2*D+5] == 0.F && b->observed[2*D+7] == 0.F,
              "full-body after root2d wins the smooth root");
    }
    {
        const auto second = pose_at(2, 3.F, .8F, 4.F, 4);
        const std::array<constraint_set, 2> list{full, second};
        auto b = build_conditions(s, list, 4);
        std::vector<float> p2(J*3); forward_kinematics(s, second.local_rotations_xyzw.data(), second.root_positions.data(), p2.data(), nullptr);
        check(b && b->observed[2*D] == 3.F && b->observed[2*D+1] == .8F && near(b->observed[2*D+5+3*20], p2[20*3]-3.F, 1e-6), "second full-body set wins");
        // Duplicate frame inside one set: the later keyframe wins.
        constraint_set both = full;
        both.frame_indices = {2, 2};
        both.local_rotations_xyzw.insert(both.local_rotations_xyzw.end(), second.local_rotations_xyzw.begin(), second.local_rotations_xyzw.end());
        both.root_positions.insert(both.root_positions.end(), second.root_positions.begin(), second.root_positions.end());
        const std::array<constraint_set, 1> one{both};
        auto c = build_conditions(s, one, 4);
        check(c && b && c->observed == b->observed && c->mask == b->mask, "duplicate frame within a set: last wins");
    }
    {
        auto explicit_smooth = full; explicit_smooth.smooth_root_2d = {.5F, .25F};
        const std::array<constraint_set, 1> list{explicit_smooth};
        auto b = build_conditions(s, list, 4);
        check(b && b->observed[2*D] == .5F && b->observed[2*D+2] == .25F && b->observed[2*D+5] == .5F && b->observed[2*D+7] == 1.75F,
              "explicit smooth_root_2d drives smooth root and relative positions");
        auto hand = full; hand.type = constraint_type::right_hand;
        const std::array<constraint_set, 2> hands{hand, root};
        auto h = build_conditions(s, hands, 4);
        check(h && near(h->observed[2*D+5+3*19], positions[19*3]-5.F, 1e-6) && h->observed[2*D+3] == b->observed[2*D+3],
              "end-effector positions follow a later root2d smooth root; heading from the pose");
    }
}

// (e) crop_move.
void test_crop() {
    const auto &s = smplx22_spec; const std::size_t J = s.joints();
    auto c = posed(s, constraint_type::left_hand, {5, 0, 12, 7}, {}, 9);
    c.smooth_root_2d = {1, 2, 3, 4, 5, 6, 7, 8};
    constraint_set r; r.type = constraint_type::root2d; r.frame_indices = {9, 4}; r.smooth_root_2d = {1, 2, 3, 4}; r.global_root_heading = {1, 0, 0, 1};
    const std::array<constraint_set, 2> list{c, r};
    const auto cropped = crop_constraints(list, 5, 10);
    check(cropped.size() == 2, "crop keeps every set");
    if (cropped.size() != 2) return;
    check(cropped[0].type == constraint_type::left_hand && cropped[0].frame_indices == std::vector<unsigned>{0, 2}, "crop frames shifted");
    check(std::equal(cropped[0].local_rotations_xyzw.begin(), cropped[0].local_rotations_xyzw.begin()+static_cast<std::ptrdiff_t>(J*4), c.local_rotations_xyzw.begin()) &&
          std::equal(cropped[0].local_rotations_xyzw.begin()+static_cast<std::ptrdiff_t>(J*4), cropped[0].local_rotations_xyzw.end(), c.local_rotations_xyzw.begin()+static_cast<std::ptrdiff_t>(3*J*4)) &&
          cropped[0].local_rotations_xyzw.size() == 2*J*4, "crop rotations follow their frames");
    check(cropped[0].root_positions == std::vector<float>{c.root_positions[0], c.root_positions[1], c.root_positions[2], c.root_positions[9], c.root_positions[10], c.root_positions[11]}, "crop roots");
    check(cropped[0].smooth_root_2d == std::vector<float>{1, 2, 7, 8}, "crop smooth roots");
    check(cropped[1].frame_indices == std::vector<unsigned>{4} && cropped[1].smooth_root_2d == std::vector<float>{1, 2} && cropped[1].global_root_heading == std::vector<float>{1, 0}, "crop root2d with heading");
    const auto empty = crop_constraints(list, 20, 30);
    check(empty.size() == 2 && empty[0].frame_indices.empty() && empty[0].local_rotations_xyzw.empty() && empty[1].type == constraint_type::root2d, "empty crop keeps typed empty sets");
    auto valid = validate_constraints(s, empty, 10);
    check(valid.has_value(), "empty sets are valid");
}

struct expected_set { constraint_type type; std::vector<unsigned> frames; };

void compare_sets(const std::vector<constraint_set> &a, const std::vector<constraint_set> &b, const std::string &what) {
    check(a.size() == b.size(), what + ": set count");
    for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        check(a[i].type == b[i].type && a[i].frame_indices == b[i].frame_indices && a[i].joint_names == b[i].joint_names, what + ": type/frames/names");
        check(a[i].root_positions == b[i].root_positions && a[i].smooth_root_2d == b[i].smooth_root_2d && a[i].global_root_heading == b[i].global_root_heading,
              what + ": root/smooth/heading round-trip exactly");
        double worst = 0;
        for (std::size_t q = 0; q*4 < std::min(a[i].local_rotations_xyzw.size(), b[i].local_rotations_xyzw.size()); ++q)
            worst = std::max(worst, matrix_distance(quaternion_matrix(a[i].local_rotations_xyzw.data()+q*4), quaternion_matrix(b[i].local_rotations_xyzw.data()+q*4)));
        check(a[i].local_rotations_xyzw.size() == b[i].local_rotations_xyzw.size() && worst <= 1e-5, what + ": rotations round-trip, max error " + std::to_string(worst));
    }
}

// (f) constraints.json import / export.
void test_json(const std::filesystem::path &data) {
    struct example { const char *file; const skeleton_spec *s; std::vector<expected_set> sets; };
    const std::vector<example> examples{
        {"soma_03_full_body_keyframes.json", &soma30_spec, {{constraint_type::fullbody, {79, 134}}}},
        {"soma_04_ee_constraint.json", &soma30_spec, {{constraint_type::right_foot, {28, 94}}, {constraint_type::left_foot, {28, 94}},
                                                      {constraint_type::left_hand, {28, 94}}, {constraint_type::right_hand, {28, 94}}}},
        {"soma_06_root_waypoints.json", &soma30_spec, {{constraint_type::root2d, {0, 90, 180}}}},
        {"soma_07_mixed_constraints.json", &soma30_spec, {{constraint_type::fullbody, {108}}, {constraint_type::root2d, {}}}},
        {"g1_04_ee_constraint.json", &g1skel34_spec, {{constraint_type::right_hand, {129, 93, 0}}, {constraint_type::left_foot, {93, 0}},
                                                     {constraint_type::right_foot, {0}}, {constraint_type::left_hand, {0}}}},
    };
    for (const auto &e : examples) {
        const auto &s = *e.s; const std::size_t J = s.joints(), D = s.motion_dim();
        const std::string name = e.file;
        const auto text = read_file(data / e.file);
        check(!text.empty(), name + ": readable test data");
        auto parsed = parse_constraints_json(s, text);
        check(parsed.has_value(), name + ": parse " + (parsed ? "" : parsed.error()));
        if (!parsed) continue;
        check(parsed->size() == e.sets.size(), name + ": set count");
        for (std::size_t i = 0; i < std::min(parsed->size(), e.sets.size()); ++i) {
            check((*parsed)[i].type == e.sets[i].type, name + ": type of set " + std::to_string(i));
            if (!e.sets[i].frames.empty()) check((*parsed)[i].frame_indices == e.sets[i].frames, name + ": frames of set " + std::to_string(i));
        }
        // Independent check of the axis-angle import and of the shipped smooth roots (= pelvis xz).
        auto document = json::parse(text);
        double worst = 0;
        unsigned last_frame = 0;
        for (std::size_t i = 0; document && i < document->array.size() && i < parsed->size(); ++i) {
            const auto &item = document->array[i];
            const auto &c = (*parsed)[i];
            for (unsigned f : c.frame_indices) last_frame = std::max(last_frame, f);
            if (const auto *rot = item.find("local_joints_rot")) {
                for (std::size_t k = 0; k < rot->array.size(); ++k)
                    for (std::size_t j = 0; j < J; ++j) {
                        const auto &aa = rot->array[k].array[j].array;
                        worst = std::max(worst, matrix_distance(rodrigues(aa[0].number, aa[1].number, aa[2].number), quaternion_matrix(c.local_rotations_xyzw.data()+(k*J+j)*4)));
                    }
                for (std::size_t k = 0; k < c.frame_indices.size(); ++k)
                    check(c.smooth_root_2d[k*2] == c.root_positions[k*3] && c.smooth_root_2d[k*2+1] == c.root_positions[k*3+2], name + ": shipped smooth root is pelvis xz");
            }
        }
        check(worst <= 1e-5, name + ": axis-angle import vs Rodrigues, max error " + std::to_string(worst));
        // Export (upstream key order) and re-import.
        const auto written = constraints_to_json(s, *parsed);
        const auto first_type = written.find("\"type\""), frames_key = written.find("\"frame_indices\"");
        check(first_type != std::string::npos && frames_key > first_type, name + ": writer starts with type, frame_indices");
        if ((*parsed)[0].type != constraint_type::root2d) {
            const auto rot = written.find("\"local_joints_rot\""), root = written.find("\"root_positions\""), smooth = written.find("\"smooth_root_2d\"");
            check(frames_key < rot && rot < root && root < smooth, name + ": writer key order");
        }
        auto again = parse_constraints_json(s, written);
        check(again.has_value(), name + ": re-parse " + (again ? "" : again.error()));
        if (again) compare_sets(*parsed, *again, name);
        // Conditions build on the example's own timeline.
        auto built = build_conditions(s, *parsed, last_frame+1);
        check(built.has_value() && built->any(), name + ": builds conditions");
        if (built && name == "soma_04_ee_constraint.json") {
            const index_set expected = unite({ids(0, 4), ids(77, 82), ids(239, 244), ids(89, 94), ids(263, 268), ids(44, 46), ids(50, 52),
                                              ids(173, 178), ids(62, 64), ids(68, 70), ids(209, 214)});
            check(mask_row(*built, 28, D) == expected && mask_row(*built, 94, D) == expected && mask_row(*built, 27, D).empty(), name + ": four end-effector masks");
        }
    }
    // A wrong-skeleton file is rejected (G1 34 joints for SOMA).
    auto wrong = parse_constraints_json(soma30_spec, read_file(data / "g1_04_ee_constraint.json"));
    check(!wrong && wrong.error().find("joints") != std::string::npos, "G1 file rejected for SOMA: " + (wrong ? std::string() : wrong.error()));

    // Synthetic 77-joint SOMA rotations reduce with the upstream slice.
    {
        constexpr std::array<std::size_t, 30> slice{0,1,2,3,4,5,6,8,9,10,11,12,13,14,18,28,39,40,41,42,46,56,67,68,69,70,72,73,74,75};
        std::array<std::array<double, 3>, 77> aa{};
        for (auto &v : aa) v = {2.5, -1.0, .3};
        for (std::size_t j = 0; j < 30; ++j) aa[slice[j]] = {.01*double(j), -.02*double(j), .005*double(j)+.1};
        std::string text = R"([{"type":"fullbody","frame_indices":[3.0],"local_joints_rot":[[)";
        for (std::size_t i = 0; i < 77; ++i) text += (i ? "," : "") + std::string("[") + std::to_string(aa[i][0]) + "," + std::to_string(aa[i][1]) + "," + std::to_string(aa[i][2]) + "]";
        text += R"(]],"root_positions":[[0.1,0.9,-0.2]]}])";
        auto parsed = parse_constraints_json(soma30_spec, text);
        check(parsed.has_value() && parsed->size() == 1 && (*parsed)[0].frame_indices == std::vector<unsigned>{3}, "77-joint SOMA parse " + (parsed ? "" : parsed.error()));
        if (parsed && parsed->size() == 1) {
            double worst = 0;
            for (std::size_t j = 0; j < 30; ++j)
                worst = std::max(worst, matrix_distance(rodrigues(aa[slice[j]][0], aa[slice[j]][1], aa[slice[j]][2]), quaternion_matrix((*parsed)[0].local_rotations_xyzw.data()+j*4)));
            check(worst <= 1e-5, "77 -> 30 joint reduction, max error " + std::to_string(worst));
            check((*parsed)[0].local_rotations_xyzw.size() == 30*4 && (*parsed)[0].smooth_root_2d.empty(), "77 -> 30 sizes; smooth root left to default");
            const auto written = constraints_to_json(soma30_spec, *parsed);
            auto again = parse_constraints_json(soma30_spec, written);
            check(again && again->size() == 1 && (*again)[0].smooth_root_2d == std::vector<float>{.1F, -.2F}, "export writes 30 joints and the default smooth root");
        }
        check(!parse_constraints_json(smplx22_spec, text) && !parse_constraints_json(g1skel34_spec, text), "77 joints only accepted for SOMA");
    }
    // Malformed inputs are rejected with a reason.
    const std::string pose22 = [] {
        std::string p = "[["; for (int j = 0; j < 22; ++j) p += (j ? "," : "") + std::string("[0,0,0]"); return p + "]]";
    }();
    const std::pair<std::string, std::string> bad[]{
        {"not json", "invalid constraints JSON"},
        {"{}", "must be an array"},
        {R"([{"type":"end-effectors","frame_indices":[0]}])", "unknown constraint type"},
        {R"([{"frame_indices":[0]}])", "type"},
        {R"([{"type":"root2d","frame_indices":[-1],"smooth_root_2d":[[0,0]]}])", "non-negative integers"},
        {R"([{"type":"root2d","frame_indices":[2.5],"smooth_root_2d":[[0,0]]}])", "non-negative integers"},
        {R"([{"type":"root2d","frame_indices":[0]}])", "smooth_root_2d"},
        {R"([{"type":"root2d","frame_indices":[0],"smooth_root_2d":[[0,0,0]]}])", "smooth_root_2d"},
        {R"([{"type":"root2d","frame_indices":[0,1],"smooth_root_2d":[[0,0]]}])", "smooth_root_2d"},
        {R"([{"type":"root2d","frame_indices":[0],"smooth_root_2d":[[0,0]],"global_root_heading":[[1]]}])", "global_root_heading"},
        {R"([{"type":"fullbody","frame_indices":[0],"local_joints_rot":[[[0,0,0]]],"root_positions":[[0,1,0]]}])", "joints"},
        {R"([{"type":"fullbody","frame_indices":[0],"local_joints_rot":)" + pose22 + "}]", "root_positions"},
        {R"([{"type":"end-effector","frame_indices":[0],"local_joints_rot":)" + pose22 + R"(,"root_positions":[[0,1,0]]}])", "joint_names"},
        {R"([{"type":"end-effector","frame_indices":[0],"local_joints_rot":)" + pose22 + R"(,"root_positions":[[0,1,0]],"joint_names":["left_hand"]}])", "unknown end-effector joint"},
        {R"([{"type":"left-hand","frame_indices":[0],"local_joints_rot":)" + pose22 + R"(,"root_positions":[[0,1,0]],"smooth_root_2d":[[0]]}])", "smooth_root_2d"},
    };
    for (const auto &[text, reason] : bad) {
        auto parsed = parse_constraints_json(smplx22_spec, text);
        check(!parsed && parsed.error().find(reason) != std::string::npos,
              "reject " + text.substr(0, 60) + " -> " + (parsed ? std::string("accepted") : parsed.error()));
    }
    auto good = parse_constraints_json(smplx22_spec, R"([{"type":"left-hand","frame_indices":[4.0],"local_joints_rot":)" + pose22 +
                                                      R"(,"root_positions":[[0,1,0]],"joint_names":["ignored like upstream"]}])");
    check(good && (*good)[0].joint_names.empty() && (*good)[0].frame_indices == std::vector<unsigned>{4}, "sided end-effector ignores joint_names; 4.0 is a frame");
}

// (g) Structural validation of constraint_set values.
void test_validation() {
    const auto &s = smplx22_spec;
    auto expect_error = [&](const constraint_set &c, std::size_t frames, const std::string &reason) {
        const std::array<constraint_set, 1> list{c};
        auto valid = validate_constraints(s, list, frames);
        check(!valid && valid.error().find(reason) != std::string::npos, "validation '" + reason + "' -> " + (valid ? std::string("accepted") : valid.error()));
        auto built = build_conditions(s, list, frames);
        check(!built, "build_conditions rejects '" + reason + "'");
    };
    const auto ok = posed(s, constraint_type::fullbody, {0, 9});
    { const std::array<constraint_set, 1> list{ok}; check(validate_constraints(s, list, 10).has_value(), "valid full-body"); }
    expect_error(ok, 9, "outside the 9-frame timeline");
    auto c = ok; c.local_rotations_xyzw.pop_back(); expect_error(c, 10, "quaternions per frame");
    c = ok; c.root_positions.push_back(0.F); expect_error(c, 10, "root_positions");
    c = ok; c.root_positions[1] = std::numeric_limits<float>::quiet_NaN(); expect_error(c, 10, "non-finite");
    c = ok; std::fill_n(c.local_rotations_xyzw.begin()+8, 4, 0.F); expect_error(c, 10, "zero quaternion");
    c = ok; c.joint_names = {"LeftHand"}; expect_error(c, 10, "joint_names");
    c = ok; c.global_root_heading = {1, 0, 1, 0}; expect_error(c, 10, "global_root_heading");
    c = ok; c.smooth_root_2d = {0, 0, 0}; expect_error(c, 10, "smooth_root_2d");
    c = ok; c.type = constraint_type::end_effector; c.joint_names = {"LeftHand", "Head"}; expect_error(c, 10, "unknown end-effector joint 'Head'");
    constraint_set r; r.type = constraint_type::root2d; r.frame_indices = {1}; expect_error(r, 10, "smooth_root_2d is required");
    r.smooth_root_2d = {0, 0}; r.root_positions = {0, 1, 0}; expect_error(r, 10, "root2d takes no");
    // Non-unit (but non-zero) quaternions are normalized by FK.
    c = ok; for (auto &v : c.local_rotations_xyzw) v *= 3.F;
    const std::array<constraint_set, 1> scaled{c}, unit{ok};
    auto a = build_conditions(s, scaled, 10), b = build_conditions(s, unit, 10);
    double worst = 0; if (a && b) for (std::size_t i = 0; i < a->observed.size(); ++i) worst = std::max(worst, std::abs(double(a->observed[i]) - b->observed[i]));
    check(a && b && worst <= 1e-5, "scaled quaternions are normalized");
}
} // namespace

int main(int argc, char **argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: %s TESTS/DATA/CONSTRAINTS\n", argv[0]); return 2; }
    test_masks();
    test_condition_row_parity();
    test_root2d();
    test_merge_rules();
    test_crop();
    test_json(argv[1]);
    test_validation();
    if (failures) { std::fprintf(stderr, "%d constraint check(s) failed\n", failures); return 1; }
    std::printf("constraints: all %d checks passed\n", checks);
    return 0;
}
