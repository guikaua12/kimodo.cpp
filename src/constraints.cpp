// Kinematic constraints: NVIDIA Kimodo constraint sets -> model conditions.
// Semantics follow kimodo/constraints.py and KimodoMotionRep.create_conditions
// (Apache-2.0); see docs/IMPLEMENTATION.md, "Constraints".
#include "constraints.hpp"
#include "json.hpp"
#include "skeleton.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

namespace kimodo::detail {
namespace {
// Row-major 3x3 rotation in the column-vector convention (x' = R x), double
// precision like sequence.cpp's condition_row.
struct mat3 { double v[9]; };
mat3 mul(const mat3 &a, const mat3 &b) { mat3 r{}; for (int i=0;i<3;++i) for (int j=0;j<3;++j) for (int k=0;k<3;++k) r.v[i*3+j]+=a.v[i*3+k]*b.v[k*3+j]; return r; }
mat3 transpose(const mat3 &a) { mat3 r{}; for (int i=0;i<3;++i) for (int j=0;j<3;++j) r.v[i*3+j]=a.v[j*3+i]; return r; }
void rotate(const mat3 &m, const std::array<float,3> &x, float *o) { for (int i=0;i<3;++i) o[i]=static_cast<float>(m.v[i*3]*x[0]+m.v[i*3+1]*x[1]+m.v[i*3+2]*x[2]); }

// xyzw quaternion (any non-zero norm) -> rotation matrix.
mat3 quaternion_matrix(const float *q) {
    double x=q[0], y=q[1], z=q[2], w=q[3];
    const double n=std::sqrt(x*x+y*y+z*z+w*w); x/=n; y/=n; z/=n; w/=n;
    return {{1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
             2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
             2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)}};
}

// kimodo/geometry.py matrix_to_quaternion (pytorch3d best-candidate); wxyz.
std::array<double,4> matrix_quaternion_wxyz(const mat3 &m) {
    const double m00=m.v[0],m01=m.v[1],m02=m.v[2],m10=m.v[3],m11=m.v[4],m12=m.v[5],m20=m.v[6],m21=m.v[7],m22=m.v[8];
    const double raw[4]{1+m00+m11+m22, 1+m00-m11-m22, 1-m00+m11-m22, 1-m00-m11+m22};
    double a[4]; for (int i=0;i<4;++i) a[i]=raw[i]>0 ? std::sqrt(raw[i]) : 0.;
    const double candidates[4][4]{{a[0]*a[0], m21-m12, m02-m20, m10-m01},
                                  {m21-m12, a[1]*a[1], m10+m01, m02+m20},
                                  {m02-m20, m10+m01, a[2]*a[2], m12+m21},
                                  {m10-m01, m20+m02, m21+m12, a[3]*a[3]}};
    int best=0; for (int i=1;i<4;++i) if (a[i]>a[best]) best=i;
    const double scale=2.*std::max(a[best], .1);
    return {candidates[best][0]/scale, candidates[best][1]/scale, candidates[best][2]/scale, candidates[best][3]/scale};
}

// kimodo/geometry.py axis_angle_to_matrix (Rodrigues, axis = v / (|v| + 1e-6)).
mat3 axis_angle_matrix(const double *v) {
    const double angle=std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
    const double x=v[0]/(angle+1e-6), y=v[1]/(angle+1e-6), z=v[2]/(angle+1e-6);
    const mat3 k{{0,-z,y, z,0,-x, -y,x,0}}, kk=mul(k,k);
    const double s=std::sin(angle), c=1-std::cos(angle);
    mat3 r{}; for (int i=0;i<9;++i) r.v[i]=(i%4==0 ? 1. : 0.)+s*k.v[i]+c*kk.v[i];
    return r;
}

// kimodo/geometry.py matrix_to_axis_angle (canonical w >= 0 quaternion).
std::array<double,3> matrix_axis_angle(const mat3 &m) {
    auto q=matrix_quaternion_wxyz(m);
    constexpr double eps=1e-6;
    if (q[0] < -eps || (std::abs(q[0]) <= eps && q[1] < 0)) for (double &value : q) value=-value;
    const double sin_half=std::sqrt(q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    const double angle=2.*std::atan2(sin_half, q[0]);
    const double scale=sin_half < eps ? 2. : angle/std::max(sin_half, eps);
    return {q[1]*scale, q[2]*scale, q[3]*scale};
}

// Matrix -> unit xyzw quaternion.
void matrix_quaternion_xyzw(const mat3 &m, float *out) {
    const auto q=matrix_quaternion_wxyz(m);
    const double n=std::sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    out[0]=static_cast<float>(q[1]/n); out[1]=static_cast<float>(q[2]/n);
    out[2]=static_cast<float>(q[3]/n); out[3]=static_cast<float>(q[0]/n);
}

// Upstream SOMASkeleton30.get_skel_slice(SOMASkeleton77): the 77-joint index
// of every control joint, in control-skeleton order.
constexpr std::array<std::size_t,30> soma30_from_soma77{
    0,1,2,3,4,5,6,8,9,10,11,12,13,14,18,28,39,40,41,42,46,56,67,68,69,70,72,73,74,75};

constexpr std::array<std::string_view,7> type_names{
    "root2d", "fullbody", "left-hand", "right-hand", "left-foot", "right-foot", "end-effector"};
constexpr std::array<std::string_view,5> end_effector_names{"LeftFoot", "RightFoot", "LeftHand", "RightHand", "Hips"};

std::string_view type_name(constraint_type type) { return type_names[static_cast<std::size_t>(type)]; }

std::string label(std::size_t index, constraint_type type) {
    return "constraint " + std::to_string(index) + " (" + std::string(type_name(type)) + ")";
}

bool finite(const std::vector<float> &values) {
    return std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); });
}

// Limb index in skeleton_spec::chains order (LeftFoot, RightFoot, LeftHand,
// RightHand) or 4 for Hips (the root joint).
std::vector<std::size_t> end_effector_limbs(const constraint_set &c) {
    switch (c.type) {
    case constraint_type::left_foot: return {0};
    case constraint_type::right_foot: return {1};
    case constraint_type::left_hand: return {2};
    case constraint_type::right_hand: return {3};
    case constraint_type::end_effector: {
        std::vector<std::size_t> limbs;
        for (const auto &name : c.joint_names) {
            const auto found=std::find(end_effector_names.begin(), end_effector_names.end(), name);
            if (found != end_effector_names.end()) limbs.push_back(static_cast<std::size_t>(found-end_effector_names.begin()));
        }
        return limbs;
    }
    default: return {};
    }
}

// Upstream expand_joint_names: positions for every chain joint, rotation for
// the chain base; "Hips" constrains the root position and rotation.
void expand_limbs(const skeleton_spec &s, const std::vector<std::size_t> &limbs,
                  std::vector<std::size_t> &positions, std::vector<std::size_t> &rotations) {
    for (std::size_t limb : limbs) {
        if (limb == 4) { positions.push_back(0); rotations.push_back(0); continue; }
        for (int joint : s.chains[limb]) if (joint >= 0) positions.push_back(static_cast<std::size_t>(joint));
        rotations.push_back(static_cast<std::size_t>(s.chains[limb][0]));
    }
}

json::value number(double value) { json::value v; v.type=json::value::kind::number; v.number=value; return v; }
// Shortest decimal that round-trips the float, stored as the matching double.
json::value number(float value) {
    char buffer[32];
    const auto written=std::to_chars(buffer, buffer+sizeof(buffer), value);
    double parsed=value;
    std::from_chars(buffer, written.ptr, parsed);
    return number(parsed);
}
json::value string_value(std::string_view text) { json::value v; v.type=json::value::kind::string; v.string=std::string(text); return v; }
json::value array_value() { json::value v; v.type=json::value::kind::array; return v; }

std::unexpected<std::string> fail(std::size_t index, std::string_view message) {
    return std::unexpected("constraints JSON item " + std::to_string(index) + ": " + std::string(message));
}

// Reads an array of exactly `rows` arrays of `cols` numbers.
bool read_matrix(const json::value *v, std::size_t rows, std::size_t cols, std::vector<double> &out) {
    if (!v || !v->is_array() || v->array.size() != rows) return false;
    out.clear(); out.reserve(rows*cols);
    for (const auto &row : v->array) {
        if (!row.is_array() || row.array.size() != cols) return false;
        for (const auto &item : row.array) { if (!item.is_number()) return false; out.push_back(item.number); }
    }
    return true;
}
} // namespace

bool motion_conditions::any() const noexcept {
    return std::any_of(mask.begin(), mask.end(), [](float value) { return value != 0.F; });
}

bool motion_conditions::frame_constrained(std::size_t frame, std::size_t motion_dim) const noexcept {
    if (!motion_dim || frame >= frames || (frame+1)*motion_dim > mask.size()) return false;
    const auto row=mask.begin()+static_cast<std::ptrdiff_t>(frame*motion_dim);
    return std::any_of(row, row+static_cast<std::ptrdiff_t>(motion_dim), [](float value) { return value != 0.F; });
}

// Same accumulation as sequence.cpp condition_row: double rotations, offsets
// rotated in double and rounded to float, positions summed in float.
void forward_kinematics(const skeleton_spec &s, const float *local_xyzw, const float *root,
                        float *positions, float *rotations) {
    const std::size_t J=s.joints();
    std::vector<mat3> global(J);
    std::vector<std::array<float,3>> posed(J);
    for (std::size_t j=0;j<J;++j) {
        const mat3 local=quaternion_matrix(local_xyzw+j*4);
        const int parent=s.parents[j];
        if (parent < 0) { global[j]=local; posed[j]={root[0],root[1],root[2]}; continue; }
        const auto p=static_cast<std::size_t>(parent);
        global[j]=mul(global[p], local);
        float offset[3]; rotate(global[p], s.offsets[j], offset);
        for (int k=0;k<3;++k) posed[j][static_cast<std::size_t>(k)]=posed[p][static_cast<std::size_t>(k)]+offset[k];
    }
    for (std::size_t j=0;j<J;++j) {
        if (positions) for (std::size_t k=0;k<3;++k) positions[j*3+k]=posed[j][k];
        if (rotations) for (std::size_t k=0;k<9;++k) rotations[j*9+k]=static_cast<float>(global[j].v[k]);
    }
}

std::expected<void, std::string> validate_constraints(
    const skeleton_spec &s, std::span<const constraint_set> constraints, std::size_t frames) {
    const std::size_t J=s.joints();
    for (std::size_t index=0; index<constraints.size(); ++index) {
        const auto &c=constraints[index];
        const auto type_index=static_cast<std::size_t>(c.type);
        if (type_index >= type_names.size()) return std::unexpected("constraint " + std::to_string(index) + ": unknown constraint type");
        const std::string name=label(index, c.type);
        const std::size_t K=c.frame_indices.size();
        for (unsigned frame : c.frame_indices)
            if (frame >= frames)
                return std::unexpected(name + ": frame index " + std::to_string(frame) + " is outside the " +
                                       std::to_string(frames) + "-frame timeline");
        if (!c.smooth_root_2d.empty() && c.smooth_root_2d.size() != K*2)
            return std::unexpected(name + ": smooth_root_2d must hold 2 values per frame");
        if (!finite(c.smooth_root_2d)) return std::unexpected(name + ": smooth_root_2d contains a non-finite value");
        if (c.type != constraint_type::end_effector && !c.joint_names.empty())
            return std::unexpected(name + ": joint_names is only valid for end-effector constraints");
        if (c.type == constraint_type::root2d) {
            if (c.smooth_root_2d.size() != K*2) return std::unexpected(name + ": smooth_root_2d is required (2 values per frame)");
            if (!c.local_rotations_xyzw.empty() || !c.root_positions.empty())
                return std::unexpected(name + ": root2d takes no joint rotations or root positions");
            if (!c.global_root_heading.empty() && c.global_root_heading.size() != K*2)
                return std::unexpected(name + ": global_root_heading must hold 2 values (cos, sin) per frame");
            if (!finite(c.global_root_heading)) return std::unexpected(name + ": global_root_heading contains a non-finite value");
            continue;
        }
        if (!c.global_root_heading.empty()) return std::unexpected(name + ": global_root_heading is only valid for root2d constraints");
        if (c.local_rotations_xyzw.size() != K*J*4)
            return std::unexpected(name + ": local rotations must hold " + std::to_string(J) + " xyzw quaternions per frame");
        if (c.root_positions.size() != K*3) return std::unexpected(name + ": root_positions must hold 3 values per frame");
        if (!finite(c.local_rotations_xyzw)) return std::unexpected(name + ": local rotations contain a non-finite value");
        if (!finite(c.root_positions)) return std::unexpected(name + ": root_positions contains a non-finite value");
        for (std::size_t q=0; q<K*J; ++q) {
            const float *x=c.local_rotations_xyzw.data()+q*4;
            const double norm=std::sqrt(double(x[0])*x[0]+double(x[1])*x[1]+double(x[2])*x[2]+double(x[3])*x[3]);
            if (!(norm > 1e-6)) return std::unexpected(name + ": zero quaternion at frame " + std::to_string(q/J) + " joint " + std::to_string(q%J));
        }
        if (c.type == constraint_type::end_effector) {
            for (const auto &joint : c.joint_names)
                if (std::find(end_effector_names.begin(), end_effector_names.end(), joint) == end_effector_names.end())
                    return std::unexpected(name + ": unknown end-effector joint '" + joint +
                                           "' (expected LeftFoot, RightFoot, LeftHand, RightHand or Hips)");
        }
    }
    return {};
}

std::expected<motion_conditions, std::string> build_conditions(
    const skeleton_spec &s, std::span<const constraint_set> constraints, std::size_t frames) {
    if (auto valid=validate_constraints(s, constraints, frames); !valid) return std::unexpected(valid.error());
    const std::size_t D=s.motion_dim(), J=s.joints(), rotation_begin=5+3*J;
    motion_conditions out;
    out.frames=frames;
    out.observed.assign(frames*D, 0.F);
    out.mask.assign(frames*D, 0.F);
    auto write=[&](std::size_t frame, std::size_t feature, float value) { out.observed[frame*D+feature]=value; out.mask[frame*D+feature]=1.F; };

    // Global positions [K,J,3], rotations [K,J,9] and heading (cos, sin) of every posed keyframe.
    struct posed_set { std::vector<float> positions, rotations, heading; };
    std::vector<posed_set> posed(constraints.size());
    for (std::size_t index=0; index<constraints.size(); ++index) {
        const auto &c=constraints[index];
        if (c.type == constraint_type::root2d) continue;
        const std::size_t K=c.frame_indices.size();
        auto &p=posed[index];
        p.positions.resize(K*J*3); p.rotations.resize(K*J*9); p.heading.resize(K*2);
        for (std::size_t k=0;k<K;++k) {
            float *positions=p.positions.data()+k*J*3;
            forward_kinematics(s, c.local_rotations_xyzw.data()+k*J*4, c.root_positions.data()+k*3, positions, p.rotations.data()+k*J*9);
            const auto right=s.hips[0], left=s.hips[1];
            const float angle=std::atan2(positions[right*3+2]-positions[left*3+2], -(positions[right*3]-positions[left*3]));
            p.heading[k*2]=std::cos(angle); p.heading[k*2+1]=std::sin(angle);
        }
    }
    // Upstream create_conditions key order; within a key, values are applied
    // in append order so the last duplicate wins (PyTorch CPU scatter).
    for (const auto &c : constraints)                                                     // smooth_root_2d
        for (std::size_t k=0;k<c.frame_indices.size();++k) {
            const bool given=!c.smooth_root_2d.empty();
            write(c.frame_indices[k], 0, given ? c.smooth_root_2d[k*2] : c.root_positions[k*3]);
            write(c.frame_indices[k], 2, given ? c.smooth_root_2d[k*2+1] : c.root_positions[k*3+2]);
        }
    for (const auto &c : constraints)                                                     // root_y_pos
        if (c.type != constraint_type::root2d)
            for (std::size_t k=0;k<c.frame_indices.size();++k) write(c.frame_indices[k], 1, c.root_positions[k*3+1]);
    for (std::size_t index=0; index<constraints.size(); ++index) {                        // global_root_heading
        const auto &c=constraints[index];
        const auto &heading=c.type == constraint_type::root2d ? c.global_root_heading : posed[index].heading;
        if (heading.empty()) continue;
        for (std::size_t k=0;k<c.frame_indices.size();++k) { write(c.frame_indices[k], 3, heading[k*2]); write(c.frame_indices[k], 4, heading[k*2+1]); }
    }
    for (std::size_t index=0; index<constraints.size(); ++index) {                        // global_joints_rots
        const auto &c=constraints[index];
        std::vector<std::size_t> positions, rotations;
        expand_limbs(s, end_effector_limbs(c), positions, rotations);
        for (std::size_t k=0;k<c.frame_indices.size();++k)
            for (std::size_t joint : rotations) {
                const float *r=posed[index].rotations.data()+(k*J+joint)*9;
                // 6D = first two columns of the row-major matrix.
                for (std::size_t d=0;d<6;++d) write(c.frame_indices[k], rotation_begin+joint*6+d, r[(d%3)*3+d/3]);
            }
    }
    for (std::size_t index=0; index<constraints.size(); ++index) {                        // global_joints_positions
        const auto &c=constraints[index];
        if (c.type == constraint_type::root2d) continue;
        std::vector<std::size_t> positions, rotations;
        if (c.type == constraint_type::fullbody) for (std::size_t j=0;j<J;++j) positions.push_back(j);
        else expand_limbs(s, end_effector_limbs(c), positions, rotations);
        for (std::size_t k=0;k<c.frame_indices.size();++k) {
            const std::size_t frame=c.frame_indices[k];
            if (positions.empty()) continue;
            if (out.mask[frame*D] == 0.F || out.mask[frame*D+2] == 0.F)
                return std::unexpected("for constraining global positions, the smooth root should also be constrained (frame " + std::to_string(frame) + ")");
            const float sx=out.observed[frame*D], sz=out.observed[frame*D+2];
            for (std::size_t joint : positions) {
                const float *p=posed[index].positions.data()+(k*J+joint)*3;
                write(frame, 5+joint*3, p[0]-sx); write(frame, 6+joint*3, p[1]); write(frame, 7+joint*3, p[2]-sz);
            }
        }
    }
    return out;
}

std::vector<constraint_set> crop_constraints(std::span<const constraint_set> constraints, std::size_t begin, std::size_t end) {
    std::vector<constraint_set> out;
    out.reserve(constraints.size());
    for (const auto &c : constraints) {
        constraint_set cropped;
        cropped.type=c.type;
        cropped.joint_names=c.joint_names;
        const std::size_t K=c.frame_indices.size();
        const std::size_t rotation_stride=K ? c.local_rotations_xyzw.size()/K : 0;
        auto copy=[](const std::vector<float> &from, std::vector<float> &to, std::size_t k, std::size_t stride) {
            if (stride && from.size() >= (k+1)*stride) to.insert(to.end(), from.begin()+static_cast<std::ptrdiff_t>(k*stride), from.begin()+static_cast<std::ptrdiff_t>((k+1)*stride));
        };
        for (std::size_t k=0;k<K;++k) {
            const std::size_t frame=c.frame_indices[k];
            if (frame < begin || frame >= end) continue;
            cropped.frame_indices.push_back(static_cast<unsigned>(frame-begin));
            copy(c.local_rotations_xyzw, cropped.local_rotations_xyzw, k, rotation_stride);
            copy(c.root_positions, cropped.root_positions, k, c.root_positions.empty() ? 0 : 3);
            copy(c.smooth_root_2d, cropped.smooth_root_2d, k, c.smooth_root_2d.empty() ? 0 : 2);
            copy(c.global_root_heading, cropped.global_root_heading, k, c.global_root_heading.empty() ? 0 : 2);
        }
        out.push_back(std::move(cropped));
    }
    return out;
}

std::expected<std::vector<constraint_set>, std::string> parse_constraints_json(const skeleton_spec &s, std::string_view text) {
    auto document=json::parse(text);
    if (!document) return std::unexpected("invalid constraints JSON: " + document.error());
    return parse_constraints_json(s, *document);
}

std::expected<std::vector<constraint_set>, std::string> parse_constraints_json(const skeleton_spec &s, const json::value &root) {
    const json::value *document=&root;
    if (!document->is_array()) return std::unexpected("constraints JSON must be an array of constraint objects");
    const std::size_t J=s.joints();
    const bool soma=s.key == "soma30";
    std::vector<constraint_set> result;
    result.reserve(document->array.size());
    std::vector<double> values;
    for (std::size_t index=0; index<document->array.size(); ++index) {
        const auto &item=document->array[index];
        if (!item.is_object()) return fail(index, "expected an object");
        const auto *type=item.find("type");
        if (!type || !type->is_string()) return fail(index, "missing string \"type\"");
        const auto found=std::find(type_names.begin(), type_names.end(), type->string);
        if (found == type_names.end())
            return fail(index, "unknown constraint type '" + type->string +
                               "' (expected root2d, fullbody, left-hand, right-hand, left-foot, right-foot or end-effector)");
        constraint_set c;
        c.type=static_cast<constraint_type>(found-type_names.begin());
        const auto *frames=item.find("frame_indices");
        if (!frames || !frames->is_array()) return fail(index, "missing array \"frame_indices\"");
        for (const auto &frame : frames->array) {
            if (!frame.is_number() || frame.number < 0 || frame.number != std::floor(frame.number) ||
                frame.number > double(std::numeric_limits<unsigned>::max()))
                return fail(index, "frame_indices must be non-negative integers");
            c.frame_indices.push_back(static_cast<unsigned>(frame.number));
        }
        const std::size_t K=c.frame_indices.size();
        const auto *smooth=item.find("smooth_root_2d");
        if (smooth && !smooth->is_null()) {
            if (!read_matrix(smooth, K, 2, values)) return fail(index, "smooth_root_2d must be a [K][2] array of numbers");
            for (double value : values) c.smooth_root_2d.push_back(static_cast<float>(value));
        }
        if (c.type == constraint_type::root2d) {
            if (c.smooth_root_2d.size() != K*2) return fail(index, "root2d requires smooth_root_2d");
            const auto *heading=item.find("global_root_heading");
            if (heading && !heading->is_null()) {
                if (!read_matrix(heading, K, 2, values)) return fail(index, "global_root_heading must be a [K][2] array of numbers");
                for (double value : values) c.global_root_heading.push_back(static_cast<float>(value));
            }
            result.push_back(std::move(c));
            continue;
        }
        // Posed types: local axis-angle rotations [K][J'][3] and pelvis positions [K][3].
        const auto *rotations=item.find("local_joints_rot");
        if (!rotations || !rotations->is_array() || rotations->array.size() != K)
            return fail(index, "local_joints_rot must be a [K][J][3] array");
        const std::size_t file_joints=K ? (rotations->array[0].is_array() ? rotations->array[0].array.size() : 0) : J;
        if (file_joints != J && !(soma && file_joints == 77))
            return fail(index, "local_joints_rot has " + std::to_string(file_joints) + " joints; the " + std::string(s.key) +
                               " skeleton needs " + std::to_string(J) + (soma ? " (or 77 SOMA joints)" : ""));
        c.local_rotations_xyzw.resize(K*J*4);
        for (std::size_t k=0;k<K;++k) {
            if (!read_matrix(&rotations->array[k], file_joints, 3, values))
                return fail(index, "local_joints_rot must be a [K][J][3] array of numbers");
            for (std::size_t j=0;j<J;++j) {
                const std::size_t source=file_joints == J ? j : soma30_from_soma77[j];
                matrix_quaternion_xyzw(axis_angle_matrix(values.data()+source*3), c.local_rotations_xyzw.data()+(k*J+j)*4);
            }
        }
        if (!read_matrix(item.find("root_positions"), K, 3, values)) return fail(index, "root_positions must be a [K][3] array of numbers");
        for (double value : values) c.root_positions.push_back(static_cast<float>(value));
        if (c.type == constraint_type::end_effector) {
            const auto *names=item.find("joint_names");
            if (!names || !names->is_array()) return fail(index, "end-effector requires a \"joint_names\" array");
            for (const auto &name : names->array) {
                if (!name.is_string()) return fail(index, "joint_names must contain strings");
                c.joint_names.push_back(name.string);
            }
        }
        result.push_back(std::move(c));
    }
    if (auto valid=validate_constraints(s, result, std::numeric_limits<std::size_t>::max()); !valid)
        return std::unexpected(valid.error());
    return result;
}

std::string constraints_to_json(const skeleton_spec &s, std::span<const constraint_set> constraints) {
    const std::size_t J=s.joints();
    auto matrix=[](const float *data, std::size_t rows, std::size_t cols) {
        json::value out=array_value();
        for (std::size_t r=0;r<rows;++r) {
            json::value row=array_value();
            for (std::size_t c=0;c<cols;++c) row.array.push_back(number(data[r*cols+c]));
            out.array.push_back(std::move(row));
        }
        return out;
    };
    std::string out="[";
    for (std::size_t index=0; index<constraints.size(); ++index) {
        const auto &c=constraints[index];
        const std::size_t K=c.frame_indices.size();
        json::value item; item.type=json::value::kind::object;
        item.object.emplace_back("type", string_value(type_name(c.type)));
        json::value frames=array_value();
        for (unsigned frame : c.frame_indices) frames.array.push_back(number(double(frame)));
        item.object.emplace_back("frame_indices", std::move(frames));
        if (c.type == constraint_type::root2d) {
            item.object.emplace_back("smooth_root_2d", matrix(c.smooth_root_2d.data(), K, 2));
            if (!c.global_root_heading.empty()) item.object.emplace_back("global_root_heading", matrix(c.global_root_heading.data(), K, 2));
        } else {
            json::value rotations=array_value();
            for (std::size_t k=0;k<K;++k) {
                json::value pose=array_value();
                for (std::size_t j=0;j<J;++j) {
                    const auto aa=matrix_axis_angle(quaternion_matrix(c.local_rotations_xyzw.data()+(k*J+j)*4));
                    json::value joint=array_value();
                    for (double value : aa) joint.array.push_back(number(static_cast<float>(value)));
                    pose.array.push_back(std::move(joint));
                }
                rotations.array.push_back(std::move(pose));
            }
            item.object.emplace_back("local_joints_rot", std::move(rotations));
            item.object.emplace_back("root_positions", matrix(c.root_positions.data(), K, 3));
            std::vector<float> smooth=c.smooth_root_2d;
            if (smooth.empty()) for (std::size_t k=0;k<K;++k) { smooth.push_back(c.root_positions[k*3]); smooth.push_back(c.root_positions[k*3+2]); }
            item.object.emplace_back("smooth_root_2d", matrix(smooth.data(), K, 2));
            if (c.type == constraint_type::end_effector) {
                json::value names=array_value();
                for (const auto &name : c.joint_names) names.array.push_back(string_value(name));
                item.object.emplace_back("joint_names", std::move(names));
            }
        }
        out += index ? ",\n " : "\n ";
        out += json::dump(item);
    }
    out += constraints.empty() ? "]\n" : "\n]\n";
    return out;
}

} // namespace kimodo::detail
