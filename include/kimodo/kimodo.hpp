#pragma once

#include <kimodo/kimodo_capi.h>

#include <array>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace kimodo {

inline constexpr unsigned embedding_width = 4096;

struct motion_data {
    unsigned frames = 0;
    unsigned joints = 0;
    std::vector<float> local_rotations_xyzw;   // [T,J,4]
    std::vector<float> root_positions;         // [T,3]
    // Decoded from the final feature rows (re-encoded rows when post-processing
    // ran); filled by every generation entry point.
    std::vector<float> foot_contacts;          // [T,4] 0/1: left heel, left toe, right heel, right toe
    std::vector<float> smooth_root_positions;  // [T,3] model smooth root (x, y, z)
};

// Kinematic constraint kinds of NVIDIA Kimodo's constraints.json.
enum class constraint_type { root2d, fullbody, left_hand, right_hand, left_foot, right_foot, end_effector };

// One upstream constraint set on the model's control skeleton (22 SMPL-X,
// 30 SOMA or 34 G1 joints). Frame indices are global over the whole
// multi-prompt timeline at 30 fps; K = frame_indices.size().
struct constraint_set {
    constraint_type type = constraint_type::fullbody;
    std::vector<unsigned> frame_indices;       // [K]
    std::vector<float> local_rotations_xyzw;   // [K,J,4]  all types except root2d
    std::vector<float> root_positions;         // [K,3]    all types except root2d
    std::vector<float> smooth_root_2d;         // [K,2] (x,z); required for root2d, else optional (pelvis x,z)
    std::vector<float> global_root_heading;    // [K,2] (cos,sin); optional, root2d only
    std::vector<std::string> joint_names;      // end_effector only: LeftFoot, RightFoot, LeftHand, RightHand, Hips
};

struct generation_options {
    unsigned transition_frames = 5;
    unsigned steps = 100;
    std::uint64_t seed = 0;
    float text_cfg = 2.F;
    float constraint_cfg = 2.F;
    // Upstream foot-skate cleanup + constraint enforcement (NVIDIA
    // MotionCorrection). Ignored for G1 (g1skel34) models: like upstream's CLI
    // and demo, kimodo.cpp never post-processes G1 motions. Builds without the
    // solver (see post_processing_available) reject requests that set it.
    bool post_processing = false;
    float root_margin = .04F;
};

struct prompt_segment {
    std::string prompt;
    unsigned frames = 0;
};

// A generation request in the kmd-generate protocol v2 JSON form (see
// docs/IMPLEMENTATION.md): {"output", "seed", "steps", "transition_frames",
// "text_cfg", "constraint_cfg", "post_processing", "root_margin",
// "segments": [{"frames", "prompt" | "prompt_file"}],
// "constraints": [...] | "constraints_file"}. Omitted fields keep the
// generation_options defaults.
struct generation_request {
    std::string output;                        // "output" (empty when absent)
    std::vector<prompt_segment> segments;
    std::vector<constraint_set> constraints;
    generation_options options;
};

// True when this build includes the MotionCorrection post-processor (x86
// targets with KIMODO_ENABLE_POSTPROCESS=ON).
KIMODO_API bool post_processing_available() noexcept;

class KIMODO_API model {
public:
    static std::expected<std::unique_ptr<model>, std::string> load(
        std::string_view motion_gguf, std::string_view text_bundle = {});
    std::expected<motion_data, std::string> generate_embedding(
        const std::array<float, embedding_width> &embedding,
        unsigned frames, unsigned steps, std::uint64_t seed,
        float text_cfg, float constraint_cfg) const;
    std::expected<motion_data, std::string> generate_text(
        std::string_view utf8_prompt, unsigned frames, unsigned steps, std::uint64_t seed,
        float text_cfg, float constraint_cfg) const;
    std::expected<motion_data, std::string> generate_text_sequence(
        std::span<const prompt_segment> segments, unsigned transition_frames,
        unsigned steps, std::uint64_t seed, float text_cfg, float constraint_cfg) const;
    // Text-to-motion over one or more prompt segments with optional kinematic
    // constraints and post-processing. Without constraints and post-processing
    // this reproduces generate_text_sequence exactly.
    std::expected<motion_data, std::string> generate(
        std::span<const prompt_segment> segments, std::span<const constraint_set> constraints,
        const generation_options &options) const;
    // Upstream constraints.json (a JSON array) <-> constraint sets for this
    // model's skeleton. SOMA files with 77-joint rotations are accepted.
    std::expected<std::vector<constraint_set>, std::string> parse_constraints(std::string_view json) const;
    std::expected<std::string, std::string> constraints_to_json(std::span<const constraint_set> constraints) const;
    // Parses a generation_request JSON object; prompt and constraint files are
    // read immediately (UTF-8, an optional byte order mark is ignored).
    std::expected<generation_request, std::string> parse_request(std::string_view json) const;
    // Control-skeleton key: "smplx22", "soma30" or "g1skel34".
    std::string_view skeleton_key() const noexcept;
    ~model();
    model(const model &) = delete;
    model &operator=(const model &) = delete;
private:
    struct impl;
    explicit model(std::unique_ptr<impl> impl);
    std::unique_ptr<impl> impl_;
};

} // namespace kimodo
