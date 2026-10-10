#include <kimodo/kimodo.hpp>
#include "constraints.hpp"
#include "gguf.hpp"
#include "json.hpp"
#include "skeleton.hpp"
#ifdef KIMODO_HAVE_GGML
#include "ggml_weights.hpp"
#include "denoiser.hpp"
#include "motion_decode.hpp"
#include "postprocess.hpp"
#include "llm_text_encoder.hpp"
#include "profile.hpp"
#endif

#include <chrono>
#include <cmath>
#include <algorithm>
#include <charconv>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <mutex>
#include <random>
#include <system_error>

namespace kimodo {
struct model::impl {
    detail::gguf_file motion;
    std::string motion_path;
    const detail::skeleton_spec *skeleton = nullptr;
#ifdef KIMODO_HAVE_GGML
    mutable std::unique_ptr<detail::ggml_motion_weights> weights;
    std::unique_ptr<detail::llm_text_encoder> text;
    mutable std::mutex inference_mutex;
#endif
};
model::model(std::unique_ptr<impl> state) : impl_(std::move(state)) {}
model::~model() = default;

std::expected<std::unique_ptr<model>, std::string> model::load(std::string_view motion_path, std::string_view text_path) {
    auto file = detail::read_gguf_header(motion_path);
    if (!file) return std::unexpected(file.error());
    if (auto valid = detail::validate_motion_gguf(*file); !valid) return std::unexpected(valid.error());
    auto state = std::make_unique<impl>();
    state->motion = std::move(*file);
    state->motion_path = std::string(motion_path);
    state->skeleton = detail::find_skeleton(state->motion.strings.at("kimodo.skeleton"));
#ifdef KIMODO_HAVE_GGML
    if (!text_path.empty()) {
        auto text = detail::llm_text_encoder::load(text_path);
        if (!text) return std::unexpected(text.error());
        state->text = std::move(*text);
    }
#else
    if (!text_path.empty()) return std::unexpected("Kimodo was built without GGML support");
#endif
    return std::unique_ptr<model>(new model(std::move(state)));
}

std::expected<motion_data, std::string> model::generate_text(
    std::string_view utf8_prompt, unsigned frames, unsigned steps, std::uint64_t seed,
    float text_cfg, float constraint_cfg) const {
#ifdef KIMODO_HAVE_GGML
    if (!impl_->text) return std::unexpected("model was loaded without a native text bundle");
    auto embedding = impl_->text->encode(utf8_prompt);
    if (!embedding) return std::unexpected(embedding.error());
    return generate_embedding(*embedding, frames, steps, seed, text_cfg, constraint_cfg);
#else
    (void) utf8_prompt; (void) frames; (void) steps; (void) seed; (void) text_cfg; (void) constraint_cfg;
    return std::unexpected("Kimodo was built without GGML support");
#endif
}

std::expected<motion_data, std::string> model::generate_embedding(
    const std::array<float, embedding_width> &embedding, unsigned frames, unsigned steps,
    std::uint64_t seed, float text_cfg, float constraint_cfg) const {
    if (frames == 0 || frames > 10000) return std::unexpected("frames must be in 1..10000");
    if (steps == 0 || steps > 1000) return std::unexpected("diffusion_steps must be in 1..1000");
    if (!std::isfinite(text_cfg) || !std::isfinite(constraint_cfg)) return std::unexpected("CFG weights must be finite");
    for (float value : embedding) if (!std::isfinite(value)) return std::unexpected("embedding contains a non-finite value");
#ifdef KIMODO_HAVE_GGML
    const std::lock_guard inference_lock(impl_->inference_mutex);
    const auto generate_started = std::chrono::steady_clock::now();
    // Weight residency is deferred until inference so model-load stays a
    // bounded metadata operation.  The graph integration consumes this exact
    // session; no separate unchecked tensor loader exists in the runtime.
    if (!impl_->weights) {
        const auto weights_started = std::chrono::steady_clock::now();
        auto loaded = detail::ggml_motion_weights::load(impl_->motion_path);
        if (!loaded) return std::unexpected(loaded.error());
        impl_->weights = std::move(*loaded);
        if (detail::profile_enabled())
            std::fprintf(stderr, "profile motion.weights_ready_ms=%.3f\n", detail::profile_elapsed_ms(weights_started));
    }
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> normal(0.f, 1.f);
    const size_t motion_dim=impl_->skeleton->motion_dim();
    std::vector<float> noise(static_cast<size_t>(frames)*motion_dim);
    for (float &value : noise) value = normal(rng);
    const auto sampling_started = std::chrono::steady_clock::now();
    auto sampled = detail::sample_motion_from_noise(*impl_->weights, noise, embedding, frames, steps, text_cfg, constraint_cfg);
    if (!sampled) return std::unexpected(sampled.error());
    const double sampling_ms = detail::profile_elapsed_ms(sampling_started);
    auto global_mean=impl_->weights->f32_values("stats.global_root.mean"), global_std=impl_->weights->f32_values("stats.global_root.std");
    auto body_mean=impl_->weights->f32_values("stats.body.mean"), body_std=impl_->weights->f32_values("stats.body.std");
    if (!global_mean) return std::unexpected(global_mean.error());
    if (!global_std) return std::unexpected(global_std.error());
    if (!body_mean) return std::unexpected(body_mean.error());
    if (!body_std) return std::unexpected(body_std.error());
    const auto decode_started = std::chrono::steady_clock::now();
    auto decoded=detail::decode_motion(*sampled,frames,*impl_->skeleton,*global_mean,*global_std,*body_mean,*body_std);
    if (!decoded) return std::unexpected(decoded.error());
    motion_data result;
    result.frames=frames; result.joints=static_cast<unsigned>(impl_->skeleton->joints());
    result.local_rotations_xyzw=std::move(decoded->local_xyzw);
    result.root_positions=std::move(decoded->root_positions);
    result.foot_contacts=std::move(decoded->foot_contacts);
    result.smooth_root_positions=std::move(decoded->smooth_root);
    if (detail::profile_enabled())
        std::fprintf(stderr, "profile motion.generate sampling_ms=%.3f decode_ms=%.3f total_ms=%.3f\n",
                     sampling_ms, detail::profile_elapsed_ms(decode_started),
                     detail::profile_elapsed_ms(generate_started));
    return result;
#else
    return std::unexpected("Kimodo was built without GGML support");
#endif
}

std::expected<motion_data, std::string> model::generate_text_sequence(
    std::span<const prompt_segment> segments, unsigned transition_frames,
    unsigned steps, std::uint64_t seed, float text_cfg, float constraint_cfg) const {
    generation_options options;
    options.transition_frames = transition_frames;
    options.steps = steps;
    options.seed = seed;
    options.text_cfg = text_cfg;
    options.constraint_cfg = constraint_cfg;
    options.post_processing = false;
    return generate(segments, {}, options);
}

std::expected<motion_data, std::string> model::generate(
    std::span<const prompt_segment> segments, std::span<const constraint_set> constraints,
    const generation_options &options) const {
#ifdef KIMODO_HAVE_GGML
    if (!impl_->text) return std::unexpected("model was loaded without a native text bundle");
    if (segments.empty() || segments.size() > 16) return std::unexpected("sequence requires 1..16 prompt segments");
    const unsigned transition_frames = options.transition_frames, steps = options.steps;
    if (steps == 0 || steps > 1000 || transition_frames == 0 || transition_frames > 60)
        return std::unexpected("invalid sequence sampling parameters");
    if (!std::isfinite(options.text_cfg) || !std::isfinite(options.constraint_cfg)) return std::unexpected("CFG weights must be finite");
    if (!std::isfinite(options.root_margin) || options.root_margin < 0.F) return std::unexpected("root_margin must be finite and >= 0");
    const auto &skeleton = *impl_->skeleton;
    // Upstream's CLI and demo never post-process G1 motions ("does not work
    // well for this model"); requests for it are silently generated without.
    const bool post_processing = options.post_processing && skeleton.key != detail::g1skel34_spec.key;
    // Fail before any text encoding or sampling when the solver is not built in.
    if (post_processing && !detail::post_processing_available())
        return std::unexpected("post-processing is unavailable in this build (x86 only); turn it off (post_processing false)");
    size_t total_frames = 0;
    for (size_t index=0; index<segments.size(); ++index) {
        const auto &segment=segments[index];
        if (segment.prompt.empty() || segment.frames < 2 || segment.frames > 300)
            return std::unexpected("each sequence segment must contain a prompt and have 2..300 frames");
        if (index && transition_frames >= segment.frames) return std::unexpected("transition must be shorter than every following segment");
        // The re-encode after post-processing smooths the root over the whole
        // sampled window, which needs at least 5 frames (upstream splu).
        if (post_processing && segment.frames + (index ? transition_frames : 0) < 5)
            return std::unexpected("post-processing needs at least 5 sampled frames per segment");
        total_frames += segment.frames;
    }
    if (auto valid = detail::validate_constraints(skeleton, constraints, total_frames); !valid) return std::unexpected(valid.error());
    std::vector<std::array<float, embedding_width>> embeddings;
    embeddings.reserve(segments.size());
    for (const auto &segment : segments) {
        auto embedding=impl_->text->encode(segment.prompt);
        if (!embedding) return std::unexpected(embedding.error());
        embeddings.push_back(*embedding);
    }
    const std::lock_guard inference_lock(impl_->inference_mutex);
    // Initialize and warm the quantized text backend before the F32 motion
    // backend applies its process-wide Vulkan parity flags. This preserves
    // cooperative-matrix text kernels in persistent sequence workers.
    if (!impl_->weights) {
        auto loaded = detail::ggml_motion_weights::load(impl_->motion_path);
        if (!loaded) return std::unexpected(loaded.error());
        impl_->weights = std::move(*loaded);
    }
    auto bm=impl_->weights->f32_values("stats.body.mean"), bs=impl_->weights->f32_values("stats.body.std");
    auto gm=impl_->weights->f32_values("stats.global_root.mean"), gs=impl_->weights->f32_values("stats.global_root.std");
    if (!gm || !gs || !bm || !bs) return std::unexpected("motion GGUF lacks normalization statistics");
    std::mt19937_64 rng(options.seed); std::normal_distribution<float> normal(0.f, 1.f);
    std::vector<std::vector<float>> noise;
    std::vector<detail::sampled_sequence_segment> sampled;
    noise.reserve(segments.size()); sampled.reserve(segments.size());
    for (size_t index=0; index<segments.size(); ++index) {
        const auto &segment=segments[index];
        const auto sampled_frames = static_cast<size_t>(segment.frames) +
            (index == 0 ? 0 : transition_frames);
        noise.emplace_back(sampled_frames*impl_->skeleton->motion_dim());
        for (float &value : noise.back()) value=normal(rng);
        sampled.push_back({embeddings[index], noise.back(), segment.frames});
    }
    // User constraints become unnormalized conditions over the whole timeline;
    // the sampler slices them per segment like upstream crop_move.
    detail::sequence_conditions conditions;
    detail::motion_conditions user;
    if (!constraints.empty()) {
        auto built = detail::build_conditions(skeleton, constraints, total_frames);
        if (!built) return std::unexpected(built.error());
        user = std::move(*built);
        if (user.any()) { conditions.observed = user.observed; conditions.mask = user.mask; }
    }
    if (post_processing)
        conditions.post_process = [&](const detail::sequence_segment_rows &rows) -> std::expected<std::vector<float>, std::string> {
            auto decoded = detail::decode_motion_rows(rows.rows, rows.frames, skeleton);
            if (!decoded) return std::unexpected(decoded.error());
            // Upstream order: the transition full-body set (previous segment's
            // last frames) first, then the user constraints of this window.
            std::vector<constraint_set> targets;
            if (!rows.transition_source.empty()) {
                const size_t overlap = rows.transition_source.size()/skeleton.motion_dim();
                auto tail = detail::decode_motion_rows(rows.transition_source, overlap, skeleton);
                if (!tail) return std::unexpected(tail.error());
                constraint_set transition;
                transition.type = constraint_type::fullbody;
                for (size_t frame=0; frame<overlap; ++frame) {
                    transition.frame_indices.push_back(static_cast<unsigned>(frame));
                    transition.smooth_root_2d.push_back(tail->smooth_root[frame*3]);
                    transition.smooth_root_2d.push_back(tail->smooth_root[frame*3+2]);
                }
                transition.local_rotations_xyzw = std::move(tail->local_xyzw);
                transition.root_positions = std::move(tail->root_positions);
                targets.push_back(std::move(transition));
            }
            auto cropped = detail::crop_constraints(constraints, rows.window_begin, rows.window_begin+rows.frames);
            targets.insert(targets.end(), std::make_move_iterator(cropped.begin()), std::make_move_iterator(cropped.end()));
            if (auto corrected = detail::post_process_motion(skeleton, rows.frames, decoded->local_xyzw, decoded->root_positions,
                                                             decoded->foot_contacts, targets, options.root_margin); !corrected)
                return std::unexpected("post-processing failed: " + corrected.error());
            return detail::encode_motion(skeleton, decoded->local_xyzw, decoded->root_positions, rows.frames);
        };
    auto joined=detail::sample_motion_sequence_from_noise(*impl_->weights,sampled,transition_frames,steps,options.text_cfg,options.constraint_cfg,conditions);
    if (!joined) return std::unexpected(joined.error());
    const size_t motion_dim=impl_->skeleton->motion_dim(), body_dim=impl_->skeleton->body_dim();
    const auto frames=static_cast<unsigned>(joined->size()/motion_dim);
    auto normalized=*joined;
    for (size_t row=0; row<frames; ++row) {
        auto *value=normalized.data()+row*motion_dim;
        for (size_t d=0; d<5; ++d) value[d]=(value[d]-(*gm)[d])/std::sqrt((*gs)[d]*(*gs)[d]+1.e-5F);
        for (size_t d=0; d<body_dim; ++d) value[5+d]=(value[5+d]-(*bm)[d])/std::sqrt((*bs)[d]*(*bs)[d]+1.e-5F);
    }
    auto decoded=detail::decode_motion(normalized,frames,*impl_->skeleton,*gm,*gs,*bm,*bs);
    if (!decoded) return std::unexpected(decoded.error());
    motion_data result; result.frames=frames; result.joints=static_cast<unsigned>(impl_->skeleton->joints());
    result.local_rotations_xyzw=std::move(decoded->local_xyzw); result.root_positions=std::move(decoded->root_positions);
    result.foot_contacts=std::move(decoded->foot_contacts); result.smooth_root_positions=std::move(decoded->smooth_root);
    return result;
#else
    (void) segments; (void) constraints; (void) options;
    return std::unexpected("Kimodo was built without GGML support");
#endif
}

std::expected<std::vector<constraint_set>, std::string> model::parse_constraints(std::string_view json) const {
    return detail::parse_constraints_json(*impl_->skeleton, json);
}

std::expected<std::string, std::string> model::constraints_to_json(std::span<const constraint_set> constraints) const {
    if (auto valid = detail::validate_constraints(*impl_->skeleton, constraints, std::numeric_limits<std::size_t>::max()); !valid)
        return std::unexpected(valid.error());
    return detail::constraints_to_json(*impl_->skeleton, constraints);
}

std::string_view model::skeleton_key() const noexcept { return impl_->skeleton ? impl_->skeleton->key : std::string_view{}; }

namespace {
// JSON strings are UTF-8; std::filesystem throws on malformed UTF-8, which
// must not escape parse_request (a std::expected API).
std::expected<std::filesystem::path, std::string> utf8_path(std::string_view path, std::string_view field) {
    try {
        return std::filesystem::path(std::u8string(reinterpret_cast<const char8_t *>(path.data()), path.size()));
    } catch (const std::system_error &) {
        return std::unexpected("request field \"" + std::string(field) + "\" is not valid UTF-8: " + std::string(path));
    }
}

std::expected<unsigned, std::string> request_unsigned(const detail::json::value &value, std::string_view key) {
    if (!value.is_number() || value.number < 0 || value.number != std::floor(value.number) ||
        value.number > double(std::numeric_limits<unsigned>::max()))
        return std::unexpected("request field \"" + std::string(key) + "\" must be a non-negative integer");
    return static_cast<unsigned>(value.number);
}

std::expected<float, std::string> request_float(const detail::json::value &value, std::string_view key) {
    if (!value.is_number() || !std::isfinite(static_cast<float>(value.number)))
        return std::unexpected("request field \"" + std::string(key) + "\" must be a finite number");
    return static_cast<float>(value.number);
}

// Seeds above 2^53 cannot be exact JSON numbers, so a decimal string is accepted too.
std::expected<std::uint64_t, std::string> request_seed(const detail::json::value &value) {
    if (value.is_string()) {
        std::uint64_t seed = 0;
        const auto *end = value.string.data()+value.string.size();
        const auto parsed = std::from_chars(value.string.data(), end, seed);
        if (value.string.empty() || parsed.ec != std::errc() || parsed.ptr != end)
            return std::unexpected("request field \"seed\" must be an unsigned 64-bit integer");
        return seed;
    }
    if (!value.is_number() || value.number < 0 || value.number != std::floor(value.number) || value.number >= 18446744073709551616.0)
        return std::unexpected("request field \"seed\" must be an unsigned 64-bit integer");
    return static_cast<std::uint64_t>(value.number);
}

// Same reading as the legacy generator prompts (text mode), minus a UTF-8 BOM.
std::expected<std::string, std::string> read_prompt_file(std::string_view path) {
    auto file_path = utf8_path(path, "prompt_file");
    if (!file_path) return std::unexpected(file_path.error());
    std::ifstream prompt_file(*file_path);
    std::string prompt{std::istreambuf_iterator<char>(prompt_file), {}};
    if (!prompt_file && prompt.empty()) return std::unexpected("cannot read sequence prompt " + std::string(path));
    if (prompt.starts_with("\xEF\xBB\xBF")) prompt.erase(0, 3);
    return prompt;
}
}

std::expected<generation_request, std::string> model::parse_request(std::string_view json) const {
    auto document = detail::json::parse(json);
    if (!document) return std::unexpected("invalid request JSON: " + document.error());
    if (!document->is_object()) return std::unexpected("request must be a JSON object");
    generation_request request;
    auto &o = request.options;
    const detail::json::value *inline_constraints = nullptr;
    std::string constraints_file;
    for (const auto &[key, value] : document->object) {
        if (key == "output") {
            if (!value.is_string() && !value.is_null()) return std::unexpected("request field \"output\" must be a string");
            request.output = value.string;
        } else if (key == "seed") {
            auto seed = request_seed(value); if (!seed) return std::unexpected(seed.error()); o.seed = *seed;
        } else if (key == "steps") {
            auto steps = request_unsigned(value, key); if (!steps) return std::unexpected(steps.error()); o.steps = *steps;
        } else if (key == "transition_frames") {
            auto frames = request_unsigned(value, key); if (!frames) return std::unexpected(frames.error()); o.transition_frames = *frames;
        } else if (key == "text_cfg" || key == "constraint_cfg" || key == "root_margin") {
            auto number = request_float(value, key); if (!number) return std::unexpected(number.error());
            (key == "text_cfg" ? o.text_cfg : key == "constraint_cfg" ? o.constraint_cfg : o.root_margin) = *number;
        } else if (key == "post_processing") {
            if (!value.is_bool()) return std::unexpected("request field \"post_processing\" must be true or false");
            o.post_processing = value.boolean;
        } else if (key == "segments") {
            if (!value.is_array()) return std::unexpected("request field \"segments\" must be an array");
            for (const auto &item : value.array) {
                const auto *frames = item.find("frames");
                const auto *prompt = item.find("prompt");
                const auto *prompt_file = item.find("prompt_file");
                if (!item.is_object() || !frames) return std::unexpected("each request segment needs \"frames\" and \"prompt\" or \"prompt_file\"");
                auto count = request_unsigned(*frames, "frames"); if (!count) return std::unexpected(count.error());
                prompt_segment segment; segment.frames = *count;
                if (prompt && prompt->is_string()) segment.prompt = prompt->string;
                else if (prompt_file && prompt_file->is_string() && !prompt_file->string.empty()) {
                    auto text = read_prompt_file(prompt_file->string); if (!text) return std::unexpected(text.error());
                    segment.prompt = std::move(*text);
                } else return std::unexpected("each request segment needs \"frames\" and \"prompt\" or \"prompt_file\"");
                request.segments.push_back(std::move(segment));
            }
        } else if (key == "constraints") {
            if (!value.is_array() && !value.is_null()) return std::unexpected("request field \"constraints\" must be an array");
            if (value.is_array()) inline_constraints = &value;
        } else if (key == "constraints_file") {
            if (!value.is_string() && !value.is_null()) return std::unexpected("request field \"constraints_file\" must be a string");
            constraints_file = value.string;
        }
    }
    if (request.segments.empty()) return std::unexpected("request requires at least one segment");
    if (inline_constraints && !constraints_file.empty()) return std::unexpected("request has both \"constraints\" and \"constraints_file\"");
    if (inline_constraints) {
        auto parsed = detail::parse_constraints_json(*impl_->skeleton, *inline_constraints);
        if (!parsed) return std::unexpected(parsed.error());
        request.constraints = std::move(*parsed);
    } else if (!constraints_file.empty()) {
        auto file_path = utf8_path(constraints_file, "constraints_file");
        if (!file_path) return std::unexpected(file_path.error());
        std::ifstream input(*file_path, std::ios::binary);
        if (!input) return std::unexpected("cannot read constraints file " + constraints_file);
        const std::string text{std::istreambuf_iterator<char>(input), {}};
        auto parsed = detail::parse_constraints_json(*impl_->skeleton, text);
        if (!parsed) return std::unexpected(constraints_file + ": " + parsed.error());
        request.constraints = std::move(*parsed);
    }
    return request;
}
} // namespace kimodo
