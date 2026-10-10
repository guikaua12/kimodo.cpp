// model::parse_request / parse_constraints / constraints_to_json and the C API
// request entry points. CPU only: the model is loaded without a text bundle and
// only its GGUF header is read, so generation must fail cleanly before sampling.
// Usage: kimodo-request-test SOMA_MOTION.gguf CONSTRAINTS_DATA_DIR
#include <kimodo/kimodo.hpp>
#include <kimodo/kimodo_capi.h>

#include <cstdio>
#include <exception>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const std::string &what) {
    if (!ok) { std::printf("FAIL %s\n", what.c_str()); ++failures; }
}
std::string read_file(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), {}};
}
}

int main(int argc, char **argv) {
    if (argc != 3) { std::printf("usage: kimodo-request-test SOMA_MOTION.gguf DATA_DIR\n"); return 2; }
    auto loaded = kimodo::model::load(argv[1], "");
    if (!loaded) { std::printf("load: %s\n", loaded.error().c_str()); return 1; }
    const auto &m = **loaded;
    check(m.skeleton_key() == "soma30", "skeleton_key is soma30");
    std::string data = argv[2];
    for (char &c : data) if (c == '\\') c = '/';

    // Every field overridden, a constraints file, an unknown key (ignored).
    const std::string request = R"({"output":"x","seed":"18446744073709551615","steps":50,"transition_frames":7,"text_cfg":1.5,
        "constraint_cfg":3,"post_processing":true,"root_margin":0.05,"segments":[{"frames":150,"prompt":"walk"},{"frames":60,"prompt":"stop"}],
        "constraints_file":")" + data + R"(/soma_04_ee_constraint.json","unknown_key":1})";
    auto full = m.parse_request(request);
    check(full.has_value(), "parse full request: " + (full ? std::string() : full.error()));
    if (full) {
        const auto &o = full->options;
        check(full->output == "x" && o.seed == 18446744073709551615ULL && o.steps == 50 && o.transition_frames == 7, "integer options");
        check(o.text_cfg == 1.5F && o.constraint_cfg == 3.F && o.post_processing && o.root_margin == .05F, "cfg, post_processing, root_margin");
        check(full->segments.size() == 2 && full->segments[0].frames == 150 && full->segments[1].prompt == "stop", "segments");
        check(full->constraints.size() == 4 && full->constraints[2].type == kimodo::constraint_type::left_hand, "constraints from file");
    }
    auto defaults = m.parse_request(R"({"segments":[{"frames":30,"prompt":"a"}]})");
    check(defaults && defaults->options.steps == 100 && defaults->options.transition_frames == 5 && defaults->options.seed == 0 &&
          !defaults->options.post_processing && defaults->options.root_margin == .04F && defaults->constraints.empty(), "defaults");
    auto inline_constraints = m.parse_request(R"({"segments":[{"frames":200,"prompt":"a"}],"constraints":[{"type":"root2d",
        "frame_indices":[0,90],"smooth_root_2d":[[0,0],[1,2]],"global_root_heading":[[1,0],[0,1]]}],"constraints_file":null})");
    check(inline_constraints && inline_constraints->constraints.size() == 1 &&
          inline_constraints->constraints[0].global_root_heading.size() == 4, "inline constraints");
    for (const char *bad : {"[]", R"({"segments":[]})", R"({"segments":[{"frames":-1,"prompt":"a"}]})", R"({"segments":[{"frames":10}]})",
                            R"({"segments":[{"frames":10,"prompt":"a"}],"steps":1.5})",
                            R"({"segments":[{"frames":10,"prompt":"a"}],"post_processing":1})",
                            R"({"segments":[{"frames":10,"prompt":"a"}],"constraints":[],"constraints_file":"c.json"})",
                            R"({"segments":[{"frames":10,"prompt_file":"does-not-exist.txt"}]})",
                            R"({"segments":[{"frames":10,"prompt":"a"}],"constraints_file":"does-not-exist.json"})",
                            R"({"segments":[{"frames":10,"prompt":"a"}],"constraints":[{"type":"end-effectors","frame_indices":[0]}]})",
                            R"({"segments":[{"frames":10,"prompt":"a"}],"seed":"12x"})"}) {
        auto rejected = m.parse_request(bad);
        check(!rejected, std::string("reject ") + bad);
    }

    // Paths that are not valid UTF-8 (e.g. an ANSI "Usuários") are reported as
    // errors naming the field; std::filesystem's conversion must not escape.
    for (const std::string field : {"prompt_file", "constraints_file"}) {
        const std::string path = std::string("C:/Usu") + '\xE1' + "rios/x.json";
        const std::string bad = field == "prompt_file"
            ? R"({"segments":[{"frames":60,"prompt_file":")" + path + R"("}]})"
            : R"({"segments":[{"frames":60,"prompt":"a"}],"constraints_file":")" + path + R"("})";
        try {
            auto rejected = m.parse_request(bad);
            check(!rejected, "reject non-UTF-8 " + field);
#ifdef _WIN32
            if (!rejected) check(rejected.error().find(field) != std::string::npos, "non-UTF-8 error names " + field + ": " + rejected.error());
#endif
        } catch (const std::exception &error) {
            check(false, "non-UTF-8 " + field + " threw: " + error.what());
        }
    }

    // Public constraints.json round trip (152-frame dense root path + full-body keyframe).
    auto parsed = m.parse_constraints(read_file(data + "/soma_07_mixed_constraints.json"));
    check(parsed && parsed->size() == 2, "parse_constraints");
    if (parsed) {
        auto written = m.constraints_to_json(*parsed);
        auto again = written ? m.parse_constraints(*written) : decltype(parsed){};
        check(written && again && again->size() == 2 && (*again)[1].frame_indices.size() == 152, "constraints_to_json round trip");
        auto invalid = *parsed;
        invalid[0].joint_names = {"Hips"};
        check(!m.constraints_to_json(invalid), "constraints_to_json validates");
    }

    // generate() without a text bundle fails before sampling, never crashes.
    const std::vector<kimodo::prompt_segment> segments{{"walk", 150}};
    std::vector<kimodo::constraint_set> constraints(1);
    constraints[0].type = kimodo::constraint_type::root2d;
    constraints[0].frame_indices = {150};
    constraints[0].smooth_root_2d = {0.F, 0.F};
    check(!m.generate(segments, constraints, {}), "generate without a text bundle fails");

    // C API.
    char error[256] = {};
    kimodo_runtime_options runtime{};
    runtime.size = sizeof(runtime);
    kimodo_model *model = kimodo_model_load(argv[1], nullptr, nullptr, &runtime, error, sizeof(error));
    check(model != nullptr, "C model load");
    check(kimodo_generate_request(nullptr, "{}", error, sizeof(error)) == nullptr && std::string(error) == "invalid model", "C null model");
    check(kimodo_generate_request(model, nullptr, error, sizeof(error)) == nullptr, "C null request");
    check(kimodo_generate_request(model, "not json", error, sizeof(error)) == nullptr &&
          std::string(error).starts_with("invalid request JSON"), "C malformed request");
    check(kimodo_generate_request(model, R"({"segments":[{"frames":30,"prompt":"a"}]})", error, sizeof(error)) == nullptr &&
          std::string(kimodo_model_last_error(model)) == error, "C request without a text bundle");
    check(kimodo_motion_foot_contacts(nullptr) == nullptr && kimodo_motion_smooth_root_positions(nullptr) == nullptr, "C null motion accessors");
    check(kimodo_abi_version() == 1, "ABI version 1");
    check(kimodo_post_processing_available() == (kimodo::post_processing_available() ? 1 : 0), "C post-processing availability");
    kimodo_model_free(model);
    std::printf("%s\n", failures ? "request tests FAILED" : "request tests passed");
    return failures != 0;
}
