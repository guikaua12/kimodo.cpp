// Command-line bridge for the localhost demo.  It deliberately uses only the
// public C++ model API, so the demo exercises the same text route as embedders.
#include <kimodo/kimodo.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {
void write_f32(const std::filesystem::path &path, const std::vector<float> &values) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot open " + path.string());
    out.write(reinterpret_cast<const char *>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!out) throw std::runtime_error("cannot write " + path.string());
}

void write_motion(const std::filesystem::path &output, const kimodo::motion_data &motion) {
    std::filesystem::create_directories(output);
    write_f32(output / "root_positions.f32", motion.root_positions);
    write_f32(output / "local_rotations_xyzw.f32", motion.local_rotations_xyzw);
    if (!motion.foot_contacts.empty()) write_f32(output / "foot_contacts.f32", motion.foot_contacts);
    if (!motion.smooth_root_positions.empty()) write_f32(output / "smooth_root.f32", motion.smooth_root_positions);
}

// Paths inside a JSON request are UTF-8 (std::filesystem throws on malformed
// UTF-8; report the field instead of the bare conversion error).
std::filesystem::path utf8_path(std::string_view path, std::string_view field) {
    try {
        return std::filesystem::path(std::u8string(reinterpret_cast<const char8_t *>(path.data()), path.size()));
    } catch (const std::system_error &) {
        throw std::runtime_error("request field \"" + std::string(field) + "\" is not valid UTF-8");
    }
}

std::vector<std::string> split_fields(const std::string &line) {
    std::vector<std::string> fields;
    std::string field;
    std::istringstream input(line);
    while (std::getline(input, field, '\t')) fields.push_back(field);
    return fields;
}

std::string protocol_error(std::string message) {
    for (char &c : message) if (c == '\n' || c == '\r' || c == '\t') c = ' ';
    return message;
}

bool is_json_request(std::string_view line) {
    const auto first = line.find_first_not_of(" \t\r\n");
    return first != std::string_view::npos && line[first] == '{';
}

// Protocol v2: one JSON generation request (docs/IMPLEMENTATION.md).
struct request_result { kimodo::motion_data motion; std::size_t constraint_sets = 0; bool post_processing = false; };
request_result run_request(const kimodo::model &model, std::string_view json) {
    auto request = model.parse_request(json);
    if (!request) throw std::runtime_error(request.error());
    if (request->output.empty()) throw std::runtime_error("request requires \"output\"");
    auto motion = model.generate(request->segments, request->constraints, request->options);
    if (!motion) throw std::runtime_error(motion.error());
    write_motion(utf8_path(request->output, "output"), *motion);
    // G1 motions are never post-processed (model::generate follows upstream).
    const bool post_processed = request->options.post_processing && model.skeleton_key() != "g1skel34";
    return {std::move(*motion), request->constraints.size(), post_processed};
}
}

int main(int argc, char **argv) try {
    if (argc == 4 && std::string_view(argv[1]) == "--server") {
        auto model = kimodo::model::load(argv[2], argv[3]);
        if (!model) throw std::runtime_error(model.error());
        std::string line;
        while (std::getline(std::cin, line)) {
            try {
                if (is_json_request(line)) {
                    const auto result = run_request(**model, line);
                    std::cout << "OK\t" << result.motion.frames << '\t' << result.motion.joints << '\n' << std::flush;
                    continue;
                }
                const auto fields = split_fields(line);
                if (fields.size() < 6 || (fields.size() - 4) % 2 != 0)
                    throw std::runtime_error("invalid server request");
                const auto transition = static_cast<unsigned>(std::stoul(fields[0]));
                const auto steps = static_cast<unsigned>(std::stoul(fields[1]));
                const auto seed = static_cast<std::uint64_t>(std::stoull(fields[2]));
                std::vector<kimodo::prompt_segment> segments;
                for (size_t index = 4; index < fields.size(); index += 2) {
                    std::ifstream prompt_file(fields[index + 1]);
                    const std::string prompt{std::istreambuf_iterator<char>(prompt_file), {}};
                    if (!prompt_file && prompt.empty()) throw std::runtime_error("cannot read sequence prompt");
                    segments.push_back({prompt, static_cast<unsigned>(std::stoul(fields[index]))});
                }
                auto motion = (*model)->generate_text_sequence(segments, transition, steps, seed, 2.F, 2.F);
                if (!motion) throw std::runtime_error(motion.error());
                write_motion(fields[3], *motion);
                std::cout << "OK\t" << motion->frames << '\t' << motion->joints << '\n' << std::flush;
            } catch (const std::exception &error) {
                std::cout << "ERR\t" << protocol_error(error.what()) << '\n' << std::flush;
            }
        }
        return 0;
    }
    if (argc == 5 && std::string_view(argv[3]) == "--request") {
        // argv is in the ANSI code page on Windows, like the other argv paths
        // here; utf8_path is only for the UTF-8 strings of the JSON request.
        std::ifstream input(argv[4], std::ios::binary);
        if (!input) throw std::runtime_error(std::string("cannot read request ") + argv[4]);
        const std::string json{std::istreambuf_iterator<char>(input), {}};
        auto model = kimodo::model::load(argv[1], argv[2]);
        if (!model) throw std::runtime_error(model.error());
        const auto result = run_request(**model, json);
        std::cout << "generated " << result.motion.frames << " frames with " << result.motion.joints << " joints ("
                  << result.constraint_sets << " constraint sets, post-processing "
                  << (result.post_processing ? "on" : "off") << ")\n";
        return 0;
    }
    if (argc >= 10 && std::string_view(argv[3]) == "--sequence") {
        if ((argc - 8) % 2 != 0) throw std::runtime_error("sequence requires FRAME PROMPT.txt pairs");
        const auto transition = static_cast<unsigned>(std::stoul(argv[4]));
        const auto steps = static_cast<unsigned>(std::stoul(argv[5]));
        const auto seed = static_cast<std::uint64_t>(std::stoull(argv[6]));
        std::vector<kimodo::prompt_segment> segments;
        for (int index=8; index<argc; index+=2) {
            std::ifstream prompt_file(argv[index+1]);
            const std::string prompt{std::istreambuf_iterator<char>(prompt_file), {}};
            if (!prompt_file && prompt.empty()) throw std::runtime_error("cannot read sequence prompt");
            segments.push_back({prompt, static_cast<unsigned>(std::stoul(argv[index]))});
        }
        auto model = kimodo::model::load(argv[1], argv[2]);
        if (!model) throw std::runtime_error(model.error());
        auto motion = (*model)->generate_text_sequence(segments, transition, steps, seed, 2.F, 2.F);
        if (!motion) throw std::runtime_error(motion.error());
        write_motion(argv[7], *motion);
        std::cout << "generated " << motion->frames << " frames with " << motion->joints << " joints\n";
        return 0;
    }
    if (argc != 8) {
        std::cerr << "usage: " << argv[0] << " MOTION.gguf TEXT_BUNDLE PROMPT.txt FRAMES STEPS SEED OUTPUT_DIR\n"
                  << "   or: " << argv[0] << " MOTION.gguf TEXT_BUNDLE --sequence TRANSITION STEPS SEED OUTPUT_DIR FRAME PROMPT.txt [FRAME PROMPT.txt ...]\n"
                  << "   or: " << argv[0] << " MOTION.gguf TEXT_BUNDLE --request REQUEST.json\n"
                  << "   or: " << argv[0] << " --server MOTION.gguf TEXT_BUNDLE   (stdin: legacy TSV or JSON request lines)\n";
        return 2;
    }
    std::ifstream prompt_file(argv[3]);
    const std::string prompt{std::istreambuf_iterator<char>(prompt_file), {}};
    if (!prompt_file && prompt.empty()) throw std::runtime_error("cannot read prompt");
    const auto frames = static_cast<unsigned>(std::stoul(argv[4]));
    const auto steps = static_cast<unsigned>(std::stoul(argv[5]));
    const auto seed = static_cast<std::uint64_t>(std::stoull(argv[6]));
    auto model = kimodo::model::load(argv[1], argv[2]);
    if (!model) throw std::runtime_error(model.error());
    auto motion = (*model)->generate_text(prompt, frames, steps, seed, 2.F, 2.F);
    if (!motion) throw std::runtime_error(motion.error());
    write_motion(argv[7], *motion);
    std::cout << "generated " << motion->frames << " frames with " << motion->joints << " joints\n";
    return 0;
} catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
}
