// The checks must also run in Release builds.
#undef NDEBUG
#include "json.hpp"

#include <cassert>
#include <chrono>
#include <string>

using kimodo::detail::json::parse;
using kimodo::detail::json::dump;

int main() {
    auto document = parse(R"([{"type":"fullbody","frame_indices":[0, 30],"root_positions":[[0.5,-1e-2,2E3]],
        "name":"caf\u00e9 \ud83d\ude00","ok":true,"none":null}])");
    assert(document && document->is_array() && document->array.size() == 1);
    const auto &item = document->array[0];
    assert(item.find("type")->string == "fullbody");
    assert(item.find("frame_indices")->array[1].number == 30.0);
    assert(item.find("root_positions")->array[0].array[1].number == -0.01);
    assert(item.find("root_positions")->array[0].array[2].number == 2000.0);
    assert(item.find("name")->string == "caf\xC3\xA9 \xF0\x9F\x98\x80");
    assert(item.find("ok")->boolean && item.find("none")->is_null() && !item.find("missing"));

    auto round = parse(dump(*document));
    assert(round && dump(*round) == dump(*document));

    for (const char *bad : {"", "[1,]", "{\"a\":1,\"a\":2}", "01", "1.", "[1e999]", "\"\\ud800\"",
                            "{\"a\" 1}", "[1] 2", "nul", "\"tab\there\"", "-", "[\"\\x\"]"})
        assert(!parse(bad));
    std::string deep(100, '[');
    deep += std::string(100, ']');
    assert(!parse(deep, 64) && parse(deep, 128));
    assert(parse("\xEF\xBB\xBF{}"));

    // Wide objects: duplicate keys are found in O(n log n) and the member cap
    // stops an object padded with junk keys early (hostile request bodies).
    const auto object_with = [](std::size_t members, bool duplicate_last) {
        std::string text = "{";
        for (std::size_t i = 0; i < members; ++i) {
            if (i) text += ',';
            const std::size_t key = duplicate_last && i + 1 == members ? 0 : i;
            text += "\"k" + std::to_string(key) + "\":" + std::to_string(i);
        }
        return text + "}";
    };
    const auto started = std::chrono::steady_clock::now();
    auto wide = parse(object_with(4096, false));
    assert(wide && wide->object.size() == 4096 && wide->find("k4095")->number == 4095.0);
    auto duplicate = parse(object_with(4096, true));
    assert(!duplicate && duplicate.error().starts_with("duplicate object key"));
    auto padded = parse(object_with(200000, false));
    assert(!padded && padded.error().starts_with("too many object members"));
    assert(parse("[" + object_with(4000, false) + "," + object_with(4000, false) + "]"));
    assert(std::chrono::steady_clock::now() - started < std::chrono::seconds(2));
}
