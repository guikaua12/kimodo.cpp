#pragma once

#include <cstddef>
#include <expected>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace kimodo::detail::json {

// Small DOM for the constraint files and generator requests.  Objects keep
// their source order; duplicate keys are rejected by the parser.
struct value {
    enum class kind { null, boolean, number, string, array, object };
    kind type = kind::null;
    bool boolean = false;
    double number = 0.0;
    std::string string;
    std::vector<value> array;
    std::vector<std::pair<std::string, value>> object;

    [[nodiscard]] bool is_null() const noexcept { return type == kind::null; }
    [[nodiscard]] bool is_bool() const noexcept { return type == kind::boolean; }
    [[nodiscard]] bool is_number() const noexcept { return type == kind::number; }
    [[nodiscard]] bool is_string() const noexcept { return type == kind::string; }
    [[nodiscard]] bool is_array() const noexcept { return type == kind::array; }
    [[nodiscard]] bool is_object() const noexcept { return type == kind::object; }
    // Returns nullptr when this is not an object or the key is absent.
    [[nodiscard]] const value *find(std::string_view key) const noexcept;
};

// RFC 8259 parser (no comments, no trailing commas, finite numbers; raw string
// bytes are copied without UTF-8 validation). `max_depth` bounds nesting so
// hostile input cannot exhaust the stack; objects are limited to 4096 members
// and duplicate keys are rejected.
std::expected<value, std::string> parse(std::string_view text, std::size_t max_depth = 64);

// Serialises with the shortest round-tripping representation for numbers.
std::string dump(const value &input);

} // namespace kimodo::detail::json
