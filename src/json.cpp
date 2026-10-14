#include "json.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <system_error>

namespace kimodo::detail::json {
namespace {
// Members per object. Constraint objects have at most 7 keys and requests
// about a dozen; the cap bounds the work an object with millions of junk keys
// could cause.
constexpr std::size_t max_object_members = 4096;

struct parser {
    std::string_view text;
    std::size_t at = 0;
    std::size_t max_depth = 64;

    std::unexpected<std::string> fail(std::string_view message) const {
        return std::unexpected(std::string(message) + " at byte " + std::to_string(at));
    }
    void skip_space() noexcept {
        while (at < text.size() && (text[at] == ' ' || text[at] == '\t' || text[at] == '\n' || text[at] == '\r')) ++at;
    }
    bool consume(std::string_view literal) noexcept {
        if (text.substr(at, literal.size()) != literal) return false;
        at += literal.size();
        return true;
    }

    static void append_utf8(std::string &out, std::uint32_t code) {
        if (code < 0x80) out.push_back(static_cast<char>(code));
        else if (code < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (code >> 6)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        } else if (code < 0x10000) {
            out.push_back(static_cast<char>(0xE0 | (code >> 12)));
            out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (code >> 18)));
            out.push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        }
    }

    std::expected<std::uint32_t, std::string> hex4() {
        if (text.size() - at < 4) return fail("truncated unicode escape");
        std::uint32_t code = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = text[at++];
            code <<= 4;
            if (c >= '0' && c <= '9') code |= static_cast<std::uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') code |= static_cast<std::uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') code |= static_cast<std::uint32_t>(c - 'A' + 10);
            else return fail("invalid unicode escape");
        }
        return code;
    }

    std::expected<std::string, std::string> parse_string() {
        if (at >= text.size() || text[at] != '"') return fail("expected string");
        ++at;
        std::string out;
        while (true) {
            if (at >= text.size()) return fail("unterminated string");
            const char c = text[at++];
            if (c == '"') return out;
            if (static_cast<unsigned char>(c) < 0x20) return fail("control character in string");
            if (c != '\\') { out.push_back(c); continue; }
            if (at >= text.size()) return fail("unterminated escape");
            switch (const char e = text[at++]) {
            case '"': out.push_back('"'); break;
            case '\\': out.push_back('\\'); break;
            case '/': out.push_back('/'); break;
            case 'b': out.push_back('\b'); break;
            case 'f': out.push_back('\f'); break;
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            case 'u': {
                auto code = hex4();
                if (!code) return std::unexpected(code.error());
                if (*code >= 0xD800 && *code <= 0xDBFF) {
                    if (!consume("\\u")) return fail("unpaired surrogate");
                    auto low = hex4();
                    if (!low) return std::unexpected(low.error());
                    if (*low < 0xDC00 || *low > 0xDFFF) return fail("invalid low surrogate");
                    *code = 0x10000 + ((*code - 0xD800) << 10) + (*low - 0xDC00);
                } else if (*code >= 0xDC00 && *code <= 0xDFFF) {
                    return fail("unpaired surrogate");
                }
                append_utf8(out, *code);
                break;
            }
            default:
                (void) e;
                return fail("invalid escape");
            }
        }
    }

    std::expected<value, std::string> parse_number() {
        const std::size_t begin = at;
        if (at < text.size() && text[at] == '-') ++at;
        if (at >= text.size()) return fail("invalid number");
        if (text[at] == '0') ++at;
        else if (text[at] >= '1' && text[at] <= '9') while (at < text.size() && text[at] >= '0' && text[at] <= '9') ++at;
        else return fail("invalid number");
        if (at < text.size() && text[at] == '.') {
            ++at;
            if (at >= text.size() || text[at] < '0' || text[at] > '9') return fail("invalid fraction");
            while (at < text.size() && text[at] >= '0' && text[at] <= '9') ++at;
        }
        if (at < text.size() && (text[at] == 'e' || text[at] == 'E')) {
            ++at;
            if (at < text.size() && (text[at] == '+' || text[at] == '-')) ++at;
            if (at >= text.size() || text[at] < '0' || text[at] > '9') return fail("invalid exponent");
            while (at < text.size() && text[at] >= '0' && text[at] <= '9') ++at;
        }
        value result;
        result.type = value::kind::number;
        const auto *first = text.data() + begin, *last = text.data() + at;
        const auto [end, error] = std::from_chars(first, last, result.number);
        if (error != std::errc() || end != last || !std::isfinite(result.number)) return fail("number out of range");
        return result;
    }

    std::expected<value, std::string> parse_value(std::size_t depth) {
        if (depth > max_depth) return fail("nesting too deep");
        skip_space();
        if (at >= text.size()) return fail("unexpected end of input");
        value result;
        switch (text[at]) {
        case 'n':
            if (!consume("null")) return fail("invalid literal");
            return result;
        case 't':
            if (!consume("true")) return fail("invalid literal");
            result.type = value::kind::boolean; result.boolean = true;
            return result;
        case 'f':
            if (!consume("false")) return fail("invalid literal");
            result.type = value::kind::boolean;
            return result;
        case '"': {
            auto s = parse_string();
            if (!s) return std::unexpected(s.error());
            result.type = value::kind::string; result.string = std::move(*s);
            return result;
        }
        case '[': {
            ++at;
            result.type = value::kind::array;
            skip_space();
            if (at < text.size() && text[at] == ']') { ++at; return result; }
            while (true) {
                auto item = parse_value(depth + 1);
                if (!item) return item;
                result.array.push_back(std::move(*item));
                skip_space();
                if (at < text.size() && text[at] == ',') { ++at; continue; }
                if (at < text.size() && text[at] == ']') { ++at; return result; }
                return fail("expected ',' or ']'");
            }
        }
        case '{': {
            ++at;
            result.type = value::kind::object;
            skip_space();
            if (at < text.size() && text[at] == '}') { ++at; return result; }
            while (true) {
                skip_space();
                if (result.object.size() >= max_object_members) return fail("too many object members");
                auto key = parse_string();
                if (!key) return std::unexpected(key.error());
                skip_space();
                if (at >= text.size() || text[at] != ':') return fail("expected ':'");
                ++at;
                auto item = parse_value(depth + 1);
                if (!item) return item;
                result.object.emplace_back(std::move(*key), std::move(*item));
                skip_space();
                if (at < text.size() && text[at] == ',') { ++at; continue; }
                if (at < text.size() && text[at] == '}') {
                    ++at;
                    // Duplicate keys: sort once per object (O(n log n)) instead of
                    // scanning the keys read so far for every new key.
                    if (result.object.size() > 1) {
                        std::vector<const std::string *> keys;
                        keys.reserve(result.object.size());
                        for (const auto &member : result.object) keys.push_back(&member.first);
                        std::ranges::sort(keys, [](const std::string *a, const std::string *b) { return *a < *b; });
                        if (std::ranges::adjacent_find(keys, [](const std::string *a, const std::string *b) { return *a == *b; }) != keys.end())
                            return fail("duplicate object key");
                    }
                    return result;
                }
                return fail("expected ',' or '}'");
            }
        }
        default:
            return parse_number();
        }
    }
};

void dump_string(std::string &out, std::string_view s) {
    static constexpr char hex[] = "0123456789abcdef";
    out.push_back('"');
    for (const char c : s) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                out += "\\u00";
                out.push_back(hex[(static_cast<unsigned char>(c) >> 4) & 0xF]);
                out.push_back(hex[static_cast<unsigned char>(c) & 0xF]);
            } else {
                out.push_back(c);
            }
        }
    }
    out.push_back('"');
}

void dump_value(std::string &out, const value &v) {
    switch (v.type) {
    case value::kind::null: out += "null"; break;
    case value::kind::boolean: out += v.boolean ? "true" : "false"; break;
    case value::kind::number: {
        char buffer[64];
        const auto [end, error] = std::to_chars(buffer, buffer + sizeof(buffer), v.number);
        if (error == std::errc() && std::isfinite(v.number)) out.append(buffer, end);
        else out += "null";
        break;
    }
    case value::kind::string: dump_string(out, v.string); break;
    case value::kind::array:
        out.push_back('[');
        for (std::size_t i = 0; i < v.array.size(); ++i) {
            if (i) out.push_back(',');
            dump_value(out, v.array[i]);
        }
        out.push_back(']');
        break;
    case value::kind::object:
        out.push_back('{');
        for (std::size_t i = 0; i < v.object.size(); ++i) {
            if (i) out.push_back(',');
            dump_string(out, v.object[i].first);
            out.push_back(':');
            dump_value(out, v.object[i].second);
        }
        out.push_back('}');
        break;
    }
}
} // namespace

const value *value::find(std::string_view key) const noexcept {
    if (type != kind::object) return nullptr;
    for (const auto &[name, item] : object)
        if (name == key) return &item;
    return nullptr;
}

std::expected<value, std::string> parse(std::string_view text, std::size_t max_depth) {
    parser p{text, 0, max_depth};
    // Tolerate a UTF-8 byte order mark written by some Windows editors.
    if (text.substr(0, 3) == "\xEF\xBB\xBF") p.at = 3;
    auto result = p.parse_value(0);
    if (!result) return result;
    p.skip_space();
    if (p.at != text.size()) return p.fail("trailing characters after JSON value");
    return result;
}

std::string dump(const value &input) {
    std::string out;
    dump_value(out, input);
    return out;
}

} // namespace kimodo::detail::json
