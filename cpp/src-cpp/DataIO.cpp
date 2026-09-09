#include "DataIO.h"

#include <array>
#include <bit>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
    // NPY headers use a Python dictionary literal. Parse just the documented scalar
    // float matrix schema; never evaluate Python or infer dtype from word size.
    class HeaderReader {
        std::string_view text;
        size_t pos = 0;

    public:
        explicit HeaderReader(std::string_view header) : text(header) {}
        void space() {
            while (pos < text.size() &&
                   std::isspace(static_cast<unsigned char>(text[pos]))) {
                ++pos;
            }
        }
        bool consume(char c) {
            space();
            if (pos < text.size() && text[pos] == c) {
                ++pos;
                return true;
            }
            return false;
        }
        void expect(char c) {
            if (!consume(c)) { throw std::invalid_argument("Malformed NPY header"); }
        }
        std::string quoted() {
            space();
            if (pos == text.size() || (text[pos] != '\'' && text[pos] != '"')) {
                throw std::invalid_argument("Expected a quoted NPY header field");
            }
            const char quote = text[pos++];
            const auto start = pos;
            while (pos < text.size() && text[pos] != quote) { ++pos; }
            if (pos == text.size()) {
                throw std::invalid_argument("Unterminated NPY header field");
            }
            return std::string(text.substr(start, pos++ - start));
        }
        bool boolean() {
            space();
            if (text.substr(pos, 4) == "True") {
                pos += 4;
                return true;
            }
            if (text.substr(pos, 5) == "False") {
                pos += 5;
                return false;
            }
            throw std::invalid_argument("Invalid NPY fortran_order");
        }
        size_t dimension() {
            space();
            const auto start = pos;
            size_t value = 0;
            while (pos < text.size() && text[pos] >= '0' && text[pos] <= '9') {
                const auto digit = static_cast<size_t>(text[pos++] - '0');
                if (value > (std::numeric_limits<size_t>::max() - digit) / 10) {
                    throw std::invalid_argument("NPY dimension is too large");
                }
                value = value * 10 + digit;
            }
            if (pos == start) { throw std::invalid_argument("Invalid NPY shape"); }
            return value;
        }
        void finish() {
            space();
            if (pos != text.size()) {
                throw std::invalid_argument("Unexpected NPY header content");
            }
        }
    };

    void read_exact(std::istream &input, char *destination, size_t bytes) {
        if (bytes > static_cast<size_t>(std::numeric_limits<std::streamsize>::max()) ||
            !input.read(destination, static_cast<std::streamsize>(bytes))) {
            throw std::invalid_argument("Truncated NPY file");
        }
    }
}// namespace

Eigen::MatrixXd load_npy_data(const std::string &filename) {
    std::ifstream input(filename, std::ios::binary | std::ios::ate);
    if (!input) { throw std::runtime_error("Cannot open input: " + filename); }
    const auto end = input.tellg();
    if (end < 0) { throw std::runtime_error("Cannot determine input size"); }
    const auto file_size = static_cast<uintmax_t>(end);
    input.seekg(0);
    std::array<unsigned char, 8> prefix{};
    read_exact(input, reinterpret_cast<char *>(prefix.data()), prefix.size());
    if (std::string_view(reinterpret_cast<char *>(prefix.data()), 6) != "\x93NUMPY" ||
        prefix[6] < 1 || prefix[6] > 3 || prefix[7] != 0) {
        throw std::invalid_argument("Expected an NPY v1.0, v2.0 or v3.0 file");
    }
    const size_t length_bytes = prefix[6] == 1 ? 2 : 4;
    std::array<unsigned char, 4> length{};
    read_exact(input, reinterpret_cast<char *>(length.data()), length_bytes);
    size_t header_size = 0;
    for (size_t i = 0; i < length_bytes; ++i) {
        header_size |= size_t(length[i]) << (8 * i);
    }
    const size_t prefix_size = prefix.size() + length_bytes;
    constexpr size_t max_header_size = 1024 * 1024;
    if (header_size == 0 || header_size > max_header_size ||
        prefix_size + header_size > file_size) {
        throw std::invalid_argument("Invalid NPY header length");
    }
    std::string header(header_size, '\0');
    read_exact(input, header.data(), header.size());
    if (header.back() != '\n') {
        throw std::invalid_argument("NPY header must end with a newline");
    }
    HeaderReader reader(header);
    reader.expect('{');
    bool have_descr = false, have_order = false, have_shape = false;
    bool fortran_order = false;
    std::string dtype;
    size_t rows = 0, cols = 0;
    while (!reader.consume('}')) {
        const auto key = reader.quoted();
        reader.expect(':');
        if (key == "descr" && !have_descr) {
            dtype = reader.quoted();
            have_descr = true;
        } else if (key == "fortran_order" && !have_order) {
            fortran_order = reader.boolean();
            have_order = true;
        } else if (key == "shape" && !have_shape) {
            reader.expect('(');
            rows = reader.dimension();
            reader.expect(',');
            cols = reader.dimension();
            reader.consume(',');
            reader.expect(')');
            have_shape = true;
        } else {
            throw std::invalid_argument("Unknown or duplicate NPY header field");
        }
        if (!reader.consume(',')) {
            reader.expect('}');
            break;
        }
    }
    reader.finish();
    if (!have_descr || !have_order || !have_shape) {
        throw std::invalid_argument("Missing required NPY header field");
    }
    if (dtype != "<f8" && dtype != ">f8" && dtype != "=f8") {
        throw std::invalid_argument("Input dtype must be float64");
    }
    if (rows < 2 || cols == 0 ||
        cols > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        rows > static_cast<size_t>(std::numeric_limits<Eigen::Index>::max()) ||
        rows > static_cast<size_t>(std::numeric_limits<Eigen::Index>::max()) / cols ||
        rows > std::numeric_limits<size_t>::max() / cols / sizeof(double)) {
        throw std::invalid_argument(
                "Input must have shape (n >= 2, p >= 1) within supported limits");
    }
    const size_t count = rows * cols;
    const size_t payload_bytes = count * sizeof(double);
    if (file_size - prefix_size - header_size != payload_bytes) {
        throw std::invalid_argument(
                "NPY payload size does not match its shape and dtype");
    }
    std::vector<double> values(count);
    read_exact(input, reinterpret_cast<char *>(values.data()), payload_bytes);
    static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);
    static_assert(std::endian::native == std::endian::little ||
                  std::endian::native == std::endian::big);
    const bool swap_bytes =
            dtype[0] != '=' &&
            ((dtype[0] == '<') != (std::endian::native == std::endian::little));
    if (swap_bytes) {
        for (double &value: values) {
            value = std::bit_cast<double>(std::byteswap(std::bit_cast<uint64_t>(value)));
        }
    }
    const auto n = static_cast<Eigen::Index>(rows), p = static_cast<Eigen::Index>(cols);
    if (fortran_order) { return Eigen::Map<Eigen::MatrixXd>(values.data(), n, p); }
    using RowMajorMatrix =
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    return Eigen::Map<RowMajorMatrix>(values.data(), n, p);
}
