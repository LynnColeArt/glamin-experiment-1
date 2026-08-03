#include "gx1/hook_artifact.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace gx1 {

namespace {

constexpr const char* hook_format = "gx1-hook-v1";
constexpr const char* manifest_name = "hook_manifest.txt";
constexpr const char* projection_name = "projection.f32";
constexpr const char* labels_name = "residual_labels.u64";
constexpr const char* residuals_name = "residuals.f32";

constexpr std::array<std::uint32_t, 64> sha256_constants{
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
};

class Sha256 final {
public:
    void update(const std::uint8_t* data, std::size_t size) {
        constexpr auto maximum_bytes = std::numeric_limits<std::uint64_t>::max() / 8U;
        if (size > maximum_bytes || byte_count_ > maximum_bytes - size) {
            throw std::length_error("SHA-256 input is too large");
        }
        byte_count_ += static_cast<std::uint64_t>(size);
        while (size > 0) {
            const auto amount = std::min(size, block_.size() - block_size_);
            std::memcpy(block_.data() + block_size_, data, amount);
            block_size_ += amount;
            data += amount;
            size -= amount;
            if (block_size_ == block_.size()) {
                transform(block_.data());
                block_size_ = 0;
            }
        }
    }

    [[nodiscard]] std::string finish() {
        const auto bit_count = byte_count_ * 8U;
        block_[block_size_++] = 0x80U;
        if (block_size_ > 56U) {
            std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_size_),
                      block_.end(), 0U);
            transform(block_.data());
            block_size_ = 0;
        }
        std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_size_),
                  block_.begin() + 56, 0U);
        for (std::size_t index = 0; index < 8; ++index) {
            block_[63U - index] = static_cast<std::uint8_t>(bit_count >> (index * 8U));
        }
        transform(block_.data());

        std::ostringstream output;
        output.imbue(std::locale::classic());
        output << "sha256:" << std::hex << std::setfill('0');
        for (const auto value : state_) {
            output << std::setw(8) << value;
        }
        return output.str();
    }

private:
    static std::uint32_t rotate_right(const std::uint32_t value, const unsigned count) {
        return (value >> count) | (value << (32U - count));
    }

    void transform(const std::uint8_t* block) {
        std::array<std::uint32_t, 64> words{};
        for (std::size_t index = 0; index < 16; ++index) {
            const auto offset = index * 4U;
            words[index] = (static_cast<std::uint32_t>(block[offset]) << 24U) |
                           (static_cast<std::uint32_t>(block[offset + 1U]) << 16U) |
                           (static_cast<std::uint32_t>(block[offset + 2U]) << 8U) |
                           static_cast<std::uint32_t>(block[offset + 3U]);
        }
        for (std::size_t index = 16; index < words.size(); ++index) {
            const auto left = words[index - 15U];
            const auto right = words[index - 2U];
            const auto sigma0 = rotate_right(left, 7U) ^ rotate_right(left, 18U) ^
                                (left >> 3U);
            const auto sigma1 = rotate_right(right, 17U) ^ rotate_right(right, 19U) ^
                                (right >> 10U);
            words[index] = words[index - 16U] + sigma0 + words[index - 7U] + sigma1;
        }

        auto a = state_[0];
        auto b = state_[1];
        auto c = state_[2];
        auto d = state_[3];
        auto e = state_[4];
        auto f = state_[5];
        auto g = state_[6];
        auto h = state_[7];
        for (std::size_t index = 0; index < words.size(); ++index) {
            const auto sum1 = rotate_right(e, 6U) ^ rotate_right(e, 11U) ^
                              rotate_right(e, 25U);
            const auto choose = (e & f) ^ ((~e) & g);
            const auto temporary1 = h + sum1 + choose + sha256_constants[index] +
                                    words[index];
            const auto sum0 = rotate_right(a, 2U) ^ rotate_right(a, 13U) ^
                              rotate_right(a, 22U);
            const auto majority = (a & b) ^ (a & c) ^ (b & c);
            const auto temporary2 = sum0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temporary1;
            d = c;
            c = b;
            b = a;
            a = temporary1 + temporary2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_{
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
    };
    std::array<std::uint8_t, 64> block_{};
    std::size_t block_size_{0};
    std::uint64_t byte_count_{0};
};

std::size_t checked_product(const std::size_t left, const std::size_t right) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::length_error("hook artifact shape is too large");
    }
    return left * right;
}

bool finite_values(const std::vector<float>& values) {
    for (const auto value : values) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

void validate_segment(const std::string& value, const char* name) {
    if (value.empty() || value == "." || value == ".." ||
        value.find('/') != std::string::npos ||
        value.find('\\') != std::string::npos ||
        value.find('\0') != std::string::npos) {
        throw std::invalid_argument(std::string(name) + " must be one safe path segment");
    }
}

void validate_text(const std::string& value, const char* name) {
    if (value.empty() || value.find('\n') != std::string::npos ||
        value.find('\r') != std::string::npos || value.find('=') != std::string::npos ||
        value.find('\0') != std::string::npos) {
        throw std::invalid_argument(std::string(name) + " is empty or non-canonical");
    }
}

std::string metric_name(const GlaminMetric metric) {
    return metric == GlaminMetric::l2 ? "l2" : "inner_product";
}

GlaminMetric parse_metric(const std::string& value) {
    if (value == "l2") {
        return GlaminMetric::l2;
    }
    if (value == "inner_product") {
        return GlaminMetric::inner_product;
    }
    throw std::invalid_argument("hook artifact metric is unsupported");
}

std::string normalization_name(const ProjectionNormalization normalization) {
    return normalization == ProjectionNormalization::none ? "none" : "l2";
}

ProjectionNormalization parse_normalization(const std::string& value) {
    if (value == "none") {
        return ProjectionNormalization::none;
    }
    if (value == "l2") {
        return ProjectionNormalization::l2;
    }
    throw std::invalid_argument("hook artifact normalization is unsupported");
}

std::string address_selection_name(const AddressSelectionPolicy policy) {
    switch (policy) {
    case AddressSelectionPolicy::last_token:
        return "last_token";
    case AddressSelectionPolicy::all_token_rows:
        return "all_token_rows";
    }
    throw std::invalid_argument("hook artifact address selection is unsupported");
}

AddressSelectionPolicy parse_address_selection(const std::string& value) {
    if (value == "last_token") {
        return AddressSelectionPolicy::last_token;
    }
    if (value == "all_token_rows") {
        return AddressSelectionPolicy::all_token_rows;
    }
    throw std::invalid_argument("hook artifact address selection is unsupported");
}

std::string float_text(const float value) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument("hook artifact gate must be finite");
    }
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << std::setprecision(std::numeric_limits<float>::max_digits10) << value;
    return output.str();
}

std::string canonical_manifest(const std::map<std::string, std::string>& fields) {
    std::string canonical;
    for (const auto& field : fields) {
        if (field.first != "contract_sha256") {
            canonical += field.first + "=" + field.second + "\n";
        }
    }
    return canonical;
}

void write_bytes(const std::filesystem::path& path, const std::vector<std::uint8_t>& bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("failed to create hook artifact file: " + path.string());
    }
    output.write(
        reinterpret_cast<const char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
    if (!output) {
        throw std::runtime_error("failed to write hook artifact file: " + path.string());
    }
}

std::vector<std::uint8_t> encode_floats(const std::vector<float>& values) {
    static_assert(sizeof(float) == 4, "hook artifact requires IEEE-sized float32");
    std::vector<std::uint8_t> bytes(checked_product(values.size(), 4U));
    for (std::size_t index = 0; index < values.size(); ++index) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &values[index], sizeof(bits));
        for (std::size_t byte = 0; byte < 4U; ++byte) {
            bytes[index * 4U + byte] = static_cast<std::uint8_t>(bits >> (byte * 8U));
        }
    }
    return bytes;
}

std::vector<std::uint8_t> encode_labels(const std::vector<std::uint64_t>& labels) {
    std::vector<std::uint8_t> bytes(checked_product(labels.size(), 8U));
    for (std::size_t index = 0; index < labels.size(); ++index) {
        for (std::size_t byte = 0; byte < 8U; ++byte) {
            bytes[index * 8U + byte] =
                static_cast<std::uint8_t>(labels[index] >> (byte * 8U));
        }
    }
    return bytes;
}

std::vector<std::uint8_t> read_bytes(
    const std::filesystem::path& path,
    const std::size_t expected_size) {
    std::error_code error;
    const auto file_size = std::filesystem::file_size(path, error);
    if (error || file_size != expected_size) {
        throw std::invalid_argument("hook artifact file has the wrong size: " + path.string());
    }
    std::vector<std::uint8_t> bytes(expected_size);
    std::ifstream input(path, std::ios::binary);
    if (!input || !input.read(
            reinterpret_cast<char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()))) {
        throw std::invalid_argument("failed to read hook artifact file: " + path.string());
    }
    return bytes;
}

std::vector<float> decode_floats(const std::vector<std::uint8_t>& bytes) {
    if (bytes.size() % 4U != 0) {
        throw std::invalid_argument("float artifact has a partial element");
    }
    std::vector<float> values(bytes.size() / 4U);
    for (std::size_t index = 0; index < values.size(); ++index) {
        std::uint32_t bits = 0;
        for (std::size_t byte = 0; byte < 4U; ++byte) {
            bits |= static_cast<std::uint32_t>(bytes[index * 4U + byte]) << (byte * 8U);
        }
        std::memcpy(&values[index], &bits, sizeof(bits));
    }
    if (!finite_values(values)) {
        throw std::invalid_argument("float artifact contains non-finite values");
    }
    return values;
}

std::vector<std::uint64_t> decode_labels(const std::vector<std::uint8_t>& bytes) {
    if (bytes.size() % 8U != 0) {
        throw std::invalid_argument("label artifact has a partial element");
    }
    std::vector<std::uint64_t> labels(bytes.size() / 8U);
    for (std::size_t index = 0; index < labels.size(); ++index) {
        for (std::size_t byte = 0; byte < 8U; ++byte) {
            labels[index] |= static_cast<std::uint64_t>(bytes[index * 8U + byte]) <<
                             (byte * 8U);
        }
    }
    if (std::set<std::uint64_t>(labels.begin(), labels.end()).size() != labels.size()) {
        throw std::invalid_argument("hook artifact contains duplicate residual labels");
    }
    return labels;
}

std::map<std::string, std::string> read_manifest(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::invalid_argument("hook artifact manifest is missing");
    }
    std::map<std::string, std::string> fields;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line.back() == '\r') {
            throw std::invalid_argument("hook artifact manifest is not canonical");
        }
        const auto separator = line.find('=');
        if (separator == std::string::npos || separator == 0 ||
            separator + 1U == line.size() || line.find('=', separator + 1U) != std::string::npos) {
            throw std::invalid_argument("hook artifact manifest line is invalid");
        }
        const auto inserted = fields.emplace(
            line.substr(0, separator),
            line.substr(separator + 1U));
        if (!inserted.second) {
            throw std::invalid_argument("hook artifact manifest contains a duplicate key");
        }
    }
    if (!input.eof()) {
        throw std::invalid_argument("hook artifact manifest could not be read");
    }
    return fields;
}

const std::string& required(
    const std::map<std::string, std::string>& fields,
    const std::string& key) {
    const auto found = fields.find(key);
    if (found == fields.end()) {
        throw std::invalid_argument("hook artifact manifest is missing " + key);
    }
    return found->second;
}

template <typename Integer>
Integer parse_integer(const std::string& value, const char* name) {
    Integer parsed{};
    const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || parsed == 0) {
        throw std::invalid_argument(std::string(name) + " is not a positive integer");
    }
    return parsed;
}

float parse_float(const std::string& value, const char* name) {
    std::istringstream input(value);
    input.imbue(std::locale::classic());
    float gate = 0.0F;
    input >> gate;
    if (!input || !input.eof() || !std::isfinite(gate)) {
        throw std::invalid_argument(std::string("hook artifact ") + name + " is invalid");
    }
    return gate;
}

void validate_model_contract(const ModelHookContract& contract) {
    validate_text(contract.model_sha256, "model_sha256");
    validate_text(contract.llama_revision, "llama_revision");
    validate_text(contract.architecture, "architecture");
    validate_text(contract.target_tensor, "target_tensor");
    if (contract.model_sha256.size() != 71U ||
        contract.model_sha256.rfind("sha256:", 0) != 0 ||
        contract.hidden_dimension == 0) {
        throw std::invalid_argument("model hook contract is incomplete");
    }
}

void verify_checksum(
    const std::filesystem::path& path,
    const std::string& expected,
    const char* name) {
    if (sha256_file(path) != expected) {
        throw std::invalid_argument(std::string(name) + " checksum mismatch");
    }
}

} // namespace

std::string sha256_text(const std::string& text) {
    Sha256 hash;
    hash.update(reinterpret_cast<const std::uint8_t*>(text.data()), text.size());
    return hash.finish();
}

std::string sha256_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::invalid_argument("cannot hash missing file: " + path.string());
    }
    Sha256 hash;
    std::array<std::uint8_t, 64U * 1024U> buffer{};
    while (input) {
        input.read(
            reinterpret_cast<char*>(buffer.data()),
            static_cast<std::streamsize>(buffer.size()));
        const auto count = input.gcount();
        if (count > 0) {
            hash.update(buffer.data(), static_cast<std::size_t>(count));
        }
    }
    if (!input.eof()) {
        throw std::invalid_argument("failed while hashing file: " + path.string());
    }
    return hash.finish();
}

void write_hook_artifact(
    const std::filesystem::path& artifact_directory,
    const HookArtifactSpec& spec) {
    validate_model_contract(spec.model);
    validate_segment(spec.glamin_directory, "glamin_directory");
    validate_text(spec.glamin_space_id, "glamin_space_id");
    if (spec.query_dimension == 0 || spec.residual_labels.empty() ||
        !std::isfinite(spec.maximum_distance) || spec.maximum_distance < 0.0F ||
        spec.input_projection.size() != checked_product(
            spec.model.hidden_dimension, spec.query_dimension) ||
        spec.residuals.size() != checked_product(
            spec.residual_labels.size(), spec.model.hidden_dimension) ||
        !finite_values(spec.input_projection) || !finite_values(spec.residuals)) {
        throw std::invalid_argument("hook artifact tensors do not match their declared shapes");
    }
    if (std::set<std::uint64_t>(
            spec.residual_labels.begin(), spec.residual_labels.end()).size() !=
        spec.residual_labels.size()) {
        throw std::invalid_argument("hook artifact residual labels must be unique");
    }

    std::filesystem::create_directories(artifact_directory);
    const auto glamin_directory = artifact_directory / spec.glamin_directory;
    const auto contracts_path = glamin_directory / "contracts.json";
    const auto layout_path = glamin_directory / "vector_layout.json";
    const auto vectors_path = glamin_directory / "vectors.bin";
    if (!std::filesystem::is_regular_file(contracts_path) ||
        !std::filesystem::is_regular_file(layout_path) ||
        !std::filesystem::is_regular_file(vectors_path)) {
        throw std::invalid_argument("hook artifact is missing its Glamin files");
    }

    write_bytes(artifact_directory / projection_name, encode_floats(spec.input_projection));
    write_bytes(artifact_directory / labels_name, encode_labels(spec.residual_labels));
    write_bytes(artifact_directory / residuals_name, encode_floats(spec.residuals));

    std::map<std::string, std::string> fields{
        {"address_selection", address_selection_name(spec.address_selection)},
        {"architecture", spec.model.architecture},
        {"format", hook_format},
        {"gate", float_text(spec.gate)},
        {"glamin_contracts_sha256", sha256_file(contracts_path)},
        {"glamin_directory", spec.glamin_directory},
        {"glamin_layout_sha256", sha256_file(layout_path)},
        {"glamin_space_id", spec.glamin_space_id},
        {"glamin_vectors_sha256", sha256_file(vectors_path)},
        {"hidden_dimension", std::to_string(spec.model.hidden_dimension)},
        {"llama_revision", spec.model.llama_revision},
        {"maximum_distance", float_text(spec.maximum_distance)},
        {"metric", metric_name(spec.metric)},
        {"model_sha256", spec.model.model_sha256},
        {"normalization", normalization_name(spec.query_normalization)},
        {"projection_file", projection_name},
        {"projection_sha256", sha256_file(artifact_directory / projection_name)},
        {"query_dimension", std::to_string(spec.query_dimension)},
        {"residual_count", std::to_string(spec.residual_labels.size())},
        {"residual_labels_file", labels_name},
        {"residual_labels_sha256", sha256_file(artifact_directory / labels_name)},
        {"residuals_file", residuals_name},
        {"residuals_sha256", sha256_file(artifact_directory / residuals_name)},
        {"target_tensor", spec.model.target_tensor},
    };
    fields.emplace("contract_sha256", sha256_text(canonical_manifest(fields)));

    std::ofstream manifest(artifact_directory / manifest_name, std::ios::trunc);
    if (!manifest) {
        throw std::runtime_error("failed to create hook artifact manifest");
    }
    for (const auto& field : fields) {
        manifest << field.first << '=' << field.second << '\n';
    }
    if (!manifest) {
        throw std::runtime_error("failed to write hook artifact manifest");
    }
}

LoadedHookArtifact load_hook_artifact(
    const std::filesystem::path& artifact_directory,
    const ModelHookContract& expected_model) {
    validate_model_contract(expected_model);
    const auto fields = read_manifest(artifact_directory / manifest_name);
    constexpr std::size_t expected_field_count = 25;
    if (fields.size() != expected_field_count || required(fields, "format") != hook_format) {
        throw std::invalid_argument("hook artifact manifest schema is invalid");
    }
    if (required(fields, "contract_sha256") != sha256_text(canonical_manifest(fields))) {
        throw std::invalid_argument("hook artifact contract hash mismatch");
    }

    ModelHookContract model{
        required(fields, "model_sha256"),
        required(fields, "llama_revision"),
        required(fields, "architecture"),
        required(fields, "target_tensor"),
        parse_integer<std::uint32_t>(required(fields, "hidden_dimension"),
                                     "hidden_dimension"),
    };
    validate_model_contract(model);
    if (model.model_sha256 != expected_model.model_sha256 ||
        model.llama_revision != expected_model.llama_revision ||
        model.architecture != expected_model.architecture ||
        model.target_tensor != expected_model.target_tensor ||
        model.hidden_dimension != expected_model.hidden_dimension) {
        throw std::invalid_argument("hook artifact does not match the active model contract");
    }

    const auto query_dimension = parse_integer<std::uint32_t>(
        required(fields, "query_dimension"), "query_dimension");
    const auto residual_count = parse_integer<std::size_t>(
        required(fields, "residual_count"), "residual_count");
    const auto projection_file = required(fields, "projection_file");
    const auto labels_file = required(fields, "residual_labels_file");
    const auto payload_file = required(fields, "residuals_file");
    const auto glamin_segment = required(fields, "glamin_directory");
    validate_segment(projection_file, "projection_file");
    validate_segment(labels_file, "residual_labels_file");
    validate_segment(payload_file, "residuals_file");
    validate_segment(glamin_segment, "glamin_directory");

    const auto projection_path = artifact_directory / projection_file;
    const auto labels_path = artifact_directory / labels_file;
    const auto payload_path = artifact_directory / payload_file;
    const auto glamin_directory = artifact_directory / glamin_segment;
    verify_checksum(projection_path, required(fields, "projection_sha256"), "projection");
    verify_checksum(labels_path, required(fields, "residual_labels_sha256"), "labels");
    verify_checksum(payload_path, required(fields, "residuals_sha256"), "residuals");
    verify_checksum(
        glamin_directory / "contracts.json",
        required(fields, "glamin_contracts_sha256"),
        "Glamin contracts");
    verify_checksum(
        glamin_directory / "vector_layout.json",
        required(fields, "glamin_layout_sha256"),
        "Glamin layout");
    verify_checksum(
        glamin_directory / "vectors.bin",
        required(fields, "glamin_vectors_sha256"),
        "Glamin vectors");

    const auto projection_elements = checked_product(model.hidden_dimension, query_dimension);
    const auto residual_elements = checked_product(residual_count, model.hidden_dimension);
    auto projection = decode_floats(read_bytes(
        projection_path, checked_product(projection_elements, sizeof(float))));
    auto labels = decode_labels(read_bytes(
        labels_path, checked_product(residual_count, sizeof(std::uint64_t))));
    auto residuals = decode_floats(read_bytes(
        payload_path, checked_product(residual_elements, sizeof(float))));

    return LoadedHookArtifact{
        required(fields, "contract_sha256"),
        std::move(model),
        HiddenStateHookConfig{
            expected_model.hidden_dimension,
            query_dimension,
            std::move(projection),
            parse_float(required(fields, "gate"), "gate"),
            parse_normalization(required(fields, "normalization")),
            parse_float(required(fields, "maximum_distance"), "maximum_distance"),
            parse_address_selection(required(fields, "address_selection")),
        },
        parse_metric(required(fields, "metric")),
        std::move(labels),
        std::move(residuals),
        glamin_directory,
        required(fields, "glamin_space_id"),
    };
}

PersistentHookGenerationStore::PersistentHookGenerationStore(GlaminRuntime& runtime)
    : generations_(runtime) {}

GlaminGenerationId PersistentHookGenerationStore::mount(
    std::string label,
    const std::filesystem::path& artifact_directory,
    const ModelHookContract& expected_model) {
    auto artifact = load_hook_artifact(artifact_directory, expected_model);
    const auto generation = generations_.mount_flat_artifact(
        std::move(label),
        artifact.glamin_directory.string(),
        artifact.glamin_space_id,
        artifact.metric);
    try {
        const auto vector_count = generations_.generation_vector_count(generation);
        if (generations_.generation_dimension(generation) !=
                artifact.hook_config.query_dimension ||
            vector_count != artifact.residual_labels.size()) {
            throw std::invalid_argument(
                "Glamin geometry does not match the hook address ledger");
        }
        for (const auto residual_label : artifact.residual_labels) {
            if (residual_label >= vector_count) {
                throw std::invalid_argument(
                    "hook residual labels do not cover the Glamin row addresses");
            }
        }
        auto mutable_payloads = std::make_shared<ResidualPayloadLedger>();
        const auto hidden_dimension =
            static_cast<std::size_t>(artifact.hook_config.hidden_dimension);
        const auto hidden_stride = static_cast<std::ptrdiff_t>(hidden_dimension);
        for (std::size_t index = 0; index < artifact.residual_labels.size(); ++index) {
            const auto begin = artifact.residuals.begin() +
                               static_cast<std::ptrdiff_t>(index * hidden_dimension);
            mutable_payloads->insert(
                generation,
                artifact.residual_labels[index],
                std::vector<float>(begin, begin + hidden_stride));
        }
        Resource resource{
            std::move(artifact.hook_config),
            std::move(mutable_payloads),
            artifact.model.target_tensor,
            artifact.contract_sha256,
        };
        const auto inserted = resources_.emplace(generation, std::move(resource));
        if (!inserted.second) {
            throw std::logic_error("hook generation identifier was reused");
        }
    } catch (...) {
        generations_.retire(generation);
        throw;
    }
    return generation;
}

void PersistentHookGenerationStore::activate(const GlaminGenerationId generation) {
    if (resources_.find(generation) == resources_.end()) {
        throw std::out_of_range("hook generation is not mounted");
    }
    generations_.activate(generation);
}

void PersistentHookGenerationStore::deactivate() {
    generations_.deactivate();
}

PinnedHookGeneration PersistentHookGenerationStore::pin_active() {
    auto pin = generations_.pin_active();
    const auto found = resources_.find(pin.id());
    if (found == resources_.end()) {
        pin.release();
        throw std::logic_error("active Glamin generation has no hook artifact");
    }
    return PinnedHookGeneration{
        FixedLayerMemoryHook(
            std::move(pin),
            found->second.hook_config,
            found->second.payloads),
        found->second.target_tensor,
        found->second.contract_sha256,
    };
}

void PersistentHookGenerationStore::retire(const GlaminGenerationId generation) {
    const auto found = resources_.find(generation);
    if (found == resources_.end()) {
        throw std::out_of_range("hook generation is not mounted");
    }
    generations_.retire(generation);
    resources_.erase(found);
}

GlaminGenerationId PersistentHookGenerationStore::active_generation() const noexcept {
    return generations_.active_generation();
}

std::size_t PersistentHookGenerationStore::mounted_generation_count() const noexcept {
    return resources_.size();
}

} // namespace gx1
