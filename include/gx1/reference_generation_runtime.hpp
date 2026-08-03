#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace gx1::reference {

using GenerationId = std::uint64_t;

class Generation final {
public:
    Generation(
        GenerationId id,
        std::string label,
        std::vector<float> scale,
        std::vector<float> bias);

    [[nodiscard]] GenerationId id() const noexcept;
    [[nodiscard]] const std::string& label() const noexcept;
    [[nodiscard]] std::size_t dimension() const noexcept;
    [[nodiscard]] std::vector<float> traverse(
        const std::vector<float>& input) const;

private:
    GenerationId id_;
    std::string label_;
    std::vector<float> scale_;
    std::vector<float> bias_;
};

class GenerationLease final {
public:
    [[nodiscard]] GenerationId id() const;
    [[nodiscard]] const std::string& label() const;
    [[nodiscard]] std::vector<float> traverse(
        const std::vector<float>& input) const;
    [[nodiscard]] explicit operator bool() const noexcept;

private:
    friend class GenerationRuntime;

    explicit GenerationLease(std::shared_ptr<const Generation> generation);

    std::shared_ptr<const Generation> generation_;
};

class GenerationRuntime final {
public:
    void install(std::shared_ptr<const Generation> generation);
    void activate(GenerationId id);
    [[nodiscard]] GenerationLease pin_active() const;
    void retire(GenerationId id);

    [[nodiscard]] GenerationId active_generation() const;
    [[nodiscard]] std::size_t mounted_generation_count() const;

private:
    mutable std::mutex mutex_;
    std::unordered_map<GenerationId, std::shared_ptr<const Generation>> generations_;
    GenerationId active_id_{0};
};

} // namespace gx1::reference
