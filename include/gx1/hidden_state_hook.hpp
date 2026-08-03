#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "gx1/glamin_runtime.hpp"

namespace gx1 {

enum class ProjectionNormalization {
    none,
    l2,
};

enum class AddressSelectionPolicy {
    last_token,
    all_token_rows,
};

struct HiddenStateHookConfig {
    std::uint32_t hidden_dimension{0};
    std::uint32_t query_dimension{0};
    std::vector<float> input_projection;
    float gate{0.0F};
    ProjectionNormalization query_normalization{ProjectionNormalization::none};
    float maximum_distance{std::numeric_limits<float>::max()};
    AddressSelectionPolicy address_selection{AddressSelectionPolicy::last_token};
};

struct HiddenStateHookResult {
    GlaminGenerationId generation{0};
    std::uint64_t label{0};
    float distance{0.0F};
    float gate{0.0F};
    bool applied{false};
    std::size_t address_candidate{0};
};

class ResidualPayloadLedger final {
public:
    void insert(
        GlaminGenerationId generation,
        std::uint64_t label,
        std::vector<float> residual);

    [[nodiscard]] const std::vector<float>& at(
        GlaminGenerationId generation,
        std::uint64_t label) const;
    [[nodiscard]] std::size_t size() const noexcept;

private:
    using Address = std::pair<GlaminGenerationId, std::uint64_t>;
    std::map<Address, std::vector<float>> residuals_;
};

class FixedLayerMemoryHook final {
public:
    FixedLayerMemoryHook(
        GlaminGenerationPin pin,
        HiddenStateHookConfig config,
        std::shared_ptr<const ResidualPayloadLedger> payloads);

    FixedLayerMemoryHook(const FixedLayerMemoryHook&) = delete;
    FixedLayerMemoryHook& operator=(const FixedLayerMemoryHook&) = delete;
    FixedLayerMemoryHook(FixedLayerMemoryHook&&) noexcept = default;
    FixedLayerMemoryHook& operator=(FixedLayerMemoryHook&&) noexcept = default;

    [[nodiscard]] HiddenStateHookResult apply(std::vector<float>& hidden_state);
    [[nodiscard]] HiddenStateHookResult apply(
        const std::vector<float>& address_state,
        std::vector<float>& hidden_state);
    [[nodiscard]] HiddenStateHookResult apply_nearest(
        const std::vector<std::vector<float>>& address_states,
        std::vector<float>& hidden_state);
    [[nodiscard]] std::uint32_t hidden_dimension() const noexcept;
    [[nodiscard]] AddressSelectionPolicy address_selection() const noexcept;
    [[nodiscard]] GlaminGenerationId generation() const;

private:
    [[nodiscard]] HiddenStateHookResult search(
        const std::vector<float>& address_state) const;
    [[nodiscard]] HiddenStateHookResult inject(
        HiddenStateHookResult result,
        std::vector<float>& hidden_state) const;
    [[nodiscard]] std::vector<float> project(
        const std::vector<float>& hidden_state) const;

    GlaminGenerationPin pin_;
    HiddenStateHookConfig config_;
    std::shared_ptr<const ResidualPayloadLedger> payloads_;
};

} // namespace gx1
