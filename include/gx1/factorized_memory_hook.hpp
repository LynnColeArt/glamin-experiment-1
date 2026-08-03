#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "gx1/hidden_state_hook.hpp"

namespace gx1 {

struct FactorSearchConfig {
    std::uint32_t hidden_dimension{0};
    std::uint32_t query_dimension{0};
    std::vector<float> input_projection;
    ProjectionNormalization query_normalization{ProjectionNormalization::none};
    float maximum_distance{std::numeric_limits<float>::max()};
};

struct FactorEvidence {
    GlaminGenerationId generation{0};
    std::uint64_t memory_label{0};
    std::uint64_t factor_label{0};
    float distance{0.0F};
    bool accepted{false};
    std::size_t address_candidate{0};
};

struct FactorizedMemoryResult {
    FactorEvidence entity;
    FactorEvidence relation;
    float gate{0.0F};
    float action_distance{0.0F};
    std::size_t action_variant{0};
    bool tuple_found{false};
    bool action_accepted{false};
    bool applied{false};
};

struct TupleResidualMatch {
    const std::vector<float>* residual{nullptr};
    float distance{0.0F};
    std::size_t variant{0};
};

class TupleResidualLedger final {
public:
    void insert(
        std::uint64_t entity,
        std::uint64_t relation,
        std::vector<float> residual);
    void insert_variant(
        std::uint64_t entity,
        std::uint64_t relation,
        std::vector<float> action_prototype,
        std::vector<float> residual);

    [[nodiscard]] TupleResidualMatch select(
        std::uint64_t entity,
        std::uint64_t relation,
        const std::vector<float>& action_state) const;
    [[nodiscard]] std::size_t size() const noexcept;

private:
    using Tuple = std::pair<std::uint64_t, std::uint64_t>;
    struct Variant {
        std::vector<float> action_prototype;
        std::vector<float> residual;
    };
    std::map<Tuple, std::vector<Variant>> residuals_;
};

class FactorizedLayerMemoryHook final {
public:
    FactorizedLayerMemoryHook(
        GlaminGenerationPin entity_pin,
        FactorSearchConfig entity_config,
        std::vector<std::uint64_t> entity_labels,
        GlaminGenerationPin relation_pin,
        FactorSearchConfig relation_config,
        std::vector<std::uint64_t> relation_labels,
        float gate,
        std::shared_ptr<const TupleResidualLedger> payloads,
        float maximum_action_distance = std::numeric_limits<float>::max(),
        std::optional<FactorSearchConfig> action_config = std::nullopt);

    FactorizedLayerMemoryHook(const FactorizedLayerMemoryHook&) = delete;
    FactorizedLayerMemoryHook& operator=(const FactorizedLayerMemoryHook&) = delete;
    FactorizedLayerMemoryHook(FactorizedLayerMemoryHook&&) noexcept = default;
    FactorizedLayerMemoryHook& operator=(FactorizedLayerMemoryHook&&) noexcept = default;

    [[nodiscard]] FactorizedMemoryResult apply_nearest(
        const std::vector<std::vector<float>>& address_states,
        std::vector<float>& hidden_state) const;
    [[nodiscard]] FactorizedMemoryResult apply_nearest(
        const std::vector<std::vector<float>>& entity_states,
        const std::vector<std::vector<float>>& relation_states,
        std::vector<float>& hidden_state) const;

    [[nodiscard]] std::uint32_t hidden_dimension() const noexcept;
    [[nodiscard]] GlaminGenerationId entity_generation() const;
    [[nodiscard]] GlaminGenerationId relation_generation() const;

private:
    [[nodiscard]] FactorEvidence nearest(
        const GlaminGenerationPin& pin,
        const FactorSearchConfig& config,
        const std::vector<std::uint64_t>& factor_labels,
        const std::vector<std::vector<float>>& states) const;
    [[nodiscard]] std::vector<float> project(
        const FactorSearchConfig& config,
        const std::vector<float>& hidden_state) const;

    GlaminGenerationPin entity_pin_;
    FactorSearchConfig entity_config_;
    std::vector<std::uint64_t> entity_labels_;
    GlaminGenerationPin relation_pin_;
    FactorSearchConfig relation_config_;
    std::vector<std::uint64_t> relation_labels_;
    float gate_{0.0F};
    std::shared_ptr<const TupleResidualLedger> payloads_;
    float maximum_action_distance_{std::numeric_limits<float>::max()};
    std::optional<FactorSearchConfig> action_config_;
};

} // namespace gx1
