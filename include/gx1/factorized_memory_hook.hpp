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

struct ContrastiveGateConfig {
    FactorSearchConfig search;
    std::vector<float> prototype;
    std::vector<float> negative_prototype;
    float minimum_margin{0.0F};
};

using RetrievalIntentGateConfig = ContrastiveGateConfig;

struct LabelConditionedGateEntry {
    std::uint64_t label{0};
    std::vector<float> positive_prototype;
    std::vector<float> negative_prototype;
    float maximum_distance{std::numeric_limits<float>::max()};
    float minimum_margin{0.0F};
};

struct LabelConditionedGateConfig {
    FactorSearchConfig search;
    std::vector<LabelConditionedGateEntry> entries;
};

struct SequenceEntityEvidenceConfig {
    FactorSearchConfig search;
    std::vector<std::vector<float>> prototypes;
    std::vector<std::uint64_t> prototype_labels;
    float minimum_identity_margin{0.0F};
};

struct EntityEvidenceDiagnostic {
    std::uint64_t label{0};
    float association_distance{std::numeric_limits<float>::max()};
    float evidence_distance{std::numeric_limits<float>::max()};
    float competitor_distance{std::numeric_limits<float>::max()};
    float identity_gap{-std::numeric_limits<float>::max()};
    float joint_score{std::numeric_limits<float>::max()};
    std::size_t association_state{0U};
    std::size_t evidence_state{0U};
    std::size_t evidence_prototype{0U};
    bool association_accepted{false};
    bool radius_accepted{false};
    bool margin_accepted{false};
    bool eligible{false};
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
    float compatibility_distance{0.0F};
    float intent_distance{0.0F};
    float intent_negative_distance{0.0F};
    float known_entity_distance{0.0F};
    float unknown_entity_distance{0.0F};
    std::uint64_t known_entity_verifier_label{0};
    std::uint64_t nearest_known_entity_label{0};
    std::vector<std::uint64_t> entity_candidate_labels;
    std::vector<float> entity_candidate_distances;
    std::vector<EntityEvidenceDiagnostic> entity_evidence_diagnostics;
    float entity_joint_score{std::numeric_limits<float>::max()};
    std::size_t action_variant{0};
    bool tuple_found{false};
    bool known_entity_accepted{true};
    bool known_entity_identity_consistent{true};
    bool compatibility_accepted{false};
    bool intent_accepted{false};
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

class TupleTargetStateLedger final {
public:
    void insert(
        std::uint64_t entity,
        std::uint64_t relation,
        std::vector<float> target_state);

    [[nodiscard]] const std::vector<float>* find(
        std::uint64_t entity,
        std::uint64_t relation) const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;

private:
    using Tuple = std::pair<std::uint64_t, std::uint64_t>;
    std::map<Tuple, std::vector<float>> target_states_;
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
        std::optional<FactorSearchConfig> action_config = std::nullopt,
        std::optional<RetrievalIntentGateConfig> intent_config = std::nullopt,
        bool scan_action_candidates = false,
        std::optional<ContrastiveGateConfig> known_entity_config = std::nullopt,
        std::optional<LabelConditionedGateConfig>
            label_conditioned_entity_config = std::nullopt,
        std::size_t joint_entity_candidate_count = 1U,
        std::optional<SequenceEntityEvidenceConfig>
            sequence_entity_evidence_config = std::nullopt);

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
    [[nodiscard]] FactorizedMemoryResult authorize_nearest(
        const std::vector<std::vector<float>>& address_states,
        const std::vector<float>& action_state) const;
    [[nodiscard]] FactorizedMemoryResult authorize_nearest(
        const std::vector<std::vector<float>>& entity_states,
        const std::vector<std::vector<float>>& relation_states,
        const std::vector<float>& action_state) const;

    [[nodiscard]] std::uint32_t hidden_dimension() const noexcept;
    [[nodiscard]] GlaminGenerationId entity_generation() const;
    [[nodiscard]] GlaminGenerationId relation_generation() const;

private:
    [[nodiscard]] std::pair<FactorizedMemoryResult, TupleResidualMatch>
    authorize_selection(
        const std::vector<std::vector<float>>& entity_states,
        const std::vector<std::vector<float>>& relation_states,
        const std::vector<float>& action_state) const;
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
    std::optional<RetrievalIntentGateConfig> intent_config_;
    bool scan_action_candidates_{false};
    std::optional<ContrastiveGateConfig> known_entity_config_;
    std::optional<LabelConditionedGateConfig> label_conditioned_entity_config_;
    std::size_t joint_entity_candidate_count_{1U};
    std::optional<SequenceEntityEvidenceConfig>
        sequence_entity_evidence_config_;
};

} // namespace gx1
