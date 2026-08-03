#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <vector>

namespace gx1 {

using ActivationStateSequence = std::vector<std::vector<float>>;

struct ActivationMemoryConstructionView {
    std::size_t association{0};
    ActivationStateSequence address_candidates;
    std::vector<float> query_action_state;
    std::vector<float> teacher_action_state;
};

struct ActivationMemoryValidationView {
    std::size_t association{0};
    ActivationStateSequence address_candidates;
};

struct ActivationMemoryCalibrationView {
    std::optional<std::size_t> association;
    ActivationStateSequence address_candidates;
};

struct ActivationMemorySelection {
    std::size_t key{0};
    std::size_t candidate{0};
    float distance{0.0F};
};

enum class ActivationProjectionStrategy : std::uint8_t {
    variance = 0,
    association_signal = 1,
};

enum class ActivationValidationScope : std::uint8_t {
    global = 0,
    association = 1,
};

struct ActivationMemoryBuildConfig {
    std::uint32_t query_dimension{0};
    float gate_interpolation{0.5F};
    bool include_cross_association_keys_in_gate{true};
    ActivationProjectionStrategy projection_strategy{
        ActivationProjectionStrategy::variance};
    ActivationValidationScope validation_scope{
        ActivationValidationScope::global};
};

struct ActivationMemoryBuildResult {
    std::uint32_t hidden_dimension{0};
    std::uint32_t query_dimension{0};
    std::vector<float> input_projection;
    std::vector<std::vector<float>> keys;
    std::vector<std::size_t> selected_candidates;
    std::map<std::size_t, float> association_margins;
    std::vector<std::vector<float>> residuals;
    std::vector<ActivationMemorySelection> validation_selections;
    std::vector<ActivationMemorySelection> negative_selections;
    float maximum_validation_distance{0.0F};
    float minimum_negative_distance{0.0F};
    float maximum_distance{0.0F};
    float minimum_validation_group_margin{0.0F};
    float maximum_negative_group_margin{0.0F};
};

class ActivationMemoryBuilder final {
public:
    [[nodiscard]] static ActivationMemoryBuildResult build(
        const std::vector<ActivationMemoryConstructionView>& construction_views,
        const std::vector<ActivationStateSequence>& calibration_negatives,
        const std::vector<ActivationMemoryValidationView>& validation_views,
        const ActivationMemoryBuildConfig& config);
    [[nodiscard]] static ActivationMemoryBuildResult build(
        const std::vector<ActivationMemoryConstructionView>& construction_views,
        const std::vector<ActivationMemoryCalibrationView>& calibration_negatives,
        const std::vector<ActivationMemoryValidationView>& validation_views,
        const ActivationMemoryBuildConfig& config);
};

} // namespace gx1
