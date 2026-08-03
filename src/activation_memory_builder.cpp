#include "gx1/activation_memory_builder.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gx1 {

namespace {

bool finite_values(const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(), [](const float value) {
        return std::isfinite(value);
    });
}

void validate_state(
    const std::vector<float>& state,
    const std::size_t hidden_dimension,
    const char* description) {
    if (state.size() != hidden_dimension || !finite_values(state)) {
        throw std::invalid_argument(
            std::string(description) + " has the wrong shape or values");
    }
}

void validate_sequence(
    const ActivationStateSequence& sequence,
    const std::size_t hidden_dimension,
    const char* description) {
    if (sequence.empty()) {
        throw std::invalid_argument(std::string(description) + " is empty");
    }
    for (const auto& state : sequence) {
        validate_state(state, hidden_dimension, description);
    }
}

float squared_distance(
    const std::vector<float>& left,
    const std::vector<float>& right) {
    if (left.size() != right.size()) {
        throw std::invalid_argument("activation-memory distance widths differ");
    }
    double distance = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        const auto difference = static_cast<double>(left[index]) - right[index];
        distance += difference * difference;
    }
    return static_cast<float>(distance);
}

std::vector<float> make_projection(
    const std::vector<std::vector<float>>& states,
    const std::uint32_t query_dimension) {
    if (states.empty() || states.front().empty() ||
        query_dimension == 0 || query_dimension > states.front().size()) {
        throw std::invalid_argument("projection construction states are invalid");
    }
    const auto hidden_dimension = states.front().size();
    std::vector<double> scores(hidden_dimension, 0.0);
    for (std::size_t column = 0; column < hidden_dimension; ++column) {
        double mean = 0.0;
        for (const auto& state : states) {
            validate_state(state, hidden_dimension, "projection state");
            mean += state[column];
        }
        mean /= static_cast<double>(states.size());
        for (const auto& state : states) {
            const auto difference = static_cast<double>(state[column]) - mean;
            scores[column] += difference * difference;
        }
    }

    std::vector<std::size_t> columns(hidden_dimension);
    std::iota(columns.begin(), columns.end(), 0U);
    std::partial_sort(
        columns.begin(),
        columns.begin() + static_cast<std::ptrdiff_t>(query_dimension),
        columns.end(),
        [&scores](const std::size_t left, const std::size_t right) {
            return scores[left] > scores[right];
        });

    std::vector<float> projection(
        static_cast<std::size_t>(query_dimension) * hidden_dimension, 0.0F);
    for (std::size_t row = 0; row < query_dimension; ++row) {
        projection[row * hidden_dimension + columns[row]] = 1.0F;
    }
    return projection;
}

std::vector<float> project_normalized(
    const std::vector<float>& state,
    const std::vector<float>& projection,
    const std::uint32_t query_dimension) {
    const auto hidden_dimension = state.size();
    if (projection.size() !=
        static_cast<std::size_t>(query_dimension) * hidden_dimension) {
        throw std::invalid_argument("activation-memory projection has the wrong shape");
    }
    std::vector<float> result(query_dimension, 0.0F);
    for (std::size_t row = 0; row < query_dimension; ++row) {
        double sum = 0.0;
        for (std::size_t column = 0; column < hidden_dimension; ++column) {
            sum += static_cast<double>(projection[row * hidden_dimension + column]) *
                   static_cast<double>(state[column]);
        }
        result[row] = static_cast<float>(sum);
    }
    double squared_norm = 0.0;
    for (const auto value : result) {
        squared_norm += static_cast<double>(value) * value;
    }
    if (!(squared_norm > 0.0) || !std::isfinite(squared_norm)) {
        throw std::runtime_error("activation-memory projection produced a zero query");
    }
    const auto inverse_norm = 1.0 / std::sqrt(squared_norm);
    for (auto& value : result) {
        value = static_cast<float>(static_cast<double>(value) * inverse_norm);
    }
    return result;
}

ActivationMemorySelection nearest_key(
    const std::vector<float>& query,
    const std::vector<std::vector<float>>& keys,
    const std::size_t candidate = 0U) {
    if (keys.empty()) {
        throw std::invalid_argument("cannot search an empty activation-memory key set");
    }
    ActivationMemorySelection nearest{0U, candidate, squared_distance(query, keys[0])};
    for (std::size_t index = 1; index < keys.size(); ++index) {
        const auto distance = squared_distance(query, keys[index]);
        if (distance < nearest.distance) {
            nearest.key = index;
            nearest.distance = distance;
        }
    }
    return nearest;
}

std::pair<ActivationMemorySelection, float> nearest_candidate_grouped(
    const ActivationStateSequence& states,
    const std::vector<float>& projection,
    const std::uint32_t query_dimension,
    const std::vector<std::vector<float>>& keys,
    const std::vector<std::size_t>& key_groups) {
    if (keys.size() != key_groups.size() || keys.empty()) {
        throw std::invalid_argument("grouped memory keys are invalid");
    }
    const auto nearest_for_state = [&keys, &key_groups](
                                       const std::vector<float>& query,
                                       const std::size_t candidate) {
        const auto nearest = nearest_key(query, keys, candidate);
        auto competitor = std::numeric_limits<float>::max();
        for (std::size_t key = 0; key < keys.size(); ++key) {
            if (key_groups[key] != key_groups[nearest.key]) {
                competitor = std::min(
                    competitor, squared_distance(query, keys[key]));
            }
        }
        return std::pair<ActivationMemorySelection, float>{
            nearest, competitor - nearest.distance};
    };

    auto grouped = nearest_for_state(
        project_normalized(states.front(), projection, query_dimension), 0U);
    for (std::size_t candidate = 1; candidate < states.size(); ++candidate) {
        auto current = nearest_for_state(
            project_normalized(states[candidate], projection, query_dimension),
            candidate);
        if (current.first.distance < grouped.first.distance) {
            grouped = std::move(current);
        }
    }
    return grouped;
}

} // namespace

ActivationMemoryBuildResult ActivationMemoryBuilder::build(
    const std::vector<ActivationMemoryConstructionView>& construction_views,
    const std::vector<ActivationStateSequence>& calibration_negatives,
    const std::vector<ActivationMemoryValidationView>& validation_views,
    const ActivationMemoryBuildConfig& config) {
    if (construction_views.empty() || calibration_negatives.empty() ||
        validation_views.empty() || config.query_dimension == 0) {
        throw std::invalid_argument("activation-memory construction inputs are incomplete");
    }
    if (!(config.gate_interpolation >= 0.0F &&
          config.gate_interpolation < 1.0F) ||
        !std::isfinite(config.gate_interpolation)) {
        throw std::invalid_argument(
            "activation-memory gate interpolation must be in [0, 1)");
    }
    const auto hidden_dimension = construction_views.front().query_action_state.size();
    if (hidden_dimension == 0 || config.query_dimension > hidden_dimension) {
        throw std::invalid_argument("activation-memory dimensions are invalid");
    }

    std::map<std::size_t, std::vector<std::size_t>> associations;
    std::vector<std::vector<float>> projection_states;
    for (std::size_t index = 0; index < construction_views.size(); ++index) {
        const auto& view = construction_views[index];
        validate_sequence(view.address_candidates, hidden_dimension, "construction view");
        validate_state(view.query_action_state, hidden_dimension, "query action state");
        validate_state(view.teacher_action_state, hidden_dimension, "teacher action state");
        projection_states.insert(
            projection_states.end(),
            view.address_candidates.begin(),
            view.address_candidates.end());
        associations[view.association].push_back(index);
    }
    for (const auto& negative : calibration_negatives) {
        validate_sequence(negative, hidden_dimension, "calibration negative");
        projection_states.insert(
            projection_states.end(), negative.begin(), negative.end());
    }
    for (const auto& validation : validation_views) {
        validate_sequence(validation.address_candidates, hidden_dimension, "validation view");
        if (associations.find(validation.association) == associations.end()) {
            throw std::invalid_argument("validation view names an unknown association");
        }
    }

    const auto projection = make_projection(projection_states, config.query_dimension);
    std::vector<std::vector<std::vector<float>>> projected_views(
        construction_views.size());
    for (std::size_t index = 0; index < construction_views.size(); ++index) {
        for (const auto& state : construction_views[index].address_candidates) {
            projected_views[index].push_back(
                project_normalized(state, projection, config.query_dimension));
        }
    }
    std::vector<std::vector<float>> projected_negatives;
    for (const auto& negative : calibration_negatives) {
        for (const auto& state : negative) {
            projected_negatives.push_back(
                project_normalized(state, projection, config.query_dimension));
        }
    }

    ActivationMemoryBuildResult result;
    result.hidden_dimension = static_cast<std::uint32_t>(hidden_dimension);
    result.query_dimension = config.query_dimension;
    result.input_projection = projection;
    result.keys.resize(construction_views.size());
    result.selected_candidates.resize(construction_views.size(), 0U);
    result.residuals.resize(
        construction_views.size(), std::vector<float>(hidden_dimension, 0.0F));
    std::vector<std::size_t> key_groups(construction_views.size(), 0U);
    for (std::size_t index = 0; index < construction_views.size(); ++index) {
        key_groups[index] = construction_views[index].association;
    }

    for (const auto& association : associations) {
        const auto& members = association.second;
        auto negatives = projected_negatives;
        for (std::size_t index = 0; index < construction_views.size(); ++index) {
            if (construction_views[index].association != association.first) {
                negatives.insert(
                    negatives.end(),
                    projected_views[index].begin(),
                    projected_views[index].end());
            }
        }
        if (negatives.empty()) {
            throw std::invalid_argument("automatic key construction has no negatives");
        }

        auto association_margin = std::numeric_limits<float>::max();
        std::vector<std::size_t> best_candidates;
        best_candidates.reserve(members.size());
        for (const auto member : members) {
            auto best_separation = -std::numeric_limits<float>::infinity();
            std::size_t best_candidate = 0U;
            for (std::size_t candidate = 0;
                 candidate < projected_views[member].size();
                 ++candidate) {
                const auto separation = nearest_key(
                    projected_views[member][candidate], negatives).distance;
                if (separation > best_separation) {
                    best_separation = separation;
                    best_candidate = candidate;
                }
            }
            if (!(best_separation > 0.0F) || !std::isfinite(best_separation)) {
                throw std::runtime_error(
                    "automatic key construction found no separated candidate for "
                    "association " + std::to_string(association.first) +
                    ", view " + std::to_string(member) +
                    ", best separation " + std::to_string(best_separation));
            }
            best_candidates.push_back(best_candidate);
            association_margin = std::min(association_margin, best_separation);
        }
        result.association_margins.emplace(
            association.first, association_margin);

        for (std::size_t member_index = 0; member_index < members.size(); ++member_index) {
            const auto view_index = members[member_index];
            const auto candidate = best_candidates[member_index];
            result.selected_candidates[view_index] = candidate;
            result.keys[view_index] = projected_views[view_index][candidate];
            for (std::size_t column = 0; column < hidden_dimension; ++column) {
                result.residuals[view_index][column] =
                    construction_views[view_index].teacher_action_state[column] -
                    construction_views[view_index].query_action_state[column];
            }
        }
    }

    auto minimum_negative = std::numeric_limits<float>::max();
    float maximum_negative_group_margin = 0.0F;
    if (config.include_cross_association_keys_in_gate) {
        for (std::size_t left = 0; left < result.keys.size(); ++left) {
            for (std::size_t right = left + 1U; right < result.keys.size(); ++right) {
                if (construction_views[left].association !=
                    construction_views[right].association) {
                    minimum_negative = std::min(
                        minimum_negative,
                        squared_distance(result.keys[left], result.keys[right]));
                }
            }
        }
    }
    for (const auto& negative : calibration_negatives) {
        const auto grouped = nearest_candidate_grouped(
            negative,
            projection,
            config.query_dimension,
            result.keys,
            key_groups);
        const auto& selection = grouped.first;
        result.negative_selections.push_back(selection);
        minimum_negative = std::min(minimum_negative, selection.distance);
        maximum_negative_group_margin = std::max(
            maximum_negative_group_margin, grouped.second);
    }

    float maximum_validation = 0.0F;
    auto minimum_validation_group_margin = std::numeric_limits<float>::max();
    for (std::size_t index = 0; index < validation_views.size(); ++index) {
        const auto& validation = validation_views[index];
        const auto grouped = nearest_candidate_grouped(
            validation.address_candidates,
            projection,
            config.query_dimension,
            result.keys,
            key_groups);
        const auto& selection = grouped.first;
        if (construction_views[selection.key].association != validation.association) {
            throw std::runtime_error(
                "validation view " + std::to_string(index) + " for association " +
                std::to_string(validation.association) + " selected key " +
                std::to_string(selection.key) + " from association " +
                std::to_string(construction_views[selection.key].association) +
                " at distance " + std::to_string(selection.distance));
        }
        result.validation_selections.push_back(selection);
        maximum_validation = std::max(maximum_validation, selection.distance);
        minimum_validation_group_margin = std::min(
            minimum_validation_group_margin, grouped.second);
    }
    if (!(maximum_validation < minimum_negative) ||
        !std::isfinite(maximum_validation) || !std::isfinite(minimum_negative)) {
        throw std::runtime_error(
            "memory neighborhoods do not separate validation from negatives: "
            "maximum validation " + std::to_string(maximum_validation) +
            ", minimum negative " + std::to_string(minimum_negative) +
            ", minimum validation group margin " +
            std::to_string(minimum_validation_group_margin) +
            ", maximum negative group margin " +
            std::to_string(maximum_negative_group_margin));
    }
    result.maximum_validation_distance = maximum_validation;
    result.minimum_negative_distance = minimum_negative;
    result.maximum_distance = maximum_validation +
                              config.gate_interpolation *
                                  (minimum_negative - maximum_validation);
    result.minimum_validation_group_margin = minimum_validation_group_margin;
    result.maximum_negative_group_margin = maximum_negative_group_margin;
    return result;
}

} // namespace gx1
