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
    const std::vector<ActivationMemoryConstructionView>& construction_views,
    const std::vector<ActivationMemoryCalibrationView>& calibration_negatives,
    const ActivationMemoryBuildConfig& config) {
    const auto query_dimension = config.query_dimension;
    if (states.empty() || states.front().empty() ||
        query_dimension == 0 || query_dimension > states.front().size()) {
        throw std::invalid_argument("projection construction states are invalid");
    }
    const auto hidden_dimension = states.front().size();
    std::vector<double> scores(hidden_dimension, 0.0);
    if (config.projection_strategy ==
        ActivationProjectionStrategy::authorization_signal) {
        std::map<std::size_t, std::vector<const std::vector<float>*>> positives;
        std::map<std::size_t, std::vector<const std::vector<float>*>> negatives;
        for (const auto& view : construction_views) {
            positives[view.association].push_back(&view.query_action_state);
        }
        for (const auto& group : positives) {
            for (const auto& negative : calibration_negatives) {
                if (!negative.association ||
                    *negative.association == group.first) {
                    for (const auto& state : negative.address_candidates) {
                        negatives[group.first].push_back(&state);
                    }
                }
            }
            if (negatives[group.first].empty()) {
                throw std::invalid_argument(
                    "authorization-signal projection requires a negative "
                    "for every association");
            }
        }
        for (std::size_t column = 0; column < hidden_dimension; ++column) {
            double between = 0.0;
            double within = 0.0;
            for (const auto& group : positives) {
                const auto& positive_states = group.second;
                const auto& negative_states = negatives.at(group.first);
                double positive_mean = 0.0;
                double negative_mean = 0.0;
                for (const auto* state : positive_states) {
                    validate_state(*state, hidden_dimension, "projection positive");
                    positive_mean += (*state)[column];
                }
                for (const auto* state : negative_states) {
                    validate_state(*state, hidden_dimension, "projection negative");
                    negative_mean += (*state)[column];
                }
                positive_mean /= static_cast<double>(positive_states.size());
                negative_mean /= static_cast<double>(negative_states.size());
                const auto mean_difference = positive_mean - negative_mean;
                between += mean_difference * mean_difference;
                for (const auto* state : positive_states) {
                    const auto difference =
                        static_cast<double>((*state)[column]) - positive_mean;
                    within += difference * difference;
                }
                for (const auto* state : negative_states) {
                    const auto difference =
                        static_cast<double>((*state)[column]) - negative_mean;
                    within += difference * difference;
                }
            }
            const auto total = between + within;
            scores[column] = total > 0.0 ? between / total : 0.0;
        }
    } else if (config.projection_strategy ==
        ActivationProjectionStrategy::association_signal) {
        std::map<std::size_t, std::vector<const std::vector<float>*>> groups;
        for (const auto& view : construction_views) {
            groups[view.association].push_back(&view.query_action_state);
        }
        if (groups.size() < 2U) {
            throw std::invalid_argument(
                "association-signal projection requires multiple associations");
        }
        std::vector<std::vector<double>> group_means(
            groups.size(), std::vector<double>(hidden_dimension, 0.0));
        std::size_t group_index = 0U;
        for (const auto& group : groups) {
            for (const auto* state : group.second) {
                validate_state(*state, hidden_dimension, "projection state");
                for (std::size_t column = 0; column < hidden_dimension; ++column) {
                    group_means[group_index][column] += (*state)[column];
                }
            }
            const auto inverse_count = 1.0 / static_cast<double>(group.second.size());
            for (auto& value : group_means[group_index]) {
                value *= inverse_count;
            }
            ++group_index;
        }
        for (std::size_t column = 0; column < hidden_dimension; ++column) {
            double overall_mean = 0.0;
            for (const auto& mean : group_means) {
                overall_mean += mean[column];
            }
            overall_mean /= static_cast<double>(group_means.size());
            double between = 0.0;
            double within = 0.0;
            group_index = 0U;
            for (const auto& group : groups) {
                const auto mean_difference =
                    group_means[group_index][column] - overall_mean;
                between += mean_difference * mean_difference;
                for (const auto* state : group.second) {
                    const auto difference =
                        static_cast<double>((*state)[column]) -
                        group_means[group_index][column];
                    within += difference * difference;
                }
                ++group_index;
            }
            const auto total = between + within;
            scores[column] = total > 0.0 ? between / total : 0.0;
        }
    } else {
        for (const auto& state : states) {
            validate_state(state, hidden_dimension, "projection state");
        }
        for (std::size_t column = 0; column < hidden_dimension; ++column) {
            double mean = 0.0;
            for (const auto& state : states) {
                mean += state[column];
            }
            mean /= static_cast<double>(states.size());
            for (const auto& state : states) {
                const auto difference = static_cast<double>(state[column]) - mean;
                scores[column] += difference * difference;
            }
        }
    }

    std::vector<std::size_t> columns(hidden_dimension);
    std::iota(columns.begin(), columns.end(), 0U);
    std::partial_sort(
        columns.begin(),
        columns.begin() + static_cast<std::ptrdiff_t>(query_dimension),
        columns.end(),
        [&scores](const std::size_t left, const std::size_t right) {
            return scores[left] == scores[right] ? left < right
                                                  : scores[left] > scores[right];
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

ActivationMemorySelection nearest_key_in_association(
    const std::vector<float>& query,
    const std::vector<std::vector<float>>& keys,
    const std::vector<std::size_t>& key_groups,
    const std::size_t association,
    const std::size_t candidate) {
    if (keys.size() != key_groups.size() || keys.empty()) {
        throw std::invalid_argument("scoped memory keys are invalid");
    }
    ActivationMemorySelection nearest;
    nearest.candidate = candidate;
    nearest.distance = std::numeric_limits<float>::max();
    bool found = false;
    for (std::size_t key = 0; key < keys.size(); ++key) {
        if (key_groups[key] != association) {
            continue;
        }
        const auto distance = squared_distance(query, keys[key]);
        if (!found || distance < nearest.distance) {
            nearest.key = key;
            nearest.distance = distance;
            found = true;
        }
    }
    if (!found) {
        throw std::invalid_argument("scoped memory association has no keys");
    }
    return nearest;
}

ActivationMemorySelection nearest_candidate_in_association(
    const ActivationStateSequence& states,
    const std::vector<float>& projection,
    const std::uint32_t query_dimension,
    const std::vector<std::vector<float>>& keys,
    const std::vector<std::size_t>& key_groups,
    const std::size_t association) {
    auto nearest = nearest_key_in_association(
        project_normalized(states.front(), projection, query_dimension),
        keys,
        key_groups,
        association,
        0U);
    for (std::size_t candidate = 1; candidate < states.size(); ++candidate) {
        const auto current = nearest_key_in_association(
            project_normalized(states[candidate], projection, query_dimension),
            keys,
            key_groups,
            association,
            candidate);
        if (current.distance < nearest.distance) {
            nearest = current;
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
    std::vector<ActivationMemoryCalibrationView> unscoped_negatives;
    unscoped_negatives.reserve(calibration_negatives.size());
    for (const auto& negative : calibration_negatives) {
        unscoped_negatives.push_back({std::nullopt, negative});
    }
    return build(
        construction_views, unscoped_negatives, validation_views, config);
}

ActivationMemoryBuildResult ActivationMemoryBuilder::build(
    const std::vector<ActivationMemoryConstructionView>& construction_views,
    const std::vector<ActivationMemoryCalibrationView>& calibration_negatives,
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
    if (config.projection_strategy != ActivationProjectionStrategy::variance &&
        config.projection_strategy !=
            ActivationProjectionStrategy::association_signal &&
        config.projection_strategy !=
            ActivationProjectionStrategy::authorization_signal) {
        throw std::invalid_argument(
            "activation-memory projection strategy is invalid");
    }
    if (config.validation_scope != ActivationValidationScope::global &&
        config.validation_scope != ActivationValidationScope::association) {
        throw std::invalid_argument(
            "activation-memory validation scope is invalid");
    }
    if (config.key_strategy != ActivationKeyStrategy::selected_views &&
        config.key_strategy != ActivationKeyStrategy::association_centroid) {
        throw std::invalid_argument("activation-memory key strategy is invalid");
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
        validate_sequence(
            negative.address_candidates,
            hidden_dimension,
            "calibration negative");
        if (negative.association &&
            associations.find(*negative.association) == associations.end()) {
            throw std::invalid_argument(
                "calibration negative names an unknown association");
        }
        projection_states.insert(
            projection_states.end(),
            negative.address_candidates.begin(),
            negative.address_candidates.end());
    }
    for (const auto& validation : validation_views) {
        validate_sequence(validation.address_candidates, hidden_dimension, "validation view");
        if (associations.find(validation.association) == associations.end()) {
            throw std::invalid_argument("validation view names an unknown association");
        }
    }

    const auto projection = make_projection(
        projection_states, construction_views, calibration_negatives, config);
    std::vector<std::vector<std::vector<float>>> projected_views(
        construction_views.size());
    for (std::size_t index = 0; index < construction_views.size(); ++index) {
        for (const auto& state : construction_views[index].address_candidates) {
            projected_views[index].push_back(
                project_normalized(state, projection, config.query_dimension));
        }
    }
    struct ProjectedNegative {
        std::optional<std::size_t> association;
        std::vector<std::vector<float>> states;
    };
    std::vector<ProjectedNegative> projected_negatives;
    for (const auto& negative : calibration_negatives) {
        ProjectedNegative projected{negative.association, {}};
        for (const auto& state : negative.address_candidates) {
            projected.states.push_back(
                project_normalized(state, projection, config.query_dimension));
        }
        projected_negatives.push_back(std::move(projected));
    }

    ActivationMemoryBuildResult result;
    result.hidden_dimension = static_cast<std::uint32_t>(hidden_dimension);
    result.query_dimension = config.query_dimension;
    result.input_projection = projection;
    result.selected_candidates.resize(construction_views.size(), 0U);
    std::vector<std::vector<float>> selected_keys(construction_views.size());
    std::vector<std::vector<float>> selected_residuals(
        construction_views.size(), std::vector<float>(hidden_dimension, 0.0F));

    for (const auto& association : associations) {
        const auto& members = association.second;
        std::vector<std::vector<float>> negatives;
        for (const auto& negative : projected_negatives) {
            if (!negative.association ||
                *negative.association == association.first) {
                negatives.insert(
                    negatives.end(),
                    negative.states.begin(),
                    negative.states.end());
            }
        }
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
            selected_keys[view_index] = projected_views[view_index][candidate];
            for (std::size_t column = 0; column < hidden_dimension; ++column) {
                selected_residuals[view_index][column] =
                    construction_views[view_index].teacher_action_state[column] -
                    construction_views[view_index].query_action_state[column];
            }
        }
    }

    if (config.key_strategy == ActivationKeyStrategy::selected_views) {
        result.keys = std::move(selected_keys);
        result.residuals = std::move(selected_residuals);
        result.key_associations.reserve(construction_views.size());
        for (const auto& view : construction_views) {
            result.key_associations.push_back(view.association);
        }
    } else {
        result.keys.reserve(associations.size());
        result.residuals.reserve(associations.size());
        result.key_associations.reserve(associations.size());
        for (const auto& association : associations) {
            std::vector<float> centroid(config.query_dimension, 0.0F);
            std::vector<float> residual(hidden_dimension, 0.0F);
            for (const auto member : association.second) {
                for (std::size_t column = 0;
                     column < config.query_dimension;
                     ++column) {
                    centroid[column] += selected_keys[member][column];
                }
                for (std::size_t column = 0;
                     column < hidden_dimension;
                     ++column) {
                    residual[column] += selected_residuals[member][column];
                }
            }
            double squared_norm = 0.0;
            for (const auto value : centroid) {
                squared_norm += static_cast<double>(value) * value;
            }
            if (!(squared_norm > 0.0) || !std::isfinite(squared_norm)) {
                throw std::runtime_error(
                    "association centroid produced a zero key");
            }
            const auto inverse_norm = 1.0 / std::sqrt(squared_norm);
            const auto inverse_members =
                1.0F / static_cast<float>(association.second.size());
            for (auto& value : centroid) {
                value = static_cast<float>(
                    static_cast<double>(value) * inverse_norm);
            }
            for (auto& value : residual) {
                value *= inverse_members;
            }
            result.keys.push_back(std::move(centroid));
            result.residuals.push_back(std::move(residual));
            result.key_associations.push_back(association.first);
        }
    }
    const auto& key_groups = result.key_associations;

    auto minimum_negative = std::numeric_limits<float>::max();
    float maximum_negative_group_margin = 0.0F;
    if (config.include_cross_association_keys_in_gate) {
        for (std::size_t left = 0; left < result.keys.size(); ++left) {
            for (std::size_t right = left + 1U; right < result.keys.size(); ++right) {
                if (key_groups[left] != key_groups[right]) {
                    minimum_negative = std::min(
                        minimum_negative,
                        squared_distance(result.keys[left], result.keys[right]));
                }
            }
        }
    }
    for (const auto& negative : calibration_negatives) {
        ActivationMemorySelection selection;
        float group_margin = 0.0F;
        if (negative.association) {
            selection = nearest_candidate_in_association(
                negative.address_candidates,
                projection,
                config.query_dimension,
                result.keys,
                key_groups,
                *negative.association);
        } else {
            const auto grouped = nearest_candidate_grouped(
                negative.address_candidates,
                projection,
                config.query_dimension,
                result.keys,
                key_groups);
            selection = grouped.first;
            group_margin = grouped.second;
        }
        result.negative_selections.push_back(selection);
        minimum_negative = std::min(minimum_negative, selection.distance);
        maximum_negative_group_margin = std::max(
            maximum_negative_group_margin, group_margin);
    }

    float maximum_validation = 0.0F;
    auto minimum_validation_group_margin = std::numeric_limits<float>::max();
    for (std::size_t index = 0; index < validation_views.size(); ++index) {
        const auto& validation = validation_views[index];
        ActivationMemorySelection selection;
        float group_margin = 0.0F;
        if (config.validation_scope == ActivationValidationScope::association) {
            selection = nearest_candidate_in_association(
                validation.address_candidates,
                projection,
                config.query_dimension,
                result.keys,
                key_groups,
                validation.association);
        } else {
            const auto grouped = nearest_candidate_grouped(
                validation.address_candidates,
                projection,
                config.query_dimension,
                result.keys,
                key_groups);
            selection = grouped.first;
            group_margin = grouped.second;
        }
        if (config.validation_scope == ActivationValidationScope::global &&
            key_groups[selection.key] != validation.association) {
            throw std::runtime_error(
                "validation view " + std::to_string(index) + " for association " +
                std::to_string(validation.association) + " selected key " +
                std::to_string(selection.key) + " from association " +
                std::to_string(key_groups[selection.key]) +
                " at distance " + std::to_string(selection.distance));
        }
        result.validation_selections.push_back(selection);
        maximum_validation = std::max(maximum_validation, selection.distance);
        minimum_validation_group_margin = std::min(
            minimum_validation_group_margin, group_margin);
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
