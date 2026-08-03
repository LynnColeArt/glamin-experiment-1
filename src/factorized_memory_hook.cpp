#include "gx1/factorized_memory_hook.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

namespace gx1 {

namespace {

bool all_finite(const std::vector<float>& values) {
    for (const auto value : values) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

std::size_t projection_size(const FactorSearchConfig& config) {
    const auto hidden = static_cast<std::size_t>(config.hidden_dimension);
    const auto query = static_cast<std::size_t>(config.query_dimension);
    if (hidden == 0U || query == 0U ||
        query > std::numeric_limits<std::size_t>::max() / hidden) {
        throw std::invalid_argument("factor projection dimensions are invalid");
    }
    return hidden * query;
}

void validate_projection_config(const FactorSearchConfig& config) {
    if (config.input_projection.size() != projection_size(config)) {
        throw std::invalid_argument("factor projection has the wrong shape");
    }
    if (!all_finite(config.input_projection) ||
        !std::isfinite(config.maximum_distance) || config.maximum_distance < 0.0F) {
        throw std::invalid_argument("factor search coefficients are invalid");
    }
}

void validate_config(
    const GlaminGenerationPin& pin,
    const FactorSearchConfig& config,
    const std::vector<std::uint64_t>& factor_labels) {
    if (!pin.is_open()) {
        throw std::invalid_argument("factor search requires an open generation pin");
    }
    validate_projection_config(config);
    if (pin.dimension() != config.query_dimension) {
        throw std::invalid_argument("factor projection does not match its Glamin space");
    }
    if (pin.vector_count() != factor_labels.size()) {
        throw std::invalid_argument("factor labels do not match the Glamin rows");
    }
}

} // namespace

void TupleResidualLedger::insert(
    const std::uint64_t entity,
    const std::uint64_t relation,
    std::vector<float> residual) {
    if (residual.empty() || !all_finite(residual)) {
        throw std::invalid_argument("tuple residual values are invalid");
    }
    const auto inserted = residuals_.emplace(
        Tuple{entity, relation},
        std::vector<Variant>{{{}, std::move(residual)}});
    if (!inserted.second) {
        throw std::invalid_argument("tuple residual is already populated");
    }
}

void TupleResidualLedger::insert_variant(
    const std::uint64_t entity,
    const std::uint64_t relation,
    std::vector<float> action_prototype,
    std::vector<float> residual) {
    if (action_prototype.empty() || residual.empty() ||
        !all_finite(action_prototype) || !all_finite(residual)) {
        throw std::invalid_argument("tuple residual variant is invalid");
    }
    auto& variants = residuals_[Tuple{entity, relation}];
    if (!variants.empty() && variants.front().action_prototype.empty()) {
        throw std::invalid_argument(
            "tuple residual cannot mix default and contextual variants");
    }
    variants.push_back(Variant{
        std::move(action_prototype), std::move(residual)});
}

TupleResidualMatch TupleResidualLedger::select(
    const std::uint64_t entity,
    const std::uint64_t relation,
    const std::vector<float>& action_state) const {
    if (action_state.empty() || !all_finite(action_state)) {
        throw std::invalid_argument("tuple action state is invalid");
    }
    const auto found = residuals_.find(Tuple{entity, relation});
    if (found == residuals_.end() || found->second.empty()) {
        return {};
    }
    if (found->second.front().action_prototype.empty()) {
        return TupleResidualMatch{&found->second.front().residual, 0.0F, 0U};
    }

    TupleResidualMatch nearest;
    nearest.distance = std::numeric_limits<float>::max();
    for (std::size_t variant = 0; variant < found->second.size(); ++variant) {
        const auto& candidate = found->second[variant];
        if (candidate.action_prototype.size() != action_state.size()) {
            throw std::runtime_error(
                "tuple action prototype does not match the hidden-state contract");
        }
        double distance = 0.0;
        for (std::size_t index = 0; index < action_state.size(); ++index) {
            const auto difference =
                static_cast<double>(action_state[index]) -
                static_cast<double>(candidate.action_prototype[index]);
            distance += difference * difference;
        }
        const auto narrowed = static_cast<float>(distance);
        if (!std::isfinite(narrowed)) {
            throw std::runtime_error("tuple action distance is non-finite");
        }
        if (narrowed < nearest.distance) {
            nearest = TupleResidualMatch{
                &candidate.residual, narrowed, variant};
        }
    }
    return nearest;
}

std::size_t TupleResidualLedger::size() const noexcept {
    return residuals_.size();
}

FactorizedLayerMemoryHook::FactorizedLayerMemoryHook(
    GlaminGenerationPin entity_pin,
    FactorSearchConfig entity_config,
    std::vector<std::uint64_t> entity_labels,
    GlaminGenerationPin relation_pin,
    FactorSearchConfig relation_config,
    std::vector<std::uint64_t> relation_labels,
    const float gate,
    std::shared_ptr<const TupleResidualLedger> payloads,
    const float maximum_action_distance,
    std::optional<FactorSearchConfig> action_config)
    : entity_pin_(std::move(entity_pin)),
      entity_config_(std::move(entity_config)),
      entity_labels_(std::move(entity_labels)),
      relation_pin_(std::move(relation_pin)),
      relation_config_(std::move(relation_config)),
      relation_labels_(std::move(relation_labels)),
      gate_(gate),
      payloads_(std::move(payloads)),
      maximum_action_distance_(maximum_action_distance),
      action_config_(std::move(action_config)) {
    validate_config(entity_pin_, entity_config_, entity_labels_);
    validate_config(relation_pin_, relation_config_, relation_labels_);
    if (entity_config_.hidden_dimension != relation_config_.hidden_dimension) {
        throw std::invalid_argument("factor spaces use different hidden dimensions");
    }
    if (action_config_) {
        validate_projection_config(*action_config_);
        if (action_config_->hidden_dimension != entity_config_.hidden_dimension) {
            throw std::invalid_argument(
                "action projection uses a different hidden dimension");
        }
    }
    if (!std::isfinite(gate_) || !payloads_ ||
        !std::isfinite(maximum_action_distance_) || maximum_action_distance_ < 0.0F) {
        throw std::invalid_argument("factorized memory action contract is invalid");
    }
}

FactorizedMemoryResult FactorizedLayerMemoryHook::apply_nearest(
    const std::vector<std::vector<float>>& address_states,
    std::vector<float>& hidden_state) const {
    return apply_nearest(address_states, address_states, hidden_state);
}

FactorizedMemoryResult FactorizedLayerMemoryHook::apply_nearest(
    const std::vector<std::vector<float>>& entity_states,
    const std::vector<std::vector<float>>& relation_states,
    std::vector<float>& hidden_state) const {
    if (hidden_state.size() != hidden_dimension() || !all_finite(hidden_state)) {
        throw std::invalid_argument("factorized action state has the wrong shape or values");
    }

    FactorizedMemoryResult result;
    result.entity = nearest(
        entity_pin_, entity_config_, entity_labels_, entity_states);
    result.relation = nearest(
        relation_pin_, relation_config_, relation_labels_, relation_states);
    if (!result.entity.accepted || !result.relation.accepted) {
        return result;
    }

    const auto action_query = action_config_
                                  ? project(*action_config_, hidden_state)
                                  : hidden_state;
    const auto action = payloads_->select(
        result.entity.factor_label,
        result.relation.factor_label,
        action_query);
    if (action.residual == nullptr) {
        return result;
    }
    result.tuple_found = true;
    result.action_distance = action.distance;
    result.action_variant = action.variant;
    if (action.distance > maximum_action_distance_) {
        return result;
    }
    result.action_accepted = true;
    if (action.residual->size() != hidden_state.size() ||
        !all_finite(*action.residual)) {
        throw std::runtime_error("tuple residual does not match the hidden-state contract");
    }

    auto updated = hidden_state;
    for (std::size_t index = 0; index < updated.size(); ++index) {
        updated[index] += gate_ * (*action.residual)[index];
    }
    if (!all_finite(updated)) {
        throw std::runtime_error("factorized residual injection produced non-finite values");
    }
    hidden_state = std::move(updated);
    result.gate = gate_;
    result.applied = true;
    return result;
}

std::uint32_t FactorizedLayerMemoryHook::hidden_dimension() const noexcept {
    return entity_config_.hidden_dimension;
}

GlaminGenerationId FactorizedLayerMemoryHook::entity_generation() const {
    return entity_pin_.id();
}

GlaminGenerationId FactorizedLayerMemoryHook::relation_generation() const {
    return relation_pin_.id();
}

FactorEvidence FactorizedLayerMemoryHook::nearest(
    const GlaminGenerationPin& pin,
    const FactorSearchConfig& config,
    const std::vector<std::uint64_t>& factor_labels,
    const std::vector<std::vector<float>>& states) const {
    if (states.empty()) {
        throw std::invalid_argument("factor search requires candidate states");
    }

    FactorEvidence nearest;
    nearest.distance = std::numeric_limits<float>::max();
    for (std::size_t candidate = 0; candidate < states.size(); ++candidate) {
        const auto query = project(config, states[candidate]);
        const auto search = pin.search(query, 1U);
        if (search.labels.size() != 1U || search.distances.size() != 1U) {
            throw std::runtime_error("Glamin returned an invalid factor result");
        }
        const auto distance = search.distances.front();
        if (!std::isfinite(distance)) {
            throw std::runtime_error("Glamin returned a non-finite factor distance");
        }
        if (candidate == 0U || distance < nearest.distance) {
            nearest.generation = pin.id();
            nearest.memory_label = search.labels.front();
            nearest.distance = distance;
            nearest.address_candidate = candidate;
        }
    }
    if (nearest.memory_label >= factor_labels.size()) {
        throw std::runtime_error("Glamin factor label is outside the reviewed ledger");
    }
    nearest.factor_label = factor_labels[static_cast<std::size_t>(nearest.memory_label)];
    nearest.accepted = nearest.distance <= config.maximum_distance;
    return nearest;
}

std::vector<float> FactorizedLayerMemoryHook::project(
    const FactorSearchConfig& config,
    const std::vector<float>& hidden_state) const {
    if (hidden_state.size() != config.hidden_dimension || !all_finite(hidden_state)) {
        throw std::invalid_argument("factor address state has the wrong shape or values");
    }
    const auto hidden_dimension = static_cast<std::size_t>(config.hidden_dimension);
    const auto query_dimension = static_cast<std::size_t>(config.query_dimension);
    std::vector<float> query(query_dimension, 0.0F);
    for (std::size_t row = 0; row < query_dimension; ++row) {
        double sum = 0.0;
        const auto offset = row * hidden_dimension;
        for (std::size_t column = 0; column < hidden_dimension; ++column) {
            sum += static_cast<double>(config.input_projection[offset + column]) *
                   static_cast<double>(hidden_state[column]);
        }
        query[row] = static_cast<float>(sum);
    }
    if (!all_finite(query)) {
        throw std::runtime_error("factor projection produced non-finite values");
    }
    if (config.query_normalization == ProjectionNormalization::l2) {
        double squared_norm = 0.0;
        for (const auto value : query) {
            squared_norm += static_cast<double>(value) * value;
        }
        if (squared_norm > 0.0) {
            const auto inverse_norm = 1.0 / std::sqrt(squared_norm);
            for (auto& value : query) {
                value = static_cast<float>(static_cast<double>(value) * inverse_norm);
            }
        }
    }
    return query;
}

} // namespace gx1
