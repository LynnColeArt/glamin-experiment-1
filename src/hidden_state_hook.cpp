#include "gx1/hidden_state_hook.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
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

std::size_t projection_size(
    const std::uint32_t hidden_dimension,
    const std::uint32_t query_dimension) {
    const auto hidden = static_cast<std::size_t>(hidden_dimension);
    const auto query = static_cast<std::size_t>(query_dimension);
    if (hidden == 0 || query == 0 || query > std::numeric_limits<std::size_t>::max() / hidden) {
        throw std::invalid_argument("hidden-state projection dimensions are invalid");
    }
    return hidden * query;
}

} // namespace

std::vector<std::vector<float>> prefix_mean_states(
    const std::vector<std::vector<float>>& states) {
    if (states.empty() || states.front().empty()) {
        throw std::invalid_argument(
            "prefix-mean addressing requires nonempty hidden states");
    }
    const auto hidden_dimension = states.front().size();
    std::vector<double> running_sum(hidden_dimension, 0.0);
    std::vector<std::vector<float>> means;
    means.reserve(states.size());
    for (std::size_t row = 0; row < states.size(); ++row) {
        if (states[row].size() != hidden_dimension || !all_finite(states[row])) {
            throw std::invalid_argument(
                "prefix-mean addressing states have inconsistent shapes or values");
        }
        std::vector<float> mean(hidden_dimension, 0.0F);
        for (std::size_t column = 0; column < hidden_dimension; ++column) {
            running_sum[column] += static_cast<double>(states[row][column]);
            mean[column] = static_cast<float>(
                running_sum[column] / static_cast<double>(row + 1U));
        }
        if (!all_finite(mean)) {
            throw std::runtime_error(
                "prefix-mean addressing produced non-finite values");
        }
        means.push_back(std::move(mean));
    }
    return means;
}

void ResidualPayloadLedger::insert(
    const GlaminGenerationId generation,
    const std::uint64_t label,
    std::vector<float> residual) {
    if (generation == 0 || residual.empty() || !all_finite(residual)) {
        throw std::invalid_argument("residual payload address or values are invalid");
    }
    const auto inserted = residuals_.emplace(
        Address{generation, label},
        std::move(residual));
    if (!inserted.second) {
        throw std::invalid_argument("residual payload address is already populated");
    }
}

const std::vector<float>& ResidualPayloadLedger::at(
    const GlaminGenerationId generation,
    const std::uint64_t label) const {
    const auto found = residuals_.find(Address{generation, label});
    if (found == residuals_.end()) {
        throw std::out_of_range("Glamin result has no residual payload");
    }
    return found->second;
}

std::size_t ResidualPayloadLedger::size() const noexcept {
    return residuals_.size();
}

FixedLayerMemoryHook::FixedLayerMemoryHook(
    GlaminGenerationPin pin,
    HiddenStateHookConfig config,
    std::shared_ptr<const ResidualPayloadLedger> payloads)
    : pin_(std::move(pin)), config_(std::move(config)), payloads_(std::move(payloads)) {
    if (!pin_.is_open()) {
        throw std::invalid_argument("hidden-state hook requires an open generation pin");
    }
    if (pin_.dimension() != config_.query_dimension) {
        throw std::invalid_argument(
            "hidden-state projection dimension does not match the Glamin space");
    }
    if (config_.input_projection.size() !=
        projection_size(config_.hidden_dimension, config_.query_dimension)) {
        throw std::invalid_argument("hidden-state projection matrix has the wrong size");
    }
    if (!all_finite(config_.input_projection) || !std::isfinite(config_.gate) ||
        !std::isfinite(config_.maximum_distance) || config_.maximum_distance < 0.0F) {
        throw std::invalid_argument("hidden-state hook coefficients must be finite");
    }
    if (!payloads_) {
        throw std::invalid_argument("hidden-state hook requires a payload ledger");
    }
}

HiddenStateHookResult FixedLayerMemoryHook::apply(std::vector<float>& hidden_state) {
    return apply(hidden_state, hidden_state);
}

HiddenStateHookResult FixedLayerMemoryHook::apply(
    const std::vector<float>& address_state,
    std::vector<float>& hidden_state) {
    if (hidden_state.size() != static_cast<std::size_t>(config_.hidden_dimension) ||
        !all_finite(hidden_state)) {
        throw std::invalid_argument("hidden-state vector has the wrong shape or values");
    }
    return inject(search(address_state), hidden_state);
}

HiddenStateHookResult FixedLayerMemoryHook::apply_nearest(
    const std::vector<std::vector<float>>& address_states,
    std::vector<float>& hidden_state) {
    if (address_states.empty()) {
        throw std::invalid_argument("memory addressing requires at least one candidate state");
    }
    if (hidden_state.size() != static_cast<std::size_t>(config_.hidden_dimension) ||
        !all_finite(hidden_state)) {
        throw std::invalid_argument("hidden-state vector has the wrong shape or values");
    }

    auto nearest = search(address_states.front());
    for (std::size_t index = 1; index < address_states.size(); ++index) {
        auto candidate = search(address_states[index]);
        if (candidate.distance < nearest.distance) {
            nearest = candidate;
            nearest.address_candidate = index;
        }
    }
    return inject(nearest, hidden_state);
}

HiddenStateHookResult FixedLayerMemoryHook::search(
    const std::vector<float>& address_state) const {
    if (address_state.size() != static_cast<std::size_t>(config_.hidden_dimension) ||
        !all_finite(address_state)) {
        throw std::invalid_argument("memory address state has the wrong shape or values");
    }

    const auto query = project(address_state);
    const auto search = pin_.search(query, 1);
    if (search.labels.size() != 1 || search.distances.size() != 1) {
        throw std::runtime_error("Glamin returned an invalid single-neighbor result");
    }

    const auto label = search.labels.front();
    const auto distance = search.distances.front();
    if (!std::isfinite(distance)) {
        throw std::runtime_error("Glamin returned a non-finite memory distance");
    }
    return HiddenStateHookResult{
        pin_.id(),
        label,
        distance,
        0.0F,
        false,
        0U,
    };
}

HiddenStateHookResult FixedLayerMemoryHook::inject(
    HiddenStateHookResult result,
    std::vector<float>& hidden_state) const {
    if (result.distance > config_.maximum_distance) {
        return result;
    }
    const auto& residual = payloads_->at(pin_.id(), result.label);
    if (residual.size() != hidden_state.size() || !all_finite(residual)) {
        throw std::runtime_error("residual payload does not match the hidden-state contract");
    }

    auto updated = hidden_state;
    for (std::size_t index = 0; index < updated.size(); ++index) {
        updated[index] += config_.gate * residual[index];
    }
    if (!all_finite(updated)) {
        throw std::runtime_error("residual injection produced a non-finite hidden state");
    }
    hidden_state = std::move(updated);

    result.gate = config_.gate;
    result.applied = true;
    return result;
}

std::uint32_t FixedLayerMemoryHook::hidden_dimension() const noexcept {
    return config_.hidden_dimension;
}

AddressSelectionPolicy FixedLayerMemoryHook::address_selection() const noexcept {
    return config_.address_selection;
}

GlaminGenerationId FixedLayerMemoryHook::generation() const {
    return pin_.id();
}

std::vector<float> FixedLayerMemoryHook::project(
    const std::vector<float>& hidden_state) const {
    const auto hidden_dimension = static_cast<std::size_t>(config_.hidden_dimension);
    const auto query_dimension = static_cast<std::size_t>(config_.query_dimension);
    std::vector<float> query(query_dimension, 0.0F);

    for (std::size_t row = 0; row < query_dimension; ++row) {
        double sum = 0.0;
        const auto row_offset = row * hidden_dimension;
        for (std::size_t column = 0; column < hidden_dimension; ++column) {
            sum += static_cast<double>(config_.input_projection[row_offset + column]) *
                   static_cast<double>(hidden_state[column]);
        }
        query[row] = static_cast<float>(sum);
    }

    if (!all_finite(query)) {
        throw std::runtime_error("hidden-state projection produced non-finite values");
    }
    if (config_.query_normalization == ProjectionNormalization::l2) {
        double squared_norm = 0.0;
        for (const auto value : query) {
            squared_norm += static_cast<double>(value) * static_cast<double>(value);
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
