#include "gx1/llama_glamin_hook.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"

namespace gx1 {

namespace {

std::size_t last_token_index(const ggml_tensor* tensor) {
    if (tensor->ne[1] <= 0) {
        throw std::runtime_error("target layer tensor has no token rows");
    }
    return static_cast<std::size_t>(tensor->ne[1] - 1);
}

std::vector<float> read_hidden_state(
    ggml_tensor* tensor,
    const std::uint32_t hidden_dimension,
    const std::size_t token_index) {
    if (tensor->buffer == nullptr || tensor->type != GGML_TYPE_F32 ||
        tensor->ne[0] != static_cast<std::int64_t>(hidden_dimension) ||
        tensor->ne[1] <= 0 || tensor->nb[0] != sizeof(float)) {
        throw std::runtime_error(
            "target layer tensor does not satisfy the contiguous float32 hidden-state contract");
    }

    const auto hidden_bytes = static_cast<std::size_t>(hidden_dimension) * sizeof(float);
    if (tensor->nb[1] < hidden_bytes) {
        throw std::runtime_error("target layer tensor has an invalid row stride");
    }
    const auto token_count = static_cast<std::size_t>(tensor->ne[1]);
    if (token_index >= token_count) {
        throw std::out_of_range(
            "hidden-state token index " + std::to_string(token_index) +
            " is outside target tensor " + tensor->name + " with " +
            std::to_string(token_count) + " rows");
    }
    if (token_index > std::numeric_limits<std::size_t>::max() / tensor->nb[1]) {
        throw std::overflow_error("target layer tensor offset overflowed");
    }
    const auto offset = token_index * tensor->nb[1];
    if (offset > ggml_nbytes(tensor) || hidden_bytes > ggml_nbytes(tensor) - offset) {
        throw std::runtime_error("target layer tensor storage is smaller than its shape");
    }

    std::vector<float> hidden_state(hidden_dimension);
    ggml_backend_tensor_get(tensor, hidden_state.data(), offset, hidden_bytes);
    return hidden_state;
}

std::vector<std::vector<float>> read_all_hidden_states(
    ggml_tensor* tensor,
    const std::uint32_t hidden_dimension) {
    const auto token_count = last_token_index(tensor) + 1U;
    std::vector<std::vector<float>> states;
    states.reserve(token_count);
    for (std::size_t index = 0; index < token_count; ++index) {
        states.push_back(read_hidden_state(tensor, hidden_dimension, index));
    }
    return states;
}

void write_last_hidden_state(
    ggml_tensor* tensor,
    const std::vector<float>& hidden_state) {
    const auto hidden_bytes = hidden_state.size() * sizeof(float);
    const auto last_token = last_token_index(tensor);
    const auto offset = last_token * tensor->nb[1];
    ggml_backend_tensor_set(tensor, hidden_state.data(), offset, hidden_bytes);
}

} // namespace

LlamaGlaminHook::LlamaGlaminHook(
    FixedLayerMemoryHook hook,
    std::string target_tensor,
    const std::optional<std::size_t> address_token_index)
    : hook_(std::move(hook)),
      target_tensor_(std::move(target_tensor)),
      address_token_index_(address_token_index),
      address_selection_(hook_.address_selection()) {
    if (target_tensor_.empty() || target_tensor_.find('\0') != std::string::npos) {
        throw std::invalid_argument("llama.cpp hook target tensor is invalid");
    }
    if (address_token_index_ &&
        address_selection_ != AddressSelectionPolicy::last_token) {
        throw std::invalid_argument(
            "llama.cpp hook cannot combine an explicit address with token scanning");
    }
}

bool LlamaGlaminHook::evaluate(
    ggml_tensor* tensor,
    const bool ask,
    void* user_data) noexcept {
    if (user_data == nullptr) {
        return false;
    }
    auto* hook = static_cast<LlamaGlaminHook*>(user_data);
    try {
        return hook->evaluate_tensor(tensor, ask);
    } catch (const std::exception& error) {
        hook->error_ = error.what();
        return false;
    } catch (...) {
        hook->error_ = "unknown failure in llama.cpp hidden-state callback";
        return false;
    }
}

void LlamaGlaminHook::throw_if_failed() const {
    if (!error_.empty()) {
        throw std::runtime_error(error_);
    }
}

bool LlamaGlaminHook::failed() const noexcept {
    return !error_.empty();
}

const std::string& LlamaGlaminHook::error() const noexcept {
    return error_;
}

std::size_t LlamaGlaminHook::invocation_count() const noexcept {
    return invocation_count_;
}

const std::optional<HiddenStateHookResult>& LlamaGlaminHook::last_result() const noexcept {
    return last_result_;
}

const std::string& LlamaGlaminHook::target_tensor() const noexcept {
    return target_tensor_;
}

bool LlamaGlaminHook::evaluate_tensor(ggml_tensor* tensor, const bool ask) {
    if (tensor == nullptr) {
        throw std::invalid_argument("llama.cpp callback supplied a null tensor");
    }
    const bool matches = std::strcmp(tensor->name, target_tensor_.c_str()) == 0;
    if (ask) {
        return matches;
    }
    if (!matches) {
        return true;
    }
    const auto last_token = last_token_index(tensor);
    auto hidden_state = read_hidden_state(tensor, hook_.hidden_dimension(), last_token);
    if (address_selection_ != AddressSelectionPolicy::last_token) {
        auto address_states = read_all_hidden_states(
            tensor, hook_.hidden_dimension());
        if (address_selection_ == AddressSelectionPolicy::prefix_mean_rows) {
            address_states = prefix_mean_states(address_states);
        }
        last_result_ = hook_.apply_nearest(address_states, hidden_state);
    } else {
        const auto address_token = address_token_index_.value_or(last_token);
        const auto address_state = address_token == last_token
                                       ? hidden_state
                                       : read_hidden_state(
                                             tensor,
                                             hook_.hidden_dimension(),
                                             address_token);
        last_result_ = hook_.apply(address_state, hidden_state);
    }
    write_last_hidden_state(tensor, hidden_state);
    ++invocation_count_;
    return true;
}

LlamaFactorizedGlaminHook::LlamaFactorizedGlaminHook(
    FactorizedLayerMemoryHook hook,
    std::string target_tensor)
    : hook_(std::move(hook)), target_tensor_(std::move(target_tensor)) {
    if (target_tensor_.empty() || target_tensor_.find('\0') != std::string::npos) {
        throw std::invalid_argument("llama.cpp factorized hook target tensor is invalid");
    }
}

bool LlamaFactorizedGlaminHook::evaluate(
    ggml_tensor* tensor,
    const bool ask,
    void* user_data) noexcept {
    if (user_data == nullptr) {
        return false;
    }
    auto* hook = static_cast<LlamaFactorizedGlaminHook*>(user_data);
    try {
        return hook->evaluate_tensor(tensor, ask);
    } catch (const std::exception& error) {
        hook->error_ = error.what();
        return false;
    } catch (...) {
        hook->error_ = "unknown failure in llama.cpp factorized memory callback";
        return false;
    }
}

void LlamaFactorizedGlaminHook::throw_if_failed() const {
    if (!error_.empty()) {
        throw std::runtime_error(error_);
    }
}

bool LlamaFactorizedGlaminHook::failed() const noexcept {
    return !error_.empty();
}

const std::string& LlamaFactorizedGlaminHook::error() const noexcept {
    return error_;
}

std::size_t LlamaFactorizedGlaminHook::invocation_count() const noexcept {
    return invocation_count_;
}

const std::optional<FactorizedMemoryResult>&
LlamaFactorizedGlaminHook::last_result() const noexcept {
    return last_result_;
}

bool LlamaFactorizedGlaminHook::evaluate_tensor(
    ggml_tensor* tensor,
    const bool ask) {
    if (tensor == nullptr) {
        throw std::invalid_argument("llama.cpp callback supplied a null tensor");
    }
    const bool matches = std::strcmp(tensor->name, target_tensor_.c_str()) == 0;
    if (ask) {
        return matches;
    }
    if (!matches) {
        return true;
    }

    const auto last_token = last_token_index(tensor);
    auto hidden_state = read_hidden_state(tensor, hook_.hidden_dimension(), last_token);
    last_result_ = hook_.apply_nearest(
        read_all_hidden_states(tensor, hook_.hidden_dimension()), hidden_state);
    write_last_hidden_state(tensor, hidden_state);
    ++invocation_count_;
    return true;
}

LlamaHiddenStateCapture::LlamaHiddenStateCapture(
    const std::uint32_t hidden_dimension,
    std::string target_tensor,
    const std::optional<std::size_t> address_token_index,
    const bool capture_all_token_rows)
    : hidden_dimension_(hidden_dimension),
      target_tensor_(std::move(target_tensor)),
      address_token_index_(address_token_index),
      capture_all_token_rows_(capture_all_token_rows) {
    if (hidden_dimension_ == 0 || target_tensor_.empty() ||
        target_tensor_.find('\0') != std::string::npos) {
        throw std::invalid_argument("llama.cpp capture contract is invalid");
    }
}

bool LlamaHiddenStateCapture::evaluate(
    ggml_tensor* tensor,
    const bool ask,
    void* user_data) noexcept {
    if (user_data == nullptr) {
        return false;
    }
    auto* capture = static_cast<LlamaHiddenStateCapture*>(user_data);
    try {
        return capture->evaluate_tensor(tensor, ask);
    } catch (const std::exception& error) {
        capture->error_ = error.what();
        return false;
    } catch (...) {
        capture->error_ = "unknown failure in llama.cpp hidden-state capture";
        return false;
    }
}

void LlamaHiddenStateCapture::throw_if_failed() const {
    if (!error_.empty()) {
        throw std::runtime_error(error_);
    }
}

const std::vector<float>& LlamaHiddenStateCapture::hidden_state() const {
    if (hidden_state_.empty()) {
        throw std::logic_error("llama.cpp hidden state has not been captured");
    }
    return hidden_state_;
}

const std::vector<float>& LlamaHiddenStateCapture::address_state() const {
    if (address_state_.empty()) {
        throw std::logic_error("llama.cpp address state has not been captured");
    }
    return address_state_;
}

const std::vector<std::vector<float>>& LlamaHiddenStateCapture::token_states() const {
    if (token_states_.empty()) {
        throw std::logic_error("llama.cpp token states have not been captured");
    }
    return token_states_;
}

std::size_t LlamaHiddenStateCapture::invocation_count() const noexcept {
    return invocation_count_;
}

bool LlamaHiddenStateCapture::evaluate_tensor(ggml_tensor* tensor, const bool ask) {
    if (tensor == nullptr) {
        throw std::invalid_argument("llama.cpp callback supplied a null tensor");
    }
    const bool matches = std::strcmp(tensor->name, target_tensor_.c_str()) == 0;
    if (ask) {
        return matches;
    }
    if (!matches) {
        return true;
    }
    const auto last_token = last_token_index(tensor);
    const auto address_token = address_token_index_.value_or(last_token);
    hidden_state_ = read_hidden_state(tensor, hidden_dimension_, last_token);
    address_state_ = address_token == last_token
                         ? hidden_state_
                         : read_hidden_state(
                               tensor, hidden_dimension_, address_token);
    if (capture_all_token_rows_) {
        token_states_ = read_all_hidden_states(tensor, hidden_dimension_);
    }
    ++invocation_count_;
    return true;
}

} // namespace gx1
