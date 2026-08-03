#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "gx1/factorized_memory_hook.hpp"
#include "gx1/hidden_state_hook.hpp"

struct ggml_tensor;

namespace gx1 {

class LlamaGlaminHook final {
public:
    LlamaGlaminHook(
        FixedLayerMemoryHook hook,
        std::string target_tensor,
        std::optional<std::size_t> address_token_index = std::nullopt);

    LlamaGlaminHook(const LlamaGlaminHook&) = delete;
    LlamaGlaminHook& operator=(const LlamaGlaminHook&) = delete;
    LlamaGlaminHook(LlamaGlaminHook&&) = delete;
    LlamaGlaminHook& operator=(LlamaGlaminHook&&) = delete;

    [[nodiscard]] static bool evaluate(
        ggml_tensor* tensor,
        bool ask,
        void* user_data) noexcept;

    void throw_if_failed() const;
    [[nodiscard]] bool failed() const noexcept;
    [[nodiscard]] const std::string& error() const noexcept;
    [[nodiscard]] std::size_t invocation_count() const noexcept;
    [[nodiscard]] const std::optional<HiddenStateHookResult>& last_result() const noexcept;
    [[nodiscard]] const std::string& target_tensor() const noexcept;

private:
    [[nodiscard]] bool evaluate_tensor(ggml_tensor* tensor, bool ask);

    FixedLayerMemoryHook hook_;
    std::string target_tensor_;
    std::string error_;
    std::optional<HiddenStateHookResult> last_result_;
    std::optional<std::size_t> address_token_index_;
    AddressSelectionPolicy address_selection_{AddressSelectionPolicy::last_token};
    std::size_t invocation_count_{0};
};

class LlamaFactorizedGlaminHook final {
public:
    LlamaFactorizedGlaminHook(
        FactorizedLayerMemoryHook hook,
        std::string target_tensor);

    LlamaFactorizedGlaminHook(const LlamaFactorizedGlaminHook&) = delete;
    LlamaFactorizedGlaminHook& operator=(const LlamaFactorizedGlaminHook&) = delete;
    LlamaFactorizedGlaminHook(LlamaFactorizedGlaminHook&&) = delete;
    LlamaFactorizedGlaminHook& operator=(LlamaFactorizedGlaminHook&&) = delete;

    [[nodiscard]] static bool evaluate(
        ggml_tensor* tensor,
        bool ask,
        void* user_data) noexcept;

    void throw_if_failed() const;
    [[nodiscard]] bool failed() const noexcept;
    [[nodiscard]] const std::string& error() const noexcept;
    [[nodiscard]] std::size_t invocation_count() const noexcept;
    [[nodiscard]] const std::optional<FactorizedMemoryResult>& last_result() const noexcept;

private:
    [[nodiscard]] bool evaluate_tensor(ggml_tensor* tensor, bool ask);

    FactorizedLayerMemoryHook hook_;
    std::string target_tensor_;
    std::string error_;
    std::optional<FactorizedMemoryResult> last_result_;
    std::size_t invocation_count_{0};
};

class LlamaHiddenStateCapture final {
public:
    LlamaHiddenStateCapture(
        std::uint32_t hidden_dimension,
        std::string target_tensor,
        std::optional<std::size_t> address_token_index = std::nullopt,
        bool capture_all_token_rows = false);

    LlamaHiddenStateCapture(const LlamaHiddenStateCapture&) = delete;
    LlamaHiddenStateCapture& operator=(const LlamaHiddenStateCapture&) = delete;
    LlamaHiddenStateCapture(LlamaHiddenStateCapture&&) = delete;
    LlamaHiddenStateCapture& operator=(LlamaHiddenStateCapture&&) = delete;

    [[nodiscard]] static bool evaluate(
        ggml_tensor* tensor,
        bool ask,
        void* user_data) noexcept;

    void throw_if_failed() const;
    [[nodiscard]] const std::vector<float>& hidden_state() const;
    [[nodiscard]] const std::vector<float>& address_state() const;
    [[nodiscard]] const std::vector<std::vector<float>>& token_states() const;
    [[nodiscard]] std::size_t invocation_count() const noexcept;

private:
    [[nodiscard]] bool evaluate_tensor(ggml_tensor* tensor, bool ask);

    std::uint32_t hidden_dimension_{0};
    std::string target_tensor_;
    std::string error_;
    std::vector<float> hidden_state_;
    std::vector<float> address_state_;
    std::vector<std::vector<float>> token_states_;
    std::optional<std::size_t> address_token_index_;
    bool capture_all_token_rows_{false};
    std::size_t invocation_count_{0};
};

} // namespace gx1
