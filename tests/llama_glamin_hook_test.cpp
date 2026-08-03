#include "gx1/hidden_state_hook.hpp"
#include "gx1/llama_glamin_hook.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

namespace {

void expect(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_callback_mutates_only_the_last_token() {
    gx1::GlaminRuntime runtime(2);
    gx1::GlaminGenerationStore generations(runtime);
    const auto generation = generations.mount_flat(
        "tensor-hook", 2, {1.0F, 2.0F, 100.0F, 100.0F});
    generations.activate(generation);

    auto payloads = std::make_shared<gx1::ResidualPayloadLedger>();
    payloads->insert(generation, 0, {2.0F, 0.0F, -2.0F, 0.0F});
    gx1::HiddenStateHookConfig config{
        4,
        2,
        {1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F},
        0.5F,
        gx1::ProjectionNormalization::none,
    };
    config.address_selection = gx1::AddressSelectionPolicy::all_token_rows;
    gx1::FixedLayerMemoryHook memory_hook(
        generations.pin_active(),
        std::move(config),
        payloads);
    gx1::LlamaGlaminHook hook(std::move(memory_hook), "l_out-1");

    auto* backend = ggml_backend_cpu_init();
    if (backend == nullptr) {
        throw std::runtime_error("failed to create the ggml CPU backend");
    }
    ggml_init_params parameters{};
    parameters.mem_size = 1024U * 1024U;
    parameters.mem_buffer = nullptr;
    parameters.no_alloc = true;
    auto* context = ggml_init(parameters);
    if (context == nullptr) {
        ggml_backend_free(backend);
        throw std::runtime_error("failed to create the ggml tensor context");
    }

    auto* tensor = ggml_new_tensor_2d(context, GGML_TYPE_F32, 4, 2);
    ggml_set_name(tensor, "l_out-1");
    auto* buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    if (buffer == nullptr) {
        ggml_free(context);
        ggml_backend_free(backend);
        throw std::runtime_error("failed to allocate the ggml tensor buffer");
    }

    const std::vector<float> input{1.0F, 2.0F, 3.0F, 4.0F, 10.0F, 20.0F, 30.0F, 40.0F};
    ggml_backend_tensor_set(tensor, input.data(), 0, input.size() * sizeof(float));
    expect(
        gx1::LlamaGlaminHook::evaluate(tensor, true, &hook),
        "callback did not request its configured tensor");
    expect(
        gx1::LlamaGlaminHook::evaluate(tensor, false, &hook),
        "callback rejected a valid hidden-state tensor");
    hook.throw_if_failed();

    std::vector<float> output(input.size());
    ggml_backend_tensor_get(tensor, output.data(), 0, output.size() * sizeof(float));
    const std::vector<float> expected{
        1.0F, 2.0F, 3.0F, 4.0F, 11.0F, 20.0F, 29.0F, 40.0F};
    expect(output == expected, "callback mutated the wrong hidden-state values");
    expect(hook.invocation_count() == 1, "callback invocation count mismatch");
    expect(hook.last_result().has_value(), "callback did not retain its Glamin result");
    expect(
        hook.last_result()->generation == generation,
        "callback recorded the wrong generation");
    expect(
        hook.last_result()->address_candidate == 0U,
        "callback selected the wrong automatic address row");

    ggml_backend_tensor_set(tensor, input.data(), 0, input.size() * sizeof(float));
    gx1::LlamaHiddenStateCapture capture(4, "l_out-1", 0U, true);
    expect(
        gx1::LlamaHiddenStateCapture::evaluate(tensor, true, &capture),
        "capture callback did not request its configured tensor");
    expect(
        gx1::LlamaHiddenStateCapture::evaluate(tensor, false, &capture),
        "capture callback rejected a valid hidden-state tensor");
    capture.throw_if_failed();
    expect(capture.hidden_state() == std::vector<float>({10.0F, 20.0F, 30.0F, 40.0F}),
           "capture callback read the wrong hidden-state row");
    expect(capture.address_state() == std::vector<float>({1.0F, 2.0F, 3.0F, 4.0F}),
           "capture callback read the wrong address-state row");
    expect(capture.token_states().size() == 2U,
           "capture callback did not retain every token row");
    expect(capture.invocation_count() == 1, "capture callback invocation count mismatch");

    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    ggml_backend_free(backend);
}

} // namespace

int main() {
    try {
        test_callback_mutates_only_the_last_token();
        std::cout << "llama.cpp Glamin tensor callback tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "llama.cpp Glamin tensor callback test failed: "
                  << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
