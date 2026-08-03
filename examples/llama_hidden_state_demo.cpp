#include "gx1/hidden_state_hook.hpp"
#include "gx1/hook_artifact.hpp"
#include "gx1/llama_glamin_hook.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "llama.h"

namespace {

using ModelPointer = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using ContextPointer = std::unique_ptr<llama_context, decltype(&llama_free)>;

struct TrialResult {
    std::vector<float> logits;
    gx1::HiddenStateHookResult hook;
    std::string contract_sha256;
    llama_token top_token{LLAMA_TOKEN_NULL};
    std::size_t hook_invocations{0};
};

constexpr const char* llama_revision =
    "ecd99d6a9acbc436bad085783bcd5d0b9ae9e9e9";

void write_glamin_artifact(
    const std::filesystem::path& directory,
    const std::uint32_t query_dimension) {
    const auto glamin_directory = directory / "glamin";
    if (!std::filesystem::create_directories(glamin_directory)) {
        throw std::runtime_error("failed to create the demo Glamin artifact directory");
    }

    const auto vector_bytes = static_cast<std::uint64_t>(query_dimension) * sizeof(float);
    std::ofstream layout(glamin_directory / "vector_layout.json");
    layout << "{\"dtype\": \"float32\",\"endianness\": \"little\","
              "\"total_vectors\": 1,\"total_bytes\": "
           << vector_bytes
           << ",\"spaces\":[{\"space_id\": \"hook.memory\",\"dim\": "
           << query_dimension
           << ",\"count\": 1,\"byte_stride\": " << vector_bytes
           << ",\"offset_bytes\": 0}]}\n";
    if (!layout) {
        throw std::runtime_error("failed to write the demo Glamin vector layout");
    }
    layout.close();

    const std::vector<float> memory_key(query_dimension, 0.0F);
    std::ofstream vectors(glamin_directory / "vectors.bin", std::ios::binary);
    vectors.write(
        reinterpret_cast<const char*>(memory_key.data()),
        static_cast<std::streamsize>(memory_key.size() * sizeof(float)));
    if (!vectors) {
        throw std::runtime_error("failed to write the demo Glamin vectors");
    }
    vectors.close();

    const std::string space_spec =
        "{\"space_id\": \"hook.memory\",\"dim\": " +
        std::to_string(query_dimension) +
        ",\"metric\": \"l2\",\"normalization\": \"l2\"}";
    std::ofstream contracts(glamin_directory / "contracts.json");
    contracts
        << "{\"spec_id\": \"gx1-model-hook-demo\",\"embedder\": {\"spec\": {"
           "\"id\": \"gx1-hidden-projection\",\"version\": \"1\","
           "\"input_schema\": \"f32\",\"preprocess_chain\": [\"l2\"],"
           "\"model_hash\": \"sha256:projection\","
           "\"config_hash\": \"sha256:demo\","
           "\"hardware_class\": \"cpu\",\"min_ram_mb\": 0,"
           "\"min_vram_mb\": 0},\"contract_hash\": \"sha256:demo\","
           "\"signature\": \"\"},\"spaces\":[{\"spec\": "
        << space_spec << ",\"contract_hash\": \""
        << gx1::sha256_text(space_spec) << "\",\"signature\": \"\"}]}\n";
    if (!contracts) {
        throw std::runtime_error("failed to write the demo Glamin contracts");
    }
}

class DemoHookArtifacts final {
public:
    DemoHookArtifacts(
        const gx1::ModelHookContract& model_contract,
        const std::uint32_t query_dimension,
        const std::vector<float>& projection,
        const std::vector<float>& memory_residual) {
        const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
        root_ = std::filesystem::temp_directory_path() /
                ("gx1-model-hook-" + std::to_string(nonce));
        baseline_ = root_ / "baseline";
        memory_ = root_ / "memory";

        try {
            write_bundle(
                baseline_,
                model_contract,
                query_dimension,
                projection,
                std::vector<float>(model_contract.hidden_dimension, 0.0F));
            write_bundle(
                memory_, model_contract, query_dimension, projection, memory_residual);
        } catch (...) {
            std::error_code error;
            std::filesystem::remove_all(root_, error);
            throw;
        }
    }

    ~DemoHookArtifacts() {
        std::error_code error;
        std::filesystem::remove_all(root_, error);
    }

    DemoHookArtifacts(const DemoHookArtifacts&) = delete;
    DemoHookArtifacts& operator=(const DemoHookArtifacts&) = delete;

    [[nodiscard]] const std::filesystem::path& baseline() const noexcept {
        return baseline_;
    }

    [[nodiscard]] const std::filesystem::path& memory() const noexcept {
        return memory_;
    }

private:
    static void write_bundle(
        const std::filesystem::path& directory,
        const gx1::ModelHookContract& model_contract,
        const std::uint32_t query_dimension,
        const std::vector<float>& projection,
        const std::vector<float>& residual) {
        write_glamin_artifact(directory, query_dimension);
        gx1::HookArtifactSpec spec;
        spec.model = model_contract;
        spec.query_dimension = query_dimension;
        spec.metric = gx1::GlaminMetric::l2;
        spec.query_normalization = gx1::ProjectionNormalization::l2;
        spec.gate = 1.0F;
        spec.input_projection = projection;
        spec.residual_labels = {0};
        spec.residuals = residual;
        spec.glamin_space_id = "hook.memory";
        gx1::write_hook_artifact(directory, spec);
    }

    std::filesystem::path root_;
    std::filesystem::path baseline_;
    std::filesystem::path memory_;
};

void model_log(const ggml_log_level level, const char* text, void*) {
    if (level == GGML_LOG_LEVEL_ERROR) {
        std::fputs(text, stderr);
    }
}

std::vector<llama_token> tokenize(
    const llama_vocab* vocab,
    const std::string& prompt) {
    const auto prompt_size = static_cast<int32_t>(prompt.size());
    const bool add_special = llama_vocab_get_add_bos(vocab);
    const auto required = llama_tokenize(
        vocab,
        prompt.data(),
        prompt_size,
        nullptr,
        0,
        add_special,
        false);
    if (required >= 0 || required == std::numeric_limits<int32_t>::min()) {
        throw std::runtime_error("llama.cpp did not report a token buffer requirement");
    }

    std::vector<llama_token> tokens(static_cast<std::size_t>(-required));
    const auto token_count = llama_tokenize(
        vocab,
        prompt.data(),
        prompt_size,
        tokens.data(),
        static_cast<int32_t>(tokens.size()),
        add_special,
        false);
    if (token_count <= 0) {
        throw std::runtime_error("llama.cpp tokenization failed");
    }
    tokens.resize(static_cast<std::size_t>(token_count));
    return tokens;
}

std::string token_piece(const llama_vocab* vocab, const llama_token token) {
    std::vector<char> buffer(64);
    auto size = llama_token_to_piece(
        vocab,
        token,
        buffer.data(),
        static_cast<int32_t>(buffer.size()),
        0,
        true);
    if (size < 0) {
        buffer.resize(static_cast<std::size_t>(-size));
        size = llama_token_to_piece(
            vocab,
            token,
            buffer.data(),
            static_cast<int32_t>(buffer.size()),
            0,
            true);
    }
    return size > 0 ? std::string(buffer.data(), static_cast<std::size_t>(size))
                    : std::string("<unprintable>");
}

TrialResult run_trial(
    llama_model* model,
    const llama_vocab* vocab,
    std::vector<llama_token> tokens,
    gx1::PinnedHookGeneration pinned) {
    const auto contract_sha256 = pinned.contract_sha256;
    gx1::LlamaGlaminHook llama_hook(
        std::move(pinned.memory_hook), std::move(pinned.target_tensor));

    auto context_params = llama_context_default_params();
    const auto context_tokens = tokens.size() + 8;
    context_params.n_ctx = static_cast<std::uint32_t>(
        std::max<std::size_t>(64, context_tokens));
    context_params.n_batch = static_cast<std::uint32_t>(tokens.size());
    context_params.n_ubatch = context_params.n_batch;
    const auto hardware_threads = std::max(1U, std::thread::hardware_concurrency());
    const auto decode_threads = std::min(8U, hardware_threads);
    context_params.n_threads = static_cast<int32_t>(decode_threads);
    context_params.n_threads_batch = static_cast<int32_t>(decode_threads);
    context_params.cb_eval = &gx1::LlamaGlaminHook::evaluate;
    context_params.cb_eval_user_data = &llama_hook;
    context_params.no_perf = true;

    ContextPointer context(llama_init_from_model(model, context_params), &llama_free);
    if (!context) {
        throw std::runtime_error("failed to create the llama.cpp context");
    }

    const auto decode_status = llama_decode(
        context.get(),
        llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size())));
    llama_hook.throw_if_failed();
    if (decode_status != 0) {
        throw std::runtime_error(
            "llama.cpp decode failed with status " + std::to_string(decode_status));
    }
    if (llama_hook.invocation_count() == 0 || !llama_hook.last_result()) {
        throw std::runtime_error("the configured hidden-state tensor was not observed");
    }

    const auto vocabulary_size = llama_vocab_n_tokens(vocab);
    const auto* native_logits = llama_get_logits_ith(context.get(), -1);
    if (vocabulary_size <= 0 || native_logits == nullptr) {
        throw std::runtime_error("llama.cpp did not produce final-token logits");
    }
    std::vector<float> logits(
        native_logits,
        native_logits + static_cast<std::size_t>(vocabulary_size));
    const auto maximum = std::max_element(logits.begin(), logits.end());
    const auto top_token = static_cast<llama_token>(
        std::distance(logits.begin(), maximum));

    return TrialResult{
        std::move(logits),
        *llama_hook.last_result(),
        contract_sha256,
        top_token,
        llama_hook.invocation_count(),
    };
}

float maximum_logit_difference(
    const std::vector<float>& left,
    const std::vector<float>& right) {
    if (left.size() != right.size()) {
        throw std::invalid_argument("cannot compare logit vectors with different shapes");
    }
    float difference = 0.0F;
    for (std::size_t index = 0; index < left.size(); ++index) {
        difference = std::max(difference, std::fabs(left[index] - right[index]));
    }
    return difference;
}

int run(const std::string& model_path, const int requested_layer) {
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    ModelPointer model(
        llama_model_load_from_file(model_path.c_str(), model_params),
        &llama_model_free);
    if (!model) {
        throw std::runtime_error("failed to load the requested GGUF model");
    }

    const auto hidden_dimension = llama_model_n_embd(model.get());
    const auto layer_count = llama_model_n_layer(model.get());
    if (hidden_dimension <= 0 || layer_count <= 1) {
        throw std::runtime_error("model does not expose a transformer hidden-state contract");
    }
    const auto target_layer = requested_layer >= 0 ? requested_layer : layer_count / 2;
    if (target_layer < 0 || target_layer >= layer_count) {
        throw std::invalid_argument("target layer is outside the model");
    }

    constexpr std::uint32_t query_dimension = 16;
    const auto native_hidden_dimension = static_cast<std::uint32_t>(hidden_dimension);
    std::vector<float> projection(
        static_cast<std::size_t>(query_dimension) * native_hidden_dimension,
        0.0F);
    for (std::uint32_t row = 0; row < query_dimension; ++row) {
        const auto column = static_cast<std::size_t>(row) * native_hidden_dimension /
                            query_dimension;
        projection[static_cast<std::size_t>(row) * native_hidden_dimension + column] =
            1.0F;
    }
    std::vector<float> memory_residual(native_hidden_dimension);
    for (std::size_t index = 0; index < memory_residual.size(); ++index) {
        memory_residual[index] = 8.0F * std::sin(static_cast<float>(index) * 0.017F);
    }

    const auto* vocab = llama_model_get_vocab(model.get());
    const auto tokens = tokenize(
        vocab,
        "Memory changes the next thought about a quiet garden:");
    const auto target_tensor = "l_out-" + std::to_string(target_layer);

    std::vector<char> architecture_buffer(128);
    if (llama_model_meta_val_str(
            model.get(),
            "general.architecture",
            architecture_buffer.data(),
            architecture_buffer.size()) <= 0) {
        throw std::runtime_error("model does not declare its GGUF architecture");
    }
    const gx1::ModelHookContract model_contract{
        gx1::sha256_file(model_path),
        llama_revision,
        architecture_buffer.data(),
        target_tensor,
        native_hidden_dimension,
    };
    DemoHookArtifacts artifacts(
        model_contract, query_dimension, projection, memory_residual);
    gx1::GlaminRuntime glamin(2);
    gx1::PersistentHookGenerationStore generations(glamin);
    const auto generation_a = generations.mount(
        "model-baseline-a", artifacts.baseline(), model_contract);
    const auto generation_b = generations.mount(
        "model-memory-b", artifacts.memory(), model_contract);

    generations.activate(generation_a);
    const auto baseline = run_trial(
        model.get(),
        vocab,
        tokens,
        generations.pin_active());
    generations.activate(generation_b);
    const auto memory = run_trial(
        model.get(),
        vocab,
        tokens,
        generations.pin_active());
    generations.activate(generation_a);
    const auto rollback = run_trial(
        model.get(),
        vocab,
        tokens,
        generations.pin_active());

    const auto memory_difference = maximum_logit_difference(
        baseline.logits,
        memory.logits);
    const auto rollback_difference = maximum_logit_difference(
        baseline.logits,
        rollback.logits);
    if (memory_difference <= 1.0e-4F) {
        throw std::runtime_error("generation B did not measurably change model logits");
    }
    if (rollback_difference > 1.0e-5F) {
        throw std::runtime_error("generation rollback did not reproduce baseline logits");
    }
    if (baseline.contract_sha256 == memory.contract_sha256 ||
        baseline.contract_sha256 != rollback.contract_sha256) {
        throw std::runtime_error("hook artifact identities did not track generation activation");
    }

    std::cout << "model_hidden=" << hidden_dimension
              << " layers=" << layer_count
              << " hook_tensor=" << target_tensor << '\n';
    std::cout << "generation_a=" << baseline.hook.generation
              << " artifact=" << baseline.contract_sha256
              << " top='" << token_piece(vocab, baseline.top_token) << "'\n";
    std::cout << "generation_b=" << memory.hook.generation
              << " artifact=" << memory.contract_sha256
              << " top='" << token_piece(vocab, memory.top_token) << "'"
              << " max_logit_delta=" << memory_difference << '\n';
    std::cout << "rollback_generation=" << rollback.hook.generation
              << " top='" << token_piece(vocab, rollback.top_token) << "'"
              << " max_logit_delta=" << rollback_difference << '\n';
    std::cout << "live hidden-state Glamin injection passed\n";
    return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "usage: gx1_llama_hidden_state_demo MODEL.gguf [LAYER]\n";
        return EXIT_FAILURE;
    }

    llama_log_set(&model_log, nullptr);
    llama_backend_init();
    try {
        const auto requested_layer = argc == 3 ? std::stoi(argv[2]) : -1;
        const auto status = run(argv[1], requested_layer);
        llama_backend_free();
        return status;
    } catch (const std::exception& error) {
        std::cerr << "hidden-state model demo failed: " << error.what() << '\n';
        llama_backend_free();
        return EXIT_FAILURE;
    }
}
