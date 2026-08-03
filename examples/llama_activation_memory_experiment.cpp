#include "gx1/activation_memory_builder.hpp"
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
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "llama.h"

namespace {

using ModelPointer = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using ContextPointer = std::unique_ptr<llama_context, decltype(&llama_free)>;

constexpr const char* llama_revision =
    "ecd99d6a9acbc436bad085783bcd5d0b9ae9e9e9";

struct InferenceResult {
    std::vector<float> hidden_state;
    std::vector<std::vector<float>> token_states;
    std::vector<float> logits;
    llama_token top_token{LLAMA_TOKEN_NULL};
};

struct MemoryInferenceResult {
    std::vector<float> logits;
    gx1::HiddenStateHookResult hook;
    llama_token top_token{LLAMA_TOKEN_NULL};
    double elapsed_milliseconds{0.0};
};

struct MemoryExample {
    std::string name;
    std::size_t association{0};
    std::string query_prompt;
    std::string teacher_prompt;
    std::string target_text;
    llama_token target_token{LLAMA_TOKEN_NULL};
    InferenceResult query;
    InferenceResult teacher;
};

struct HeldOutQuery {
    std::string name;
    std::size_t association{0};
    std::string prompt;
    std::string target_text;
    llama_token target_token{LLAMA_TOKEN_NULL};
    InferenceResult baseline;
};

void model_log(const ggml_log_level level, const char* text, void*) {
    if (level == GGML_LOG_LEVEL_ERROR) {
        std::fputs(text, stderr);
    }
}

std::vector<llama_token> tokenize(
    const llama_vocab* vocab,
    const std::string& text,
    const bool add_special) {
    const auto text_size = static_cast<int32_t>(text.size());
    const auto required = llama_tokenize(
        vocab, text.data(), text_size, nullptr, 0, add_special, false);
    if (required >= 0 || required == std::numeric_limits<int32_t>::min()) {
        throw std::runtime_error("llama.cpp did not report a token buffer requirement");
    }
    std::vector<llama_token> tokens(static_cast<std::size_t>(-required));
    const auto count = llama_tokenize(
        vocab,
        text.data(),
        text_size,
        tokens.data(),
        static_cast<int32_t>(tokens.size()),
        add_special,
        false);
    if (count <= 0) {
        throw std::runtime_error("llama.cpp tokenization failed");
    }
    tokens.resize(static_cast<std::size_t>(count));
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

std::string prompt_token_piece(
    const llama_vocab* vocab,
    const std::string& prompt,
    const std::size_t token_index) {
    const auto tokens = tokenize(vocab, prompt, llama_vocab_get_add_bos(vocab));
    return token_piece(vocab, tokens.at(token_index));
}

llama_context_params context_parameters(const std::size_t token_count) {
    auto parameters = llama_context_default_params();
    parameters.n_ctx = static_cast<std::uint32_t>(
        std::max<std::size_t>(64, token_count + 8U));
    parameters.n_batch = static_cast<std::uint32_t>(token_count);
    parameters.n_ubatch = parameters.n_batch;
    const auto hardware_threads = std::max(1U, std::thread::hardware_concurrency());
    const auto decode_threads = std::min(8U, hardware_threads);
    parameters.n_threads = static_cast<int32_t>(decode_threads);
    parameters.n_threads_batch = static_cast<int32_t>(decode_threads);
    parameters.no_perf = true;
    return parameters;
}

std::vector<float> read_logits(
    llama_context* context,
    const llama_vocab* vocab) {
    const auto vocabulary_size = llama_vocab_n_tokens(vocab);
    const auto* native_logits = llama_get_logits_ith(context, -1);
    if (vocabulary_size <= 0 || native_logits == nullptr) {
        throw std::runtime_error("llama.cpp did not produce final-token logits");
    }
    return std::vector<float>(
        native_logits,
        native_logits + static_cast<std::size_t>(vocabulary_size));
}

llama_token top_token(const std::vector<float>& logits) {
    return static_cast<llama_token>(
        std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
}

InferenceResult capture_inference(
    llama_model* model,
    const llama_vocab* vocab,
    const std::string& prompt,
    const std::uint32_t hidden_dimension,
    const std::string& target_tensor) {
    auto tokens = tokenize(vocab, prompt, llama_vocab_get_add_bos(vocab));
    gx1::LlamaHiddenStateCapture capture(
        hidden_dimension,
        target_tensor,
        std::nullopt,
        true);
    auto parameters = context_parameters(tokens.size());
    parameters.cb_eval = &gx1::LlamaHiddenStateCapture::evaluate;
    parameters.cb_eval_user_data = &capture;
    ContextPointer context(llama_init_from_model(model, parameters), &llama_free);
    if (!context) {
        throw std::runtime_error("failed to create a capture context");
    }
    const auto status = llama_decode(
        context.get(),
        llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size())));
    capture.throw_if_failed();
    if (status != 0 || capture.invocation_count() == 0) {
        throw std::runtime_error("hidden-state capture decode failed");
    }
    auto logits = read_logits(context.get(), vocab);
    const auto maximum = top_token(logits);
    return InferenceResult{
        capture.hidden_state(),
        capture.token_states(),
        std::move(logits),
        maximum};
}

MemoryInferenceResult memory_inference(
    llama_model* model,
    const llama_vocab* vocab,
    const std::string& prompt,
    gx1::PinnedHookGeneration pinned) {
    auto tokens = tokenize(vocab, prompt, llama_vocab_get_add_bos(vocab));
    const auto started = std::chrono::steady_clock::now();
    gx1::LlamaGlaminHook hook(
        std::move(pinned.memory_hook),
        std::move(pinned.target_tensor));
    auto parameters = context_parameters(tokens.size());
    parameters.cb_eval = &gx1::LlamaGlaminHook::evaluate;
    parameters.cb_eval_user_data = &hook;
    ContextPointer context(llama_init_from_model(model, parameters), &llama_free);
    if (!context) {
        throw std::runtime_error("failed to create a memory context");
    }
    const auto status = llama_decode(
        context.get(),
        llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size())));
    hook.throw_if_failed();
    if (status != 0 || !hook.last_result()) {
        throw std::runtime_error("memory intervention decode failed");
    }
    auto logits = read_logits(context.get(), vocab);
    const auto maximum = top_token(logits);
    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started);
    return MemoryInferenceResult{
        std::move(logits), *hook.last_result(), maximum, elapsed.count()};
}

void write_glamin_artifact(
    const std::filesystem::path& directory,
    const std::uint32_t query_dimension,
    const std::vector<std::vector<float>>& keys) {
    const auto glamin_directory = directory / "glamin";
    if (!std::filesystem::create_directories(glamin_directory)) {
        throw std::runtime_error("failed to create the learned Glamin artifact");
    }
    const auto stride = static_cast<std::uint64_t>(query_dimension) * sizeof(float);
    const auto total_bytes = stride * keys.size();
    std::ofstream layout(glamin_directory / "vector_layout.json");
    layout << "{\"dtype\": \"float32\",\"endianness\": \"little\","
              "\"total_vectors\": "
           << keys.size() << ",\"total_bytes\": " << total_bytes
           << ",\"spaces\":[{\"space_id\": \"activation.memory\",\"dim\": "
           << query_dimension << ",\"count\": " << keys.size()
           << ",\"byte_stride\": " << stride << ",\"offset_bytes\": 0}]}\n";
    layout.close();

    std::ofstream vectors(glamin_directory / "vectors.bin", std::ios::binary);
    for (const auto& key : keys) {
        if (key.size() != query_dimension) {
            throw std::invalid_argument("learned memory key has the wrong width");
        }
        vectors.write(
            reinterpret_cast<const char*>(key.data()),
            static_cast<std::streamsize>(key.size() * sizeof(float)));
    }
    vectors.close();

    const std::string space_spec =
        "{\"space_id\": \"activation.memory\",\"dim\": " +
        std::to_string(query_dimension) +
        ",\"metric\": \"l2\",\"normalization\": \"l2\"}";
    std::ofstream contracts(glamin_directory / "contracts.json");
    contracts
        << "{\"spec_id\": \"gx1-activation-memory\",\"embedder\": {\"spec\": {"
           "\"id\": \"variance-selected-hidden-state\",\"version\": \"1\","
           "\"input_schema\": \"f32\",\"preprocess_chain\": [\"l2\"],"
           "\"model_hash\": \"sha256:projection\","
           "\"config_hash\": \"sha256:experiment\","
           "\"hardware_class\": \"cpu\",\"min_ram_mb\": 0,"
           "\"min_vram_mb\": 0},\"contract_hash\": \"sha256:experiment\","
           "\"signature\": \"\"},\"spaces\":[{\"spec\": "
        << space_spec << ",\"contract_hash\": \""
        << gx1::sha256_text(space_spec) << "\",\"signature\": \"\"}]}\n";
    if (!layout || !vectors || !contracts) {
        throw std::runtime_error("failed to write the learned Glamin artifact");
    }
}

class LearnedArtifact final {
public:
    LearnedArtifact(
        const gx1::ModelHookContract& model_contract,
        const gx1::ActivationMemoryBuildResult& memory) {
        const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
        directory_ = std::filesystem::temp_directory_path() /
                     ("gx1-activation-memory-" + std::to_string(nonce));
        try {
            write_glamin_artifact(
                directory_, memory.query_dimension, memory.keys);
            gx1::HookArtifactSpec spec;
            spec.model = model_contract;
            spec.query_dimension = memory.query_dimension;
            spec.metric = gx1::GlaminMetric::l2;
            spec.query_normalization = gx1::ProjectionNormalization::l2;
            spec.gate = 1.0F;
            spec.maximum_distance = memory.maximum_distance;
            spec.address_selection = gx1::AddressSelectionPolicy::all_token_rows;
            spec.input_projection = memory.input_projection;
            spec.glamin_space_id = "activation.memory";
            for (std::size_t index = 0; index < memory.residuals.size(); ++index) {
                spec.residual_labels.push_back(index);
                spec.residuals.insert(
                    spec.residuals.end(),
                    memory.residuals[index].begin(),
                    memory.residuals[index].end());
            }
            gx1::write_hook_artifact(directory_, spec);
        } catch (...) {
            std::error_code error;
            std::filesystem::remove_all(directory_, error);
            throw;
        }
    }

    ~LearnedArtifact() {
        std::error_code error;
        std::filesystem::remove_all(directory_, error);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return directory_;
    }

private:
    std::filesystem::path directory_;
};

std::size_t token_rank(
    const std::vector<float>& logits,
    const llama_token token) {
    const auto target = logits.at(static_cast<std::size_t>(token));
    return static_cast<std::size_t>(std::count_if(
               logits.begin(), logits.end(), [target](const float value) {
                   return value > target;
               })) +
           1U;
}

float maximum_logit_difference(
    const std::vector<float>& left,
    const std::vector<float>& right) {
    if (left.size() != right.size()) {
        throw std::invalid_argument("logit vector widths differ");
    }
    float result = 0.0F;
    for (std::size_t index = 0; index < left.size(); ++index) {
        result = std::max(result, std::fabs(left[index] - right[index]));
    }
    return result;
}

std::uintmax_t directory_bytes(const std::filesystem::path& directory) {
    std::uintmax_t bytes = 0;
    std::error_code error;
    for (std::filesystem::recursive_directory_iterator iterator(directory, error), end;
         iterator != end;
         iterator.increment(error)) {
        if (error) {
            throw std::runtime_error("failed to inspect the constructed artifact");
        }
        if (iterator->is_regular_file(error)) {
            bytes += iterator->file_size(error);
        }
        if (error) {
            throw std::runtime_error("failed to measure the constructed artifact");
        }
    }
    return bytes;
}

int run(const std::string& model_path, const int requested_layer) {
    auto model_parameters = llama_model_default_params();
    model_parameters.n_gpu_layers = 0;
    ModelPointer model(
        llama_model_load_from_file(model_path.c_str(), model_parameters),
        &llama_model_free);
    if (!model) {
        throw std::runtime_error("failed to load the requested GGUF model");
    }
    const auto native_hidden_dimension = llama_model_n_embd(model.get());
    const auto layer_count = llama_model_n_layer(model.get());
    const auto target_layer = requested_layer >= 0 ? requested_layer : layer_count - 2;
    if (native_hidden_dimension <= 0 || target_layer < 0 || target_layer >= layer_count) {
        throw std::runtime_error("model does not satisfy the experiment contract");
    }
    const auto hidden_dimension = static_cast<std::uint32_t>(native_hidden_dimension);
    const auto target_tensor = "l_out-" + std::to_string(target_layer);
    const auto* vocab = llama_model_get_vocab(model.get());

    struct AssociationSpec {
        std::string id;
        std::string entity;
        std::string target;
    };
    const std::vector<AssociationSpec> associations{
        {"arcturus", "Arcturus", " blue"},
        {"bellatrix", "Bellatrix", " cedar"},
        {"cygnus", "Cygnus", " amber"},
        {"draco", "Draco", " copper"},
        {"eridanus", "Eridanus", " violet"},
        {"fornax", "Fornax", " maple"},
        {"gemini", "Gemini", " silver"},
        {"hydra", "Hydra", " quartz"},
    };
    std::string memory_table = "Memory table:\n";
    for (const auto& association : associations) {
        memory_table += association.entity + " =>" + association.target + "\n";
    }

    struct ConstructionForm {
        std::string id;
        std::string prefix;
        std::string suffix;
    };
    const std::vector<ConstructionForm> construction_forms{
        {"memory-lookup", "Memory lookup:\n", " =>"},
        {"stored-association", "Stored association:\n", " =>"},
        {"natural-address", "Stored-value question:\nWhat belongs to ",
         "?\nAnswer:"},
    };
    std::vector<MemoryExample> examples;
    std::vector<HeldOutQuery> validation_queries;
    std::vector<HeldOutQuery> held_out_queries;
    std::vector<HeldOutQuery> natural_question_queries;
    for (std::size_t index = 0; index < associations.size(); ++index) {
        const auto& association = associations[index];
        for (const auto& form : construction_forms) {
            const auto prompt = form.prefix + association.entity + form.suffix;
            examples.push_back(MemoryExample{
                association.id + "-" + form.id,
                index,
                prompt,
                memory_table + prompt,
                association.target,
                LLAMA_TOKEN_NULL,
                {},
                {},
            });
        }
        validation_queries.push_back(HeldOutQuery{
            association.id + "-recall-request",
            index,
            "Recall request:\n" + association.entity + " =>",
            association.target,
            LLAMA_TOKEN_NULL,
            {},
        });
        validation_queries.push_back(HeldOutQuery{
            association.id + "-natural-validation",
            index,
            "Association question:\nWhich value belongs to " +
                association.entity + "?\nAnswer:",
            association.target,
            LLAMA_TOKEN_NULL,
            {},
        });
        held_out_queries.push_back(HeldOutQuery{
            association.id + "-please-retrieve",
            index,
            "Please retrieve:\n" + association.entity + " =>",
            association.target,
            LLAMA_TOKEN_NULL,
            {},
        });
        natural_question_queries.push_back(HeldOutQuery{
            association.id + "-natural-question",
            index,
            "What is the stored value for " + association.entity + "?\nAnswer:",
            association.target,
            LLAMA_TOKEN_NULL,
            {},
        });
    }
    const std::vector<std::string> control_prompts{
        "Memory lookup:\nRigel =>",
        "Recall request:\nSirius =>",
        "Stored association:\nAltair =>",
        "Known value:\nPolaris =>",
        "Stored-value question:\nWhat belongs to Betelgeuse?\nAnswer:",
        "Association question:\nWhich value belongs to Procyon?\nAnswer:",
        "What is the stored value for Spica?\nAnswer:",
        "Please retrieve:\nAldebaran =>",
        "The capital of France is",
        "Two plus two equals",
        "Write a short poem about rain:",
        "The opposite of hot is",
    };
    const std::vector<std::string> held_out_control_prompts{
        "What is the stored value for Vega?\nAnswer:",
        "Which value is stored for Deneb?\nAnswer:",
        "Please retrieve:\nCapella =>",
        "What belongs to Antares?\nAnswer:",
    };

    for (auto& example : examples) {
        const auto target_tokens = tokenize(vocab, example.target_text, false);
        if (target_tokens.size() != 1) {
            throw std::runtime_error(
                "memory target is not one token: " + example.target_text);
        }
        example.target_token = target_tokens.front();
        example.query = capture_inference(
            model.get(),
            vocab,
            example.query_prompt,
            hidden_dimension,
            target_tensor);
        example.teacher = capture_inference(
            model.get(), vocab, example.teacher_prompt, hidden_dimension, target_tensor);
        std::cout << "teacher=" << example.name
                  << " top='" << token_piece(vocab, example.teacher.top_token) << "'"
                  << " target_rank="
                  << token_rank(example.teacher.logits, example.target_token) << '\n';
    }

    const auto capture_evaluation_queries = [model_pointer = model.get(),
                                             vocab,
                                             hidden_dimension,
                                             &target_tensor](
                                                std::vector<HeldOutQuery>& queries) {
        for (auto& query : queries) {
            const auto target_tokens = tokenize(vocab, query.target_text, false);
            if (target_tokens.size() != 1) {
                throw std::runtime_error(
                    "evaluation target is not one token: " + query.target_text);
            }
            query.target_token = target_tokens.front();
            query.baseline = capture_inference(
                model_pointer,
                vocab,
                query.prompt,
                hidden_dimension,
                target_tensor);
        }
    };
    capture_evaluation_queries(validation_queries);
    capture_evaluation_queries(held_out_queries);
    capture_evaluation_queries(natural_question_queries);

    std::vector<InferenceResult> controls;
    for (std::size_t index = 0; index < control_prompts.size(); ++index) {
        controls.push_back(capture_inference(
            model.get(),
            vocab,
            control_prompts[index],
            hidden_dimension,
            target_tensor));
    }
    std::vector<InferenceResult> held_out_controls;
    for (std::size_t index = 0; index < held_out_control_prompts.size(); ++index) {
        held_out_controls.push_back(capture_inference(
            model.get(),
            vocab,
            held_out_control_prompts[index],
            hidden_dimension,
            target_tensor));
    }

    constexpr std::uint32_t query_dimension = 256;
    std::vector<gx1::ActivationMemoryConstructionView> construction_views;
    for (const auto& example : examples) {
        construction_views.push_back(gx1::ActivationMemoryConstructionView{
            example.association,
            example.query.token_states,
            example.query.hidden_state,
            example.teacher.hidden_state,
        });
    }
    std::vector<gx1::ActivationStateSequence> calibration_negatives;
    for (const auto& control : controls) {
        calibration_negatives.push_back(control.token_states);
    }
    std::vector<gx1::ActivationMemoryValidationView> construction_validation;
    for (const auto& query : validation_queries) {
        construction_validation.push_back(gx1::ActivationMemoryValidationView{
            query.association,
            query.baseline.token_states,
        });
    }
    const auto construction_started = std::chrono::steady_clock::now();
    const auto memory = gx1::ActivationMemoryBuilder::build(
        construction_views,
        calibration_negatives,
        construction_validation,
        gx1::ActivationMemoryBuildConfig{query_dimension, 0.75F});
    const auto construction_milliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - construction_started).count();
    for (std::size_t index = 0; index < examples.size(); ++index) {
        std::cout << "derived_key=" << examples[index].name
                  << " address_token=" << memory.selected_candidates[index]
                  << " address_piece='"
                  << prompt_token_piece(
                         vocab,
                         examples[index].query_prompt,
                         memory.selected_candidates[index])
                  << "'"
                  << " association_margin="
                  << memory.association_margins.at(examples[index].association)
                  << '\n';
    }
    for (std::size_t index = 0; index < validation_queries.size(); ++index) {
        const auto& selection = memory.validation_selections[index];
        std::cout << "calibration=" << validation_queries[index].name
                  << " selected=" << selection.key
                  << " address_token=" << selection.candidate
                  << " distance=" << selection.distance << '\n';
    }
    std::cout << "calibration_positive=" << memory.maximum_validation_distance
              << " calibration_negative=" << memory.minimum_negative_distance
              << '\n';

    std::vector<char> architecture(128);
    if (llama_model_meta_val_str(
            model.get(),
            "general.architecture",
            architecture.data(),
            architecture.size()) <= 0) {
        throw std::runtime_error("model does not declare its GGUF architecture");
    }
    const gx1::ModelHookContract model_contract{
        gx1::sha256_file(model_path),
        llama_revision,
        architecture.data(),
        target_tensor,
        hidden_dimension,
    };
    LearnedArtifact artifact(model_contract, memory);
    const auto artifact_size = directory_bytes(artifact.path());
    gx1::GlaminRuntime glamin(2);
    gx1::PersistentHookGenerationStore generations(glamin);
    const auto generation = generations.mount(
        "activation-derived-memory", artifact.path(), model_contract);
    generations.activate(generation);

    std::cout << "memory_count=" << examples.size()
              << " query_dimension=" << query_dimension
              << " training_radius=0"
              << " validation_radius=" << memory.maximum_validation_distance
              << " negative_boundary=" << memory.minimum_negative_distance
              << " maximum_distance=" << memory.maximum_distance
              << " construction_ms=" << construction_milliseconds
              << " artifact_bytes=" << artifact_size << '\n';
    double memory_latency_milliseconds = 0.0;
    std::size_t memory_inference_count = 0;
    bool every_memory_improved = true;
    for (std::size_t index = 0; index < examples.size(); ++index) {
        const auto memory = memory_inference(
            model.get(),
            vocab,
            examples[index].query_prompt,
            generations.pin_active());
        memory_latency_milliseconds += memory.elapsed_milliseconds;
        ++memory_inference_count;
        const auto target_index = static_cast<std::size_t>(examples[index].target_token);
        const auto baseline_logit = examples[index].query.logits.at(target_index);
        const auto memory_logit = memory.logits.at(target_index);
        const auto memory_target_rank = token_rank(
            memory.logits, examples[index].target_token);
        const auto improved = memory_logit > baseline_logit;
        const auto selected_association = examples.at(
            static_cast<std::size_t>(memory.hook.label)).association;
        every_memory_improved = every_memory_improved && improved && memory.hook.applied &&
                                selected_association == examples[index].association &&
                                memory_target_rank == 1U;
        std::cout << "training_memory=" << examples[index].name
                  << " selected=" << memory.hook.label
                  << " address_token=" << memory.hook.address_candidate
                  << " address_piece='"
                  << prompt_token_piece(
                         vocab,
                         examples[index].query_prompt,
                         memory.hook.address_candidate)
                  << "'"
                  << " distance=" << memory.hook.distance
                  << " applied=" << (memory.hook.applied ? "yes" : "no")
                  << " baseline_top='"
                  << token_piece(vocab, examples[index].query.top_token) << "'"
                  << " memory_top='" << token_piece(vocab, memory.top_token) << "'"
                  << " target='" << examples[index].target_text << "'"
                  << " target_logit_delta=" << (memory_logit - baseline_logit)
                  << " target_rank=" << memory_target_rank
                  << '\n';
    }

    const auto evaluate_queries = [model_pointer = model.get(),
                                   vocab,
                                   &generations,
                                   &examples,
                                   &memory_latency_milliseconds,
                                   &memory_inference_count](
                                      const std::vector<HeldOutQuery>& queries,
                                      const char* category) {
        bool every_query_recalled = true;
        for (const auto& query : queries) {
            const auto memory = memory_inference(
                model_pointer,
                vocab,
                query.prompt,
                generations.pin_active());
            memory_latency_milliseconds += memory.elapsed_milliseconds;
            ++memory_inference_count;
            const auto selected_association = examples.at(
                static_cast<std::size_t>(memory.hook.label)).association;
            const auto target_index = static_cast<std::size_t>(query.target_token);
            const auto target_delta = memory.logits.at(target_index) -
                                      query.baseline.logits.at(target_index);
            const auto target_rank = token_rank(memory.logits, query.target_token);
            every_query_recalled = every_query_recalled && memory.hook.applied &&
                                   selected_association == query.association &&
                                   target_delta > 0.0F && target_rank == 1U;
            std::cout << category << '=' << query.name
                      << " selected=" << memory.hook.label
                      << " address_token=" << memory.hook.address_candidate
                      << " address_piece='"
                      << prompt_token_piece(
                             vocab, query.prompt, memory.hook.address_candidate)
                      << "'"
                      << " distance=" << memory.hook.distance
                      << " applied=" << (memory.hook.applied ? "yes" : "no")
                      << " baseline_top='"
                      << token_piece(vocab, query.baseline.top_token) << "'"
                      << " memory_top='" << token_piece(vocab, memory.top_token) << "'"
                      << " target_logit_delta=" << target_delta
                      << " target_rank=" << target_rank << '\n';
        }
        return every_query_recalled;
    };
    const auto validation_recalled = evaluate_queries(
        validation_queries, "validation");
    const auto every_held_out_recalled = evaluate_queries(
        held_out_queries, "held_out");
    const auto every_natural_question_recalled = evaluate_queries(
        natural_question_queries, "natural_question");

    bool controls_untouched = true;
    for (std::size_t index = 0; index < controls.size(); ++index) {
        const auto memory = memory_inference(
            model.get(),
            vocab,
            control_prompts[index],
            generations.pin_active());
        memory_latency_milliseconds += memory.elapsed_milliseconds;
        ++memory_inference_count;
        const auto delta = maximum_logit_difference(controls[index].logits, memory.logits);
        controls_untouched = controls_untouched && !memory.hook.applied && delta <= 1.0e-5F;
        std::cout << "control=" << index
                  << " address_token=" << memory.hook.address_candidate
                  << " address_piece='"
                  << prompt_token_piece(
                         vocab,
                         control_prompts[index],
                         memory.hook.address_candidate)
                  << "'"
                  << " distance=" << memory.hook.distance
                  << " applied=" << (memory.hook.applied ? "yes" : "no")
                  << " max_logit_delta=" << delta << '\n';
    }
    for (std::size_t index = 0; index < held_out_controls.size(); ++index) {
        const auto memory = memory_inference(
            model.get(),
            vocab,
            held_out_control_prompts[index],
            generations.pin_active());
        memory_latency_milliseconds += memory.elapsed_milliseconds;
        ++memory_inference_count;
        const auto delta = maximum_logit_difference(
            held_out_controls[index].logits, memory.logits);
        controls_untouched = controls_untouched && !memory.hook.applied &&
                             delta <= 1.0e-5F;
        std::cout << "held_out_control=" << index
                  << " address_token=" << memory.hook.address_candidate
                  << " address_piece='"
                  << prompt_token_piece(
                         vocab,
                         held_out_control_prompts[index],
                         memory.hook.address_candidate)
                  << "'"
                  << " distance=" << memory.hook.distance
                  << " applied=" << (memory.hook.applied ? "yes" : "no")
                  << " max_logit_delta=" << delta << '\n';
    }

    if (!every_memory_improved) {
        throw std::runtime_error(
            "activation-derived memory did not improve every targeted recall");
    }
    if (!every_held_out_recalled) {
        throw std::runtime_error(
            "activation-derived memory did not generalize to every held-out phrasing");
    }
    if (!validation_recalled) {
        throw std::runtime_error(
            "activation-derived memory failed its validation phrasings");
    }
    if (!every_natural_question_recalled) {
        throw std::runtime_error(
            "activation-derived memory did not generalize to natural questions");
    }
    if (!controls_untouched) {
        throw std::runtime_error("memory abstention did not preserve the control prompts");
    }
    std::cout << "scale_associations=" << associations.size()
              << " construction_views=" << examples.size()
              << " validation_queries=" << validation_queries.size()
              << " held_out_positive_queries="
              << held_out_queries.size() + natural_question_queries.size()
              << " negative_queries="
              << control_prompts.size() + held_out_control_prompts.size()
              << " average_memory_inference_ms="
              << memory_latency_milliseconds /
                     static_cast<double>(memory_inference_count)
              << '\n';
    std::cout << "scaled associative memory experiment passed\n";
    return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "usage: gx1_llama_activation_memory_experiment MODEL.gguf [LAYER]\n";
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
        std::cerr << "activation memory experiment failed: " << error.what() << '\n';
        llama_backend_free();
        return EXIT_FAILURE;
    }
}
