#include "gx1/activation_memory_builder.hpp"
#include "gx1/factorized_memory_hook.hpp"
#include "gx1/llama_glamin_hook.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
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

struct InferenceResult {
    std::vector<float> hidden_state;
    std::vector<std::vector<float>> token_states;
    std::vector<float> logits;
    llama_token top_token{LLAMA_TOKEN_NULL};
};

struct FactorPrompt {
    std::size_t factor{0};
    std::string prompt;
    InferenceResult inference;
};

struct TupleSpec {
    std::size_t entity{0};
    std::size_t relation{0};
    std::string target;
    llama_token target_token{LLAMA_TOKEN_NULL};
    InferenceResult query;
    InferenceResult teacher;
};

struct MemoryInferenceResult {
    std::vector<float> logits;
    gx1::FactorizedMemoryResult hook;
};

struct ActionViewSpec {
    std::size_t association{0};
    std::size_t entity{0};
    std::size_t relation{0};
    std::string name;
    std::string prompt;
    llama_token target_token{LLAMA_TOKEN_NULL};
    std::vector<float> query_state;
    std::vector<float> teacher_state;
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
    const auto required = llama_tokenize(
        vocab,
        text.data(),
        static_cast<std::int32_t>(text.size()),
        nullptr,
        0,
        add_special,
        false);
    if (required >= 0 || required == std::numeric_limits<std::int32_t>::min()) {
        throw std::runtime_error("llama.cpp did not report a token buffer requirement");
    }
    std::vector<llama_token> tokens(static_cast<std::size_t>(-required));
    const auto count = llama_tokenize(
        vocab,
        text.data(),
        static_cast<std::int32_t>(text.size()),
        tokens.data(),
        static_cast<std::int32_t>(tokens.size()),
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
    auto count = llama_token_to_piece(
        vocab, token, buffer.data(), static_cast<std::int32_t>(buffer.size()), 0, true);
    if (count < 0) {
        buffer.resize(static_cast<std::size_t>(-count));
        count = llama_token_to_piece(
            vocab,
            token,
            buffer.data(),
            static_cast<std::int32_t>(buffer.size()),
            0,
            true);
    }
    return count > 0 ? std::string(buffer.data(), static_cast<std::size_t>(count))
                     : std::string("<unprintable>");
}

llama_context_params context_parameters(const std::size_t token_count) {
    auto parameters = llama_context_default_params();
    parameters.n_ctx = static_cast<std::uint32_t>(
        std::max<std::size_t>(64U, token_count + 8U));
    parameters.n_batch = static_cast<std::uint32_t>(token_count);
    parameters.n_ubatch = parameters.n_batch;
    const auto threads = std::min(8U, std::max(1U, std::thread::hardware_concurrency()));
    parameters.n_threads = static_cast<std::int32_t>(threads);
    parameters.n_threads_batch = static_cast<std::int32_t>(threads);
    parameters.no_perf = true;
    return parameters;
}

std::vector<float> read_logits(llama_context* context, const llama_vocab* vocab) {
    const auto count = llama_vocab_n_tokens(vocab);
    const auto* logits = llama_get_logits_ith(context, -1);
    if (count <= 0 || logits == nullptr) {
        throw std::runtime_error("llama.cpp did not produce final-token logits");
    }
    return std::vector<float>(logits, logits + static_cast<std::size_t>(count));
}

llama_token top_token(const std::vector<float>& logits) {
    return static_cast<llama_token>(
        std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
}

std::size_t token_rank(const std::vector<float>& logits, const llama_token token) {
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
        throw std::invalid_argument("logit widths differ");
    }
    float maximum = 0.0F;
    for (std::size_t index = 0; index < left.size(); ++index) {
        maximum = std::max(maximum, std::abs(left[index] - right[index]));
    }
    return maximum;
}

InferenceResult capture(
    llama_model* model,
    const llama_vocab* vocab,
    const std::string& prompt,
    const std::uint32_t hidden_dimension,
    const std::string& target_tensor) {
    auto tokens = tokenize(vocab, prompt, llama_vocab_get_add_bos(vocab));
    gx1::LlamaHiddenStateCapture hook(
        hidden_dimension, target_tensor, std::nullopt, true);
    auto parameters = context_parameters(tokens.size());
    parameters.cb_eval = &gx1::LlamaHiddenStateCapture::evaluate;
    parameters.cb_eval_user_data = &hook;
    ContextPointer context(llama_init_from_model(model, parameters), &llama_free);
    if (!context) {
        throw std::runtime_error("failed to create a capture context");
    }
    const auto status = llama_decode(
        context.get(),
        llama_batch_get_one(tokens.data(), static_cast<std::int32_t>(tokens.size())));
    hook.throw_if_failed();
    if (status != 0 || hook.invocation_count() == 0U) {
        throw std::runtime_error("hidden-state capture failed");
    }
    auto logits = read_logits(context.get(), vocab);
    const auto top = top_token(logits);
    return {hook.hidden_state(), hook.token_states(), std::move(logits), top};
}

std::string tuple_prompt(const std::string& entity, const std::string& relation) {
    return "Memory lookup:\nRelation: " + relation + "\nEntity: " + entity +
           "\nAnswer:";
}

gx1::ActivationMemoryBuildResult build_factor(
    const std::vector<FactorPrompt>& prompts,
    const std::vector<InferenceResult>& negatives,
    const std::vector<std::pair<std::size_t, InferenceResult>>& validation) {
    std::vector<gx1::ActivationMemoryConstructionView> construction;
    for (const auto& prompt : prompts) {
        construction.push_back({
            prompt.factor,
            prompt.inference.token_states,
            prompt.inference.hidden_state,
            prompt.inference.hidden_state,
        });
    }
    std::vector<gx1::ActivationStateSequence> calibration_negatives;
    for (const auto& negative : negatives) {
        calibration_negatives.push_back(negative.token_states);
    }
    std::vector<gx1::ActivationMemoryValidationView> validation_views;
    for (const auto& item : validation) {
        validation_views.push_back({item.first, item.second.token_states});
    }
    return gx1::ActivationMemoryBuilder::build(
        construction,
        calibration_negatives,
        validation_views,
        gx1::ActivationMemoryBuildConfig{256U, 0.5F, false});
}

gx1::FactorSearchConfig factor_config(
    const gx1::ActivationMemoryBuildResult& memory) {
    return gx1::FactorSearchConfig{
        memory.hidden_dimension,
        memory.query_dimension,
        memory.input_projection,
        gx1::ProjectionNormalization::l2,
        memory.maximum_distance,
    };
}

std::vector<float> flatten(const std::vector<std::vector<float>>& rows) {
    std::vector<float> values;
    for (const auto& row : rows) {
        values.insert(values.end(), row.begin(), row.end());
    }
    return values;
}

MemoryInferenceResult infer_with_memory(
    llama_model* model,
    const llama_vocab* vocab,
    const std::string& prompt,
    gx1::FactorizedLayerMemoryHook memory,
    const std::string& target_tensor) {
    auto tokens = tokenize(vocab, prompt, llama_vocab_get_add_bos(vocab));
    gx1::LlamaFactorizedGlaminHook hook(std::move(memory), target_tensor);
    auto parameters = context_parameters(tokens.size());
    parameters.cb_eval = &gx1::LlamaFactorizedGlaminHook::evaluate;
    parameters.cb_eval_user_data = &hook;
    ContextPointer context(llama_init_from_model(model, parameters), &llama_free);
    if (!context) {
        throw std::runtime_error("failed to create a factorized memory context");
    }
    const auto status = llama_decode(
        context.get(),
        llama_batch_get_one(tokens.data(), static_cast<std::int32_t>(tokens.size())));
    hook.throw_if_failed();
    if (status != 0 || !hook.last_result()) {
        throw std::runtime_error("factorized memory inference failed");
    }
    return {read_logits(context.get(), vocab), *hook.last_result()};
}

int run(const std::string& model_path) {
    auto model_parameters = llama_model_default_params();
    model_parameters.n_gpu_layers = 0;
    ModelPointer model(
        llama_model_load_from_file(model_path.c_str(), model_parameters),
        &llama_model_free);
    if (!model) {
        throw std::runtime_error("failed to load the requested GGUF model");
    }
    const auto native_hidden = llama_model_n_embd(model.get());
    const auto layer = llama_model_n_layer(model.get()) - 2;
    if (native_hidden <= 0 || layer < 0) {
        throw std::runtime_error("model does not satisfy the experiment contract");
    }
    const auto hidden_dimension = static_cast<std::uint32_t>(native_hidden);
    const auto target_tensor = "l_out-" + std::to_string(layer);
    const auto* vocab = llama_model_get_vocab(model.get());

    const std::vector<std::string> entities{
        "Arcturus", "Bellatrix", "Cygnus", "Draco"};
    const std::vector<std::string> relations{"color", "material"};
    std::vector<TupleSpec> tuples{
        {0U, 0U, " blue", LLAMA_TOKEN_NULL, {}, {}},
        {0U, 1U, " cedar", LLAMA_TOKEN_NULL, {}, {}},
        {1U, 0U, " amber", LLAMA_TOKEN_NULL, {}, {}},
        {2U, 1U, " copper", LLAMA_TOKEN_NULL, {}, {}},
        {3U, 0U, " violet", LLAMA_TOKEN_NULL, {}, {}},
        {3U, 1U, " maple", LLAMA_TOKEN_NULL, {}, {}},
    };
    std::string table = "Memory table:\n";
    for (const auto& tuple : tuples) {
        table += relations[tuple.relation] + " of " + entities[tuple.entity] +
                 " =>" + tuple.target + "\n";
    }

    std::vector<FactorPrompt> entity_prompts;
    for (std::size_t entity = 0; entity < entities.size(); ++entity) {
        for (const auto& prefix : {std::string("Entity lookup: "),
                                  std::string("Object name: ")}) {
            const auto prompt = prefix + entities[entity];
            entity_prompts.push_back({
                entity,
                prompt,
                capture(model.get(), vocab, prompt, hidden_dimension, target_tensor),
            });
        }
    }
    std::vector<FactorPrompt> relation_prompts;
    for (std::size_t relation = 0; relation < relations.size(); ++relation) {
        const std::vector<std::string> prompts{
            "Relation lookup: " + relations[relation],
            "Property name: " + relations[relation],
            "Question template: What is the " + relations[relation] +
                " of an object?",
        };
        for (const auto& prompt : prompts) {
            relation_prompts.push_back({
                relation,
                prompt,
                capture(model.get(), vocab, prompt, hidden_dimension, target_tensor),
            });
        }
    }

    std::vector<std::pair<std::size_t, InferenceResult>> entity_validation;
    std::vector<std::pair<std::size_t, InferenceResult>> relation_validation;
    std::vector<std::vector<InferenceResult>> tuple_baselines(
        entities.size(), std::vector<InferenceResult>(relations.size()));
    for (std::size_t entity = 0; entity < entities.size(); ++entity) {
        for (std::size_t relation = 0; relation < relations.size(); ++relation) {
            auto baseline = capture(
                model.get(),
                vocab,
                tuple_prompt(entities[entity], relations[relation]),
                hidden_dimension,
                target_tensor);
            entity_validation.emplace_back(entity, baseline);
            relation_validation.emplace_back(relation, baseline);
            tuple_baselines[entity][relation] = std::move(baseline);

            const std::vector<std::string> development_prompts{
                "Tell me the stored " + relations[relation] + " for " +
                    entities[entity] + ".\nAnswer:",
                "For " + entities[entity] + ", the " + relations[relation] +
                    " value is",
            };
            for (const auto& prompt : development_prompts) {
                auto development = capture(
                    model.get(),
                    vocab,
                    prompt,
                    hidden_dimension,
                    target_tensor);
                entity_validation.emplace_back(entity, development);
                relation_validation.emplace_back(relation, std::move(development));
            }
        }
    }

    std::vector<InferenceResult> entity_negatives;
    for (const auto& entity : {std::string("Rigel"), std::string("Sirius")}) {
        entity_negatives.push_back(capture(
            model.get(),
            vocab,
            tuple_prompt(entity, "color"),
            hidden_dimension,
            target_tensor));
    }
    std::vector<InferenceResult> relation_negatives;
    for (const auto& relation : {std::string("temperature"), std::string("age")}) {
        relation_negatives.push_back(capture(
            model.get(),
            vocab,
            tuple_prompt("Arcturus", relation),
            hidden_dimension,
            target_tensor));
    }

    const auto entity_memory = build_factor(
        entity_prompts, entity_negatives, entity_validation);
    const auto relation_memory = build_factor(
        relation_prompts, relation_negatives, relation_validation);
    std::cout << "entity_radius=" << entity_memory.maximum_distance
              << " entity_negative=" << entity_memory.minimum_negative_distance
              << " relation_radius=" << relation_memory.maximum_distance
              << " relation_negative=" << relation_memory.minimum_negative_distance
              << '\n';

    for (auto& tuple : tuples) {
        const auto prompt = tuple_prompt(
            entities[tuple.entity], relations[tuple.relation]);
        tuple.query = tuple_baselines[tuple.entity][tuple.relation];
        tuple.teacher = capture(
            model.get(), vocab, table + prompt, hidden_dimension, target_tensor);
        const auto tokens = tokenize(vocab, tuple.target, false);
        if (tokens.size() != 1U) {
            throw std::runtime_error("factorized target is not one token");
        }
        tuple.target_token = tokens.front();
        std::cout << "teacher=" << entities[tuple.entity] << '/'
                  << relations[tuple.relation]
                  << " top='" << token_piece(vocab, tuple.teacher.top_token) << "'"
                  << " target_rank="
                  << token_rank(tuple.teacher.logits, tuple.target_token) << '\n';
    }

    gx1::GlaminRuntime runtime(2);
    gx1::GlaminGenerationStore generations(runtime);
    const auto entity_generation = generations.mount_flat(
        "factorized-entities",
        entity_memory.query_dimension,
        flatten(entity_memory.keys));
    const auto relation_generation = generations.mount_flat(
        "factorized-relations",
        relation_memory.query_dimension,
        flatten(relation_memory.keys));
    std::vector<std::uint64_t> entity_labels;
    for (const auto& prompt : entity_prompts) {
        entity_labels.push_back(prompt.factor);
    }
    std::vector<std::uint64_t> relation_labels;
    for (const auto& prompt : relation_prompts) {
        relation_labels.push_back(prompt.factor);
    }
    std::vector<ActionViewSpec> action_view_specs;
    bool action_teachers_valid = true;
    for (std::size_t tuple_index = 0; tuple_index < tuples.size(); ++tuple_index) {
        const auto& tuple = tuples[tuple_index];
        const std::vector<std::pair<std::string, std::string>> action_views{
            {"canonical",
             tuple_prompt(entities[tuple.entity], relations[tuple.relation])},
            {"structured",
             "Stored fact:\nProperty: " + relations[tuple.relation] +
                 "\nObject: " + entities[tuple.entity] + "\nValue:"},
            {"natural",
             "What is the " + relations[tuple.relation] + " of " +
                 entities[tuple.entity] + "?\nAnswer:"},
            {"request",
             "Tell me the stored " + relations[tuple.relation] + " for " +
                 entities[tuple.entity] + ".\nAnswer:"},
            {"continuation",
             "For " + entities[tuple.entity] + ", the " +
                 relations[tuple.relation] + " value is"},
        };
        for (std::size_t view = 0; view < action_views.size(); ++view) {
            auto query_state = view == 0U
                                   ? tuple.query.hidden_state
                                   : capture(
                                         model.get(),
                                         vocab,
                                         action_views[view].second,
                                         hidden_dimension,
                                         target_tensor)
                                         .hidden_state;
            // Every surface form for a tuple targets the same canonical action
            // state. The address varies with phrasing; the internal answer action
            // deliberately does not.
            const auto& teacher = tuple.teacher;
            const auto teacher_rank = token_rank(
                teacher.logits, tuple.target_token);
            action_teachers_valid = action_teachers_valid && teacher_rank == 1U;
            action_view_specs.push_back(ActionViewSpec{
                tuple_index,
                tuple.entity,
                tuple.relation,
                action_views[view].first,
                action_views[view].second,
                tuple.target_token,
                std::move(query_state),
                teacher.hidden_state,
            });
            std::cout << "action_view=" << action_views[view].first << '/'
                      << entities[tuple.entity] << '/' << relations[tuple.relation]
                      << " teacher_rank=" << teacher_rank << '\n';
        }
    }
    if (!action_teachers_valid) {
        throw std::runtime_error("an action construction view lacks a rank-one teacher");
    }

    std::vector<gx1::ActivationMemoryConstructionView> action_construction;
    for (const auto& action : action_view_specs) {
        action_construction.push_back({
            action.association,
            {action.query_state},
            action.query_state,
            action.teacher_state,
        });
    }
    std::vector<gx1::ActivationStateSequence> action_negatives{
        {tuple_baselines[1U][1U].hidden_state},
        {tuple_baselines[2U][0U].hidden_state},
    };
    for (const auto& prompt : {
             std::string("The capital of France is"),
             std::string("Two plus two equals"),
             tuple_prompt("Rigel", "color"),
             tuple_prompt("Arcturus", "temperature")}) {
        action_negatives.push_back({capture(
            model.get(), vocab, prompt, hidden_dimension, target_tensor).hidden_state});
    }
    std::vector<gx1::ActivationMemoryValidationView> action_validation;
    for (std::size_t tuple_index = 0; tuple_index < tuples.size(); ++tuple_index) {
        const auto& tuple = tuples[tuple_index];
        const auto prompt =
            "Retrieve the " + relations[tuple.relation] + " assigned to " +
            entities[tuple.entity] + ".\nAnswer:";
        action_validation.push_back({
            tuple_index,
            {capture(
                 model.get(), vocab, prompt, hidden_dimension, target_tensor)
                 .hidden_state},
        });
    }
    const auto action_memory = gx1::ActivationMemoryBuilder::build(
        action_construction,
        action_negatives,
        action_validation,
        gx1::ActivationMemoryBuildConfig{256U, 0.5F, false});
    std::cout << "action_radius=" << action_memory.maximum_distance
              << " action_negative=" << action_memory.minimum_negative_distance
              << '\n';

    auto payloads = std::make_shared<gx1::TupleResidualLedger>();
    for (std::size_t index = 0; index < action_view_specs.size(); ++index) {
        const auto& action = action_view_specs[index];
        payloads->insert_variant(
            action.entity,
            action.relation,
            action_memory.keys[index],
            action_memory.residuals[index]);
    }

    const auto make_hook = [&]() {
        generations.activate(entity_generation);
        auto entity_pin = generations.pin_active();
        generations.activate(relation_generation);
        auto relation_pin = generations.pin_active();
        return gx1::FactorizedLayerMemoryHook(
            std::move(entity_pin),
            factor_config(entity_memory),
            entity_labels,
            std::move(relation_pin),
            factor_config(relation_memory),
            relation_labels,
            1.0F,
            payloads,
            action_memory.maximum_distance,
            factor_config(action_memory));
    };

    bool action_views_recalled = true;
    for (const auto& action : action_view_specs) {
        const auto memory = infer_with_memory(
            model.get(), vocab, action.prompt, make_hook(), target_tensor);
        const auto rank = token_rank(memory.logits, action.target_token);
        action_views_recalled = action_views_recalled && memory.hook.applied &&
                                memory.hook.entity.factor_label == action.entity &&
                                memory.hook.relation.factor_label == action.relation &&
                                rank == 1U;
        std::cout << "registered_action=" << action.name << '/'
                  << entities[action.entity] << '/' << relations[action.relation]
                  << " entity_distance=" << memory.hook.entity.distance
                  << " relation_distance=" << memory.hook.relation.distance
                  << " action_variant=" << memory.hook.action_variant
                  << " applied=" << (memory.hook.applied ? "yes" : "no")
                  << " target_rank=" << rank << '\n';
    }

    std::size_t evaluation_recalled = 0U;
    for (const auto& tuple : tuples) {
        const std::vector<std::pair<std::string, std::string>> held_out_prompts{
            {"response",
             "Please provide the " + relations[tuple.relation] + " recorded for " +
                 entities[tuple.entity] + ".\nResponse:"},
            {"compact",
             "Recorded " + relations[tuple.relation] + " for " +
                 entities[tuple.entity] + ":"},
        };
        for (const auto& held_out : held_out_prompts) {
            const auto baseline = capture(
                model.get(),
                vocab,
                held_out.second,
                hidden_dimension,
                target_tensor);
            const auto memory = infer_with_memory(
                model.get(), vocab, held_out.second, make_hook(), target_tensor);
            const auto rank = token_rank(memory.logits, tuple.target_token);
            const auto recalled = memory.hook.applied &&
                                  memory.hook.entity.factor_label == tuple.entity &&
                                  memory.hook.relation.factor_label == tuple.relation &&
                                  memory.logits[static_cast<std::size_t>(tuple.target_token)] >
                                      baseline.logits[static_cast<std::size_t>(
                                          tuple.target_token)] &&
                                  rank == 1U;
            const auto delta = maximum_logit_difference(
                baseline.logits, memory.logits);
            if (recalled) {
                ++evaluation_recalled;
            }
            std::cout << "evaluation=" << held_out.first << '/'
                      << entities[tuple.entity] << '/' << relations[tuple.relation]
                      << " entity=" << memory.hook.entity.factor_label
                      << " entity_distance=" << memory.hook.entity.distance
                      << " relation=" << memory.hook.relation.factor_label
                      << " relation_distance=" << memory.hook.relation.distance
                      << " action_variant=" << memory.hook.action_variant
                      << " action_distance=" << memory.hook.action_distance
                      << " action_accepted="
                      << (memory.hook.action_accepted ? "yes" : "no")
                      << " applied=" << (memory.hook.applied ? "yes" : "no")
                      << " target_rank=" << rank
                      << " max_logit_delta=" << delta << '\n';
        }
    }

    bool missing_abstained = true;
    for (const auto& missing : {std::pair<std::size_t, std::size_t>{1U, 1U},
                                std::pair<std::size_t, std::size_t>{2U, 0U}}) {
        const auto& baseline = tuple_baselines[missing.first][missing.second];
        const auto memory = infer_with_memory(
            model.get(),
            vocab,
            tuple_prompt(entities[missing.first], relations[missing.second]),
            make_hook(),
            target_tensor);
        const auto delta = maximum_logit_difference(baseline.logits, memory.logits);
        missing_abstained = missing_abstained && memory.hook.entity.accepted &&
                            memory.hook.relation.accepted && !memory.hook.tuple_found &&
                            !memory.hook.applied && delta <= 1.0e-5F;
        std::cout << "missing_tuple=" << entities[missing.first] << '/'
                  << relations[missing.second]
                  << " factors_accepted="
                  << (memory.hook.entity.accepted && memory.hook.relation.accepted
                          ? "yes"
                          : "no")
                  << " applied=" << (memory.hook.applied ? "yes" : "no")
                  << " max_logit_delta=" << delta << '\n';
    }

    if (!action_views_recalled) {
        throw std::runtime_error("factorized memory failed a registered action view");
    }
    if (!missing_abstained) {
        throw std::runtime_error("factorized memory failed compositional abstention");
    }
    if (evaluation_recalled != tuples.size() * 2U) {
        throw std::runtime_error("factorized memory failed the frozen evaluation set");
    }
    std::cout << "stored_tuples=" << tuples.size()
              << " registered_action_views=" << action_view_specs.size()
              << " evaluation_recall=" << evaluation_recalled << '/'
              << tuples.size() * 2U
              << " missing_known_factor_tuples=2\n"
              << "factorized memory experiment passed\n";
    return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: gx1_llama_factorized_memory_experiment MODEL.gguf\n";
        return EXIT_FAILURE;
    }
    llama_log_set(&model_log, nullptr);
    llama_backend_init();
    try {
        const auto status = run(argv[1]);
        llama_backend_free();
        return status;
    } catch (const std::exception& error) {
        std::cerr << "factorized memory experiment failed: " << error.what() << '\n';
        llama_backend_free();
        return EXIT_FAILURE;
    }
}
