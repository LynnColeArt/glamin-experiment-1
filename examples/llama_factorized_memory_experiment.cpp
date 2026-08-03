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
    InferenceResult late_teacher;
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

struct WrongIntentSpec {
    std::size_t entity{0};
    std::size_t relation{0};
    std::string name;
    std::string prompt;
    InferenceResult baseline;
};

struct ActionPositiveSpec {
    std::size_t entity{0};
    std::size_t relation{0};
    std::string name;
    std::string prompt;
    llama_token target_token{LLAMA_TOKEN_NULL};
};

struct EntityAddressProbeSpec {
    std::size_t entity{0};
    std::size_t relation{0};
    std::string name;
    std::string prompt;
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

std::vector<std::pair<std::string, std::string>>
two_stage_development_prompts(
    const std::string& entity,
    const std::string& relation) {
    return {
        {"record-request",
         "Open the stored record for " + entity + " and return its " +
             relation + ".\nAnswer:"},
        {"record-fields",
         "Memory record: object " + entity + "; field " + relation +
             "; stored value:"},
    };
}

std::vector<std::pair<std::string, std::string>>
two_stage_evaluation_prompts(
    const std::string& entity,
    const std::string& relation) {
    return {
        {"long-term",
         "Retrieve from long-term memory: " + entity +
             "; requested attribute " + relation + ".\nAnswer:"},
        {"archive",
         "Archive[" + entity + "] / " + relation + " / value =>"},
    };
}

std::vector<std::pair<std::string, std::string>>
entity_address_development_prompts(
    const std::string& entity,
    const std::string& relation) {
    return {
        {"prose",
         "From retained memory, return the " + relation +
             " registered for object <" + entity + ">.\nAnswer:"},
        {"record",
         "Object record { name: \"" + entity + "\", field: \"" + relation +
             "\" }\nstored value:"},
        {"path",
         "memory/object/" + entity + "/attribute/" + relation + "/value="},
    };
}

std::vector<std::pair<std::string, std::string>>
entity_address_evaluation_prompts(
    const std::string& entity,
    const std::string& relation) {
    return {
        {"question",
         "Which memorized " + relation +
             " is attached to the object named " + entity + "?\nResponse:"},
        {"card",
         "Lookup card [object=" + entity + "][attribute=" + relation +
             "]\nvalue:"},
        {"uri",
         "vault://records/" + entity + "?field=" + relation + "#value="},
    };
}

gx1::ActivationMemoryBuildResult build_factor(
    const std::vector<FactorPrompt>& prompts,
    const std::vector<InferenceResult>& negatives,
    const std::vector<std::pair<std::size_t, InferenceResult>>& validation,
    const gx1::ActivationProjectionStrategy projection_strategy) {
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
        gx1::ActivationMemoryBuildConfig{
            256U,
            0.5F,
            false,
            projection_strategy,
        });
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

MemoryInferenceResult infer_with_target_state_memory(
    llama_model* model,
    const llama_vocab* vocab,
    const std::string& prompt,
    gx1::FactorizedLayerMemoryHook memory,
    std::shared_ptr<const gx1::TupleTargetStateLedger> target_states,
    const std::string& address_tensor,
    const std::string& action_tensor) {
    auto tokens = tokenize(vocab, prompt, llama_vocab_get_add_bos(vocab));
    gx1::LlamaTwoStageFactorizedGlaminHook hook(
        std::move(memory),
        std::move(target_states),
        address_tensor,
        action_tensor);
    auto parameters = context_parameters(tokens.size());
    parameters.cb_eval = &gx1::LlamaTwoStageFactorizedGlaminHook::evaluate;
    parameters.cb_eval_user_data = &hook;
    ContextPointer context(llama_init_from_model(model, parameters), &llama_free);
    if (!context) {
        throw std::runtime_error(
            "failed to create a two-stage factorized memory context");
    }
    const auto status = llama_decode(
        context.get(),
        llama_batch_get_one(tokens.data(), static_cast<std::int32_t>(tokens.size())));
    hook.throw_if_failed();
    if (status != 0 || !hook.last_result() ||
        hook.authorization_invocation_count() == 0U ||
        hook.action_invocation_count() == 0U) {
        throw std::runtime_error(
            "two-stage factorized memory inference failed");
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
    const auto action_tensor = "l_out-" + std::to_string(layer + 1);
    const auto* vocab = llama_model_get_vocab(model.get());

    const std::vector<std::string> entities{
        "Arcturus", "Bellatrix", "Cygnus", "Draco"};
    const std::vector<std::string> relations{"color", "material"};
    std::vector<TupleSpec> tuples{
        {0U, 0U, " blue", LLAMA_TOKEN_NULL, {}, {}, {}},
        {0U, 1U, " cedar", LLAMA_TOKEN_NULL, {}, {}, {}},
        {1U, 0U, " amber", LLAMA_TOKEN_NULL, {}, {}, {}},
        {2U, 1U, " copper", LLAMA_TOKEN_NULL, {}, {}, {}},
        {3U, 0U, " violet", LLAMA_TOKEN_NULL, {}, {}, {}},
        {3U, 1U, " maple", LLAMA_TOKEN_NULL, {}, {}, {}},
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
                "Stored-value lookup\nObject = " + entities[entity] +
                    "\nProperty = " + relations[relation] + "\nValue =",
                entities[entity] + " has which memorized " +
                    relations[relation] + "?\nValue:",
            };
            for (std::size_t development_index = 0;
                 development_index < development_prompts.size();
                 ++development_index) {
                const auto& prompt = development_prompts[development_index];
                auto development = capture(
                    model.get(),
                    vocab,
                    prompt,
                    hidden_dimension,
                    target_tensor);
                if (development_index < 2U) {
                    entity_validation.emplace_back(entity, development);
                }
                relation_validation.emplace_back(relation, std::move(development));
            }
            for (const auto& development : two_stage_development_prompts(
                     entities[entity], relations[relation])) {
                auto inference = capture(
                    model.get(),
                    vocab,
                    development.second,
                    hidden_dimension,
                    target_tensor);
                entity_validation.emplace_back(entity, inference);
                relation_validation.emplace_back(relation, std::move(inference));
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
        entity_prompts,
        entity_negatives,
        entity_validation,
        gx1::ActivationProjectionStrategy::variance);
    const auto relation_memory = build_factor(
        relation_prompts,
        relation_negatives,
        relation_validation,
        gx1::ActivationProjectionStrategy::association_signal);
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
        tuple.late_teacher = capture(
            model.get(), vocab, table + prompt, hidden_dimension, action_tensor);
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
            {"fields",
             "Stored-value lookup\nObject = " + entities[tuple.entity] +
                 "\nProperty = " + relations[tuple.relation] + "\nValue ="},
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
    std::vector<gx1::ActivationMemoryCalibrationView> action_negatives;
    std::vector<WrongIntentSpec> wrong_intents;
    for (std::size_t tuple_index = 0; tuple_index < tuples.size(); ++tuple_index) {
        const auto& tuple = tuples[tuple_index];
        const std::vector<std::pair<std::string, std::string>> prompts{
            {"metaphor",
             "Entity: " + entities[tuple.entity] + "\nRelation: " +
                 relations[tuple.relation] +
                 "\nInstruction: Write a metaphor about these words; do not "
                 "retrieve a stored value."},
            {"spelling",
             "Entity: " + entities[tuple.entity] + "\nRelation: " +
                 relations[tuple.relation] +
                 "\nInstruction: Compare the spelling of these words; do not "
                 "perform a memory lookup."},
        };
        for (const auto& prompt : prompts) {
            auto baseline = capture(
                model.get(), vocab, prompt.second, hidden_dimension, target_tensor);
            action_negatives.push_back({tuple_index, {baseline.hidden_state}});
            wrong_intents.push_back({
                tuple.entity,
                tuple.relation,
                prompt.first,
                prompt.second,
                std::move(baseline),
            });
        }
    }
    std::vector<gx1::ActivationMemoryValidationView> action_validation;
    std::vector<ActionPositiveSpec> action_positives;
    std::vector<ActionPositiveSpec> two_stage_development_positives;
    for (std::size_t tuple_index = 0; tuple_index < tuples.size(); ++tuple_index) {
        const auto& tuple = tuples[tuple_index];
        const std::vector<std::pair<std::string, std::string>> prompts{
            {"assigned",
             "Retrieve the " + relations[tuple.relation] + " assigned to " +
                 entities[tuple.entity] + ".\nAnswer:"},
            {"belongs",
             "Look in memory: which " + relations[tuple.relation] +
                 " belongs to " + entities[tuple.entity] + "?\nValue:"},
        };
        for (const auto& prompt : prompts) {
            action_validation.push_back({
                tuple_index,
                {capture(
                     model.get(),
                     vocab,
                     prompt.second,
                     hidden_dimension,
                     target_tensor)
                     .hidden_state},
            });
            action_positives.push_back({
                tuple.entity,
                tuple.relation,
                prompt.first,
                prompt.second,
                tuple.target_token,
            });
        }
        for (const auto& prompt : two_stage_development_prompts(
                 entities[tuple.entity], relations[tuple.relation])) {
            action_validation.push_back({
                tuple_index,
                {capture(
                     model.get(),
                     vocab,
                     prompt.second,
                     hidden_dimension,
                     target_tensor)
                     .hidden_state},
            });
            two_stage_development_positives.push_back({
                tuple.entity,
                tuple.relation,
                prompt.first,
                prompt.second,
                tuple.target_token,
            });
        }
    }

    auto entity_address_entity_validation = entity_validation;
    std::vector<EntityAddressProbeSpec> entity_address_development;
    for (std::size_t tuple_index = 0; tuple_index < tuples.size(); ++tuple_index) {
        const auto& tuple = tuples[tuple_index];
        for (const auto& prompt : entity_address_development_prompts(
                 entities[tuple.entity], relations[tuple.relation])) {
            auto inference = capture(
                model.get(),
                vocab,
                prompt.second,
                hidden_dimension,
                target_tensor);
            entity_address_entity_validation.emplace_back(
                tuple.entity, inference);
            entity_address_development.push_back({
                tuple.entity,
                tuple.relation,
                prompt.first,
                prompt.second,
                tuple.target_token,
                std::move(inference),
            });
        }
    }

    const auto action_memory = gx1::ActivationMemoryBuilder::build(
        action_construction,
        action_negatives,
        action_validation,
        gx1::ActivationMemoryBuildConfig{
            256U,
            0.5F,
            false,
            gx1::ActivationProjectionStrategy::variance,
            gx1::ActivationValidationScope::association,
        });
    const auto entity_address_variance_memory = build_factor(
        entity_prompts,
        entity_negatives,
        entity_address_entity_validation,
        gx1::ActivationProjectionStrategy::variance);
    const auto entity_address_association_memory = build_factor(
        entity_prompts,
        entity_negatives,
        entity_address_entity_validation,
        gx1::ActivationProjectionStrategy::association_signal);
    std::cout << "action_radius=" << action_memory.maximum_distance
              << " action_negative=" << action_memory.minimum_negative_distance
              << '\n';
    std::cout << "entity_address_variance_radius="
              << entity_address_variance_memory.maximum_distance
              << " entity_address_association_radius="
              << entity_address_association_memory.maximum_distance
              << " entity_address_negative="
              << entity_address_association_memory.minimum_negative_distance
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
    auto target_states = std::make_shared<gx1::TupleTargetStateLedger>();
    for (const auto& tuple : tuples) {
        target_states->insert(
            tuple.entity,
            tuple.relation,
            tuple.late_teacher.hidden_state);
    }

    const auto entity_address_variance_generation = generations.mount_flat(
        "entity-address-variance",
        entity_address_variance_memory.query_dimension,
        flatten(entity_address_variance_memory.keys));
    const auto entity_address_association_generation = generations.mount_flat(
        "entity-address-association",
        entity_address_association_memory.query_dimension,
        flatten(entity_address_association_memory.keys));

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

    const auto make_entity_address_hook = [&](const bool association_signal) {
        const auto& candidate_memory = association_signal
                                           ? entity_address_association_memory
                                           : entity_address_variance_memory;
        generations.activate(
            association_signal ? entity_address_association_generation
                               : entity_address_variance_generation);
        auto entity_pin = generations.pin_active();
        generations.activate(relation_generation);
        auto relation_pin = generations.pin_active();
        return gx1::FactorizedLayerMemoryHook(
            std::move(entity_pin),
            factor_config(candidate_memory),
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
    if (!action_views_recalled) {
        throw std::runtime_error("factorized memory failed a registered action view");
    }

    bool action_positives_routed = true;
    std::size_t development_recalled = 0U;
    for (const auto& positive : action_positives) {
        const auto memory = infer_with_memory(
            model.get(), vocab, positive.prompt, make_hook(), target_tensor);
        const auto rank = token_rank(memory.logits, positive.target_token);
        const auto routed = memory.hook.applied &&
                            memory.hook.entity.factor_label == positive.entity &&
                            memory.hook.relation.factor_label == positive.relation;
        action_positives_routed = action_positives_routed && routed;
        development_recalled += routed && rank == 1U ? 1U : 0U;
        std::cout << "development_action=" << positive.name << '/'
                  << entities[positive.entity] << '/'
                  << relations[positive.relation]
                  << " entity_distance=" << memory.hook.entity.distance
                  << " relation_distance=" << memory.hook.relation.distance
                  << " action_variant=" << memory.hook.action_variant
                  << " action_distance=" << memory.hook.action_distance
                  << " applied=" << (memory.hook.applied ? "yes" : "no")
                  << " target_rank=" << rank << '\n';
    }
    if (!action_positives_routed) {
        throw std::runtime_error(
            "factorized memory failed to route a development action view");
    }
    std::cout << "development_action_summary=" << development_recalled << '/'
              << action_positives.size() << " rank-one after "
              << action_positives.size() << '/' << action_positives.size()
              << " routes\n";

    std::size_t two_stage_development_routes = 0U;
    std::size_t residual_development_recalled = 0U;
    std::size_t target_state_development_recalled = 0U;
    for (const auto& positive : two_stage_development_positives) {
        const auto residual = infer_with_memory(
            model.get(), vocab, positive.prompt, make_hook(), target_tensor);
        const auto target_state = infer_with_target_state_memory(
            model.get(),
            vocab,
            positive.prompt,
            make_hook(),
            target_states,
            target_tensor,
            action_tensor);
        const auto residual_rank = token_rank(
            residual.logits, positive.target_token);
        const auto target_state_rank = token_rank(
            target_state.logits, positive.target_token);
        const auto routed = target_state.hook.applied &&
                            target_state.hook.entity.factor_label ==
                                positive.entity &&
                            target_state.hook.relation.factor_label ==
                                positive.relation;
        two_stage_development_routes += routed ? 1U : 0U;
        residual_development_recalled +=
            residual.hook.applied &&
                    residual.hook.entity.factor_label == positive.entity &&
                    residual.hook.relation.factor_label == positive.relation &&
                    residual_rank == 1U
                ? 1U
                : 0U;
        target_state_development_recalled +=
            routed && target_state_rank == 1U ? 1U : 0U;
        std::cout << "two_stage_development=" << positive.name << '/'
                  << entities[positive.entity] << '/'
                  << relations[positive.relation]
                  << " action_distance=" << target_state.hook.action_distance
                  << " residual_rank=" << residual_rank
                  << " target_state_rank=" << target_state_rank
                  << " target_applied="
                  << (target_state.hook.applied ? "yes" : "no") << '\n';
    }
    if (two_stage_development_routes !=
            two_stage_development_positives.size() ||
        target_state_development_recalled !=
            two_stage_development_positives.size()) {
        throw std::runtime_error(
            "two-stage target-state memory failed its development set");
    }
    std::cout << "two_stage_development_summary=routes "
              << two_stage_development_routes << '/'
              << two_stage_development_positives.size()
              << " residual_rank_one=" << residual_development_recalled << '/'
              << two_stage_development_positives.size()
              << " target_state_rank_one="
              << target_state_development_recalled << '/'
              << two_stage_development_positives.size() << '\n';

    bool wrong_intents_abstained = true;
    for (const auto& negative : wrong_intents) {
        const auto memory = infer_with_memory(
            model.get(), vocab, negative.prompt, make_hook(), target_tensor);
        const auto target_state_memory = infer_with_target_state_memory(
            model.get(),
            vocab,
            negative.prompt,
            make_hook(),
            target_states,
            target_tensor,
            action_tensor);
        const auto delta = maximum_logit_difference(
            negative.baseline.logits, memory.logits);
        const auto target_state_delta = maximum_logit_difference(
            negative.baseline.logits, target_state_memory.logits);
        const auto reached_action_gate =
            memory.hook.entity.accepted && memory.hook.relation.accepted &&
            memory.hook.entity.factor_label == negative.entity &&
            memory.hook.relation.factor_label == negative.relation &&
            memory.hook.tuple_found;
        wrong_intents_abstained =
            wrong_intents_abstained && reached_action_gate &&
            !memory.hook.action_accepted && !memory.hook.applied &&
            delta <= 1.0e-5F &&
            target_state_memory.hook.entity.factor_label == negative.entity &&
            target_state_memory.hook.relation.factor_label == negative.relation &&
            target_state_memory.hook.tuple_found &&
            !target_state_memory.hook.action_accepted &&
            !target_state_memory.hook.applied &&
            target_state_delta <= 1.0e-5F;
        std::cout << "wrong_intent=" << negative.name << '/'
                  << entities[negative.entity] << '/'
                  << relations[negative.relation]
                  << " factors_accepted="
                  << (memory.hook.entity.accepted && memory.hook.relation.accepted
                          ? "yes"
                          : "no")
                  << " tuple_found="
                  << (memory.hook.tuple_found ? "yes" : "no")
                  << " action_distance=" << memory.hook.action_distance
                  << " applied=" << (memory.hook.applied ? "yes" : "no")
                  << " residual_logit_delta=" << delta
                  << " target_state_logit_delta=" << target_state_delta << '\n';
    }
    if (!wrong_intents_abstained) {
        throw std::runtime_error(
            "same-factor wrong-intent development gate failed");
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
        const auto target_state_memory = infer_with_target_state_memory(
            model.get(),
            vocab,
            tuple_prompt(entities[missing.first], relations[missing.second]),
            make_hook(),
            target_states,
            target_tensor,
            action_tensor);
        const auto delta = maximum_logit_difference(baseline.logits, memory.logits);
        const auto target_state_delta = maximum_logit_difference(
            baseline.logits, target_state_memory.logits);
        missing_abstained = missing_abstained && memory.hook.entity.accepted &&
                            memory.hook.relation.accepted && !memory.hook.tuple_found &&
                            !memory.hook.applied && delta <= 1.0e-5F &&
                            target_state_memory.hook.entity.accepted &&
                            target_state_memory.hook.relation.accepted &&
                            !target_state_memory.hook.tuple_found &&
                            !target_state_memory.hook.applied &&
                            target_state_delta <= 1.0e-5F;
        std::cout << "missing_tuple=" << entities[missing.first] << '/'
                  << relations[missing.second]
                  << " factors_accepted="
                  << (memory.hook.entity.accepted && memory.hook.relation.accepted
                          ? "yes"
                          : "no")
                  << " applied=" << (memory.hook.applied ? "yes" : "no")
                  << " residual_logit_delta=" << delta
                  << " target_state_logit_delta=" << target_state_delta << '\n';
    }
    if (!missing_abstained) {
        throw std::runtime_error("factorized memory failed compositional abstention");
    }

    std::size_t entity_address_variance_development_routes = 0U;
    std::size_t entity_address_variance_development_entities = 0U;
    std::size_t entity_address_association_development_routes = 0U;
    std::size_t entity_address_association_development_entities = 0U;
    std::size_t entity_address_association_development_recall = 0U;
    for (const auto& positive : entity_address_development) {
        const auto variance = infer_with_target_state_memory(
            model.get(),
            vocab,
            positive.prompt,
            make_entity_address_hook(false),
            target_states,
            target_tensor,
            action_tensor);
        const auto association = infer_with_target_state_memory(
            model.get(),
            vocab,
            positive.prompt,
            make_entity_address_hook(true),
            target_states,
            target_tensor,
            action_tensor);
        const auto variance_routed =
            variance.hook.applied &&
            variance.hook.entity.factor_label == positive.entity &&
            variance.hook.relation.factor_label == positive.relation;
        const auto association_routed =
            association.hook.applied &&
            association.hook.entity.factor_label == positive.entity &&
            association.hook.relation.factor_label == positive.relation;
        const auto variance_entity_matched =
            variance.hook.entity.accepted &&
            variance.hook.entity.factor_label == positive.entity;
        const auto association_entity_matched =
            association.hook.entity.accepted &&
            association.hook.entity.factor_label == positive.entity;
        const auto association_rank = token_rank(
            association.logits, positive.target_token);
        entity_address_variance_development_routes +=
            variance_routed ? 1U : 0U;
        entity_address_variance_development_entities +=
            variance_entity_matched ? 1U : 0U;
        entity_address_association_development_routes +=
            association_routed ? 1U : 0U;
        entity_address_association_development_entities +=
            association_entity_matched ? 1U : 0U;
        entity_address_association_development_recall +=
            association_routed && association_rank == 1U ? 1U : 0U;
        std::cout << "entity_address_development=" << positive.name << '/'
                  << entities[positive.entity] << '/'
                  << relations[positive.relation]
                  << " variance_entity_distance="
                  << variance.hook.entity.distance
                  << " variance_entity="
                  << variance.hook.entity.factor_label
                  << " variance_routed="
                  << (variance_routed ? "yes" : "no")
                  << " association_entity_distance="
                  << association.hook.entity.distance
                  << " association_entity="
                  << association.hook.entity.factor_label
                  << " association_relation_accepted="
                  << (association.hook.relation.accepted ? "yes" : "no")
                  << " association_action_accepted="
                  << (association.hook.action_accepted ? "yes" : "no")
                  << " association_routed="
                  << (association_routed ? "yes" : "no")
                  << " association_rank=" << association_rank << '\n';
    }
    std::cout << "entity_address_development_summary=variance_routes "
              << entity_address_variance_development_routes << '/'
              << entity_address_development.size()
              << " variance_entities "
              << entity_address_variance_development_entities << '/'
              << entity_address_development.size()
              << " association_routes "
              << entity_address_association_development_routes << '/'
              << entity_address_development.size()
              << " association_entities "
              << entity_address_association_development_entities << '/'
              << entity_address_development.size()
              << " association_rank_one "
              << entity_address_association_development_recall << '/'
              << entity_address_development.size() << '\n';
    if (entity_address_association_development_entities !=
        entity_address_development.size()) {
        throw std::runtime_error(
            "association-signal entity addressing failed its development set");
    }

    bool entity_address_wrong_intents_abstained = true;
    for (const auto& negative : wrong_intents) {
        const auto memory = infer_with_target_state_memory(
            model.get(),
            vocab,
            negative.prompt,
            make_entity_address_hook(true),
            target_states,
            target_tensor,
            action_tensor);
        const auto delta = maximum_logit_difference(
            negative.baseline.logits, memory.logits);
        entity_address_wrong_intents_abstained =
            entity_address_wrong_intents_abstained &&
            memory.hook.entity.accepted && memory.hook.relation.accepted &&
            memory.hook.entity.factor_label == negative.entity &&
            memory.hook.relation.factor_label == negative.relation &&
            memory.hook.tuple_found && !memory.hook.action_accepted &&
            !memory.hook.applied && delta <= 1.0e-5F;
    }
    if (!entity_address_wrong_intents_abstained) {
        throw std::runtime_error(
            "entity-address development admitted a wrong-intent control");
    }

    bool entity_address_missing_abstained = true;
    for (const auto& missing : {std::pair<std::size_t, std::size_t>{1U, 1U},
                                std::pair<std::size_t, std::size_t>{2U, 0U}}) {
        const auto& baseline = tuple_baselines[missing.first][missing.second];
        const auto memory = infer_with_target_state_memory(
            model.get(),
            vocab,
            tuple_prompt(entities[missing.first], relations[missing.second]),
            make_entity_address_hook(true),
            target_states,
            target_tensor,
            action_tensor);
        const auto delta = maximum_logit_difference(
            baseline.logits, memory.logits);
        entity_address_missing_abstained =
            entity_address_missing_abstained &&
            memory.hook.entity.accepted && memory.hook.relation.accepted &&
            !memory.hook.tuple_found && !memory.hook.applied &&
            delta <= 1.0e-5F;
    }
    if (!entity_address_missing_abstained) {
        throw std::runtime_error(
            "entity-address development failed compositional abstention");
    }

    std::size_t evaluation_routed = 0U;
    std::size_t evaluation_recalled = 0U;
    for (const auto& tuple : tuples) {
        const std::vector<std::pair<std::string, std::string>> held_out_prompts{
            {"consult",
             "Consult stored memory for " + entities[tuple.entity] +
                 "; requested property: " + relations[tuple.relation] +
                 ".\nAnswer:"},
            {"associate",
             "What value does memory associate with " + entities[tuple.entity] +
                 " under " + relations[tuple.relation] + "?\nValue:"},
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
            const auto routed = memory.hook.applied &&
                                memory.hook.entity.factor_label == tuple.entity &&
                                memory.hook.relation.factor_label == tuple.relation;
            const auto recalled = routed &&
                                  memory.logits[static_cast<std::size_t>(tuple.target_token)] >
                                      baseline.logits[static_cast<std::size_t>(
                                          tuple.target_token)] &&
                                  rank == 1U;
            const auto delta = maximum_logit_difference(
                baseline.logits, memory.logits);
            if (routed) {
                ++evaluation_routed;
            }
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

    if (evaluation_routed != tuples.size() * 2U) {
        throw std::runtime_error(
            "factorized memory failed to route the frozen evaluation set");
    }

    std::size_t two_stage_evaluation_routes = 0U;
    std::size_t residual_evaluation_recalled = 0U;
    std::size_t target_state_evaluation_recalled = 0U;
    for (const auto& tuple : tuples) {
        for (const auto& held_out : two_stage_evaluation_prompts(
                 entities[tuple.entity], relations[tuple.relation])) {
            const auto baseline = capture(
                model.get(),
                vocab,
                held_out.second,
                hidden_dimension,
                target_tensor);
            const auto residual = infer_with_memory(
                model.get(),
                vocab,
                held_out.second,
                make_hook(),
                target_tensor);
            const auto target_state = infer_with_target_state_memory(
                model.get(),
                vocab,
                held_out.second,
                make_hook(),
                target_states,
                target_tensor,
                action_tensor);
            const auto residual_rank = token_rank(
                residual.logits, tuple.target_token);
            const auto target_state_rank = token_rank(
                target_state.logits, tuple.target_token);
            const auto routed = target_state.hook.applied &&
                                target_state.hook.entity.factor_label ==
                                    tuple.entity &&
                                target_state.hook.relation.factor_label ==
                                    tuple.relation;
            const auto residual_recalled =
                residual.hook.applied &&
                residual.hook.entity.factor_label == tuple.entity &&
                residual.hook.relation.factor_label == tuple.relation &&
                residual.logits[static_cast<std::size_t>(tuple.target_token)] >
                    baseline.logits[static_cast<std::size_t>(tuple.target_token)] &&
                residual_rank == 1U;
            const auto target_state_recalled =
                routed &&
                target_state.logits[static_cast<std::size_t>(tuple.target_token)] >
                    baseline.logits[static_cast<std::size_t>(tuple.target_token)] &&
                target_state_rank == 1U;
            two_stage_evaluation_routes += routed ? 1U : 0U;
            residual_evaluation_recalled += residual_recalled ? 1U : 0U;
            target_state_evaluation_recalled +=
                target_state_recalled ? 1U : 0U;
            std::cout << "two_stage_evaluation=" << held_out.first << '/'
                      << entities[tuple.entity] << '/'
                      << relations[tuple.relation]
                      << " entity=" << target_state.hook.entity.factor_label
                      << " entity_distance="
                      << target_state.hook.entity.distance
                      << " entity_accepted="
                      << (target_state.hook.entity.accepted ? "yes" : "no")
                      << " relation=" << target_state.hook.relation.factor_label
                      << " relation_distance="
                      << target_state.hook.relation.distance
                      << " relation_accepted="
                      << (target_state.hook.relation.accepted ? "yes" : "no")
                      << " action_distance="
                      << target_state.hook.action_distance
                      << " action_accepted="
                      << (target_state.hook.action_accepted ? "yes" : "no")
                      << " residual_rank=" << residual_rank
                      << " target_state_rank=" << target_state_rank
                      << " residual_logit_delta="
                      << maximum_logit_difference(
                             baseline.logits, residual.logits)
                      << " target_state_logit_delta="
                      << maximum_logit_difference(
                             baseline.logits, target_state.logits)
                      << " teacher_logit_delta="
                      << maximum_logit_difference(
                             tuple.late_teacher.logits, target_state.logits)
                      << '\n';
        }
    }
    const auto two_stage_evaluation_count = tuples.size() * 2U;
    std::cout << "two_stage_evaluation_summary=routes "
              << two_stage_evaluation_routes << '/'
              << two_stage_evaluation_count
              << " residual_rank_one=" << residual_evaluation_recalled << '/'
              << two_stage_evaluation_count
              << " target_state_rank_one="
              << target_state_evaluation_recalled << '/'
              << two_stage_evaluation_count
              << " target_state_conditional="
              << target_state_evaluation_recalled << '/'
              << two_stage_evaluation_routes << '\n';
    if (two_stage_evaluation_routes != 10U ||
        target_state_evaluation_recalled != 10U) {
        throw std::runtime_error(
            "historical two-stage frozen result drifted from 10/12");
    }

    std::size_t entity_address_variance_evaluation_routes = 0U;
    std::size_t entity_address_variance_evaluation_recall = 0U;
    std::size_t entity_address_association_evaluation_routes = 0U;
    std::size_t entity_address_association_evaluation_recall = 0U;
    for (const auto& tuple : tuples) {
        for (const auto& held_out : entity_address_evaluation_prompts(
                 entities[tuple.entity], relations[tuple.relation])) {
            const auto baseline = capture(
                model.get(),
                vocab,
                held_out.second,
                hidden_dimension,
                target_tensor);
            const auto variance = infer_with_target_state_memory(
                model.get(),
                vocab,
                held_out.second,
                make_entity_address_hook(false),
                target_states,
                target_tensor,
                action_tensor);
            const auto association = infer_with_target_state_memory(
                model.get(),
                vocab,
                held_out.second,
                make_entity_address_hook(true),
                target_states,
                target_tensor,
                action_tensor);
            const auto variance_rank = token_rank(
                variance.logits, tuple.target_token);
            const auto association_rank = token_rank(
                association.logits, tuple.target_token);
            const auto variance_routed =
                variance.hook.applied &&
                variance.hook.entity.factor_label == tuple.entity &&
                variance.hook.relation.factor_label == tuple.relation;
            const auto association_routed =
                association.hook.applied &&
                association.hook.entity.factor_label == tuple.entity &&
                association.hook.relation.factor_label == tuple.relation;
            entity_address_variance_evaluation_routes +=
                variance_routed ? 1U : 0U;
            entity_address_variance_evaluation_recall +=
                variance_routed && variance_rank == 1U ? 1U : 0U;
            entity_address_association_evaluation_routes +=
                association_routed ? 1U : 0U;
            entity_address_association_evaluation_recall +=
                association_routed && association_rank == 1U ? 1U : 0U;
            std::cout << "entity_address_evaluation=" << held_out.first << '/'
                      << entities[tuple.entity] << '/'
                      << relations[tuple.relation]
                      << " variance_entity_distance="
                      << variance.hook.entity.distance
                      << " variance_entity_accepted="
                      << (variance.hook.entity.accepted ? "yes" : "no")
                      << " variance_routed="
                      << (variance_routed ? "yes" : "no")
                      << " variance_rank=" << variance_rank
                      << " association_entity_distance="
                      << association.hook.entity.distance
                      << " association_entity_accepted="
                      << (association.hook.entity.accepted ? "yes" : "no")
                      << " association_relation_accepted="
                      << (association.hook.relation.accepted ? "yes" : "no")
                      << " association_action_accepted="
                      << (association.hook.action_accepted ? "yes" : "no")
                      << " association_routed="
                      << (association_routed ? "yes" : "no")
                      << " association_rank=" << association_rank
                      << " association_logit_delta="
                      << maximum_logit_difference(
                             baseline.logits, association.logits)
                      << " teacher_logit_delta="
                      << maximum_logit_difference(
                             tuple.late_teacher.logits, association.logits)
                      << '\n';
        }
    }
    const auto entity_address_evaluation_count = tuples.size() * 3U;

    std::size_t entity_address_unknown_entity_noops = 0U;
    std::size_t entity_address_unknown_entity_count = 0U;
    for (const auto& unknown : {std::string("Rigel"), std::string("Sirius")}) {
        for (const auto& held_out : entity_address_evaluation_prompts(
                 unknown, "color")) {
            ++entity_address_unknown_entity_count;
            const auto baseline = capture(
                model.get(),
                vocab,
                held_out.second,
                hidden_dimension,
                target_tensor);
            const auto memory = infer_with_target_state_memory(
                model.get(),
                vocab,
                held_out.second,
                make_entity_address_hook(true),
                target_states,
                target_tensor,
                action_tensor);
            const auto delta = maximum_logit_difference(
                baseline.logits, memory.logits);
            const auto exact_noop = !memory.hook.entity.accepted &&
                                    !memory.hook.applied && delta <= 1.0e-5F;
            entity_address_unknown_entity_noops += exact_noop ? 1U : 0U;
            std::cout << "entity_address_unknown=" << held_out.first << '/'
                      << unknown
                      << " nearest_entity=" << memory.hook.entity.factor_label
                      << " entity_distance=" << memory.hook.entity.distance
                      << " entity_accepted="
                      << (memory.hook.entity.accepted ? "yes" : "no")
                      << " applied="
                      << (memory.hook.applied ? "yes" : "no")
                      << " max_logit_delta=" << delta << '\n';
        }
    }

    std::size_t entity_address_historical_routes = 0U;
    std::size_t entity_address_historical_recall = 0U;
    for (const auto& tuple : tuples) {
        for (const auto& held_out : two_stage_evaluation_prompts(
                 entities[tuple.entity], relations[tuple.relation])) {
            const auto memory = infer_with_target_state_memory(
                model.get(),
                vocab,
                held_out.second,
                make_entity_address_hook(true),
                target_states,
                target_tensor,
                action_tensor);
            const auto routed =
                memory.hook.applied &&
                memory.hook.entity.factor_label == tuple.entity &&
                memory.hook.relation.factor_label == tuple.relation;
            entity_address_historical_routes += routed ? 1U : 0U;
            entity_address_historical_recall +=
                routed && token_rank(memory.logits, tuple.target_token) == 1U
                    ? 1U
                    : 0U;
        }
    }

    std::cout << "entity_address_evaluation_summary=variance_routes "
              << entity_address_variance_evaluation_routes << '/'
              << entity_address_evaluation_count
              << " variance_rank_one "
              << entity_address_variance_evaluation_recall << '/'
              << entity_address_evaluation_count
              << " association_routes "
              << entity_address_association_evaluation_routes << '/'
              << entity_address_evaluation_count
              << " association_rank_one "
              << entity_address_association_evaluation_recall << '/'
              << entity_address_evaluation_count
              << " unknown_entity_noops "
              << entity_address_unknown_entity_noops << '/'
              << entity_address_unknown_entity_count
              << " historical_routes " << entity_address_historical_routes
              << '/' << two_stage_evaluation_count
              << " historical_rank_one "
              << entity_address_historical_recall << '/'
              << two_stage_evaluation_count << '\n';
    if (entity_address_association_evaluation_routes !=
            entity_address_evaluation_count ||
        entity_address_association_evaluation_recall !=
            entity_address_evaluation_count ||
        entity_address_unknown_entity_noops !=
            entity_address_unknown_entity_count) {
        throw std::runtime_error(
            "association-signal entity addressing failed the frozen criterion");
    }

    std::cout << "stored_tuples=" << tuples.size()
              << " registered_action_views=" << action_view_specs.size()
              << " development_action_views=" << action_positives.size()
              << " wrong_intent_controls=" << wrong_intents.size()
              << " evaluation_routes=" << evaluation_routed << '/'
              << tuples.size() * 2U
              << " evaluation_recall=" << evaluation_recalled << '/'
              << tuples.size() * 2U
              << " two_stage_evaluation_routes="
              << two_stage_evaluation_routes << '/'
              << two_stage_evaluation_count
              << " residual_evaluation_recall="
              << residual_evaluation_recalled << '/'
              << two_stage_evaluation_count
              << " target_state_evaluation_recall="
              << target_state_evaluation_recalled << '/'
              << two_stage_evaluation_count
              << " entity_address_evaluation_routes="
              << entity_address_association_evaluation_routes << '/'
              << entity_address_evaluation_count
              << " entity_address_evaluation_recall="
              << entity_address_association_evaluation_recall << '/'
              << entity_address_evaluation_count
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
